"""Voice-safe tool layer.

`voice_tool` wraps a plain async function with the behaviour every voice tool needs:
  * spoken filler if the call is slow (only when the line is otherwise silent)
  * hard timeout + bounded retry (retries only for read-only tools)
  * friendly, code-free error messages for the LLM to relay
  * enforced confirmation for risky actions: the action cannot run until the user
    has taken another turn after the agent read the action back
  * idempotency: a completed side-effecting action is never executed twice
    (e.g. after a reconnect, a failover or a repeated "yes")
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from livekit.agents import RunContext, ToolError, function_tool, get_job_context

from . import backend
from .config import load_settings
from .state import PendingConfirmation, SessionState
from .telemetry import redact

logger = logging.getLogger("voice-agent")

# call_id -> SessionTelemetry; set by main so tools can report failures
TELEMETRY: dict[str, Any] = {}

_SPOKEN_DIGITS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def digits_from_speech(value: str) -> str:
    """'four five six, seven eight' / '45 678' / 'double five 6' -> digits only."""
    tokens = re.findall(r"[a-z]+|\d+", value.lower())
    out: list[str] = []
    repeat = 1
    for tok in tokens:
        if tok in ("double", "triple"):
            repeat = 2 if tok == "double" else 3
            continue
        digit = tok if tok.isdigit() else _SPOKEN_DIGITS.get(tok)
        if digit is None:
            continue
        out.append(digit * repeat if len(digit) == 1 else digit)
        repeat = 1
    return "".join(out)


def count_user_turns(chat_ctx: Any) -> int:
    """User messages in the history. Counted from the history itself (not a hook) so it
    is correct for every input path: voice turns, text chat, and replayed checkpoints."""
    return sum(1 for i in chat_ctx.items if getattr(i, "role", None) == "user")


def _action_key(name: str, kwargs: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(kwargs, sort_keys=True, default=str)}"


def voice_tool(
    *,
    side_effect: bool = False,
    confirm: bool = False,
    filler: str | None = "One moment while I check that.",
    timeout: float | None = None,
    retries: int = 1,
) -> Callable[[Callable[..., Awaitable[str]]], Any]:
    def deco(fn: Callable[..., Awaitable[str]]) -> Any:
        name = fn.__name__
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        async def wrapper(ctx: RunContext[SessionState], *args: Any, **kw: Any) -> str | None:
            bound = sig.bind(ctx, *args, **kw)
            kwargs = {k: v for k, v in bound.arguments.items() if k != "ctx"}
            settings = load_settings()
            state = ctx.userdata
            user_turn = count_user_turns(ctx.session.history)
            key = _action_key(name, kwargs)
            telemetry = TELEMETRY.get(state.call_id)
            if telemetry:
                telemetry.on_tool(ctx.speech_handle.id, name)

            if side_effect and key in state.completed_actions:
                return f"Already completed earlier in this call: {state.completed_actions[key]}"

            if confirm:
                pending = state.pending_confirmations.get(key)
                if pending is None or pending.user_turn >= user_turn:
                    summary = ", ".join(f"{k} {v}" for k, v in kwargs.items())
                    state.pending_confirmations[key] = PendingConfirmation(
                        key=key, summary=summary, user_turn=user_turn
                    )
                    return (
                        f"NOT EXECUTED YET. Read back the action ({name.replace('_', ' ')}: {summary}) "
                        "and ask the user to confirm. Call this tool again with the same arguments "
                        "only after the user clearly says yes."
                    )

            # never start an irreversible action for a reply the user already talked over
            if side_effect and ctx.speech_handle.interrupted:
                return None

            attempts = 1 if side_effect else 1 + retries
            last_exc: BaseException | None = None
            started = time.perf_counter()
            for attempt in range(attempts):
                try:
                    filler_cm = (
                        ctx.with_filler(filler, delay=settings.tool_filler_delay) if filler else _null()
                    )
                    async with filler_cm:
                        result = await asyncio.wait_for(
                            fn(ctx, **kwargs), timeout=timeout or settings.tool_timeout
                        )
                    break
                except backend.BackendError as e:  # business-rule answer, not a failure
                    raise ToolError(str(e)) from None
                except ToolError:
                    raise
                except (TimeoutError, Exception) as e:
                    last_exc = e
                    logger.warning(
                        "tool failed",
                        extra={"json": {"tool": name, "attempt": attempt + 1, "error": redact(repr(e))}},
                    )
            else:
                if telemetry:
                    telemetry.tool_failures += 1
                raise ToolError(
                    "The system didn't respond in time. Apologize briefly without technical details "
                    "and offer to try again or connect the caller to a person."
                ) from last_exc

            logger.info(
                "tool ok", extra={"json": {"tool": name, "ms": round((time.perf_counter() - started) * 1000)}}
            )
            if side_effect:
                state.completed_actions[key] = result
                state.pending_confirmations.pop(key, None)
            return result

        return function_tool(wrapper, on_duplicate="reject" if side_effect else "allow")

    return deco


class _null:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> None:
        return None


# ------------------------------------------------------------------ domain tools


@voice_tool(filler="Let me pull up that order.")
async def get_order_status(ctx: RunContext[SessionState], order_id: str) -> str:
    """Look up an order's status, items, total and delivery date.

    Args:
        order_id: The order number, digits only (convert spoken digits like "four five six" to "456").
    """
    oid = digits_from_speech(order_id)
    if not oid:
        raise ToolError("That order number didn't contain any digits. Ask the caller to repeat it.")
    o = await backend.get_order(oid)
    ctx.userdata.slots["order_id"] = oid
    eta = f" Expected delivery {o.eta}." if o.eta else ""
    return f"Order {oid}: status {o.status}; items: {', '.join(o.items)}; total ${o.total:.2f}.{eta}"


@voice_tool(side_effect=True, confirm=True, filler="Okay, processing that now.")
async def cancel_order(ctx: RunContext[SessionState], order_id: str) -> str:
    """Cancel an order that has not shipped yet. Irreversible.

    Args:
        order_id: The order number, digits only.
    """
    return await backend.cancel_order(digits_from_speech(order_id))


@voice_tool(filler=None, timeout=1.5)
async def search_knowledge_base(ctx: RunContext[SessionState], query: str) -> str:
    """Search store policies (returns, refunds, shipping, hours, warranty). Use for any policy question.

    Args:
        query: A short search query.
    """
    hits = await backend.search_kb(query)
    return " ".join(hits) if hits else "No matching policy found. Do not guess; offer a human agent."


# ------------------------------------------------------------------ call-control tools


@function_tool
async def hold_on(ctx: RunContext[SessionState]) -> str:
    """Call when the user asks you to wait ("hold on", "give me a second", "let me find it")."""
    ctx.userdata.on_hold_until = time.time() + load_settings().hold_timeout
    return "Hold started. Reply with a very short acknowledgment like 'Sure, take your time.'"


@function_tool
async def end_call(ctx: RunContext[SessionState], farewell: str) -> None:
    """End the call when the user says goodbye, the task is done and they need nothing else,
    or after one warning to an abusive caller.

    Args:
        farewell: One short goodbye sentence to say before hanging up.
    """
    state = ctx.userdata
    if state.ending:
        return None
    state.ending = True
    _say_then_hangup(ctx, farewell)
    return None


@function_tool
async def transfer_to_human(ctx: RunContext[SessionState], reason: str, summary: str) -> None:
    """Transfer to a human agent when the user asks for a person, is frustrated, or you cannot help.

    Args:
        reason: Short reason for the transfer.
        summary: Two-sentence summary of the issue and what was already tried, for the human agent.
    """
    state = ctx.userdata
    if state.transferred:
        return None
    state.transferred = True
    logger.info(
        "transfer",
        extra={
            "json": {
                "call": state.call_id,
                "reason": reason,
                "summary": redact(summary),
                "slots": state.slots,
            }
        },
    )
    # Telephony: replace the hangup with a SIP REFER / warm transfer, e.g.
    # await get_job_context().api.sip.transfer_sip_participant(...)
    _say_then_hangup(
        ctx,
        "I'm connecting you with a member of our team now. I've passed along the details, "
        "so you won't need to repeat yourself.",
    )
    return None


def _say_then_hangup(ctx: RunContext[SessionState], text: str) -> None:
    """Queue the last words and hang up after they finish playing.

    Must not be awaited inside the tool: the queued speech plays only after the tool's
    own turn completes, so awaiting it from the tool would deadlock.
    """
    session = ctx.session
    handle = session.say(text, allow_interruptions=False)

    async def _run() -> None:
        try:
            await asyncio.wait_for(handle.wait_for_playout(), timeout=20)
        except Exception:
            logger.warning("farewell playout did not complete", exc_info=True)
        _hangup(session)

    _BACKGROUND.add(task := asyncio.create_task(_run()))
    task.add_done_callback(_BACKGROUND.discard)


_BACKGROUND: set[asyncio.Task[None]] = set()


def _hangup(session: Any) -> None:
    session.shutdown(drain=True)
    try:
        get_job_context().delete_room()
    except RuntimeError:  # no job context (tests / console)
        pass


ALL_TOOLS = [get_order_status, cancel_order, search_knowledge_base, hold_on, end_call, transfer_to_human]
