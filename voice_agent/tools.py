"""Voice-safe tool layer and the assistant's tools.

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
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from livekit.agents import RunContext, ToolError, function_tool, get_job_context

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


def require_confirmation(ctx: RunContext[SessionState], name: str, kwargs: dict[str, Any]) -> str | None:
    """Enforce confirmation in code: an action may run only if it was read back to the user in an
    earlier turn and the user has spoken since. Returns the instruction for the LLM while the
    action is still unconfirmed, or None once it may run (the pending entry is then consumed)."""
    state = ctx.userdata
    user_turn = count_user_turns(ctx.session.history)
    key = _action_key(name, kwargs)
    pending = state.pending_confirmations.get(key)
    if pending is None or pending.user_turn >= user_turn:
        summary = ", ".join(f"{k} {v}" for k, v in kwargs.items())
        state.pending_confirmations[key] = PendingConfirmation(key=key, summary=summary, user_turn=user_turn)
        return (
            f"NOT EXECUTED YET. Read back the action ({name.replace('_', ' ')}: {summary}) "
            "in plain words and ask the user to confirm. Call this tool again with the same "
            "arguments only after the user clearly says yes."
        )
    state.pending_confirmations.pop(key, None)
    return None


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
            key = _action_key(name, kwargs)
            telemetry = TELEMETRY.get(state.call_id)
            if telemetry:
                telemetry.on_tool(ctx.speech_handle.id, name)

            if side_effect and key in state.completed_actions:
                return f"Already completed earlier in this call: {state.completed_actions[key]}"

            if confirm:
                pending_msg = require_confirmation(ctx, name, kwargs)
                if pending_msg:
                    return pending_msg

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
                except ToolError:  # a real answer for the user (e.g. a rule said no), not a failure
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


# ------------------------------------------------------------------ tools


@function_tool
async def get_current_datetime(ctx: RunContext[SessionState]) -> str:
    """Get the current date, time, weekday and timezone. Call this for any question about today's
    date, the time, the day of the week, or how long until or since a date."""
    tz_name = load_settings().timezone
    now = datetime.now(ZoneInfo(tz_name)) if tz_name else datetime.now().astimezone()
    return now.strftime("%A, %d %B %Y, %I:%M %p, %Z time")


@function_tool
async def end_call(ctx: RunContext[SessionState], farewell: str) -> None:
    """End the conversation. Call this whenever the user says goodbye or bye, "that's all",
    "stop", "stop it", "end this" or "I'm done". Saying goodbye without calling this leaves
    the session open.

    Args:
        farewell: One short goodbye sentence to say before ending.
    """
    state = ctx.userdata
    if state.ending:
        return None
    state.ending = True
    say_then_hangup(ctx.session, farewell)
    return None


def say_then_hangup(session: Any, text: str) -> None:
    """Queue the last words and hang up after they finish playing.

    Must not be awaited inside the tool: the queued speech plays only after the tool's
    own turn completes, so awaiting it from the tool would deadlock.
    """
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
