"""End-to-end agent behaviour through a real AgentSession (text mode) with a scripted LLM."""

from __future__ import annotations

import asyncio

import pytest
from livekit.agents import AgentSession, RunContext, ToolError

from voice_agent.agent import VoiceAgent, ignore_reason
from voice_agent.config import load_settings
from voice_agent.state import SessionState
from voice_agent.tools import voice_tool

from .fakes import ScriptedLLM, ToolCall

# Test-only tools: the voice_tool safety layer (confirmation, idempotency, timeouts) is generic
# and kept for future real-world tools (smart home, messages, calendar...).
SENT: list[str] = []
SLOW = {"seconds": 0.0}


@voice_tool(side_effect=True, confirm=True, filler=None)
async def send_message(ctx: RunContext[SessionState], to: str, text: str) -> str:
    """Send a text message. Irreversible.

    Args:
        to: Recipient name.
        text: Message text.
    """
    SENT.append(f"{to}:{text}")
    return f"Message sent to {to}."


@voice_tool(filler=None)
async def lookup(ctx: RunContext[SessionState], query: str) -> str:
    """Look something up.

    Args:
        query: What to look up.
    """
    await asyncio.sleep(SLOW["seconds"])
    if query == "forbidden":
        raise ToolError("That isn't allowed.")
    return f"result for {query}"


@pytest.fixture(autouse=True)
def reset():
    SENT.clear()
    SLOW["seconds"] = 0.0
    yield


def _tool_outputs(result) -> list[str]:
    return [e.item.output for e in result.events if e.type == "function_call_output"]


async def _session(fake: ScriptedLLM) -> tuple[AgentSession, SessionState]:
    state = SessionState(call_id="test")
    session = AgentSession(llm=fake, vad=None, userdata=state, turn_handling={"turn_detection": "manual"})
    return session, state


async def _start(session: AgentSession) -> VoiceAgent:
    agent = VoiceAgent(load_settings())

    async def no_greeting() -> None:  # greeting needs TTS; not under test here
        return None

    agent.on_enter = no_greeting  # type: ignore[method-assign]
    await agent.update_tools([*agent.tools, send_message, lookup])
    await session.start(agent)
    return agent


async def test_risky_action_needs_a_user_turn_before_executing():
    fake = ScriptedLLM(
        [
            # turn 1: model tries to act immediately, then tries AGAIN in the same turn
            ToolCall("send_message", to="Mom", text="late"),
            ToolCall("send_message", to="Mom", text="late"),
            "Just to confirm, send Mom 'late'?",
            # turn 2: user said yes -> now it may execute
            ToolCall("send_message", to="Mom", text="late"),
            "Sent.",
            # turn 3: a repeated "yes" must not run the action twice
            ToolCall("send_message", to="Mom", text="late"),
            "Already sent.",
        ]
    )
    session, _ = await _session(fake)
    async with session:
        await _start(session)

        r1 = await session.run(user_input="Text Mom that I'm late")
        outs = _tool_outputs(r1)
        assert len(outs) == 2 and all("NOT EXECUTED" in o for o in outs)
        assert SENT == []

        r2 = await session.run(user_input="Yes, send it")
        assert "Message sent" in _tool_outputs(r2)[0]
        assert SENT == ["Mom:late"]

        r3 = await session.run(user_input="yes")
        assert _tool_outputs(r3)[0].startswith("Already completed")
        assert SENT == ["Mom:late"]


async def test_current_datetime_tool_returns_real_date():
    from datetime import datetime

    fake = ScriptedLLM([ToolCall("get_current_datetime"), "It's today."])
    session, _ = await _session(fake)
    async with session:
        await _start(session)
        r = await session.run(user_input="What's the date today?")
        assert str(datetime.now().year) in _tool_outputs(r)[0]


async def test_tool_rule_rejection_is_relayed_as_error():
    fake = ScriptedLLM([ToolCall("lookup", query="forbidden"), "Sorry, I can't do that."])
    session, _ = await _session(fake)
    async with session:
        await _start(session)
        r = await session.run(user_input="look up forbidden")
        ev = next(e for e in r.events if e.type == "function_call_output")
        assert ev.item.is_error and "isn't allowed" in ev.item.output


async def test_slow_tool_times_out_with_friendly_error(monkeypatch):
    monkeypatch.setenv("TOOL_TIMEOUT", "0.2")
    SLOW["seconds"] = 5
    fake = ScriptedLLM([ToolCall("lookup", query="x"), "Sorry, that's slow."])
    session, _ = await _session(fake)
    async with session:
        await _start(session)
        r = await session.run(user_input="look up x")
        ev = next(e for e in r.events if e.type == "function_call_output")
        assert ev.item.is_error
        assert "didn't respond in time" in ev.item.output
        assert "Timeout" not in ev.item.output  # no technical details leak to the user


@pytest.mark.parametrize(
    "user, agent, reason",
    [
        ("...", None, "noise"),
        ("  ", "Hello", "noise"),
        # mic picked up the agent's own voice (speakerphone / failed AEC)
        (
            "your order four five six seven eight shipped today",
            "Your order four five six seven eight shipped today and arrives Friday.",
            "echo",
        ),
        (
            "Your order 45678 shipped today and arrives friday",
            "Your order 45678 shipped today and arrives Friday.",
            "echo",
        ),
        # real answers that reuse the agent's words must still get a reply
        ("yes my order number is four five six", "Is your order number four five six seven eight?", None),
        ("yes", "Should I cancel order one two three four five?", None),
        ("no thanks that's all", None, None),
    ],
)
def test_ignore_reason(user, agent, reason):
    assert ignore_reason(user, agent) == reason


async def test_end_call_says_farewell_and_shuts_down_without_deadlock():
    fake = ScriptedLLM([ToolCall("end_call", farewell="Thanks for calling, goodbye!")])
    session, state = await _session(fake)
    closed = asyncio.Event()
    session.on("close", lambda ev: closed.set())
    async with session:
        await _start(session)
        r = await asyncio.wait_for(session.run(user_input="that's all, bye"), timeout=5)
        assert state.ending
        said = [e.item.text_content for e in r.events if e.type == "message"]
        assert "Thanks for calling, goodbye!" in said
        await asyncio.wait_for(closed.wait(), timeout=5)
        assert len(fake.requests) == 1  # no extra LLM reply after the farewell
