"""End-to-end agent behaviour through a real AgentSession (text mode) with a scripted LLM."""

from __future__ import annotations

import asyncio

import pytest
from livekit.agents import AgentSession

from voice_agent import backend
from voice_agent.agent import VoiceAgent, ignore_reason
from voice_agent.config import load_settings
from voice_agent.state import SessionState

from .fakes import ScriptedLLM, ToolCall


@pytest.fixture(autouse=True)
def reset_backend():
    backend._ORDERS["12345"].status = "processing"
    yield
    backend._ORDERS["12345"].status = "processing"


def _tool_outputs(result) -> list[str]:
    return [e.item.output for e in result.events if e.type == "function_call_output"]


async def _session(fake: ScriptedLLM, monkeypatch=None) -> tuple[AgentSession, SessionState]:
    state = SessionState(call_id="test")
    session = AgentSession(llm=fake, vad=None, userdata=state, turn_handling={"turn_detection": "manual"})
    return session, state


async def _start(session: AgentSession) -> None:
    agent = VoiceAgent(load_settings())

    async def no_greeting() -> None:  # greeting needs TTS; not under test here
        return None

    agent.on_enter = no_greeting  # type: ignore[method-assign]
    await session.start(agent)


async def test_risky_action_needs_a_user_turn_before_executing():
    fake = ScriptedLLM(
        [
            # turn 1: model tries to cancel immediately, then tries AGAIN in the same turn
            ToolCall("cancel_order", order_id="12345"),
            ToolCall("cancel_order", order_id="12345"),
            "Just to confirm, cancel order one two three four five?",
            # turn 2: user said yes -> now it may execute
            ToolCall("cancel_order", order_id="12345"),
            "Done, it's cancelled.",
            # turn 3: a repeated "yes" must not run the action twice
            ToolCall("cancel_order", order_id="12345"),
            "That order is already cancelled.",
        ]
    )
    session, state = await _session(fake)
    async with session:
        await _start(session)

        r1 = await session.run(user_input="Cancel my order one two three four five")
        outs = _tool_outputs(r1)
        assert len(outs) == 2 and all("NOT EXECUTED" in o for o in outs)
        assert backend._ORDERS["12345"].status == "processing"

        r2 = await session.run(user_input="Yes, cancel it")
        assert "has been cancelled" in _tool_outputs(r2)[0]
        assert backend._ORDERS["12345"].status == "cancelled"

        r3 = await session.run(user_input="yes")
        assert _tool_outputs(r3)[0].startswith("Already completed")


async def test_spoken_order_number_is_normalized_for_lookup():
    fake = ScriptedLLM([ToolCall("get_order_status", order_id="four five six seven eight"), "It shipped."])
    session, state = await _session(fake)
    async with session:
        await _start(session)
        r = await session.run(user_input="Where is order four five six seven eight?")
        assert "Order 45678: status shipped" in _tool_outputs(r)[0]
        assert state.slots["order_id"] == "45678"


async def test_business_rule_rejection_is_relayed_not_executed():
    fake = ScriptedLLM(
        [
            ToolCall("cancel_order", order_id="45678"),
            "Confirm?",
            ToolCall("cancel_order", order_id="45678"),
            "Sorry, it already shipped.",
        ]
    )
    session, _ = await _session(fake)
    async with session:
        await _start(session)
        await session.run(user_input="cancel order 45678")
        r = await session.run(user_input="yes")
        ev = next(e for e in r.events if e.type == "function_call_output")
        assert ev.item.is_error and "already shipped" in ev.item.output


async def test_slow_tool_times_out_with_friendly_error(monkeypatch):
    monkeypatch.setenv("TOOL_TIMEOUT", "0.2")

    async def slow(order_id: str):
        await asyncio.sleep(5)

    monkeypatch.setattr(backend, "get_order", slow)
    fake = ScriptedLLM([ToolCall("get_order_status", order_id="45678"), "Sorry, the system is slow."])
    session, _ = await _session(fake)
    async with session:
        await _start(session)
        r = await session.run(user_input="status of 45678")
        ev = next(e for e in r.events if e.type == "function_call_output")
        assert ev.item.is_error
        assert "didn't respond in time" in ev.item.output
        assert "Timeout" not in ev.item.output  # no technical details leak to the caller


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
