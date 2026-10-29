"""Worker entrypoint: one AgentSession per room (= per call).

Run:
  uv run python -m voice_agent.main console   # talk to it in the terminal
  uv run python -m voice_agent.main dev       # connect to LiveKit, hot reload
  uv run python -m voice_agent.main start     # production
"""

from __future__ import annotations

import asyncio
import logging
import time

from dotenv import load_dotenv

load_dotenv()

from livekit import rtc  # noqa: E402
from livekit.agents import (  # noqa: E402
    AgentServer,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    cli,
)
from livekit.agents.voice import room_io  # noqa: E402

from .agent import VoiceAgent  # noqa: E402
from .config import load_settings  # noqa: E402
from .normalize import speech_normalizer  # noqa: E402
from .pipeline import build_llm, build_stt, build_tts, build_turn_handling, build_vad  # noqa: E402
from .prompts import (  # noqa: E402
    HOLD_EXPIRED_PROMPT,
    MAX_DURATION_PROMPT,
    SILENCE_GOODBYE,
    SILENCE_PROMPT,
    STT_TIMEOUT_PROMPT,
)
from .state import CheckpointStore, SessionState  # noqa: E402
from .telemetry import SessionTelemetry, redact, setup_logging  # noqa: E402
from .tools import TELEMETRY  # noqa: E402

logger = logging.getLogger("voice-agent")
server = AgentServer()


@server.rtc_session(agent_name="voice-agent")
async def entrypoint(ctx: JobContext) -> None:
    setup_logging()
    settings = load_settings()
    call_id = ctx.room.name
    store = CheckpointStore(settings.redis_url)
    telemetry = SessionTelemetry(call_id)
    TELEMETRY[call_id] = telemetry

    # ---- resume after a worker crash/failover if a checkpoint exists for this room
    restored = await store.load(call_id)
    if restored:
        state, chat = restored
        agent = VoiceAgent(settings, chat_ctx=ChatContext.from_dict({"items": chat}), resumed=True)
        logger.info("resuming call from checkpoint", extra={"json": {"call": call_id}})
    else:
        state = SessionState(call_id=call_id)
        agent = VoiceAgent(settings)

    vad = build_vad()
    session: AgentSession[SessionState] = AgentSession(
        vad=vad,
        stt=build_stt(settings, vad),
        llm=build_llm(settings),
        tts=build_tts(settings),
        turn_handling=build_turn_handling(settings),
        userdata=state,
        max_tool_steps=settings.max_tool_steps,
        tts_text_transforms=[
            "filter_markdown",
            "filter_emoji",
            speech_normalizer(date_order=settings.date_order, lang=settings.language),
        ],
        user_away_timeout=settings.user_away_timeout,
        transcription_timeout=settings.transcription_timeout,
    )

    # ------------------------------------------------------------------ telemetry
    @session.on("metrics_collected")
    def _on_metrics(ev) -> None:
        telemetry.on_metrics(ev.metrics)

    @session.on("agent_false_interruption")
    def _on_false_interruption(ev) -> None:
        telemetry.false_interruptions += 1

    @session.on("error")
    def _on_error(ev) -> None:
        recoverable = getattr(ev.error, "recoverable", True)
        logger.error(
            "pipeline error",
            extra={
                "json": {
                    "call": call_id,
                    "source": type(ev.source).__name__,
                    "recoverable": recoverable,
                    "error": redact(str(ev.error)),
                }
            },
        )

    # ------------------------------------------------------------------ per-turn bookkeeping
    @session.on("conversation_item_added")
    def _on_item(ev) -> None:
        item = ev.item
        if not isinstance(item, ChatMessage):
            return
        if item.role == "user":
            logger.info("user", extra={"json": {"call": call_id, "text": redact(item.text_content or "")}})
            return
        if item.role != "assistant":
            return
        telemetry.on_assistant_message(item.metrics.get("e2e_latency"), bool(item.interrupted))
        logger.info(
            "agent",
            extra={
                "json": {
                    "call": call_id,
                    "text": redact(item.text_content or ""),
                    "interrupted": bool(item.interrupted),
                }
            },
        )
        # checkpoint after every agent turn so a replacement worker can resume the call
        chat = agent.chat_ctx.to_dict(exclude_image=True, exclude_audio=True)["items"]
        asyncio.create_task(store.save(session.userdata, chat))
        agent.maybe_compact()

    @session.on("user_transcription_timeout")
    def _on_stt_timeout(ev) -> None:
        # VAD heard speech but STT produced nothing (STT hiccup / audio withheld)
        telemetry.stt_timeouts += 1
        if not state.ending:
            session.say(STT_TIMEOUT_PROMPT)

    # ------------------------------------------------------------------ silence handling
    silence_task: asyncio.Task | None = None

    async def silence_watch() -> None:
        """Runs while the caller is silent ('away'): re-prompt, then hang up."""
        while not state.ending:
            if time.time() < state.on_hold_until:  # caller asked us to wait
                await asyncio.sleep(state.on_hold_until - time.time())
                if session.user_state == "away" and not state.ending:
                    state.on_hold_until = 0.0
                    await session.say(HOLD_EXPIRED_PROMPT)
                    await asyncio.sleep(settings.user_away_timeout)
                continue
            if state.silence_prompts >= settings.max_silence_prompts:
                state.ending = True
                await session.say(SILENCE_GOODBYE, allow_interruptions=False)
                session.shutdown()
                ctx.delete_room()
                return
            state.silence_prompts += 1
            await session.say(SILENCE_PROMPT)
            await asyncio.sleep(settings.user_away_timeout)

    @session.on("user_state_changed")
    def _on_user_state(ev) -> None:
        nonlocal silence_task
        if ev.new_state == "away" and session.agent_state != "speaking":
            if silence_task is None or silence_task.done():
                silence_task = asyncio.create_task(silence_watch())
        elif ev.new_state == "speaking":
            if silence_task and not silence_task.done():
                silence_task.cancel()
            state.silence_prompts = 0

    # ------------------------------------------------------------------ max call duration
    async def duration_guard() -> None:
        remaining = settings.max_call_duration - (time.time() - state.started_at)
        await asyncio.sleep(max(0.0, remaining - 60))
        if not state.ending:
            await session.generate_reply(instructions=MAX_DURATION_PROMPT)
        await asyncio.sleep(60)
        if not state.ending:
            state.ending = True
            await session.say(
                "I have to end the call now. Thanks for calling, goodbye!", allow_interruptions=False
            )
            session.shutdown()
            ctx.delete_room()

    guard = asyncio.create_task(duration_guard())

    # ------------------------------------------------------------------ shutdown
    async def on_shutdown() -> None:
        guard.cancel()
        if silence_task:
            silence_task.cancel()
        logger.info("session summary", extra={"json": telemetry.summary()})
        TELEMETRY.pop(call_id, None)
        if state.ending or state.transferred:  # finished normally: nothing to resume
            await store.delete(call_id)

    ctx.add_shutdown_callback(on_shutdown)

    audio_input: room_io.AudioInputOptions | bool = True
    if settings.noise_cancellation:  # LiveKit Cloud enhanced NC (BVC); telephony-tuned for SIP
        from livekit.plugins import noise_cancellation

        audio_input = room_io.AudioInputOptions(
            noise_cancellation=lambda params: (
                noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC()
            ),
            auto_gain_control=True,  # soft/loud talkers
        )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(audio_input=audio_input, close_on_disconnect=True),
    )


if __name__ == "__main__":
    cli.run_app(server)
