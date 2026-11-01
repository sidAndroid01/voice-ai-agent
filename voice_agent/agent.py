"""The conversational agent: turn hooks, echo/noise guards, context compaction, greeting cache."""

from __future__ import annotations

import asyncio
import difflib
import logging
import re

from livekit import rtc
from livekit.agents import Agent, ChatContext, ChatMessage, StopResponse, llm

from .config import Settings
from .prompts import GREETING, system_prompt
from .state import SessionState
from .tools import ALL_TOOLS

logger = logging.getLogger("voice-agent")

# text -> synthesized frames; shared by every call in this worker process
_GREETING_CACHE: dict[str, list[rtc.AudioFrame]] = {}

_SUMMARY_INSTRUCTIONS = (
    "Summarize this earlier part of a customer support voice conversation in at most 5 short sentences. "
    "Keep every concrete fact: names, order numbers, amounts, dates, decisions, actions taken, "
    "and anything still unresolved. Output plain text only."
)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def ignore_reason(user_text: str, last_agent_text: str | None) -> str | None:
    """Why a finalized user transcript should not get a reply, or None if it should."""
    # Noise that made it through VAD + STT as punctuation or nothing.
    if not re.search(r"\w", user_text):
        return "noise"
    # Echo: if AEC failed (speakerphone, bad device), STT transcribes the agent's own voice.
    # A user turn that closely matches what the agent just said is not real input.
    if last_agent_text and len(user_text.split()) >= 4:
        said, heard = _norm(last_agent_text), _norm(user_text)
        if heard and (heard in said or difflib.SequenceMatcher(None, heard, said).ratio() > 0.85):
            return "echo"
    return None


class VoiceAgent(Agent):
    def __init__(self, settings: Settings, *, chat_ctx: ChatContext | None = None, resumed: bool = False):
        super().__init__(instructions=system_prompt(settings), tools=ALL_TOOLS, chat_ctx=chat_ctx)
        self._settings = settings
        self._resumed = resumed
        self._compacting = False

    # ------------------------------------------------------------------ lifecycle

    async def on_enter(self) -> None:
        if self._resumed:
            # failover: the caller heard a short gap, not a dropped call
            await self.session.generate_reply(
                instructions="The call briefly glitched and you are resuming. Say a very short "
                "'Sorry about that' and continue exactly where the conversation left off."
            )
            return
        text = GREETING.format(company=self._settings.company_name, agent=self._settings.agent_name)
        frames = _GREETING_CACHE.get(text)
        if frames:
            self.session.say(text, audio=_replay(frames), allow_interruptions=True)
        else:
            self.session.say(text, allow_interruptions=True)
            asyncio.create_task(self._cache_greeting(text))

    async def _cache_greeting(self, text: str) -> None:
        try:
            frames = [ev.frame async for ev in self.session.tts.synthesize(text)]  # type: ignore[union-attr]
            _GREETING_CACHE[text] = frames
        except Exception:
            logger.debug("greeting cache warmup failed", exc_info=True)

    # ------------------------------------------------------------------ turn hook

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: ChatMessage) -> None:
        state: SessionState = self.session.userdata
        text = (new_message.text_content or "").strip()

        last_agent = next(
            (m for m in reversed(turn_ctx.items) if getattr(m, "role", None) == "assistant"), None
        )
        reason = ignore_reason(text, last_agent.text_content if last_agent else None)
        if reason:
            logger.info("ignored user turn", extra={"json": {"reason": reason}})
            raise StopResponse()

        state.user_turns += 1
        state.silence_prompts = 0
        state.on_hold_until = 0.0

    # ------------------------------------------------------------------ context compaction

    def maybe_compact(self) -> None:
        """Summarize older turns once history grows, keeping latency and cost flat on long calls.
        Runs in the background after a reply, never on the critical path."""
        if self._compacting or len(self.chat_ctx.items) <= self._settings.summarize_after_items:
            return
        self._compacting = True
        asyncio.create_task(self._compact())

    async def _compact(self) -> None:
        try:
            ctx = self.chat_ctx.copy()
            items = ctx.items
            head = [i for i in items[:2] if getattr(i, "role", None) in ("system", "developer")]
            body = items[len(head) :]
            cut = len(body) - self._settings.keep_recent_items
            # never split a tool call from its output: start the kept tail on a user message
            while cut < len(body) and getattr(body[cut], "role", None) != "user":
                cut += 1
            if cut <= 0 or cut >= len(body):
                return
            old, recent = body[:cut], body[cut:]
            transcript = "\n".join(
                f"{m.role}: {m.text_content}"
                for m in old
                if getattr(m, "type", "") == "message" and m.text_content
            )
            state: SessionState = self.session.userdata
            if state.conversation_summary:
                transcript = f"Earlier summary: {state.conversation_summary}\n{transcript}"

            req = ChatContext.empty()
            req.add_message(role="system", content=_SUMMARY_INSTRUCTIONS)
            req.add_message(role="user", content=transcript)
            summary = ""
            async with self.session.llm.chat(chat_ctx=req) as stream:  # type: ignore[union-attr]
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        summary += chunk.delta.content
            summary = summary.strip()
            if not summary:
                return

            state.conversation_summary = summary
            # merge with anything appended while we were summarizing
            live = self.chat_ctx.copy()
            appended = live.items[len(items) :]
            new_ctx = ChatContext(list(head))
            new_ctx.add_message(role="system", content=f"Summary of the call so far: {summary}")
            new_ctx.items.extend(recent + appended)
            await self.update_chat_ctx(new_ctx)
            logger.info("context compacted", extra={"json": {"from": len(items), "to": len(new_ctx.items)}})
        except Exception:
            logger.warning("context compaction failed", exc_info=True)
        finally:
            self._compacting = False


async def _replay(frames: list[rtc.AudioFrame]):
    for f in frames:
        yield f
