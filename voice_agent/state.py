"""Per-call structured state plus checkpointing for failover.

The conversation history lives in the framework's ChatContext; this holds the
structured side (slots, confirmations, executed actions) that tools rely on.
Both are checkpointed after every turn so a replacement worker can resume a
call in the same room with no repeated questions and no duplicated actions.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("voice-agent")

CHECKPOINT_TTL_S = 2 * 60 * 60


@dataclass
class PendingConfirmation:
    key: str
    summary: str
    user_turn: int  # user turn index when the confirmation was requested


@dataclass
class SessionState:
    call_id: str = ""
    caller: str | None = None
    started_at: float = field(default_factory=time.time)
    user_turns: int = 0
    slots: dict[str, Any] = field(default_factory=dict)
    pending_confirmations: dict[str, PendingConfirmation] = field(default_factory=dict)
    # idempotency key -> result of a side-effecting tool that already ran
    completed_actions: dict[str, str] = field(default_factory=dict)
    on_hold_until: float = 0.0
    silence_prompts: int = 0
    transferred: bool = False
    ending: bool = False
    conversation_summary: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> SessionState:
        data = json.loads(raw)
        data["pending_confirmations"] = {
            k: PendingConfirmation(**v) for k, v in data.get("pending_confirmations", {}).items()
        }
        return cls(**data)


class CheckpointStore:
    """Redis-backed when REDIS_URL is set; otherwise a no-op (single-node dev)."""

    def __init__(self, redis_url: str) -> None:
        self._redis = None
        if redis_url:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(redis_url, socket_timeout=0.5)

    @staticmethod
    def _key(call_id: str) -> str:
        return f"voice-agent:checkpoint:{call_id}"

    async def save(self, state: SessionState, chat_items: list[dict[str, Any]]) -> None:
        if self._redis is None:
            return
        payload = json.dumps({"state": state.to_json(), "chat": chat_items})
        try:
            await self._redis.set(self._key(state.call_id), payload, ex=CHECKPOINT_TTL_S)
        except Exception:  # a checkpoint failure must never break the live call
            logger.warning("checkpoint save failed", exc_info=True)

    async def load(self, call_id: str) -> tuple[SessionState, list[dict[str, Any]]] | None:
        if self._redis is None:
            return None
        try:
            raw = await self._redis.get(self._key(call_id))
        except Exception:
            logger.warning("checkpoint load failed", exc_info=True)
            return None
        if not raw:
            return None
        data = json.loads(raw)
        return SessionState.from_json(data["state"]), data["chat"]

    async def delete(self, call_id: str) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.delete(self._key(call_id))
        except Exception:
            logger.warning("checkpoint delete failed", exc_info=True)
