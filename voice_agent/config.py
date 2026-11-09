"""Environment-driven settings. Every tunable of the pipeline lives here so that
latency/robustness trade-offs can be changed per deployment without code edits."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


def _list(name: str, default: str) -> list[str]:
    return [p.strip() for p in _env(name, default).split(",") if p.strip()]


@dataclass(frozen=True)
class Settings:
    # --- model chains: "provider:model", first = primary, rest = failover order ---
    stt_chain: list[str] = field(
        default_factory=lambda: _list("STT_CHAIN", "deepgram:nova-3,openai:gpt-4o-mini-transcribe")
    )
    llm_chain: list[str] = field(
        default_factory=lambda: _list(
            "LLM_CHAIN", "ollama:qwen3:8b,anthropic:claude-haiku-4-5,openai:gpt-4.1-mini"
        )
    )
    tts_chain: list[str] = field(
        default_factory=lambda: _list(
            "TTS_CHAIN",
            "deepgram:aura-2-andromeda-en,cartesia:sonic-3,elevenlabs:eleven_flash_v2_5,openai:gpt-4o-mini-tts",
        )
    )
    language: str = field(default_factory=lambda: _env("AGENT_LANGUAGE", "en"))
    keyterms: list[str] = field(default_factory=lambda: _list("STT_KEYTERMS", ""))

    # --- turn taking ---
    # "semantic": LiveKit streaming end-of-turn model (cloud v1, local mini fallback)
    # "stt":      STT-integrated end-of-turn (use with deepgram:flux-general-en)
    # "vad":      silence only (fallback / debugging)
    turn_detection: str = field(default_factory=lambda: _env("TURN_DETECTION", "semantic"))
    endpointing_mode: str = field(default_factory=lambda: _env("ENDPOINTING_MODE", "dynamic"))
    min_endpointing_delay: float = field(default_factory=lambda: _float("MIN_ENDPOINTING_DELAY", 0.3))
    max_endpointing_delay: float = field(default_factory=lambda: _float("MAX_ENDPOINTING_DELAY", 2.5))
    preemptive_generation: bool = field(default_factory=lambda: _bool("PREEMPTIVE_GENERATION", True))
    preemptive_tts: bool = field(default_factory=lambda: _bool("PREEMPTIVE_TTS", True))

    # --- interruptions (barge-in) ---
    min_interruption_duration: float = field(
        default_factory=lambda: _float("MIN_INTERRUPTION_DURATION", 0.35)
    )
    false_interruption_timeout: float = field(
        default_factory=lambda: _float("FALSE_INTERRUPTION_TIMEOUT", 1.5)
    )

    # --- silence / call lifecycle ---
    user_away_timeout: float = field(default_factory=lambda: _float("USER_AWAY_TIMEOUT", 8.0))
    # 0 = never nag a quiet user (assistant mode); >0 = "still there?" prompts, then end the session
    max_silence_prompts: int = field(default_factory=lambda: int(_env("MAX_SILENCE_PROMPTS", "0")))
    max_call_duration: float = field(default_factory=lambda: _float("MAX_CALL_DURATION", 1800.0))
    transcription_timeout: float = field(default_factory=lambda: _float("TRANSCRIPTION_TIMEOUT", 3.0))

    # --- tools ---
    tool_timeout: float = field(default_factory=lambda: _float("TOOL_TIMEOUT", 8.0))
    tool_filler_delay: float = field(default_factory=lambda: _float("TOOL_FILLER_DELAY", 0.7))
    max_tool_steps: int = field(default_factory=lambda: int(_env("MAX_TOOL_STEPS", "4")))

    # --- context management ---
    summarize_after_items: int = field(default_factory=lambda: int(_env("SUMMARIZE_AFTER_ITEMS", "30")))
    keep_recent_items: int = field(default_factory=lambda: int(_env("KEEP_RECENT_ITEMS", "12")))

    # --- infra ---
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", ""))
    noise_cancellation: bool = field(default_factory=lambda: _bool("NOISE_CANCELLATION", False))
    date_order: str = field(default_factory=lambda: _env("DATE_ORDER", "DMY"))
    agent_name: str = field(default_factory=lambda: _env("AGENT_NAME", "Ava"))
    timezone: str = field(
        default_factory=lambda: _env("AGENT_TIMEZONE", "")
    )  # e.g. Asia/Kolkata; "" = system


def load_settings() -> Settings:
    return Settings()
