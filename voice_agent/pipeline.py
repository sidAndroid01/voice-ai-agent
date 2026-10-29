"""Builds the cascaded STT -> LLM -> TTS pipeline with provider failover.

Each stage is a chain "provider:model,provider:model,...". Providers whose API key
is missing are skipped, so the same config works in dev (one key) and prod (all keys).
"livekit:<provider/model>" routes through LiveKit Inference (only LiveKit keys needed).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from livekit.agents import inference, llm, stt, tts, vad
from livekit.agents.voice.turn import TurnHandlingOptions

from .config import Settings

logger = logging.getLogger("voice-agent")

_KEYS = {
    "deepgram": "DEEPGRAM_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cartesia": "CARTESIA_API_KEY",
    "elevenlabs": "ELEVEN_API_KEY",
    "livekit": "LIVEKIT_API_KEY",
}


def _available(spec: str) -> tuple[str, str] | None:
    provider, _, model = spec.partition(":")
    env = _KEYS.get(provider)
    if env is None:
        raise ValueError(f"unknown provider '{provider}' in '{spec}'")
    if not os.getenv(env):
        logger.info("skipping %s (%s not set)", spec, env)
        return None
    return provider, model


def _build_chain(specs: list[str], make: Callable[[str, str], Any], stage: str) -> list[Any]:
    out = [make(p, m) for p, m in filter(None, map(_available, specs))]
    if not out:
        raise RuntimeError(f"No usable {stage} provider in {specs}. Set an API key for at least one of them.")
    return out


def is_flux(s: Settings) -> bool:
    first = next(filter(None, map(_available, s.stt_chain)), None)
    return bool(first and first[0] == "deepgram" and first[1].startswith("flux"))


def build_vad() -> vad.VAD:
    # Silero, run locally (<1 ms per frame). Shared by the session and the STT fallback.
    return inference.VAD(model="silero")


def build_stt(s: Settings, vad_: vad.VAD | None = None) -> stt.STT:
    def make(provider: str, model: str) -> stt.STT:
        if provider == "deepgram":
            from livekit.plugins import deepgram

            if model.startswith("flux"):  # STT-integrated end-of-turn detection
                return deepgram.STTv2(model=model, keyterm=s.keyterms or [], eager_eot_threshold=0.5)
            return deepgram.STT(
                model=model,
                language="multi" if s.language == "multi" else s.language,
                interim_results=True,
                smart_format=False,  # we want raw words; normalization happens downstream
                filler_words=True,  # "um"/"uh" help the turn detector see an unfinished thought
                endpointing_ms=25,  # let VAD + turn detector own end-of-turn
                keyterm=s.keyterms or [],
            )
        if provider == "openai":
            from livekit.plugins import openai

            return openai.STT(model=model, language=s.language)
        if provider == "livekit":
            return inference.STT.from_model_string(model)
        raise ValueError(f"unsupported STT provider {provider}")

    chain = _build_chain(s.stt_chain, make, "STT")
    return (
        chain[0]
        if len(chain) == 1
        else stt.FallbackAdapter(
            chain,
            vad=vad_ or build_vad(),
            attempt_timeout=5.0,  # VAD segments audio for non-streaming STTs
        )
    )


def build_llm(s: Settings) -> llm.LLM:
    def make(provider: str, model: str) -> llm.LLM:
        if provider == "anthropic":
            from livekit.plugins import anthropic

            return anthropic.LLM(model=model, temperature=0.4, caching="ephemeral")
        if provider == "openai":
            from livekit.plugins import openai

            return openai.LLM(model=model, temperature=0.4)
        if provider == "livekit":
            return inference.llm_from_model_string(model)
        raise ValueError(f"unsupported LLM provider {provider}")

    chain = _build_chain(s.llm_chain, make, "LLM")
    # attempt_timeout bounds time-to-first-token on the primary before failing over;
    # never retry once audio has started, or the caller hears the reply restart.
    return (
        chain[0]
        if len(chain) == 1
        else llm.FallbackAdapter(chain, attempt_timeout=3.0, retry_on_chunk_sent=False)
    )


def build_tts(s: Settings) -> tts.TTS:
    def make(provider: str, model: str) -> tts.TTS:
        if provider == "cartesia":
            from livekit.plugins import cartesia

            return cartesia.TTS(model=model, language=s.language if s.language != "multi" else "en")
        if provider == "elevenlabs":
            from livekit.plugins import elevenlabs

            return elevenlabs.TTS(model=model)
        if provider == "openai":
            from livekit.plugins import openai

            return openai.TTS(model=model)
        if provider == "livekit":
            return inference.TTS.from_model_string(model)
        raise ValueError(f"unsupported TTS provider {provider}")

    chain = _build_chain(s.tts_chain, make, "TTS")
    return chain[0] if len(chain) == 1 else tts.FallbackAdapter(chain, max_retry_per_tts=1)


def build_turn_handling(s: Settings) -> TurnHandlingOptions:
    if s.turn_detection == "stt" or (s.turn_detection == "semantic" and is_flux(s)):
        turn_detection: Any = "stt"
    elif s.turn_detection == "vad":
        turn_detection = "vad"
    else:
        # streaming semantic end-of-turn model: cloud v1 with automatic local fallback
        turn_detection = inference.TurnDetector()

    return TurnHandlingOptions(
        turn_detection=turn_detection,
        endpointing={
            "mode": s.endpointing_mode,  # "dynamic" learns each caller's pause length
            "min_delay": s.min_endpointing_delay,
            "max_delay": s.max_endpointing_delay,
        },
        interruption={
            "enabled": True,
            # adaptive (ML) barge-in separates backchannels ("mm-hmm") from real
            # interruptions when available; the framework falls back to VAD otherwise
            "min_duration": s.min_interruption_duration,
            "resume_false_interruption": True,  # cough/noise: resume instead of going silent
            "false_interruption_timeout": s.false_interruption_timeout,
        },
        preemptive_generation={
            "enabled": s.preemptive_generation,  # start LLM before end-of-turn is confirmed
            "preemptive_tts": s.preemptive_tts,
            "max_speech_duration": 10.0,
        },
    )
