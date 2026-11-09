"""Per-turn latency tracing, session-level SLO summary, and PII redaction for logs.

Every agent reply gets one structured log line with its latency attribution:
  end-of-utterance wait -> STT finalization -> LLM TTFT -> TTS TTFB -> e2e
so a slow p95 can be traced to the stage that caused it.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from livekit.agents import metrics as lk_metrics

logger = logging.getLogger("voice-agent.telemetry")

# ---------------------------------------------------------------- PII redaction

_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{7,}\d(?![\w]|\.\d)")
_DECIMAL = re.compile(r"\d+\.\d+")  # timestamps, measurements: not phone numbers
_LONG_DIGITS = re.compile(r"(?<!\.)\b\d{6,}\b(?!\.\d)")
# Spoken digit runs coming out of STT: "four five six seven eight nine"
_SPOKEN_DIGITS = re.compile(
    r"\b(?:(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)[\s,-]+){5,}"
    r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)\b",
    re.I,
)


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def redact(text: str) -> str:
    """Strip card numbers, emails, phone numbers and long digit runs from text bound for logs."""

    def card(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "[CARD]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else m.group(0)

    text = _CARD.sub(card, text)
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PHONE.sub(lambda m: m.group(0) if _DECIMAL.fullmatch(m.group(0)) else "[PHONE]", text)
    text = _LONG_DIGITS.sub("[NUMBER]", text)
    text = _SPOKEN_DIGITS.sub("[NUMBER]", text)
    return text


# ---------------------------------------------------------------- turn tracing


@dataclass
class TurnTrace:
    speech_id: str
    eou_delay: float | None = None  # user stopped speaking -> turn committed
    transcription_delay: float | None = None  # user stopped speaking -> final transcript
    llm_ttft: float | None = None
    tts_ttfb: float | None = None
    e2e_latency: float | None = None  # measured by the framework on the assistant message
    llm_prompt_tokens: int = 0
    llm_cached_tokens: int = 0
    llm_completion_tokens: int = 0
    tts_chars: int = 0
    interrupted: bool = False
    tool_calls: list[str] = field(default_factory=list)


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1]


class SessionTelemetry:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.started_at = time.time()
        self._turns: dict[str, TurnTrace] = {}
        self._usage = lk_metrics.ModelUsageCollector()
        self.false_interruptions = 0
        self.interruptions = 0
        self.tool_failures = 0
        self.stt_timeouts = 0

    def _turn(self, speech_id: str | None) -> TurnTrace | None:
        if not speech_id:
            return None
        return self._turns.setdefault(speech_id, TurnTrace(speech_id))

    def on_metrics(self, m: Any) -> None:
        self._usage.collect(m)
        if isinstance(m, lk_metrics.EOUMetrics):
            t = self._turn(m.speech_id)
            if t:
                t.eou_delay = m.end_of_utterance_delay
                t.transcription_delay = m.transcription_delay
        elif isinstance(m, lk_metrics.LLMMetrics):
            t = self._turn(m.speech_id)
            if t and t.llm_ttft is None and not m.cancelled:
                t.llm_ttft = m.ttft
            if t:
                t.llm_prompt_tokens += m.prompt_tokens
                t.llm_cached_tokens += m.prompt_cached_tokens
                t.llm_completion_tokens += m.completion_tokens
        elif isinstance(m, lk_metrics.TTSMetrics):
            t = self._turn(m.speech_id)
            if t and t.tts_ttfb is None and not m.cancelled:
                t.tts_ttfb = m.ttfb
            if t:
                t.tts_chars += m.characters_count

    def on_assistant_message(self, e2e: float | None, interrupted: bool) -> None:
        # metrics for a reply arrive before its chat item, so the newest trace is this reply's
        if not self._turns:
            return
        t = next(reversed(self._turns.values()))
        if t.e2e_latency is not None:  # already logged (e.g. a second message from a tool step)
            return
        t.e2e_latency = e2e
        t.interrupted = interrupted
        if interrupted:
            self.interruptions += 1
        logger.info("turn", extra={"json": {"session": self.session_id, **_round(t.__dict__)}})

    def on_tool(self, speech_id: str | None, name: str) -> None:
        t = self._turn(speech_id)
        if t:
            t.tool_calls.append(name)

    def summary(self) -> dict[str, Any]:
        e2e = [t.e2e_latency for t in self._turns.values() if t.e2e_latency is not None]
        ttft = [t.llm_ttft for t in self._turns.values() if t.llm_ttft is not None]
        ttfb = [t.tts_ttfb for t in self._turns.values() if t.tts_ttfb is not None]
        eou = [t.eou_delay for t in self._turns.values() if t.eou_delay is not None]
        return _round(
            {
                "session": self.session_id,
                "duration_s": time.time() - self.started_at,
                "turns": len(e2e),
                "e2e_p50": _pct(e2e, 50),
                "e2e_p95": _pct(e2e, 95),
                "llm_ttft_p50": _pct(ttft, 50),
                "tts_ttfb_p50": _pct(ttfb, 50),
                "eou_delay_p50": _pct(eou, 50),
                "interruptions": self.interruptions,
                "false_interruptions": self.false_interruptions,
                "tool_failures": self.tool_failures,
                "stt_timeouts": self.stt_timeouts,
                "usage": [u.model_dump(exclude_none=True) for u in self._usage.flatten()],
            }
        )


def _round(d: dict[str, Any]) -> dict[str, Any]:
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}


class JsonFormatter(logging.Formatter):
    """Emit structured payloads (``extra={"json": {...}}``) as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "json", None)
        base = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if payload:
            base.update(payload)
        return json.dumps(base, default=str)


def setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    for name in ("voice-agent", "voice-agent.telemetry"):
        lg = logging.getLogger(name)
        lg.handlers[:] = [handler]
        lg.setLevel(logging.INFO)
        lg.propagate = False
