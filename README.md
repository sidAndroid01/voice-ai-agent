# Voice AI Agent

A real-time, streaming voice assistant for web and mobile apps. You talk, it listens, thinks and talks back in about half a second, and you can interrupt it at any point like a real person.

It is built on [LiveKit Agents](https://docs.livekit.io/agents/) as a **cascaded pipeline**: three separate services for listening (STT), thinking (LLM) and speaking (TTS), each with automatic failover to a backup provider.

The demo plays **"Ava"**, a customer-support agent for a fictional online store. The store backend is a mock (`voice_agent/backend.py`) with three sample orders and a few help articles. Replace it with your real APIs.

> **Status: v0 prototype.** The conversation engine is complete and covered by offline tests, but it has not yet been run on a real call or deployed. See [Roadmap](#roadmap). The full design is in [PLAN.md](PLAN.md).

---

## How it works

```
 Browser / Mobile app
        │  WebRTC (Opus audio, echo cancellation, noise suppression on device)
        ▼
   LiveKit (media server: codecs, jitter buffer)
        │  20 ms audio frames
        ▼
 ┌──────────────── Voice Agent (one session per conversation) ────────────────┐
 │                                                                            │
 │  VAD ──► Turn detection ──► STT ──► LLM + tools ──► text cleanup ──► TTS ──┼──► back to the user
 │  "is someone   "have they    speech   Claude decides   "$45.50" →    voice │
 │   speaking?"    finished?"   → text   what to say      "forty-five…"       │
 │                                                                            │
 │  Interruption handling · silence/hold timers · latency tracing · PII-safe  │
 │  logs · per-turn checkpoint to Redis (crash recovery)                      │
 └────────────────────────────────────────────────────────────────────────────┘
                                        │
                          Tools / backend (orders, help articles)
```

Every stage streams and stages overlap. The LLM starts drafting a reply *before* the user has definitely finished speaking, and TTS speaks the first clause while the LLM is still writing the next.

## Providers

| Stage | Primary → failover | Notes |
|---|---|---|
| VAD | Silero (local) | < 1 ms per frame |
| Turn detection | LiveKit semantic turn detector | Tells a pause from a finished thought |
| STT | Deepgram Nova-3 → OpenAI gpt-4o-mini-transcribe | Streaming, keyword boosting |
| LLM | **Local** Qwen3-8B via Ollama (thinking off) → Claude Haiku 4.5 → GPT-4.1-mini | Runs on your machine with no key; cloud models are optional failover |
| TTS | Deepgram Aura-2 → Cartesia Sonic-3 → ElevenLabs Flash → OpenAI | One Deepgram key covers STT and TTS |

Providers without an API key are skipped automatically, and any extra keys become failover. Each chain can be changed in `.env` without touching code (`STT_CHAIN`, `LLM_CHAIN`, `TTS_CHAIN`).

## Quick start

**Requirements:** Python 3.12, [uv](https://docs.astral.sh/uv/), a [Deepgram](https://deepgram.com) key (speech in and out), and [Ollama](https://ollama.com) with `ollama pull qwen3:8b` (local LLM). Console mode needs nothing else. A [LiveKit Cloud](https://cloud.livekit.io) project (free tier) or `livekit-server --dev` is needed only for browser and app clients.

```bash
git clone https://github.com/sidAndroid01/voice-ai-agent.git
cd voice-ai-agent

cp .env.example .env                          # then fill in your keys (this file is git-ignored)
uv sync                                       # install dependencies
uv run -m livekit.agents download-files       # one-time: local VAD + turn-detector models

uv run python -m voice_agent.main console     # talk to it from your terminal (mic + speakers)
uv run python -m voice_agent.main dev         # join LiveKit rooms; test at https://agents-playground.livekit.io
uv run pytest                                 # 42 offline tests, no API keys needed
```

Minimum `.env` for a working conversation:

```
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
DEEPGRAM_API_KEY=...      # STT
ANTHROPIC_API_KEY=...     # LLM
CARTESIA_API_KEY=...      # TTS
```

All other settings (timeouts, interruption sensitivity, voice, agent name) are documented in [`.env.example`](.env.example).

## Project structure

| File | What it does |
|---|---|
| `voice_agent/main.py` | Entry point. Starts a session per conversation, wires the pipeline, handles silence, hold and maximum duration, saves checkpoints, records metrics |
| `voice_agent/pipeline.py` | Builds the STT / LLM / TTS chains with failover, plus turn-taking and interruption settings |
| `voice_agent/agent.py` | Ava herself: cached greeting, echo guard, summarizing long conversations, resuming after a crash |
| `voice_agent/tools.py` | What Ava can do: order lookup, cancel order, help search, hold, end call, hand off to a human. The wrapper adds fillers, timeouts, retries, enforced confirmation and duplicate protection |
| `voice_agent/normalize.py` | Rewrites prices, dates, times, phone numbers, IDs, emails and URLs into speakable words, while streaming |
| `voice_agent/telemetry.py` | Per-turn latency breakdown, p50/p95 per session, redaction of personal data from logs |
| `voice_agent/state.py` | Session state and Redis checkpoints |
| `voice_agent/prompts.py` | The voice-specific system prompt |
| `voice_agent/backend.py` | **Mock** store backend. Business rules are enforced here, not in the prompt |
| `voice_agent/config.py` | Every setting, read from environment variables |
| `tests/` | Offline tests with fake providers |

## What it handles

- **Natural turn-taking:** waits through "um…" and mid-sentence pauses, and replies quickly once a thought is finished.
- **Interruptions:** stops speaking immediately when the user talks over it, ignores coughs and "mm-hmm", and resumes after a false alarm.
- **Silence and hold:** asks "Are you still there?" after 8 s, honours "hold on a second" for up to 60 s.
- **Speakable output:** no markdown or emojis. `$45.50` becomes "forty-five dollars and fifty cents", and order `45678` is read digit by digit.
- **Safe tools:** spoken filler while a lookup is slow, an 8 s timeout and one retry, confirmation required before irreversible actions, and the same action is never run twice.
- **Grounding and prompt injection:** facts come only from tools, and rules such as "only unshipped orders can be cancelled" live in the backend.
- **Long conversations:** older turns are summarized in the background.
- **Reliability:** provider failover at every stage, and a checkpoint after every turn so a replacement worker can resume.
- **Privacy:** card numbers, emails, phone numbers and long digit runs are redacted from logs.

The full edge-case catalogue, with IDs, is in [PLAN.md §7](PLAN.md).

## Roadmap

Scope: **web and mobile app only** (no phone calls).

**Next**
- [ ] Switch to Deepgram for both STT (Flux, with built-in end-of-turn detection) and TTS (Aura-2), keeping non-Deepgram backups
- [ ] First end-to-end test with real API keys
- [ ] Use the logged-in app user's identity so callers can only see and change **their own** orders
- [ ] Check for an explicit "yes" in code before confirmed actions, not only via the LLM
- [ ] Store idempotency keys in the backend, so an action can't repeat after a crash

**Production**
- [ ] Storage: transcripts (redacted), recordings, per-call metrics
- [ ] Dashboards: OpenTelemetry traces, Prometheus/Grafana for p50/p95 latency
- [ ] Real store APIs and proper help-article search (RAG)
- [ ] In-app human handoff (a support agent joins the room, or a ticket is opened)
- [ ] Output safety and prompt-injection checks, plus memory across conversations
- [ ] Docker image, deployment and load testing

**Later**
- [ ] Simulated-caller evals (accents, noise, interruptions) in CI
- [ ] Multilingual / code-switching, model routing, expressive TTS

## Security

- Real keys live only in `.env`, which is git-ignored. Never commit it. `.env.example` contains placeholders only.
- Logs are redacted before they are written.
