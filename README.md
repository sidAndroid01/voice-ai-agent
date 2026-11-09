# Voice AI Agent

A real-time, general-purpose voice assistant you can ask anything, like the voice mode in ChatGPT. You talk, it listens, thinks and talks back, and you can interrupt it at any point like a real person. By default the "brain" runs **locally** (Ollama); only speech goes to Deepgram.

It is built on [LiveKit Agents](https://docs.livekit.io/agents/) as a **cascaded pipeline**: three separate services for listening (STT), thinking (LLM) and speaking (TTS), each with automatic failover to a backup provider.

> **Status: v0 prototype.** Tested live in console mode on a MacBook (M4, 16 GB) with Deepgram + local Qwen3-8B: about 3 s from end of speech to first audio (p50). Not deployed yet. See [Roadmap](#roadmap). The full design is in [PLAN.md](PLAN.md).

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
                          Tools (date/time, end conversation)
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
uv run pytest                                 # offline tests, no API keys needed
```

Minimum `.env` for a working conversation (local LLM, console mode):

```
DEEPGRAM_API_KEY=...          # speech to text and text to speech
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey        # console mode needs placeholders; real keys only for browser/app clients
LIVEKIT_API_SECRET=secret
```

Then make sure Ollama is running (`ollama serve` or the Ollama app) with `qwen3:8b` pulled.

All other settings (timeouts, interruption sensitivity, voice, agent name) are documented in [`.env.example`](.env.example).

## Project structure

| File | What it does |
|---|---|
| `voice_agent/main.py` | Entry point. Starts a session per conversation, wires the pipeline, optional silence prompts, maximum duration, checkpoints, metrics |
| `voice_agent/pipeline.py` | Builds the STT / LLM / TTS chains with failover, plus turn-taking and interruption settings |
| `voice_agent/agent.py` | The assistant: cached greeting, echo guard, summarizing long conversations, resuming after a crash |
| `voice_agent/tools.py` | Tools (current date/time, end conversation) plus `@voice_tool`, a wrapper for future tools that adds fillers, timeouts, retries, enforced confirmation and duplicate protection |
| `voice_agent/normalize.py` | Rewrites prices, dates, times, phone numbers, IDs, emails and URLs into speakable words, while streaming |
| `voice_agent/telemetry.py` | Per-turn latency breakdown, p50/p95 per session, redaction of personal data from logs |
| `voice_agent/state.py` | Session state and Redis checkpoints |
| `voice_agent/prompts.py` | The general-assistant voice prompt (short spoken answers, honesty about live info) |
| `voice_agent/config.py` | Every setting, read from environment variables |
| `tests/` | Offline tests with fake providers |

## What it handles

- **Ask anything:** general knowledge, explanations, ideas, advice, practice conversations. Short spoken answers by default; longer when you ask for detail.
- **Honesty:** says when it isn't sure, and that it can't check live information (news, weather, scores). Gets today's date and time from a tool instead of guessing.
- **Natural turn-taking:** waits through "um…" and mid-sentence pauses, and replies once a thought is finished.
- **Interruptions:** stops speaking immediately when you talk over it, ignores coughs and "mm-hmm", and resumes after a false alarm.
- **Silence:** waits quietly, with no "are you still there?" nagging (configurable with `MAX_SILENCE_PROMPTS`).
- **Speakable output:** no markdown or emojis. `$45.50` becomes "forty-five dollars and fifty cents", `1653` becomes "sixteen fifty-three", codes are read digit by digit.
- **Ending:** "bye" or "stop" ends the session through the `end_call` tool.
- **Long conversations:** older turns are summarized in the background.
- **Reliability:** provider failover at every stage, and a checkpoint after every turn so a replacement worker can resume.
- **Privacy:** the LLM runs locally by default. Card numbers, emails, phone numbers and long digit runs are redacted from logs.

The full edge-case catalogue, with IDs, is in [PLAN.md §7](PLAN.md). Store-specific cases there (orders, confirmation, handoff) are now generic tool-safety cases.

## Roadmap

Scope: **general voice assistant for web and mobile apps** (no phone calls).

**Next**
- [ ] Web search tool so it can answer questions about current events
- [ ] Lower latency: Deepgram Flux (built-in end-of-turn detection), shorter endpointing, faster local model or optional cloud LLM
- [ ] Browser client: local `livekit-server --dev` + agents playground
- [ ] Memory across conversations (name, preferences)

**Production**
- [ ] Storage: transcripts (redacted) and per-session metrics
- [ ] Dashboards: OpenTelemetry traces, Prometheus/Grafana for p50/p95 latency
- [ ] Output safety and prompt-injection checks
- [ ] Docker image, deployment and load testing

**Evals**
- [ ] Simulated-user evals (accents, noise, interruptions), latency and turn-taking metrics, LLM-judge answer quality, in CI

**Later**
- [ ] Multilingual / code-switching, expressive TTS, vision (camera/screen sharing)

## Security

- Real keys live only in `.env`, which is git-ignored. Never commit it. `.env.example` contains placeholders only.
- Logs are redacted before they are written.
