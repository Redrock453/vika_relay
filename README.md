# vika_relay

An LLM gateway behind a Telegram assistant. One `ask()` call, four cloud providers and an optional local model behind it: if one fails, runs out of balance or returns nothing, the next one answers — and with no internet at all, the local model does. On top of that — tool calling, RAG over a private knowledge base, guest access with a hard tool allowlist, and a cost guard for peak pricing hours.

Extracted from the author's personal assistant; this repository contains the gateway core and a minimal bot around it.

## Features

- **Fallback chain** — DeepSeek → OpenRouter → Groq → Gemini, all through OpenAI-compatible APIs. Each provider gets 3 attempts with 1 s / 3 s backoff. Auth, billing and unknown-model errors (401/402/403/404) skip straight to the next provider. An empty answer counts as a failure, not as a reply.
- **Local model** — an optional last link on Ollama. It answers when every cloud provider is unreachable. It runs without tools, so tool results from earlier steps of the turn reach it as text. Off unless `VIKA_LOCAL_MODEL` is set.
- **Tool calling loop** — the model calls tools, results go back to it, until it answers in text. At the round limit it writes an honest summary of what was done instead of cutting off.
- **Cost guard** — during DeepSeek peak hours (weekdays 01–04 and 06–10 UTC) the pro model and heavy requests (6,000+ characters) are routed around DeepSeek, with Groq first.
- **Per-request routing** — preferred provider, model, reasoning effort and cancellation are set per turn in thread-local state, so concurrent chats don't affect each other.
- **Access modes** — `owner` gets web search and the knowledge base; `guest` gets web search only. The allowlist is enforced twice: in the tool list the model sees and in the executor. Unknown modes get guest rights.
- **RAG** — bge-m3 embeddings via Ollama, Qdrant search with a relevance threshold; short messages skip retrieval. Retrieved context goes at the end of the user turn, so the prompt prefix stays stable and provider-side prompt caching keeps hitting.
- **Telegram bot** — text and voice (Whisper via Groq), per-user history, user allowlist.

## Architecture

```mermaid
flowchart LR
    U[Telegram user] --> B[Bot<br/>allowlist · voice to text]
    B --> A[Agent<br/>mode: owner / guest]
    A --> H[(History)]
    A -->|owner only| K[(Knowledge base<br/>Qdrant + bge-m3)]
    A --> R{LLM router}
    R --> P1[DeepSeek]
    P1 -.->|fails| P2[OpenRouter]
    P2 -.->|fails| P3[Groq]
    P3 -.->|fails| P4[Gemini]
    P4 -.->|fails| P5[Local model<br/>Ollama]
    R <-->|tool calls| T[Tools<br/>web_search · rag_search]
    T --> S[SearXNG]
```

| Path | What it does |
| --- | --- |
| `gate/router.py` | Provider chain with the local fallback, retries, tool loop, peak guard |
| `gate/agent.py` | Prompt, history and RAG assembly, access modes |
| `gate/rag.py` | Embeddings via Ollama, search in Qdrant |
| `gate/search.py` | Web search via SearXNG |
| `bot/telegram.py` | Telegram front end |

## Quick start

```bash
git clone https://github.com/Redrock453/vika_relay.git
cd vika_relay
cp .env.example .env   # add a Telegram token, your Telegram ID and at least one provider key
```

With Docker (bot + SearXNG + Qdrant):

```bash
docker compose up -d --build
```

To add the local model and RAG embeddings, set `VIKA_LOCAL_MODEL` in `.env` (any Ollama chat model; `qwen3:8b` here is an example), then start Ollama and pull the models:

```bash
docker compose --profile local up -d --build
docker compose exec ollama ollama pull qwen3:8b
docker compose exec ollama ollama pull bge-m3
```

Without Docker (Python 3.11+, ffmpeg for voice):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

RAG and the local model are optional: without Ollama and Qdrant the bot answers through the cloud providers, without a knowledge base.

## Configuration

| Variable | Purpose |
| --- | --- |
| `TELEGRAM_BOT_TOKEN`, `ALLOWED_IDS` | Bot token and allowed Telegram user IDs |
| `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY` | Providers; any without a key is skipped |
| `VIKA_*_MODEL` | Model override per provider |
| `VIKA_DEEPSEEK_PEAK_GUARD` | `0` turns the cost guard off |
| `VIKA_LOCAL_MODEL` | Local model on Ollama, last in the chain; empty = off |
| `OLLAMA_URL` | Ollama endpoint for the local model and embeddings |
| `QDRANT_HOST`, `VIKA_RAG_COLLECTION` | Knowledge base |
| `SEARXNG_URL`, `SEARXNG_SECRET` | Web search |

The full list with defaults is in [`.env.example`](.env.example).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover routing (fallback order, retries, fail-fast errors, empty answers, tool loop, round limit, peak guard, local fallback) and access modes, using scripted fake providers — no API keys or network needed.

## Українською

**vika_relay** — шлюз до LLM для Telegram-асистента: п'ять хмарних провайдерів і локальна модель на Ollama з автоматичним перемиканням, виклик інструментів, RAG на Qdrant, гостьовий режим з жорстким списком дозволених інструментів і захист від пікових тарифів. Виділено з особистого асистента автора.
