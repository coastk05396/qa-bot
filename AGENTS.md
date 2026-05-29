# Instructions

This repository contains three independent prototype exercises. Work in the relevant subproject root and its `scaffold/` directory; there is no shared app entrypoint or repo-wide test runner.

## Repository layout

- `qr_code_generator/` — FastAPI QR shortener with SQLite, SQLAlchemy, in-memory redirect caching, QR image generation, soft delete, expiration, and scan analytics.
- `chatgpt_task/` — MCP stdio task scheduler with SQLite persistence, hourly time buckets, a watcher thread, a worker thread, and dotted tool names such as `task.create`.
- `knowledge_base_qa_bot/` — FastAPI Q&A bot with two guided paths:
  - `scaffold/markdown_kb/` for Markdown section indexing + BM25
  - `scaffold/vector_rag/` for chunking + FAISS retrieval

The `answers/` trees mirror the scaffolds and can be used as reference implementations.

## Commands

### QR Code Generator

```bash
cd qr_code_generator/scaffold
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Single verification:

```bash
curl -X POST http://localhost:8000/api/qr/create \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com"}'
```

Use the curl checks in `PROMPT.md` for redirect, update, delete, image, and analytics behavior.

### ChatGPT Task Scheduler

```bash
cd chatgpt_task/scaffold
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.mcp_server
```

Inspector run:

```bash
npx @modelcontextprotocol/inspector python -m app.mcp_server
```

Single verification: use the inspector to call one tool at a time (`task.create`, `task.status`, `task.cancel`, or `task.list`) and confirm the JSON response.

### Knowledge Base Q&A Bot

Markdown KB:

```bash
cd knowledge_base_qa_bot/scaffold/markdown_kb
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Vector RAG:

```bash
cd knowledge_base_qa_bot/scaffold/vector_rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Set `OPENAI_API_KEY` before using either guided track. Single verification: `curl http://localhost:8000/health`, then `POST /index`, then `POST /chat` with a question grounded in `docs/*.md`.

## Architecture

### QR Code Generator

- `app/main.py` creates the FastAPI app and wires the router.
- `app/routes.py` owns the API surface: create, redirect, update, delete, image, and analytics.
- `app/models.py` stores `UrlMapping` and `ScanEvent` in SQLite.
- Redirect handling should stay cache-first, then fall back to the DB, and must distinguish missing (`404`) from deleted/expired (`410`) links.

### ChatGPT Task Scheduler

- `app/mcp_server.py` defines MCP tools, a registry-based dispatcher, and the stdio server entrypoint.
- `app/scheduler.py` owns the watcher/worker loops and the in-memory queue.
- `app/models.py` stores jobs with `time_bucket`, status, result, and timestamps.
- Tool names follow `namespace.action` style, and the registry should map those names directly to handlers.

### Knowledge Base Q&A Bot

- `app/main.py` loads any persisted index on startup.
- `app/routes.py` exposes the shared API: `GET /health`, `POST /index`, `POST /chat`.
- `app/indexer.py` builds the retrieval index from `docs/*.md` and persists it under `.kb/`.
- `app/retrieval.py` formats the prompt and returns grounded answers with sources.
- Both strategies cite sources as `filename#heading` and should fall back to an honest “cannot confirm” response when retrieval is weak.

## Conventions

- Keep changes local to the active exercise; do not mix scaffold and reference code unless explicitly asked.
- Use the existing FastAPI/Pydantic/SQLAlchemy patterns already present in each scaffold.
- Keep route handlers thin and move business logic into helper modules.
- Preserve source metadata and heading paths in the knowledge-base projects so citations stay inspectable.
- For the scheduler, keep the hourly bucket format stable (`%Y%m%d%H`) and treat the watcher/worker loops as background daemon threads.
- Prefer explicit response models and exact status-code semantics over ad hoc dictionaries.
