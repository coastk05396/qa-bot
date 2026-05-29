# qa-bot

This repository contains independent prototype exercises. There is no repo-wide app entrypoint or single test runner. Work inside the relevant project folder.

## Projects

- `knowledge_base_qa_bot/` - FastAPI knowledge-base Q&A bot with two guided retrieval strategies:
  - `scaffold/markdown_kb/` for Markdown section indexing + BM25
  - `scaffold/vector_rag/` for chunking + FAISS vector retrieval
- `chatgpt_task/` - MCP stdio task scheduler with SQLite persistence and tool routing

## Knowledge Base Q&A Bot

This is the main web app in the repo right now. Both guided tracks expose the same API:

- `GET /health`
- `POST /index`
- `POST /chat`

### Gemini API Key

Both guided tracks use Gemini for answer generation.

You can provide the key in either of these ways:

1. Export it in your shell:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

2. Or place it in a `.env` file that the app can load, for example:

```bash
GEMINI_API_KEY=your-gemini-api-key
```

Recommended location: keep the `.env` in the scaffold you are running, or export the variable before starting the server.

### Run Markdown KB

```bash
cd knowledge_base_qa_bot/scaffold/markdown_kb
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Run Vector RAG

```bash
cd knowledge_base_qa_bot/scaffold/vector_rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

### Quick Verification

After the server starts:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/index
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What shipping options are available?","provider":"gemini"}'
```

If you run Vector RAG on port `8001`, use `http://127.0.0.1:8001` instead.

### Notes

- Retrieval indices are persisted under `.kb/`
- Re-run `POST /index` after changing `docs/*.md`
- The shared browser UI is served by both retrieval backends

## ChatGPT Task Scheduler

```bash
cd chatgpt_task/scaffold
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.mcp_server
```

Inspector:

```bash
npx @modelcontextprotocol/inspector python -m app.mcp_server
```

## Working Style

- Keep changes local to the active exercise
- Prefer the scaffold folders for implementation work
- Use project-specific README and `PROMPT.md` files for deeper exercise details