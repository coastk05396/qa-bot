# Guided Track Scaffolds

Choose one retrieval strategy:

```bash
# Recommended default
cd markdown_kb

# Traditional RAG comparison
cd vector_rag
```

Both folders expose the same API:

```text
GET /health
POST /index
POST /chat
POST /api/memory/compact
POST /api/memory/dream
```

The chat endpoint now expects a client-generated session ID:

```json
{
	"query": "How long do refunds take?",
	"provider": "gemini",
	"session_id": "client-session-uuid"
}
```

Memory behavior is flat-markdown only:

- L1 daily chat logs append to `.kb/logs/YYYY-MM-DD.md`
- L3 dreamed knowledge appends to `.kb/wiki/index.md`
- Dream sync re-runs the active backend's existing index builder so wiki content is searchable

For nightly dreaming, use external cron to call the active backend instead of embedding a scheduler in the FastAPI process:

```bash
0 0 * * * curl -X POST http://localhost:8000/api/memory/dream -H "Content-Type: application/json" -d '{"provider":"gemini"}'
0 0 * * * curl -X POST http://localhost:8001/api/memory/dream -H "Content-Type: application/json" -d '{"provider":"gemini"}'
```

Both folders require a Gemini API key before running the server:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

Markdown KB uses Gemini for final answer generation with `gemini-flash-lite-latest`. Vector RAG also uses Gemini embeddings with `gemini-embedding-001`.

Start with `markdown_kb` if you want the smallest dependency surface and the easiest debugging path.
