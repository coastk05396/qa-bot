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
```

Both folders require a Gemini API key before running the server:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

Markdown KB uses Gemini for final answer generation with `gemini-flash-lite-latest`. Vector RAG also uses Gemini embeddings with `gemini-embedding-001`.

Start with `markdown_kb` if you want the smallest dependency surface and the easiest debugging path.
