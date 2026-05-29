# qa-bot

Run from the repo root with `make`:

```bash
make install
make markdown
make vector
make run-both
```
## 3-Tier RAG Memory System
```
( User asks Question )
            │
            ▼
 ┌────────────────────────────────┐      ┌────────────────────────────────┐      ┌────────────────────────────────┐
 │ 🟦 L1: Daily Logs              │      │ 🟩 L2: Dream ETL               │      │ 🟪 L3: Wiki Memory             │
 │    (Working & Episodic)        │      │    (Consolidation Pipeline)    │      │    (Semantic Storage)          │
 │                                │      │                                │      │                                │
 │      [ Frontend Buffer ]       │      │        [ Group VALID ]         │      │          [ index.md ]          │
 │               │                │      │               │                │      │               │                │
 │           (Compact)            │──Cron─▶       (LLM Generate)          │──Save─▶          (Rebuild)            │
 │               │                │      │               │                │      │               │                │
 │               ▼                │      │               ▼                │      │               ▼                │
 │      [ Flush .kb/logs ]        │      │       [ Structured Q&A ]       │      │        [ Vector/BM25 ]         │
 │                                │      │                                │      │                                │
 └────────────────────────────────┘      └────────────────────────────────┘      └────────────────────────────────┘
                                                                                                 │
                                                                                                 ▼
                                                                                   ( Future Retrieval Answers )
```

| Tier | Component | Description | Storage Path |
| :--- | :--- | :--- | :--- |
| **`L1`** | **Working Memory** | Active in-browser session buffer. Cleared upon clicking `Compact Session` or trash. | *Frontend (Browser)* |
| **`L2`** | **Episodic Memory**| Compacted raw Q+A logs. Flushed from L1 for short-term daily storage. | `.kb/logs/YYYY-MM-DD.md` |
| **`L3`** | **Semantic Memory**| Dreamed wiki memory. Distilled persistent knowledge ready for RAG indexing. | `.kb/wiki/index.md` |




## Gemini-Key

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

## APIs

Both backends expose the same endpoints:

- `GET /health`
- `POST /index`
- `POST /chat`
- `POST /api/memory/compact`
- `POST /api/memory/dream`
