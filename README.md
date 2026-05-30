# qa-bot

<img width="1500" height="1032" alt="twat1_middle" src="https://github.com/user-attachments/assets/2287759a-b964-47cd-b51b-8948a6146a1f" />

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
| **`L1`** | **Short Memory** | Active in-browser session buffer. Cleared upon clicking `Compact Session` or the trash button to log daily conversation. | *Frontend (Browser)* |
| **`L2`** | **Dreaming** | A background ETL process that clusters L1 logs using **embedding similarity**. When similar queries hit 3+ times, an LLM structures and promotes them. | *Processing Pipeline* |
| **`L3`** | **Wiki** | Persistent semantic memory. Distilled, structured knowledge ready for backend RAG indexing and future retrieval. | `.kb/wiki/index.md` |

Run from the repo root with `make`:

```bash
make install
make markdown
make vector
make run-both
```


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
