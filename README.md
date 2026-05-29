# qa-bot

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

Run from the repo root with `make`:

```bash
make markdown
make vector
make run-both
```
## Memory Flow

The app uses three memory levels:

- `L1`: compacted raw Q+A logs in `knowledge_base_qa_bot/.kb/logs/YYYY-MM-DD.md`
- `L2`: active in-browser session buffer, kept in the frontend until you click `Compact Session` or the trash button
- `L3`: dreamed wiki memory in `knowledge_base_qa_bot/.kb/wiki/index.md`

## Diagram

```mermaid
flowchart LR
    classDef outer fill:#fffdf8,stroke:#5f6f7a,stroke-width:2px,stroke-dasharray: 8 6,color:#1f2933
    classDef step fill:#fffefb,stroke:#7a8a96,stroke-width:1.5px,color:#1f2933

    U[User asks question]:::step

    subgraph L2[L2 Session Memory]
        direction TB
        B1[Hold Q+A in frontend buffer]:::step
        B2[Wait for Compact Session or trash]:::step
    end

    subgraph L1[L1 Log Memory]
        direction TB
        C1[Flush Q+A into .kb/logs]:::step
        C2[Store raw session history]:::step
    end

    subgraph L3[L3 Wiki Memory]
        direction TB
        D1[Group repeated VALID questions]:::step
        D2[LLM writes structured question + answer]:::step
        D3[Save wiki entry in .kb/wiki/index.md]:::step
    end

    I[Index rebuild]:::step
    R[Future retrieval answers from wiki]:::step

    style L2 fill:#f7fbff,stroke:#5c7c99,stroke-width:2px,stroke-dasharray: 8 6
    style L1 fill:#f7fcf7,stroke:#658a65,stroke-width:2px,stroke-dasharray: 8 6
    style L3 fill:#fbf7ff,stroke:#7b66a1,stroke-width:2px,stroke-dasharray: 8 6

    U --> B1
    B1 --> B2
    B2 -->|flush session| C1
    C1 --> C2
    C2 -->|Dream to Wiki| D1
    D1 --> D2
    D2 --> D3
    D3 --> I
    I --> R
```

## APIs

Both backends expose the same endpoints:

- `GET /health`
- `POST /index`
- `POST /chat`
- `POST /api/memory/compact`
- `POST /api/memory/dream`