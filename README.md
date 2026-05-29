# qa-bot

The main app in this repo is the Knowledge Base Q&A Bot.

## Quick start

1. Go to the default app:

```bash
cd knowledge_base_qa_bot/scaffold/markdown_kb
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set your Gemini API key:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

5. Start the server:

```bash
uvicorn app.main:app --reload
```

6. Check that it is running:

```bash
curl http://127.0.0.1:8000/health
```

## API

- `GET /health`
- `POST /index`
- `POST /chat`

Index the docs before chatting:

```bash
curl -X POST http://127.0.0.1:8000/index
```

## Other options

- `knowledge_base_qa_bot/scaffold/vector_rag` - same API, different retrieval strategy
- `chatgpt_task/scaffold` - separate MCP scheduler exercise
