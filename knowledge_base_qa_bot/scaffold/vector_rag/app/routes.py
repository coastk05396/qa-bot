from pathlib import Path

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from .indexer import EMBEDDING_MODEL, build_index
from .memory import append_chat_logs, compact_session, run_dreaming_etl
from .retrieval import CANNOT_CONFIRM, GEMINI_CHAT_MODEL, gemini_key_is_configured, query
from .schemas import (
    ChatRequest,
    ChatResponse,
    IndexResponse,
    MemoryCompactRequest,
    MemoryCompactResponse,
    MemoryDreamRequest,
    MemoryDreamResponse,
)

router = APIRouter()
FRONTEND_PATH = Path(__file__).resolve().parents[3] / "frontend" / "index.html"
DEFAULT_QUESTIONS = {
    "Summarize the refund policy and include the expected timeline.",
    "What shipping options are available, and what should customers expect for delivery timing?",
    "What account-help steps can the bot confirm from the knowledge base?",
}


@router.get("/", include_in_schema=False)
def frontend():
    return FileResponse(FRONTEND_PATH)


@router.get("/health")
def health():
    return {
        "status": "ok",
        "provider": "gemini",
        "api_key_provided": gemini_key_is_configured(),
        "chat_model": GEMINI_CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }


@router.post("/index", response_model=IndexResponse)
def index_docs():
    try:
        files_count, sections_count = build_index()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IndexResponse(files_indexed=files_count, sections_indexed=sections_count)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        response = await run_in_threadpool(query, req.query, req.provider)
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/memory/compact", response_model=MemoryCompactResponse)
async def compact_memory(req: MemoryCompactRequest):
    try:
        flushed_entries = await append_chat_logs(
            req.session_id,
            [entry.model_dump() for entry in req.entries],
        )
        return MemoryCompactResponse(**compact_session(req.session_id, flushed_entries))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/memory/dream", response_model=MemoryDreamResponse)
async def dream_memory(req: MemoryDreamRequest):
    try:
        payload = await run_dreaming_etl(build_index, req.provider)
        return MemoryDreamResponse(**payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
