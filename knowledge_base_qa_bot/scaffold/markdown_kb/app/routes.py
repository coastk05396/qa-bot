from pathlib import Path

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import FileResponse

from .indexer import build_index
from .retrieval import GEMINI_CHAT_MODEL, gemini_key_is_configured, query
from .schemas import ChatRequest, ChatResponse, IndexResponse

router = APIRouter()
FRONTEND_PATH = Path(__file__).resolve().parents[3] / "frontend" / "index.html"


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
    }


@router.post("/index", response_model=IndexResponse)
def index_docs():
    try:
        files_count, sections_count = build_index()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IndexResponse(files_indexed=files_count, sections_indexed=sections_count)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return query(req.query, req.provider)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
