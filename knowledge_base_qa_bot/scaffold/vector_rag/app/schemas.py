from typing import Literal

from pydantic import BaseModel


ProviderMode = Literal["gemini"]


class IndexResponse(BaseModel):
    files_indexed: int
    sections_indexed: int


class ChatRequest(BaseModel):
    query: str
    provider: ProviderMode = "gemini"
    session_id: str


class SourceInfo(BaseModel):
    source: str
    heading: str
    score: float
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class MemoryLogEntry(BaseModel):
    question: str
    answer: str
    status: str


class MemoryCompactRequest(BaseModel):
    session_id: str
    entries: list[MemoryLogEntry] = []


class MemoryCompactResponse(BaseModel):
    ok: bool
    previous_session_id: str
    buffers_flushed: int
    message: str


class MemoryDreamRequest(BaseModel):
    provider: ProviderMode = "gemini"


class MemoryDreamResponse(BaseModel):
    logs_scanned: int
    valid_blocks: int
    rejected_blocks_removed: int
    processed_session_ids: list[str]
    wiki_updated: bool
    kept_blocks: int
    compacted_files: int
    deleted_files: int
