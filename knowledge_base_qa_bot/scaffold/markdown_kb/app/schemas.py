from typing import Literal

from pydantic import BaseModel


ProviderMode = Literal["gemini"]


class IndexResponse(BaseModel):
    files_indexed: int
    sections_indexed: int


class ChatRequest(BaseModel):
    query: str
    provider: ProviderMode = "gemini"


class SourceInfo(BaseModel):
    source: str
    heading: str
    score: float
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
