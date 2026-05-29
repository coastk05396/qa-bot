from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .indexer import load_index_json
from .memory import ensure_memory_directories
from .routes import router

app = FastAPI(title="Markdown Knowledge Base Q&A Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.on_event("startup")
def load_persisted_index():
    ensure_memory_directories()
    load_index_json()
