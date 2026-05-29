import json
import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .memory import WIKI_DIR


load_dotenv()


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_DIR = Path(__file__).resolve().parents[3] / ".kb" / "faiss_index"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " "],
)

vectorstore: FAISS | None = None
_embeddings = None
files_indexed = 0
sections_indexed = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def get_embeddings():
    global _embeddings
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the server environment")
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )
    return _embeddings


def load_markdown_sections(path: Path) -> list[Document]:
    documents: list[Document] = []
    heading_stack: list[tuple[int, str]] = []
    current_path: list[str] = []
    content_lines: list[str] = []

    def flush_current() -> None:
        if not current_path:
            return
        content = "\n".join(content_lines).strip()
        heading = " > ".join(current_path)
        source = f"{path.name}#{slugify(current_path[-1])}"
        documents.append(
            Document(
                page_content=f"Heading: {heading}\n\n{content or current_path[-1]}",
                metadata={
                    "source": source,
                    "heading": heading,
                },
            )
        )

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        heading_match = HEADING_RE.match(raw_line)
        if heading_match:
            flush_current()
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading))
            current_path = [item[1] for item in heading_stack]
            content_lines = []
            continue
        if current_path:
            content_lines.append(raw_line)

    flush_current()
    return documents


def iter_markdown_files(docs_dir: Path = DOCS_DIR, wiki_dir: Path | None = None) -> list[Path]:
    resolved_wiki_dir = wiki_dir or WIKI_DIR
    markdown_files = sorted(docs_dir.glob("*.md"))
    if resolved_wiki_dir.exists():
        markdown_files.extend(sorted(resolved_wiki_dir.glob("*.md")))
    return markdown_files


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed

    vectorstore = None
    files_indexed = 0
    sections_indexed = 0
    markdown_files = iter_markdown_files(docs_dir)
    documents: list[Document] = []

    for markdown_path in markdown_files:
        documents.extend(load_markdown_sections(markdown_path))

    files_indexed = len(markdown_files)
    chunks = splitter.split_documents(documents)
    sections_indexed = len(chunks)

    if chunks:
        vectorstore = FAISS.from_documents(chunks, get_embeddings())

    save_vector_index(INDEX_DIR)
    return files_indexed, sections_indexed


def save_vector_index(index_dir: Path = INDEX_DIR) -> None:
    if vectorstore is None:
        if index_dir.exists():
            shutil.rmtree(index_dir)
        return

    if index_dir.exists():
        shutil.rmtree(index_dir)

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "files_indexed": files_indexed,
        "sections_indexed": sections_indexed,
    }
    (index_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_vector_index(index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed

    faiss_path = index_dir / "index.faiss"
    pickle_path = index_dir / "index.pkl"
    metadata_path = index_dir / "metadata.json"

    if not faiss_path.exists() or not pickle_path.exists():
        vectorstore = None
        files_indexed = 0
        sections_indexed = 0
        return 0, 0

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("embedding_model") != EMBEDDING_MODEL:
            raise RuntimeError("Persisted FAISS index uses a different embedding model")

    vectorstore = FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    files_indexed = metadata.get("files_indexed", 0)
    sections_indexed = metadata.get("sections_indexed", 0)
    return files_indexed, sections_indexed


def search(query: str, k: int = 3) -> list[tuple[Document, float]]:
    if vectorstore is None:
        return []
    return vectorstore.similarity_search_with_score(query, k=k)
