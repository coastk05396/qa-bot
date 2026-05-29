import math
import re
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

from .memory import WIKI_DIR


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_PATH = Path(__file__).resolve().parents[3] / ".kb" / "index.json"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "my",
    "of",
    "the",
    "to",
    "what",
    "when",
    "which",
}


@dataclass
class Section:
    id: str
    file: str
    heading: str
    heading_path: list[str]
    content: str
    tokens: list[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "content": self.content,
            "tokens": self.tokens,
        }


sections: list[Section] = []
doc_freq: Counter[str] = Counter()
avg_doc_len = 0.0
files_indexed = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def parse_markdown(path: Path) -> list[Section]:
    parsed_sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    current_path: list[str] = []
    content_lines: list[str] = []

    def flush_current() -> None:
        if not current_path:
            return
        content = "\n".join(content_lines).strip()
        tokens = tokenize(f"{' '.join(current_path)}\n{content}")
        parsed_sections.append(
            Section(
                id=f"{path.name}#{slugify(current_path[-1])}",
                file=path.name,
                heading=current_path[-1],
                heading_path=current_path.copy(),
                content=content,
                tokens=tokens,
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
    return parsed_sections


def iter_markdown_files(docs_dir: Path = DOCS_DIR, wiki_dir: Path | None = None) -> list[Path]:
    resolved_wiki_dir = wiki_dir or WIKI_DIR
    markdown_files = sorted(docs_dir.glob("*.md"))
    if resolved_wiki_dir.exists():
        markdown_files.extend(sorted(resolved_wiki_dir.glob("*.md")))
    return markdown_files


def write_index_json(index_path: Path = INDEX_PATH) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sections": [section.to_dict() for section in sections],
        "stats": {
            "files_indexed": files_indexed,
            "sections_indexed": len(sections),
            "avg_doc_len": avg_doc_len,
        },
    }
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rebuild_stats() -> None:
    global doc_freq, avg_doc_len, files_indexed

    doc_freq = Counter()
    files_indexed = len({section.file for section in sections})
    total_tokens = 0

    for section in sections:
        total_tokens += len(section.tokens)
        doc_freq.update(set(section.tokens))

    avg_doc_len = (total_tokens / len(sections)) if sections else 0.0


def load_index_json(index_path: Path = INDEX_PATH) -> tuple[int, int]:
    global sections

    if not index_path.exists():
        sections = []
        rebuild_stats()
        return 0, 0

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    sections = [Section(**section) for section in payload.get("sections", [])]
    rebuild_stats()
    return files_indexed, len(sections)


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global sections, doc_freq, avg_doc_len, files_indexed

    sections = []
    doc_freq = Counter()
    avg_doc_len = 0.0
    files_indexed = 0

    for markdown_path in iter_markdown_files(docs_dir):
        sections.extend(parse_markdown(markdown_path))

    rebuild_stats()
    write_index_json(INDEX_PATH)
    return files_indexed, len(sections)


def bm25_score(query_tokens: list[str], section: Section, k1: float = 1.5, b: float = 0.75) -> float:
    if not query_tokens or not section.tokens or not sections:
        return 0.0

    term_freq = Counter(section.tokens)
    doc_len = max(len(section.tokens), 1)
    mean_len = avg_doc_len or 1.0
    section_count = len(sections)
    heading_tokens = set(tokenize(" ".join(section.heading_path)))
    score = 0.0

    for token in query_tokens:
        frequency = term_freq.get(token, 0)
        if frequency == 0:
            continue
        frequency_in_docs = doc_freq.get(token, 0)
        idf = math.log(1 + (section_count - frequency_in_docs + 0.5) / (frequency_in_docs + 0.5))
        denominator = frequency + k1 * (1 - b + b * (doc_len / mean_len))
        score += idf * ((frequency * (k1 + 1)) / denominator)
        if token in heading_tokens:
            score += 0.2

    return score


def search(query: str, k: int = 3) -> list[tuple[Section, float]]:
    query_tokens = tokenize(query)
    ranked = [
        (section, bm25_score(query_tokens, section))
        for section in sections
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [(section, score) for section, score in ranked[:k] if score > 0]
