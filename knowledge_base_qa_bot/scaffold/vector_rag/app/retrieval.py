import os
import re

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from . import indexer


load_dotenv()


SYSTEM_PROMPT = """
You are a knowledge-base Q&A assistant.

Answer only from the provided CONTEXT.
Give a direct answer first in plain language.
Return plain text only.
Do not use Markdown, bold markers, bullets, numbered lists, headings, links, or code fences.
If the context does not clearly answer the question, say: "I cannot confirm from the knowledge base."
Do not use outside knowledge.
Do not answer with source names alone.
"""

CANNOT_CONFIRM = "I cannot confirm from the knowledge base."

GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
_llms: dict[str, object] = {}


def get_gemini_api_key() -> str:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def gemini_key_is_configured() -> bool:
    return bool(get_gemini_api_key())


def get_llm(provider: str = "gemini"):
    normalized_provider = (provider or "gemini").lower()
    if normalized_provider not in _llms:
        api_key = get_gemini_api_key()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the server environment")
        _llms[normalized_provider] = ChatGoogleGenerativeAI(
            model=GEMINI_CHAT_MODEL,
            google_api_key=api_key,
            temperature=0,
            max_retries=1,
        )
    return _llms[normalized_provider]


def build_prompt(query: str, ranked_chunks: list) -> str:
    context_blocks = []
    for doc, score in ranked_chunks:
        context_blocks.append(
            "\n".join(
                [
                    f"[Source: {doc.metadata.get('source', 'unknown')}]",
                    f"Heading: {doc.metadata.get('heading', 'unknown')}",
                    f"Score: {round(float(score), 3)}",
                    doc.page_content,
                ]
            )
        )
    context = "\n\n".join(context_blocks) if context_blocks else "(no context)"
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def _extract_summary(content: str) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if lines and lines[0].startswith("Heading:"):
        lines = lines[1:]

    cleaned = " ".join(lines)
    if not cleaned:
        return "This section is present in the knowledge base, but no additional body text was indexed."

    sentences = [part.strip() for part in cleaned.replace(";", ". ").split(".") if part.strip()]
    if not sentences:
        return cleaned
    return ". ".join(sentences[:2]).strip() + "."


def _strip_markdown(text: str) -> str:
    cleaned = text.replace("\r\n", "\n")
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", cleaned)
    cleaned = re.sub(r"^[ \t]{0,3}[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^[ \t]{0,3}\d+[.)]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^[ \t]{0,3}>\s?", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _leaf_heading(heading: str) -> str:
    parts = [part.strip() for part in heading.split(">") if part.strip()]
    return parts[-1] if parts else heading.strip() or "Section"


def _build_grounded_fallback(ranked_chunks: list) -> str:
    details = []
    seen_ids = set()

    for doc, _score in ranked_chunks:
        source_id = doc.metadata.get("source", "unknown")
        if source_id in seen_ids:
            continue
        seen_ids.add(source_id)
        details.append(f"{_leaf_heading(doc.metadata.get('heading', 'Section'))}: {_extract_summary(doc.page_content)}")

    if not details:
        return CANNOT_CONFIRM

    return "From the knowledge base, I can confirm:\n" + "\n".join(details)


def _finalize_answer(raw_answer: object, ranked_chunks: list) -> str:
    answer = _strip_markdown(getattr(raw_answer, "content", str(raw_answer)).strip())
    if not answer:
        return _build_grounded_fallback(ranked_chunks)

    normalized = answer.lower()
    if normalized.startswith(CANNOT_CONFIRM.lower()):
        return _build_grounded_fallback(ranked_chunks)

    return answer


def query(question: str, provider: str = "gemini") -> dict:
    if indexer.vectorstore is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked_chunks = indexer.search(question, k=3)
    if not ranked_chunks:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    response = get_llm(provider).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_prompt(question, ranked_chunks)),
        ]
    )

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "heading": doc.metadata.get("heading", "unknown"),
            "score": round(float(score), 3),
            "content": doc.page_content[:240],
        }
        for doc, score in ranked_chunks
    ]

    return {
        "answer": _finalize_answer(response, ranked_chunks),
        "sources": sources,
    }
