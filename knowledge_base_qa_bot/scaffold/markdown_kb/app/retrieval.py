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


def build_prompt(query: str, ranked_sections: list) -> str:
    context_blocks = []
    for section, _score in ranked_sections:
        context_blocks.append(
            "\n".join(
                [
                    f"[Source: {section.id}]",
                    f"Heading: {' > '.join(section.heading_path)}",
                    section.content or "(No body content for this heading)",
                ]
            )
        )
    context = "\n\n".join(context_blocks) if context_blocks else "(no context)"
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def _extract_summary(content: str) -> str:
    cleaned = " ".join(line.strip() for line in content.splitlines() if line.strip())
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


def _build_grounded_fallback(ranked_sections: list) -> str:
    details = []
    seen_ids = set()

    for section, _score in ranked_sections:
        if section.id in seen_ids:
            continue
        seen_ids.add(section.id)
        details.append(f"{section.heading}: {_extract_summary(section.content)}")

    if not details:
        return CANNOT_CONFIRM

    return "From the knowledge base, I can confirm:\n" + "\n".join(details)


def _finalize_answer(raw_answer: object, ranked_sections: list) -> str:
    answer = _strip_markdown(getattr(raw_answer, "content", str(raw_answer)).strip())
    if not answer:
        return _build_grounded_fallback(ranked_sections)

    normalized = answer.lower()
    if normalized.startswith(CANNOT_CONFIRM.lower()):
        return _build_grounded_fallback(ranked_sections)

    return answer


def query(question: str, provider: str = "gemini") -> dict:
    if not indexer.sections:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked_sections = indexer.search(question, k=5)
    grounded_sections = [(section, score) for section, score in ranked_sections if section.content.strip()] or ranked_sections
    grounded_sections = grounded_sections[:3]

    if not grounded_sections:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    response = get_llm(provider).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_prompt(question, grounded_sections)),
        ]
    )

    sources = [
        {
            "source": section.id,
            "heading": " > ".join(section.heading_path),
            "score": round(score, 3),
            "content": section.content[:240],
        }
        for section, score in grounded_sections
    ]

    return {
        "answer": _finalize_answer(response, grounded_sections),
        "sources": sources,
    }
