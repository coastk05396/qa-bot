import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from langchain.schema import Document

from app import indexer, retrieval, routes
from app.schemas import ChatRequest


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(len(lowered)),
            float(lowered.count("refund")),
            float(lowered.count("email")),
        ]


class FakeVectorStore:
    def __init__(self, results):
        self._results = results

    def similarity_search_with_score(self, _query: str, k: int = 3):
        return self._results[:k]


class VectorKnowledgeBaseTests(unittest.TestCase):
    def tearDown(self) -> None:
        indexer.vectorstore = None
        indexer.files_indexed = 0
        indexer.sections_indexed = 0

    def test_load_markdown_sections_preserves_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            doc_path = Path(tmp_dir) / "account_help.md"
            doc_path.write_text(
                "# Account Help\n\n## Change Email Address\n\nCustomers can change their email address from Account Settings.\n",
                encoding="utf-8",
            )

            documents = indexer.load_markdown_sections(doc_path)

        self.assertEqual(documents[1].metadata["source"], "account_help.md#change-email-address")
        self.assertEqual(documents[1].metadata["heading"], "Account Help > Change Email Address")

    def test_build_index_and_reload_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            docs_dir = Path(tmp_dir) / "docs"
            docs_dir.mkdir()
            (docs_dir / "refund_policy.md").write_text(
                "# Refund Policy\n\n## Refund Timeline\n\nApproved refunds are processed within 5-7 business days.\n",
                encoding="utf-8",
            )
            index_dir = Path(tmp_dir) / ".kb" / "faiss_index"
            embeddings = FakeEmbeddings()

            with patch.object(indexer, "INDEX_DIR", index_dir), patch.object(indexer, "get_embeddings", return_value=embeddings):
                files_indexed, sections_indexed = indexer.build_index(docs_dir)
                self.assertEqual(files_indexed, 1)
                self.assertGreater(sections_indexed, 0)
                self.assertTrue((index_dir / "metadata.json").exists())

                indexer.vectorstore = None
                indexer.files_indexed = 0
                indexer.sections_indexed = 0
                loaded_files, loaded_sections = indexer.load_vector_index(index_dir)

        self.assertEqual(loaded_files, 1)
        self.assertEqual(loaded_sections, sections_indexed)
        self.assertIsNotNone(indexer.vectorstore)

    def test_query_returns_pre_index_message(self) -> None:
        response = retrieval.query("How long do refunds take?")
        self.assertIn("not been indexed", response["answer"])
        self.assertEqual(response["sources"], [])

    def test_query_returns_cannot_confirm_when_no_chunks_match(self) -> None:
        indexer.vectorstore = FakeVectorStore([])

        response = retrieval.query("Which restaurants are nearby?")

        self.assertEqual(response["answer"], "I cannot confirm from the knowledge base.")
        self.assertEqual(response["sources"], [])

    def test_query_uses_llm_and_returns_sources(self) -> None:
        indexer.vectorstore = FakeVectorStore(
            [
                (
                    Document(
                        page_content="Heading: Refund Policy > Refund Timeline\n\nApproved refunds are processed within 5-7 business days.",
                        metadata={
                            "source": "refund_policy.md#refund-timeline",
                            "heading": "Refund Policy > Refund Timeline",
                        },
                    ),
                    0.123,
                )
            ]
        )
        fake_llm = SimpleNamespace(
            invoke=lambda _messages: SimpleNamespace(
                content="Approved refunds are processed within 5-7 business days. [refund_policy.md#refund-timeline]"
            )
        )

        with patch.object(retrieval, "get_llm", return_value=fake_llm):
            response = retrieval.query("How long do refunds take?")

        self.assertIn("refund_policy.md#refund-timeline", response["answer"])
        self.assertEqual(response["sources"][0]["source"], "refund_policy.md#refund-timeline")

    def test_frontend_route_points_to_shared_ui(self) -> None:
        response = routes.frontend()
        self.assertTrue(routes.FRONTEND_PATH.exists())
        self.assertEqual(Path(response.path), routes.FRONTEND_PATH)

    def test_chat_route_returns_json_http_error_on_backend_failure(self) -> None:
        with patch.object(routes, "query", side_effect=RuntimeError("OPENAI_API_KEY is not set")):
            with self.assertRaises(HTTPException) as context:
                routes.chat(ChatRequest(query="How long do refunds take?"))

        self.assertEqual(context.exception.status_code, 500)
        self.assertIn("OPENAI_API_KEY", context.exception.detail)


if __name__ == "__main__":
    unittest.main()
