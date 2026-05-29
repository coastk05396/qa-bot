import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app import indexer, retrieval, routes
from app.schemas import ChatRequest


class MarkdownKnowledgeBaseTests(unittest.TestCase):
    def tearDown(self) -> None:
        indexer.sections = []
        indexer.rebuild_stats()

    def test_parse_markdown_creates_section_ids_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            doc_path = Path(tmp_dir) / "faq.md"
            doc_path.write_text(
                "# FAQ\n\n## Refund Timeline\n\nRefunds take 5-7 business days.\n",
                encoding="utf-8",
            )

            sections = indexer.parse_markdown(doc_path)

        self.assertEqual([section.id for section in sections], ["faq.md#faq", "faq.md#refund-timeline"])
        self.assertEqual(sections[1].heading_path, ["FAQ", "Refund Timeline"])
        self.assertIn("refunds", sections[1].tokens)

    def test_build_index_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            docs_dir = Path(tmp_dir) / "docs"
            docs_dir.mkdir()
            (docs_dir / "refund_policy.md").write_text(
                "# Refund Policy\n\n## Refund Timeline\n\nApproved refunds are processed within 5-7 business days.\n",
                encoding="utf-8",
            )
            index_path = Path(tmp_dir) / ".kb" / "index.json"

            with patch.object(indexer, "INDEX_PATH", index_path):
                files_indexed, sections_indexed = indexer.build_index(docs_dir)
                self.assertEqual(files_indexed, 1)
                self.assertGreaterEqual(sections_indexed, 2)
                self.assertTrue(index_path.exists())

                indexer.sections = []
                indexer.rebuild_stats()
                loaded_files, loaded_sections = indexer.load_index_json(index_path)

        self.assertEqual(loaded_files, 1)
        self.assertEqual(loaded_sections, sections_indexed)
        self.assertEqual(indexer.sections[1].id, "refund_policy.md#refund-timeline")

    def test_query_returns_pre_index_message(self) -> None:
        response = retrieval.query("How long do refunds take?")
        self.assertIn("not been indexed", response["answer"])
        self.assertEqual(response["sources"], [])

    def test_query_returns_cannot_confirm_when_search_is_empty(self) -> None:
        indexer.sections = [
            indexer.Section(
                id="refund_policy.md#refund-timeline",
                file="refund_policy.md",
                heading="Refund Timeline",
                heading_path=["Refund Policy", "Refund Timeline"],
                content="Approved refunds are processed within 5-7 business days.",
                tokens=indexer.tokenize("Refund Policy Refund Timeline Approved refunds are processed within 5-7 business days."),
            )
        ]
        indexer.rebuild_stats()

        response = retrieval.query("Which restaurants are nearby?")

        self.assertEqual(response["answer"], "I cannot confirm from the knowledge base.")
        self.assertEqual(response["sources"], [])

    def test_query_uses_llm_and_returns_sources(self) -> None:
        indexer.sections = [
            indexer.Section(
                id="refund_policy.md#refund-timeline",
                file="refund_policy.md",
                heading="Refund Timeline",
                heading_path=["Refund Policy", "Refund Timeline"],
                content="Approved refunds are processed within 5-7 business days.",
                tokens=indexer.tokenize("Refund Policy Refund Timeline Approved refunds are processed within 5-7 business days."),
            )
        ]
        indexer.rebuild_stats()

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
