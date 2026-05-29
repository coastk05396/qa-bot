import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app import indexer, main, memory, retrieval, routes
from app.schemas import ChatRequest, MemoryCompactRequest, MemoryLogEntry


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
            wiki_dir = Path(tmp_dir) / ".kb" / "wiki"
            wiki_dir.mkdir(parents=True)
            (wiki_dir / "index.md").write_text(
                "# Learned Refunds\n\n## Repeated Question\n\nRefund requests often ask about timing.\n",
                encoding="utf-8",
            )
            index_path = Path(tmp_dir) / ".kb" / "index.json"

            with patch.object(indexer, "INDEX_PATH", index_path), patch.object(indexer, "WIKI_DIR", wiki_dir):
                files_indexed, sections_indexed = indexer.build_index(docs_dir)
                self.assertEqual(files_indexed, 2)
                self.assertGreaterEqual(sections_indexed, 2)
                self.assertTrue(index_path.exists())

                indexer.sections = []
                indexer.rebuild_stats()
                loaded_files, loaded_sections = indexer.load_index_json(index_path)

                self.assertEqual(loaded_files, 2)
                self.assertEqual(loaded_sections, sections_indexed)
                self.assertEqual(indexer.sections[1].id, "refund_policy.md#refund-timeline")
                self.assertIn("index.md#repeated-question", [section.id for section in indexer.sections])

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

    def test_query_preserves_explicit_cannot_confirm_response(self) -> None:
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
            invoke=lambda _messages: SimpleNamespace(content="I cannot confirm from the knowledge base.")
        )

        with patch.object(retrieval, "get_llm", return_value=fake_llm):
            response = retrieval.query("How long do refunds take?")

        self.assertEqual(response["answer"], "I cannot confirm from the knowledge base.")

    def test_frontend_route_points_to_shared_ui(self) -> None:
        response = routes.frontend()
        self.assertTrue(routes.FRONTEND_PATH.exists())
        self.assertEqual(Path(response.path), routes.FRONTEND_PATH)

    def test_startup_creates_memory_directories_before_loading_index(self) -> None:
        with patch.object(main, "ensure_memory_directories") as ensure_dirs, patch.object(main, "load_index_json") as load_index:
            main.load_persisted_index()

        ensure_dirs.assert_called_once_with()
        load_index.assert_called_once_with()

    def test_chat_route_returns_answer_without_logging_immediately(self) -> None:
        append_logs = AsyncMock()
        with patch.object(routes, "query", return_value={"answer": "Refunds take 5-7 business days.", "sources": [{"source": "refund_policy.md#refund-timeline", "heading": "Refund Policy > Refund Timeline", "score": 0.9, "content": "Refunds take 5-7 business days."}]}), patch.object(routes, "append_chat_logs", append_logs):
            response = asyncio.run(routes.chat(ChatRequest(query="How long do refunds take?", session_id="session-1")))

        self.assertEqual(response["answer"], "Refunds take 5-7 business days.")
        append_logs.assert_not_awaited()

    def test_compact_memory_flushes_buffered_entries(self) -> None:
        append_logs = AsyncMock(return_value=2)
        request = MemoryCompactRequest(
            session_id="session-default",
            entries=[
                MemoryLogEntry(question="Q1", answer="A1", status="VALID"),
                MemoryLogEntry(question="Q2", answer="A2", status="DEFAULT"),
            ],
        )

        with patch.object(routes, "append_chat_logs", append_logs):
            response = asyncio.run(routes.compact_memory(request))

        append_logs.assert_awaited_once_with(
            "session-default",
            [
                {"question": "Q1", "answer": "A1", "status": "VALID"},
                {"question": "Q2", "answer": "A2", "status": "DEFAULT"},
            ],
        )
        self.assertTrue(response.ok)
        self.assertEqual(response.buffers_flushed, 2)

    def test_dreaming_etl_compacts_logs_and_appends_wiki(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            kb_root = Path(tmp_dir) / ".kb"
            logs_dir = kb_root / "logs"
            wiki_dir = kb_root / "wiki"
            wiki_index_path = wiki_dir / "index.md"
            logs_dir.mkdir(parents=True)
            wiki_dir.mkdir(parents=True)
            log_path = logs_dir / "2026-05-30.md"
            log_path.write_text(
                "".join(
                    [
                        memory.format_log_block("2026-05-30T10:00:00+00:00", "session-1", "What is the refund timing?", "Refunds take 5-7 business days.", "VALID"),
                        memory.format_log_block("2026-05-30T10:05:00+00:00", "session-2", "What is the refund timing?", "Refunds take 5-7 business days.", "VALID"),
                        memory.format_log_block("2026-05-30T10:07:00+00:00", "session-4", "timing the refund what is", "Refunds take 5-7 business days.", "VALID"),
                        memory.format_log_block("2026-05-30T10:10:00+00:00", "session-3", "What is the refund timing?", "I cannot confirm from the knowledge base.", "REJECTED"),
                    ]
                ),
                encoding="utf-8",
            )

            fake_llm = SimpleNamespace(
                invoke=lambda _messages: SimpleNamespace(
                    content='{"entries": [{"question": "What is the refund timing?", "answer": "Approved refunds are processed within 5-7 business days."}]}'
                )
            )
            build_calls: list[str] = []

            def fake_build_index() -> tuple[int, int]:
                build_calls.append("called")
                return 2, 3

            with patch.object(memory, "KB_ROOT", kb_root), patch.object(memory, "LOGS_DIR", logs_dir), patch.object(memory, "WIKI_DIR", wiki_dir), patch.object(memory, "WIKI_INDEX_PATH", wiki_index_path), patch.object(memory, "get_llm", return_value=fake_llm), patch.object(memory, "_embed_questions", return_value=[[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]]):
                payload = asyncio.run(memory.run_dreaming_etl(fake_build_index))

            self.assertTrue(payload["wiki_updated"])
            self.assertEqual(payload["processed_session_ids"], ["session-1", "session-2", "session-4"])
            self.assertEqual(payload["rejected_blocks_removed"], 1)
            self.assertEqual(build_calls, ["called"])
            self.assertTrue(wiki_index_path.exists())
            wiki_text = wiki_index_path.read_text(encoding="utf-8")
            self.assertIn("## What is the refund timing?", wiki_text)
            self.assertIn("Approved refunds are processed within 5-7 business days.", wiki_text)
            self.assertFalse(log_path.exists())

    def test_dreaming_etl_requires_three_matching_question_word_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            kb_root = Path(tmp_dir) / ".kb"
            logs_dir = kb_root / "logs"
            wiki_dir = kb_root / "wiki"
            wiki_index_path = wiki_dir / "index.md"
            logs_dir.mkdir(parents=True)
            wiki_dir.mkdir(parents=True)
            log_path = logs_dir / "2026-05-30.md"
            log_path.write_text(
                "".join(
                    [
                        memory.format_log_block("2026-05-30T10:00:00+00:00", "session-1", "refund timing", "Refunds take 5-7 business days.", "VALID"),
                        memory.format_log_block("2026-05-30T10:05:00+00:00", "session-2", "timing refund", "Refunds take 5-7 business days.", "VALID"),
                    ]
                ),
                encoding="utf-8",
            )

            with patch.object(memory, "KB_ROOT", kb_root), patch.object(memory, "LOGS_DIR", logs_dir), patch.object(memory, "WIKI_DIR", wiki_dir), patch.object(memory, "WIKI_INDEX_PATH", wiki_index_path), patch.object(memory, "get_llm") as get_llm, patch.object(memory, "_embed_questions", return_value=[[1.0, 0.0], [0.0, 1.0]]):
                payload = asyncio.run(memory.run_dreaming_etl(lambda: (0, 0)))

            self.assertEqual(payload["valid_blocks"], 2)
            self.assertEqual(payload["processed_session_ids"], [])
            self.assertFalse(payload["wiki_updated"])
            get_llm.assert_not_called()
            remaining = log_path.read_text(encoding="utf-8")
            self.assertIn("session-1", remaining)
            self.assertIn("session-2", remaining)

    def test_dreaming_etl_groups_questions_by_embedding_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            kb_root = Path(tmp_dir) / ".kb"
            logs_dir = kb_root / "logs"
            wiki_dir = kb_root / "wiki"
            wiki_index_path = wiki_dir / "index.md"
            logs_dir.mkdir(parents=True)
            wiki_dir.mkdir(parents=True)
            log_path = logs_dir / "2026-05-30.md"
            log_path.write_text(
                "".join(
                    [
                        memory.format_log_block("2026-05-30T10:00:00+00:00", "session-1", "when can i get money back", "A1", "VALID"),
                        memory.format_log_block("2026-05-30T10:05:00+00:00", "session-2", "when will i get my money", "A2", "VALID"),
                        memory.format_log_block("2026-05-30T10:10:00+00:00", "session-3", "get money when back", "A3", "VALID"),
                    ]
                ),
                encoding="utf-8",
            )

            fake_llm = SimpleNamespace(
                invoke=lambda _messages: SimpleNamespace(content='{"entries": [{"question": "when will i get money back ?", "answer": "Approved refunds are processed within 5-7 business days."}]}')
            )

            with patch.object(memory, "KB_ROOT", kb_root), patch.object(memory, "LOGS_DIR", logs_dir), patch.object(memory, "WIKI_DIR", wiki_dir), patch.object(memory, "WIKI_INDEX_PATH", wiki_index_path), patch.object(memory, "get_llm", return_value=fake_llm), patch.object(memory, "_embed_questions", return_value=[[1.0, 0.0], [0.96, 0.04], [0.95, 0.05]]):
                payload = asyncio.run(memory.run_dreaming_etl(lambda: (0, 0)))

            self.assertTrue(payload["wiki_updated"])
            self.assertEqual(payload["processed_session_ids"], ["session-1", "session-2", "session-3"])
            self.assertFalse(log_path.exists())
            wiki_text = wiki_index_path.read_text(encoding="utf-8")
            self.assertIn("## when will i get money back ?", wiki_text)
            self.assertIn("Approved refunds are processed within 5-7 business days.", wiki_text)

    def test_dreaming_etl_ignores_default_prompts_for_wiki_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            kb_root = Path(tmp_dir) / ".kb"
            logs_dir = kb_root / "logs"
            wiki_dir = kb_root / "wiki"
            wiki_index_path = wiki_dir / "index.md"
            logs_dir.mkdir(parents=True)
            wiki_dir.mkdir(parents=True)
            log_path = logs_dir / "2026-05-30.md"
            log_path.write_text(
                memory.format_log_block(
                    "2026-05-30T10:00:00+00:00",
                    "session-default",
                    "What account-help steps can the bot confirm from the knowledge base?",
                    "Customers can reset their password from the sign-in page.",
                    "DEFAULT",
                ),
                encoding="utf-8",
            )

            with patch.object(memory, "KB_ROOT", kb_root), patch.object(memory, "LOGS_DIR", logs_dir), patch.object(memory, "WIKI_DIR", wiki_dir), patch.object(memory, "WIKI_INDEX_PATH", wiki_index_path), patch.object(memory, "get_llm") as get_llm:
                payload = asyncio.run(memory.run_dreaming_etl(lambda: (0, 0)))

            self.assertEqual(payload["valid_blocks"], 0)
            self.assertFalse(payload["wiki_updated"])
            get_llm.assert_not_called()
            self.assertFalse(wiki_index_path.exists())
            self.assertFalse(log_path.exists())

    def test_chat_route_returns_json_http_error_on_backend_failure(self) -> None:
        with patch.object(routes, "query", side_effect=RuntimeError("OPENAI_API_KEY is not set")):
            with self.assertRaises(HTTPException) as context:
                asyncio.run(routes.chat(ChatRequest(query="How long do refunds take?", session_id="session-1")))

        self.assertEqual(context.exception.status_code, 500)
        self.assertIn("OPENAI_API_KEY", context.exception.detail)


if __name__ == "__main__":
    unittest.main()
