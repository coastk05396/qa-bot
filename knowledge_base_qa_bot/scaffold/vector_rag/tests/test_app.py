import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from langchain.schema import Document

from app import indexer, main, memory, retrieval, routes
from app.schemas import ChatRequest, MemoryCompactRequest, MemoryLogEntry


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
            wiki_dir = Path(tmp_dir) / ".kb" / "wiki"
            wiki_dir.mkdir(parents=True)
            (wiki_dir / "index.md").write_text(
                "# Learned Refunds\n\n## Repeated Question\n\nRefund timing is the most repeated request in logs.\n",
                encoding="utf-8",
            )
            index_dir = Path(tmp_dir) / ".kb" / "faiss_index"
            embeddings = FakeEmbeddings()

            with patch.object(indexer, "INDEX_DIR", index_dir), patch.object(indexer, "WIKI_DIR", wiki_dir), patch.object(indexer, "get_embeddings", return_value=embeddings):
                files_indexed, sections_indexed = indexer.build_index(docs_dir)
                self.assertEqual(files_indexed, 2)
                self.assertGreater(sections_indexed, 0)
                self.assertTrue((index_dir / "metadata.json").exists())

                indexer.vectorstore = None
                indexer.files_indexed = 0
                indexer.sections_indexed = 0
                loaded_files, loaded_sections = indexer.load_vector_index(index_dir)

                self.assertEqual(loaded_files, 2)
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

    def test_query_allows_nearest_neighbor_context_without_token_overlap_gate(self) -> None:
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
            response = retrieval.query("hi")

        self.assertIn("refund_policy.md#refund-timeline", response["answer"])
        self.assertEqual(response["sources"][0]["source"], "refund_policy.md#refund-timeline")

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

    def test_startup_creates_memory_directories_before_loading_index(self) -> None:
        with patch.object(main, "ensure_memory_directories") as ensure_dirs, patch.object(main, "load_vector_index") as load_index:
            main.load_persisted_index()

        ensure_dirs.assert_called_once_with()
        load_index.assert_called_once_with()

    def test_chat_route_returns_answer_without_logging_immediately(self) -> None:
        append_logs = AsyncMock()
        with patch.object(routes, "query", return_value={"answer": "I cannot confirm from the knowledge base.", "sources": []}), patch.object(routes, "append_chat_logs", append_logs):
            response = asyncio.run(routes.chat(ChatRequest(query="Which restaurants are nearby?", session_id="session-2")))

        self.assertEqual(response["answer"], "I cannot confirm from the knowledge base.")
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

            with patch.object(memory, "KB_ROOT", kb_root), patch.object(memory, "LOGS_DIR", logs_dir), patch.object(memory, "WIKI_DIR", wiki_dir), patch.object(memory, "WIKI_INDEX_PATH", wiki_index_path), patch.object(memory, "get_llm") as get_llm, patch.object(memory, "_embed_questions", return_value=[[1.0, 0.0]]):
                payload = asyncio.run(memory.run_dreaming_etl(lambda: (0, 0)))

            self.assertEqual(payload["valid_blocks"], 0)
            self.assertFalse(payload["wiki_updated"])
            get_llm.assert_not_called()
            self.assertFalse(wiki_index_path.exists())
            self.assertFalse(log_path.exists())

    def test_dreaming_etl_deletes_empty_logs_after_processing(self) -> None:
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
                        memory.format_log_block(
                            "2026-05-30T10:00:00+00:00",
                            "session-9",
                            "Can I change my email?",
                            "Customers can change their email from Account Settings.",
                            "VALID",
                        ),
                        memory.format_log_block(
                            "2026-05-30T10:05:00+00:00",
                            "session-10",
                            "change can i my email",
                            "Customers can change their email from Account Settings.",
                            "VALID",
                        ),
                        memory.format_log_block(
                            "2026-05-30T10:10:00+00:00",
                            "session-11",
                            "can i change my email",
                            "Customers can change their email from Account Settings.",
                            "VALID",
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            fake_llm = SimpleNamespace(
                invoke=lambda _messages: SimpleNamespace(
                    content='{"entries": [{"question": "Can I change my email?", "answer": "Customers can change their email from Account Settings."}]}'
                )
            )
            build_calls: list[str] = []

            def fake_build_index() -> tuple[int, int]:
                build_calls.append("called")
                return 2, 2

            with patch.object(memory, "KB_ROOT", kb_root), patch.object(memory, "LOGS_DIR", logs_dir), patch.object(memory, "WIKI_DIR", wiki_dir), patch.object(memory, "WIKI_INDEX_PATH", wiki_index_path), patch.object(memory, "get_llm", return_value=fake_llm), patch.object(memory, "_embed_questions", return_value=[[1.0, 0.0], [0.98, 0.02], [0.99, 0.01]]):
                payload = asyncio.run(memory.run_dreaming_etl(fake_build_index))

            self.assertTrue(payload["wiki_updated"])
            self.assertEqual(payload["processed_session_ids"], ["session-9", "session-10", "session-11"])
            self.assertEqual(payload["deleted_files"], 1)
            self.assertEqual(build_calls, ["called"])
            self.assertFalse(log_path.exists())
            wiki_text = wiki_index_path.read_text(encoding="utf-8")
            self.assertIn("## Can I change my email?", wiki_text)
            self.assertIn("Customers can change their email from Account Settings.", wiki_text)

    def test_dreaming_etl_keeps_valid_logs_below_three_word_matches(self) -> None:
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
                        memory.format_log_block(
                            "2026-05-30T10:00:00+00:00",
                            "session-9",
                            "change email",
                            "Customers can change their email from Account Settings.",
                            "VALID",
                        ),
                        memory.format_log_block(
                            "2026-05-30T10:05:00+00:00",
                            "session-10",
                            "email change",
                            "Customers can change their email from Account Settings.",
                            "VALID",
                        ),
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
            self.assertIn("session-9", remaining)
            self.assertIn("session-10", remaining)

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

            with patch.object(memory, "KB_ROOT", kb_root), patch.object(memory, "LOGS_DIR", logs_dir), patch.object(memory, "WIKI_DIR", wiki_dir), patch.object(memory, "WIKI_INDEX_PATH", wiki_index_path), patch.object(memory, "get_llm", return_value=fake_llm), patch.object(memory, "_embed_questions", return_value=[[1.0, 0.0], [0.97, 0.03], [0.96, 0.04]]):
                payload = asyncio.run(memory.run_dreaming_etl(lambda: (0, 0)))

            self.assertTrue(payload["wiki_updated"])
            self.assertEqual(payload["processed_session_ids"], ["session-1", "session-2", "session-3"])
            self.assertFalse(log_path.exists())
            wiki_text = wiki_index_path.read_text(encoding="utf-8")
            self.assertIn("## when will i get money back ?", wiki_text)
            self.assertIn("Approved refunds are processed within 5-7 business days.", wiki_text)

    def test_chat_route_returns_json_http_error_on_backend_failure(self) -> None:
        with patch.object(routes, "query", side_effect=RuntimeError("OPENAI_API_KEY is not set")):
            with self.assertRaises(HTTPException) as context:
                asyncio.run(routes.chat(ChatRequest(query="How long do refunds take?", session_id="session-2")))

        self.assertEqual(context.exception.status_code, 500)
        self.assertIn("OPENAI_API_KEY", context.exception.detail)


if __name__ == "__main__":
    unittest.main()
