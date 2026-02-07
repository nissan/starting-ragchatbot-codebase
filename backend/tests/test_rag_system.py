"""Tests for rag_system.py — RAGSystem.query() orchestration."""

from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass

import pytest


@dataclass
class MockConfig:
    """Minimal config with all attributes RAGSystem.__init__ needs."""

    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    CHROMA_PATH: str = "./test_chroma"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MAX_RESULTS: int = 5
    ANTHROPIC_API_KEY: str = "test-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    MAX_HISTORY: int = 2
    use_ollama: bool = False


@pytest.fixture
def rag_mocks():
    """Patch all four dependencies at their import location in rag_system and return mocks."""
    with (
        patch("rag_system.VectorStore") as MockVS,
        patch("rag_system.AIGenerator") as MockAI,
        patch("rag_system.DocumentProcessor") as MockDP,
        patch("rag_system.SessionManager") as MockSM,
    ):
        # Configure mock instances
        mock_vs_instance = MockVS.return_value
        mock_ai_instance = MockAI.return_value
        mock_sm_instance = MockSM.return_value

        # Default return values
        mock_ai_instance.generate_response.return_value = "This is the AI response."
        mock_sm_instance.get_conversation_history.return_value = None

        yield {
            "VectorStore": MockVS,
            "AIGenerator": MockAI,
            "DocumentProcessor": MockDP,
            "SessionManager": MockSM,
            "vs": mock_vs_instance,
            "ai": mock_ai_instance,
            "sm": mock_sm_instance,
        }


@pytest.fixture
def rag_system(rag_mocks):
    """Create a RAGSystem with all dependencies mocked."""
    from rag_system import RAGSystem

    return RAGSystem(MockConfig())


# ===================================================================
# RAGSystem.query()
# ===================================================================


class TestRAGSystemQuery:

    def test_query_wraps_prompt(self, rag_system, rag_mocks):
        """generate_response receives the wrapped prompt."""
        rag_system.query("What is RAG?")

        call_kwargs = rag_mocks["ai"].generate_response.call_args.kwargs
        assert (
            call_kwargs["query"]
            == "Answer this question about course materials: What is RAG?"
        )

    def test_query_passes_tools_and_manager(self, rag_system, rag_mocks):
        """generate_response called with tool definitions and tool_manager."""
        rag_system.query("Tell me about embeddings")

        call_kwargs = rag_mocks["ai"].generate_response.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] is not None

    def test_query_with_session_history(self, rag_system, rag_mocks):
        """Gets history from session manager and passes to generate_response."""
        rag_mocks["sm"].get_conversation_history.return_value = (
            "User: Hi\nAssistant: Hello!"
        )

        rag_system.query("Follow-up question", session_id="sess_1")

        rag_mocks["sm"].get_conversation_history.assert_called_once_with("sess_1")
        call_kwargs = rag_mocks["ai"].generate_response.call_args.kwargs
        assert call_kwargs["conversation_history"] == "User: Hi\nAssistant: Hello!"

    def test_query_no_session_no_history(self, rag_system, rag_mocks):
        """When session_id=None, history is None and add_exchange not called."""
        rag_system.query("Standalone question", session_id=None)

        call_kwargs = rag_mocks["ai"].generate_response.call_args.kwargs
        assert call_kwargs["conversation_history"] is None
        rag_mocks["sm"].add_exchange.assert_not_called()

    def test_query_retrieves_and_resets_sources(self, rag_system, rag_mocks):
        """Sources from get_last_sources() returned; reset_sources() called after."""
        # The tool_manager is a real ToolManager instance on rag_system, so we mock its methods
        rag_system.tool_manager.get_last_sources = MagicMock(
            return_value=[{"text": "Intro to RAG", "link": "https://example.com"}]
        )
        rag_system.tool_manager.reset_sources = MagicMock()

        response, sources = rag_system.query("question")

        assert sources == [{"text": "Intro to RAG", "link": "https://example.com"}]
        rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_updates_session_history(self, rag_system, rag_mocks):
        """add_exchange called with original query and response."""
        rag_mocks["ai"].generate_response.return_value = "The answer is 42."

        rag_system.query("What is the meaning of life?", session_id="sess_1")

        rag_mocks["sm"].add_exchange.assert_called_once_with(
            "sess_1", "What is the meaning of life?", "The answer is 42."
        )

    def test_query_does_not_return_query_failed(self, rag_system, rag_mocks):
        """Response should not contain 'query failed' or 'error' (basic smoke check)."""
        response, sources = rag_system.query("Tell me about RAG")

        assert "query failed" not in response.lower()
        assert "error" not in response.lower()
