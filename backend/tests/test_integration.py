"""Integration tests — real ChromaDB + real embeddings, only Anthropic API mocked."""

import os
import sys
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest

# Backend imports (conftest.py already adds backend to sys.path)
from vector_store import VectorStore
from document_processor import DocumentProcessor
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
COURSE3_PATH = os.path.join(DOCS_DIR, "course3_script.txt")


# ---------------------------------------------------------------------------
# Session-scoped fixtures (expensive — run once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_vector_store(tmp_path_factory):
    """Real VectorStore backed by a temporary ChromaDB directory.

    Indexes ``docs/course3_script.txt`` via DocumentProcessor so all
    downstream tests operate on real embedded data.
    """
    chroma_dir = str(tmp_path_factory.mktemp("chroma"))
    store = VectorStore(
        chroma_path=chroma_dir,
        embedding_model="all-MiniLM-L6-v2",
        max_results=5,
    )

    processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
    course, chunks = processor.process_course_document(COURSE3_PATH)

    store.add_course_metadata(course)
    store.add_course_content(chunks)

    return store


@pytest.fixture(scope="session")
def indexed_course():
    """Independently process course3_script.txt to get expected values."""
    processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
    return processor.process_course_document(COURSE3_PATH)


# ---------------------------------------------------------------------------
# Function-scoped fixtures (cheap — fresh per test)
# ---------------------------------------------------------------------------


@pytest.fixture
def real_search_tool(real_vector_store):
    return CourseSearchTool(real_vector_store)


@pytest.fixture
def real_outline_tool(real_vector_store):
    return CourseOutlineTool(real_vector_store)


@pytest.fixture
def real_tool_manager(real_search_tool, real_outline_tool):
    tm = ToolManager()
    tm.register_tool(real_search_tool)
    tm.register_tool(real_outline_tool)
    return tm


# ===================================================================
# TestVectorStoreSearch (5 tests)
# ===================================================================


class TestVectorStoreSearch:
    """Search behaviour against real ChromaDB + embeddings."""

    def test_search_returns_relevant_results(self, real_vector_store):
        results = real_vector_store.search(query="embeddings")

        assert not results.is_empty()
        assert results.error is None
        # Results should contain actual course content
        for doc in results.documents:
            assert len(doc) > 0

    def test_search_with_course_name_filter(self, real_vector_store, indexed_course):
        course, _ = indexed_course
        # Use a partial name — the embedding similarity resolver should match
        results = real_vector_store.search(
            query="retrieval", course_name="Advanced Retrieval"
        )

        assert not results.is_empty()
        for meta in results.metadata:
            assert meta["course_title"] == course.title

    def test_search_with_lesson_number_filter(self, real_vector_store):
        results = real_vector_store.search(query="retrieval", lesson_number=1)

        assert not results.is_empty()
        for meta in results.metadata:
            assert meta["lesson_number"] == 1

    def test_search_with_combined_filters(self, real_vector_store, indexed_course):
        course, _ = indexed_course
        results = real_vector_store.search(
            query="retrieval",
            course_name="Advanced Retrieval",
            lesson_number=1,
        )

        assert not results.is_empty()
        for meta in results.metadata:
            assert meta["course_title"] == course.title
            assert meta["lesson_number"] == 1

    def test_search_gibberish_returns_results(self, real_vector_store):
        """Nonsensical query shouldn't crash; distances should be worse than a good query."""
        gibberish = real_vector_store.search(query="xyzzy plugh zork")
        relevant = real_vector_store.search(query="embeddings retrieval")

        assert gibberish.error is None
        # Both return results (nearest-neighbour always returns something)
        assert not gibberish.is_empty()
        assert not relevant.is_empty()
        # Average distance for gibberish should be higher (worse) than relevant
        avg_gibberish = sum(gibberish.distances) / len(gibberish.distances)
        avg_relevant = sum(relevant.distances) / len(relevant.distances)
        assert avg_gibberish > avg_relevant


# ===================================================================
# TestVectorStoreMetadata (3 tests)
# ===================================================================


class TestVectorStoreMetadata:
    """Course catalog metadata stored in and retrieved from ChromaDB."""

    def test_get_course_outline_returns_full_structure(
        self, real_vector_store, indexed_course
    ):
        course, _ = indexed_course
        outline = real_vector_store.get_course_outline(course.title)

        assert outline is not None
        assert outline["title"] == course.title
        assert outline["course_link"] is not None
        assert len(outline["lessons"]) == len(course.lessons)

        # Each lesson has the expected keys
        for lesson in outline["lessons"]:
            assert "lesson_number" in lesson
            assert "lesson_title" in lesson

    def test_resolve_course_name_partial_match(self, real_vector_store, indexed_course):
        """Informal / partial name resolves to exact stored title via embeddings."""
        course, _ = indexed_course
        resolved = real_vector_store._resolve_course_name("Advanced Retrieval")
        assert resolved == course.title

    def test_get_existing_course_titles(self, real_vector_store, indexed_course):
        course, _ = indexed_course
        titles = real_vector_store.get_existing_course_titles()

        assert course.title in titles
        assert real_vector_store.get_course_count() >= 1


# ===================================================================
# TestSearchToolIntegration (3 tests)
# ===================================================================


class TestSearchToolIntegration:
    """CourseSearchTool and CourseOutlineTool with real VectorStore."""

    def test_execute_returns_formatted_content(self, real_search_tool):
        result = real_search_tool.execute(query="embeddings retrieval")

        # Formatted output should contain bracketed headers
        assert "[" in result
        assert "]" in result
        # Should contain actual course text (not empty)
        assert len(result) > 50

    def test_execute_populates_sources_with_real_links(self, real_search_tool):
        real_search_tool.execute(query="query expansion")

        sources = real_search_tool.last_sources
        assert len(sources) > 0
        for source in sources:
            assert "text" in source
            assert source["link"] is not None

    def test_outline_tool_returns_real_lessons(self, real_outline_tool, indexed_course):
        course, _ = indexed_course
        result = real_outline_tool.execute(course_name=course.title)

        assert f"Course: {course.title}" in result
        assert f"Total lessons: {len(course.lessons)}" in result
        # Each lesson line present
        for lesson in course.lessons:
            assert f"Lesson {lesson.lesson_number}:" in result


# ===================================================================
# TestToolManagerIntegration (2 tests)
# ===================================================================


class TestToolManagerIntegration:
    """ToolManager wired to real search + outline tools."""

    def test_execute_tool_by_name_with_real_search(self, real_tool_manager):
        result = real_tool_manager.execute_tool(
            "search_course_content", query="vector database"
        )

        assert len(result) > 0
        assert "[" in result  # formatted header present

    def test_source_tracking_through_tool_manager(self, real_tool_manager):
        real_tool_manager.execute_tool(
            "search_course_content", query="reranking cross encoder"
        )

        sources = real_tool_manager.get_last_sources()
        assert len(sources) > 0
        assert "text" in sources[0]

        real_tool_manager.reset_sources()
        assert real_tool_manager.get_last_sources() == []


# ===================================================================
# TestRAGSystemIntegration (2 tests)
# ===================================================================


@dataclass
class _IntegrationConfig:
    """Config that points at a real ChromaDB path for integration tests."""

    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    CHROMA_PATH: str = ""  # filled in by fixture
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MAX_RESULTS: int = 5
    ANTHROPIC_API_KEY: str = "test-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    MAX_HISTORY: int = 2
    use_ollama: bool = False


class TestRAGSystemIntegration:
    """RAGSystem with real VectorStore + mock AIGenerator."""

    @pytest.fixture
    def rag_system_real_store(self, real_vector_store):
        """RAGSystem backed by real ChromaDB; only ai_generator is mocked."""
        from rag_system import RAGSystem

        # Get the ChromaDB path from the real store's client
        chroma_path = real_vector_store.client._identifier
        cfg = _IntegrationConfig(CHROMA_PATH=chroma_path)

        with patch("rag_system.AIGenerator") as MockAI:
            mock_ai = MockAI.return_value
            rag = RAGSystem(cfg)
            # Replace VectorStore so it shares the same data
            rag.vector_store = real_vector_store
            # Rewire search tools to point at the shared store
            rag.search_tool = CourseSearchTool(real_vector_store)
            rag.outline_tool = CourseOutlineTool(real_vector_store)
            rag.tool_manager = ToolManager()
            rag.tool_manager.register_tool(rag.search_tool)
            rag.tool_manager.register_tool(rag.outline_tool)

            yield rag, mock_ai

    def test_query_with_tool_use_returns_sources(self, rag_system_real_store):
        """Mock AI invokes real tool_manager → returns real sources."""
        rag, mock_ai = rag_system_real_store

        def fake_generate(query, conversation_history, tools, tool_manager):
            # Simulate Claude calling the search tool
            tool_manager.execute_tool("search_course_content", query="embeddings")
            return "Here is the answer based on course content."

        mock_ai.generate_response.side_effect = fake_generate

        response, sources = rag.query("Tell me about embeddings")

        assert response == "Here is the answer based on course content."
        assert len(sources) > 0
        assert "text" in sources[0]
        assert sources[0]["link"] is not None

    def test_query_without_tool_use_returns_direct_response(
        self, rag_system_real_store
    ):
        """Mock AI returns direct answer → empty sources, no crash."""
        rag, mock_ai = rag_system_real_store
        mock_ai.generate_response.return_value = "Hello! How can I help?"

        response, sources = rag.query("Hello")

        assert response == "Hello! How can I help?"
        assert sources == []
