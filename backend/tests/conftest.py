import sys
import os
from unittest.mock import MagicMock

import pytest

# Add backend directory to path so bare imports (vector_store, search_tools, etc.) work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager


# ---------------------------------------------------------------------------
# SearchResults fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_search_results():
    """SearchResults with 2 documents, metadata, and distances."""
    return SearchResults(
        documents=[
            "Lesson 1 covers the basics of retrieval-augmented generation.",
            "Lesson 2 dives into vector databases and embeddings.",
        ],
        metadata=[
            {"course_title": "Intro to RAG", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Intro to RAG", "lesson_number": 2, "chunk_index": 0},
        ],
        distances=[0.25, 0.42],
    )


@pytest.fixture
def empty_search_results():
    """SearchResults with no documents (empty but no error)."""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """SearchResults representing an error."""
    return SearchResults.empty("Search error: connection refused")


# ---------------------------------------------------------------------------
# Mock VectorStore
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store(sample_search_results):
    """A MagicMock standing in for VectorStore with common methods configured."""
    store = MagicMock()
    store.search.return_value = sample_search_results
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    store.get_course_link.return_value = "https://example.com/course/intro-rag"
    store.get_course_outline.return_value = {
        "title": "Intro to RAG",
        "course_link": "https://example.com/course/intro-rag",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "What is RAG?"},
            {"lesson_number": 2, "lesson_title": "Vector Databases"},
            {"lesson_number": 3, "lesson_title": "Putting It Together"},
        ],
    }
    return store


# ---------------------------------------------------------------------------
# Tool fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def course_search_tool(mock_vector_store):
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def course_outline_tool(mock_vector_store):
    return CourseOutlineTool(mock_vector_store)


@pytest.fixture
def tool_manager_with_tools(course_search_tool, course_outline_tool):
    """ToolManager with both CourseSearchTool and CourseOutlineTool registered."""
    tm = ToolManager()
    tm.register_tool(course_search_tool)
    tm.register_tool(course_outline_tool)
    return tm
