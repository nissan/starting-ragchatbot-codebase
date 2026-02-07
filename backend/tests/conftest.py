import sys
import os
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional

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


# ---------------------------------------------------------------------------
# API test fixtures — standalone FastAPI app (avoids static-files mount)
# ---------------------------------------------------------------------------

# Response models matching app.py
class _Source(BaseModel):
    text: str
    link: Optional[str] = None

class _QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class _QueryResponse(BaseModel):
    answer: str
    sources: List[_Source]
    session_id: str

class _CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


@pytest.fixture
def mock_rag_system():
    """A MagicMock standing in for RAGSystem with common methods configured."""
    rag = MagicMock()
    rag.query.return_value = (
        "RAG stands for Retrieval-Augmented Generation.",
        [{"text": "Lesson 1: Intro to RAG", "link": "https://example.com/lesson/1"}],
    )
    rag.session_manager.create_session.return_value = "session_1"
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Intro to RAG", "Advanced Embeddings"],
    }
    return rag


def _build_test_app(rag_system):
    """Build a minimal FastAPI app with the same routes as app.py but no static files."""
    test_app = FastAPI()

    @test_app.post("/api/query", response_model=_QueryResponse)
    async def query_documents(request: _QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return _QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        rag_system.session_manager.delete_session(session_id)
        return {"status": "ok"}

    @test_app.get("/api/courses", response_model=_CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return _CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return test_app


@pytest.fixture
def test_app(mock_rag_system):
    """A FastAPI test app wired to mock_rag_system (no static files)."""
    return _build_test_app(mock_rag_system)


@pytest.fixture
def client(test_app):
    """Starlette TestClient for the test app."""
    return TestClient(test_app)
