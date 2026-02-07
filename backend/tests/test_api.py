"""Tests for FastAPI API endpoints.

Uses a standalone test app (built in conftest.py) that mirrors the routes
in app.py but skips the static-files mount and startup event, so tests
run without a real frontend directory or ChromaDB instance.
"""

import pytest


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    """Tests for POST /api/query."""

    def test_query_returns_answer_and_sources(self, client, mock_rag_system):
        resp = client.post("/api/query", json={"query": "What is RAG?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "RAG stands for Retrieval-Augmented Generation."
        assert len(body["sources"]) == 1
        assert body["sources"][0]["text"] == "Lesson 1: Intro to RAG"
        assert body["sources"][0]["link"] == "https://example.com/lesson/1"

    def test_query_creates_session_when_missing(self, client, mock_rag_system):
        resp = client.post("/api/query", json={"query": "Hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "session_1"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_uses_provided_session_id(self, client, mock_rag_system):
        resp = client.post(
            "/api/query",
            json={"query": "Hello", "session_id": "my-session"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "my-session"
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_query_passes_question_to_rag_system(self, client, mock_rag_system):
        client.post(
            "/api/query",
            json={"query": "Explain embeddings", "session_id": "s1"},
        )
        mock_rag_system.query.assert_called_once_with("Explain embeddings", "s1")

    def test_query_returns_500_on_rag_error(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("ChromaDB unavailable")
        resp = client.post("/api/query", json={"query": "anything"})
        assert resp.status_code == 500
        assert "ChromaDB unavailable" in resp.json()["detail"]

    def test_query_missing_body_returns_422(self, client):
        resp = client.post("/api/query")
        assert resp.status_code == 422

    def test_query_missing_query_field_returns_422(self, client):
        resp = client.post("/api/query", json={"session_id": "s1"})
        assert resp.status_code == 422

    def test_query_empty_string_query(self, client, mock_rag_system):
        """Empty string is still valid JSON; the endpoint forwards it to the RAG system."""
        resp = client.post("/api/query", json={"query": ""})
        assert resp.status_code == 200
        mock_rag_system.query.assert_called_once()

    def test_query_sources_with_null_link(self, client, mock_rag_system):
        mock_rag_system.query.return_value = (
            "Some answer",
            [{"text": "Source without link", "link": None}],
        )
        resp = client.post("/api/query", json={"query": "test"})
        assert resp.status_code == 200
        assert resp.json()["sources"][0]["link"] is None

    def test_query_no_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("Direct answer", [])
        resp = client.post("/api/query", json={"query": "hi"})
        assert resp.status_code == 200
        assert resp.json()["sources"] == []


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:
    """Tests for GET /api/courses."""

    def test_courses_returns_stats(self, client, mock_rag_system):
        resp = client.get("/api/courses")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_courses"] == 2
        assert "Intro to RAG" in body["course_titles"]
        assert "Advanced Embeddings" in body["course_titles"]

    def test_courses_returns_500_on_error(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("DB down")
        resp = client.get("/api/courses")
        assert resp.status_code == 500
        assert "DB down" in resp.json()["detail"]

    def test_courses_empty_catalog(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        resp = client.get("/api/courses")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_courses"] == 0
        assert body["course_titles"] == []


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

class TestDeleteSessionEndpoint:
    """Tests for DELETE /api/session/{session_id}."""

    def test_delete_session_returns_ok(self, client, mock_rag_system):
        resp = client.delete("/api/session/sess-abc")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        mock_rag_system.session_manager.delete_session.assert_called_once_with("sess-abc")

    def test_delete_nonexistent_session_still_ok(self, client, mock_rag_system):
        """Mirrors app.py behaviour — delete_session is fire-and-forget."""
        resp = client.delete("/api/session/does-not-exist")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Request / response edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Cross-cutting edge cases for all endpoints."""

    def test_wrong_method_on_query_returns_405(self, client):
        resp = client.get("/api/query")
        assert resp.status_code == 405

    def test_wrong_method_on_courses_returns_405(self, client):
        resp = client.post("/api/courses")
        assert resp.status_code == 405

    def test_unknown_route_returns_404(self, client):
        resp = client.get("/api/nonexistent")
        assert resp.status_code == 404

    def test_query_content_type_must_be_json(self, client):
        resp = client.post("/api/query", content="not json", headers={"content-type": "text/plain"})
        assert resp.status_code == 422
