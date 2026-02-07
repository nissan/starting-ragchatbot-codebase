# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials. Uses ChromaDB for vector storage, Anthropic Claude for AI generation, and a vanilla HTML/JS frontend served by FastAPI.

## Commands

### Run the app
```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```
App serves at http://localhost:8000. Requires `ANTHROPIC_API_KEY` in a `.env` file at the project root.

### Install dependencies
```bash
uv sync
```

### No tests exist yet
There is no test suite in this codebase.

## Architecture

### Query Flow
User question → FastAPI (`POST /api/query`) → `RAGSystem.query()` → `AIGenerator` sends prompt + `search_course_content` tool to Claude API → Claude either answers directly or calls the tool → `CourseSearchTool` queries ChromaDB → results returned to Claude for a second API call (synthesis) → response + sources returned to frontend.

### Key Design Decisions
- **Tool-based RAG**: Instead of always retrieving context, Claude decides whether to search via Anthropic's tool-use feature. The tool definition is in `search_tools.py:CourseSearchTool.get_tool_definition()`.
- **Two ChromaDB collections**: `course_catalog` (course metadata for semantic name resolution) and `course_content` (chunked lesson text for content search). Both live in `backend/chroma_db/`.
- **Two-pass Claude calls**: When the tool is invoked, `AIGenerator._handle_tool_execution()` makes a second API call without tools to synthesize the final answer from tool results.
- **Session management is in-memory**: `SessionManager` stores conversation history in a dict keyed by session ID. History is passed to Claude as formatted text in the system prompt, not as message history.
- **Document format convention**: Course docs in `docs/` follow a structured text format with `Course Title:`, `Course Link:`, `Course Instructor:` headers, then `Lesson N: <title>` markers separating content.

### Backend Module Responsibilities
- `rag_system.py` — Orchestrator that wires all components; entry point is `query()` and `add_course_folder()`
- `ai_generator.py` — Claude API client with tool-use loop; owns the system prompt
- `vector_store.py` — ChromaDB wrapper; handles course name resolution, filtering, and search
- `search_tools.py` — Tool abstraction (`Tool` ABC, `CourseSearchTool`, `ToolManager`); formats search results and tracks sources
- `document_processor.py` — Parses course files, extracts metadata, chunks text with sentence-aware splitting
- `session_manager.py` — In-memory conversation history per session
- `config.py` — Dataclass with all tunable parameters (chunk size, model, etc.)
- `models.py` — Pydantic models: `Course`, `Lesson`, `CourseChunk`

### Frontend
Vanilla HTML/CSS/JS in `frontend/`. Uses `marked.js` for markdown rendering. Served as static files by FastAPI's `StaticFiles` mount at `/`. All API calls go to relative `/api/*` paths.

### Startup Behavior
On app startup (`app.py:startup_event`), all `.txt`/`.pdf`/`.docx` files in `docs/` are processed and indexed into ChromaDB. Courses already in the DB (matched by title) are skipped.
