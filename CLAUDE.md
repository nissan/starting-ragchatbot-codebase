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
App serves at http://localhost:8000. With `ANTHROPIC_API_KEY` in a `.env` file at the project root it uses Claude; without it, falls back to Ollama (`llama3.2:3b` by default).

### Install dependencies
```bash
uv sync
```

### Run tests
```bash
cd backend && uv run pytest tests/ -v
```
54 tests across `test_ai_generator.py`, `test_search_tools.py`, `test_rag_system.py`, and `test_integration.py`.

### Code quality
```bash
./scripts/quality.sh          # Run all checks (formatting + tests)
./scripts/quality.sh format   # Auto-format code with black
./scripts/quality.sh check    # Check formatting without changes
./scripts/quality.sh test     # Run tests only
```
Black is configured in `pyproject.toml` (line-length 88, Python 3.13 target).

## Architecture

### Query Flow
User question → FastAPI (`POST /api/query`) → `RAGSystem.query()` → `AIGenerator` (or `OllamaGenerator`) sends prompt + tools to the LLM → LLM either answers directly or calls a tool → tool queries ChromaDB → results returned to LLM → LLM may chain a second tool call (up to `MAX_TOOL_ROUNDS=2`) or synthesize a final answer → response + sources returned to frontend.

### Key Design Decisions
- **Tool-based RAG**: Instead of always retrieving context, the LLM decides whether to search via tool-use. Tool definitions live in `search_tools.py` (`CourseSearchTool`, `CourseOutlineTool`).
- **Two generator backends**: `AIGenerator` (Anthropic Claude API) and `OllamaGenerator` (local Ollama). Selected via `config.use_ollama` (True when no `ANTHROPIC_API_KEY`). Both share the same `generate_response()` interface and system prompt.
- **Multi-round tool execution** (`MAX_TOOL_ROUNDS = 2`): `_handle_tool_execution()` loops up to 2 rounds. Non-final rounds include tools so the LLM can chain calls (e.g., get outline → search by topic). The final round omits tools to force synthesis. Early exit if the LLM responds with text before hitting the limit.
- **Two ChromaDB collections**: `course_catalog` (course metadata for semantic name resolution) and `course_content` (chunked lesson text for content search). Both live in `backend/chroma_db/`.
- **Source accumulation**: `ToolManager.get_last_sources()` collects deduplicated sources from all tools that were called, not just the first. This ensures multi-round calls surface sources from every tool invoked.
- **Session management is in-memory**: `SessionManager` stores conversation history in a dict keyed by session ID. History is passed to the LLM as formatted text in the system prompt, not as message history.
- **Document format convention**: Course docs in `docs/` follow a structured text format with `Course Title:`, `Course Link:`, `Course Instructor:` headers, then `Lesson N: <title>` markers separating content.

### Backend Module Responsibilities
- `rag_system.py` — Orchestrator that wires all components; entry point is `query()` and `add_course_folder()`
- `ai_generator.py` — Anthropic Claude API client with multi-round tool-use loop; owns the system prompt and `MAX_TOOL_ROUNDS`
- `ollama_generator.py` — Local Ollama client with the same multi-round loop; handles small-model quirks (JSON text tool calls, schema-as-value arguments)
- `vector_store.py` — ChromaDB wrapper; handles course name resolution, filtering, and search
- `search_tools.py` — Tool abstraction (`Tool` ABC, `CourseSearchTool`, `CourseOutlineTool`, `ToolManager`); formats search results and tracks sources across multi-round calls
- `document_processor.py` — Parses course files, extracts metadata, chunks text with sentence-aware splitting
- `session_manager.py` — In-memory conversation history per session
- `config.py` — Dataclass with all tunable parameters (chunk size, model, etc.)
- `models.py` — Pydantic models: `Course`, `Lesson`, `CourseChunk`

### Frontend
Vanilla HTML/CSS/JS in `frontend/`. Uses `marked.js` for markdown rendering. Served as static files by FastAPI's `StaticFiles` mount at `/`. All API calls go to relative `/api/*` paths.

### Startup Behavior
On app startup (`app.py:startup_event`), all `.txt`/`.pdf`/`.docx` files in `docs/` are processed and indexed into ChromaDB. Courses already in the DB (matched by title) are skipped.
