"""Tests for search_tools.py — CourseSearchTool, CourseOutlineTool, ToolManager."""

from unittest.mock import MagicMock

from vector_store import SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager

# ===================================================================
# CourseSearchTool.execute()
# ===================================================================


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute()"""

    def test_execute_successful_search(self, course_search_tool, mock_vector_store):
        result = course_search_tool.execute(query="RAG basics")

        mock_vector_store.search.assert_called_once_with(
            query="RAG basics", course_name=None, lesson_number=None
        )
        assert "Lesson 1 covers the basics" in result
        assert "Lesson 2 dives into" in result

    def test_execute_with_course_filter(self, course_search_tool, mock_vector_store):
        course_search_tool.execute(query="embeddings", course_name="Intro to RAG")

        mock_vector_store.search.assert_called_once_with(
            query="embeddings", course_name="Intro to RAG", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, course_search_tool, mock_vector_store):
        course_search_tool.execute(query="embeddings", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="embeddings", course_name=None, lesson_number=2
        )

    def test_execute_with_both_filters(self, course_search_tool, mock_vector_store):
        course_search_tool.execute(
            query="embeddings", course_name="Intro to RAG", lesson_number=2
        )

        mock_vector_store.search.assert_called_once_with(
            query="embeddings", course_name="Intro to RAG", lesson_number=2
        )

    def test_execute_empty_results(
        self, course_search_tool, mock_vector_store, empty_search_results
    ):
        mock_vector_store.search.return_value = empty_search_results

        result = course_search_tool.execute(query="nonexistent topic")
        assert result == "No relevant content found."

    def test_execute_empty_with_course_filter(
        self, course_search_tool, mock_vector_store, empty_search_results
    ):
        mock_vector_store.search.return_value = empty_search_results

        result = course_search_tool.execute(query="x", course_name="Advanced ML")
        assert result == "No relevant content found in course 'Advanced ML'."

    def test_execute_empty_with_lesson_filter(
        self, course_search_tool, mock_vector_store, empty_search_results
    ):
        mock_vector_store.search.return_value = empty_search_results

        result = course_search_tool.execute(query="x", lesson_number=5)
        assert result == "No relevant content found in lesson 5."

    def test_execute_empty_with_both_filters(
        self, course_search_tool, mock_vector_store, empty_search_results
    ):
        mock_vector_store.search.return_value = empty_search_results

        result = course_search_tool.execute(
            query="x", course_name="Advanced ML", lesson_number=5
        )
        assert (
            result == "No relevant content found in course 'Advanced ML' in lesson 5."
        )

    def test_execute_error(
        self, course_search_tool, mock_vector_store, error_search_results
    ):
        mock_vector_store.search.return_value = error_search_results

        result = course_search_tool.execute(query="anything")
        assert result == "Search error: connection refused"

    def test_format_results_populates_sources(
        self, course_search_tool, mock_vector_store
    ):
        """last_sources should contain deduplicated {text, link} entries."""
        # Use results where both docs share the same course — dedup should reduce to 2
        course_search_tool.execute(query="RAG basics")

        sources = course_search_tool.last_sources
        assert len(sources) == 2  # lesson 1 and lesson 2 have different text
        assert sources[0]["text"] == "Intro to RAG - Lesson 1"
        assert sources[0]["link"] == "https://example.com/lesson/1"

    def test_format_results_lesson_link_fallback(
        self, course_search_tool, mock_vector_store
    ):
        """Falls back to get_course_link() when get_lesson_link() returns None."""
        mock_vector_store.get_lesson_link.return_value = None

        course_search_tool.execute(query="RAG basics")

        sources = course_search_tool.last_sources
        # Should fall back to course link
        assert sources[0]["link"] == "https://example.com/course/intro-rag"

    def test_get_tool_definition(self, course_search_tool):
        defn = course_search_tool.get_tool_definition()

        assert defn["name"] == "search_course_content"
        assert "input_schema" in defn
        assert "query" in defn["input_schema"]["properties"]
        assert defn["input_schema"]["required"] == ["query"]


# ===================================================================
# CourseOutlineTool.execute()
# ===================================================================


class TestCourseOutlineToolExecute:

    def test_outline_valid_course(self, course_outline_tool, mock_vector_store):
        result = course_outline_tool.execute(course_name="Intro to RAG")

        mock_vector_store.get_course_outline.assert_called_once_with("Intro to RAG")
        assert "Course: Intro to RAG" in result
        assert "Link: https://example.com/course/intro-rag" in result
        assert "Total lessons: 3" in result
        assert "Lesson 1: What is RAG?" in result
        assert "Lesson 3: Putting It Together" in result
        # Should populate last_sources
        assert course_outline_tool.last_sources == [
            {"text": "Intro to RAG", "link": "https://example.com/course/intro-rag"}
        ]

    def test_outline_invalid_course(self, course_outline_tool, mock_vector_store):
        mock_vector_store.get_course_outline.return_value = None

        result = course_outline_tool.execute(course_name="Nonexistent Course")
        assert result == "No course found matching 'Nonexistent Course'"

    def test_outline_tool_definition(self, course_outline_tool):
        defn = course_outline_tool.get_tool_definition()

        assert defn["name"] == "get_course_outline"
        assert "input_schema" in defn
        assert "course_name" in defn["input_schema"]["properties"]
        assert defn["input_schema"]["required"] == ["course_name"]


# ===================================================================
# ToolManager
# ===================================================================


class TestToolManager:

    def test_register_and_get_definitions(self, tool_manager_with_tools):
        defs = tool_manager_with_tools.get_tool_definitions()
        assert len(defs) == 2
        names = {d["name"] for d in defs}
        assert names == {"search_course_content", "get_course_outline"}

    def test_execute_unregistered_tool(self, tool_manager_with_tools):
        result = tool_manager_with_tools.execute_tool("nonexistent_tool")
        assert result == "Tool 'nonexistent_tool' not found"

    def test_source_tracking_and_reset(self, tool_manager_with_tools):
        # Execute the search tool to populate sources
        tool_manager_with_tools.execute_tool(
            "search_course_content", query="RAG basics"
        )

        sources = tool_manager_with_tools.get_last_sources()
        assert len(sources) > 0

        tool_manager_with_tools.reset_sources()
        assert tool_manager_with_tools.get_last_sources() == []

    def test_multi_tool_source_accumulation(self, tool_manager_with_tools):
        """get_last_sources returns deduplicated sources from ALL tools after multi-round calls."""
        # Round 1: outline tool populates its sources
        tool_manager_with_tools.execute_tool(
            "get_course_outline", course_name="Intro to RAG"
        )
        # Round 2: search tool populates its sources
        tool_manager_with_tools.execute_tool(
            "search_course_content", query="RAG basics"
        )

        sources = tool_manager_with_tools.get_last_sources()
        source_texts = [s["text"] for s in sources]

        # Should have sources from BOTH tools
        assert "Intro to RAG" in source_texts  # from outline tool
        assert "Intro to RAG - Lesson 1" in source_texts  # from search tool
        assert "Intro to RAG - Lesson 2" in source_texts  # from search tool

    def test_multi_tool_source_deduplication(self, tool_manager_with_tools):
        """Duplicate source texts across tools are deduplicated."""
        # Both tools produce sources with "Intro to RAG" text
        tool_manager_with_tools.execute_tool(
            "get_course_outline", course_name="Intro to RAG"
        )
        tool_manager_with_tools.execute_tool(
            "search_course_content", query="RAG basics"
        )

        sources = tool_manager_with_tools.get_last_sources()
        source_texts = [s["text"] for s in sources]

        # "Intro to RAG" appears in both outline and search sources,
        # but should only appear once in combined output
        assert source_texts.count("Intro to RAG") == 1

    def test_multi_tool_reset_clears_all(self, tool_manager_with_tools):
        """reset_sources clears sources from all tools."""
        tool_manager_with_tools.execute_tool(
            "get_course_outline", course_name="Intro to RAG"
        )
        tool_manager_with_tools.execute_tool(
            "search_course_content", query="RAG basics"
        )

        tool_manager_with_tools.reset_sources()

        assert tool_manager_with_tools.get_last_sources() == []
