"""Tests for ai_generator.py — AIGenerator response generation and tool-use flow."""

from unittest.mock import patch, MagicMock

import pytest

from ai_generator import AIGenerator
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage


def make_anthropic_response(
    text=None,
    tool_use=None,
    stop_reason="end_turn",
    model="claude-sonnet-4-20250514",
):
    """Build a realistic anthropic.types.Message object."""
    content = []
    if text is not None:
        content.append(TextBlock(type="text", text=text))
    if tool_use is not None:
        content.append(
            ToolUseBlock(
                type="tool_use",
                id=tool_use["id"],
                name=tool_use["name"],
                input=tool_use["input"],
            )
        )
    return Message(
        id="msg_test_123",
        type="message",
        role="assistant",
        model=model,
        content=content,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=100, output_tokens=50),
    )


@pytest.fixture
def mock_anthropic_client():
    """Patch anthropic.Anthropic so AIGenerator.__init__ gets a mock client."""
    with patch("ai_generator.anthropic.Anthropic") as MockCls:
        client = MagicMock()
        MockCls.return_value = client
        yield client


@pytest.fixture
def generator(mock_anthropic_client):
    """AIGenerator wired to the mock Anthropic client."""
    return AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")


# ===================================================================
# Direct response (no tool use)
# ===================================================================

class TestDirectResponse:

    def test_direct_response(self, generator, mock_anthropic_client):
        """When stop_reason='end_turn', return content[0].text with a single API call."""
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            text="Paris is the capital of France.", stop_reason="end_turn"
        )

        result = generator.generate_response(query="What is the capital of France?")

        assert result == "Paris is the capital of France."
        assert mock_anthropic_client.messages.create.call_count == 1


# ===================================================================
# System prompt construction
# ===================================================================

class TestSystemPrompt:

    def test_conversation_history_in_system_prompt(self, generator, mock_anthropic_client):
        """System prompt includes conversation history when provided."""
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            text="Follow-up answer.", stop_reason="end_turn"
        )
        history = "User: Hi\nAssistant: Hello!"

        generator.generate_response(query="Tell me more", conversation_history=history)

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert "Previous conversation:" in call_kwargs["system"]
        assert history in call_kwargs["system"]

    def test_no_history_system_prompt(self, generator, mock_anthropic_client):
        """System prompt is just SYSTEM_PROMPT when no history provided."""
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            text="Answer.", stop_reason="end_turn"
        )

        generator.generate_response(query="Question")

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == AIGenerator.SYSTEM_PROMPT


# ===================================================================
# Tool passing
# ===================================================================

class TestToolPassing:

    def test_tools_passed_to_api(self, generator, mock_anthropic_client):
        """API call includes tools and tool_choice when tools are provided."""
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            text="Direct answer.", stop_reason="end_turn"
        )
        tools = [{"name": "search_course_content", "input_schema": {}}]

        generator.generate_response(query="Search something", tools=tools)

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == {"type": "auto"}


# ===================================================================
# Tool-use two-pass flow
# ===================================================================

class TestToolUseTwoPass:

    def test_tool_use_two_pass_flow(self, generator, mock_anthropic_client):
        """When stop_reason='tool_use': executes tool, makes 2nd call, returns synthesized text."""
        # First API call returns a tool_use response
        first_response = make_anthropic_response(
            text="Let me search for that.",
            tool_use={
                "id": "toolu_abc123",
                "name": "search_course_content",
                "input": {"query": "RAG basics"},
            },
            stop_reason="tool_use",
        )
        # Second API call returns the final synthesized response
        second_response = make_anthropic_response(
            text="RAG combines retrieval with generation.", stop_reason="end_turn"
        )
        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        # Mock tool manager
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Found: RAG is a technique..."

        tools = [{"name": "search_course_content", "input_schema": {}}]
        result = generator.generate_response(
            query="What is RAG?", tools=tools, tool_manager=tool_manager
        )

        # Should return the synthesized response
        assert result == "RAG combines retrieval with generation."

        # Should have made exactly 2 API calls
        assert mock_anthropic_client.messages.create.call_count == 2

        # Tool manager should have been called with correct args
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="RAG basics"
        )

        # Validate the second API call's message structure
        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1].kwargs
        messages = second_call_kwargs["messages"]

        # messages: [user, assistant (tool_use blocks), user (tool_result)]
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == first_response.content
        assert messages[2]["role"] == "user"
        # tool_result content
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "toolu_abc123"
        assert messages[2]["content"][0]["content"] == "Found: RAG is a technique..."

        # Second call should NOT include tools
        assert "tools" not in second_call_kwargs


# ===================================================================
# Edge cases: tool_manager is None
# ===================================================================

class TestToolManagerNone:

    def test_tool_use_without_tool_manager(self, generator, mock_anthropic_client):
        """When tool_manager=None and response has TextBlock first, falls through to text."""
        response = make_anthropic_response(
            text="I would search but no tool manager.",
            tool_use={
                "id": "toolu_xyz",
                "name": "search_course_content",
                "input": {"query": "test"},
            },
            stop_reason="tool_use",
        )
        mock_anthropic_client.messages.create.return_value = response

        # tool_manager=None → skips tool execution, falls through to content[0].text
        result = generator.generate_response(query="Search something")

        assert result == "I would search but no tool manager."
        assert mock_anthropic_client.messages.create.call_count == 1

    def test_tool_use_only_toolblock_no_manager(self, generator, mock_anthropic_client):
        """Latent bug: when tool_manager=None and content[0] is ToolUseBlock, raises AttributeError."""
        # Build response with ONLY a ToolUseBlock (no TextBlock)
        response = make_anthropic_response(
            tool_use={
                "id": "toolu_xyz",
                "name": "search_course_content",
                "input": {"query": "test"},
            },
            stop_reason="tool_use",
        )
        mock_anthropic_client.messages.create.return_value = response

        # content[0] is ToolUseBlock which has no .text attribute → AttributeError
        with pytest.raises(AttributeError):
            generator.generate_response(query="Search something")
