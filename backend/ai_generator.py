import logging
import anthropic
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **search_course_content**: Use for questions about specific course content or detailed educational materials.
- **get_course_outline**: Use when the user asks about a course's structure, outline, syllabus, lesson list, what topics a course covers, or how many lessons it has. Format the course link as a markdown hyperlink using the course title as the display text. List every lesson with its number and title.
- You may make up to 2 sequential tool calls per query when needed. Use a single tool call for straightforward lookups.
- **When to chain 2 tool calls**: If the user asks about a specific lesson's topic and then wants to compare or find similar content elsewhere, first search/get the specific content, then make a second search WITHOUT a course_name filter to find results across all courses.
- Synthesize tool results into accurate, fact-based responses
- Always use the full course title (not abbreviations) when referencing courses in your response
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course outline/structure questions**: Use get_course_outline, then present the course title as a markdown hyperlink (e.g. [Course Title](url)) and every lesson (number and title)
- **Course content questions**: Use search_course_content, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls across up to MAX_TOOL_ROUNDS rounds.

        Each round: append assistant tool-use message, execute tools, append results,
        call API. Tools are included in all rounds except the last, forcing Claude
        to synthesize a final answer.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters (includes messages, system, tools)
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        messages = base_params["messages"].copy()
        current_response = initial_response

        for round_num in range(self.MAX_TOOL_ROUNDS):
            # Append assistant's tool-use response
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls, collect results
            tool_results = []
            for block in current_response.content:
                if block.type == "tool_use":
                    logger.info(f"Tool round {round_num + 1}/{self.MAX_TOOL_ROUNDS}: "
                                f"calling {block.name}({block.input})")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_manager.execute_tool(block.name, **block.input)
                    })

            # Append tool results
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Build next API call — include tools only if another round is allowed
            is_last_round = (round_num == self.MAX_TOOL_ROUNDS - 1)
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"]
            }
            if not is_last_round:
                api_params["tools"] = base_params["tools"]
                api_params["tool_choice"] = {"type": "auto"}

            # Call API
            current_response = self.client.messages.create(**api_params)

            # If no more tool calls, break early
            if current_response.stop_reason != "tool_use":
                break

        logger.info(f"Tool execution completed after {round_num + 1} round(s)")
        return current_response.content[0].text