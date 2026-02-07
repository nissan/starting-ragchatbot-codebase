import json
import re
import ollama
from typing import List, Optional, Dict, Any


class OllamaGenerator:
    """Handles interactions with a local Ollama instance for generating responses"""

    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **search_course_content**: Use for questions about specific course content or detailed educational materials.
- **get_course_outline**: Use when the user asks about a course's structure, outline, syllabus, lesson list, what topics a course covers, or how many lessons it has. Format the course link as a markdown hyperlink using the course title as the display text. List every lesson with its number and title.
- **One tool call per query maximum**
- Synthesize tool results into accurate, fact-based responses
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

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.client = ollama.Client(host=base_url)
        print(f"Using Ollama with model: {model}")

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use (Anthropic format, will be converted)
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        # Convert Anthropic tool format to Ollama/OpenAI format
        ollama_tools = None
        if tools:
            ollama_tools = [self._convert_tool(t) for t in tools]

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=ollama_tools,
                options={"temperature": 0},
            )
        except ConnectionError:
            return "Error: Could not connect to Ollama. Make sure Ollama is running (`ollama serve`)."
        except ollama.ResponseError as e:
            if "not found" in str(e).lower():
                return f"Error: Model '{self.model}' not found. Pull it with `ollama pull {self.model}`."
            return f"Ollama error: {e}"

        # Handle tool calls if present (via proper tool_calls field)
        tool_calls = response.message.tool_calls
        content = response.message.content

        if tool_calls and tool_manager:
            return self._handle_tool_execution(response, messages, query, tool_manager)

        # Small models sometimes dump the tool call as JSON text instead of
        # using the tool_calls mechanism. Detect and handle that case.
        if tool_manager and content:
            parsed = self._try_parse_text_tool_call(content)
            if parsed:
                return self._execute_parsed_tool_call(parsed, messages, query, tool_manager)

        return content or ""

    def _handle_tool_execution(self, initial_response, messages: List[Dict],
                               fallback_query: str, tool_manager) -> str:
        """Handle tool calls from Ollama and get a synthesis response."""
        messages = messages.copy()
        messages.append({"role": "assistant", "content": "Let me search for that information."})

        # Execute all tool calls and collect results
        all_results = []
        for tool_call in initial_response.message.tool_calls:
            args = self._fix_tool_arguments(
                tool_call.function.arguments, fallback_query,
                tool_name=tool_call.function.name,
            )
            tool_result = tool_manager.execute_tool(
                tool_call.function.name,
                **args,
            )
            all_results.append(tool_result)

        combined = "\n\n".join(all_results)
        return self._synthesize(messages, combined)

    def _synthesize(self, messages: List[Dict], tool_result: str) -> str:
        """Send tool results as a user message and get a synthesis response.

        Small models (e.g. llama3.2:3b) often ignore {"role": "tool"} messages.
        Sending the results as a user message works reliably.
        """
        messages = messages.copy()
        messages.append({
            "role": "user",
            "content": (
                "Here are the search results from the course database:\n\n"
                f"{tool_result}\n\n"
                "Please answer the question based on these results."
            ),
        })

        try:
            final_response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0},
            )
        except (ConnectionError, ollama.ResponseError) as e:
            return f"Ollama error during synthesis: {e}"

        result = final_response.message.content or ""
        # Guard against synthesis returning another tool call JSON
        if result.strip().startswith("{") and self._try_parse_text_tool_call(result):
            return tool_result
        return result

    @staticmethod
    def _is_stringified_schema(value: str) -> bool:
        """Detect if a string value is actually a stringified JSON Schema definition.

        Small models sometimes pass the schema definition itself as a string, e.g.:
            "{'type': 'string', 'description': 'Course title (partial matches work)'}"
        These contain no actual user value and should be discarded.
        """
        return ("'type':" in value or '"type":' in value) and (
            "'description':" in value or '"description":' in value
        )

    @staticmethod
    def _extract_value_from_dict(d: Dict) -> Optional[str]:
        """Extract the actual value from a schema-like dict that a small model
        produces instead of a plain argument value.

        Handles variations like:
            {"type": "string", "description": "outline"}  → "outline"
            {"type": "Advanced Retrieval for AI"}          → "Advanced Retrieval for AI"
            {"description": "MCP"}                         → "MCP"
        """
        _schema_types = {"string", "integer", "number", "boolean", "object", "array"}

        # If "description" exists and looks like the actual value, use it
        desc = d.get("description")
        type_val = d.get("type")

        if desc and str(desc).strip().lower() not in _schema_types:
            return str(desc)
        # If "type" holds the actual value (not a JSON Schema type), use it
        if type_val and str(type_val).strip().lower() not in _schema_types:
            return str(type_val)
        # If description is a schema type but type is also, nothing useful
        if desc:
            return str(desc)
        return None

    @staticmethod
    def _fix_tool_arguments(args: Dict[str, Any], fallback_query: str = "",
                            tool_name: str = "") -> Dict[str, Any]:
        """Fix tool arguments when the model passes schema objects instead of values.

        Small models sometimes produce arguments like:
            {"query": {"type": "string", "description": "outline"}}
        instead of:
            {"query": "outline"}
        This extracts the actual value, drops nonsensical placeholder values,
        and coerces types.
        """
        _skip_values = {"none specified", "none", "n/a", "not specified", "null"}
        fixed = {}
        for key, value in args.items():
            if isinstance(value, dict):
                extracted = OllamaGenerator._extract_value_from_dict(value)
                if extracted is None or extracted.strip().lower() in _skip_values:
                    continue
                fixed[key] = extracted
            elif isinstance(value, str) and OllamaGenerator._is_stringified_schema(value):
                # Model pasted the schema definition as a string — discard it
                continue
            else:
                if isinstance(value, str) and value.strip().lower() in _skip_values:
                    continue
                fixed[key] = value

        # Coerce lesson_number to int if present
        if "lesson_number" in fixed:
            try:
                fixed["lesson_number"] = int(fixed["lesson_number"])
            except (ValueError, TypeError):
                del fixed["lesson_number"]

        # If course_name is missing, try to extract it from the fallback query
        # Look for quoted course names like "MCP: Build Rich-Context AI Apps"
        if not fixed.get("course_name") and fallback_query:
            match = re.search(r'["\u201c]([^"\u201d]+)["\u201d]', fallback_query)
            if match:
                fixed["course_name"] = match.group(1)

        # Only inject query fallback for tools that require it
        if tool_name != "get_course_outline":
            if not fixed.get("query") and fallback_query:
                fixed["query"] = fallback_query

        return fixed

    def _try_parse_text_tool_call(self, content: str) -> Optional[Dict]:
        """Try to parse a tool call that the model dumped as JSON text.

        Small models sometimes produce malformed JSON with missing closing braces.
        This method attempts to fix unbalanced braces before giving up.
        """
        content = content.strip()
        if not content.startswith("{"):
            return None
        # Try parsing as-is first
        data = self._try_json_loads(content)
        if data is None:
            # Try fixing unbalanced braces
            fixed = content
            while fixed.count("{") > fixed.count("}"):
                fixed += "}"
            data = self._try_json_loads(fixed)
        if data and "name" in data and "parameters" in data:
            return data
        return None

    @staticmethod
    def _try_json_loads(text: str) -> Optional[Dict]:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    def _execute_parsed_tool_call(self, parsed: Dict, messages: List[Dict],
                                  fallback_query: str, tool_manager) -> str:
        """Execute a tool call that was parsed from text content and synthesize."""
        messages = messages.copy()
        messages.append({"role": "assistant", "content": "Let me search for that information."})

        args = self._fix_tool_arguments(
            parsed.get("parameters", {}), fallback_query,
            tool_name=parsed["name"],
        )
        tool_result = tool_manager.execute_tool(parsed["name"], **args)

        return self._synthesize(messages, tool_result)

    @staticmethod
    def _convert_tool(anthropic_tool: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an Anthropic tool definition to Ollama/OpenAI format."""
        return {
            "type": "function",
            "function": {
                "name": anthropic_tool["name"],
                "description": anthropic_tool.get("description", ""),
                "parameters": anthropic_tool.get("input_schema", {}),
            },
        }
