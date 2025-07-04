from openai import AzureOpenAI
from dotenv import load_dotenv
import json
import os
from internal_state import InternalState
from tools.tool_dispatch import tool_dispatch
from helper_methods import *
from tools.openai_client import client

# Load environment variables from .env
load_dotenv()

# Model and version
MODEL_4o = os.getenv("MODEL_4o")
AZURE_OPEN_VERSION_4o = os.getenv("AZURE_OPEN_VERSION_4o")

# Read Azure credentials
AZURE_OPENAI_API_KEY = os.getenv("CLASS_OPEN_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("SUBSCRIPTION_OPENAI_ENDPOINT")

DEBUG = os.getenv("DEBUG", "0").lower() == "1"


def _check_azure_connection() -> None:
    """Ping Azure OpenAI with a minimal request to verify credentials/connectivity."""
    try:
        test_msg = [{"role": "system", "content": "ping"}]
        client.chat.completions.create(
            model=MODEL_4o,
            messages=test_msg,
            max_tokens=1,
        )
        print("✅ Azure OpenAI connectivity verified.")
    except Exception as conn_error:
        print("🚨 Azure OpenAI connectivity check failed:", conn_error)
        print(
            "Please verify your AZURE_OPENAI_ENDPOINT, API key, and network connectivity."
        )
        # Exit early to avoid entering the agent loop without connectivity
        import sys

        sys.exit(1)


# Run the connectivity check once at startup
_check_azure_connection()

# Agent Internal State
state = InternalState()


# Agent Tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "online_search",
            "description": "Search the internet for information about a specific entity's attribute. Returns the attribute value found through web search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "an_entity": {
                        "type": "string",
                        "description": "The entity to search for (e.g., 'Abraham Lincoln', 'Apple Inc.', 'Mount Everest')",
                    },
                    "an_attribute": {
                        "type": "string",
                        "description": "The specific attribute to find about the entity (e.g., 'birthplace', 'CEO', 'height', 'founding date')",
                    },
                    "max_retries": {
                        "type": "integer",
                        "description": "Maximum number of retry attempts for processing (optional, defaults to 3)",
                        "default": 3,
                    },
                },
                "required": ["an_entity", "an_attribute"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes the given content to a file on disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_content": {
                        "type": "string",
                        "description": "Text or data to be written into the file",
                    },
                    "fn": {
                        "type": "string",
                        "description": "The file name or path where the content should be written",
                    },
                },
                "required": ["file_content", "fn"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_entities_from_file",
            "description": "Extracts entities of a given type (like city, person) from a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The path to the text file.",
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "Type of entity to extract (e.g., city, person, organization).",
                    },
                },
                "required": ["file_name", "entity_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gen_plot_prog",
            "description": "Generates a Python plotting program based on a natural language request. The code will read from a CSV file and save the resulting plot to a .png file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_request": {
                        "type": "string",
                        "description": "A description of the plot to generate, like 'plot average grade by year'",
                    },
                    "input_file": {
                        "type": "string",
                        "description": "Path to the input CSV file",
                    },
                    "columns": {
                        "type": "string",
                        "description": "Comma-separated list of columns available in the CSV",
                    },
                    "gen_output_program_fn": {
                        "type": "string",
                        "description": "Filename where the generated Python script should be written",
                    },
                    "output_png": {
                        "type": "string",
                        "description": "Filename for the resulting plot image",
                    },
                    "knowledge_base": {
                        "type": "object",
                        "description": "Extracted data from the file resources and knowledge base to help the tool generate the plot.",
                    },
                },
                "required": [
                    "plot_request",
                    "input_file",
                    "columns",
                    "gen_output_program_fn",
                    "output_png",
                    "knowledge_base",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "debug_and_regenerate_prog",
            "description": "Debugs and regenerates a Python program based on an error message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "program_fn": {
                        "type": "string",
                        "description": "The path to the Python program file to debug.",
                    },
                    "errors": {
                        "type": "string",
                        "description": "The error message to of the program execution to deubg.",
                    },
                    "plot_request": {
                        "type": "string",
                        "description": "The plot request that the program is meant to provide a plot for.",
                    },
                    "data_file": {
                        "type": "string",
                        "description": "The data file that the program uses to generate the plot.",
                    },
                    "columns": {
                        "type": "string",
                        "description": "The columns of the original data file that the program uses to generate the plot. You can use the file description to find the columns.",
                    },
                },
                "required": [
                    "program_fn",
                    "errors",
                    "plot_request",
                    "data_file",
                    "columns",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_Python_prog",
            "description": "Executes a Python program and returns the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "program_fn": {
                        "type": "string",
                        "description": "The path to the Python program file to execute.",
                    },
                },
                "required": ["program_fn"],
            },
        },
    },
]


TOOL_LLM_COST = {
    "debug_and_regenerate_prog": 1,
    "extract_entities_from_file": 1,
    "gen_plot_prog": 1,
    "online_search": 1,
}


# Tool execution function
def execute_tool(tool_name: str, **kwargs) -> str:
    """Execute specified tool and charge LLM-call budget if the tool uses GPT."""
    # Check LLM cost before execution
    extra_llm_cost = TOOL_LLM_COST.get(tool_name, 0)
    if state.llm_calls + extra_llm_cost > state.max_llm_calls:
        return json.dumps(
            {
                "error": "LLM call budget exceeded by tool execution",
                "tool": tool_name,
                "required_llm_calls": extra_llm_cost,
                "remaining_llm_calls": state.max_llm_calls - state.llm_calls,
            }
        )

    try:
        tool_func = tool_dispatch.get(tool_name)
        if not tool_func:
            return json.dumps(
                {
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": list(tool_dispatch.keys()),
                }
            )

        return tool_func(**kwargs)

    except Exception as e:
        return json.dumps(
            {
                "error": f"Critical error in execute_tool: {str(e)}",
                "tool": tool_name,
                "parameters": kwargs,
            }
        )


def analyze_input_file(file_name: str):
    with open(file_name, "r") as f:
        content = json.load(f)  # Parse JSON directly from file object
    query_file_name = content["query_name"]
    state.set_query_data(query_file_name)
    state.add_reflection(f"Extracted query file data: {query_file_name}")
    for obj in content["file_resources"]:
        file_name = obj["file_name"]
        description = obj["description"]
        state.file_resources[file_name] = description
        state.add_reflection(
            f"Analyzed file {file_name} with description: {description}",
        )
    file_reflection = (
        f"Analyzed {len(content['file_resources'])} files"
        if len(content["file_resources"]) > 0
        else "No resource file availlable for usage."
    )
    state.add_reflection(file_reflection)


initial_system_prompt = """You are a ReAct-style agent. You solve queries by interleaving three steps:

1. Thought — Reflect on the current task and decide the next action.
2. Action — Call one of the tools if needed, or give a final answer.
3. Observation — The result from the tool or a confirmation.

You must continue this loop until the task is completed or an error prevents further progress.

You have access to the following tools:
- extract_entities_from_file(file_name: str, entity_type: str)
- Internet_search_attribute(entity: str, attribute: str)
- gen_plot_prog(plot_request: str, input_file: str, columns: str, gen_output_program_fn: str, output_png: str, knowledge_base: dict)
- execute_Python_prog(program_fn: str)
- debug_and_regenerate_prog(program_fn: str, errors: str)
- write_file(file_content: str, fn: str)

You can solve the query based on the given tools, and the files available to you.
If you experience an error with a tool, debug and analyze the error, if you can fix it and you believe you should call the tool again, do so. 
otherwise, you should not call the tool again, and you should find a different set of actions to solve the task.

In case you encounter an error, and after debugging and analyzing alternatives steps with given tools - you conclude there is no action you can take to solve the task,
you should respond with "Final response is = I cannot solve the task with the tools available to me."

If you need to perform an action, **use one of these tools directly by calling it**.

If you believe you have fully answered the user's query, STOP and respond like this:
\"Final response is = <the final answer to the user query content>\"

Do NOT take any more actions after you have given the Final Response.
Do not call tools after providing the Final Response.

"""

state.messages.append({"role": "system", "content": initial_system_prompt})

analyze_input_file("input.json")

initial_user_prompt = f"""
Your task is to answer the query.
Query: {state.query_text}
Given a list of file name and their description available for the task:
{state.file_resources}
If the list is empty, it means no file is available for usage.

File usage is it at your disposal. If you feel you can answer the query without using the files, do so.
If you feel you need to use the files, use the tools provided to you analyze the files.
"""

state.messages.append({"role": "user", "content": initial_user_prompt})


def debug_conversation_state(messages, context=""):
    """Debug utility to print current conversation state"""
    print(f"\n🔍 CONVERSATION DEBUG {context}")
    print(f"Total messages: {len(messages)}")

    tool_calls_found = []
    tool_responses_found = []

    for i, msg in enumerate(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"  [{i}] ASSISTANT with {len(msg.tool_calls)} tool calls:")
            for tc in msg.tool_calls:
                tool_calls_found.append(tc.id)
                print(f"    - {tc.id}: {tc.function.name}")
        elif isinstance(msg, dict):
            if msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                tool_responses_found.append(tc_id)
                print(
                    f"  [{i}] TOOL RESPONSE: {tc_id} ({msg.get('name', 'unknown')})"
                )
            else:
                print(
                    f"  [{i}] {msg.get('role', 'unknown').upper()}: {len(msg.get('content', ''))} chars"
                )
        else:
            role = getattr(msg, "role", "unknown")
            content_len = len(getattr(msg, "content", "") or "")
            print(f"  [{i}] {role.upper()}: {content_len} chars")

    # Check for unmatched tool calls
    unmatched = [
        tc_id
        for tc_id in tool_calls_found
        if tc_id not in tool_responses_found
    ]
    if unmatched:
        print(f"❌ UNMATCHED TOOL CALLS: {unmatched}")
        return False
    else:
        print("✅ All tool calls have responses")
        return True


# ============================================================================
# TOOL EXECUTION HELPER FUNCTIONS
# ============================================================================


def _execute_single_tool_call(
    state: InternalState, tool_call, tool_call_index: int, total_calls: int
) -> bool:
    """
    Execute a single tool call and handle all associated logic.

    Args:
        state: The internal state object
        tool_call: The tool call object from OpenAI
        tool_call_index: Index of current tool call (0-based)
        total_calls: Total number of tool calls being processed

    Returns:
        bool: True if tool response was successfully added, False otherwise
    """
    tool_response_added = False
    print(
        f"Processing tool call {tool_call_index + 1}/{total_calls}: {tool_call.id}"
    )

    try:
        # Extract function name and arguments
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        print(f"Executing tool: {func_name} with args: {args}")

        # Handle redundant calls
        if state.already_called_tool(func_name, args):
            prev = get_previous_tool_result(state, func_name, args)

            tool_response_added, reflection_msg, observation_msg = (
                add_redundant_tool_response(
                    state, tool_call, func_name, args, prev_result=prev
                )
            )

        # Register and execute tool
        else:
            tool_response_added, reflection_msg, observation_msg = (
                _handle_tool_execution(state, tool_call, func_name, args)
            )

    except Exception as critical_error:
        print(
            f"Critical error processing tool call {tool_call.id}: {critical_error}"
        )

        if not tool_response_added:
            tool_response_added, reflection_msg, observation_msg = (
                add_emergency_tool_response(state, tool_call, critical_error)
            )

        # Log error but continue processing
        state.log_error(
            "Tool processing error",
            str(critical_error),
            {
                "tool_call_id": tool_call.id,
                "function_name": getattr(
                    tool_call.function, "name", "unknown"
                ),
            },
        )

    return tool_response_added, reflection_msg, observation_msg


def _handle_tool_execution(
    state: InternalState, tool_call, func_name: str, args: dict
) -> bool:
    """
    Handle the actual tool execution and result processing.

    Args:
        state: The internal state object
        tool_call: The tool call object
        func_name: Name of the function to execute
        args: Arguments for the function

    Returns:
        bool: True if tool response was successfully added
    """
    # Update state tracking
    try:
        state.register_tool_call(func_name, args)
    except Exception as state_error:
        print(f"Warning: State tracking error: {state_error}")

    # Execute the tool
    try:
        result = execute_tool(func_name, **args)

        # Ensure result is a string
        if not isinstance(result, str):
            result = json.dumps(
                {
                    "error": "Tool returned non-string result",
                    "result": str(result),
                }
            )

        print(f"Tool {func_name} execution completed.\nResults: {result}")

    except Exception as tool_error:
        print(f"Tool execution failed: {tool_error}")
        result = json.dumps(
            {
                "error": f"Tool execution failed: {str(tool_error)}",
                "tool": func_name,
                "args": args,
            }
        )

    # Process tool results
    returned_reflection, returned_observation = _process_tool_results(
        state, tool_call, func_name, args, result
    )

    return True, returned_reflection, returned_observation


def _process_tool_results(
    state: InternalState, tool_call, func_name: str, args: dict, result: str
):
    """
    Process tool execution results and update state accordingly.

    Args:
        state: The internal state object
        tool_call: The tool call object
        func_name: Name of the executed function
        args: Arguments used for the function
        result: Result returned by the tool
    """
    # Add tool response to conversation
    add_tool_response(state, tool_call, func_name, result)

    # Update state tracking
    try:
        state.exit_tool(func_name)
    except Exception as state_error:
        print(f"Warning: State exit error: {state_error}")

    # Add detailed reflection with actual tool results
    returned_reflection = _add_detailed_tool_reflection(
        state, func_name, args, result
    )

    # Update knowledge base on successful execution
    try:
        update_knowledge_base(state, func_name, args, result)
    except Exception as kb_error:
        print(f"Warning: Knowledge base update failed: {kb_error}")

    if func_name == "debug_and_regenerate_prog":
        state.tool_history = [
            entry
            for entry in state.tool_history
            if entry[0] != "execute_Python_prog"
        ]

    # Add observation message for ReAct framework
    # state.messages.append(
    #     {
    #         "role": "assistant",
    #         "content": f"Observation: The tool `{func_name}` returned:\n{result}",
    #     }
    # )
    observation_msg = (
        f"Observation: The tool `{func_name}` returned:\n{result}"
    )

    return returned_reflection, observation_msg


def _add_detailed_tool_reflection(
    state: InternalState, func_name: str, args: dict, result: str
):
    """
    Add a detailed reflection that includes the actual tool execution result.

    Args:
        state: The internal state object
        func_name: Name of the executed function
        args: Arguments used for the function
        result: Result returned by the tool
    """
    try:
        # Try to parse result as JSON to determine if it's an error or success
        parsed_result = json.loads(result)

        if isinstance(parsed_result, dict) and "error" in parsed_result:
            # Tool returned an error
            error_msg = parsed_result.get("error", "Unknown error")
            reflection = f"Tool '{func_name}' failed with error: {error_msg}. Args: {args}"

        elif (
            isinstance(parsed_result, dict)
            and parsed_result.get("status") == "skipped"
        ):
            # Redundant call
            reflection = f"Tool '{func_name}' was skipped (redundant call with args: {args})"

        else:
            # Successful execution
            reflection = f"Tool '{func_name}' executed successfully with args {args}. Result: {result[:200]}{'...' if len(result) > 200 else ''}"

    except json.JSONDecodeError:
        # Result is not JSON, treat as plain text success
        reflection = f"Tool '{func_name}' executed successfully with args {args}. Result: {result[:200]}{'...' if len(result) > 200 else ''}"

    # Add the detailed reflection (this goes to reflection_log)
    # state.add_reflection(reflection)
    return reflection


def _process_tool_calls(state: InternalState, msg) -> None:
    """
    Process all tool calls from an assistant message.

    Args:
        state: The internal state object
        msg: The assistant message containing tool calls
    """
    # Add the assistant message with tool_calls FIRST
    state.messages.append(msg)
    print(f"Added assistant message with {len(msg.tool_calls)} tool calls")

    # Process each tool call
    reflections = []
    observations = []
    for i, tool_call in enumerate(msg.tool_calls):
        _, reflection_msg, observation_msg = _execute_single_tool_call(
            state, tool_call, i, len(msg.tool_calls)
        )
        reflections.append(reflection_msg)
        observations.append(observation_msg)
    for r in reflections:
        state.add_reflection(r)
    for o in observations:
        state.messages.append({"role": "assistant", "content": o})


# ---------------------------------------------------------------------------
# LLM WRAPPER WITH RATE-LIMIT HANDLING
# ---------------------------------------------------------------------------


def _chat_completion_with_retry(messages: list, max_attempts: int = 3):
    """Call Azure OpenAI chat completion with basic 429-retry logic."""
    attempt = 0
    while attempt < max_attempts:
        try:
            return client.chat.completions.create(
                model=MODEL_4o,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        except Exception as err:
            # Extract error information
            err_str = str(err)
            if "429" in err_str or "Rate limit" in err_str:
                wait_seconds = 60
                print(
                    f"⚠️ Rate limit hit. Waiting {wait_seconds}s before retry (attempt {attempt+1}/{max_attempts})…"
                )
                import time

                time.sleep(wait_seconds)
                attempt += 1
                continue
            # 400 conversation integrity issues → abort loop
            if "role 'tool'" in err_str and "Invalid parameter" in err_str:
                raise
            # Any other error – re-raise so it is logged by outer handler
            raise
    # If we get here all retries failed
    raise RuntimeError("Exceeded retry attempts due to continual 429 errors")


# ============================================================================
# MAIN AGENT EXECUTION LOOP
# ============================================================================

if __name__ == "__main__":

    while state.can_continue():
        try:
            print("Calling LLM for next tool to invoke")

            if state.llm_calls > 0:
                state.add_next_step_prompt()

            # TODO - Wrap costs and LLM call in a loop
            state.register_llm_call()
            # Debug conversation state before API call
            if DEBUG:
                debug_conversation_state(state.messages, "BEFORE API CALL")

            # --- LLM CALL WITH RATE-LIMIT HANDLING ---
            response = _chat_completion_with_retry(state.messages)
            msg = response.choices[0].message
            if DEBUG:
                print("returned message: ", msg)

            # Check if the model wants to call tools
            if msg.tool_calls and len(msg.tool_calls) > 0:
                _process_tool_calls(state, msg)

            else:
                # Add the assistant message for non-tool responses
                state.messages.append(msg)

                # Handle regular text responses and final answers
                if msg.content and msg.content.strip().startswith(
                    "Final response"
                ):
                    state.register_final_answer(msg.content)

                else:
                    # Handle other text responses
                    state.add_reflection(
                        "Received a response, but no final answer or tool call."
                    )
        except Exception as e:
            state.log_error("Unknown error", str(e), {})
            continue

    if not state.done:
        # Decide reason
        if (
            state.llm_calls > state.max_llm_calls
            or state.tool_calls > state.max_tool_calls
        ):
            reason = "budget exhausted"
        else:
            reason = "unable to solve with available tools"

        fail_msg = f"Final response is = I cannot solve the task with the tools available to me. (Reason: {reason})"
        state.register_final_answer(fail_msg)
        if DEBUG:
            print(fail_msg)
