from openai import AzureOpenAI
from dotenv import load_dotenv
import json
from openai import AzureOpenAI
import os
import subprocess
from internal_state import InternalState
from tools.online_search_tool import internet_search_attribute
from tools.write_file_tool import write_file

# Load environment variables from .env
load_dotenv()

# Model and version
MODEL_4o = os.getenv("MODEL_4o")
AZURE_OPEN_VERSION_4o = os.getenv("AZURE_OPEN_VERSION_4o")

# Read Azure credentials
AZURE_OPENAI_API_KEY = os.getenv("CLASS_OPEN_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("SUBSCRIPTION_OPENAI_ENDPOINT")

# Initialize the OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPEN_VERSION_4o,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

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
    # Inside tool_schemas list
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
]


# Tool execution function
def execute_tool(tool_name: str, **kwargs):
    """Execute the specified tool with given arguments - always returns a string"""
    try:
        if tool_name == "online_search":
            # Validate required parameters
            an_entity = kwargs.get("an_entity")
            an_attribute = kwargs.get("an_attribute")

            if (
                not an_entity
                or not isinstance(an_entity, str)
                or not an_entity.strip()
            ):
                return json.dumps(
                    {
                        "error": "Invalid or missing 'an_entity' parameter",
                        "received": an_entity,
                        "tool": tool_name,
                    }
                )

            if (
                not an_attribute
                or not isinstance(an_attribute, str)
                or not an_attribute.strip()
            ):
                return json.dumps(
                    {
                        "error": "Invalid or missing 'an_attribute' parameter",
                        "received": an_attribute,
                        "tool": tool_name,
                    }
                )

            print(
                f"[TOOL] Executing {tool_name} for entity: '{an_entity}', attribute: '{an_attribute}'"
            )

            try:
                result = internet_search_attribute(
                    an_entity=an_entity,
                    an_attribute=an_attribute,
                    max_results=kwargs.get("max_retries", 3),
                )

                # Ensure result is a valid string
                if not isinstance(result, str):
                    return json.dumps(
                        {
                            "error": "Tool returned invalid response type",
                            "expected": "string",
                            "received": type(result).__name__,
                            "tool": tool_name,
                        }
                    )

                return result

            except Exception as tool_error:
                return json.dumps(
                    {
                        "error": f"Tool execution failed: {str(tool_error)}",
                        "tool": tool_name,
                        "entity": an_entity,
                        "attribute": an_attribute,
                    }
                )
        elif tool_name == "write_file":
            file_content = kwargs.get("file_content")
            fn = kwargs.get("fn")
            return write_file(file_content, fn)
        else:
            error_response = {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": ["online_search"],
                "received_tool": tool_name,
            }
            print(f"[TOOL ERROR] {error_response['error']}")
            return json.dumps(error_response)

    except Exception as e:
        # This should never happen, but just in case
        error_response = {
            "error": f"Critical error in execute_tool: {str(e)}",
            "tool": tool_name,
            "parameters": kwargs,
        }
        print(f"[CRITICAL ERROR] {error_response['error']}")
        return json.dumps(error_response)


def analyze_input_file(file_name: str):
    with open(file_name, "r") as f:
        content = json.load(f)  # Parse JSON directly from file object
    query_file_name = content["query_name"]
    state.set_query_data(query_file_name)
    state.add_reflection(
        f"Extracted query file data: {query_file_name}", log=False
    )
    for obj in content["file_resources"]:
        file_name = obj["file_name"]
        description = obj["description"]
        state.file_resources[file_name] = description
        state.add_reflection(
            f"Analyzed file {file_name} with description: {description}",
            log=False,
        )
    file_reflection = (
        f"Analyzed {len(content['file_resources'])} files"
        if len(content["file_resources"]) > 0
        else "No resource file availlable for usage."
    )
    state.add_reflection(file_reflection, log=False)


initial_system_prompt = """You are a ReAct-style agent. You solve queries by interleaving three steps:

1. Thought — Reflect on the current task and decide the next action.
2. Action — Call one of the tools if needed, or give a final answer.
3. Observation — The result from the tool or a confirmation.

You must continue this loop until the task is completed or an error prevents further progress.

You have access to the following tools:
- extract_entities_from_file(file_name: str, entity_type: str)
- Internet_search_attribute(entity: str, attribute: str)
- gen_plot_prog(plot_request: str, input_file: str, columns: str, gen_output_program_fn: str, output_png: str)
- execute_Python_prog(program_fn: str)
- debug_and_regenerate_prog(program_fn: str, errors: str)
- write_file(file_content: str, fn: str)

You can solve the query based on the given tools, and the files available to you.
If you experience an error with a tool, debug and analyze the error, if you can fix it and you believe you should call the tool again, do so. 
otherwise, you should not call the tool again, and you should find a different set of actions to solve the task.

In case you encounter an error, and after debugging and analyzing alternatives steps with given tools - you conclude there is no action you can take to solve the task,
you should respond with "Final Answer: I cannot solve the task with the tools available to me."

If you need to perform an action, **use one of these tools directly by calling it**.

If you believe you have fully answered the user's query, STOP and respond like this:
Final Answer: <the final answer to the user query content>

Do NOT take any more actions after you have given the Final Answer.
Do not call tools after providing the Final Answer.


Do not continue after you have given the Final Answer.
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


if __name__ == "__main__":
    # Test the tool integration first
    # test_tool_integration()

    # Then run your normal agent logic
    while state.can_continue():
        try:
            print("Calling LLM for next tool to invoke")
            if state.llm_calls > 0:
                state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "Given the above observation and prior steps, think carefully about the next action.\n"
                            "If you have completed the task and all required information has been gathered, produce your final answer with 'Final Answer: ...'.\n"
                            "Otherwise, explain your reasoning and choose the next appropriate tool or step."
                        ),
                    }
                )

            state.register_llm_call()

            # Debug conversation state before API call
            debug_conversation_state(state.messages, "BEFORE API CALL")

            response = client.chat.completions.create(
                model=MODEL_4o,
                messages=state.messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            print("returned message: ", msg)

            # Check if the model wants to call tools
            if msg.tool_calls and len(msg.tool_calls) > 0:
                # Add the assistant message with tool_calls FIRST
                state.messages.append(msg)
                print(
                    f"Added assistant message with {len(msg.tool_calls)} tool calls"
                )

                # Process each tool call with guaranteed response messages
                for i, tool_call in enumerate(msg.tool_calls):
                    tool_response_added = False
                    print(
                        f"Processing tool call {i+1}/{len(msg.tool_calls)}: {tool_call.id}"
                    )

                    try:
                        # Extract function name and arguments correctly for new format
                        func_name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)

                        print(f"Executing tool: {func_name} with args: {args}")

                        # Check for redundant calls
                        if state.already_called_tool(func_name, args):
                            print(
                                f"🔄 REDUNDANT CALL DETECTED for {func_name}"
                            )

                            # Add tool response for skipped call with explicit debugging
                            redundant_response = {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": func_name,
                                "content": json.dumps(
                                    {
                                        "status": "skipped",
                                        "reason": "Redundant call - same arguments used previously",
                                        "tool": func_name,
                                        "args": args,
                                    }
                                ),
                            }

                            state.messages.append(redundant_response)
                            state.add_reflection(
                                f"Skipping redundant call to {func_name} with same args."
                            )
                            tool_response_added = True
                            print(
                                f"✅ Added redundant tool response for {tool_call.id}"
                            )

                            # Validate that the message was actually added
                            last_msg = state.messages[-1]
                            if last_msg.get("tool_call_id") == tool_call.id:
                                print(
                                    f"✅ Confirmed: Tool response added successfully"
                                )
                            else:
                                print(
                                    f"❌ ERROR: Tool response not found in messages!"
                                )
                                print(f"Last message: {last_msg}")

                            continue

                        # Update state tracking (with error handling)
                        try:
                            state.register_tool_call(func_name, args)
                        except Exception as state_error:
                            print(
                                f"Warning: State tracking error: {state_error}"
                            )

                        # Execute the tool with comprehensive error handling
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
                            print(
                                f"Tool {func_name} execution completed.\n Results: {result}"
                            )

                        except Exception as tool_error:
                            print(f"Tool execution failed: {tool_error}")
                            result = json.dumps(
                                {
                                    "error": f"Tool execution failed: {str(tool_error)}",
                                    "tool": func_name,
                                    "args": args,
                                }
                            )

                        # Always add tool result to conversation
                        tool_response = {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": func_name,
                            "content": f"Tool {func_name} execution completed.\n Results: {result}",
                        }
                        state.messages.append(tool_response)

                        tool_response_added = True
                        print(f"✅ Added tool response for {tool_call.id}")

                        # Update state tracking exit (with error handling)
                        try:
                            state.exit_tool(func_name)
                        except Exception as state_error:
                            print(f"Warning: State exit error: {state_error}")

                        state.add_reflection(
                            f"Tool {func_name} completed. Preparing next step based on result."
                        )
                        # After Reflection, add assistant Observation message before next thought
                        state.messages.append(
                            {
                                "role": "assistant",
                                "content": f"Observation: The tool `{func_name}` returned:\n{result}",
                            }
                        )

                    except Exception as critical_error:
                        # This is a critical error - ensure we still add a tool response
                        print(
                            f"Critical error processing tool call {tool_call.id}: {critical_error}"
                        )

                        if not tool_response_added:
                            # Add emergency error response to maintain conversation integrity
                            emergency_response = {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": getattr(
                                    tool_call.function,
                                    "name",
                                    "unknown_tool",
                                ),
                                "content": json.dumps(
                                    {
                                        "error": f"Critical tool processing error: {str(critical_error)}",
                                        "tool_call_id": tool_call.id,
                                    }
                                ),
                            }
                            state.messages.append(emergency_response)
                            print(
                                f"🚨 Added emergency response for {tool_call.id}"
                            )

                        # Log the error but continue processing other tool calls
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

            else:
                # Add the assistant message for non-tool responses
                state.messages.append(msg)

                # Handle regular text responses and final answers
                if msg.content and msg.content.strip().startswith(
                    "Final Answer:"
                ):
                    state.register_final_answer(msg.content)
                    print(f"Final answer received: {msg.content}")
                else:
                    # Handle other text responses
                    state.add_reflection(
                        "Received a response, but no final answer or tool call."
                    )
        except Exception as e:
            state.log_error("Unknown error", str(e), {})
            continue
