import json
from internal_state import InternalState


def add_redundant_tool_response(
    state: InternalState, tool_call, func_name, args
):
    # print(f"🔄 REDUNDANT CALL DETECTED for {func_name} with same args {args}")

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

    # Enhanced reflection with more details
    reflection_msg = f"Skipped redundant call to '{func_name}' with args {args}. Tool was already executed successfully."

    state.add_reflection(reflection_msg)
    # print(f"✅ Added redundant tool response for {tool_call.id}")

    # Sanity Check
    last_msg = state.messages[-1]
    if last_msg.get("tool_call_id") != tool_call.id:
        print(f"❌ ERROR: Tool response not found in messages!")
        print(f"Last message: {last_msg}")
        add_emergency_tool_response(
            state, tool_call, "Tool response not found in messages!"
        )


def get_previous_tool_result(
    state: InternalState, func_name: str, args: dict
) -> str:
    """
    Retrieve the result from a previous tool execution with the same arguments.

    Args:
        state: The internal state object
        func_name: Name of the tool function
        args: Arguments used for the function

    Returns:
        str: The previous result or empty string if not found
    """
    # Try to get from knowledge base
    result_key = f"last_result_{func_name}"
    stored_result = state.get_knowledge(result_key)

    if stored_result and isinstance(stored_result, dict):
        stored_args = stored_result.get("args", {})
        if stored_args == args:  # Same arguments
            return stored_result.get("result", "")

    return ""


def get_tool_execution_summary(state: InternalState) -> dict:
    """
    Get a summary of all tool executions from the knowledge base.

    Args:
        state: The internal state object

    Returns:
        dict: Summary of tool executions including successes and errors
    """
    summary = {
        "successful_tools": {},
        "failed_tools": {},
        "total_executions": 0,
    }

    for key, value in state.knowledge_base.items():
        if key.startswith("last_result_"):
            tool_name = key.replace("last_result_", "")
            summary["successful_tools"][tool_name] = value
            summary["total_executions"] += 1
        elif key.startswith("last_error_"):
            tool_name = key.replace("last_error_", "")
            summary["failed_tools"][tool_name] = value
            summary["total_executions"] += 1

    return summary


def add_tool_response(state: InternalState, tool_call, func_name, result):
    tool_response = {
        "tool_call_id": tool_call.id,
        "role": "tool",
        "name": func_name,
        "content": f"Tool {func_name} execution completed.\n Results: {result}",
    }
    state.messages.append(tool_response)
    # print(f"✅ Added tool response for {tool_call.id}")


def add_emergency_tool_response(
    state: InternalState, tool_call, critical_error
):
    emergency_response = {
        "tool_call_id": tool_call.id,
        "role": "tool",
        "name": getattr(tool_call.function, "name", "unknown_tool"),
        "content": json.dumps(
            {
                "error": f"Critical tool processing error: {str(critical_error)}",
                "tool_call_id": tool_call.id,
            }
        ),
    }
    state.messages.append(emergency_response)
    # print(f"🚨 Added emergency response for {tool_call.id}")


def update_knowledge_base(state: InternalState, func_name, args, result):
    """
    Parse tool outputs and store structured, reusable data in the knowledge base.
    The knowledge base remains a flat dict with clear keys to avoid nested complexity.
    """

    try:
        parsed = json.loads(result)
    except Exception:
        return  # Non-JSON result or unusable output

    # --- TOOL: extract_entities_from_file ---
    if func_name == "extract_entities_from_file":
        try:
            entities = json.loads(result)  # Parse the JSON array
        except:
            entities = []  # Fallback if parsing fails

        entity_type = args.get("entity_type", "unknown").lower()

        if entities:
            key = f"{entity_type}_entities"
            existing = state.knowledge_base.get(key, [])
            combined = list(set(existing) | set(entities))  # Avoid duplicates
            state.update_knowledge(key, combined)
            # print(f"✅ Stored {len(entities)} entities under '{key}'")

    # --- TOOL: online_search ---
    elif func_name == "online_search":
        entity = args.get("an_entity")
        attribute = args.get("an_attribute")
        value = parsed.get("attribute_value")

        if entity and attribute and value:
            key = f"{entity.lower()}_{attribute.lower()}"
            state.update_knowledge(key, value)
            # print(f"✅ Stored search result under '{key}': {value}")

    elif func_name == "write_file":
        filename = args.get("fn", "last_written_file_name")
        content = args.get("file_content", "Empty file content")
        state.update_knowledge(filename, content)


def get_recent_tool_results(state: InternalState, limit: int = 5) -> list:
    """
    Get recent tool execution results from the reflection log.

    Args:
        state: The internal state object
        limit: Number of recent tool results to return

    Returns:
        list: Recent tool execution reflections
    """
    tool_reflections = [
        reflection
        for reflection in state.reflection_log
        if any(
            word in reflection.lower()
            for word in ["tool", "executed", "failed", "skipped"]
        )
    ]

    return tool_reflections[-limit:] if tool_reflections else []
