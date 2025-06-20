import json
from typing import Optional

from tools.tool_manager import ToolManager


def write_file(file_content: str, fn: str,tool_manager: Optional[ToolManager] = None) -> str:
    """
    Writes the given content to a file and returns a JSON-formatted confirmation.

    Args:
        file_content: The content to write.
        fn: The file name or path to write to.

    Returns:
        A JSON string confirming success or describing the error.
    """

    if tool_manager:
        if not tool_manager.can_call_tool():
            return json.dumps({
                "status": "error",
                "file_name": fn,
                "error": "Tool usage limit exceeded"
            })
        tool_manager.register_tool_call("write_file")


    try:
        with open(fn, "w", encoding="utf-8") as f:
            f.write(file_content)

        return json.dumps(
            {
                "status": "success",
                "file_name": fn,
                "message": f"Content successfully written to {fn}.",
            }
        )
    except Exception as e:
        return json.dumps(
            {"status": "error", "file_name": fn, "error": str(e)}
        )
