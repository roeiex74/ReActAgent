import json
from typing import Optional


def write_file(file_content: str, fn: str) -> str:
    """
    Writes the given content to a file and returns a JSON-formatted confirmation.

    Args:
        file_content: The content to write.
        fn: The file name or path to write to.

    Returns:
        A JSON string confirming success or describing the error.
    """

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
