import subprocess
import json
import os


def execute_Python_prog(program_fn: str) -> str:
    """
    Execute a Python program file and return structured results.

    Args:
        program_fn: Path to the Python file to execute
        tool_manager: Optional tool manager for usage tracking

    Returns:
        str: JSON string containing execution results with structure:
        {
            "status": "success|error|limit_exceeded",
            "program_file": str,
            "stdout": str,
            "stderr": str,
            "return_code": int,
            "execution_time": float,
            "error": str (only if error occurred)
        }
    """
    # Input validation
    if not program_fn or not isinstance(program_fn, str):
        return json.dumps(
            {
                "status": "error",
                "program_file": program_fn,
                "error": "Invalid program_fn: must be a non-empty string",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_time": 0.0,
            }
        )

    # Normalize the file path
    program_fn = program_fn.strip()

    # Check if file exists
    if not os.path.exists(program_fn):
        return json.dumps(
            {
                "status": "error",
                "program_file": program_fn,
                "error": f"File not found: {program_fn}",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_time": 0.0,
            }
        )

    # Check if file has .py extension
    if not program_fn.endswith(".py"):
        return json.dumps(
            {
                "status": "error",
                "program_file": program_fn,
                "error": f"Invalid file type: {program_fn}. Expected .py file",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_time": 0.0,
            }
        )

    try:
        import time

        start_time = time.time()

        # Execute the Python program
        result = subprocess.run(
            ["python", program_fn], capture_output=True, text=True, timeout=30
        )

        execution_time = time.time() - start_time

        # Prepare response based on execution result
        if result.returncode == 0:
            return json.dumps(
                {
                    "status": "success",
                    "program_file": program_fn,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                    "return_code": result.returncode,
                    "execution_time": round(execution_time, 3),
                }
            )
        else:
            return json.dumps(
                {
                    "status": "error",
                    "program_file": program_fn,
                    "error": f"Program execution failed with return code {result.returncode}",
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                    "return_code": result.returncode,
                    "execution_time": round(execution_time, 3),
                }
            )

    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "status": "error",
                "program_file": program_fn,
                "error": "Program execution timed out (30 seconds limit)",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_time": 30.0,
            }
        )

    except FileNotFoundError:
        return json.dumps(
            {
                "status": "error",
                "program_file": program_fn,
                "error": "Python interpreter not found. Ensure Python is installed and in PATH",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_time": 0.0,
            }
        )

    except PermissionError:
        return json.dumps(
            {
                "status": "error",
                "program_file": program_fn,
                "error": f"Permission denied: Cannot execute {program_fn}",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_time": 0.0,
            }
        )

    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "program_file": program_fn,
                "error": f"Unexpected error during execution: {str(e)}",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_time": 0.0,
            }
        )
