import subprocess

def execute_Python_prog(program_fn: str) -> str:
    print("**Entering tool execute_Python_prog**")
    print(f"Parameter program_fn = {program_fn}")

    try:
        result = subprocess.run(
            ["python", program_fn],
            capture_output=True,
            text=True,
            timeout=10  # Optional: avoid infinite loops
        )
        if result.returncode == 0:
            print("**Exiting tool execute_Python_prog**")
            return 'Program executed successfully'
        else:
            print("**Exiting tool execute_Python_prog**")
            return result.stderr.strip()
    except Exception as e:
        print("**Exiting tool execute_Python_prog**")
        return str(e)
