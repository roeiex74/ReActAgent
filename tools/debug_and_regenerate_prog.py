from typing import Optional
from openai_client import client, MODEL_4o
from tools.tool_manager import ToolManager

def debug_and_regenerate_prog(program_fn: str, errors: str, tool_manager: Optional[ToolManager] = None) -> str:

    if tool_manager:
        tool_manager.register_tool_call("debug_and_regenerate_prog")
        if not tool_manager.can_call_llm():
            return "LLM call limit exceeded"

        tool_manager.register_llm_call()


    print("**Entering tool debug_and_regenerate_prog**")
    print(f"Parameter program_fn = {program_fn}")
    print(f"Parameter errors = {errors[:50]}...")

    try:
        with open(program_fn, 'r') as file:
            faulty_code = file.read()
    except Exception as e:
        print("**Exiting tool debug_and_regenerate_prog**")
        return f"Failed to read program file: {str(e)}"

    system_prompt = "You are an expert Python programmer and code fixer."
    user_prompt = f"""You are given a faulty Python program and an error message.
Your job is to fix the problem and return the corrected program only.
Do not include explanations or anything else.

### Faulty code:
{faulty_code}

### Error message:
{errors}

### Corrected code:
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_4o,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        corrected_code = response.choices[0].message.content.strip()

        with open(program_fn, 'w') as file:
            file.write(corrected_code)

        print("**Exiting tool debug_and_regenerate_prog**")
        return corrected_code

    except Exception as e:
        print("**Exiting tool debug_and_regenerate_prog**")
        return f"Failed to regenerate code: {str(e)}"
