from openai_client import client, MODEL_4o
import os

def debug_and_regenerate_prog(program_fn: str, errors: str) -> str:
    print("**Entering tool debug_and_regenerate_prog**")
    print(f"Parameter program_fn = {program_fn}")
    print(f"Parameter errors = {errors[:50]}...")  # Truncate long errors

    try:
        with open(program_fn, 'r') as f:
            faulty_code = f.read()
    except Exception as e:
        print("**Exiting tool debug_and_regenerate_prog**")
        return f"Failed to read program file: {str(e)}"

    # Prompt the LLM
    system_prompt = "You are a senior Python developer tasked with fixing code."
    user_prompt = f"""
You are given a Python program and its error output from execution.
Your task is to fix the errors. Return ONLY the corrected code. Do NOT include explanation.

--- Program ---
{faulty_code}

--- Errors ---
{errors}

--- Fixed Program ---
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

        # Overwrite the file with the corrected version
        with open(program_fn, 'w') as f:
            f.write(corrected_code)

        print("**Exiting tool debug_and_regenerate_prog**")
        return corrected_code

    except Exception as e:
        print("**Exiting tool debug_and_regenerate_prog**")
        return f"Failed to regenerate code: {str(e)}"
