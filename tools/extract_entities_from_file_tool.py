import os
import json
from typing import Optional
from tools.openai_client import client, MODEL_4o


def extract_entities_from_file(
    file_name: str,
    entity_type: str,
) -> str:
    """
    Extracts entities of a specified type from a text file using the LLM.

    Args:
        file_name (str): The path to the text file.
        entity_type (str): Type of entity to extract (e.g., 'city', 'person', 'organization').

    Returns:
        str: JSON-formatted list of entities: '["Entity1", "Entity2", ...]'
    """

    try:
        if not os.path.exists(file_name):
            return json.dumps(
                {"error": f"File not found: {file_name}", "entities": []}
            )

        with open(file_name, "r", encoding="utf-8") as f:
            content = f.read()

        prompt = (
            f"Extract all entities of type '{entity_type}' from the following text. "
            'Return them as a Python list of strings (e.g., ["New York", "Chicago"]). '
            "In case you cannot find any entities, return an empty list. ONLY RETURN THE LIST, NO OTHER TEXT."
            "Text:\n---\n" + content + "\n---"
        )

        completion = client.chat.completions.create(
            model=MODEL_4o,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts entities from text. You are an expert in the field of data extraction and analysis. Go over the entire text content and extract all entities of the specified type.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        response_text = completion.choices[0].message.content.strip()

        # Ensure response is a valid list string
        if response_text.startswith("[") and response_text.endswith("]"):
            return response_text
        else:
            return json.dumps(
                {
                    "error": "LLM response is not a valid list.",
                    "raw_response": response_text,
                }
            )

    except Exception as e:
        return json.dumps({"error": str(e), "entities": []})


if __name__ == "__main__":
    file_name = "students.txt"
    print(extract_entities_from_file(file_name, "cars"))
