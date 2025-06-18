import requests
import os
from openai import AzureOpenAI
import json
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")


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


def bing_search_contexts(query: str, max_results: int = 3):
    """Search using Serper API and return formatted contexts"""
    try:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": SERPER_API_KEY}
        payload = {"q": query}

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        results = response.json().get("organic", [])

        contexts = []
        for r in results[:max_results]:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            context = f"{title}\n{snippet}\nURL: {link}"
            contexts.append(context)

        return contexts

    except requests.exceptions.RequestException as e:
        raise Exception(f"Search failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Search error: {str(e)}")


def internet_search_attribute(
    an_entity: str, an_attribute: str, max_results: int = 3
) -> str:
    query = f"{an_entity} {an_attribute}"
    try:
        snippets = bing_search_contexts(query, max_results)

        # Construct prompt for LLM
        prompt = (
            f"Entity: {an_entity}\nAttribute: {an_attribute}\n"
            f"Search Results:\n- " + "\n- ".join(snippets) + "\n\n"
            "Based on the search results, return a JSON object like:\n"
            '{"entity": "...", "attribute": "...", "attribute_value": "..."}'
        )
        messages = [
            {
                "role": "system",
                "content": """You extract structured info from web search results. Do not use any data that is not provided in the search results.
                If the answer to the attribute is not provided in the search results, return "not found".
                """,
            },
            {"role": "user", "content": prompt},
        ]

        completion = client.chat.completions.create(
            model=MODEL_4o, messages=messages
        )

        return completion.choices[0].message.content

    except Exception as e:
        return json.dumps(
            {
                "entity": an_entity,
                "attribute": an_attribute,
                "error": f"Search or parsing failed: {str(e)}",
            }
        )


if __name__ == "__main__":
    print(
        internet_search_attribute("Maccabi Haifa", "first championship year")
    )
