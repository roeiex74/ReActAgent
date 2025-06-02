from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import (
    RatelimitException,
    DuckDuckGoSearchException,
)
from dotenv import load_dotenv
import os
import time
import json
import logging
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion
from requests.exceptions import RequestException, Timeout, ConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_4o = "gpt-4o-mini"
AZURE_OPEN_VERSION_4o = "2024-08-01-preview"

# Read Azure credentials
AZURE_OPENAI_API_KEY = os.getenv("CLASS_OPEN_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("SUBSCRIPTION_OPENAI_ENDPOINT")

TEMPERATURE = 0.5

# Validate environment variables
if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
    logger.error(
        "Missing required environment variables: CLASS_OPEN_API_KEY or SUBSCRIPTION_OPENAI_ENDPOINT"
    )
    raise ValueError(
        "Azure OpenAI credentials not found in environment variables"
    )

try:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPEN_VERSION_4o,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    raise


def generate_search_query(
    an_entity: str,
    an_attribute: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """
    Generate a search query for the given entity and attribute with comprehensive error handling.

    Args:
        an_entity: The entity to search for
        an_attribute: The attribute to find
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        str: Generated search query, or fallback query if generation fails

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If all retry attempts fail
    """
    # Input validation
    if (
        not an_entity
        or not isinstance(an_entity, str)
        or not an_entity.strip()
    ):
        raise ValueError("Entity must be a non-empty string")
    if (
        not an_attribute
        or not isinstance(an_attribute, str)
        or not an_attribute.strip()
    ):
        raise ValueError("Attribute must be a non-empty string")

    an_entity = an_entity.strip()
    an_attribute = an_attribute.strip()

    # Fallback query in case of API failure
    fallback_query = f'"{an_entity}" {an_attribute}'

    prompt = f"Generate a search query for the given entity and attribute. Entity: {an_entity}, Attribute: {an_attribute}"
    system_message = "You are a helpful assistant that generates search queries for the given entity and attribute. Provide a clear and concise search query that will return relevant results."

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Generating search query attempt {attempt + 1}/{max_retries}"
            )

            response = client.chat.completions.create(
                model=MODEL_4o,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                timeout=30,  # 30 second timeout
            )

            generated_query = response.choices[0].message.content

            if generated_query and generated_query.strip():
                logger.info(
                    f"Successfully generated search query: {generated_query}"
                )
                return generated_query.strip()
            else:
                logger.warning("Generated query is empty, using fallback")
                return fallback_query

        except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1} failed to generate search query: {e}"
            )

            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(
                    f"All {max_retries} attempts failed. Using fallback query: {fallback_query}"
                )
                return fallback_query


def search_duckduckgo(
    query: str,
    max_results: int = 3,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> list:
    """
    Search DuckDuckGo with comprehensive error handling and rate limit management.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries in seconds

    Returns:
        list: List of search result dictionaries, empty list if all attempts fail
    """
    # Input validation
    if not query or not isinstance(query, str) or not query.strip():
        logger.error("Query must be a non-empty string")
        return []

    if max_results <= 0:
        logger.error("max_results must be positive")
        return []

    query = query.strip()

    for attempt in range(max_retries):
        try:
            logger.info(
                f"DuckDuckGo search attempt {attempt + 1}/{max_retries} for query: {query}"
            )

            # Add delay to respect rate limits
            if attempt > 0:
                wait_time = retry_delay * (
                    2 ** (attempt - 1)
                )  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

            # Perform search
            results = DDGS().text(query, max_results=max_results)
            search_results = []

            for result in results:
                try:
                    search_result = {
                        "title": result.get("title", "No title available"),
                        "url": result.get("href", ""),
                        "description": result.get(
                            "body", "No description available"
                        ),
                    }
                    search_results.append(search_result)
                except Exception as e:
                    logger.warning(
                        f"Error processing individual search result: {e}"
                    )
                    continue

            logger.info(
                f"Successfully retrieved {len(search_results)} search results"
            )
            return search_results

        except RatelimitException:
            logger.warning(f"Rate limit hit on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                rate_limit_wait = 60  # Wait 60 seconds for rate limit
                logger.info(
                    f"Waiting {rate_limit_wait} seconds for rate limit..."
                )
                time.sleep(rate_limit_wait)
            else:
                logger.error("Rate limit exceeded on final attempt")

        except DuckDuckGoSearchException as e:
            logger.warning(
                f"DuckDuckGo search exception on attempt {attempt + 1}: {e}"
            )
            if "Ratelimit" in str(e):
                if attempt < max_retries - 1:
                    logger.info("Rate limit detected, waiting 60 seconds...")
                    time.sleep(60)
                else:
                    logger.error("Rate limit on final attempt")

        except (ConnectionError, Timeout, RequestException) as e:
            logger.warning(f"Network error on attempt {attempt + 1}: {e}")

        except Exception as e:
            logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")

    logger.error(
        f"All {max_retries} search attempts failed for query: {query}"
    )
    return []


def generate_search_results(
    an_entity: str, an_attribute: str, max_retries: int = 3
) -> str:
    """
    Generate search results and extract attribute value with comprehensive error handling.

    Args:
        an_entity: The entity to search for
        an_attribute: The attribute to extract
        max_retries: Maximum number of retry attempts for GPT processing

    Returns:
        str: JSON string with results or error information
    """
    try:
        # Input validation
        if (
            not an_entity
            or not isinstance(an_entity, str)
            or not an_entity.strip()
        ):
            error_result = {
                "entity": an_entity if an_entity else "INVALID",
                "attribute": an_attribute if an_attribute else "INVALID",
                "attribute_value": "ERROR: Invalid entity provided",
                "error": "Input validation failed",
            }
            return json.dumps(error_result)

        if (
            not an_attribute
            or not isinstance(an_attribute, str)
            or not an_attribute.strip()
        ):
            error_result = {
                "entity": an_entity,
                "attribute": an_attribute if an_attribute else "INVALID",
                "attribute_value": "ERROR: Invalid attribute provided",
                "error": "Input validation failed",
            }
            return json.dumps(error_result)

        an_entity = an_entity.strip()
        an_attribute = an_attribute.strip()

        # Step 1: Generate search query
        try:
            query = generate_search_query(an_entity, an_attribute)
        except Exception as e:
            logger.error(f"Failed to generate search query: {e}")
            error_result = {
                "entity": an_entity,
                "attribute": an_attribute,
                "attribute_value": "ERROR: Failed to generate search query",
                "error": str(e),
            }
            return json.dumps(error_result)

        # Step 2: Perform search
        try:
            search_results = search_duckduckgo(query)
            if not search_results:
                error_result = {
                    "entity": an_entity,
                    "attribute": an_attribute,
                    "attribute_value": "ERROR: No search results found",
                    "error": "Search returned empty results",
                }
                return json.dumps(error_result)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            error_result = {
                "entity": an_entity,
                "attribute": an_attribute,
                "attribute_value": "ERROR: Search operation failed",
                "error": str(e),
            }
            return json.dumps(error_result)

        # Step 3: Process results with GPT
        system_message = f"""
        You are a helpful assistant that reviews search results for the given entity and attribute. 
        You are provided with online search results from a given query, in a list of dictionaries, where each dictionary contains:
        1. Result title
        2. Result URL
        3. Result Description
        
        You are to review the provided results and extract the value of the attribute from the results, and return an answer ONLY in the given JSON format:
         "entity": "{an_entity}", "attribute": "{an_attribute}",
    "attribute_value": <value>

    where <value> is the value extracted from the Internet
    search results. 
        
        THE OUTPUT MUST BE ONLY IN THE GIVEN JSON FORMAT, NO OTHER TEXT OR MARKDOWN.
        """

        user_message = f"""
        Here are the search results:
        {search_results}
        for the query: {query}
        with entity: {an_entity} and attribute: {an_attribute}
        """

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Processing search results with GPT, attempt {attempt + 1}/{max_retries}"
                )

                response = client.chat.completions.create(
                    model=MODEL_4o,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=TEMPERATURE,
                    timeout=30,
                )

                result_content = response.choices[0].message.content

                # Validate JSON format
                try:
                    parsed_result = json.loads(result_content)
                    # Ensure required fields are present
                    if all(
                        key in parsed_result
                        for key in ["entity", "attribute", "attribute_value"]
                    ):
                        logger.info("Successfully processed search results")
                        return result_content
                    else:
                        logger.warning("GPT response missing required fields")

                except json.JSONDecodeError:
                    logger.warning(
                        f"GPT returned invalid JSON on attempt {attempt + 1}"
                    )

                if attempt < max_retries - 1:
                    time.sleep(2)  # Brief delay before retry

            except Exception as e:
                logger.warning(
                    f"GPT processing failed on attempt {attempt + 1}: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2)

        # Final fallback if all GPT attempts fail
        logger.error("All GPT processing attempts failed")
        fallback_result = {
            "entity": an_entity,
            "attribute": an_attribute,
            "attribute_value": "ERROR: Failed to process search results",
            "error": "GPT processing failed after all retry attempts",
            "raw_search_results": search_results[
                :2
            ],  # Include first 2 results for debugging
        }
        return json.dumps(fallback_result)

    except Exception as e:
        logger.error(f"Unexpected error in generate_search_results: {e}")
        error_result = {
            "entity": an_entity if "an_entity" in locals() else "UNKNOWN",
            "attribute": (
                an_attribute if "an_attribute" in locals() else "UNKNOWN"
            ),
            "attribute_value": "ERROR: Unexpected system error",
            "error": str(e),
        }
        return json.dumps(error_result)


if __name__ == "__main__":
    try:
        an_entity = "Abraham Lincoln"
        an_attribute = "birthplace"
        result = generate_search_results(an_entity, an_attribute)
        print(result)
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        error_output = {
            "entity": "Abraham Lincoln",
            "attribute": "birthplace",
            "attribute_value": "ERROR: System failure",
            "error": str(e),
        }
        print(json.dumps(error_output))
