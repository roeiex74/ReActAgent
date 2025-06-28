# openai_client.py

import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

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

__all__ = ["client", "MODEL_4o"]
