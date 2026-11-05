
"""Gemini LLM configuration for the HR RAG application."""

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from hr_assistant import config


@lru_cache(maxsize=1)
def get_llm():
    """
    Return a cached Gemini chat model.

    Uses the Gemini Developer API (Google AI Studio),
    not Vertex AI or Google Cloud credentials.
    """

    if not config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )

    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL_NAME,
        api_key=config.GEMINI_API_KEY,
        temperature=0,
        max_retries=2,
    )
