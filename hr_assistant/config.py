
"""Configuration for the HR RAG application.

Supports:
- Local development through .env
- Streamlit Community Cloud through st.secrets
"""

import os
from dotenv import load_dotenv

# Load local environment variables.
load_dotenv()


def get_setting(name: str, default=None):
    """Read a setting from environment variables or Streamlit secrets.

    Environment variables take priority, allowing local .env overrides.
    """

    value = os.getenv(name)

    if value is not None and value != "":
        return value

    try:
        import streamlit as st

        value = st.secrets.get(name, default)
        return value

    except Exception:
        return default


# --------------------------------------------------
# API KEYS AND CONNECTIONS
# --------------------------------------------------

GEMINI_API_KEY = get_setting("GEMINI_API_KEY")
JINA_API_KEY = get_setting("JINA_API_KEY")

QDRANT_URL = get_setting("QDRANT_URL")
QDRANT_API_KEY = get_setting("QDRANT_API_KEY")

GROQ_API_KEY = get_setting("GROQ_API_KEY")
GROQ_BASE_URL = get_setting(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1"
)

PROJECT_ID = get_setting("PROJECT_ID")
LOCATION = get_setting("LOCATION")
GCS_BUCKET_NAME = get_setting("GCS_BUCKET_NAME")


# --------------------------------------------------
# QDRANT COLLECTIONS
# --------------------------------------------------

QDRANT_COLLECTION_NAME = get_setting(
    "QDRANT_COLLECTION_NAME",
    "hr_policies"
)

QDRANT_NOISY_COLLECTION_NAME = get_setting(
    "QDRANT_NOISY_COLLECTION_NAME",
    "hr_policies_noisy_demo"
)


# --------------------------------------------------
# EMPLOYEE ACCESS
# --------------------------------------------------

ALLOWED_EMPLOYEE_EMAILS = {
    email.strip()
    for email in get_setting(
        "ALLOWED_EMPLOYEE_EMAILS", ""
    ).split(",")
    if email.strip()
}


# --------------------------------------------------
# LLM CONFIGURATION
# --------------------------------------------------

LLM_MODEL_NAME = get_setting(
    "LLM_MODEL_NAME",
    "gemini-3.5-flash-lite"
)

FALLBACK_MODEL_NAME = get_setting(
    "FALLBACK_MODEL_NAME",
    "openai/gpt-oss-20b"
)


# --------------------------------------------------
# EMBEDDINGS AND RERANKING
# --------------------------------------------------

EMBEDDING_MODEL_NAME = get_setting(
    "EMBEDDING_MODEL_NAME",
    "jina-embeddings-v2-base-en"
)

RERANKER_MODEL_NAME = get_setting(
    "RERANKER_MODEL_NAME",
    "jina-reranker-v2-base-multilingual"
)


# --------------------------------------------------
# DOCUMENT PROCESSING
# --------------------------------------------------

CHUNK_SIZE = 500
CHUNK_OVERLAP = 60


# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

TOP_K_RESULTS = 5
RERANK_CANDIDATE_K = 12

HR_POLICY_CATEGORIES = {
    "Leave",
    "Work From Home",
    "Probation",
    "Notice Period",
    "Reimbursement",
    "Code of Conduct",
    "Holidays",
    "Maternity Paternity",
    "Travel Expense",
    "Exit Process",
}

RELEVANCE_THRESHOLD = 0.35


# --------------------------------------------------
# GUARDRAILS
# --------------------------------------------------

GUARDRAIL_PROVIDER = get_setting(
    "GUARDRAIL_PROVIDER",
    "gemini_lite"
)

MODEL_ARMOR_LOCATION = get_setting(
    "MODEL_ARMOR_LOCATION",
    "us"
)

MODEL_ARMOR_TEMPLATE_ID = get_setting(
    "MODEL_ARMOR_TEMPLATE_ID",
    "hr-assistant-guardrail"
)

GUARDRAIL_FAIL_OPEN_INPUT = (
    str(get_setting("GUARDRAIL_FAIL_OPEN_INPUT", "false"))
    .strip()
    .lower() == "true"
)

GUARDRAIL_FAIL_OPEN_OUTPUT = (
    str(get_setting("GUARDRAIL_FAIL_OPEN_OUTPUT", "false"))
    .strip()
    .lower() == "true"
)

GUARDRAIL_HISTORY_TURNS = int(
    get_setting("GUARDRAIL_HISTORY_TURNS", "6")
)


# --------------------------------------------------
# SEMANTIC CACHE
# --------------------------------------------------

SEMANTIC_CACHE_THRESHOLD = 0.93

SEMANTIC_CACHE_MAX_ENTRIES = int(
    get_setting("SEMANTIC_CACHE_MAX_ENTRIES", "500")
)

SEMANTIC_CACHE_TTL_SECONDS = int(
    get_setting("SEMANTIC_CACHE_TTL_SECONDS", "3600")
)


# --------------------------------------------------
# LANGSMITH
# --------------------------------------------------

LANGSMITH_TRACING = get_setting(
    "LANGSMITH_TRACING",
    "false"
)

LANGSMITH_ENDPOINT = get_setting(
    "LANGSMITH_ENDPOINT",
    "https://api.smith.langchain.com"
)

LANGSMITH_API_KEY = get_setting("LANGSMITH_API_KEY")

LANGSMITH_PROJECT = get_setting(
    "LANGSMITH_PROJECT",
    "hr-policy-assistant"
)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

JUDGE_MODEL_NAME = get_setting(
    "JUDGE_MODEL_NAME",
    "openai/gpt-oss-120b"
)


# --------------------------------------------------
# VALIDATE REQUIRED SETTINGS
# --------------------------------------------------

def check_api_keys() -> None:
    """Validate credentials required by the HR RAG application."""

    required = {
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "JINA_API_KEY": JINA_API_KEY,
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
    }

    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:
        raise ValueError(
            "Missing required configuration values: "
            + ", ".join(missing)
        )
