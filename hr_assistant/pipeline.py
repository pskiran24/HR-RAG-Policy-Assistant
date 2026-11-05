
"""17 · pipeline — wire the components into ready-to-use agents.

    ask()       input guardrail -> semantic cache -> agent
                -> output guardrail -> cache store.

    ask_plain() raw agent call without guardrails or semantic cache.

    Ingestion is local. The builders connect to the Qdrant collection
    already created and bootstrap it through local ingestion only when
    the collection does not exist.
"""

import logging

from langgraph.checkpoint.memory import InMemorySaver

from hr_assistant import config, thread_memory
from hr_assistant.agent import create_hr_agent, create_reliability_agent
from hr_assistant.guardrails import check_input, check_output
from hr_assistant.llm import get_llm
from hr_assistant.semantic_cache import SemanticCache
from hr_assistant.tools import create_guarded_search_tool, create_search_tool
from hr_assistant.vector_store import collection_exists, load_vector_store

logger = logging.getLogger(__name__)

INPUT_BLOCKED_MESSAGE = (
    "I can't process that request — it was flagged by "
    "the input safety guardrail."
)

OUTPUT_BLOCKED_MESSAGE = (
    "I can't share that answer as generated — it was flagged "
    "by the output safety guardrail."
)


# ── Builders ──────────────────────────────────────────────────────────────

def _bootstrap_collection(collection_name: str, ingest_fn) -> None:
    """Create a missing Qdrant collection using local ingestion."""

    if collection_exists(collection_name):
        logger.info(
            "Qdrant collection '%s' already exists.",
            collection_name,
        )
        return

    logger.info(
        "Collection '%s' not found. Starting local ingestion.",
        collection_name,
    )

    # Run local ingestion.
    ingest_fn()

    # Verify successful collection creation.
    if not collection_exists(collection_name):
        raise RuntimeError(
            f"Collection '{collection_name}' was not created "
            "successfully after local ingestion."
        )

    logger.info(
        "Collection '%s' created successfully.",
        collection_name,
    )


def _resolve_checkpointer(checkpointer):
    """Resolve the LangGraph checkpointer.

    None:
        Use InMemorySaver for CLI and Streamlit.

    False:
        Disable checkpointer for LangGraph Studio.

    Other:
        Use the supplied checkpointer object.
    """

    if checkpointer is None:
        return InMemorySaver()

    if checkpointer is False:
        return None

    return checkpointer


def _build_guarded_assistant(
    collection_name: str,
    ingest_fn,
    checkpointer=None,
):
    """Build the secure HR assistant.

    Returns:
        (agent, cache)
    """

    config.check_api_keys()

    # Bootstrap collection if missing.
    _bootstrap_collection(collection_name, ingest_fn)

    # Connect to Qdrant.
    vector_store = load_vector_store(collection_name)

    # Build guarded LangGraph agent.
    agent = create_reliability_agent(
        get_llm(),
        [create_guarded_search_tool(vector_store)],
        checkpointer=_resolve_checkpointer(checkpointer),
    )

    return agent, SemanticCache()


def build_hr_assistant(checkpointer=None):
    """Build the main HR assistant using the clean HR collection.

    Collection:
        hr_policies
    """

    from hr_assistant.ingestion import ingest_hr_policies

    return _build_guarded_assistant(
        config.QDRANT_COLLECTION_NAME,
        ingest_hr_policies,
        checkpointer=checkpointer,
    )


def build_reliability_assistant(checkpointer=None):
    """Build the reliability assistant using the noisy collection.

    Collection:
        hr_policies_noisy_demo
    """

    from hr_assistant.ingestion import ingest_noisy_corpus

    return _build_guarded_assistant(
        config.QDRANT_NOISY_COLLECTION_NAME,
        ingest_noisy_corpus,
        checkpointer=checkpointer,
    )


def build_plain_assistant(checkpointer=None):
    """Build an assistant without guardrails or semantic caching.

    Used for:
        - Red-team baseline testing
        - Plain answer-quality evaluation
    """

    from hr_assistant.ingestion import ingest_hr_policies

    config.check_api_keys()

    _bootstrap_collection(
        config.QDRANT_COLLECTION_NAME,
        ingest_hr_policies,
    )

    vector_store = load_vector_store(
        config.QDRANT_COLLECTION_NAME
    )

    return create_hr_agent(
        get_llm(),
        [create_search_tool(vector_store)],
        checkpointer=_resolve_checkpointer(checkpointer),
    )


# ── Asking ───────────────────────────────────────────────────────────────

def ask_plain(
    agent,
    question: str,
    thread_id: str = "default-session",
) -> str:
    """Invoke the raw agent without guardrails or semantic cache."""

    return thread_memory.invoke_agent(
        agent,
        question,
        thread_id,
    ).text


def ask(
    agent,
    cache,
    question: str,
    thread_id: str = "default-session",
) -> str:
    """Execute the complete guarded HR assistant workflow.

    Flow:
        1. Screen current question and recent user history.
        2. Apply input safety guardrail.
        3. Check semantic cache.
        4. Invoke LangGraph agent if there is no cache hit.
        5. Apply output safety guardrail.
        6. Store safe answers in semantic cache.

    Both generated and cached answers pass through the output guardrail.
    """

    # Step 1: Prepare input for safety screening.
    screen_text = thread_memory.input_text_for_screening(
        agent,
        question,
        thread_id,
    )

    # Step 2: Input guardrail.
    input_ok, input_reason = check_input(screen_text)

    logger.info(
        "INPUT GUARDRAIL: %s (%s)",
        "pass" if input_ok else "block",
        input_reason,
    )

    if not input_ok:
        return INPUT_BLOCKED_MESSAGE

    # Step 3: Semantic cache lookup.
    cached_answer = cache.lookup(question)

    if cached_answer is not None:
        logger.info("SEMANTIC CACHE: hit")

        # IMPORTANT:
        # Cached answers must also pass the output guardrail.
        output_ok, output_reason = check_output(cached_answer)

        logger.info(
            "CACHED ANSWER GUARDRAIL: %s (%s)",
            "pass" if output_ok else "block",
            output_reason,
        )

        if not output_ok:
            return OUTPUT_BLOCKED_MESSAGE

        # Record only safe cached answers in conversation memory.
        thread_memory.record_turn(
            agent,
            thread_id,
            question,
            cached_answer,
        )

        return cached_answer

    logger.info("SEMANTIC CACHE: miss")

    # Step 4: Invoke the agent.
    message = thread_memory.invoke_agent(
        agent,
        question,
        thread_id,
    )

    answer = message.text

    # Step 5: Output guardrail.
    output_ok, output_reason = check_output(answer)

    logger.info(
        "OUTPUT GUARDRAIL: %s (%s)",
        "pass" if output_ok else "block",
        output_reason,
    )

    if not output_ok:
        # Replace unsafe generated answer in conversation memory.
        thread_memory.overwrite_answer(
            agent,
            thread_id,
            message,
            OUTPUT_BLOCKED_MESSAGE,
        )

        return OUTPUT_BLOCKED_MESSAGE

    # Step 6: Store only safe answers in semantic cache.
    cache.store(question, answer)

    return answer


# ── Backwards compatibility ──────────────────────────────────────────────

ask_reliably = ask
