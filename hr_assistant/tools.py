
"""11 · tools — retrieval and reranking tools for the HR agents.

Two variants:

create_search_tool():
    Plain retrieval baseline for comparison and red-team testing.

create_guarded_search_tool():
    HR-category-filtered retrieval with Jina reranking
    and a relevance threshold.

Retrieval flow:
    Qdrant hybrid retrieval
        -> Jina reranking
        -> relevance filtering (guarded tool)
        -> source-attributed results
"""


from langchain.tools import tool

from hr_assistant import config
from hr_assistant.reranker import rerank, rerank_with_scores
from hr_assistant.vector_store import get_retriever


NOT_FOUND_SENTINEL = (
    "NOT_FOUND: no relevant HR policy content matched this question."
)


def _format_document(document, score=None):
    """Format a retrieved document with its source attribution."""

    source = document.metadata.get("source", "Unknown source")
    content = document.page_content

    if score is None:
        return (
            f"[Source: {source}]\n"
            f"{content}"
        )

    return (
        f"[Source: {source}] "
        f"(relevance: {score:.4f})\n"
        f"{content}"
    )


# ── Plain search tool ────────────────────────────────────────────────────

def create_search_tool(vector_store):
    """Create the unguarded search tool.

    Used for:
        - Plain baseline comparisons
        - Red-team experiments
        - Retrieval and answer-quality evaluation

    No HR category filtering or relevance threshold is applied.
    """

    @tool
    def search_hr_policy(question: str) -> str:
        """Search the HR policy documents for information about:

        - Leave
        - Work from home
        - Probation
        - Notice period
        - Reimbursement
        - Code of conduct
        - Holidays
        - Maternity and paternity leave
        - Travel expenses
        - Exit process
        """

        retriever = get_retriever(
            vector_store,
            k=config.RERANK_CANDIDATE_K,
        )

        candidates = retriever.invoke(question)

        if not candidates:
            return NOT_FOUND_SENTINEL

        top_chunks = rerank(
            question,
            candidates,
            top_n=config.TOP_K_RESULTS,
        )

        if not top_chunks:
            return NOT_FOUND_SENTINEL

        return "\n\n".join(
            _format_document(document)
            for document in top_chunks
        )

    return search_hr_policy


# ── Guarded search tool ──────────────────────────────────────────────────

def create_guarded_search_tool(vector_store):
    """Create the guarded HR search tool.

    Two retrieval restrictions:

    1. Category filtering:
       Only retrieve documents belonging to the configured
       HR_POLICY_CATEGORIES allow-list.

    2. Relevance threshold:
       The highest Jina reranking score must meet
       RELEVANCE_THRESHOLD.

    If either condition fails, return NOT_FOUND.
    """

    @tool
    def search_hr_policy(question: str) -> str:
        """Search only the company's HR policy documents.

        Supported topics:
        - Leave
        - Work from home
        - Probation
        - Notice period
        - Reimbursement
        - Code of conduct
        - Holidays
        - Maternity and paternity leave
        - Travel expenses
        - Exit process

        Do not use this tool for Finance, Sales, Operations,
        Business or other non-HR questions.
        """

        # Step 1: Retrieve only configured HR categories.
        retriever = get_retriever(
            vector_store,
            k=config.RERANK_CANDIDATE_K,
            filter_categories=config.HR_POLICY_CATEGORIES,
        )

        candidates = retriever.invoke(question)

        # Step 2: Handle empty retrieval results.
        if not candidates:
            return NOT_FOUND_SENTINEL

        # Step 3: Rerank candidates using Jina.
        ranked = rerank_with_scores(
            question,
            candidates,
            top_n=config.TOP_K_RESULTS,
        )

        # Step 4: Handle empty reranking results.
        if not ranked:
            return NOT_FOUND_SENTINEL

        # Step 5: Apply the relevance threshold.
        best_score = ranked[0][1]

        if best_score < config.RELEVANCE_THRESHOLD:
            return NOT_FOUND_SENTINEL

        # Step 6: Format the results with source attribution.
        return "\n\n".join(
            _format_document(document, score)
            for document, score in ranked
        )

    return search_hr_policy
