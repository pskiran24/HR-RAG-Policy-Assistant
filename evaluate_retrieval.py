
"""Evaluate HR RAG retrieval before and after Jina reranking."""

import json
from pathlib import Path
from statistics import mean

from hr_assistant import config
from hr_assistant.vector_store import load_vector_store, get_retriever
from hr_assistant.reranker import rerank_with_scores


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "evaluation_dataset.json"
OUTPUT_PATH = PROJECT_ROOT / "retrieval_evaluation_results.json"

K = 5
CANDIDATE_K = 12


def unique_sources(documents):
    """Preserve ranking while keeping only the first chunk per source."""

    sources = []
    seen = set()

    for doc in documents:
        source = doc.metadata.get("source", "unknown")

        if source not in seen:
            sources.append(source)
            seen.add(source)

    return sources


def calculate_metrics(ranked_sources, relevant_sources):
    """Calculate source-level retrieval metrics at K."""

    top_k = ranked_sources[:K]
    relevant = set(relevant_sources)

    relevant_count = sum(
        1 for source in top_k if source in relevant
    )

    recall = (
        relevant_count / len(relevant)
        if relevant else 0.0
    )

    # Standard fixed-K Precision@K.
    precision = relevant_count / K

    reciprocal_rank = 0.0

    for rank, source in enumerate(top_k, start=1):
        if source in relevant:
            reciprocal_rank = 1 / rank
            break

    hit = 1 if relevant_count > 0 else 0

    return {
        f"recall@{K}": recall,
        f"precision@{K}": precision,
        "mrr": reciprocal_rank,
        f"hit_rate@{K}": hit,
    }


def average_metrics(results):
    """Calculate mean metrics across evaluation questions."""

    if not results:
        raise ValueError("No evaluation results available.")

    return {
        metric: mean(result[metric] for result in results)
        for metric in results[0]
    }


def evaluate_collection(collection_name, questions):
    """Evaluate one Qdrant collection."""

    print(f"\n{'=' * 70}")
    print(f"COLLECTION: {collection_name}")
    print("=" * 70)

    store = load_vector_store(
        collection_name=collection_name,
        hybrid=True,
    )

    retriever = get_retriever(
        store,
        k=CANDIDATE_K,
    )

    before_results = []
    after_results = []
    question_results = []

    for item in questions:
        query = item["query"]
        relevant_sources = item["relevant_sources"]

        if not isinstance(relevant_sources, list):
            raise ValueError(
                f"Question {item['id']}: relevant_sources must be a list."
            )

        print(f"\nQuestion {item['id']}: {query}")

        # Stage 1: Initial hybrid retrieval
        candidates = retriever.invoke(query)

        before_sources = unique_sources(candidates)

        before_metrics = calculate_metrics(
            before_sources,
            relevant_sources,
        )

        # Stage 2: Jina reranking
        reranked = rerank_with_scores(
            query,
            candidates,
            top_n=CANDIDATE_K,
        )

        reranked_docs = [
            doc for doc, score in reranked
        ]

        after_sources = unique_sources(reranked_docs)

        after_metrics = calculate_metrics(
            after_sources,
            relevant_sources,
        )

        before_results.append(before_metrics)
        after_results.append(after_metrics)

        question_results.append({
            "question_id": item["id"],
            "query": query,
            "relevant_sources": relevant_sources,

            "before_reranking": {
                "ranked_sources": before_sources[:K],
                "metrics": before_metrics,
            },

            "after_reranking": {
                "ranked_sources": after_sources[:K],
                "metrics": after_metrics,
            },
        })

        print("Expected sources:", relevant_sources)
        print("Before reranking:", before_sources[:K])
        print("After reranking:", after_sources[:K])

        print(
            f"Recall@{K}: "
            f"{before_metrics[f'recall@{K}']:.2f} -> "
            f"{after_metrics[f'recall@{K}']:.2f}"
        )

        print(
            f"Precision@{K}: "
            f"{before_metrics[f'precision@{K}']:.2f} -> "
            f"{after_metrics[f'precision@{K}']:.2f}"
        )

        print(
            f"MRR: "
            f"{before_metrics['mrr']:.2f} -> "
            f"{after_metrics['mrr']:.2f}"
        )

    before_average = average_metrics(before_results)
    after_average = average_metrics(after_results)

    print(f"\n{'-' * 70}")
    print("AGGREGATE RESULTS")
    print("-" * 70)

    print(f"{'Metric':<18}{'Before':>12}{'After':>12}")

    for metric in before_average:
        print(
            f"{metric:<18}"
            f"{before_average[metric]:>12.4f}"
            f"{after_average[metric]:>12.4f}"
        )

    return {
        "collection": collection_name,
        "number_of_questions": len(questions),
        "before_reranking": before_average,
        "after_reranking": after_average,
        "question_details": question_results,
    }


def main():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    questions = [
        question
        for question in questions
        if question.get("is_in_scope", True)
    ]

    if not questions:
        raise ValueError("No in-scope evaluation questions found.")

    results = []

    collections = [
        config.QDRANT_COLLECTION_NAME,
        config.QDRANT_NOISY_COLLECTION_NAME,
    ]

    for collection_name in collections:
        results.append(
            evaluate_collection(collection_name, questions)
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nDetailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
