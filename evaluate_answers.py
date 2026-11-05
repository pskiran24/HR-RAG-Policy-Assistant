
"""
End-to-end Answer Evaluation for the HR RAG application.

Features:
- Incremental result saving after every question.
- Resume previously completed questions.
- Retry questions with status "error".
- Evaluate clean and noisy collections independently.
- Citation precision, recall, and match rate.
"""

import json
import logging
import re

from pathlib import Path
from datetime import datetime, timezone

from hr_assistant.pipeline import (
    build_hr_assistant,
    build_reliability_assistant,
    ask,
    INPUT_BLOCKED_MESSAGE,
    OUTPUT_BLOCKED_MESSAGE,
)

from hr_assistant.guardrails import GuardrailProviderError


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_ROOT / "evaluation_dataset.json"
OUTPUT_PATH = PROJECT_ROOT / "answer_evaluation_results.json"

# Start with three questions. Change to 27 after testing.
QUESTION_LIMIT = 3

# Select which collections to evaluate.
EVALUATE_CLEAN = True
EVALUATE_NOISY = True

# Resume completed questions from the previous run.
RESUME_EXISTING_RESULTS = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ==================================================
# LOAD DATASET
# ==================================================

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("Evaluation dataset must be a JSON list.")

    required_fields = {
        "id",
        "category",
        "query",
        "is_in_scope",
        "relevant_sources",
        "ground_truth",
    }

    for item in dataset:
        missing = required_fields - set(item.keys())

        if missing:
            raise ValueError(
                f"Question {item.get('id')} is missing: {missing}"
            )

    return dataset


# ==================================================
# CITATION EXTRACTION
# ==================================================

def extract_citations(answer):

    if not answer:
        return []

    pattern = r"(?i)\b[a-z0-9][a-z0-9_-]*\.txt\b"

    citations = re.findall(pattern, answer)

    return list(
        dict.fromkeys(
            citation.lower()
            for citation in citations
        )
    )


# ==================================================
# RESPONSE CLASSIFICATION
# ==================================================

def classify_response(answer):

    if answer == INPUT_BLOCKED_MESSAGE:
        return "blocked", "Input safety guardrail blocked the request."

    if answer == OUTPUT_BLOCKED_MESSAGE:
        return "blocked", "Output safety guardrail blocked the answer."

    if not answer:
        return "error", "Empty response returned by the pipeline."

    return "success", None


# ==================================================
# CITATION METRICS
# ==================================================

def calculate_citation_metrics(expected_sources, actual_citations):

    expected = {
        source.lower()
        for source in expected_sources
    }

    actual = {
        source.lower()
        for source in actual_citations
    }

    correct = expected.intersection(actual)

    precision = (
        len(correct) / len(actual)
        if actual else 0.0
    )

    recall = (
        len(correct) / len(expected)
        if expected else None
    )

    match = (
        bool(correct)
        if expected else None
    )

    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_match": match,
    }


# ==================================================
# RESULT FILE MANAGEMENT
# ==================================================

def create_empty_output(questions):

    return {
        "schema_version": 2,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "end_to_end_answer_generation",
        "question_limit": QUESTION_LIMIT,
        "total_dataset_questions": None,
        "evaluated_questions": len(questions),
        "question_ids": [item["id"] for item in questions],
        "clean_collection": [],
        "noisy_collection": [],
    }


def load_existing_output(questions, total_dataset_questions):

    if not RESUME_EXISTING_RESULTS or not OUTPUT_PATH.exists():
        output = create_empty_output(questions)
        output["total_dataset_questions"] = total_dataset_questions
        return output

    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as file:
            existing = json.load(file)

        expected_ids = [item["id"] for item in questions]

        # Only resume files created by this updated evaluator
        # with the same selected question set.
        if (
            existing.get("schema_version") != 2
            or existing.get("question_ids") != expected_ids
        ):
            logger.info(
                "Existing results are incompatible. Starting a new evaluation."
            )

            output = create_empty_output(questions)
            output["total_dataset_questions"] = total_dataset_questions
            return output

        logger.info("Existing evaluation results found. Resuming.")

        existing["total_dataset_questions"] = total_dataset_questions

        return existing

    except (json.JSONDecodeError, OSError, TypeError):
        logger.warning(
            "Could not read existing results. Starting a new evaluation."
        )

        output = create_empty_output(questions)
        output["total_dataset_questions"] = total_dataset_questions
        return output


def save_results(output):

    output["last_saved_at"] = datetime.now(timezone.utc).isoformat()

    temporary_path = OUTPUT_PATH.with_suffix(".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # Replace the previous result file only after the new file is complete.
    temporary_path.replace(OUTPUT_PATH)

    logger.info("Checkpoint saved: %s", OUTPUT_PATH.name)


def get_completed_ids(results):

    return {
        str(result["id"])
        for result in results
        if result.get("status") in {"success", "blocked"}
    }


# ==================================================
# EVALUATE ONE COLLECTION
# ==================================================

def evaluate_collection(
    collection_name,
    assistant_builder,
    questions,
    output,
    output_key,
):

    existing_results = output.get(output_key, [])

    completed_ids = get_completed_ids(existing_results)

    pending_questions = [
        item
        for item in questions
        if str(item["id"]) not in completed_ids
    ]

    if not pending_questions:

        logger.info(
            "All questions already completed for %s. Skipping.",
            collection_name,
        )

        return

    logger.info("Building assistant: %s", collection_name)

    agent, cache = assistant_builder()

    for index, item in enumerate(pending_questions, start=1):

        question_id = item["id"]
        query = item["query"]

        thread_id = (
            f"answer-eval-{collection_name}-q{question_id}"
        )

        logger.info(
            "[%s/%s] Question %s: %s",
            index,
            len(pending_questions),
            question_id,
            query,
        )

        result = {
            "id": question_id,
            "category": item["category"],
            "query": query,
            "is_in_scope": item["is_in_scope"],
            "ground_truth": item["ground_truth"],
            "expected_sources": item["relevant_sources"],
            "collection": collection_name,
            "thread_id": thread_id,
            "answer": None,
            "actual_citations": [],
            "citation_precision": None,
            "citation_recall": None,
            "citation_match": None,
            "status": None,
            "error": None,
        }

        try:

            answer = ask(
                agent,
                cache,
                query,
                thread_id=thread_id,
            )

            status, reason = classify_response(answer)

            result["answer"] = answer
            result["status"] = status

            if status == "blocked":

                result["error"] = reason

                logger.warning(
                    "Question %s genuinely blocked: %s",
                    question_id,
                    reason,
                )

            elif status == "error":

                result["error"] = reason

                logger.error(
                    "Question %s returned an empty response.",
                    question_id,
                )

            else:

                citations = extract_citations(answer)

                metrics = calculate_citation_metrics(
                    item["relevant_sources"],
                    citations,
                )

                result.update({
                    "actual_citations": citations,
                    **metrics,
                })

                logger.info(
                    "Answer generated. Citations: %s",
                    citations,
                )

        except GuardrailProviderError as exc:

            result["status"] = "error"
            result["error"] = str(exc)

            logger.error(
                "Guardrail provider unavailable for question %s: %s",
                question_id,
                exc,
            )

        except Exception as exc:

            result["status"] = "error"

            result["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            logger.exception(
                "Evaluation failed for question %s",
                question_id,
            )

        # Replace any previous error result for this question.
        existing_results = [
            old_result
            for old_result in existing_results
            if str(old_result["id"]) != str(question_id)
        ]

        existing_results.append(result)

        output[output_key] = existing_results

        # Save immediately after every question.
        save_results(output)


# ==================================================
# SUMMARY
# ==================================================

def print_summary(collection_name, results):

    successful = [
        result for result in results
        if result["status"] == "success"
    ]

    blocked = [
        result for result in results
        if result["status"] == "blocked"
    ]

    errors = [
        result for result in results
        if result["status"] == "error"
    ]

    print("\n" + "=" * 65)
    print(collection_name.upper())
    print("=" * 65)

    print("Total questions:", len(results))
    print("Successful:", len(successful))
    print("Blocked:", len(blocked))
    print("Errors:", len(errors))

    if successful:

        precision = sum(
            result["citation_precision"]
            for result in successful
        ) / len(successful)

        recall_values = [
            result["citation_recall"]
            for result in successful
            if result["citation_recall"] is not None
        ]

        recall = (
            sum(recall_values) / len(recall_values)
            if recall_values else None
        )

        match_values = [
            result["citation_match"]
            for result in successful
            if result["citation_match"] is not None
        ]

        match_rate = (
            sum(match_values) / len(match_values)
            if match_values else None
        )

        print("\nCitation metrics (successful answers only):")

        print("Average Citation Precision:", round(precision, 4))

        print(
            "Average Citation Recall:",
            round(recall, 4) if recall is not None else "N/A",
        )

        print(
            "Citation Match Rate:",
            round(match_rate, 4) if match_rate is not None else "N/A",
        )

    else:

        print("\nNo successful answers available for citation metrics.")

    print("\nIndividual results:")

    for result in results:

        print(
            f"\nQ{result['id']} - {result['status'].upper()}"
        )

        print("Question:", result["query"])
        print("Answer:", result["answer"])
        print("Citations:", result["actual_citations"])
        print("Error:", result["error"])


# ==================================================
# MAIN
# ==================================================

def main():

    if not EVALUATE_CLEAN and not EVALUATE_NOISY:
        raise ValueError(
            "Enable at least one collection for evaluation."
        )

    dataset = load_dataset()

    logger.info("Total dataset questions: %s", len(dataset))

    questions = [
        item
        for item in dataset
        if item["is_in_scope"]
    ][:QUESTION_LIMIT]

    if not questions:
        raise ValueError("No in-scope questions found.")

    logger.info("Questions selected: %s", len(questions))

    output = load_existing_output(
        questions,
        total_dataset_questions=len(dataset),
    )

    if EVALUATE_CLEAN:

        evaluate_collection(
            collection_name="hr_policies",
            assistant_builder=build_hr_assistant,
            questions=questions,
            output=output,
            output_key="clean_collection",
        )

    if EVALUATE_NOISY:

        evaluate_collection(
            collection_name="hr_policies_noisy_demo",
            assistant_builder=build_reliability_assistant,
            questions=questions,
            output=output,
            output_key="noisy_collection",
        )

    save_results(output)

    if EVALUATE_CLEAN:
        print_summary(
            "Clean Collection",
            output["clean_collection"],
        )

    if EVALUATE_NOISY:
        print_summary(
            "Noisy Collection",
            output["noisy_collection"],
        )

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)

    print("\nResults saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
