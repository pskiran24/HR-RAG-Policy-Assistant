
"""Local ingestion pipeline for the HR RAG application.

Local data files
    -> parse PDF/DOCX/PPTX once into processed/noise/
    -> load documents
    -> split into chunks
    -> generate embeddings
    -> store in Qdrant hybrid collections

The clean collection contains HR policies only.
The noisy collection contains HR policies and additional noise documents.
"""

import logging

from hr_assistant import config
from hr_assistant.document_loader import (
    load_documents_from_local,
    load_all_local_documents,
    load_processed_documents_from_local,
)
from hr_assistant.processor import process_raw_to_json
from hr_assistant.splitter import split_into_chunks
from hr_assistant.vector_store import (
    build_vector_store,
    collection_exists,
)

logger = logging.getLogger(__name__)


def ingest_hr_policies(force: bool = False) -> None:
    """Build the clean HR-only Qdrant collection."""

    name = config.QDRANT_COLLECTION_NAME

    if collection_exists(name) and not force:
        logger.info(
            "Collection '%s' already populated. "
            "Skipping ingestion (use --force to rebuild).",
            name,
        )
        return

    documents = load_documents_from_local()

    if not documents:
        raise RuntimeError(
            "No HR policy documents found. Check the data/ directory."
        )

    chunks = split_into_chunks(documents)

    if not chunks:
        raise RuntimeError("No chunks generated from HR policy documents.")

    build_vector_store(
        chunks,
        hybrid=True,
        collection_name=name,
    )

    logger.info(
        "Ingested %d HR document(s) -> %d chunk(s) into '%s'.",
        len(documents),
        len(chunks),
        name,
    )


def ingest_noisy_corpus(force: bool = False) -> None:
    """Build the mixed HR + noise Qdrant collection."""

    name = config.QDRANT_NOISY_COLLECTION_NAME

    if collection_exists(name) and not force:
        logger.info(
            "Collection '%s' already populated. "
            "Skipping ingestion (use --force to rebuild).",
            name,
        )
        return

    # Parse binary documents when rebuilding or when no cached
    # processed documents are available.
    processed_documents = load_processed_documents_from_local()

    if force or not processed_documents:
        logger.info("Processing local PDF, DOCX and PPTX files...")
        process_raw_to_json()

    # Load HR policies, noise TXT files and processed binary documents.
    documents = load_all_local_documents()

    if not documents:
        raise RuntimeError(
            "No documents found for the noisy collection. "
            "Check data/ and processed/noise/."
        )

    chunks = split_into_chunks(documents)

    if not chunks:
        raise RuntimeError(
            "No chunks generated for the noisy collection."
        )

    build_vector_store(
        chunks,
        hybrid=True,
        collection_name=name,
    )

    logger.info(
        "Ingested %d document(s) -> %d chunk(s) into '%s'.",
        len(documents),
        len(chunks),
        name,
    )


def run_ingestion(
    force: bool = False,
    hr: bool = True,
    noisy: bool = True,
    upload: bool | None = None,
) -> None:
    """Run the selected local ingestion jobs.

    The upload parameter is retained for compatibility with the existing
    CLI, but is no longer used because ingestion is local.
    """

    config.check_api_keys()

    if upload:
        logger.info(
            "The upload option is ignored: local ingestion does not use GCS."
        )

    if hr:
        ingest_hr_policies(force=force)

    if noisy:
        ingest_noisy_corpus(force=force)
