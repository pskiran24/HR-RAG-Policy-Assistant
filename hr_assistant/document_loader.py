
"""Local document loader for the HR RAG application."""

import json
from pathlib import Path

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
HR_DATA_DIR = DATA_DIR
NOISE_DATA_DIR = DATA_DIR / "noise"

PROCESSED_NOISE_DIR = PROJECT_ROOT / "processed" / "noise"


def extract_policy_category(text: str) -> str:
    """Extract the policy category from the beginning of a document."""

    for line in text.splitlines()[:5]:
        stripped = line.strip().lower()

        if stripped.startswith("policy category:"):
            return line.split(":", 1)[1].strip()

        if stripped.startswith("category:"):
            return line.split(":", 1)[1].strip()

    return "Unknown"


def load_local_text_documents(directory: Path) -> list[Document]:
    """Load all TXT documents from a directory."""

    documents = []

    if not directory.exists():
        return documents

    for path in sorted(directory.glob("*.txt")):
        if not path.is_file():
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "policy_category": extract_policy_category(text),
                    "file_type": path.suffix.lower(),
                    "local_path": str(path),
                },
            )
        )

    return documents


def load_documents_from_local() -> list[Document]:
    """Load HR policy TXT files from data/."""

    return load_local_text_documents(HR_DATA_DIR)


def load_processed_documents_from_local() -> list[Document]:
    """Load previously processed PDF, DOCX and PPTX JSON records."""

    documents = []

    if not PROCESSED_NOISE_DIR.exists():
        return documents

    for json_path in sorted(PROCESSED_NOISE_DIR.glob("*.json")):
        try:
            record = json.loads(
                json_path.read_text(encoding="utf-8")
            )

            text = record.get("text", "")

            if not text.strip():
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": record.get("source", json_path.name),
                        "policy_category": record.get(
                            "policy_category",
                            extract_policy_category(text),
                        ),
                        "file_type": record.get(
                            "file_type",
                            Path(record.get("source", "")).suffix.lower(),
                        ),
                        "local_path": record.get(
                            "local_path",
                            str(json_path),
                        ),
                    },
                )
            )

        except (json.JSONDecodeError, OSError, TypeError) as exc:
            print(f"Could not load {json_path.name}: {exc}")

    return documents


def load_all_local_documents() -> list[Document]:
    """Load HR TXT files, noise TXT files and processed binary documents."""

    documents = []

    # HR policy TXT files
    documents.extend(load_documents_from_local())

    # Noise TXT files
    documents.extend(
        load_local_text_documents(NOISE_DATA_DIR)
    )

    # Previously processed PDF, DOCX and PPTX files
    documents.extend(
        load_processed_documents_from_local()
    )

    return documents


# Backward-compatible function names for existing modules.
def load_documents_from_gcs(*args, **kwargs):
    return load_documents_from_local()


def load_processed_documents_from_gcs(*args, **kwargs):
    return load_all_local_documents()
