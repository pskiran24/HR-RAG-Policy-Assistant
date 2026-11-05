
"""Local document processor for the HR RAG application.

Parses PDF, DOCX and PPTX files from data/noise/ into plain text
and stores the results as JSON in processed/noise/.

TXT files are handled directly by document_loader.py.
"""

import io
import json
import logging
from pathlib import Path

from docx import Document as DocxReader
from pptx import Presentation
from pypdf import PdfReader

from hr_assistant.document_loader import extract_policy_category

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NOISE_DIR = DATA_DIR / "noise"
PROCESSED_DIR = PROJECT_ROOT / "processed"
PROCESSED_NOISE_DIR = PROCESSED_DIR / "noise"


def _parse_pdf(raw_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw_bytes))
    return "\n".join(
        page.extract_text() or "" for page in reader.pages
    )


def _parse_docx(raw_bytes: bytes) -> str:
    doc = DocxReader(io.BytesIO(raw_bytes))
    return "\n".join(
        paragraph.text for paragraph in doc.paragraphs
    )


def _parse_pptx(raw_bytes: bytes) -> str:
    presentation = Presentation(io.BytesIO(raw_bytes))
    lines = []

    for slide in presentation.slides:
        if slide.shapes.title is not None:
            lines.append(slide.shapes.title.text)

        for shape in slide.shapes:
            if shape.has_text_frame and shape != slide.shapes.title:
                lines.append(shape.text_frame.text)

    return "\n".join(lines)


_PARSERS = {
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".pptx": _parse_pptx,
}


def parse_local_file(file_path: Path) -> str:
    """Read a local file and return its extracted plain text."""

    extension = file_path.suffix.lower()

    if extension == ".txt":
        return file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    parser = _PARSERS.get(extension)

    if parser is None:
        raise ValueError(
            f"No parser registered for extension: {extension}"
        )

    raw_bytes = file_path.read_bytes()
    return parser(raw_bytes)


def process_raw_to_json() -> int:
    """Parse supported files from data/noise/ and save JSON records.

    Returns the number of successfully processed files.
    """

    if not NOISE_DIR.exists():
        logger.warning("Noise directory does not exist: %s", NOISE_DIR)
        return 0

    PROCESSED_NOISE_DIR.mkdir(parents=True, exist_ok=True)

    count = 0

    supported_extensions = {".pdf", ".docx", ".pptx"}

    for file_path in sorted(NOISE_DIR.iterdir()):

        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        # TXT files are loaded directly by document_loader.py.
        if extension == ".txt":
            continue

        if extension not in supported_extensions:
            logger.info("Skipping unsupported file: %s", file_path.name)
            continue

        try:
            extracted_text = parse_local_file(file_path)

            if not extracted_text.strip():
                logger.warning(
                    "No text extracted from: %s",
                    file_path.name,
                )
                continue

            record = {
                "source": file_path.name,
                "policy_category": extract_policy_category(extracted_text),
                "text": extracted_text,
                "local_path": str(file_path),
                "file_type": extension,
            }

            output_name = file_path.stem + ".json"
            output_path = PROCESSED_NOISE_DIR / output_name

            output_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            logger.info(
                "Processed %s -> %s",
                file_path.name,
                output_path,
            )

            count += 1

        except Exception:
            logger.exception(
                "Failed to process file: %s",
                file_path.name,
            )

    logger.info("Successfully processed %d files.", count)

    return count
