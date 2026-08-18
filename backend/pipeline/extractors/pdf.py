import logging
from collections.abc import Callable
from pathlib import Path
import pdfplumber
from backend.pipeline.extractors.base import ExtractedContent

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]


async def extract_pdf(filepath: str, log: LogCallback | None = None) -> ExtractedContent:
    emit = lambda message: log(message) if log else logger.info(message)
    emit(f"Extracting PDF from {filepath}")
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())

    if not pages:
        raise RuntimeError("No readable text found in PDF")

    full_text = "\n\n".join(pages)
    title = path.stem.replace("_", " ").replace("-", " ").title()

    emit(f"Extracted {len(pages)} pages, {len(full_text.split())} words from '{title}'")
    return ExtractedContent(
        source_type="pdf",
        title=title,
        text=full_text,
        metadata={"page_count": len(pages), "filepath": filepath},
    )
