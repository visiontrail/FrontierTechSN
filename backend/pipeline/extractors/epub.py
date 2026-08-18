import logging
from collections.abc import Callable
from pathlib import Path
from ebooklib import epub
from bs4 import BeautifulSoup
from backend.pipeline.extractors.base import ExtractedContent

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


async def extract_epub(filepath: str, log: LogCallback | None = None) -> ExtractedContent:
    emit = lambda message: log(message) if log else logger.info(message)
    emit(f"Extracting EPUB from {filepath}")
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"EPUB file not found: {filepath}")

    book = epub.read_epub(str(path))

    title = book.get_metadata("DC", "title")
    title = title[0][0] if title else path.stem

    author = book.get_metadata("DC", "creator")
    author = author[0][0] if author else "Unknown"

    chapters = []
    for item in book.get_items_of_type(9):  # ITEM_DOCUMENT
        html = item.get_content().decode("utf-8", errors="ignore")
        text = _html_to_text(html)
        if len(text.split()) > 20:
            chapters.append(text)

    if not chapters:
        raise RuntimeError("No readable chapters found in EPUB")

    full_text = "\n\n---\n\n".join(chapters)
    emit(f"Extracted {len(chapters)} chapters, {len(full_text.split())} words from '{title}'")

    return ExtractedContent(
        source_type="epub",
        title=title,
        text=full_text,
        metadata={"author": author, "chapter_count": len(chapters), "filepath": filepath},
    )
