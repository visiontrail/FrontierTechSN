import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend import config
from backend.pipeline.extractors.base import ExtractedContent
from backend.pipeline.extractors.epub import extract_epub
from backend.pipeline.process_logging import stream_subprocess

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

# The Isla-Reader promotion step shells out to a Swift app plus an AI selection
# pass. It previously had no timeout and could hang the pipeline indefinitely;
# cap it generously and fall back to plain EPUB extraction if it overruns.
CURATED_TIMEOUT = 1800


def _promotion_script_available() -> tuple[bool, str]:
    script = config.ISLA_READER_PROMOTION_SCRIPT
    if not script.exists():
        return False, f"Isla-Reader promotion script not found: {script}"
    if not script.is_file():
        return False, f"Isla-Reader promotion script is not a file: {script}"
    if shutil.which("swift") is None:
        return False, "Swift CLI is not available on PATH"
    return True, ""


def _find_selected_stage2(output_root: Path) -> Path:
    direct = output_root / "selected.stage2.json"
    if direct.exists():
        return direct

    matches = sorted(output_root.rglob("selected.stage2.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]

    raise FileNotFoundError(f"selected.stage2.json was not produced under {output_root}")


def _highlight_text(item: dict[str, Any]) -> str:
    return str(item.get("highlightText") or item.get("highlight_text") or item.get("text") or "").strip()


def _format_curated_highlights(selected: list[dict[str, Any]]) -> str:
    lines = ["Curated book highlights selected by Isla-Reader:"]
    for i, item in enumerate(selected, start=1):
        chapter = str(item.get("chapterTitle") or item.get("chapter_title") or "Unknown chapter").strip()
        highlight = _highlight_text(item)
        note = str(item.get("noteText") or item.get("note_text") or "").strip()
        reason = str(item.get("selectionReason") or item.get("selection_reason") or "").strip()
        title = str(item.get("postTitle") or item.get("post_title") or "").strip()
        description = str(item.get("postDescription") or item.get("post_description") or "").strip()

        parts = [f"{i}. Chapter: {chapter}", f"Quote: {highlight}"]
        if note:
            parts.append(f"Reader note: {note}")
        if reason:
            parts.append(f"Why it matters: {reason}")
        if title:
            parts.append(f"Angle: {title}")
        if description:
            parts.append(f"Discussion cue: {description}")
        lines.append("\n".join(parts))

    return "\n\n".join(lines)


async def extract_epub_curated(
    filepath: str,
    output_root: str | None = None,
    log: LogCallback | None = None,
) -> ExtractedContent:
    emit = lambda message: log(message) if log else logger.info(message)

    def warn(message: str):
        logger.warning(message)
        if log:
            log(message)

    available, reason = _promotion_script_available()
    if not available:
        warn(f"{reason}; falling back to standard EPUB extraction")
        fallback = await extract_epub(filepath)
        fallback.metadata["processing_mode"] = "full_text"
        fallback.metadata["curated_fallback_reason"] = reason
        return fallback

    epub_path = Path(filepath)
    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB file not found: {filepath}")

    output_dir = Path(output_root) if output_root else config.OUTPUTS_DIR / "isla-reader" / epub_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    script = config.ISLA_READER_PROMOTION_SCRIPT
    command = [
        str(script),
        "--epub",
        str(epub_path),
        "--output",
        str(output_dir),
        "--style",
        "none",
    ]
    emit("Running Isla-Reader curated highlights: %s" % " ".join(command))

    try:
        returncode, output = await stream_subprocess(
            name="Isla-Reader curated highlights",
            command=command,
            logger=logger,
            log=log,
            cwd=script.parent.parent,
            timeout=CURATED_TIMEOUT,
        )
    except TimeoutError as exc:
        warn(f"Isla-Reader promotion timed out; falling back to EPUB extraction: {exc}")
        fallback = await extract_epub(filepath)
        fallback.metadata["processing_mode"] = "full_text"
        fallback.metadata["curated_fallback_reason"] = str(exc)
        return fallback

    if returncode != 0:
        message = output[-500:].strip()
        warn(f"Isla-Reader promotion failed (exit {returncode}); falling back to EPUB extraction: {message}")
        fallback = await extract_epub(filepath)
        fallback.metadata["processing_mode"] = "full_text"
        fallback.metadata["curated_fallback_reason"] = message or "Isla-Reader promotion failed"
        return fallback

    selected_path = _find_selected_stage2(output_dir)
    data = json.loads(selected_path.read_text())
    selected = data.get("selected") if isinstance(data, dict) else data
    if not isinstance(selected, list) or not selected:
        raise RuntimeError(f"No curated highlights found in {selected_path}")

    highlights = [item for item in selected if isinstance(item, dict) and _highlight_text(item)]
    if not highlights:
        raise RuntimeError(f"Curated highlights in {selected_path} contain no readable text")

    title = epub_path.stem
    manifest_path = selected_path.parent / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            source = manifest.get("source") if isinstance(manifest, dict) else None
            if isinstance(source, dict):
                title = source.get("title") or title
        except json.JSONDecodeError:
            logger.warning("Unable to parse Isla-Reader manifest at %s", manifest_path)

    text = _format_curated_highlights(highlights)
    emit(f"Loaded {len(highlights)} curated highlights from {selected_path}")

    return ExtractedContent(
        source_type="epub",
        title=title,
        text=text,
        metadata={
            "filepath": filepath,
            "processing_mode": "curated_highlights",
            "curated_highlights": highlights,
            "selected_stage2_path": str(selected_path),
            "curated_output_dir": str(selected_path.parent),
        },
    )
