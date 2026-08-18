"""Script-driven viral thumbnail generation through ChatGPT Web + OpenCLI.

The prompt formula is deliberately stored in ``backend/prompts`` and loaded at
call time, so an Admin -> Prompts edit affects the next task without a restart.
The generated prompt, image, and provenance manifest stay together under the
task output directory.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import config
from backend.pipeline.digester import _chat, _resolve_provider
from backend.pipeline.opencli import OpenCLIError, first_json, run_opencli

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_IMAGE_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
)


@dataclass(frozen=True)
class ThumbnailArtifact:
    prompt_path: str
    image_path: str
    manifest_path: str
    conversation_url: str = ""


def _log(log: LogCallback | None, message: str) -> None:
    if log is not None:
        log(message)
    else:
        logger.info(message)


def _clean_model_prompt(value: str) -> str:
    clean = value.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if not clean:
        raise RuntimeError("Thumbnail art director returned an empty image prompt")
    return clean


def _rows(value: Any) -> list[dict[str, Any]]:
    """Normalize common OpenCLI JSON envelopes to a list of result rows."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("data", "items", "results", "rows"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [value]
    return []


def _field(row: dict[str, Any], name: str) -> str:
    for key, value in row.items():
        if str(key).lower() == name.lower():
            text = str(value or "").strip()
            for prefix in ("📁", "🔗"):
                if text.startswith(prefix):
                    text = text[len(prefix) :].strip()
            return text
    return ""


def _looks_like_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 32:
        return False
    with path.open("rb") as handle:
        header = handle.read(12)
    return header.startswith(_IMAGE_SIGNATURES) or (
        header.startswith(b"RIFF") and header[8:12] == b"WEBP"
    )


def _new_image_files(image_dir: Path, before: set[Path]) -> list[Path]:
    candidates = [
        path.resolve()
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in _IMAGE_SUFFIXES
        and path.resolve() not in before
        and _looks_like_image(path)
    ]
    return sorted(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _reported_image_files(rows: list[dict[str, Any]], image_dir: Path) -> list[Path]:
    root = image_dir.resolve()
    files: list[Path] = []
    for row in rows:
        raw = _field(row, "file")
        if not raw or raw == "-":
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = config.PROJECT_ROOT / path
        resolved = path.resolve()
        if (resolved == root or root in resolved.parents) and _looks_like_image(resolved):
            files.append(resolved)
    return files


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


async def generate_thumbnail_prompt(
    *,
    title: str,
    script: str,
    provider_id: int | None = None,
    ai_endpoint: str | None = None,
    ai_model: str | None = None,
    log: LogCallback | None = None,
) -> str:
    """Turn the final narration script into one complete image-generation prompt."""
    template = (config.PROMPTS_DIR / "thumbnail.txt").read_text(encoding="utf-8")
    endpoint, model, api_key = await _resolve_provider(provider_id, ai_endpoint, ai_model)
    source = json.dumps(
        {"video_title": title, "audio_script": script},
        indent=2,
        ensure_ascii=False,
    )
    result = await _chat(
        template,
        source,
        endpoint,
        model,
        api_key,
        log,
        "Thumbnail art direction",
        max_tokens=3072,
    )
    return _clean_model_prompt(result)


async def generate_thumbnail(
    *,
    task_id: str,
    task_dir: Path,
    title: str,
    script_path: Path,
    provider_id: int | None = None,
    ai_endpoint: str | None = None,
    ai_model: str | None = None,
    log: LogCallback | None = None,
) -> ThumbnailArtifact:
    """Generate, download, and provenance-log a task thumbnail without TTS."""
    thumbnail_dir = task_dir / "thumbnail"
    image_dir = thumbnail_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = thumbnail_dir / "manifest.json"
    prompt_path = thumbnail_dir / "prompt.txt"
    started_at = datetime.now(timezone.utc).isoformat()

    manifest: dict[str, Any] = {
        "task_id": task_id,
        "status": "generating",
        "provider": "chatgpt_web",
        "source_script": str(script_path),
        "prompt_path": str(prompt_path),
        "image_path": "",
        "image_paths": [],
        "conversation_url": "",
        "started_at": started_at,
        "completed_at": None,
        "error": None,
    }
    _write_manifest(manifest_path, manifest)

    try:
        script = script_path.read_text(encoding="utf-8").strip()
        if not script:
            raise RuntimeError(f"Thumbnail source script is empty at {script_path}")

        _log(log, "Thumbnail: deriving a viral cover prompt from the final audio script")
        prompt = await generate_thumbnail_prompt(
            title=title,
            script=script,
            provider_id=provider_id,
            ai_endpoint=ai_endpoint,
            ai_model=ai_model,
            log=log,
        )
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        _log(log, f"Thumbnail: prompt saved to {prompt_path}")

        before = {
            path.resolve()
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        }
        timeout = config.THUMBNAIL_CHATGPT_TIMEOUT
        result = await run_opencli(
            [
                "chatgpt",
                "image",
                prompt,
                "--op",
                str(image_dir.resolve()),
                "--timeout",
                str(timeout),
                "--window",
                "background",
                "--site-session",
                "persistent",
                "--keep-tab",
                "false",
                "-f",
                "json",
            ],
            timeout=timeout + 45,
        )
        parsed_rows = _rows(first_json(result.stdout))
        reported = _reported_image_files(parsed_rows, image_dir)
        created = _new_image_files(image_dir, before)
        images = list(dict.fromkeys([*reported, *created]))
        if not images:
            raise OpenCLIError(
                "ChatGPT Web completed without a valid downloaded image in the task thumbnail directory"
            )

        conversation_url = next(
            (_field(row, "link") for row in parsed_rows if _field(row, "link")),
            "",
        )
        artifact = ThumbnailArtifact(
            prompt_path=str(prompt_path.resolve()),
            image_path=str(images[0]),
            manifest_path=str(manifest_path.resolve()),
            conversation_url=conversation_url,
        )
        manifest.update(
            {
                "status": "ready",
                "image_path": artifact.image_path,
                "image_paths": [str(path) for path in images],
                "conversation_url": conversation_url,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_manifest(manifest_path, manifest)
        _log(log, f"Thumbnail: ChatGPT Web image saved to {artifact.image_path}")
        return artifact
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
        )
        _write_manifest(manifest_path, manifest)
        raise


def artifact_dict(artifact: ThumbnailArtifact) -> dict[str, str]:
    """Small JSON-friendly helper for scripts and acceptance checks."""
    return asdict(artifact)
