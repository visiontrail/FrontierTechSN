"""Independent Agent SDK task for generating the video's publication title."""

from __future__ import annotations

import json
import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import config
from backend.pipeline.digester import _chat, _resolve_provider
from backend.title_strategy import YOUTUBE_TITLE_MAX_CHARS, with_title_strategy

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

_TITLE_PREFIX_RE = re.compile(
    r"^(?:[-*#]+\s*|\d+[.)、]\s*|(?:video\s+)?title\s*[:：]\s*|标题\s*[:：]\s*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TitleArtifact:
    title: str
    title_path: str
    manifest_path: str


def _log(log: LogCallback | None, message: str) -> None:
    if log is not None:
        log(message)
    else:
        logger.info(message)


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def clean_generated_title(value: str) -> str:
    """Normalize a one-title Agent response without silently inventing content."""
    clean = value.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Title agent returned an empty title")
    if len(lines) != 1:
        raise RuntimeError("Title agent must return exactly one title on one line")

    title = _TITLE_PREFIX_RE.sub("", lines[0]).strip()
    title = title.strip("\"'“”‘’《》")
    title = re.sub(r"\s+", " ", title).strip()
    if not title:
        raise RuntimeError("Title agent returned an empty title")
    if any(char in title for char in "<>"):
        raise RuntimeError("Title cannot contain angle brackets")
    if len(title) > YOUTUBE_TITLE_MAX_CHARS:
        raise RuntimeError(f"Title agent returned an overlong title ({len(title)} characters)")
    return title


async def generate_title(
    *,
    task_id: str,
    task_dir: Path,
    source_title: str,
    summary: dict[str, Any] | None,
    script: str,
    provider_id: int | None = None,
    ai_endpoint: str | None = None,
    ai_model: str | None = None,
    log: LogCallback | None = None,
) -> TitleArtifact:
    """Run a dedicated AI session and persist its title as an artifact."""
    title_dir = task_dir / "title"
    title_dir.mkdir(parents=True, exist_ok=True)
    title_path = title_dir / "title.txt"
    manifest_path = title_dir / "manifest.json"
    started_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "task_id": task_id,
        "status": "generating",
        "executor": "claude_agent_sdk",
        "source_title": source_title,
        "source_script": str(task_dir / "script.txt"),
        "title": "",
        "title_path": str(title_path.resolve()),
        "started_at": started_at,
        "completed_at": None,
        "error": None,
    }
    _write_manifest(manifest_path, manifest)

    try:
        endpoint, model, api_key = await _resolve_provider(
            provider_id,
            ai_endpoint,
            ai_model,
        )
        system_prompt = with_title_strategy(
            (config.PROMPTS_DIR / "title.txt").read_text(encoding="utf-8")
        )
        manifest["prompt_sha256"] = hashlib.sha256(system_prompt.encode()).hexdigest()
        if not script.strip():
            raise RuntimeError("Title generation requires the final narration script")
        payload: dict[str, str] = {"final_audio_script": script}
        _log(log, "Title agent: starting an independent Agent SDK session")
        for attempt in range(3):
            result = await _chat(
                system_prompt,
                json.dumps(payload, indent=2, ensure_ascii=False),
                model=model,
                endpoint=endpoint,
                api_key=api_key,
                # Allow reasoning headroom before the one-line response.
                max_tokens=1024,
                enable_skills=False,
                disable_thinking=True,
                log=log,
                label="Title agent",
            )
            try:
                generated_title = clean_generated_title(result)
                break
            except RuntimeError as exc:
                if attempt == 2:
                    raise RuntimeError(f"Title failed validation after 3 attempts: {exc}") from exc
                payload["validation_feedback"] = f"{exc}. Rewrite from the final narration; do not truncate."
                _log(log, f"Title agent: repairing invalid format (attempt {attempt + 1}/3): {exc}")
        title_path.write_text(generated_title + "\n", encoding="utf-8")
        manifest.update(
            {
                "status": "ready",
                "title": generated_title,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_manifest(manifest_path, manifest)
        _log(log, f"Title agent: generated '{generated_title}'")
        return TitleArtifact(
            title=generated_title,
            title_path=str(title_path.resolve()),
            manifest_path=str(manifest_path.resolve()),
        )
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


def artifact_dict(artifact: TitleArtifact) -> dict[str, str]:
    return asdict(artifact)
