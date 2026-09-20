"""Persistent, source-bound upload copy; generation never publishes to a platform."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from twitter_text import extract_urls, parse_tweet

from backend import config, database as db
from backend.models import TaskResponse, TaskStatus
from backend.pipeline.digester import _chat, _resolve_provider
from backend.title_strategy import YOUTUBE_TITLE_MAX_CHARS, with_title_strategy

logger = logging.getLogger(__name__)
HUMANIZER_REVISION = "9862685f575c65a8247f90369951df1b3416e3d6"
HUMANIZER_DIR = Path(__file__).parent / "prompts" / "vendor" / "humanizer"
_running: dict[str, asyncio.Task] = {}
ELIGIBLE_STATUSES = {TaskStatus.COMPLETE, TaskStatus.AWAITING_REVIEW, TaskStatus.FAILED}


class CopyDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    youtube_title: str = Field(min_length=1, max_length=YOUTUBE_TITLE_MAX_CHARS)
    youtube_show_notes: str = Field(min_length=1, max_length=5000)
    x_paragraphs: list[str] = Field(min_length=1, max_length=3)


def _path(task: TaskResponse) -> Path:
    directory = Path(task.output_dir) if task.output_dir else config.OUTPUTS_DIR / task.id
    return directory / "social_copy.json"


def _inputs(task: TaskResponse) -> dict:
    if not task.script_path or not Path(task.script_path).is_file():
        raise ValueError("A saved narration script is required")
    script = Path(task.script_path).read_text(encoding="utf-8").strip()
    if not script:
        raise ValueError("The saved narration script is empty")
    references = []
    if task.source_url and task.source_url.startswith(("https://", "http://")):
        references.append({"url": task.source_url})
    dossier = _path(task).parent / "research" / "dossier.json"
    if dossier.is_file():
        # Selected stories only; rejected candidates must not enter the copy.
        for item in json.loads(dossier.read_text(encoding="utf-8")).get("selected", []):
            if isinstance(item, dict) and str(item.get("url", "")).startswith(("https://", "http://")):
                references.append({key: item[key] for key in ("title", "url", "source_name") if key in item})
    return {"final_narration": script, "source_references": references}


def _fingerprint(inputs: dict) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staged.replace(path)


def read_copy(task: TaskResponse) -> dict:
    result = {"status": "missing", "copy": None, "error": None, "stale": False}
    try:
        if _path(task).is_file():
            result.update(json.loads(_path(task).read_text(encoding="utf-8")))
    except (ValueError, OSError):
        result.update(status="failed", error="Saved copy could not be read. Generate it again.")
    try:
        current = _fingerprint(_inputs(task))
        result["stale"] = bool(result.get("source_fingerprint") and current != result["source_fingerprint"])
    except (ValueError, OSError):
        result["stale"] = bool(result.get("copy"))
    if task.id in _running:
        result["status"] = "generating"
    elif result["status"] == "generating":
        result.update(status="failed", error="Generation was interrupted. Try again.")
    return result


def _validate(raw: str, inputs: dict) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    draft = CopyDraft.model_validate_json(cleaned)
    title = draft.youtube_title.strip()
    notes = draft.youtube_show_notes.strip()
    if not title or not notes or "\n" in title or "\r" in title:
        raise ValueError("Title must be one nonempty line and Show Notes must not be blank")
    if any(char in title + notes for char in "<>"):
        raise ValueError("YouTube fields cannot contain angle brackets")
    paragraphs = [paragraph.strip() for paragraph in draft.x_paragraphs]
    if any(not paragraph or "\n" in paragraph or "\r" in paragraph for paragraph in paragraphs):
        raise ValueError("Each X item must be one nonempty paragraph on one line")
    # Preserve the writer's semantic grouping; sentence counts do not set breaks.
    post = "\n\n".join(paragraphs)
    parsed = parse_tweet(post)
    if not parsed.valid:
        raise ValueError(f"X post is invalid or exceeds 280 weighted characters (got {parsed.weightedLength})")
    if len(re.findall(r"(?<!\w)#[\w]+", post)) > 2:
        raise ValueError("X post must have at most two hashtags")
    allowed_urls = set(extract_urls(inputs["final_narration"]))
    allowed_urls.update(item["url"] for item in inputs["source_references"])
    for url in extract_urls(title + "\n" + notes + "\n" + post):
        if url not in allowed_urls:
            raise ValueError("Use only the supplied source URLs; do not invent links")
    if re.search(r"(?m)^\s*\d{1,2}:\d{2}\b", notes):
        raise ValueError("Do not invent chapter timestamps")
    return {"youtube_title": title, "youtube_show_notes": notes, "x_post": post,
            "x_weighted_length": parsed.weightedLength}


async def _generate(task: TaskResponse, previous: dict) -> None:
    path = _path(task)
    result = {**previous, "status": "generating", "error": None}
    try:
        _write(path, result)
        inputs = _inputs(task)
        fingerprint = _fingerprint(inputs)
        skill = (HUMANIZER_DIR / "SKILL.md").read_text(encoding="utf-8")
        system = with_title_strategy(
            (config.PROMPTS_DIR / "social_copy.txt").read_text(encoding="utf-8")
        )
        system += "\n\n<humanizer_skill>\n" + skill + "\n</humanizer_skill>"
        endpoint, model, api_key = await _resolve_provider(
            task.config.provider_id, task.config.ai_endpoint, task.config.ai_model,
        )
        feedback = None
        previous_draft = None
        for attempt in range(3):
            payload = {**inputs}
            if feedback:
                payload["validation_feedback"] = feedback
                payload["previous_draft"] = previous_draft
            # Humanizer's draft/critique/rewrite pass needs reasoning headroom
            # even on providers that ignore disable_thinking. Let the shared
            # transport own timeouts and primary/backup retry policy.
            raw = await _chat(
                system, json.dumps(payload, ensure_ascii=False), endpoint=endpoint,
                model=model, api_key=api_key, max_tokens=16384, enable_skills=False,
                disable_thinking=True, label="Social copy · Humanizer",
            )
            try:
                copy = _validate(raw, inputs)
                break
            except (ValueError, ValidationError) as exc:
                feedback = str(exc)
                previous_draft = raw
                logger.info("Social copy validation for %s, attempt %d/3: %s", task.id, attempt + 1, feedback)
                if attempt == 2:
                    raise ValueError(f"Copy failed validation after 3 attempts: {feedback}") from exc
        latest = await db.get_task(task.id)
        if latest is None:
            return
        if _fingerprint(_inputs(latest)) != fingerprint:
            raise ValueError("The narration changed during generation. Generate again from the saved script.")
        result.update(
            status="ready", copy=copy, source_fingerprint=fingerprint,
            generated_at=datetime.now(timezone.utc).isoformat(), stale=False,
            humanizer_revision=HUMANIZER_REVISION,
            humanizer_sha256=hashlib.sha256(skill.encode()).hexdigest(),
            prompt_sha256=hashlib.sha256(system.encode()).hexdigest(),
        )
        logger.info("Social copy ready for %s (X: %d/280)", task.id, copy["x_weighted_length"])
    except asyncio.CancelledError:
        result.update(status="failed", error="Generation was interrupted. Try again.")
        _write(path, result)
        raise
    except Exception as exc:
        logger.warning("Social copy generation failed for %s: %s", task.id, exc)
        # Provider errors may contain internal endpoints. Keep them in local logs.
        detail = str(exc) if isinstance(exc, ValueError) else "The writing service failed. Try again."
        result.update(status="failed", error=detail)
    if await db.get_task(task.id) is not None:
        _write(path, result)


def start_generation(task: TaskResponse, *, regenerate: bool = False) -> dict:
    current = read_copy(task)
    if task.id in _running:
        return current
    if current["status"] == "ready" and not current["stale"] and not regenerate:
        return current
    if task.status not in ELIGIBLE_STATUSES:
        raise ValueError("Wait until the narration is ready before generating upload copy")
    _inputs(task)
    job = asyncio.create_task(_generate(task, current), name=f"social-copy-{task.id}")
    _running[task.id] = job
    def finished(done: asyncio.Task) -> None:
        _running.pop(task.id, None)
        if not done.cancelled() and done.exception() is not None:
            logger.error("Could not save social copy for %s: %s", task.id, done.exception())

    job.add_done_callback(finished)
    return {**current, "status": "generating", "error": None}


async def stop_generations() -> None:
    jobs = list(_running.values())
    for job in jobs:
        job.cancel()
    await asyncio.gather(*jobs, return_exceptions=True)
