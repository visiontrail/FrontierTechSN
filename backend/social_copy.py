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
from twitter_text import extract_urls, extract_urls_with_indices, parse_tweet

from backend import config, database as db
from backend.models import TaskResponse, TaskStatus
from backend.pipeline.digester import _chat, _resolve_provider
from backend.title_strategy import YOUTUBE_TITLE_MAX_CHARS, with_title_strategy
from backend.daily_news.rundown import read_rundown, read_chapters, chapter_text

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


class StoryNote(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    story_id: str
    text: str = Field(min_length=1, max_length=2000)


class DailyCopyDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    youtube_title: str = Field(min_length=1, max_length=YOUTUBE_TITLE_MAX_CHARS)
    youtube_story_notes: list[StoryNote] = Field(min_length=1)
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
    inputs = {"final_narration": script, "source_references": references}
    rundown = read_rundown(_path(task).parent, script)
    if rundown:
        if not task.generated_title:
            raise ValueError("Morning Desk requires its production title before writing upload copy")
        inputs.update(
            daily_rundown=rundown, publication_title=task.generated_title,
            source_references=[{key: s[key] for key in ("story_id", "url", "source_name")}
                               for s in rundown["stories"]],
            chapters=read_chapters(_path(task).parent, rundown, task.audio_path, task.video_path),
        )
    return inputs


def _fingerprint(inputs: dict) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _daily_notes_suffix(inputs: dict) -> str:
    rundown = inputs["daily_rundown"]
    chapters = chapter_text(rundown, inputs.get("chapters", []), inputs["publication_title"])
    return ("\n\n" + chapters if chapters else "") + "\n\nSources:\n" + "\n".join(
        f"{story['source_name']}: {story['url']}" for story in rundown["stories"] if story.get("url")
    )


def _daily_notes_budget(inputs: dict) -> dict:
    reserved = len(_daily_notes_suffix(inputs))
    count = len(inputs["daily_rundown"]["story_order"])
    separators = 2 * (count - 1)
    available = 5000 - reserved
    if available < count + separators:
        raise ValueError("Measured chapters and sources leave no room for every story note")
    per_story = min(2000, (available - separators) // count)
    return {"description_max_characters": 5000, "reserved_characters": reserved,
            "joined_story_notes_max_characters": available,
            "story_separator_characters": separators,
            "suggested_characters_per_story": per_story,
            "target_characters_per_story": min(220, max(1, per_story * 7 // 10))}


def _daily_note_repair(raw: str, inputs: dict) -> list[dict] | None:
    """Make an over-budget retry an explicit edit of each approved story."""
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        draft = DailyCopyDraft.model_validate_json(cleaned)
    except (ValueError, ValidationError):
        return None
    rundown = inputs["daily_rundown"]
    if [row.story_id for row in draft.youtube_story_notes] != rundown["story_order"]:
        return None
    budget = _daily_notes_budget(inputs)
    if len("\n\n".join(row.text.strip() for row in draft.youtube_story_notes)) <= budget["joined_story_notes_max_characters"]:
        return None
    stories = {story["story_id"]: story for story in rundown["stories"]}
    return [{"story_id": row.story_id, "previous_text": row.text,
             "current_characters": len(row.text.strip()),
             "target_characters": budget["target_characters_per_story"],
             "approved_narration": stories[row.story_id]["narration"]}
            for row in draft.youtube_story_notes]


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


def _note_url_tokens(text: str) -> set[str]:
    tokens = set()
    for item in extract_urls_with_indices(text):
        token = item["url"]
        # The extractor stops bare hosts before a query/fragment/port. Include
        # those suffixes so a grounded name cannot authorize a new link.
        suffix = re.match(r"[:/?#@]\S+", text[item["indices"][1]:])
        tokens.add(token + (suffix.group() if suffix else ""))
    return tokens


def _validate(raw: str, inputs: dict) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    rundown = inputs.get("daily_rundown")
    if rundown:
        ordered = DailyCopyDraft.model_validate_json(cleaned)
        if ordered.youtube_title != inputs["publication_title"]:
            raise ValueError("Keep the exact supplied publication_title; the lead is already locked")
        if [row.story_id for row in ordered.youtube_story_notes] != rundown["story_order"]:
            raise ValueError("youtube_story_notes must cover every story once in the exact supplied story_order")
        notes = "\n\n".join(row.text.strip() for row in ordered.youtube_story_notes)
        budget = _daily_notes_budget(inputs)
        if len(notes) > budget["joined_story_notes_max_characters"]:
            raise ValueError(
                f"Story notes use {len(notes)} characters; maximum is "
                f"{budget['joined_story_notes_max_characters']} including blank-line separators. "
                f"Measured chapters and exact sources reserve {budget['reserved_characters']} "
                "of 5000 characters. Shorten prose while preserving every story and its caveats."
            )
        draft = CopyDraft(youtube_title=ordered.youtube_title, youtube_show_notes=notes,
                          x_paragraphs=ordered.x_paragraphs)
    else:
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
    unsupported_urls = sorted(set(extract_urls(title + "\n" + notes + "\n" + post)) - allowed_urls)
    if unsupported_urls:
        # Do not canonicalize away query strings: they may identify a different
        # article. Give the writer the exact mismatch so retries can repair it.
        feedback = (
            "Use only the supplied source URLs; do not invent links. "
            f"Unapproved URLs: {json.dumps(unsupported_urls, ensure_ascii=False)}. "
            "Copy supplied URLs verbatim, including query strings and fragments; "
            "do not remove tracking parameters."
        )
        supplied_variants = sorted(
            supplied for supplied in allowed_urls
            if any(supplied.split("?", 1)[0].split("#", 1)[0] ==
                   url.split("?", 1)[0].split("#", 1)[0] for url in unsupported_urls)
        )
        if supplied_variants:
            feedback += f" Exact supplied URLs for these paths: {json.dumps(supplied_variants, ensure_ascii=False)}."
        raise ValueError(feedback)
    if re.search(r"(?m)^\s*\d{1,2}:\d{2}\b", notes):
        raise ValueError("Do not invent chapter timestamps")
    if rundown:
        stories = {story["story_id"]: story for story in rundown["stories"]}
        note_links = {}
        for row in ordered.youtube_story_notes:
            # URL extraction also recognizes domain-shaped company names (Z.ai).
            # Preserve only exact bare tokens from THIS story's approved prose;
            # explicit links, paths and names from other stories still fail.
            grounded_names = {
                token for token in _note_url_tokens(stories[row.story_id]["narration"])
                if not re.search(r"[:/?#@]", token)
            }
            links = sorted(_note_url_tokens(row.text) - grounded_names)
            if links:
                note_links[row.story_id] = links
        if note_links:
            raise ValueError(
                "Keep story notes as prose; software appends sources in story order. "
                f"Remove these links from the identified story notes: {json.dumps(note_links, ensure_ascii=False)}. "
                "An exact bare company name from that story's approved narration may remain."
            )
        notes += _daily_notes_suffix(inputs)
        if len(notes) > 5000:
            raise ValueError("Show Notes plus measured chapters and sources exceed 5000 characters; shorten story notes")
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
        if inputs.get("daily_rundown"):
            notes_budget = _daily_notes_budget(inputs)
            system += """\nMORNING DESK LOCKED RUNDOWN overrides the generic output shape.
Return {"youtube_title":"exact publication_title", "youtube_story_notes":[{"story_id":"...","text":"..."}], "x_paragraphs":["..."]}.
Copy publication_title exactly; do not select or rewrite the lead. Write one short
note per story in daily_rundown.story_order. Each note covers ONLY that story's
supplied approved narration. The first note must explain the cold open's core
event and significance, preserving its caveats. Other notes follow playback order.
Do not add headings, greetings, timestamps or URLs to story notes: software appends
measured chapters and ordered source links. Preserve domain-shaped company names
such as Z.ai only when that exact bare name occurs in the same story's approved
narration; do not turn a name into a website link. X also opens with the locked lead;
any secondary highlights follow story_order. Keep all prose natural and concise.
Respect youtube_show_notes_budget: the joined story notes, including two newline
characters between notes, must fit joined_story_notes_max_characters. Chapters
and exact source links already consume reserved_characters; do not shorten or
omit them. Override the generic 150–350 word guidance: write ONE compact sentence
per story, aiming below target_characters_per_story (characters, NOT words).
Retain source attribution, uncertainty and the key development. Do not recap all
supporting details. The full narration remains in the video.
If story_note_edits is supplied, substantially condense EACH previous_text to its
target_characters using its approved_narration. Do not make only cosmetic edits.
Keep every story, the locked title and valid X paragraphs; return the full JSON.
"""
        endpoint, model, api_key = await _resolve_provider(
            task.config.provider_id, task.config.ai_endpoint, task.config.ai_model,
        )
        feedback = None
        previous_draft = None
        for attempt in range(3):
            payload = {**inputs}
            if inputs.get("daily_rundown"):
                payload["youtube_show_notes_budget"] = notes_budget
            if feedback:
                payload["validation_feedback"] = feedback
                payload["previous_draft"] = previous_draft
                if inputs.get("daily_rundown"):
                    edits = _daily_note_repair(previous_draft, inputs)
                    if edits:
                        payload["story_note_edits"] = edits
            # Humanizer's draft/critique/rewrite pass needs reasoning headroom
            # even on providers that ignore disable_thinking. Let the shared
            # transport own timeouts and primary/backup retry policy.
            raw = await _chat(
                system, json.dumps(payload, ensure_ascii=False), endpoint=endpoint,
                model=model, api_key=api_key, max_tokens=32768, enable_skills=False,
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
