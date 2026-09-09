import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from backend import config
from backend.database import update_task
from backend.models import SourceType, TaskResponse, TaskStatus, ScriptFormat
from backend.pipeline.extractors.youtube import extract_youtube
from backend.pipeline.extractors.epub import extract_epub
from backend.pipeline.extractors.epub_curated import extract_epub_curated
from backend.pipeline.extractors.pdf import extract_pdf
from backend.pipeline.extractors.topic import extract_topic
from backend.pipeline.digester import summarize, generate_script
from backend.pipeline.tts import generate_tts
from backend.pipeline.composer import compose_video
from backend.pipeline.footage import acquire_footage
from backend.pipeline.thumbnail import generate_thumbnail
from backend.pipeline.title import generate_title
from backend.pipeline.music import _probe_duration, generate_background_music
from backend.daily_news.research import (
    NewsArticle,
    ResearchDossier,
    SourceFetch,
    dossier_markdown,
    run_research,
)
from backend.daily_news.review import review_daily_script
from backend.daily_news.scriptwriter import generate_daily_script, narration_duration_report
from backend.pipeline.extractors.base import ExtractedContent

logger = logging.getLogger(__name__)

LogCallback = Callable[[str], None]

EXTRACTORS = {
    "youtube": extract_youtube,
    "epub": extract_epub,
    "pdf": extract_pdf,
}


def _daily_edition_date(task: TaskResponse):
    """Resolve the edition date from the scheduler payload, or local news time."""
    from datetime import date

    raw = task.source_url or ""
    if raw:
        try:
            payload = json.loads(raw)
            value = str(payload.get("edition_date") or "") if isinstance(payload, dict) else ""
            if value:
                return date.fromisoformat(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return datetime.now(ZoneInfo(task.config.news_timezone)).date()


def pipeline_log_path(task_dir: Path) -> Path:
    return task_dir / "logs" / "pipeline.log"


def append_pipeline_log(task_dir: Path, message: str):
    log_path = pipeline_log_path(task_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def format_pipeline_log(task_id: str, message: str, timestamp: datetime | None = None) -> str:
    event_time = timestamp or datetime.now().astimezone()
    return f"[{event_time.strftime('%Y%m%d-%H:%M:%S')}] [{task_id}] {message}"


def emit_pipeline_log(task_id: str, task_dir: Path, message: str, log: LogCallback | None = None):
    formatted = format_pipeline_log(task_id, message)
    append_pipeline_log(task_dir, formatted)
    logger.info(formatted)
    if log:
        log(formatted)


def task_output_dir(task: TaskResponse) -> Path:
    """Return the immutable artifact root persisted with an existing task."""
    return Path(task.output_dir) if task.output_dir else config.OUTPUTS_DIR / task.id


async def _acquire_task_footage(
    task: TaskResponse,
    task_dir: Path,
    task_log: LogCallback,
    *,
    title: str | None = None,
    supplied_queries: list[str] | None = None,
):
    script_path = Path(task.script_path or task_dir / "script.txt")
    manifest = await acquire_footage(
        media_provider=task.config.footage_provider,
        task_id=task.id,
        task_dir=task_dir,
        title=title or task.generated_title or task.source_title or task.id,
        script_path=script_path,
        clip_count=task.config.footage_clip_count,
        orientation=task.config.video_orientation,
        license_policy=task.config.footage_license_policy,
        provider_id=task.config.provider_id,
        ai_endpoint=task.config.ai_endpoint,
        ai_model=task.config.ai_model,
        supplied_queries=supplied_queries,
        log=task_log,
    )

    requested = int(manifest.get("requested_clip_count") or 0)
    acquired = len(manifest.get("clips") or [])
    if acquired < requested:
        missing = ", ".join(
            str(item.get("query") or "") for item in manifest.get("missing_queries", [])
        )
        detail = f"unfilled searches: {missing}. " if missing else ""
        raise RuntimeError(
            f"Public-footage acquisition incomplete: {acquired}/{requested} eligible clips; "
            f"{detail}"
            "review footage/manifest.json for rejected candidates and search failures. "
            "Retry footage acquisition to resume verified clips before rendering."
        )
    return manifest


async def _generate_task_thumbnail(
    task: TaskResponse,
    task_dir: Path,
    task_log: LogCallback,
    *,
    title: str,
):
    script_path = Path(task.script_path or task_dir / "script.txt")
    artifact = await generate_thumbnail(
        task_id=task.id,
        task_dir=task_dir,
        title=title,
        script_path=script_path,
        provider_id=task.config.provider_id,
        ai_endpoint=task.config.ai_endpoint,
        ai_model=task.config.ai_model,
        log=task_log,
    )
    await update_task(task.id, thumbnail_path=artifact.image_path)
    task.thumbnail_path = artifact.image_path
    return artifact


async def _generate_task_title(
    task: TaskResponse,
    task_dir: Path,
    task_log: LogCallback,
    *,
    source_title: str,
    summary: dict | None = None,
):
    script_path = Path(task.script_path or task_dir / "script.txt")
    script = script_path.read_text(encoding="utf-8").strip()
    if not script:
        raise RuntimeError(f"Title source script is empty at {script_path}")
    artifact = await generate_title(
        task_id=task.id,
        task_dir=task_dir,
        source_title=source_title,
        summary=summary,
        script=script,
        provider_id=task.config.provider_id,
        ai_endpoint=task.config.ai_endpoint,
        ai_model=task.config.ai_model,
        log=task_log,
    )
    await update_task(task.id, generated_title=artifact.title)
    task.generated_title = artifact.title
    return artifact


async def _after_audio(
    task: TaskResponse,
    *,
    script_path: str,
    audio_path: str,
    task_log: LogCallback,
    log: LogCallback | None,
    prefix: str = "",
):
    """Either pause for audio review or, when the task opted out, continue
    straight into the compose stage."""
    if (
        task.source_type == SourceType.NEWS_DAILY
        and task.config.target_duration_minutes is not None
    ):
        narration_seconds = await _probe_duration(Path(audio_path))
        duration_contract = narration_duration_report(
            narration_seconds,
            task.config.target_duration_minutes,
        )
        task_log(
            f"{prefix}Narration duration contract: actual "
            f"{narration_seconds / 60:.2f} min, target "
            f"{task.config.target_duration_minutes} min"
        )
        if not duration_contract["passed"]:
            raise RuntimeError(
                "Daily-news narration duration is outside target tolerance: "
                f"target {task.config.target_duration_minutes} min, actual "
                f"{narration_seconds / 60:.2f} min; required "
                f"{duration_contract['minimum_seconds'] / 60:.2f}-"
                f"{duration_contract['maximum_seconds'] / 60:.2f} min"
            )
    if not task.config.auto_render:
        await update_task(task.id, status=TaskStatus.AWAITING_REVIEW.value)
        task_log(f"{prefix}Audio ready, awaiting review before video render")
        return

    task_log(f"{prefix}Audio ready, audio review skipped — rendering video now")
    # run_compose reads the paths off the task object, which still holds the
    # values from before this run wrote them.
    task.script_path = script_path
    task.audio_path = audio_path
    await run_compose(task, log=log)


async def run_pipeline(task: TaskResponse, log: LogCallback | None = None):
    task_dir = task_output_dir(task)
    task_dir.mkdir(parents=True, exist_ok=True)
    task_log = lambda message: emit_pipeline_log(task.id, task_dir, message, log)

    await update_task(task.id, output_dir=str(task_dir))
    task.output_dir = str(task_dir)

    ai_endpoint = task.config.ai_endpoint
    ai_model = task.config.ai_model
    provider_id = task.config.provider_id

    # Stage 1: Extract
    task_log(f"Stage 1: Extracting from {task.source_type}")
    await update_task(task.id, status=TaskStatus.EXTRACTING.value)

    dossier = None
    edition_date = None
    if task.source_type == "news_daily":
        edition_date = _daily_edition_date(task)
        await update_task(task.id, status=TaskStatus.RESEARCHING.value)
        dossier = await run_research(
            edition_date,
            task_dir,
            max_stories=task.config.news_max_stories,
            window_hours=task.config.news_window_hours,
            log=task_log,
        )
        content = ExtractedContent(
            source_type="news_daily",
            title=f"ByteFront Espresso — {edition_date.isoformat()}",
            text=dossier_markdown(dossier),
            metadata={
                "origin": "daily_news",
                "edition_date": edition_date.isoformat(),
                "selected_story_ids": [article.id for article in dossier.selected],
                "research_required": True,
            },
        )
    elif task.source_type == "topic":
        content = await extract_topic(task.source_title, task.source_url, log=task_log)
    elif task.source_type == "epub" and task.config.processing_mode == "curated_highlights":
        content = await extract_epub_curated(task.source_url, str(task_dir / "isla_reader"), log=task_log)
    elif task.source_type == "youtube":
        content = await extract_youtube(task.source_url, log=task_log)
    else:
        extractor = EXTRACTORS.get(task.source_type)
        if not extractor:
            raise ValueError(f"Unsupported source type: {task.source_type}")
        content = await extractor(task.source_url, log=task_log)
    await update_task(task.id, source_title=content.title)
    task_log(f"Extracted '{content.title}' ({len(content.text.split())} words)")

    (task_dir / "extracted.json").write_text(
        json.dumps({"title": content.title, "metadata": content.metadata, "text_length": len(content.text)}, indent=2)
    )

    # Stage 2: Digest
    task_log("Stage 2: Digesting content")
    await update_task(task.id, status=TaskStatus.DIGESTING.value)

    if dossier is not None:
        summary = {
            "title": content.title,
            "thesis": "A balanced bilingual-source briefing of the most consequential current frontier-technology developments.",
            "talking_points": [
                {
                    "headline": article.title,
                    "source": article.source_name,
                    "url": article.url,
                    "category": article.category,
                    "evidence": article.evidence_text[:1800],
                }
                for article in dossier.selected
            ],
            "key_quotes": [],
            "discussion_angles": ["technical significance", "evidence quality", "near-term implications"],
            "edition_date": edition_date.isoformat(),
        }
    else:
        summary = await summarize(content, ai_endpoint, ai_model, provider_id, log=task_log)
    (task_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    if dossier is not None and edition_date is not None:
        script = await generate_daily_script(
            dossier,
            edition_date,
            target_duration_minutes=task.config.target_duration_minutes,
            language=task.config.news_language,
            opening_remarks=task.config.opening_remarks,
            closing_remarks=task.config.closing_remarks,
            ai_endpoint=ai_endpoint,
            ai_model=ai_model,
            provider_id=provider_id,
            log=task_log,
        )
        (task_dir / "script.draft.txt").write_text(script, encoding="utf-8")
        task_log("Stage 2b: Reviewing every script claim with ChatGPT Web")
        await update_task(task.id, status=TaskStatus.REVIEWING.value)
        reviewed = await review_daily_script(
            script,
            dossier,
            edition_date,
            task_dir,
            language=task.config.news_language,
            opening_remarks=task.config.opening_remarks,
            closing_remarks=task.config.closing_remarks,
            ai_endpoint=ai_endpoint,
            ai_model=ai_model,
            provider_id=provider_id,
            target_duration_minutes=task.config.target_duration_minutes,
            log=task_log,
        )
        script = reviewed.script
    else:
        script = await generate_script(
            summary,
            task.config.target_duration_minutes or 10,
            script_format=task.config.script_format.value,
            ai_endpoint=ai_endpoint,
            ai_model=ai_model,
            provider_id=provider_id,
            closing_remarks=task.config.closing_remarks,
            log=task_log,
        )
    script_path = str(task_dir / "script.txt")
    Path(script_path).write_text(script, encoding="utf-8")
    await update_task(task.id, script_path=script_path)
    task.script_path = script_path
    task_log(f"Script saved to {script_path}")

    # Stage 3: a dedicated Agent SDK session turns the completed narration into
    # the publication title. It is persisted separately from the extracted
    # source title and becomes the title consumed by every downstream stage.
    task_log("Stage 3: Generating publication title with independent Agent")
    await update_task(task.id, status=TaskStatus.TITLING.value)
    title_artifact = await _generate_task_title(
        task,
        task_dir,
        task_log,
        source_title=content.title,
        summary=summary,
    )
    publication_title = title_artifact.title

    # Stage 4: use the final narration (the exact TTS input) to derive one
    # cover-art prompt, then generate/download the image through the signed-in
    # ChatGPT web session. Cover art is an enhancement, so a web/provider
    # outage is recorded in thumbnail/manifest.json but never discards a valid
    # script or forces a costly TTS retry.
    if task.config.thumbnail_enabled:
        task_log("Stage 4: Generating script-driven viral thumbnail")
        try:
            await _generate_task_thumbnail(
                task,
                task_dir,
                task_log,
                title=publication_title,
            )
        except Exception as exc:
            task_log(f"Thumbnail generation could not complete; continuing without cover art: {exc}")

    # Enabled footage is a delivery requirement. Fail before expensive TTS
    # when scouting cannot fulfill the plan; saved scripts remain resumable.
    if task.config.footage_enabled:
        task_log("Stage 5: Scouting open-license public footage")
        await update_task(task.id, status=TaskStatus.SOURCING.value)
        try:
            await _acquire_task_footage(
                task,
                task_dir,
                task_log,
                title=publication_title,
            )
        except Exception as exc:
            task_log(f"Public footage scout blocked delivery: {exc}")
            raise

    # Stage 6: TTS
    task_log(f"Stage 6: Generating TTS audio with {task.config.tts_model}")
    await update_task(task.id, status=TaskStatus.TTS.value)

    voices = [task.config.voice_1]
    if task.config.script_format == ScriptFormat.DIALOGUE:
        voices.append(task.config.voice_2)

    audio_dir = str(task_dir / "audio")
    audio_path = await generate_tts(script_path, audio_dir, voices, task.config.tts_model, log=task_log)
    await update_task(task.id, audio_path=audio_path)

    # Pause for audio review before the (expensive) video composition. The user
    # previews the audio and triggers the compose stage via the render endpoint,
    # unless the task was created with the review step turned off.
    await _after_audio(
        task,
        script_path=script_path,
        audio_path=audio_path,
        task_log=task_log,
        log=log,
    )


async def run_regenerate(task: TaskResponse, log: LogCallback | None = None):
    """Re-run only the TTS and compose stages from an existing (possibly
    edited) script, skipping extraction and digestion."""
    task_dir = task_output_dir(task)
    task_dir.mkdir(parents=True, exist_ok=True)
    task_log = lambda message: emit_pipeline_log(task.id, task_dir, message, log)

    script_path = task.script_path or str(task_dir / "script.txt")
    if not Path(script_path).exists():
        raise FileNotFoundError(f"No script to regenerate from at {script_path}")

    summary_path = task_dir / "summary.json"
    summary = None
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            task_log("Regenerate: Existing summary is unreadable; title Agent will use the script")

    task_log("Regenerate: Updating publication title with independent Agent")
    await update_task(task.id, status=TaskStatus.TITLING.value, error_message=None)
    title_artifact = await _generate_task_title(
        task,
        task_dir,
        task_log,
        source_title=task.source_title or task.id,
        summary=summary,
    )
    title = title_artifact.title

    # Keep cover art aligned with an edited script. This still precedes the TTS
    # subprocess, so a thumbnail failure consumes no model-synthesis memory.
    if task.config.thumbnail_enabled:
        task_log("Regenerate: Updating viral thumbnail from edited script")
        try:
            await _generate_task_thumbnail(
                task,
                task_dir,
                task_log,
                title=title,
            )
        except Exception as exc:
            task_log(f"Regenerate: Thumbnail update failed; retaining prior cover: {exc}")

    # Stage 6: TTS
    task_log(f"Regenerate: Generating TTS audio with {task.config.tts_model}")
    await update_task(task.id, status=TaskStatus.TTS.value, error_message=None)

    voices = [task.config.voice_1]
    if task.config.script_format == ScriptFormat.DIALOGUE:
        voices.append(task.config.voice_2)

    audio_dir = str(task_dir / "audio")
    audio_path = await generate_tts(script_path, audio_dir, voices, task.config.tts_model, log=task_log)
    await update_task(task.id, audio_path=audio_path)

    # Pause for audio review, same as the full pipeline.
    await _after_audio(
        task,
        script_path=script_path,
        audio_path=audio_path,
        task_log=task_log,
        log=log,
        prefix="Regenerate: ",
    )


async def run_tts_resume(task: TaskResponse, log: LogCallback | None = None):
    """Resume only TTS and its normal post-audio path from a saved script.

    Unlike ``run_regenerate``, this recovery path deliberately preserves the
    existing publication title and thumbnail. It is intended for a failed TTS
    run whose upstream editorial artifacts are already complete.
    """
    task_dir = task_output_dir(task)
    task_dir.mkdir(parents=True, exist_ok=True)
    task_log = lambda message: emit_pipeline_log(task.id, task_dir, message, log)

    script_path = task.script_path or str(task_dir / "script.txt")
    if not Path(script_path).is_file():
        raise FileNotFoundError(f"No script to resume TTS from at {script_path}")

    task_log(f"TTS resume: Generating audio with {task.config.tts_model} from the saved script")
    await update_task(task.id, status=TaskStatus.TTS.value, error_message=None)

    voices = [task.config.voice_1]
    if task.config.script_format == ScriptFormat.DIALOGUE:
        voices.append(task.config.voice_2)

    audio_dir = str(task_dir / "audio")
    audio_path = await generate_tts(
        script_path,
        audio_dir,
        voices,
        task.config.tts_model,
        log=task_log,
    )
    await update_task(task.id, audio_path=audio_path)

    await _after_audio(
        task,
        script_path=script_path,
        audio_path=audio_path,
        task_log=task_log,
        log=log,
        prefix="TTS resume: ",
    )


async def run_daily_review_resume(task: TaskResponse, log: LogCallback | None = None):
    """Resume a failed daily edition from its persisted dossier and draft.

    Research and the first yhroot script call are intentionally not repeated.
    This is the recovery path for a daytime provider outage during fact-check
    correction: rerun the independent review, persist the approved script, and
    continue through the ordinary regenerate/TTS/render path.
    """
    if task.source_type != SourceType.NEWS_DAILY:
        raise ValueError("Review resume is only available for daily-news tasks")

    task_dir = task_output_dir(task)
    task_log = lambda message: emit_pipeline_log(task.id, task_dir, message, log)
    dossier_path = task_dir / "research" / "dossier.json"
    draft_path = task_dir / "script.draft.txt"
    if not dossier_path.is_file() or not draft_path.is_file():
        raise FileNotFoundError("Daily review resume requires dossier.json and script.draft.txt")

    raw = json.loads(dossier_path.read_text(encoding="utf-8"))
    dossier = ResearchDossier(
        edition_date=str(raw["edition_date"]),
        generated_at=str(raw["generated_at"]),
        window_hours=int(raw["window_hours"]),
        candidates=[NewsArticle(**item) for item in raw.get("candidates", [])],
        selected=[NewsArticle(**item) for item in raw.get("selected", [])],
        fetches=[SourceFetch(**item) for item in raw.get("fetches", [])],
    )
    edition_date = _daily_edition_date(task)
    script = draft_path.read_text(encoding="utf-8").strip()
    task_log("Review resume: reusing persisted research dossier and yhroot draft")
    await update_task(task.id, status=TaskStatus.REVIEWING.value, error_message=None)
    reviewed = await review_daily_script(
        script,
        dossier,
        edition_date,
        task_dir,
        language=task.config.news_language,
        opening_remarks=task.config.opening_remarks,
        closing_remarks=task.config.closing_remarks,
        ai_endpoint=task.config.ai_endpoint,
        ai_model=task.config.ai_model,
        provider_id=task.config.provider_id,
        target_duration_minutes=task.config.target_duration_minutes,
        log=task_log,
    )
    script_path = task_dir / "script.txt"
    script_path.write_text(reviewed.script, encoding="utf-8")
    await update_task(task.id, script_path=str(script_path))
    task.script_path = str(script_path)
    task_log(f"Review resume: approved script saved to {script_path}")
    await run_regenerate(task, log=log)


async def run_footage_acquisition(
    task: TaskResponse,
    *,
    resume_status: str,
    supplied_queries: list[str] | None = None,
    log: LogCallback | None = None,
):
    """Run or retry only the public-footage scout from an existing script."""
    task_dir = task_output_dir(task)
    task_dir.mkdir(parents=True, exist_ok=True)
    task_log = lambda message: emit_pipeline_log(task.id, task_dir, message, log)

    script_path = Path(task.script_path or task_dir / "script.txt")
    if not script_path.exists():
        raise FileNotFoundError(f"No script to scout from at {script_path}")

    task_log("Footage retry: planning and acquiring open-license B-roll")
    await update_task(task.id, status=TaskStatus.SOURCING.value, error_message=None)
    try:
        await _acquire_task_footage(
            task,
            task_dir,
            task_log,
            supplied_queries=supplied_queries,
        )
    finally:
        await update_task(task.id, status=resume_status)


async def run_compose(task: TaskResponse, log: LogCallback | None = None):
    """Resume from the compose stage using an already-generated script and
    audio. Triggered after the user has reviewed the audio preview."""
    task_dir = task_output_dir(task)
    task_dir.mkdir(parents=True, exist_ok=True)
    task_log = lambda message: emit_pipeline_log(task.id, task_dir, message, log)

    script_path = task.script_path or str(task_dir / "script.txt")
    audio_path = task.audio_path
    if not audio_path or not Path(audio_path).exists():
        raise FileNotFoundError(f"No audio to compose from for task {task.id}")
    if not Path(script_path).exists():
        raise FileNotFoundError(f"No script to compose from at {script_path}")

    title = task.generated_title or task.source_title or task.id

    background_music_path = None
    if task.config.background_music_enabled:
        music_path = task_dir / "music" / "background.wav"
        if not music_path.is_file():
            action = (
                f"Loading local program track {task.config.background_music_track_id}"
                if task.config.background_music_provider == "local_library"
                else f"Generating a matching program bed via {task.config.background_music_provider}"
            )
            task_log(f"Music: {action}")
            await update_task(task.id, status=TaskStatus.MUSIC.value, error_message=None)
            summary = None
            summary_path = task_dir / "summary.json"
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    task_log("Music: Summary unreadable; using the publication title as context")
            artifact = await generate_background_music(
                task.id,
                task_dir,
                title=title,
                summary=summary,
                provider=task.config.background_music_provider,
                track_id=task.config.background_music_track_id,
                log=task_log,
            )
            music_path = Path(artifact.audio_path)
        background_music_path = str(music_path)

    task_log(f"Render: Composing video with {task.config.video_template} template")
    await update_task(task.id, status=TaskStatus.COMPOSING.value, error_message=None)

    video_path = await compose_video(
        script_path=script_path,
        audio_path=audio_path,
        output_dir=str(task_dir),
        title=title,
        include_character=task.config.include_character,
        captions_enabled=task.config.captions_enabled,
        video_template=task.config.video_template,
        video_orientation=task.config.video_orientation,
        opening_style=task.config.opening_style,
        outro_style=task.config.outro_style,
        edition_date=(
            _daily_edition_date(task).isoformat()
            if task.source_type == SourceType.NEWS_DAILY
            else None
        ),
        collage_broll_enabled=task.config.collage_broll_enabled,
        collage_broll_count=task.config.collage_broll_count,
        news_images_enabled=task.config.news_images_enabled,
        news_image_count=task.config.news_image_count,
        is_monologue=task.config.script_format == ScriptFormat.MONOLOGUE,
        # The compose stage now runs its own AI calls (art direction, then the
        # Claude Agent SDK authoring crews), so it needs the same provider the
        # digest stage used.
        ai_endpoint=task.config.ai_endpoint,
        ai_model=task.config.ai_model,
        provider_id=task.config.provider_id,
        tts_model=task.config.tts_model,
        background_music_path=background_music_path,
        background_music_bed_db=task.config.background_music_bed_db,
        background_music_duck_db=task.config.background_music_duck_db,
        program_music_pacing_enabled=task.config.program_music_pacing_enabled,
        program_music_intro_seconds=task.config.program_music_intro_seconds,
        program_music_opening_gap_seconds=task.config.program_music_opening_gap_seconds,
        program_music_story_gap_seconds=task.config.program_music_story_gap_seconds,
        log=task_log,
    )
    completion_fields = {
        "video_path": video_path,
        # Keep COMPLETE private until the worker has either completed the
        # publication decision or deliberately skipped a duplicate publish.
        # This prevents the UI from starting a rework against a version whose
        # publication outcome is still in flight.
        "status": TaskStatus.PUBLISHING.value,
    }
    rendered_video = Path(video_path)
    # Real composers return a materialized MP4 and should persist its probed
    # duration. Test doubles and external adapters may return a future path;
    # do not turn that valid contract into an ffprobe failure.
    if rendered_video.is_file():
        completion_fields["duration_seconds"] = await _probe_duration(rendered_video)
    await update_task(task.id, **completion_fields)

    task_log(f"Render complete: {video_path}")
