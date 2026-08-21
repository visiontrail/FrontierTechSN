"""Compose the narration into a rendered video.

The stage runs in five steps:

1. **Storyboard** — cut the script into timed scenes against the audio.
2. **Direction** — an art director model decides what each scene shows.
3. **Authoring** — Claude Agent SDK crews write the HyperFrames scene files,
   starting from deterministic drafts and falling back to them scene-by-scene.
4. **Assembly** — a generated spine mounts the scenes on the audio timeline.
5. **Render** — the HyperFrames CLI captures the composition to MP4.

Steps 2 and 3 are the only places a model is involved, and neither can produce a
blank video: the spine is generated, every authored scene is gated against the
HyperFrames runtime contract, and lint failures revert the offending scene to
its deterministic draft before the render starts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

from backend import config
from backend.pipeline import (
    av_sync,
    assembler,
    collage_broll,
    director,
    footage,
    music,
    multimodal_review,
    news_images,
    scene_kit,
    storyboard as sb,
    visual_plan,
)
from backend.pipeline.process_logging import run_capture_logged, stream_subprocess
from backend.pipeline.video_format import FrameSpec, LANDSCAPE, resolve_frame_spec

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

# FFmpeg silence analysis is a quick pass; the HyperFrames render is the long
# stage and is streamed live, so this ceiling only guards a genuine hang.
SILENCE_TIMEOUT = 120

# Per-frame wall-clock ceiling used to derive the render timeout from the frame
# count (duration x fps). Scene-based compositions capture far faster than the
# old 158-clip single-page layout, but the ceiling stays generous: it must not
# kill a slow-but-live render, only a genuinely wedged one.
RENDER_SECONDS_PER_FRAME = 1.5
RENDER_TIMEOUT_FLOOR = 900  # never below 15 min, regardless of how short the clip is
# The real hang detector: HyperFrames reports capture progress continuously, so
# going silent for this long means it is wedged rather than merely slow. Warmup
# (Chrome launch, first-frame compile) is the longest legitimate quiet stretch.
RENDER_STALL_TIMEOUT = 600


def _load_cached_scene_plans(output_dir: Path, board: dict) -> list[dict] | None:
    """Recover the last fully written direction pass after a late render failure.

    ``visual_plan.json`` is written only after direction, collage attachment,
    grounding and music mixing have completed.  A missing runtime asset should
    therefore resume from that checkpoint instead of paying for the same slow
    provider calls again.  Scene ids are validated against the current
    storyboard so a changed script/audio pair can never reuse stale direction.
    """
    path = output_dir / "visual_plan.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    by_id = {
        str(item.get("id") or ""): item
        for item in payload
        if isinstance(item, dict) and str(item.get("id") or "")
    }
    scene_ids = [str(scene.get("id") or "") for scene in board.get("scenes") or []]
    if not scene_ids or any(not scene_id or scene_id not in by_id for scene_id in scene_ids):
        return None
    recovered: list[dict] = []
    for scene_id in scene_ids:
        plan = dict(by_id[scene_id])
        # The checkpoint is written after footage/collage attachment.  Those
        # fields are runtime placement state, not direction, and must be
        # stripped before the current manifests attach exactly once again.
        if plan.get("archetype") == "footage" and plan.get("footage_src"):
            plan["archetype"] = "topic"
            for key in tuple(plan):
                if key.startswith("footage_") or key.startswith("collage_"):
                    plan.pop(key, None)
        if plan.get("news_image") or plan.get("archetype") == "news_image":
            plan["archetype"] = str(
                plan.get("news_image_original_archetype") or "topic"
            )
            for key in tuple(plan):
                if key.startswith("news_image"):
                    plan.pop(key, None)
        recovered.append(plan)
    return recovered


def _news_image_placement_error(
    manifest: dict,
    *,
    attached_images: int,
    required_count: int,
) -> str:
    images = manifest.get("images") or []
    acquired_images = len(images) if isinstance(images, list) else 0
    manifest_status = str(manifest.get("status") or "unknown")
    missing_scene_ids = [
        str(scene_id) for scene_id in manifest.get("missing_scene_ids") or []
    ]
    missing_detail = (
        f"; missing scenes={','.join(missing_scene_ids)}"
        if missing_scene_ids
        else ""
    )
    return (
        "News-image placement incomplete: "
        f"scout acquired {acquired_images}/{required_count} licensed images "
        f"(manifest status={manifest_status}{missing_detail}); "
        f"{attached_images}/{required_count} reached final scenes"
    )


def _narration_completeness_failures(alignment: dict) -> list[str]:
    """Return only alignment failures that imply missing spoken content.

    Boundary uncertainty can make captions less precise, but low word/line/audio
    coverage means the WAV cannot represent the full script. That distinction
    lets rendering remain available when timing is merely approximate while
    failing closed on the one-minute-from-a-ten-minute-script failure mode.
    """
    if alignment.get("method") != "whisper_script_forced_alignment":
        return []
    failures = []
    minimum_word_coverage = config.AV_SYNC_MIN_WORD_COVERAGE_PERCENT / 100
    if float(alignment.get("word_coverage") or 0) < minimum_word_coverage:
        failures.append(
            f"matched-word coverage {float(alignment.get('word_coverage') or 0):.1%} "
            f"is below {minimum_word_coverage:.1%}"
        )
    if float(alignment.get("line_coverage") or 0) < 0.75:
        failures.append(
            f"matched-line coverage {float(alignment.get('line_coverage') or 0):.1%} "
            "is below 75.0%"
        )
    if float(alignment.get("audio_coverage") or 0) < 0.80:
        failures.append(
            f"transcript covers only {float(alignment.get('audio_coverage') or 0):.1%} "
            "of the audio"
        )
    return failures


def _orpheus_manifest_failures(
    script_path: str | Path,
    audio_path: str | Path,
    tts_model: str | None,
) -> list[str]:
    """Recheck the fail-closed Orpheus source/audio contract before render."""
    if tts_model != "orpheus-en":
        return []
    from backend.pipeline.tts import _file_sha256, _strip_speaker_labels

    audio = Path(audio_path).resolve()
    manifest_path = audio.parent / "tts_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Orpheus integrity manifest is unavailable: {exc}"]

    failures: list[str] = []
    integrity = manifest.get("integrity") or {}
    if not integrity.get("passed"):
        failures.append("Orpheus per-utterance acoustic verification did not pass")
    if float(integrity.get("verified_source_coverage") or 0) != 1.0:
        failures.append(
            "Orpheus verified source coverage is not 100% "
            f"({float(integrity.get('verified_source_coverage') or 0):.1%})"
        )

    script = Path(script_path).read_text(encoding="utf-8")
    canonical = _strip_speaker_labels(script)
    source_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if source_hash != manifest.get("source_text_sha256"):
        failures.append("the current audio script changed after Orpheus verification")
    if _file_sha256(audio) != manifest.get("output_audio_sha256"):
        failures.append("the narration WAV changed after Orpheus verification")
    return failures


def _detect_silence_boundaries(wav_path: str, log: LogCallback | None = None) -> list[float]:
    command = [
        "ffmpeg", "-i", wav_path,
        # VibeVoice inserts sub-second pauses inside sentences and longer pauses
        # between script paragraphs/turns. Treat only the latter as caption
        # boundaries; otherwise early sentence pauses consume segment slots and
        # leave the final caption on-screen for most of the episode.
        "-af", "silencedetect=noise=-30dB:d=1.2",
        "-f", "null", "-",
    ]
    # NB: do NOT pass the task `log` callback here. silencedetect emits hundreds
    # of stderr lines; fanning each one out per-line to pipeline.log (reopened
    # every line) + the SSE stream + the root logger floods the log pipe and is
    # what produced the "--- Logging error ---" spam. The full output still goes
    # to the module logger as a single record (start.sh log) for debugging, and
    # compose_video emits a concise one-line boundary summary to the task log.
    result = run_capture_logged(
        name="FFmpeg silence detect",
        command=command,
        logger=logger,
        log=None,
        timeout=SILENCE_TIMEOUT,
    )
    boundaries = []
    for line in result.stderr.splitlines():
        match = re.search(r"silence_end: ([\d.]+)", line)
        if match:
            boundaries.append(float(match.group(1)))
    return boundaries


def _project_relative(path: str | Path, project_dir: Path) -> str:
    """Path of ``path`` relative to the render project dir, POSIX-style.

    HyperFrames sandboxes asset access to the project directory, so every
    referenced file must live inside it and be referenced relatively. Falls back
    to an absolute path for assets outside the project (which HyperFrames will
    flag as missing) rather than raising.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _build_render_command(
    project_dir: Path,
    video_path: Path,
    frame: FrameSpec = LANDSCAPE,
) -> list[str]:
    """Build the render command for the current HyperFrames CLI (v0.6.x).

    The render entry is the project directory (which must contain
    ``index.html``) and dimensions come from ``--resolution``. Prefer the
    locally-installed, version-pinned binary so a render never triggers an
    on-demand ``npx`` install; fall back to a *pinned* npx invocation only if the
    local install is missing.
    """
    local_bin = config.HYPERFRAME_DIR / "node_modules" / ".bin" / "hyperframes"
    if local_bin.exists():
        base = [str(local_bin)]
    else:
        base = ["npx", "--yes", f"hyperframes@{config.HYPERFRAMES_VERSION}"]
    return base + [
        "render", str(project_dir),
        "--output", str(video_path),
        "--resolution", frame.render_resolution,
        "--fps", str(config.RENDER_FPS),
        "--quality", config.RENDER_QUALITY,
        "-w", str(config.RENDER_WORKERS),
    ]


def _mount_list(board: dict) -> list[dict]:
    """Every mount needed to cover the timeline with no gaps.

    The first narrated scene begins at zero; there is no separate title-card
    mount before the subject starts.
    """
    mounts = [
        {"id": scene["id"], "start": scene["start"], "duration": scene["duration"]}
        for scene in board["scenes"]
    ]
    mounts.append(
        {
            "id": visual_plan.OUTRO_SCENE_ID,
            "start": float(board["outro_start"]),
            "duration": float(board["outro_duration"]),
        }
    )
    return mounts


def _quality_warnings(
    alignment: dict,
    visual_grounding: dict,
    multimodal: dict | None = None,
) -> list[str]:
    """Summarize advisory A/V findings for the final delivery report."""
    warnings: list[str] = []
    if not alignment.get("passed"):
        detail = "; ".join(
            [
                *[str(value) for value in alignment.get("failure_reasons") or []],
                *[str(value) for value in alignment.get("transcription_failures") or []],
            ]
        )
        warnings.append(
            "Audio timing could not be fully verified"
            + (f": {detail}" if detail else "")
            + "; estimated timing was used."
        )

    if not visual_grounding.get("passed"):
        failed = [
            str(scene.get("id"))
            for scene in visual_grounding.get("scenes") or []
            if not scene.get("grounded")
        ]
        warnings.append(
            "Visual grounding was uncertain"
            + (f" for {', '.join(failed)}" if failed else "")
            + "; rendering continued with the available scene plans."
        )

    if multimodal and multimodal.get("status") not in {"pending", "disabled"}:
        if not multimodal.get("passed"):
            details: list[str] = []
            if multimodal.get("failed_scene_ids"):
                details.append(
                    "low-match scenes "
                    + ", ".join(str(value) for value in multimodal["failed_scene_ids"])
                )
            details.extend(str(value) for value in multimodal.get("errors") or [])
            if not details:
                average = multimodal.get("average_score")
                details.append(
                    f"average score {average} did not meet the configured threshold"
                    if average is not None
                    else "review thresholds were not met"
                )
            warnings.append(
                "Gemini rendered-frame review did not fully pass after retries: "
                + "; ".join(details)
                + "."
            )
    return warnings


def _finalize_quality_report(
    report: dict,
    alignment: dict,
    visual_grounding: dict,
    multimodal: dict,
    *,
    multimodal_enabled: bool,
) -> dict:
    """Mark quality truthfully without turning an advisory miss into no delivery."""
    multimodal_passed = (
        bool(multimodal.get("passed")) if multimodal_enabled else True
    )
    quality_passed = (
        bool(alignment.get("passed"))
        and bool(visual_grounding.get("passed"))
        and multimodal_passed
    )
    warnings = _quality_warnings(alignment, visual_grounding, multimodal)
    report.update(
        {
            "passed": quality_passed,
            "quality_status": "passed" if quality_passed else "warning",
            "delivery_status": (
                "completed_with_warnings" if warnings else "completed"
            ),
            "warnings": warnings,
            "multimodal": multimodal,
        }
    )
    return report


def _kit_plans(
    plans: list[dict],
    mounts: list[dict],
    theme: scene_kit.Theme,
    frame: FrameSpec = LANDSCAPE,
) -> list[scene_kit.ScenePlan]:
    duration_by_id = {mount["id"]: float(mount["duration"]) for mount in mounts}
    return [
        scene_kit.ScenePlan.from_dict(
            plan,
            duration=duration_by_id.get(plan["id"], 6.0),
            scene_id=plan["id"],
            theme=theme,
            frame=frame,
        )
        for plan in plans
    ]


async def compose_video(
    script_path: str,
    audio_path: str,
    output_dir: str,
    title: str = "Podcast Episode",
    include_character: bool = False,
    captions_enabled: bool = True,
    video_template: str = "podcast",
    video_orientation: str = "landscape",
    opening_style: str = "editorial_motion",
    collage_broll_enabled: bool = False,
    collage_broll_count: int = 4,
    news_images_enabled: bool = True,
    news_image_count: int = 4,
    is_monologue: bool = False,
    ai_endpoint: str | None = None,
    ai_model: str | None = None,
    provider_id: int | None = None,
    tts_model: str | None = None,
    background_music_path: str | None = None,
    background_music_bed_db: float = -25.0,
    background_music_duck_db: float = -11.0,
    log: LogCallback | None = None,
) -> str:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    frame = resolve_frame_spec(video_orientation)

    # Mirror to the task log (pipeline.log + LogPanel) when available, else the
    # module logger (start.sh log). Prefer the callback to avoid double-logging.
    emit = lambda message: log(message) if log else logger.info(message)

    # --- 1. Storyboard -----------------------------------------------------
    manifest_failures = _orpheus_manifest_failures(script_path, audio_path, tts_model)
    if manifest_failures:
        detail = "; ".join(manifest_failures)
        emit(f"Narration integrity failed; video render blocked: {detail}")
        raise RuntimeError(
            "Narration audio does not retain a 100% verified script contract; "
            f"refusing to render. {detail}"
        )
    if tts_model == "orpheus-en":
        emit("Narration integrity: Orpheus manifest verifies 100% of source utterances")
    audio_duration = sb.get_audio_duration(audio_path)
    word_transcript, transcription = await av_sync.ensure_word_transcript(
        audio_path,
        output_dir_path,
        log=emit,
    )
    boundaries = _detect_silence_boundaries(audio_path, log)
    emit(f"Audio duration: {audio_duration:.1f}s; {len(boundaries)} silence boundaries detected")

    summary_path = output_dir_path / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None

    board = sb.build_storyboard(
        script_path=script_path,
        audio_duration=audio_duration,
        title=title,
        silence_boundaries=boundaries,
        word_transcript=word_transcript,
        minimum_word_coverage=config.AV_SYNC_MIN_WORD_COVERAGE_PERCENT / 100,
        maximum_boundary_uncertainty=config.AV_SYNC_MAX_BOUNDARY_UNCERTAINTY_MS / 1000,
        is_monologue=is_monologue,
        summary=summary,
        log=emit,
    )
    board["alignment"]["transcription_backend"] = transcription.get("backend", "unavailable")
    board["alignment"]["transcription_attempts"] = transcription.get("attempts", 0)
    if transcription.get("failure_reasons"):
        board["alignment"]["transcription_failures"] = transcription["failure_reasons"]
    sb.write_storyboard(output_dir_path, board)
    completeness_failures = _narration_completeness_failures(board["alignment"])
    if completeness_failures:
        detail = "; ".join(completeness_failures)
        emit(f"Narration integrity failed; video render blocked: {detail}")
        raise RuntimeError(
            "Narration audio does not cover the full script; refusing to render "
            f"a truncated video. {detail}"
        )
    if board["alignment"].get("passed"):
        emit(
            "A/V sync timing: "
            f"{float(board['alignment'].get('word_coverage') or 0):.1%} word coverage; "
            f"max boundary uncertainty "
            f"{float(board['alignment'].get('max_boundary_uncertainty_seconds') or 0):.2f}s"
        )
    else:
        emit("A/V sync warning: acoustic timing is unverified; rendering will continue")
    emit(f"Composition {board['total_duration']:.1f}s over {board['scene_count']} scenes")

    # --- 2. Direction ------------------------------------------------------
    plans = _load_cached_scene_plans(output_dir_path, board)
    if plans is not None:
        emit(f"Visual plan: reusing {len(plans)}/{board['scene_count']} cached scene plan(s)")
    else:
        plans = await visual_plan.plan_scene_visuals(
            board,
            ai_endpoint=ai_endpoint,
            ai_model=ai_model,
            provider_id=provider_id,
            log=emit,
        )

    manifest = footage.read_manifest(output_dir_path)
    attached = visual_plan.attach_footage(plans, board, manifest, output_dir_path)
    if attached:
        emit(f"Footage: {attached} manifest clip(s) placed as full-bleed scenes")
    requested_footage = int((manifest or {}).get("requested_clip_count") or 0)
    acquired_footage = len((manifest or {}).get("clips") or [])

    force_collage_opening = opening_style == "paper_collage"
    if collage_broll_enabled or force_collage_opening:
        requested_collages = collage_broll_count if collage_broll_enabled else 1
        collage_manifest = await collage_broll.generate_collage_broll(
            board,
            output_dir_path,
            count=requested_collages,
            force_opening=force_collage_opening,
            frame=frame,
            provider_id=provider_id,
            ai_endpoint=ai_endpoint,
            ai_model=ai_model,
            log=emit,
        )
        collage_attached = collage_broll.attach_collage(
            plans, collage_manifest, output_dir_path
        )
        emit(
            f"Collage B-roll: {collage_attached}/{requested_collages} generated "
            f"clip(s) placed as {frame.aspect_ratio} full-bleed scenes"
        )

    final_public_footage = sum(
        1
        for plan in plans
        if plan.get("archetype") == "footage" and not plan.get("collage_broll")
    )
    final_collages = sum(1 for plan in plans if plan.get("collage_broll"))
    if requested_footage and final_public_footage != requested_footage:
        raise RuntimeError(
            "Public-footage placement incomplete: "
            f"{acquired_footage}/{requested_footage} clips were acquired and "
            f"{final_public_footage}/{requested_footage} reached final scenes"
        )
    if collage_broll_enabled or force_collage_opening:
        if final_collages != requested_collages:
            raise RuntimeError(
                "Collage B-roll placement incomplete: "
                f"{final_collages}/{requested_collages} requested clips reached final scenes"
            )

    news_image_inventory = {"attached": 0, "placement_modes": {"inline": 0, "fullscreen": 0}}
    eligible_image_scene_ids = {
        str(plan.get("id") or "")
        for plan in plans
        if not plan.get("collage_broll") and plan.get("archetype") != "footage"
    }
    if news_images_enabled and eligible_image_scene_ids:
        image_manifest = await news_images.acquire_news_images(
            board,
            output_dir_path,
            count=news_image_count,
            excluded_scene_ids={
                str(plan.get("id") or "")
                for plan in plans
                if str(plan.get("id") or "") not in eligible_image_scene_ids
            },
            scene_hints={str(plan.get("id") or ""): plan for plan in plans},
            log=emit,
        )
        news_image_inventory = news_images.attach_news_images(
            plans, board, image_manifest, output_dir_path
        )
        required_count = min(news_image_count, len(eligible_image_scene_ids))
        attached_images = int(news_image_inventory["attached"])
        image_modes = news_image_inventory["placement_modes"]
        if attached_images != required_count:
            raise RuntimeError(
                _news_image_placement_error(
                    image_manifest,
                    attached_images=attached_images,
                    required_count=required_count,
                )
            )
        if required_count >= 2 and (
            not image_modes.get("inline") or not image_modes.get("fullscreen")
        ):
            raise RuntimeError(
                "News-image placement incomplete: both inline and fullscreen modes "
                "are required when at least two eligible scenes exist"
            )
    emit(
        "Final B-roll inventory: "
        f"{final_public_footage} public footage clip(s), "
        f"{final_collages} paper-collage clip(s), "
        f"{news_image_inventory['attached']} news image(s) "
        f"(inline={news_image_inventory['placement_modes']['inline']}, "
        f"fullscreen={news_image_inventory['placement_modes']['fullscreen']})"
    )

    scene_plans = list(plans)
    visual_grounding = visual_plan.visual_grounding_report(scene_plans, board)
    quality_report = {
        "passed": bool(board["alignment"].get("passed")) and visual_grounding["passed"],
        "quality_status": "pending",
        "delivery_status": "pending",
        "warnings": _quality_warnings(board["alignment"], visual_grounding),
        "narration": {
            "provider": config.tts_provider_label(tts_model),
            "model": tts_model or config.TTS_DEFAULT_MODEL,
            "audio_path": str(Path(audio_path).resolve()),
        },
        "alignment_analyzer": {
            "provider": "MLX Whisper",
            "role": "post-TTS word timestamps only; does not generate narration",
            "backend": transcription.get("backend", "unavailable"),
            "attempts": transcription.get("attempts", 0),
        },
        "timing": board["alignment"],
        "visual_grounding": visual_grounding,
        "multimodal": {"status": "pending", "passed": False},
        "background_music": {"enabled": bool(background_music_path), "passed": None},
    }
    (output_dir_path / "av_sync_report.json").write_text(
        json.dumps(quality_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if visual_grounding["passed"]:
        emit(
            f"A/V sync semantics: {visual_grounding['grounded_scenes']}/"
            f"{visual_grounding['scene_count']} scene plans grounded in their narration"
        )
    else:
        failed = [scene["id"] for scene in visual_grounding["scenes"] if not scene["grounded"]]
        emit(
            "A/V sync warning: visual grounding was uncertain for "
            f"{', '.join(failed)}; rendering will continue"
        )

    program_audio_path = Path(audio_path)
    if background_music_path:
        source_music = Path(background_music_path)
        if not source_music.is_file():
            raise RuntimeError(f"Configured background music is missing: {source_music}")
        program_audio_path = await music.mix_narration_and_music(
            audio_path,
            source_music,
            output_dir_path,
            bed_db=background_music_bed_db,
            duck_db=background_music_duck_db,
            log=emit,
        )
        quality_report["background_music"] = {
            "enabled": True,
            "passed": True,
            "source": str(source_music.resolve()),
            "program_mix": str(program_audio_path.resolve()),
            "report": str((output_dir_path / "audio" / "music_mix_report.json").resolve()),
        }

    plans = scene_plans + [visual_plan.outro_plan(board)]
    (output_dir_path / "visual_plan.json").write_text(
        json.dumps(plans, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- 3. Authoring ------------------------------------------------------
    mounts = _mount_list(board)
    theme = scene_kit.resolve_theme(video_template)
    kit_plans = _kit_plans(plans, mounts, theme, frame)

    assembler.vendor_assets(output_dir_path, include_lottie=include_character)
    character_src = assembler.stage_character(output_dir_path) if include_character else None
    assembler.write_scene_files(output_dir_path, kit_plans)
    emit(f"Wrote {len(kit_plans)} deterministic scene draft(s)")

    # The crews talk to the same provider as the digest stage, so they need its
    # resolved credentials — not just the task's overrides.
    from backend.pipeline.digester import _resolve_provider

    try:
        endpoint, model, api_key = await _resolve_provider(provider_id, ai_endpoint, ai_model)
    except Exception as exc:  # noqa: BLE001 - the deterministic scenes still stand
        emit(f"Provider lookup failed ({exc}); rendering the deterministic scenes")
        endpoint, model, api_key = None, None, None

    # Image scenes stay on the deterministic renderer. This prevents an authoring
    # agent from "improving" the composition by dropping a licensed asset whose
    # exact placement is part of the delivery contract.
    director_plans = [
        plan
        for plan in scene_plans
        if not plan.get("collage_broll") and not plan.get("news_image")
    ]
    if config.DIRECTOR_ENABLED and director_plans and model:
        budget = director_plans[: config.DIRECTOR_MAX_SCENES] if config.DIRECTOR_MAX_SCENES else director_plans
        try:
            outcome = await director.direct_scenes(
                output_dir_path,
                board,
                budget,
                kit_plans,
                model=model,
                endpoint=endpoint,
                api_key=api_key,
                log=emit,
                frame=frame,
            )
            (output_dir_path / "director_report.json").write_text(
                json.dumps(
                    {
                        "authored": outcome.authored,
                        "rejected": outcome.rejected,
                        "failures": outcome.failures,
                        "agents_run": outcome.agents_run,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - the deterministic scenes still stand
            emit(f"Director unavailable ({exc}); rendering the deterministic scenes")
            outcome = director.DirectorOutcome()
    else:
        emit("Director agents disabled; rendering the deterministic scenes")
        outcome = director.DirectorOutcome()

    # --- 4. Assembly -------------------------------------------------------
    audio_src = _project_relative(program_audio_path, output_dir_path)
    spine = assembler.build_spine(
        board,
        audio_src=audio_src,
        mounts=mounts,
        theme=theme,
        frame=frame,
        character_src=character_src,
        captions_enabled=captions_enabled,
    )
    composition_path = assembler.write_spine(output_dir_path, spine)
    emit(f"Composition written to {composition_path}")

    clean, lint_output = assembler.lint_project(output_dir_path, emit)
    if not clean and outcome.authored:
        # Only agent-authored files can be wrong here — the spine and the kit are
        # generated. Revert exactly the scenes lint named, then re-check.
        blamed = director.scenes_named_in(lint_output, outcome.authored)
        if blamed:
            director.revert_scenes(output_dir_path, blamed, kit_plans)
            emit(f"Lint: reverted {len(blamed)} agent scene(s) to the deterministic draft")
            clean, lint_output = assembler.lint_project(output_dir_path, emit)
    if not clean:
        # Last resort: every scene goes back to the known-good renderer rather
        # than shipping a composition the runtime may refuse to drive.
        director.revert_scenes(output_dir_path, [plan.id for plan in kit_plans], kit_plans)
        emit("Lint still failing; reverted every scene to the deterministic kit")
        outcome.authored.clear()
        clean, lint_output = assembler.lint_project(output_dir_path, emit)
        if not clean:
            raise RuntimeError(f"Composition failed HyperFrames lint:\n{lint_output[-1200:]}")

    # Layout check. Overlapping text and content spilling out of a card do not
    # stop a render — they just make it look broken — so this loop is advisory:
    # give the agents HyperFrames' own findings, and revert anything still
    # failing afterwards rather than blocking the video.
    if config.INSPECT_ENABLED:
        layout_ok, findings = assembler.inspect_project(output_dir_path, emit)
        blamed = director.scenes_named_in_findings(findings, outcome.authored, mounts)
        if not layout_ok and blamed:
            try:
                await director.repair_scenes(
                    output_dir_path,
                    blamed,
                    findings,
                    model=model,
                    endpoint=endpoint,
                    api_key=api_key,
                    log=emit,
                    frame=frame,
                    theme=theme,
                )
            except Exception as exc:  # noqa: BLE001 - layout polish is never fatal
                emit(f"Director repair pass unavailable ({exc})")
            clean, lint_output = assembler.lint_project(output_dir_path, emit)
            if not clean:
                director.revert_scenes(output_dir_path, blamed, kit_plans)
                emit("Repair broke lint; reverted those scenes to the deterministic draft")
            else:
                layout_ok, findings = assembler.inspect_project(output_dir_path, emit)
                still_bad = director.scenes_named_in_findings(findings, blamed, mounts)
                if still_bad:
                    director.revert_scenes(output_dir_path, still_bad, kit_plans)
                    emit(
                        f"Layout still failing for {len(still_bad)} scene(s); "
                        "reverted them to the deterministic draft"
                    )

    # --- 5. Render ---------------------------------------------------------
    video_path = output_dir_path / "video.mp4"
    total_frames = max(1, round(float(board["total_duration"]) * config.RENDER_FPS))
    render_timeout = max(RENDER_TIMEOUT_FLOOR, int(300 + total_frames * RENDER_SECONDS_PER_FRAME))
    emit(
        f"Rendering ~{total_frames} frames "
        f"({board['total_duration']:.0f}s @ {config.RENDER_FPS}fps, {config.RENDER_QUALITY}, "
        f"{config.RENDER_WORKERS} worker(s)); render timeout {render_timeout}s, "
        f"stall timeout {RENDER_STALL_TIMEOUT}s"
    )

    render_command = _build_render_command(output_dir_path, video_path, frame)
    returncode, output = await stream_subprocess(
        name="HyperFrames render",
        command=render_command,
        logger=logger,
        log=log,
        cwd=config.HYPERFRAME_DIR,
        timeout=render_timeout,
        stall_timeout=RENDER_STALL_TIMEOUT,
    )

    if returncode != 0:
        raise RuntimeError(f"Video render failed (exit {returncode}): {output[-500:]}")

    if not video_path.exists():
        raise RuntimeError("Video render produced no output file")

    emit(f"Video rendered: {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")
    if config.AV_SYNC_GEMINI_REVIEW_ENABLED:
        gemini_review = await multimodal_review.review_video(
            video_path,
            board,
            output_dir_path,
            log=emit,
        )
    else:
        gemini_review = {
            "status": "disabled",
            "passed": None,
            "analyzer": "gemini-web-via-opencli",
            "scenes": [],
        }
    _finalize_quality_report(
        quality_report,
        board["alignment"],
        visual_grounding,
        gemini_review,
        multimodal_enabled=config.AV_SYNC_GEMINI_REVIEW_ENABLED,
    )
    (output_dir_path / "av_sync_report.json").write_text(
        json.dumps(quality_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if gemini_review.get("passed"):
        emit(
            "Gemini A/V match: "
            f"{gemini_review['reviewed_scenes']}/{gemini_review['scene_count']} scenes passed; "
            f"average {gemini_review['average_score']:.1f}/100"
        )
    elif config.AV_SYNC_GEMINI_REVIEW_ENABLED:
        emit(
            "A/V sync warning: Gemini review did not fully pass after retries; "
            "video delivery continues and details are recorded in av_sync_report.json"
        )
    if quality_report["warnings"]:
        emit(
            f"Video completed with {len(quality_report['warnings'])} A/V quality "
            "warning(s); see av_sync_report.json"
        )
    return str(video_path)
