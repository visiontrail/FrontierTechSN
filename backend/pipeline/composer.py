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
import math
import os
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from backend import config
from backend.pipeline import (
    av_sync,
    assembler,
    collage_broll,
    director,
    footage,
    intros,
    music,
    media_shots,
    multimodal_review,
    news_images,
    news_webpages,
    outros,
    scene_kit,
    storyboard as sb,
    visual_plan,
)
from backend.pipeline.process_logging import run_capture_logged, stream_subprocess
from backend.pipeline.video_format import (
    FrameSpec,
    LANDSCAPE,
    RenderSpec,
    resolve_frame_spec,
    resolve_render_spec,
)

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
RENDER_STALL_TIMEOUT = max(600, math.ceil(config.RENDER_PROTOCOL_TIMEOUT_MS / 1000) + 60)

VISUAL_PLAN_CACHE_FILENAME = "visual_plan.cache.json"
VISUAL_PLAN_CACHE_VERSION = 1
VISUAL_PLAN_PROMPT_CONTRACT_VERSION = 2
QUALITY_RETRY_STATE_FILENAME = "quality_retry_state.json"
_VISUAL_PLAN_CACHE_KEYS = {
    "cache_version",
    "planner_input_sha256",
    "prompt_contract_version",
    "prompt_sha256",
    "visual_plan_bytes",
    "visual_plan_sha256",
}


class QualityGateRetry(RuntimeError):
    """A rendered candidate needs another compose/review cycle, not FAILED."""


def _eligible_news_image_scene_ids(plans: list[dict], board: dict) -> set[str]:
    bookends = {
        str(scene.get("id") or "")
        for scene in board.get("scenes", [])
        if scene.get("program_segment_kind") in {"opening", "closing"}
    }
    return {
        str(plan.get("id") or "")
        for plan in plans
        if not plan.get("collage_broll")
        and plan.get("archetype") != "footage"
        and str(plan.get("id") or "") not in bookends
    }


def _record_quality_retry(
    output_dir: Path,
    quality_report: dict,
    video_sha256: str,
) -> dict:
    path = output_dir / QUALITY_RETRY_STATE_FILENAME
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous = {}
    history = previous.get("history") if isinstance(previous, dict) else []
    if not isinstance(history, list):
        history = []
    multimodal = quality_report.get("multimodal") or {}
    attempt = {
        "attempt": len(history) + 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rendered_video_sha256": video_sha256,
        "failed_scene_ids": [
            str(value) for value in multimodal.get("failed_scene_ids") or []
        ],
        "average_score": multimodal.get("average_score"),
        "warnings": [str(value) for value in quality_report.get("warnings") or []],
        "scene_reviews": [
            {
                "id": str(review.get("id") or ""),
                "score": review.get("score"),
                "review_available": review.get("rubric_consistent") is True,
                "issues": [str(value) for value in review.get("issues") or []],
                "suggested_visual": str(review.get("suggested_visual") or ""),
            }
            for review in multimodal.get("scenes") or []
            if isinstance(review, dict) and review.get("passed") is not True
        ],
    }
    history.append(attempt)
    state = {
        "status": "retrying",
        "attempt_count": len(history),
        "latest": attempt,
        "history": history[-20:],
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
    return state


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _strict_json_loads(payload: bytes) -> object:
    value = json.loads(
        payload.decode("utf-8"),
        parse_constant=_reject_json_constant,
    )

    def finite(item: object) -> bool:
        if isinstance(item, float):
            return math.isfinite(item)
        if isinstance(item, list):
            return all(finite(child) for child in item)
        if isinstance(item, dict):
            return all(isinstance(key, str) and finite(child) for key, child in item.items())
        return item is None or isinstance(item, (str, int, bool))

    if not finite(value):
        raise ValueError("JSON contains a non-finite or unsupported value")
    return value


def _finite_storyboard_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"storyboard {field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"storyboard {field} must be a finite number")
    return number


def _visual_plan_planner_input(board: dict) -> dict:
    thesis = board.get("thesis", "")
    scenes = board.get("scenes") or []
    if not isinstance(thesis, str) or not isinstance(scenes, list) or not scenes:
        raise ValueError("storyboard is missing visual-plan planner input")

    semantic_scenes: list[dict] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            raise ValueError("storyboard scenes must be objects")
        scene_id = scene.get("id")
        text = scene.get("text")
        index = scene.get("index")
        keywords = scene.get("keywords") or []
        if (
            not isinstance(scene_id, str)
            or not scene_id
            or not isinstance(text, str)
            or type(index) is not int
            or index < 0
            or not isinstance(keywords, list)
            or any(not isinstance(keyword, str) for keyword in keywords)
        ):
            raise ValueError("storyboard scene semantics are malformed")
        semantic_scenes.append(
            {
                "id": scene_id,
                "index": index,
                "text": text,
                "start": _finite_storyboard_number(scene.get("start"), f"{scene_id}.start"),
                "duration": _finite_storyboard_number(
                    scene.get("duration"), f"{scene_id}.duration"
                ),
                "keywords": keywords,
            }
        )
    return {"thesis": thesis, "scenes": semantic_scenes}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _visual_plan_cache_contract(board: dict) -> dict:
    prompt_bytes = (config.PROMPTS_DIR / "visual_plan.txt").read_bytes()
    return {
        "cache_version": VISUAL_PLAN_CACHE_VERSION,
        "planner_input_sha256": _canonical_sha256(_visual_plan_planner_input(board)),
        "prompt_contract_version": VISUAL_PLAN_PROMPT_CONTRACT_VERSION,
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_visual_plan_checkpoint(output_dir: Path, board: dict, plans: list[dict]) -> None:
    plan_path = output_dir / "visual_plan.json"
    cache_path = output_dir / VISUAL_PLAN_CACHE_FILENAME
    plan_bytes = json.dumps(
        plans,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    contract = {
        **_visual_plan_cache_contract(board),
        "visual_plan_bytes": len(plan_bytes),
        "visual_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
    }
    cache_bytes = json.dumps(
        contract,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    # The sidecar is the commit marker. Invalidate it before replacing the plan,
    # then publish the new sidecar last so every interrupted write fails closed.
    cache_path.unlink(missing_ok=True)
    _atomic_write_bytes(plan_path, plan_bytes)
    _atomic_write_bytes(cache_path, cache_bytes)


def _load_cached_scene_plans(output_dir: Path, board: dict) -> list[dict] | None:
    """Recover the last fully written direction pass after a late render failure.

    ``visual_plan.json`` is committed once immediately after direction and may be
    enriched after collage, image, and music stages complete. A current sidecar
    binds its exact bytes to the ordered storyboard semantics and visual-planner
    prompt contract. Legacy, partial, tampered, and stale checkpoints fail closed.
    """
    path = output_dir / "visual_plan.json"
    cache_path = output_dir / VISUAL_PLAN_CACHE_FILENAME
    if not path.is_file() or not cache_path.is_file():
        return None
    try:
        plan_bytes = path.read_bytes()
        cache = _strict_json_loads(cache_path.read_bytes())
        expected_contract = _visual_plan_cache_contract(board)
        if (
            not isinstance(cache, dict)
            or set(cache) != _VISUAL_PLAN_CACHE_KEYS
            or type(cache.get("cache_version")) is not int
            or type(cache.get("prompt_contract_version")) is not int
            or type(cache.get("visual_plan_bytes")) is not int
            or cache.get("visual_plan_bytes") != len(plan_bytes)
            or cache.get("visual_plan_sha256") != hashlib.sha256(plan_bytes).hexdigest()
            or any(cache.get(key) != value for key, value in expected_contract.items())
        ):
            return None
        payload = _strict_json_loads(plan_bytes)
    except (
        OSError,
        RecursionError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return None
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        return None
    scene_ids = [str(scene.get("id") or "") for scene in board.get("scenes") or []]
    expected_ids = scene_ids
    payload_ids = [item.get("id") for item in payload]
    if not scene_ids or any(not scene_id for scene_id in scene_ids) or payload_ids != expected_ids:
        return None
    by_id = {str(item["id"]): item for item in payload}
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
        for key in tuple(plan):
            if key.startswith("news_webpage") or key == "media_shots":
                plan.pop(key, None)
        recovered.append(plan)
    return recovered


def _pending_visual_repairs(output_dir: Path, plans: list[dict]) -> tuple[str, list[dict]]:
    """Use feedback only for the retained candidate, once per scene checkpoint."""
    try:
        state = _strict_json_loads((output_dir / "quality_retry_state.json").read_bytes())
        latest = state.get("latest") or {}
        source_hash = latest.get("rendered_video_sha256")
        candidate = output_dir / "video.next.mp4"
        if (
            state.get("status") != "retrying"
            or not isinstance(source_hash, str)
            or not candidate.is_file()
            or _sha256_path(candidate) != source_hash
        ):
            return "", []
        by_id = {plan["id"]: plan for plan in plans}
        failed = set(latest.get("failed_scene_ids") or [])
        feedback = []
        for review in latest.get("scene_reviews") or []:
            scene_id = review.get("id")
            previous = by_id.get(scene_id)
            if (
                scene_id not in failed
                or previous is None
                or review.get("review_available") is not True
                or (
                    previous.get("review_repair_source_sha256") == source_hash
                    and previous.get("grounding_source") != "narration_fallback"
                )
            ):
                continue
            feedback.append({
                "id": scene_id,
                "issues": review.get("issues") or [],
                "suggested_visual": review.get("suggested_visual") or "",
                "previous_copy": {
                    key: previous.get(key)
                    for key in ("headline", "body", "stat", "stat_label", "items", "quote", "attribution")
                },
            })
        return source_hash, feedback
    except (OSError, ValueError, TypeError, AttributeError):
        return "", []


async def _load_or_plan_scene_visuals(
    output_dir: Path,
    board: dict,
    *,
    ai_endpoint: str | None,
    ai_model: str | None,
    provider_id: int | None,
    log: LogCallback,
) -> list[dict]:
    plans = _load_cached_scene_plans(output_dir, board)
    if plans is not None:
        log(f"Visual plan: reusing {len(plans)}/{board['scene_count']} cached scene plan(s)")
        source_hash, feedback = _pending_visual_repairs(output_dir, plans)
        if feedback:
            failed_ids = {row["id"] for row in feedback}
            repair_scenes = [scene for scene in board["scenes"] if scene["id"] in failed_ids]
            log(f"Visual plan: repairing {len(repair_scenes)} failed scene(s) from rendered-frame feedback")
            repaired = await visual_plan.plan_scene_visuals(
                {**board, "scenes": repair_scenes, "scene_count": len(repair_scenes),
                 "visual_review_feedback": feedback},
                ai_endpoint=ai_endpoint,
                ai_model=ai_model,
                provider_id=provider_id,
                log=log,
            )
            by_id = {plan["id"]: plan for plan in repaired}
            if set(by_id) != failed_ids:
                raise RuntimeError("Visual review repair did not return every failed scene")
            unavailable = [
                plan["id"] for plan in repaired
                if plan.get("grounding_source") == "narration_fallback"
            ]
            if unavailable:
                raise RuntimeError(
                    "Visual review repair provider did not supply corrected direction for "
                    + ", ".join(unavailable)
                    + "; retaining the previous plan and pending review feedback"
                )
            unchanged = [
                row["id"] for row in feedback
                if all(by_id[row["id"]].get(key) == value
                       for key, value in row["previous_copy"].items())
            ]
            if unchanged:
                raise RuntimeError(
                    "Visual review repair returned unchanged visible content for "
                    + ", ".join(unchanged)
                )
            for repaired_plan in repaired:
                repaired_plan["review_repair_source_sha256"] = source_hash
                repaired_plan["visual_review_feedback"] = next(
                    {key: row[key] for key in ("issues", "suggested_visual")}
                    for row in feedback if row["id"] == repaired_plan["id"]
                )
            plans = [by_id.get(plan["id"], plan) for plan in plans]
            _write_visual_plan_checkpoint(output_dir, board, plans)
        return plans

    plans = await visual_plan.plan_scene_visuals(
        board,
        ai_endpoint=ai_endpoint,
        ai_model=ai_model,
        provider_id=provider_id,
        log=log,
    )
    # Direction is the expensive checkpoint. Commit it before any footage,
    # collage, image, or music stage can fail so a retry does not repeat the
    # planner call. Runtime placement fields are added only to the later rewrite.
    _write_visual_plan_checkpoint(output_dir, board, plans)
    return plans


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


def _narration_completeness_failures(
    alignment: dict,
    *,
    source_contract_verified: bool = False,
) -> list[str]:
    """Return only alignment failures that imply missing spoken content.

    Boundary uncertainty can make captions less precise, but low word/line/audio
    coverage means the WAV cannot represent the full script. That distinction
    lets rendering remain available when timing is merely approximate while
    failing closed on the one-minute-from-a-ten-minute-script failure mode.
    """
    # Orpheus is verified utterance-by-utterance before concatenation. Its
    # manifest binds the exact script and complete WAV hashes and requires
    # 100% verified source coverage. A second Whisper pass over the paced
    # program audio is useful for approximate visual timing, but long inserted
    # music gaps and fused proper-name spellings must not override that stronger
    # completeness proof. Unmanifested and non-Orpheus audio remain fail-closed.
    if source_contract_verified:
        return []
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


def _narration_manifest_failures(
    script_path: str | Path,
    audio_path: str | Path,
    tts_model: str | None,
) -> list[str]:
    """Recheck the generated narration's source/audio contract before render."""
    from backend.pipeline.tts import (
        NARRATION_PACING_POLICY,
        NARRATION_SYNTHESIS_SPEED_RATIO,
        _expand_vibevoice_pronunciations,
        _file_sha256,
        _strip_speaker_labels,
    )

    audio = Path(audio_path).resolve()
    manifest_path = audio.parent / "tts_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"narration integrity manifest is unavailable: {exc}"]
    if not isinstance(manifest, dict):
        return ["narration integrity manifest must be a JSON object"]

    failures: list[str] = []
    if manifest.get("pacing_policy") != NARRATION_PACING_POLICY:
        failures.append(
            "the narration manifest does not prove the natural-speech, "
            "visuals-follow-audio pacing policy"
        )
    speed_ratio = manifest.get("synthesis_speed_ratio")
    if (
        isinstance(speed_ratio, bool)
        or not isinstance(speed_ratio, (int, float))
        or not math.isfinite(float(speed_ratio))
        or float(speed_ratio) != NARRATION_SYNTHESIS_SPEED_RATIO
    ):
        failures.append(
            "the narration manifest does not prove natural 1.0x synthesis speed "
            f"(recorded {speed_ratio!r})"
        )
    manifest_model = str(manifest.get("model") or "")
    effective_model = tts_model or config.TTS_DEFAULT_MODEL
    if manifest_model and manifest_model != effective_model:
        failures.append(
            "the narration manifest model does not match the current task "
            f"({manifest_model} != {effective_model})"
        )

    model = config.TTS_MODELS.get(effective_model)
    if model is None:
        failures.append(f"the configured TTS model is unknown ({effective_model})")
        return failures

    script = Path(script_path).read_text(encoding="utf-8")
    canonical = (
        script.strip()
        if model.get("requires_speaker_labels")
        else _strip_speaker_labels(script)
    )
    manifest_source = canonical
    if model.get("kind") == "local_subprocess":
        manifest_source, _ = _expand_vibevoice_pronunciations(canonical)
    source_hash = hashlib.sha256(manifest_source.encode("utf-8")).hexdigest()
    if source_hash != manifest.get("source_text_sha256"):
        failures.append("the current script changed after narration generation")
    if _file_sha256(audio) != manifest.get("output_audio_sha256"):
        failures.append("the narration WAV changed after narration generation")

    if model.get("acoustic_integrity"):
        provider = str(model.get("provider") or effective_model)
        integrity = manifest.get("integrity") or {}
        if not isinstance(integrity, dict):
            failures.append(f"{provider} integrity report must be a JSON object")
            return failures
        if not integrity.get("passed"):
            failures.append(f"{provider} per-part acoustic verification did not pass")
        if float(integrity.get("verified_source_coverage") or 0) != 1.0:
            failures.append(
                f"{provider} verified source coverage is not 100% "
                f"({float(integrity.get('verified_source_coverage') or 0):.1%})"
            )
    return failures


def _manifest_program_segments(
    audio_path: str | Path,
    physical_segments: list[dict],
) -> list[dict] | None:
    """Recover exact line timing from verified concatenated TTS chunks.

    Each remote-provider part is independently hash-bound and acoustically verified.
    Matching the part texts back to physical script lines therefore provides a
    stronger program-boundary contract than transcribing the silence-padded
    full file again with Whisper.
    """
    audio = Path(audio_path).resolve()
    manifest_path = audio.parent / "tts_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    integrity = manifest.get("integrity") or {}
    parts = manifest.get("parts") or []
    if (
        not isinstance(integrity, dict)
        or integrity.get("passed") is not True
        or float(integrity.get("verified_source_coverage") or 0) != 1.0
        or not isinstance(parts, list)
        or not parts
    ):
        return None

    def canonical(value: object) -> str:
        return " ".join(str(value or "").split()).casefold()

    part_rows: list[tuple[str, float]] = []
    for part in parts:
        if not isinstance(part, dict):
            return None
        input_name = str(part.get("input") or "")
        duration = float(part.get("duration_seconds") or 0)
        input_path = audio.parent / input_name
        if not input_name or duration <= 0 or not input_path.is_file():
            return None
        part_rows.append((canonical(input_path.read_text(encoding="utf-8")), duration))

    aligned: list[dict] = []
    part_index = 0
    cursor = 0.0
    for segment in physical_segments:
        target = canonical(segment.get("text"))
        accumulated: list[str] = []
        duration = 0.0
        while part_index < len(part_rows):
            text, part_duration = part_rows[part_index]
            part_index += 1
            accumulated.append(text)
            duration += part_duration
            candidate = canonical(" ".join(accumulated))
            if candidate == target:
                break
            if not target.startswith(candidate):
                return None
        else:
            return None
        aligned.append(
            {
                **segment,
                "start": cursor,
                "duration": duration,
                "timing_source": "verified_chunk_manifest",
            }
        )
        cursor += duration
    if part_index != len(part_rows) or abs(cursor - sum(row[1] for row in part_rows)) > 0.001:
        return None
    return aligned


def _apply_manifest_program_alignment(
    alignment: dict,
    pacing_report: dict,
    manifest_segments: list[dict] | None,
    *,
    provider_label: str = "Orpheus",
) -> dict:
    """Promote the exact verified chunk/program contract over whole-file ASR."""
    if not manifest_segments or pacing_report.get("passed") is not True:
        return alignment
    original = dict(alignment)
    verified_words = sum(int(row.get("word_count") or 0) for row in manifest_segments)
    alignment.update(
        {
            "method": f"{provider_label.casefold().replace(' ', '_')}_manifest_program_timeline",
            "script_words": verified_words,
            "transcript_words": verified_words,
            "matched_words": verified_words,
            "word_coverage": 1.0,
            "line_coverage": 1.0,
            "audio_coverage": 1.0,
            "max_boundary_uncertainty_seconds": 0.0,
            "passed": True,
            "failure_reasons": [],
            "source_contract": {
                "provider": provider_label,
                "per_utterance_acoustic_coverage": 1.0,
                "program_pacing_report_passed": True,
                "physical_segment_count": len(manifest_segments),
            },
            "whole_file_asr_observation": original,
        }
    )
    return alignment


def _enforce_program_opening_copy(plans: list[dict], storyboard: dict) -> None:
    """Keep the exact show identity/date visible on the music-led opener."""
    if not plans or not storyboard.get("program_timeline"):
        return
    scenes = storyboard.get("scenes") or []
    if not scenes:
        return
    opening = str(scenes[0].get("text") or "")
    match = re.search(
        r"\bIt(?:'s| is)\s+(?P<date>.+?\d{4}),\s+and this is\s+"
        r"(?P<title>[^—–.,]+)",
        opening,
        flags=re.IGNORECASE,
    )
    if not match:
        return
    plans[0]["kicker"] = match.group("date").strip().upper()
    plans[0]["headline"] = match.group("title").strip()
    plans[0]["body"] = ""


def _orpheus_manifest_failures(
    script_path: str | Path,
    audio_path: str | Path,
    tts_model: str | None,
) -> list[str]:
    """Compatibility wrapper retained for focused integrity callers/tests."""
    if tts_model != "orpheus-en":
        return []
    return _narration_manifest_failures(script_path, audio_path, tts_model)


def _rendered_video_failures(
    video_path: Path,
    *,
    frame: FrameSpec,
    expected_duration: float,
    render: RenderSpec | None = None,
) -> list[str]:
    """Probe a staged render before it can replace the last completed cut."""
    render = render or resolve_render_spec(frame)
    result = run_capture_logged(
        name="Rendered video probe",
        command=[
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(video_path),
        ],
        logger=logger,
        timeout=30,
        stdout_log_limit=2000,
        stderr_log_limit=1000,
    )
    if result.returncode != 0:
        return [f"ffprobe exited {result.returncode}: {(result.stderr or '')[-500:]}"]
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return [f"ffprobe returned invalid JSON: {exc}"]
    streams = payload.get("streams") or []
    videos = [item for item in streams if item.get("codec_type") == "video"]
    audios = [item for item in streams if item.get("codec_type") == "audio"]
    failures: list[str] = []
    if len(videos) != 1:
        failures.append(f"expected one video stream, found {len(videos)}")
    else:
        width = int(videos[0].get("width") or 0)
        height = int(videos[0].get("height") or 0)
        if (width, height) != (render.width, render.height):
            failures.append(
                f"render dimensions are {width}x{height}, expected "
                f"{render.width}x{render.height}"
            )
        for field in ("avg_frame_rate", "r_frame_rate"):
            try:
                numerator, denominator = str(videos[0].get(field) or "0/1").split("/")
                fps = float(numerator) / float(denominator)
            except (ValueError, ZeroDivisionError):
                fps = 0
            if not math.isfinite(fps) or abs(fps - render.fps) > 0.01:
                failures.append(f"render {field} is {fps:g}, expected {render.fps} fps")
    if len(audios) != 1:
        failures.append(f"expected one audio stream, found {len(audios)}")
    try:
        duration = float(
            (payload.get("format") or {}).get("duration")
            or (videos[0].get("duration") if videos else 0)
            or 0
        )
    except (TypeError, ValueError):
        duration = 0.0
    tolerance = max(0.5, 2 / max(1, render.fps))
    if not math.isfinite(duration) or duration <= 0 or abs(duration - expected_duration) > tolerance:
        failures.append(
            f"render duration is {duration:.3f}s, expected {expected_duration:.3f}s "
            f"within {tolerance:.3f}s"
        )
    if failures:
        return failures

    # ffprobe proves only container metadata. Decode every staged stream before
    # promotion so a truncated/corrupt rerender can never replace the last
    # completed cut merely because its headers are readable.
    decode = run_capture_logged(
        name="Rendered video full decode",
        command=[
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        logger=logger,
        timeout=max(60, min(900, int(expected_duration * 2 + 30))),
        stdout_log_limit=500,
        stderr_log_limit=2000,
    )
    if decode.returncode != 0:
        return [
            "full render decode failed with ffmpeg exit "
            f"{decode.returncode}: {(decode.stderr or '')[-1000:]}"
        ]
    return []


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _promote_render_candidate(
    staged_video_path: Path,
    staged_report_path: Path,
    output_dir: Path,
    video_sha256: str,
) -> Path:
    """Publish a versioned cut without touching the prior task video pointer."""
    final_video_path = output_dir / f"video-{video_sha256[:16]}.mp4"
    os.replace(staged_video_path, final_video_path)
    # The report remains a convenience alias. If this second promotion fails,
    # the previous task.video_path is still intact and the new version is an
    # auditable orphan that a retry may safely replace.
    os.replace(staged_report_path, output_dir / "av_sync_report.json")
    return final_video_path


def _promote_quality_gated_candidate(
    staged_video_path: Path,
    staged_report_path: Path,
    output_dir: Path,
    video_sha256: str,
    quality_report: dict,
) -> Path:
    """Promote only a candidate whose final persisted quality report passed."""
    if quality_report.get("passed") is not True:
        details = [str(value) for value in quality_report.get("warnings") or []]
        state = _record_quality_retry(
            output_dir,
            quality_report,
            video_sha256,
        )
        raise QualityGateRetry(
            "Rendered candidate did not pass the final A/V quality gate and was "
            f"not promoted; automatic compose retry {state['attempt_count']} requested"
            + (f": {'; '.join(details)}" if details else "")
        )
    retry_state_path = output_dir / QUALITY_RETRY_STATE_FILENAME
    if retry_state_path.is_file():
        try:
            state = json.loads(retry_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        state.update(
            {
                "status": "passed",
                "passed_at": datetime.now(timezone.utc).isoformat(),
                "promoted_video_sha256": video_sha256,
            }
        )
        retry_state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return _promote_render_candidate(
        staged_video_path,
        staged_report_path,
        output_dir,
        video_sha256,
    )


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


def _render_timeout(duration: float, render: RenderSpec) -> int:
    frames = max(1, round(duration * render.fps))
    pixel_ratio = render.width * render.height / (1920 * 1080)
    return max(RENDER_TIMEOUT_FLOOR, math.ceil(300 + frames * RENDER_SECONDS_PER_FRAME * pixel_ratio))


def _build_render_command(
    project_dir: Path,
    video_path: Path,
    frame: FrameSpec = LANDSCAPE,
    render: RenderSpec | None = None,
) -> list[str]:
    """Build the render command for the current HyperFrames CLI (v0.6.x).

    The render entry is the project directory (which must contain
    ``index.html``) and dimensions come from ``--resolution``. Prefer the
    locally-installed, version-pinned binary so a render never triggers an
    on-demand ``npx`` install; fall back to a *pinned* npx invocation only if the
    local install is missing.
    """
    render = render or resolve_render_spec(frame)
    local_bin = config.HYPERFRAME_DIR / "node_modules" / ".bin" / "hyperframes"
    if local_bin.exists():
        base = [str(local_bin)]
    else:
        base = ["npx", "--yes", f"hyperframes@{config.HYPERFRAMES_VERSION}"]
    command = base + [
        "render", str(project_dir),
        "--output", str(video_path),
        "--resolution", render.resolution,
        "--fps", str(render.fps),
        "--quality", render.quality,
        "--sdr",
        "-w", render.workers,
        "--protocol-timeout", str(render.protocol_timeout_ms),
    ]
    variables_path = project_dir / intros.INTRO_VARIABLES_FILENAME
    if variables_path.is_file():
        command.extend(["--variables-file", str(variables_path), "--strict-variables"])
    return command


def _mount_list(board: dict) -> list[dict]:
    """Every mount needed to cover the timeline with no gaps.

    The branded intro/outro are ordinary timed mounts around the narrated
    scenes. ``scene_kind`` survives here so the spine can keep the opener above
    any preloaded content without making the model responsible for z-order.
    """
    mounts: list[dict] = []
    for scene in board["scenes"]:
        mount = {
            "id": scene["id"],
            "start": scene["start"],
            "duration": scene["duration"],
        }
        if scene.get("scene_kind"):
            mount["scene_kind"] = scene["scene_kind"]
        mounts.append(mount)
    return mounts


def _quality_warnings(
    alignment: dict,
    visual_grounding: dict,
    multimodal: dict | None = None,
    news_image_inventory: dict | None = None,
) -> list[str]:
    """Summarize A/V quality-gate failures for the final delivery report."""
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
    if (
        news_image_inventory
        and news_image_inventory.get("enabled")
        and not news_image_inventory.get("complete")
    ):
        warnings.append(
            str(news_image_inventory.get("warning") or "").strip()
            or "Licensed news-image inventory was incomplete; grounded template visuals were used."
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
    """Finalize the strict quality gate before candidate promotion."""
    # The reviewer owns score calibration and provider-unavailable fallback.
    # Exact scene bindings cannot overrule a later rendered-frame rejection:
    # a correctly bound scene can still omit a narrated claim from its pixels.
    multimodal_passed = multimodal.get("passed") is True if multimodal_enabled else True
    quality_passed = (
        alignment.get("passed") is True
        and visual_grounding.get("passed") is True
        and multimodal_passed
        and report.get("visual_coverage", {}).get("passed", True) is True
    )
    warnings = _quality_warnings(
        alignment,
        visual_grounding,
        multimodal,
        report.get("news_images"),
    )
    report.update(
        {
            "passed": quality_passed,
            # A rejected candidate is active rework, not a terminal task
            # failure. Keep the persisted report vocabulary aligned with the
            # worker state: the candidate remains deferred while another
            # compose cycle is queued automatically.
            "quality_status": "passed" if quality_passed else "retrying",
            "delivery_status": "completed" if quality_passed else "deferred",
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


def _assert_locked_visual_assets(output_dir: Path, plans: list[dict]) -> None:
    """Fail closed if a media-locked scene file drops any requested source."""
    failures: list[str] = []
    for plan in plans:
        scene_id = str(plan.get("id") or "")
        required = [
            str(plan.get("footage_src") or ""),
            str(plan.get("collage_src") or ""),
            str(plan.get("news_webpage_src") or ""),
            str(plan.get("news_image_src") or ""),
            str(plan.get("intro_logo_src") or ""),
            str(plan.get("outro_logo_src") or ""),
            str(plan.get("outro_hold_src") or ""),
            *[
                str(value or "")
                for value in plan.get("news_image_srcs") or []
            ],
            *[
                str(item.get("src") or "")
                for item in plan.get("footage_sequence") or []
                if isinstance(item, dict)
            ],
        ]
        required = list(dict.fromkeys(value for value in required if value))
        if not required:
            continue
        path = output_dir / "compositions" / f"{scene_id}.html"
        try:
            html = path.read_text(encoding="utf-8")
        except OSError as exc:
            failures.append(f"{scene_id}: unreadable composition ({exc})")
            continue
        if plan.get("outro_credits"):
            failures.extend(f"{scene_id}: {problem}" for problem in
                            director.validate_scene_html(html, scene_id, plan=plan))
        if plan.get("media_shots"):
            try:
                media_shots.assert_rendered_shots(plan, html)
            except (ValueError, TypeError, KeyError) as exc:
                failures.append(f"{scene_id}: {exc}")
            # The shot renderer canonicalizes project-relative asset paths.
            required = [value if value.startswith("../") else "../" + value for value in required]
        missing = [source for source in required if f'src="{source}"' not in html]
        if missing:
            failures.append(f"{scene_id}: missing {', '.join(missing)}")
    if failures:
        raise RuntimeError(
            "Locked visual assets did not reach generated HyperFrames scenes: "
            + "; ".join(failures)
        )


def _available_collage_storyboard(board: dict, plans: list[dict], count: int | None) -> dict:
    """Expose same-story footage to the collage editor without reserving scenes."""
    by_id = {str(plan.get("id")): plan for plan in plans}
    scenes = [{**scene, "available_public_footage": by_id.get(str(scene.get("id")), {}).get("footage_sequence", [])}
              for scene in board.get("scenes", [])
              if scene.get("program_segment_kind") not in {"opening", "closing"}]
    if count is not None and count > len(scenes):
        raise RuntimeError(
            f"Collage B-roll placement unavailable: {count} clips requested but only "
            f"{len(scenes)} narration scenes remain after program bookends"
        )
    return {**board, "scenes": scenes}


def _review_retry_fingerprint(directory: Path, request: dict, render: RenderSpec | None = None) -> str:
    """Bind review-only recovery to the unchanged render inputs and settings."""
    paths = {Path(request[key]) for key in ("script_path", "audio_path", "background_music_path") if request.get(key)}
    for name in ("index.html", "storyboard.json", "visual_plan.json", "footage/manifest.json",
                 "audio/paced_narration.wav", "audio/program_mix.wav"):
        paths.add(directory / name)
    for name in ("compositions", "assets", "news_images"):
        paths.update(path for path in (directory / name).rglob("*") if path.is_file())
    paths.update((directory / "footage").glob("*-render.mp4"))
    paths.update(Path(__file__).parent / name for name in (
        "scene_kit.py", "assembler.py", "storyboard.py", "intros.py", "outros.py",
        "composer.py", "news_images.py", "collage_broll.py", "visual_plan.py",
        "media_shots.py", "video_format.py",
    ))
    paths.add(config.PROMPTS_DIR / "media_shots.txt")
    evidence = {
        "request": request,
        "render": (render or resolve_render_spec(
            resolve_frame_spec(request.get("video_orientation"))
        )).to_dict(),
        "files": {str(path.resolve()): _sha256_path(path) if path.is_file() else None for path in sorted(paths)},
    }
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, default=str).encode()).hexdigest()


async def _resume_unavailable_visual_review(directory: Path, request: dict, frame: FrameSpec, emit, render: RenderSpec | None = None) -> str | None:
    try:
        checkpoint = json.loads((directory / "render_review_checkpoint.json").read_text())
        report_path = directory / "av_sync_report.next.json"
        report = json.loads(report_path.read_text())
        prior_review = report.get("multimodal") or {}
        interrupted = (
            report.get("quality_status") == "pending"
            and report.get("delivery_status") == "pending"
            and prior_review.get("status") == "pending"
        )
        unavailable = bool(prior_review.get("errors"))
        candidate = directory / "video.next.mp4"
        if (
            not config.AV_SYNC_GEMINI_REVIEW_ENABLED
            or not (unavailable or interrupted)
            or checkpoint.get("input_sha256") != _review_retry_fingerprint(directory, request, render)
            or not candidate.is_file()
        ):
            return None
        digest = _sha256_path(candidate)
        report_digest = report.get("rendered_video_sha256")
        if (
            digest != checkpoint.get("video_sha256")
            or (report_digest is not None and digest != report_digest)
            or (not interrupted and report_digest is None)
        ):
            return None
        board = json.loads((directory / "storyboard.json").read_text())
        if _rendered_video_failures(candidate, frame=frame, expected_duration=float(board["total_duration"]), render=render):
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    reason = "an interrupted review" if interrupted else "provider unavailability"
    emit(f"Visual review: resuming the unchanged rendered candidate after {reason}")
    # A resumed unavailable review has already exhausted the primary path.
    # Give the configured fallback one chance before repeating that cycle.
    review = await multimodal_review.review_video(
        candidate, board, directory, log=emit, prefer_fallback=unavailable
    )
    _finalize_quality_report(report, board["alignment"], report["visual_grounding"], review, multimodal_enabled=True)
    report["rendered_video_sha256"] = digest
    report["rendered_video_filename"] = f"video-{digest[:16]}.mp4"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return str(_promote_quality_gated_candidate(candidate, report_path, directory, digest, report))


def _previous_rejection_used_fallback(output_dir: Path) -> bool:
    """Let the same available reviewer assess repairs it explicitly requested."""
    fallback = str(getattr(config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", "") or "")
    if fallback != "chatgpt":
        return False
    try:
        report = json.loads((output_dir / "av_sync_report.next.json").read_text())
        review = report["multimodal"]
        batches = (review.get("calibration") or {}).get("batches") or review.get("batches")
        return review.get("passed") is False and bool(batches) and all(
            batch.get("contract_valid") is True
            and batch.get("review_provider") == fallback
            for batch in batches
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _director_scene_plans(plans: list[dict], *, quality_retry: bool) -> list[dict]:
    # Preserve unrelated bookends on retry, but route rejected bookends to the
    # overlay editor: staging their locked media cannot apply layout feedback.
    # Source rolls retain their measured geometry and complete attribution.
    return [
        plan for plan in plans
        if not plan.get("outro_credits") and ((
            plan.get("archetype") in {"intro", "outro"}
            and (not quality_retry or bool(plan.get("visual_review_feedback")))
        ) or (
            not plan.get("footage_src")
            and not plan.get("news_image")
            and not plan.get("news_webpage")
        ))
    ]


def _retain_bookend_review_feedback(staged: dict, previous: dict) -> dict:
    """Keep the selected preset while carrying pixel feedback to its editor."""
    for key in ("review_repair_source_sha256", "visual_review_feedback"):
        if key in previous:
            staged[key] = previous[key]
    return staged


def _require_public_footage_placement(plans: list[dict], manifest: dict | None, task_dir: Path) -> None:
    """Fail before expensive downstream generation and name every missing clip."""
    requested = int((manifest or {}).get("requested_clip_count") or 0)
    if not requested:
        return
    clips = (manifest or {}).get("clips") or []
    placements: dict[str, list[str]] = {}
    for plan in plans:
        if plan.get("archetype") != "footage":
            continue
        sequence = plan.get("footage_sequence") or (
            [{"src": plan.get("footage_src")}] if not plan.get("collage_broll") else []
        )
        for item in sequence:
            placements.setdefault(str(item.get("src") or ""), []).append(plan["id"])
    results = []
    for clip in clips:
        source = Path(clip.get("local_path") or clip.get("path") or "")
        source = source if source.is_absolute() else task_dir / source
        try:
            relative = source.resolve().relative_to(task_dir.resolve()).as_posix()
        except ValueError:
            relative = ""
        scenes = placements.get(relative, [])
        results.append({"id": clip.get("id"), "query": clip.get("query"),
                        "src": relative, "scene_ids": scenes, "placed": len(scenes) == 1})
    delivered = sum(item["placed"] for item in results)
    report = {"requested": requested, "acquired": len(clips), "placed": delivered,
              "passed": delivered == requested and len(clips) == requested, "clips": results}
    (task_dir / "footage-placement.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        missing = "; ".join(f"{item['id']} ({item['query']})" for item in results if not item["placed"])
        raise RuntimeError(
            f"Public-footage delivery blocked: {len(clips)}/{requested} clips were acquired and "
            f"{delivered}/{requested} reached exact narration scenes. "
            f"Unplaced or duplicated clips: {missing or 'missing acquired clips'}. "
            "An enabled Public Footage request may not silently fall back to template visuals."
        )


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
    outro_style: str = outros.DEFAULT_OUTRO_STYLE,
    edition_date: str | None = None,
    collage_broll_enabled: bool = False,
    collage_broll_count: int | None = None,
    news_images_enabled: bool = True,
    news_image_count: int | None = None,
    is_monologue: bool = False,
    ai_endpoint: str | None = None,
    ai_model: str | None = None,
    provider_id: int | None = None,
    tts_model: str | None = None,
    background_music_path: str | None = None,
    background_music_bed_db: float = -25.0,
    background_music_duck_db: float = -11.0,
    program_music_pacing_enabled: bool = False,
    program_music_intro_seconds: float = 2.0,
    program_music_opening_gap_seconds: float = 3.0,
    program_music_story_gap_seconds: float = 1.5,
    log: LogCallback | None = None,
    footage_enabled: bool = False,
) -> str:
    review_request = {key: value for key, value in locals().items() if key != "log"}
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    if footage_enabled and not footage.acquisition_is_complete(
        output_dir_path, footage.read_manifest(output_dir_path),
        Path(script_path).read_text(encoding="utf-8"),
    ):
        raise RuntimeError(
            "Public-footage delivery blocked: enabled footage acquisition has no "
            "complete, current manifest. Acquire footage before composing."
        )
    prefer_previous_reviewer = _previous_rejection_used_fallback(output_dir_path)
    quality_retry_source, _ = _pending_visual_repairs(output_dir_path, [])
    frame = resolve_frame_spec(video_orientation)
    render = resolve_render_spec(frame)

    # Mirror to the task log (pipeline.log + LogPanel) when available, else the
    # module logger (start.sh log). Prefer the callback to avoid double-logging.
    emit = lambda message: log(message) if log else logger.info(message)

    # --- 1. Storyboard -----------------------------------------------------
    effective_tts_model = tts_model or config.TTS_DEFAULT_MODEL
    tts_model_meta = config.TTS_MODELS.get(effective_tts_model, {})
    source_contract_verified = bool(tts_model_meta.get("acoustic_integrity"))
    tts_provider = str(tts_model_meta.get("provider") or effective_tts_model)
    manifest_failures = _narration_manifest_failures(script_path, audio_path, tts_model)
    if manifest_failures:
        detail = "; ".join(manifest_failures)
        emit(f"Narration integrity failed; video render blocked: {detail}")
        raise RuntimeError(
            "Narration audio does not retain a 100% verified script contract; "
            f"refusing to render. {detail}"
        )
    resumed = await _resume_unavailable_visual_review(output_dir_path, review_request, frame, emit, render)
    if resumed is not None:
        return resumed
    if source_contract_verified:
        emit(
            f"Narration integrity: {tts_provider} manifest verifies 100% of source parts"
        )
    emit("Narration pacing: natural 1.0x speech locked; scene timing follows measured audio")
    audio_duration = sb.get_audio_duration(audio_path)
    word_transcript, transcription = await av_sync.ensure_word_transcript(
        audio_path,
        output_dir_path,
        log=emit,
    )
    boundaries = _detect_silence_boundaries(audio_path, log)
    emit(f"Audio duration: {audio_duration:.1f}s; {len(boundaries)} silence boundaries detected")

    if program_music_pacing_enabled:
        if not background_music_path:
            raise RuntimeError("Program pacing requires an enabled background-music track")
        physical_segments: list[dict] = []
        for raw in Path(script_path).read_text(encoding="utf-8").splitlines():
            text = raw.strip()
            if not text:
                continue
            text = re.sub(r"^Speaker\s+\d+\s*:\s*", "", text, flags=re.IGNORECASE).strip()
            if text:
                physical_segments.append(
                    {
                        "speaker": 1,
                        "text": text,
                        "word_count": max(1, len(re.findall(r"\b[\w'-]+\b", text))),
                    }
                )
        if len(physical_segments) < 3:
            raise RuntimeError(
                "Daily program pacing requires separate physical lines for the opening, "
                "at least one news segment, and the closing"
            )
        manifest_program_segments = (
            _manifest_program_segments(audio_path, physical_segments)
            if source_contract_verified
            else None
        )
        if manifest_program_segments:
            aligned_segments = manifest_program_segments
        elif word_transcript:
            aligned_segments, _ = sb.align_lines_to_transcript(
                physical_segments,
                word_transcript,
                audio_duration,
                minimum_word_coverage=config.AV_SYNC_MIN_WORD_COVERAGE_PERCENT / 100,
                maximum_boundary_uncertainty=(
                    config.AV_SYNC_MAX_BOUNDARY_UNCERTAINTY_MS / 1000
                ),
            )
        else:
            aligned_segments = sb.assign_line_timing(
                physical_segments,
                audio_duration,
                boundaries,
            )
        paced_audio, pacing_report = await music.create_paced_narration(
            audio_path,
            output_dir_path,
            aligned_segments,
            intro_seconds=program_music_intro_seconds,
            opening_gap_seconds=program_music_opening_gap_seconds,
            story_gap_seconds=program_music_story_gap_seconds,
            log=emit,
        )
        word_transcript = music.shift_word_transcript_for_pacing(
            word_transcript,
            pacing_report,
        )
        audio_path = str(paced_audio)
        audio_duration = sb.get_audio_duration(audio_path)
        boundaries = _detect_silence_boundaries(audio_path, log)
        emit(
            f"Program audio duration: {audio_duration:.1f}s after deterministic music gaps; "
            f"{len(boundaries)} silence boundaries detected"
        )

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
    if program_music_pacing_enabled and source_contract_verified:
        _apply_manifest_program_alignment(
            board["alignment"],
            pacing_report,
            manifest_program_segments,
            provider_label=tts_provider,
        )
        if manifest_program_segments:
            board = sb.build_program_storyboard(
                pacing_report=pacing_report,
                title=title,
                alignment=board["alignment"],
                summary=summary,
                log=emit,
            )
    sb.write_storyboard(output_dir_path, board)
    completeness_failures = _narration_completeness_failures(
        board["alignment"],
        source_contract_verified=source_contract_verified,
    )
    if completeness_failures:
        detail = "; ".join(completeness_failures)
        emit(f"Narration integrity failed; video render blocked: {detail}")
        raise RuntimeError(
            "Narration audio does not cover the full script; refusing to render "
            f"a truncated video. {detail}"
        )
    if source_contract_verified and not board["alignment"].get("passed"):
        emit(
            "A/V sync warning: paced full-file ASR is incomplete, but the "
            f"hash-bound {tts_provider} manifest proves 100% per-part source coverage"
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
    plans = await _load_or_plan_scene_visuals(
        output_dir_path,
        board,
        ai_endpoint=ai_endpoint,
        ai_model=ai_model,
        provider_id=provider_id,
        log=emit,
    )

    manifest = await footage.normalize_manifest_clips(
        output_dir_path, footage.read_manifest(output_dir_path), log=emit,
    )
    attached = visual_plan.attach_footage(plans, board, manifest, output_dir_path)
    if attached:
        emit(f"Footage: {attached} manifest clip(s) placed as full-bleed scenes")
    _require_public_footage_placement(plans, manifest, output_dir_path)

    # Morning Desk has explicit spoken opening/closing scenes. Those are owned
    # by the selected system bookend preset, never by generated collage B-roll.
    force_collage_opening = (
        opening_style == "paper_collage" and not board.get("program_timeline")
    )
    if collage_broll_enabled or force_collage_opening:
        requested_collages = collage_broll_count if collage_broll_enabled else 1
        collage_board = _available_collage_storyboard(board, plans, requested_collages)
        collage_manifest = await collage_broll.generate_collage_broll(
            collage_board,
            output_dir_path,
            count=requested_collages,
            force_opening=force_collage_opening,
            frame=frame,
            provider_id=provider_id,
            ai_endpoint=ai_endpoint,
            ai_model=ai_model,
            reuse_failed=bool(quality_retry_source),
            log=emit,
        )
        collage_attached = collage_broll.attach_collage(
            plans, collage_manifest, output_dir_path
        )
        planned_collages = int(collage_manifest.get("planned_count") or 0)
        emit(
            f"Collage B-roll: {collage_attached}/{planned_collages} AI-selected "
            f"clip(s) placed as {frame.aspect_ratio} full-bleed scenes"
        )

    final_public_footage = sum(
        len(plan.get("footage_sequence") or []) or (0 if plan.get("collage_broll") else 1)
        for plan in plans
        if plan.get("archetype") == "footage"
    )
    final_collages = sum(1 for plan in plans if plan.get("collage_broll"))
    _require_public_footage_placement(plans, manifest, output_dir_path)
    if collage_broll_enabled or force_collage_opening:
        if final_collages != planned_collages and (requested_collages is not None or force_collage_opening):
            raise RuntimeError(
                "Collage B-roll placement incomplete: "
                f"{final_collages}/{planned_collages} planned clips reached final scenes"
            )
        if final_collages != planned_collages:
            emit(f"Collage: {final_collages}/{planned_collages} available; AI shot editor will replan from actual assets")

    news_image_inventory = {"attached": 0, "placement_modes": {"inline": 0, "fullscreen": 0}}
    news_image_quality = {
        "enabled": bool(news_images_enabled),
        "complete": not news_images_enabled,
        "requested": 0,
        "attached": 0,
        "manifest_status": "disabled" if not news_images_enabled else "not_needed",
        "missing_scene_ids": [],
        "warning": "",
    }
    eligible_image_scene_ids = _eligible_news_image_scene_ids(plans, board)
    if not eligible_image_scene_ids:
        news_image_quality["complete"] = True
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
        required_count = int(image_manifest.get("planned_image_count") or 0)
        attached_images = int(news_image_inventory["attached"])
        image_modes = news_image_inventory["placement_modes"]
        manifest_images = image_manifest.get("images") or []
        acquired_images = len(manifest_images) if isinstance(manifest_images, list) else 0
        manifest_status = str(image_manifest.get("status") or "unknown")
        missing_scene_ids = [
            str(scene_id) for scene_id in image_manifest.get("missing_scene_ids") or []
        ]
        shortfall = ""
        if attached_images != required_count:
            shortfall = _news_image_placement_error(
                image_manifest,
                attached_images=attached_images,
                required_count=required_count,
            )
            if manifest_status == "ready" or attached_images != acquired_images:
                raise RuntimeError(shortfall)
            emit(
                f"{shortfall}; continuing with grounded template visuals for "
                "the missing scenes"
            )
        if attached_images >= 2 and (
            not image_modes.get("inline") or not image_modes.get("fullscreen")
        ):
            mode_warning = (
                "News-image placement retained only one display mode after preserving "
                "structured scene evidence"
            )
            if manifest_status == "ready":
                raise RuntimeError(
                    f"{mode_warning}; both inline and fullscreen modes are required "
                    "when at least two eligible scenes exist"
                )
            emit(f"{mode_warning}; continuing with the verified partial inventory")
        news_image_quality = {
            "enabled": True,
            "complete": attached_images == required_count,
            "requested": required_count,
            "attached": attached_images,
            "manifest_status": manifest_status,
            "missing_scene_ids": missing_scene_ids,
            "warning": shortfall,
        }

    webpage_manifest = await news_webpages.acquire_news_webpages(
        board,
        plans,
        output_dir_path,
        log=emit,
    )
    webpage_attached = news_webpages.attach_news_webpages(
        plans,
        webpage_manifest,
        output_dir_path,
    )
    webpage_requested = int(webpage_manifest.get("requested_count") or 0)
    webpage_captured = len(webpage_manifest.get("pages") or [])
    if webpage_captured and webpage_attached != webpage_captured:
        raise RuntimeError(
            "News-webpage placement incomplete: "
            f"{webpage_attached}/{webpage_captured} verified English captures reached final scenes"
        )
    if webpage_requested and not webpage_attached:
        errors = "; ".join(
            str(item.get("message") or "capture failed")
            for item in webpage_manifest.get("errors") or []
            if isinstance(item, dict)
        )
        raise RuntimeError(
            "News-webpage overlay delivery blocked: English article scenes were eligible "
            f"but 0/{webpage_requested} captures were usable"
            + (f". {errors[:700]}" if errors else "")
        )
    emit(
        "Final B-roll inventory: "
        f"{final_public_footage} public footage clip(s), "
        f"{final_collages} paper-collage clip(s), "
        f"{news_image_inventory['attached']} news image(s) "
        f"(inline={news_image_inventory['placement_modes']['inline']}, "
        f"fullscreen={news_image_inventory['placement_modes']['fullscreen']}), "
        f"{webpage_attached} English news webpage overlay(s)"
    )

    _enforce_program_opening_copy(plans, board)
    shot_report = await media_shots.plan_media_shots(
        plans, board, output_dir_path, provider_id=provider_id,
        ai_endpoint=ai_endpoint, ai_model=ai_model, log=emit,
    )
    scene_plans = list(plans)
    visual_grounding = visual_plan.visual_grounding_report(scene_plans, board)
    quality_report = {
        "render_profile": render.to_dict(),
        "passed": bool(board["alignment"].get("passed")) and visual_grounding["passed"],
        "quality_status": "pending",
        "delivery_status": "pending",
        "warnings": _quality_warnings(
            board["alignment"],
            visual_grounding,
            news_image_inventory=news_image_quality,
        ),
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
        "news_images": news_image_quality,
        "visual_coverage": shot_report,
    }
    quality_report_path = output_dir_path / "av_sync_report.next.json"
    quality_report_path.unlink(missing_ok=True)
    quality_report_path.write_text(
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

    # Cache the narration timing before the source roll extends the closing.
    _write_visual_plan_checkpoint(output_dir_path, board, scene_plans)
    outro_plan = outros.stage_outro(
        output_dir_path, board, outro_style, media_plans=scene_plans,
        video_orientation=frame.orientation,
    )
    quality_report["outro_credits"] = {
        "entry_count": len(outro_plan["outro_credits"]),
        "manifest": str(output_dir_path / "outro_credits.json"),
        "start": board["outro_start"],
        "duration": board["outro_duration"],
        "reading_tail_seconds": board["credits_tail_duration"],
    }
    program_audio_path = Path(audio_path)
    if board["credits_tail_duration"] > 0:
        # Keep narration/caption timestamps intact. Only the credit tail adds
        # silence to narration, so the normal music mixer can score the tail.
        program_duration = float(board["audio_duration"]) + board["credits_tail_duration"]
        program_audio_path = output_dir_path / "audio" / "outro_narration.wav"
        program_audio_path.parent.mkdir(parents=True, exist_ok=True)
        returncode, output = await stream_subprocess(
            name="Outro audio tail", command=[
                "ffmpeg", "-v", "error", "-y", "-i", str(audio_path),
                "-af", f"apad=whole_dur={program_duration:.2f}",
                "-c:a", "pcm_s16le", str(program_audio_path),
            ], logger=logger, log=log, timeout=300,
        )
        if returncode:
            raise RuntimeError(f"Could not extend outro audio: {output[-500:]}")
        board["program_audio_duration"] = program_duration
    if background_music_path:
        source_music = Path(background_music_path)
        if not source_music.is_file():
            raise RuntimeError(f"Configured background music is missing: {source_music}")
        program_audio_path = await music.mix_narration_and_music(
            str(program_audio_path),
            source_music,
            output_dir_path,
            bed_db=background_music_bed_db,
            duck_db=background_music_duck_db,
            log=emit,
            original_narration_path=audio_path if board["credits_tail_duration"] > 0 else None,
        )
        quality_report["background_music"] = {
            "enabled": True,
            "passed": True,
            "source": str(source_music.resolve()),
            "program_mix": str(program_audio_path.resolve()),
            "report": str((output_dir_path / "audio" / "music_mix_report.json").resolve()),
        }

    # Cache only narration-driven direction. Branded bookends are configured
    # render stages, not scenes the visual planner should regenerate on retry.
    plans = scene_plans
    intro_plan = intros.stage_intro(
        output_dir_path,
        board,
        outro_style,
        edition_date=edition_date,
    )
    intro_index = next(
        index for index, plan in enumerate(scene_plans) if plan.get("id") == intro_plan["id"]
    )
    scene_plans[intro_index] = _retain_bookend_review_feedback(
        intro_plan, scene_plans[intro_index]
    )
    outro_index = next(
        index for index, plan in enumerate(scene_plans) if plan.get("id") == outro_plan["id"]
    )
    scene_plans[outro_index] = _retain_bookend_review_feedback(
        outro_plan, scene_plans[outro_index]
    )
    plans = scene_plans
    sb.write_storyboard(output_dir_path, board)
    emit(
        f"Intro: {intro_plan['intro_label']} bound to the "
        f"{intro_plan['duration']:.2f}s spoken opening; "
        f"HyperFrames variables inject {intro_plan['edition_weekday']}, "
        f"{intro_plan['edition_date']}"
    )
    emit(
        f"Outro: {outro_plan['outro_label']} bound to the "
        f"{outro_plan['duration']:.2f}s closing with an editable "
        f"HyperFrames overlay; {len(outro_plan['outro_credits'])} source credits, "
        f"{board['credits_tail_duration']:.2f}s reading tail"
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

    # Licensed editorial media stays on the deterministic renderer. The branded
    # intro/outro without source rolls are exceptions: their videos remain locked
    # while the video-editing agent authors only the editable HyperFrames overlay.
    director_plans = _director_scene_plans(
        scene_plans,
        quality_retry=bool(quality_retry_source) or any(
            plan.get("visual_review_feedback") for plan in scene_plans
        ),
    )
    if quality_retry_source:
        emit("Quality retry: preserving bookend media and directing overlays with frame feedback")
    if config.DIRECTOR_ENABLED and director_plans and model:
        if config.DIRECTOR_MAX_SCENES:
            budget = director_plans[: config.DIRECTOR_MAX_SCENES]
            configured_bookends = [
                plan
                for plan in director_plans
                if plan.get("archetype") in {"intro", "outro"}
            ]
            for configured_bookend in configured_bookends:
                if configured_bookend in budget or not budget:
                    continue
                replace_at = next(
                    (
                        index
                        for index in range(len(budget) - 1, -1, -1)
                        if budget[index].get("archetype") not in {"intro", "outro"}
                    ),
                    len(budget) - 1,
                )
                budget[replace_at] = configured_bookend
        else:
            budget = director_plans
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
    _assert_locked_visual_assets(output_dir_path, scene_plans)
    emit("Locked visual assets verified in generated HyperFrames scene sources")
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
            bookend_ids = {
                str(plan.get("id"))
                for plan in scene_plans
                if plan.get("archetype") in {"intro", "outro"}
            }
            blamed_bookends = [scene_id for scene_id in blamed if scene_id in bookend_ids]
            if blamed_bookends:
                director.revert_scenes(output_dir_path, blamed_bookends, kit_plans)
                emit(
                    "Layout: reverted the branded bookend overlay to its verified editable draft "
                    "instead of letting a generic caption-safe repair rewrite it"
                )
                blamed = [scene_id for scene_id in blamed if scene_id not in bookend_ids]
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

    # A repair or future agent implementation must never drop the selected
    # Gemini plate or exact ByteFront logo after the initial authoring gate.
    _assert_locked_visual_assets(output_dir_path, scene_plans)

    # --- 5. Render ---------------------------------------------------------
    staged_video_path = output_dir_path / "video.next.mp4"
    staged_video_path.unlink(missing_ok=True)
    total_frames = max(1, round(float(board["total_duration"]) * render.fps))
    render_timeout = _render_timeout(float(board["total_duration"]), render)
    emit(
        f"Rendering ~{total_frames} frames "
        f"({render.width}x{render.height}, {board['total_duration']:.0f}s @ {render.fps}fps, {render.quality}, "
        f"{render.workers} worker(s)); render timeout {render_timeout}s, "
        f"stall timeout {RENDER_STALL_TIMEOUT}s"
    )

    render_command = _build_render_command(output_dir_path, staged_video_path, frame, render)
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
        staged_video_path.unlink(missing_ok=True)
        raise RuntimeError(f"Video render failed (exit {returncode}): {output[-500:]}")

    if not staged_video_path.exists():
        raise RuntimeError("Video render produced no output file")

    probe_failures = _rendered_video_failures(
        staged_video_path,
        frame=frame,
        expected_duration=float(board["total_duration"]),
        render=render,
    )
    if probe_failures:
        staged_video_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Staged video failed validation: " + "; ".join(probe_failures)
        )
    emit(
        f"Video rendered: {staged_video_path} "
        f"({staged_video_path.stat().st_size / 1024 / 1024:.1f} MB)"
    )
    (output_dir_path / "render_review_checkpoint.json").write_text(json.dumps({
        "input_sha256": _review_retry_fingerprint(output_dir_path, review_request, render),
        "video_sha256": _sha256_path(staged_video_path),
    }))
    if config.AV_SYNC_GEMINI_REVIEW_ENABLED:
        gemini_review = await multimodal_review.review_video(
            staged_video_path,
            board,
            output_dir_path,
            log=emit,
            prefer_fallback=prefer_previous_reviewer,
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
    rendered_video_sha256 = _sha256_path(staged_video_path)
    quality_report["rendered_video_sha256"] = rendered_video_sha256
    quality_report["rendered_video_filename"] = (
        f"video-{rendered_video_sha256[:16]}.mp4"
    )
    quality_report_path.write_text(
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
            "A/V quality gate not yet passed: Gemini review requested another "
            "compose cycle; the rendered candidate and av_sync_report.next.json "
            "remain as retry evidence"
        )
    if not quality_report["passed"]:
        emit(
            f"Final A/V quality gate deferred promotion with "
            f"{len(quality_report['warnings'])} finding(s); automatic retry will continue"
        )
    video_path = _promote_quality_gated_candidate(
        staged_video_path,
        quality_report_path,
        output_dir_path,
        rendered_video_sha256,
        quality_report,
    )
    return str(video_path)
