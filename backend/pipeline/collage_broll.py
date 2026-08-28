"""Automated editorial collage B-roll through Agent SDK + signed-in web apps."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import re
import shutil
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from backend import config
from backend.pipeline.opencli import OpenCLIError, first_json, run_opencli
from backend.pipeline.video_format import FrameSpec

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

SOURCE_REPOSITORY = "https://github.com/pyang5166/gbro-collage-broll"
SOURCE_COMMIT = "a1a4ee2e2abf7d44e460026b706d0c72c2cf8a91"
CLIP_FPS = 24
MOTION_SAMPLE_FPS = 4
CACHE_CONTRACT_VERSION = 1
SELECTION_POLICY_VERSION = 3
PLAYBACK_POLICY = "play_once_then_hold_last_frame"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
_SAFE_SCENE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_COLORS = ("#D96B35", "#D2A928", "#315F4C", "#594080", "#188C85", "#B73D3D")
_FALLBACK_ART_DIRECTIONS = (
    "raw archival photomontage with torn newsprint, grease-pencil marks, and uneven deckled edges",
    "playful hand-built scrapbook with painted paper, fabric scraps, tape, and deliberately irregular silhouettes",
    "bold geometric cut-paper modernism with oversized shapes, sharp scale contrast, and off-axis balance",
    "surreal editorial assemblage mixing photographic fragments, hand-drawn marks, and impossible spatial relationships",
    "quiet translucent layering with vellum, tracing-paper diagrams, botanical fragments, and soft overlapping shadows",
    "high-energy photocopied zine collage with ripped textures, rough ink, stamps, and dense edge-to-edge rhythm",
)
_FALLBACK_COMPOSITIONS = (
    "an asymmetric diagonal build with a strong visual interruption near one edge",
    "a loose handmade cluster that leaves breathing room in an unexpected corner",
    "a monumental foreground shape opposed by several tiny satellite details",
    "an all-over composition with controlled density and multiple discovery points",
    "a layered window or portal that reveals depth through overlapping cut surfaces",
    "a split-field composition whose two visual systems collide at the narrative turning point",
)
_FALLBACK_MOTIONS = (
    "pieces tear open and peel back in staggered layers before settling",
    "elements unfold, hinge, and tumble into a slightly imperfect handmade arrangement",
    "large shapes sweep across the frame while smaller details punctuate the rhythm",
    "fragments emerge from different depths, overlap, and lock into a surreal final relationship",
    "translucent layers drift, fan open, and align with restrained tactile motion",
    "photocopied scraps slap, rip-reveal, jitter, and finally freeze into a deliberate zine spread",
)
GEMINI_VIDEO_UPLOAD_CAPABILITY_CODE = (
    "OPENCLI_CAPABILITY_UNAVAILABLE:GEMINI_VIDEO_LOCAL_FILE_UPLOAD"
)
GEMINI_VIDEO_INPUT_HYDRATION_STUCK_CODE = (
    "OPENCLI_CAPABILITY_DEGRADED:GEMINI_VIDEO_UPLOAD_INPUT_HYDRATION_STUCK"
)
CHATGPT_IMAGE_COMPOSER_NOT_READY_SIGNATURE = (
    "chatgpt-image-command-exec:composer-not-ready"
)


class GeminiVideoUploadCapabilityError(OpenCLIError):
    """This browser session cannot put local keyframes into Gemini Video."""

    def __init__(self, message: str, *, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def _seconds(value: Any, fallback: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if math.isfinite(parsed) else fallback


def _with_scene_timing(spec: dict[str, Any], scene: dict[str, Any]) -> dict[str, Any]:
    """Attach the one-play media duration contract for a narration scene."""
    maximum = max(1 / CLIP_FPS, _seconds(config.COLLAGE_GEMINI_MAX_SECONDS, 8.0))
    script_duration = max(0.0, _seconds(scene.get("duration")))
    target_duration = min(script_duration, maximum) if script_duration else maximum
    return {
        **spec,
        "script_duration_seconds": round(script_duration, 3),
        "target_duration_seconds": round(target_duration, 3),
        "gemini_max_duration_seconds": round(maximum, 3),
    }


def _duration_token(duration: float) -> str:
    return f"{duration:.3f}".rstrip("0").rstrip(".")


def _final_clip_path(item_dir: Path, duration: float) -> Path:
    return item_dir / "video" / f"final-{_duration_token(duration)}s-noaudio.mp4"


def _log(log: LogCallback | None, message: str) -> None:
    if log:
        log(message)
    else:
        logger.info(message)


def _write_json(path: Path, payload: Any) -> None:
    _assert_safe_write_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    _assert_safe_write_target(temporary)
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    _assert_safe_write_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    _assert_safe_write_target(temporary)
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _assert_no_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"Collage cache path cannot contain symlinks: {current}")


def _assert_safe_write_target(path: Path) -> None:
    _assert_no_symlink_components(path.parent)
    if path.is_symlink():
        raise RuntimeError(f"Collage cache write target cannot be a symlink: {path}")


def _assert_tree_has_no_symlinks(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"Collage cache stage cannot be a symlink: {path}")
    if not path.is_dir():
        return
    try:
        for child in path.rglob("*"):
            if child.is_symlink():
                raise RuntimeError(
                    f"Collage cache stage cannot contain symlinks: {child}"
                )
    except OSError as exc:
        raise RuntimeError(f"Unable to inspect collage cache stage {path}: {exc}") from exc


def _ensure_owned_directory(path: Path, *, parent: Path | None = None) -> Path:
    _assert_no_symlink_components(path)
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"Collage cache directory is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(path)
    resolved = path.resolve(strict=True)
    if parent is not None:
        parent_resolved = parent.resolve(strict=True)
        if path.absolute().parent != parent.absolute():
            raise RuntimeError(f"Collage cache path escapes lexical parent: {path}")
        if resolved.parent != parent_resolved:
            raise RuntimeError(f"Collage cache path escapes resolved parent: {path}")
    return resolved


def _validate_storyboard_scene_ids(storyboard: dict) -> None:
    scene_ids: list[str] = []
    for scene in storyboard.get("scenes") or []:
        scene_id = str(scene.get("id") or "")
        if not _SAFE_SCENE_ID.fullmatch(scene_id) or scene_id in {".", ".."}:
            raise ValueError(f"Unsafe collage scene id: {scene_id!r}")
        scene_ids.append(scene_id)
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Collage scene ids must be unique")


def _assert_item_stage_paths_safe(item_dir: Path) -> None:
    _assert_no_symlink_components(item_dir)
    for path in (
        item_dir / "stills",
        item_dir / "frames",
        item_dir / "video",
    ):
        _assert_tree_has_no_symlinks(path)
    for path in (
        item_dir / "image-prompt.txt",
        item_dir / "image-prompt.txt.tmp",
        item_dir / "video-prompt.txt",
        item_dir / "video-prompt.txt.tmp",
        item_dir / "stills" / "cache-contract.json",
        item_dir / "stills" / "cache-contract.json.tmp",
        item_dir / "video" / "cache-contract.json",
        item_dir / "video" / "cache-contract.json.tmp",
    ):
        if path.is_symlink():
            raise RuntimeError(f"Collage cache stage file cannot be a symlink: {path}")


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_strict_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_strict_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_strict_json_value(item)
            for key, item in value.items()
        )
    return False


def _file_sha256(path: Path) -> str:
    if path.is_symlink():
        raise RuntimeError(f"Collage cache artifact cannot be a symlink: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_contract(frame: FrameSpec) -> dict[str, Any]:
    return {
        "orientation": frame.orientation,
        "aspect_ratio": frame.aspect_ratio,
        "width": frame.width,
        "height": frame.height,
        "media_width": frame.media_width,
        "media_height": frame.media_height,
    }


def _planning_fingerprint(
    storyboard: dict,
    *,
    count: int,
    force_opening: bool,
    frame: FrameSpec,
) -> str:
    scenes = list(storyboard.get("scenes") or [])
    target_count = min(max(1, count), len(scenes)) if scenes else 0
    return _fingerprint(
        {
            "cache_contract_version": CACHE_CONTRACT_VERSION,
            "selection_policy_version": SELECTION_POLICY_VERSION,
            "contract": "collage_visual_spec",
            "video_thesis": _normalized_text(storyboard.get("thesis")),
            "target_count": target_count,
            "force_opening_scene": bool(force_opening),
            "frame": _frame_contract(frame),
            "candidate_scenes": [
                {
                    "scene_id": str(scene.get("id") or ""),
                    "start": _seconds(scene.get("start")),
                    "duration": _seconds(scene.get("duration")),
                    # Keep the complete narration here even though the planner
                    # prompt is bounded. A semantic change anywhere in a scene
                    # must invalidate the selection/design cache.
                    "narration": _normalized_text(scene.get("text")),
                    "keywords": [
                        _normalized_text(keyword)
                        for keyword in (scene.get("keywords") or [])
                    ],
                }
                for scene in scenes
            ],
        }
    )


def _scene_semantic_fingerprint(storyboard: dict, scene: dict[str, Any]) -> str:
    return _fingerprint(
        {
            "cache_contract_version": CACHE_CONTRACT_VERSION,
            "contract": "collage_scene_semantics",
            "video_thesis": _normalized_text(storyboard.get("thesis")),
            "scene_id": str(scene.get("id") or ""),
            "narration": _normalized_text(scene.get("text")),
            "keywords": [
                _normalized_text(keyword) for keyword in (scene.get("keywords") or [])
            ],
        }
    )


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _prompt_file_matches(path: Path, prompt: str) -> bool:
    if path.is_symlink():
        raise RuntimeError(f"Collage prompt cannot be a symlink: {path}")
    try:
        return path.is_file() and path.read_text(encoding="utf-8") == prompt + "\n"
    except (OSError, UnicodeError):
        return False


def _resolve_contract_artifact(
    item_dir: Path,
    relative_path: Any,
    *,
    owned_root: Path,
    suffixes: set[str],
) -> Path | None:
    if not isinstance(relative_path, str) or not relative_path:
        return None
    relative = Path(relative_path)
    if relative.is_absolute():
        return None
    _assert_tree_has_no_symlinks(owned_root)
    lexical_candidate = item_dir / relative
    _assert_no_symlink_components(lexical_candidate)
    if lexical_candidate.is_symlink():
        raise RuntimeError(
            f"Collage cache artifact cannot be a symlink: {lexical_candidate}"
        )
    try:
        candidate = lexical_candidate.resolve()
        root = owned_root.resolve()
        if not candidate.is_relative_to(root):
            return None
        if not candidate.is_file() or candidate.suffix.lower() not in suffixes:
            return None
    except (OSError, RuntimeError):
        return None
    return candidate


def _read_contract(path: Path) -> dict[str, Any] | None:
    if path.is_symlink():
        raise RuntimeError(f"Collage cache contract cannot be a symlink: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _contract_bindings_match(
    payload: dict[str, Any], expected: dict[str, Any]
) -> bool:
    return all(payload.get(key) == value for key, value in expected.items())


def _load_still_cache(
    item_dir: Path,
    *,
    expected: dict[str, Any],
    still_prompt: str,
) -> Path | None:
    payload = _read_contract(item_dir / "stills" / "cache-contract.json")
    if payload is None or not _contract_bindings_match(payload, expected):
        return None
    if not _prompt_file_matches(item_dir / "image-prompt.txt", still_prompt):
        return None
    artifact = _resolve_contract_artifact(
        item_dir,
        payload.get("artifact_path"),
        owned_root=item_dir / "stills",
        suffixes=_IMAGE_SUFFIXES,
    )
    try:
        if artifact is None or _file_sha256(artifact) != payload.get("artifact_sha256"):
            return None
    except OSError:
        return None
    return artifact


def _load_final_cache(
    item_dir: Path,
    *,
    expected: dict[str, Any],
    still_prompt: str,
    motion_prompt: str,
    final_path: Path,
) -> tuple[Path, Path] | None:
    payload = _read_contract(item_dir / "video" / "cache-contract.json")
    if payload is None or not _contract_bindings_match(payload, expected):
        return None
    if not _prompt_file_matches(item_dir / "image-prompt.txt", still_prompt):
        return None
    if not _prompt_file_matches(item_dir / "video-prompt.txt", motion_prompt):
        return None
    final = _resolve_contract_artifact(
        item_dir,
        payload.get("artifact_path"),
        owned_root=item_dir / "video",
        suffixes={".mp4"},
    )
    hold = _resolve_contract_artifact(
        item_dir,
        payload.get("hold_frame_path"),
        owned_root=item_dir / "frames",
        suffixes=_IMAGE_SUFFIXES,
    )
    try:
        if final is None or final != final_path.resolve():
            return None
        if hold is None:
            return None
        if _file_sha256(final) != payload.get("artifact_sha256"):
            return None
        if _file_sha256(hold) != payload.get("hold_frame_sha256"):
            return None
    except OSError:
        return None
    return final, hold


def _remove_owned_path(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"Refusing to clean symlinked collage cache path: {path}")
    if path.is_dir():
        _assert_tree_has_no_symlinks(path)
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink(missing_ok=True)


def _clear_downstream_stage(item_dir: Path) -> None:
    for path in (
        item_dir / "frames",
        item_dir / "video",
        item_dir / "video-prompt.txt",
        item_dir / "video-prompt.txt.tmp",
    ):
        _remove_owned_path(path)


def _clear_all_stages(item_dir: Path) -> None:
    _clear_downstream_stage(item_dir)
    for path in (
        item_dir / "stills",
        item_dir / "image-prompt.txt",
        item_dir / "image-prompt.txt.tmp",
    ):
        _remove_owned_path(path)


def _stage_has_content(path: Path) -> bool:
    try:
        return path.is_file() or (path.is_dir() and any(path.iterdir()))
    except OSError:
        return True


def _still_stage_has_content(item_dir: Path) -> bool:
    return any(
        _stage_has_content(path)
        for path in (item_dir / "stills", item_dir / "image-prompt.txt")
    )


def _downstream_stage_has_content(item_dir: Path) -> bool:
    return any(
        _stage_has_content(path)
        for path in (
            item_dir / "frames",
            item_dir / "video",
            item_dir / "video-prompt.txt",
        )
    )


def _materialize_still(still: Path, item_dir: Path) -> Path:
    still_dir = item_dir / "stills"
    _ensure_owned_directory(still_dir, parent=item_dir)
    if still.is_symlink():
        raise RuntimeError(f"Collage still cannot be a symlink: {still}")
    source = still.resolve()
    try:
        if source.is_relative_to(still_dir.resolve()):
            return source
    except (OSError, RuntimeError):
        pass
    suffix = source.suffix.lower() if source.suffix.lower() in _IMAGE_SUFFIXES else ".png"
    destination = still_dir / f"generated-still{suffix}"
    _assert_safe_write_target(destination)
    shutil.copyfile(source, destination)
    return destination.resolve()


def _artifact_relative_path(item_dir: Path, artifact: Path) -> str:
    if artifact.is_symlink():
        raise RuntimeError(f"Collage cache artifact cannot be a symlink: {artifact}")
    _assert_no_symlink_components(artifact)
    return artifact.resolve().relative_to(item_dir.resolve()).as_posix()


def _still_contract_bindings(
    *,
    scene_id: str,
    semantic_fingerprint: str,
    spec: dict[str, Any],
    still_prompt: str,
    frame: FrameSpec,
) -> dict[str, Any]:
    inputs = {
        "cache_contract_version": CACHE_CONTRACT_VERSION,
        "stage": "still",
        "scene_id": scene_id,
        "scene_semantic_fingerprint": semantic_fingerprint,
        "spec_fingerprint": _fingerprint(spec),
        "image_prompt_sha256": _prompt_sha256(still_prompt),
        "frame": _frame_contract(frame),
    }
    return {**inputs, "fingerprint": _fingerprint(inputs)}


def _final_contract_bindings(
    *,
    scene_id: str,
    semantic_fingerprint: str,
    spec: dict[str, Any],
    still_prompt: str,
    motion_prompt: str,
    frame: FrameSpec,
    target_duration: float,
) -> dict[str, Any]:
    inputs = {
        "cache_contract_version": CACHE_CONTRACT_VERSION,
        "stage": "final",
        "scene_id": scene_id,
        "scene_semantic_fingerprint": semantic_fingerprint,
        "spec_fingerprint": _fingerprint(spec),
        "image_prompt_sha256": _prompt_sha256(still_prompt),
        "video_prompt_sha256": _prompt_sha256(motion_prompt),
        "frame": _frame_contract(frame),
        "target_duration_seconds": round(target_duration, 3),
        "clip_fps": CLIP_FPS,
        "playback_policy": PLAYBACK_POLICY,
    }
    return {**inputs, "fingerprint": _fingerprint(inputs)}


def _write_still_contract(
    item_dir: Path,
    bindings: dict[str, Any],
    artifact: Path,
) -> None:
    _write_json(
        item_dir / "stills" / "cache-contract.json",
        {
            **bindings,
            "artifact_path": _artifact_relative_path(item_dir, artifact),
            "artifact_sha256": _file_sha256(artifact),
        },
    )


def _write_final_contract(
    item_dir: Path,
    bindings: dict[str, Any],
    artifact: Path,
    hold_frame: Path,
) -> None:
    _write_json(
        item_dir / "video" / "cache-contract.json",
        {
            **bindings,
            "artifact_path": _artifact_relative_path(item_dir, artifact),
            "artifact_sha256": _file_sha256(artifact),
            "hold_frame_path": _artifact_relative_path(item_dir, hold_frame),
            "hold_frame_sha256": _file_sha256(hold_frame),
        },
    )


def _json_array(value: str) -> list[dict[str, Any]]:
    text = value.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    raise RuntimeError("Collage planning agent did not return a JSON array")


def _scene_choices(storyboard: dict, count: int, force_opening: bool) -> list[dict]:
    scenes = list(storyboard.get("scenes") or [])
    if not scenes:
        return []
    count = min(max(1, count), len(scenes))
    selected: list[dict] = [scenes[0]] if force_opening else []
    remaining = [scene for scene in scenes if scene not in selected]
    needed = count - len(selected)
    if needed <= 0:
        return selected
    if needed >= len(remaining):
        return [*selected, *remaining]
    for slot in range(needed):
        index = round(slot * (len(remaining) - 1) / max(1, needed - 1))
        scene = remaining[index]
        if scene not in selected:
            selected.append(scene)
    for scene in remaining:
        if len(selected) >= count:
            break
        if scene not in selected:
            selected.append(scene)
    return sorted(selected, key=lambda scene: float(scene.get("start") or 0))


def _fallback_spec(scene: dict, index: int) -> dict[str, Any]:
    text = str(scene.get("text") or "").strip()
    meaning = re.split(r"(?<=[.!?。！？])\s*", text)[0][:220] or "A hidden process becomes visible"
    objects = ["primary symbolic subject", "supporting found fragment", "physical relationship", "visible consequence"]
    direction_index = index % len(_FALLBACK_ART_DIRECTIONS)
    return _with_scene_timing({
        "scene_id": scene["id"],
        "script_meaning": meaning,
        "emotion": "clarity",
        "visual_metaphor": f"A collage transformation makes visible how {meaning.rstrip('.。')}.",
        "background_hex": _COLORS[index % len(_COLORS)],
        "accent_colors": [],
        "art_direction": _FALLBACK_ART_DIRECTIONS[direction_index],
        "color_direction": "choose the palette from the story rather than a fixed house combination",
        "composition_direction": _FALLBACK_COMPOSITIONS[direction_index],
        "motion_direction": _FALLBACK_MOTIONS[direction_index],
        "elements": [
            {"what": item, "role": "metaphor", "motion": "interpret freely", "placement": "choose for the composition"}
            for item in objects
        ],
        "assembly_order": objects,
        "final_frame": "A resolved, story-specific collage whose visual relationship is clear without relying on a house layout.",
        "planner": "deterministic_fallback",
    }, scene)


def _normalize_spec(raw: dict, scene: dict, index: int) -> dict[str, Any]:
    fallback = _fallback_spec(scene, index)
    spec = {**fallback, **raw, "scene_id": scene["id"], "planner": "claude_agent_sdk"}
    color = str(spec.get("background_hex") or "")
    spec["background_hex"] = color.upper() if _HEX.fullmatch(color) else fallback["background_hex"]
    raw_accents = spec.get("accent_colors")
    accents = [
        str(value)[:80]
        for value in raw_accents
        if str(value).strip()
    ] if isinstance(raw_accents, list) else fallback["accent_colors"]
    spec["accent_colors"] = accents[:6]
    elements = [item for item in (spec.get("elements") or []) if isinstance(item, dict)][:8]
    spec["elements"] = elements or fallback["elements"]
    order = [str(value)[:100] for value in (spec.get("assembly_order") or []) if str(value).strip()]
    spec["assembly_order"] = order[:8] or [item["what"] for item in spec["elements"]]
    for key in (
        "script_meaning",
        "emotion",
        "visual_metaphor",
        "art_direction",
        "color_direction",
        "composition_direction",
        "motion_direction",
        "final_frame",
    ):
        spec[key] = str(spec.get(key) or fallback[key])[:800]
    return _with_scene_timing(spec, scene)


async def plan_specs(
    storyboard: dict,
    *,
    count: int,
    force_opening: bool,
    frame: FrameSpec,
    provider_id: int | None = None,
    ai_endpoint: str | None = None,
    ai_model: str | None = None,
    log: LogCallback | None = None,
) -> list[dict[str, Any]]:
    """Use a dedicated Agent SDK turn to select beats and design metaphors."""
    scenes = list(storyboard.get("scenes") or [])
    if not scenes:
        return []
    target_count = min(max(1, count), len(scenes))
    fallback_choices = _scene_choices(storyboard, target_count, force_opening)
    from backend.pipeline import agent
    from backend.pipeline.digester import _resolve_provider

    skill_path = config.PROJECT_ROOT / ".claude" / "skills" / "gbro-collage-broll" / "SKILL.md"
    system = skill_path.read_text(encoding="utf-8")
    system += (
        "\n\nFor this planning turn, do not use tools and do not generate media. "
        "Return exactly the JSON array described in the Agent visual-spec contract. "
        f"Select exactly {target_count} visually rich beats from the supplied candidate scenes, "
        f"use only their scene ids, distribute the choices across the timeline, and target {frame.aspect_ratio}. "
        + (f"The first item must be {scenes[0]['id']}. " if force_opening else "")
    )
    payload = {
        "video_thesis": storyboard.get("thesis", ""),
        "orientation": frame.orientation,
        "aspect_ratio": frame.aspect_ratio,
        "force_opening_scene": force_opening,
        "candidate_scenes": [
            {
                "scene_id": scene["id"],
                "start": scene.get("start"),
                "duration": scene.get("duration"),
                "narration": str(scene.get("text") or "")[:700],
                "keywords": scene.get("keywords") or [],
            }
            for scene in scenes
        ],
    }
    choices = fallback_choices
    try:
        endpoint, model, api_key = await _resolve_provider(provider_id, ai_endpoint, ai_model)
        _log(log, f"Collage B-roll agent: selecting and designing {target_count} visual metaphor(s)")
        answer = await agent.agent_complete(
            system,
            json.dumps(payload, indent=2, ensure_ascii=False),
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            max_tokens=4096,
            log=log,
            label="Collage B-roll agent",
        )
        raw_items = _json_array(answer)
        allowed = {str(scene["id"]): scene for scene in scenes}
        raw_by_id: dict[str, dict[str, Any]] = {}
        selected_ids: list[str] = []
        for item in raw_items:
            scene_id = str(item.get("scene_id") or "")
            if scene_id in allowed and scene_id not in selected_ids:
                raw_by_id[scene_id] = item
                selected_ids.append(scene_id)
        if force_opening and str(scenes[0]["id"]) not in selected_ids:
            selected_ids.insert(0, str(scenes[0]["id"]))
        for scene in fallback_choices:
            scene_id = str(scene["id"])
            if len(selected_ids) >= target_count:
                break
            if scene_id not in selected_ids:
                selected_ids.append(scene_id)
        selected_ids = selected_ids[:target_count]

        # Collage is the correct treatment for abstract narration that cannot
        # support a strictly grounded licensed still.  If the agent selected a
        # concrete named scene while leaving such an abstract scene outside,
        # swap the most image-rich selected scene out.  This preserves the
        # configured visual inventory without forcing the image scout to
        # invent or weakly match a subject.
        from backend.pipeline import news_images

        grounded_counts = {
            str(scene["id"]): news_images.grounded_visual_subject_count(scene)
            for scene in scenes
        }
        protected = {str(scenes[0]["id"])} if force_opening else set()
        missing_abstract = [
            str(scene["id"])
            for scene in scenes
            if not grounded_counts[str(scene["id"])]
            and str(scene["id"]) not in selected_ids
        ]
        for abstract_id in missing_abstract:
            replaceable = [
                scene_id
                for scene_id in selected_ids
                if scene_id not in protected and grounded_counts.get(scene_id, 0) > 0
            ]
            if not replaceable:
                break
            victim = max(
                replaceable,
                key=lambda scene_id: (
                    grounded_counts.get(scene_id, 0),
                    -selected_ids.index(scene_id),
                ),
            )
            selected_ids[selected_ids.index(victim)] = abstract_id
            _log(
                log,
                "Collage B-roll agent: reserved "
                f"{abstract_id} for metaphor treatment and left {victim} "
                "available for a strictly grounded licensed still",
            )
        choices = sorted(
            (allowed[scene_id] for scene_id in selected_ids),
            key=lambda scene: float(scene.get("start") or 0),
        )
    except Exception as exc:  # noqa: BLE001 - deterministic specs preserve delivery
        _log(log, f"Collage B-roll agent unavailable ({exc}); using narration-derived specs")
        raw_by_id = {}
    return [
        _normalize_spec(raw_by_id.get(scene["id"], {}), scene, index)
        for index, scene in enumerate(choices)
    ]


def image_prompt(spec: dict[str, Any], frame: FrameSpec) -> str:
    elements = "; ".join(str(item.get("what") or "") for item in spec["elements"])
    accents = ", ".join(spec["accent_colors"]) or "no mandatory accent swatches"
    return f"""Use case: documentary B-roll.
Asset type: final still frame for a {frame.aspect_ratio} image-to-video clip.
Create an original editorial collage expressing this visual metaphor: {spec['visual_metaphor']}

Treat collage as an open medium, not a house style. Exercise broad creative control over the visual era, materials, edge treatment, mark-making, density, scale, depth, and balance. The result may be raw or refined, minimal or maximal, analog or graphic, playful or severe, as the story demands. Do not fall back to a generic centered halftone-paper template.

Art direction: {spec['art_direction']}.
Color direction: {spec['color_direction']}. The empty animation keyframe uses {spec['background_hex']} as its base color; integrate that anchor naturally rather than letting it dictate the whole palette. Optional planner swatches: {accents}.
Composition direction: {spec['composition_direction']}.
Conceptual ingredients: {elements}. Reinterpret, combine, crop, abstract, or subordinate them freely; they are narrative ingredients, not a rigid object checklist.
Final relationship: {spec['final_frame']}

Make this frame feel specifically authored for this narration beat and visibly distinct from the other collage shots in the same video. Preserve enough separable visual structure for an assemble-from-empty animation.

Content exclusions only: no readable typography, letters, numerals, logos, watermarks, UI, or subtitles. These exclusions do not otherwise limit the collage aesthetic."""


def video_prompt(spec: dict[str, Any], frame: FrameSpec) -> str:
    order = " → ".join(spec["assembly_order"])
    target_duration = _seconds(spec.get("target_duration_seconds"), 8.0)
    return f"""Animate an editorial collage from Image 1, the exact empty first frame, to Image 2, the exact completed last frame. Keep one continuous {frame.aspect_ratio} shot and resolve precisely to the supplied Image 2 composition.

Motion direction: {spec['motion_direction']}.
Suggested narrative progression: {order}. Treat it as an expressive story arc, not a mandatory list of identical entrance moves. Freely vary timing, overlaps, reveals, material behavior, depth, and local movement to suit the art direction. Subtle camera or parallax motion is allowed only when it settles back into the exact supplied final framing.

Complete one non-repeating evolution across approximately {target_duration:.3f} seconds, then hold the supplied Image 2 composition. Never restart or loop any motion. Preserve the chosen collage language and do not introduce a generic stop-motion preset. Target running time: {target_duration:.3f} seconds.

No scene cuts, unrelated new objects, readable text, letters, numbers, logos, watermark, UI, or sound."""


def _rows(value: Any) -> list[dict[str, Any]]:
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
        if str(key).lower() == name:
            return str(value or "").lstrip("📁🔗 ").strip()
    return ""


async def _media_command(args: Sequence[str], *, timeout: int) -> None:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.communicate()
        raise RuntimeError(f"Media command timed out after {timeout}s: {args[0]}")
    if process.returncode:
        detail = stderr.decode(errors="replace")[-1200:] or stdout.decode(errors="replace")[-1200:]
        raise RuntimeError(f"Media command failed ({process.returncode}): {detail}")


async def _run_opencli_retry(
    args: list[str],
    *,
    timeout: int,
    label: str,
    non_retryable: Callable[[Exception], bool] | None = None,
    repeated_failure_signature: Callable[[Exception], str | None] | None = None,
    repeated_failure_threshold: int = 2,
    repeated_failure_error_code: str | None = None,
):
    """Retry transient browser failures, stopping on a proven capability gap."""
    last_error: Exception | None = None
    last_repeated_signature: str | None = None
    repeated_failure_hits = 0
    maximum_attempts = max(1, int(config.OPENCLI_MAX_ATTEMPTS))
    for attempt in range(1, maximum_attempts + 1):
        try:
            logger.info("%s attempt %s/%s", label, attempt, maximum_attempts)
            return await run_opencli(args, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - bounded browser retry
            last_error = exc
            logger.warning(
                "%s attempt %s/%s failed: %s",
                label,
                attempt,
                maximum_attempts,
                exc,
            )
            if non_retryable and non_retryable(exc):
                raise GeminiVideoUploadCapabilityError(
                    f"{label} cannot run in this browser session "
                    f"({GEMINI_VIDEO_UPLOAD_CAPABILITY_CODE}): {exc}",
                    error_code=GEMINI_VIDEO_UPLOAD_CAPABILITY_CODE,
                ) from exc
            repeated_signature = (
                repeated_failure_signature(exc)
                if repeated_failure_signature
                else None
            )
            if repeated_signature:
                if repeated_signature == last_repeated_signature:
                    repeated_failure_hits += 1
                else:
                    last_repeated_signature = repeated_signature
                    repeated_failure_hits = 1
            else:
                last_repeated_signature = None
                repeated_failure_hits = 0
            if repeated_failure_hits >= max(2, repeated_failure_threshold):
                message = (
                    f"{label} repeated the same stable failure "
                    f"{repeated_failure_hits} times "
                    f"(signature={repeated_signature}): {exc}"
                )
                if repeated_failure_error_code:
                    raise GeminiVideoUploadCapabilityError(
                        f"{message} ({repeated_failure_error_code})",
                        error_code=repeated_failure_error_code,
                    ) from exc
                raise OpenCLIError(message) from exc
            if attempt < maximum_attempts:
                await asyncio.sleep(min(45, 3 * 2 ** (attempt - 1)))
    raise OpenCLIError(
        f"{label} exhausted {maximum_attempts} OpenCLI attempts: {last_error}"
    ) from last_error


def _is_gemini_video_upload_capability_failure(error: Exception) -> bool:
    """Recognize the complete, stable Browser Bridge upload failure signature.

    One blocked mechanism is not enough: a transient DOM change can break an
    individual path. This only becomes non-retryable after native injection is
    denied, the extension allowlist blocks direct CDP evaluation, and Gemini
    clears the synthetic fallback without creating an attachment.
    """
    message = " ".join(str(error).split()).lower()
    if GEMINI_VIDEO_UPLOAD_CAPABILITY_CODE.lower() in message:
        return True
    return all(
        token in message
        for token in (
            "page.setfileinput",
            "-32000",
            "not allowed",
            "dom.setfileinputfiles",
            "cdp method not permitted",
            "runtime.evaluate",
            '"method":"datatransfer"',
            '"attachments":0',
            '"inputfiles":[[]]',
        )
    )


def _gemini_video_input_hydration_stuck_signature(
    error: Exception,
) -> str | None:
    """Return the adapter's canonical stuck-state fingerprint, if present."""
    message = str(error)
    if GEMINI_VIDEO_INPUT_HYDRATION_STUCK_CODE not in message:
        return None
    match = re.search(r'"stuckSignature":"([^"]+)"', message)
    return match.group(1) if match else None


def _chatgpt_image_composer_not_ready_signature(
    error: Exception,
) -> str | None:
    """Recognize the adapter's stable signed-out/unready image-composer error."""
    message = " ".join(str(error).split()).casefold()
    required = (
        "opencli chatgpt image failed",
        "code: command_exec",
        "message: failed to send image prompt to chatgpt",
        "help: open https://chatgpt.com/new and verify the composer is ready.",
    )
    if all(token in message for token in required):
        return CHATGPT_IMAGE_COMPOSER_NOT_READY_SIGNATURE
    return None


async def _generate_still(prompt: str, item_dir: Path) -> tuple[Path, str]:
    still_dir = item_dir / "stills"
    still_dir.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in still_dir.iterdir() if path.is_file()}
    timeout = config.COLLAGE_CHATGPT_TIMEOUT
    result = await _run_opencli_retry(
        [
            "chatgpt", "image", prompt,
            "--op", str(still_dir.resolve()),
            "--timeout", str(timeout),
            "--window", "background",
            "--site-session", "persistent",
            "--keep-tab", "false",
            "-f", "json",
        ],
        timeout=timeout + 60,
        label="ChatGPT collage still",
        repeated_failure_signature=_chatgpt_image_composer_not_ready_signature,
        repeated_failure_threshold=2,
    )
    parsed = _rows(first_json(result.stdout))
    reported: list[Path] = []
    for row in parsed:
        raw = _field(row, "file")
        if raw and raw != "-":
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = config.PROJECT_ROOT / candidate
            if candidate.is_file():
                reported.append(candidate.resolve())
    created = sorted(
        (
            path.resolve()
            for path in still_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES and path.resolve() not in before
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    images = list(dict.fromkeys([*reported, *created]))
    if not images:
        raise OpenCLIError("ChatGPT Web produced no downloaded collage still")
    conversation = next((_field(row, "link") for row in parsed if _field(row, "link")), "")
    return images[0], conversation


async def _prepare_frames(source: Path, item_dir: Path, color: str, frame: FrameSpec) -> tuple[Path, Path]:
    frames = item_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    original = frames / f"last-frame-original{source.suffix.lower()}"
    shutil.copyfile(source, original)
    first = frames / "first-frame.png"
    # JPEG keeps the browser-bridge fallback payload well below its message
    # ceiling while preserving the exact target pixels and collage detail.
    last = frames / "last-frame.jpg"
    await _media_command(
        [
            "ffmpeg", "-y", "-i", str(original), "-vf",
            f"scale={frame.media_width}:{frame.media_height}:force_original_aspect_ratio=increase,crop={frame.media_width}:{frame.media_height}",
            "-frames:v", "1", "-q:v", "3", str(last),
        ],
        timeout=120,
    )
    # Gemini Create Video can accept a visually empty paper field, but its
    # upload processor may reject an almost byte-empty, perfectly flat PNG.
    # Render deterministic paper fibre and tiny tonal variation so the first
    # frame remains an empty set while still being a real photographic texture.
    paper_hex = _color(color, "#315F4C").lstrip("#")
    base = tuple(int(paper_hex[index : index + 2], 16) for index in (0, 2, 4))
    paper = Image.new("RGB", (frame.media_width, frame.media_height), base)
    paper_draw = ImageDraw.Draw(paper, "RGBA")
    paper_rng = random.Random(
        int(hashlib.sha256(f"{paper_hex}:{frame.aspect_ratio}".encode()).hexdigest()[:16], 16)
    )
    for y in range(4, frame.media_height, 5):
        shade = paper_rng.choice((-7, -5, -3, 3, 5, 7))
        tone = tuple(max(0, min(255, channel + shade)) for channel in base)
        paper_draw.line(
            (0, y, frame.media_width, y + paper_rng.choice((-1, 0, 1))),
            fill=(*tone, 62),
        )
    for _ in range(max(6000, frame.media_width * frame.media_height // 120)):
        x = paper_rng.randrange(frame.media_width)
        y = paper_rng.randrange(frame.media_height)
        shade = paper_rng.choice((-16, -12, -8, 8, 12, 16))
        tone = tuple(max(0, min(255, channel + shade)) for channel in base)
        paper_draw.point((x, y), fill=(*tone, 105))
    for _ in range(420):
        x = paper_rng.randrange(frame.media_width)
        y = paper_rng.randrange(frame.media_height)
        length = paper_rng.randrange(8, 65)
        shade = paper_rng.choice((-10, -7, 7, 10))
        tone = tuple(max(0, min(255, channel + shade)) for channel in base)
        paper_draw.line((x, y, min(frame.media_width, x + length), y), fill=(*tone, 70))
    paper.save(first, format="PNG", optimize=True)
    return first, last


def _color(value: str, fallback: str) -> str:
    aliases = {
        "amber": "#D2A928",
        "charcoal": "#27272A",
        "cream": "#F2E7CF",
        "cyan": "#43B9C4",
        "gold": "#D2A928",
        "teal": "#188C85",
        "violet": "#7257A8",
        "warm cream": "#F2E7CF",
    }
    text = str(value or "").strip().lower()
    return str(value).upper() if _HEX.fullmatch(str(value or "")) else aliases.get(text, fallback)


async def _render_local_still(spec: dict[str, Any], item_dir: Path, frame: FrameSpec) -> Path:
    """Render a deterministic paper-cut collage when the web still is unavailable."""
    still_dir = item_dir / "stills"
    still_dir.mkdir(parents=True, exist_ok=True)
    output = still_dir / "local-paper-collage.png"
    width, height = frame.media_width, frame.media_height
    background = _color(str(spec.get("background_hex") or ""), "#315F4C")
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image, "RGBA")

    # Quiet paper fibre and registration marks keep the fallback recognisably
    # editorial without relying on fonts, logos, or network assets.
    for y in range(8, height, 24):
        alpha = 10 if (y // 24) % 2 else 7
        draw.line((0, y, width, y + 2), fill=(255, 255, 255, alpha), width=1)

    seed_source = json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")
    rng = random.Random(int(hashlib.sha256(seed_source).hexdigest()[:16], 16))
    accents = [
        _color(value, "#43B9C4") for value in (spec.get("accent_colors") or [])
    ] or ["#43B9C4", "#D2A928"]
    paper_colors = ["#F2E7CF", "#202124", *accents]
    element_count = min(6, max(3, len(spec.get("elements") or [])))
    center_x, center_y = width // 2, height // 2

    for index in range(element_count):
        piece_w = int(width * rng.uniform(0.12, 0.24))
        piece_h = int(height * rng.uniform(0.15, 0.34))
        angle = (index / max(1, element_count)) * math.tau + rng.uniform(-0.35, 0.35)
        radius_x = width * rng.uniform(0.10, 0.27)
        radius_y = height * rng.uniform(0.08, 0.24)
        x = int(center_x + radius_x * math.cos(angle) - piece_w / 2)
        y = int(center_y + radius_y * math.sin(angle) - piece_h / 2)
        x = max(40, min(width - piece_w - 40, x))
        y = max(40, min(height - piece_h - 40, y))

        piece = Image.new("RGBA", (piece_w + 48, piece_h + 48), (0, 0, 0, 0))
        mask = Image.new("L", piece.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        bounds = (24, 24, 24 + piece_w, 24 + piece_h)
        shape = index % 3
        if shape == 0:
            mask_draw.rounded_rectangle(bounds, radius=max(14, piece_w // 10), fill=255)
        elif shape == 1:
            mask_draw.ellipse(bounds, fill=255)
        else:
            mask_draw.polygon(
                [(24 + piece_w // 2, 24), (24 + piece_w, 24 + piece_h), (24, 24 + piece_h)],
                fill=255,
            )
        shadow = Image.new("RGBA", piece.size, (0, 0, 0, 0))
        shadow.putalpha(mask.filter(ImageFilter.GaussianBlur(12)))
        shadow_color = Image.new("RGBA", piece.size, (0, 0, 0, 95))
        shadow_color.putalpha(shadow.getchannel("A"))
        piece.alpha_composite(shadow_color, (8, 10))

        color = paper_colors[index % len(paper_colors)]
        fill = Image.new("RGBA", piece.size, color)
        fill.putalpha(mask)
        piece.alpha_composite(fill)
        piece_draw = ImageDraw.Draw(piece, "RGBA")
        if index % 2 == 0:
            for dot_y in range(32, piece_h + 24, 16):
                for dot_x in range(32, piece_w + 24, 16):
                    if mask.getpixel((dot_x, dot_y)) > 0:
                        piece_draw.ellipse((dot_x - 2, dot_y - 2, dot_x + 2, dot_y + 2), fill=(0, 0, 0, 95))
        rotated = piece.rotate(rng.uniform(-8, 8), resample=Image.Resampling.BICUBIC, expand=True)
        image.paste(rotated, (x - 24, y - 24), rotated)

    image.save(output, format="PNG", optimize=True)
    return output


async def _animate_still_locally(
    first: Path,
    last: Path,
    item_dir: Path,
    frame: FrameSpec,
    target_duration: float,
) -> Path:
    """Assemble staggered paper tiles once, ending on the exact completed still."""
    video_dir = item_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    raw = video_dir / "local-paper-assembly.mp4"
    width, height = frame.media_width, frame.media_height
    with Image.open(first) as source:
        background = ImageOps.fit(source.convert("RGB"), (width, height))
    with Image.open(last) as source:
        completed = ImageOps.fit(source.convert("RGB"), (width, height))

    tile_specs: list[dict[str, Any]] = []
    columns, rows = 3, 2
    directions = ((-1, 0), (0, -1), (1, 0), (-1, 0), (0, 1), (1, 0))
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            left = round(column * width / columns)
            top = round(row * height / rows)
            right = round((column + 1) * width / columns)
            bottom = round((row + 1) * height / rows)
            tile = completed.crop((left, top, right, bottom)).convert("RGBA")
            tile_specs.append(
                {
                    "image": tile,
                    "target": (left, top),
                    "direction": directions[index],
                    "rotation": (-1.4, 0.8, -0.5, 1.1, -0.9, 0.5)[index],
                    "phase": index * 0.91,
                }
            )

    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(CLIP_FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(raw),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None
    total_frames = max(1, round(target_duration * CLIP_FPS))
    try:
        for frame_index in range(total_frames):
            progress = frame_index / max(1, total_frames - 1)
            canvas = background.copy().convert("RGBA")
            for index, tile_spec in enumerate(tile_specs):
                entrance_start = 0.06 + index * 0.095
                entrance_progress = min(1.0, max(0.0, (progress - entrance_start) / 0.25))
                if entrance_progress <= 0:
                    continue
                # Back-ease gives each paper piece a physical snap on arrival.
                shifted = entrance_progress - 1
                eased = 1 + 2.70158 * shifted**3 + 1.70158 * shifted**2
                direction_x, direction_y = tile_spec["direction"]
                target_x, target_y = tile_spec["target"]
                travel = 1 - eased
                travel_x = direction_x * width * 0.72 * travel
                travel_y = direction_y * height * 0.72 * travel
                # Drift settles back to zero so the last frame is the completed
                # supplied composition, not the beginning of another cycle.
                ambient = math.sin(math.pi * progress) * entrance_progress
                drift_x = math.sin(progress * math.tau * 1.15 + tile_spec["phase"]) * 4 * ambient
                drift_y = math.cos(progress * math.tau * 0.9 + tile_spec["phase"]) * 3 * ambient
                rotation = travel * direction_x * 10 + tile_spec["rotation"] * ambient
                piece = tile_spec["image"].rotate(
                    rotation,
                    resample=Image.Resampling.BICUBIC,
                    expand=True,
                )
                shadow = Image.new("RGBA", piece.size, (0, 0, 0, 0))
                shadow.putalpha(piece.getchannel("A").filter(ImageFilter.GaussianBlur(7)))
                shadow_layer = Image.new("RGBA", piece.size, (0, 0, 0, 72))
                shadow_layer.putalpha(shadow.getchannel("A"))
                x = round(target_x + travel_x + drift_x - (piece.width - tile_spec["image"].width) / 2)
                y = round(target_y + travel_y + drift_y - (piece.height - tile_spec["image"].height) / 2)
                canvas.alpha_composite(shadow_layer, (x + 7, y + 9))
                canvas.alpha_composite(piece, (x, y))

            # A reversible camera push keeps the one-pass assembly alive while
            # still resolving to the exact completed frame.
            zoom = 1.0 + 0.025 * math.sin(math.pi * progress)
            if zoom > 1:
                enlarged = canvas.resize(
                    (round(width * zoom), round(height * zoom)),
                    Image.Resampling.BICUBIC,
                )
                x_offset = (enlarged.width - width) // 2
                y_offset = (enlarged.height - height) // 2
                canvas = enlarged.crop((x_offset, y_offset, x_offset + width, y_offset + height))
            process.stdin.write(canvas.convert("RGB").tobytes())
            if frame_index % 4 == 3:
                await process.stdin.drain()
        process.stdin.close()
        await process.stdin.wait_closed()
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
    except Exception:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(stderr.decode(errors="replace")[-1600:])
    return raw


async def _generate_video(prompt: str, first: Path, last: Path, item_dir: Path, frame: FrameSpec) -> tuple[Path, str]:
    video_dir = item_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    raw = video_dir / "gemini-web-original.mp4"
    timeout = config.COLLAGE_GEMINI_TIMEOUT
    result = await _run_opencli_retry(
        [
            "gemini", "video", prompt,
            "--first", str(first.resolve()),
            "--last", str(last.resolve()),
            "--aspect", frame.aspect_ratio,
            "--output", str(raw.resolve()),
            "--timeout", str(timeout),
            # Gemini's local-file picker does not reliably hydrate in a
            # background Chrome tab. Foreground is required to retain both
            # keyframes, matching the rendered-frame review path.
            "--window", "foreground",
            "--site-session", "persistent",
            "--keep-tab", "false",
            "-f", "json",
        ],
        timeout=timeout + 120,
        label="Gemini collage video",
        non_retryable=_is_gemini_video_upload_capability_failure,
        repeated_failure_signature=_gemini_video_input_hydration_stuck_signature,
        repeated_failure_threshold=2,
        repeated_failure_error_code=GEMINI_VIDEO_INPUT_HYDRATION_STUCK_CODE,
    )
    if not raw.is_file() or raw.stat().st_size < 1024:
        raise OpenCLIError("Gemini Web Create Video produced no downloaded MP4")
    rows = _rows(first_json(result.stdout))
    conversation = next((_field(row, "link") for row in rows if _field(row, "link")), "")
    return raw, conversation


async def _normalize_video(
    raw: Path,
    item_dir: Path,
    frame: FrameSpec,
    target_duration: float,
) -> Path:
    """Trim and normalize a generated clip without replaying source frames."""
    final = _final_clip_path(item_dir, target_duration)
    await _media_command(
        [
            "ffmpeg", "-y", "-i", str(raw),
            "-t", str(target_duration), "-vf",
            f"scale={frame.media_width}:{frame.media_height}:force_original_aspect_ratio=increase,crop={frame.media_width}:{frame.media_height},fps={CLIP_FPS}",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final),
        ],
        timeout=300,
    )
    return final


async def probe_video(
    path: Path,
    frame: FrameSpec,
    target_duration: float,
) -> dict[str, Any]:
    process = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(stderr.decode(errors="replace")[-800:])
    data = json.loads(stdout)
    videos = [stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"]
    audio = [stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"]
    stream = videos[0] if videos else {}
    rate = str(stream.get("avg_frame_rate") or "0/1").split("/")
    fps = float(rate[0]) / max(1.0, float(rate[1]))
    duration = float(data.get("format", {}).get("duration") or stream.get("duration") or 0)
    motion = await _probe_motion(
        path,
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        duration_seconds=target_duration,
    )
    required_motion_seconds = min(4.0, max(1.0, target_duration * 0.5))
    checks = {
        "dimensions": (stream.get("width"), stream.get("height")) == (frame.media_width, frame.media_height),
        "duration": abs(duration - target_duration) <= 0.15,
        "fps": abs(fps - CLIP_FPS) <= 0.1,
        "no_audio": not audio,
        "sustained_motion": motion["active_seconds"] >= required_motion_seconds,
        # The workflow starts on an empty field and ends on the assembled
        # composition. Matching endpoints indicate a loop or failed assembly.
        "non_repeating_endpoints": motion["first_last_delta"] >= 1.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "width": stream.get("width"),
        "height": stream.get("height"),
        "duration": round(duration, 3),
        "target_duration": round(target_duration, 3),
        "fps": round(fps, 3),
        "audio_streams": len(audio),
        "motion": motion,
    }


async def _probe_motion(
    path: Path,
    *,
    width: int,
    height: int,
    duration_seconds: float,
) -> dict[str, Any]:
    """Measure one-pass motion and distinguish the empty and completed endpoints."""
    if width <= 0 or height <= 0:
        return {"active_seconds": 0, "second_scores": [], "first_last_delta": 0.0}
    sample_width = 96
    sample_height = max(2, round(height * sample_width / width))
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"fps={MOTION_SAMPLE_FPS},scale={sample_width}:{sample_height},format=gray",
        "-f",
        "rawvideo",
        "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(stderr.decode(errors="replace")[-800:])
    frame_size = sample_width * sample_height
    frames = [stdout[offset : offset + frame_size] for offset in range(0, len(stdout), frame_size)]
    frames = [sample for sample in frames if len(sample) == frame_size]
    second_scores: list[float] = []
    active_seconds = 0.0
    for second in range(max(1, math.ceil(duration_seconds))):
        start = second * MOTION_SAMPLE_FPS
        end = min(len(frames) - 1, (second + 1) * MOTION_SAMPLE_FPS)
        deltas: list[float] = []
        for index in range(start, end):
            before, after = frames[index], frames[index + 1]
            deltas.append(sum(abs(left - right) for left, right in zip(before, after, strict=True)) / frame_size)
        score = round(sum(deltas) / len(deltas), 3) if deltas else 0.0
        second_scores.append(score)
        if score >= 0.35:
            active_seconds += min(1.0, max(0.0, duration_seconds - second))
    first_last_delta = 0.0
    if len(frames) >= 2:
        first_last_delta = sum(
            abs(left - right) for left, right in zip(frames[0], frames[-1], strict=True)
        ) / frame_size
    return {
        "active_seconds": round(active_seconds, 3),
        "second_scores": second_scores,
        "first_last_delta": round(first_last_delta, 3),
    }


async def _contact_sheet(
    video: Path,
    item_dir: Path,
    frame: FrameSpec,
    target_duration: float,
) -> Path:
    sheet = item_dir / "video" / "contact-sheet.jpg"
    thumb_w, thumb_h = ((256, 144) if not frame.is_portrait else (144, 256))
    columns = max(1, math.ceil(target_duration))
    await _media_command(
        [
            "ffmpeg", "-y", "-i", str(video), "-vf",
            f"fps=1,scale={thumb_w}:{thumb_h},tile={columns}x1", "-frames:v", "1", str(sheet),
        ],
        timeout=120,
    )
    return sheet


async def generate_collage_broll(
    storyboard: dict,
    task_dir: Path,
    *,
    count: int,
    force_opening: bool,
    frame: FrameSpec,
    provider_id: int | None = None,
    ai_endpoint: str | None = None,
    ai_model: str | None = None,
    log: LogCallback | None = None,
) -> dict[str, Any]:
    """Run the former three-gate workflow automatically, one web job at a time."""
    _validate_storyboard_scene_ids(storyboard)
    _ensure_owned_directory(task_dir)
    root = task_dir / "collage_broll"
    _ensure_owned_directory(root, parent=task_dir)
    manifest_path = root / "manifest.json"
    specs_path = root / "visual-spec.json"
    for cache_file in (manifest_path, specs_path, root / "visual-spec.json.tmp"):
        if cache_file.is_symlink():
            raise RuntimeError(f"Collage cache file cannot be a symlink: {cache_file}")
    planning_fingerprint = _planning_fingerprint(
        storyboard,
        count=count,
        force_opening=force_opening,
        frame=frame,
    )
    manifest: dict[str, Any] = {
        "status": "planning",
        "cache_contract_version": CACHE_CONTRACT_VERSION,
        "planning_fingerprint": planning_fingerprint,
        "visual_spec_cache": "miss",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "approval_gates": [],
        "approval_mode": "automatic",
        "planner": "claude_agent_sdk",
        "still_provider": "chatgpt_web_via_opencli",
        "video_provider": "gemini_web_create_video_via_opencli",
        "gemini_video_upload_capability": {
            "status": "unknown",
            "error_code": None,
            "detected_at_scene_id": None,
        },
        "gemini_api_key_used": False,
        "orientation": frame.orientation,
        "aspect_ratio": frame.aspect_ratio,
        "requested_count": count,
        "gemini_max_clip_seconds": config.COLLAGE_GEMINI_MAX_SECONDS,
        "playback_policy": PLAYBACK_POLICY,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "items": [],
        "errors": [],
    }
    cached_specs: list[dict[str, Any]] = []
    cache_hit = False
    scenes = list(storyboard.get("scenes") or [])
    target_count = min(max(1, count), len(scenes)) if scenes else 0
    if specs_path.is_file():
        try:
            payload = json.loads(specs_path.read_text(encoding="utf-8"))
            valid_scene_ids = {
                str(scene.get("id") or "") for scene in scenes
            }
            candidate_specs = payload.get("specs") if isinstance(payload, dict) else None
            candidate_specs_sha256 = (
                _fingerprint(candidate_specs) if isinstance(candidate_specs, list) else ""
            )
            candidate_ids = (
                [str(item.get("scene_id") or "") for item in candidate_specs]
                if isinstance(candidate_specs, list)
                and all(isinstance(item, dict) for item in candidate_specs)
                else []
            )
            if (
                isinstance(payload, dict)
                and payload.get("cache_contract_version") == CACHE_CONTRACT_VERSION
                and payload.get("planning_fingerprint") == planning_fingerprint
                and payload.get("specs_sha256") == candidate_specs_sha256
                and isinstance(candidate_specs, list)
                and len(candidate_specs) == target_count
                and len(candidate_ids) == len(set(candidate_ids))
                and (
                    not force_opening
                    or (
                        bool(candidate_specs)
                        and candidate_ids[0] == str((scenes or [{}])[0].get("id") or "")
                    )
                )
                and all(scene_id in valid_scene_ids for scene_id in candidate_ids)
            ):
                cached_specs = candidate_specs
                cache_hit = True
        except (
            OSError,
            TypeError,
            ValueError,
            UnicodeError,
            json.JSONDecodeError,
        ):
            cached_specs = []
    _write_json(manifest_path, manifest)
    if cache_hit:
        specs = cached_specs
        manifest["visual_spec_cache"] = "hit"
        _log(log, f"Collage B-roll: reusing {len(specs)} existing visual spec(s)")
    else:
        specs = await plan_specs(
            storyboard,
            count=count,
            force_opening=force_opening,
            frame=frame,
            provider_id=provider_id,
            ai_endpoint=ai_endpoint,
            ai_model=ai_model,
            log=log,
        )
    scenes_by_id = {
        str(scene.get("id") or ""): scene for scene in storyboard.get("scenes") or []
    }
    safe_specs: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        scene_id = str(spec.get("scene_id") or "")
        if scene_id not in scenes_by_id:
            continue
        scene = scenes_by_id[scene_id]
        timed_spec = _with_scene_timing(spec, scene)
        if not _is_strict_json_value(timed_spec):
            _log(
                log,
                f"Collage B-roll: rejected non-finite/non-JSON spec for {scene_id}; "
                "using narration-derived fallback",
            )
            timed_spec = _fallback_spec(scene, index)
        safe_specs.append(timed_spec)
    specs = safe_specs
    specs_sha256 = _fingerprint(specs)
    manifest["specs_sha256"] = specs_sha256
    _write_json(
        specs_path,
        {
            "cache_contract_version": CACHE_CONTRACT_VERSION,
            "planning_fingerprint": planning_fingerprint,
            "specs_sha256": specs_sha256,
            "specs": specs,
        },
    )
    manifest["status"] = "generating"
    _write_json(manifest_path, manifest)
    gemini_video_upload_unavailable = False

    for index, spec in enumerate(specs, start=1):
        scene_id = str(spec["scene_id"])
        scene = scenes_by_id[scene_id]
        target_duration = _seconds(spec.get("target_duration_seconds"), 8.0)
        item_dir = root / f"{index:02d}-{scene_id}"
        _ensure_owned_directory(item_dir, parent=root)
        _assert_item_stage_paths_safe(item_dir)
        still_prompt = image_prompt(spec, frame)
        motion_prompt = video_prompt(spec, frame)
        semantic_fingerprint = _scene_semantic_fingerprint(storyboard, scene)
        still_bindings = _still_contract_bindings(
            scene_id=scene_id,
            semantic_fingerprint=semantic_fingerprint,
            spec=spec,
            still_prompt=still_prompt,
            frame=frame,
        )
        final_bindings = _final_contract_bindings(
            scene_id=scene_id,
            semantic_fingerprint=semantic_fingerprint,
            spec=spec,
            still_prompt=still_prompt,
            motion_prompt=motion_prompt,
            frame=frame,
            target_duration=target_duration,
        )
        item: dict[str, Any] = {
            "scene_id": scene_id,
            "status": "generating",
            "spec": spec,
            "scene_semantic_fingerprint": semantic_fingerprint,
            "spec_fingerprint": still_bindings["spec_fingerprint"],
            "image_prompt_sha256": still_bindings["image_prompt_sha256"],
            "video_prompt_sha256": final_bindings["video_prompt_sha256"],
            "still_fingerprint": still_bindings["fingerprint"],
            "final_fingerprint": final_bindings["fingerprint"],
            "cache_reuse": {"still": False, "final": False},
            "still_path": "",
            "video_path": "",
            "contact_sheet": "",
            "chatgpt_url": "",
            "gemini_url": "",
            "script_duration_seconds": spec["script_duration_seconds"],
            "target_duration_seconds": target_duration,
            "playback_policy": PLAYBACK_POLICY,
            "qa": {},
            "generation_warnings": [],
            "error": None,
        }
        manifest["items"].append(item)
        _write_json(manifest_path, manifest)
        try:
            expected_final = _final_clip_path(item_dir, target_duration)
            # Validate all persisted evidence before touching either prompt.
            # A final contract is self-contained because it binds the exact
            # clip and attach hold frame as well as all semantic/prompt inputs.
            cached_final = _load_final_cache(
                item_dir,
                expected=final_bindings,
                still_prompt=still_prompt,
                motion_prompt=motion_prompt,
                final_path=expected_final,
            )
            if cached_final is not None:
                final_artifact, cached_hold = cached_final
                try:
                    cached_qa = await probe_video(
                        final_artifact, frame, target_duration
                    )
                    if cached_qa["passed"]:
                        # The sheet is a disposable review derivative; rebuild
                        # it from the verified final rather than trusting cache.
                        cached_sheet = await _contact_sheet(
                            final_artifact, item_dir, frame, target_duration
                        )
                        item.update(
                            {
                                "status": "ready",
                                "still_path": str(cached_hold.relative_to(task_dir)),
                                "video_path": str(final_artifact.relative_to(task_dir)),
                                "contact_sheet": str(cached_sheet.relative_to(task_dir)),
                                "still_provider": "existing_verified_artifact",
                                "video_provider": "existing_verified_artifact",
                                "cache_reuse": {"still": False, "final": True},
                                "qa": cached_qa,
                            }
                        )
                        _log(
                            log,
                            f"Collage B-roll {index}/{len(specs)}: "
                            f"reused semantic-verified clip for {scene_id}",
                        )
                        _write_json(manifest_path, manifest)
                        continue
                except Exception as exc:  # noqa: BLE001 - regenerate bad cache
                    _log(
                        log,
                        f"Collage B-roll {index}/{len(specs)}: "
                        f"cached final validation failed for {scene_id} ({exc})",
                    )

            cached_still = _load_still_cache(
                item_dir,
                expected=still_bindings,
                still_prompt=still_prompt,
            )
            still_stage_present = _still_stage_has_content(item_dir)
            downstream_stage_present = _downstream_stage_has_content(item_dir)
            if cached_still is None and still_stage_present:
                # A legacy/mismatched still cannot seed any new downstream
                # artifact. Remove only this item's owned pipeline stages.
                _clear_all_stages(item_dir)
                downstream_stage_present = False
            elif downstream_stage_present:
                # Missing, mismatched, incomplete, or QA-failed final evidence
                # invalidates frames/video, but an independently valid still
                # remains reusable.
                _clear_downstream_stage(item_dir)

            # Prompt writes deliberately happen after old contract/artifact
            # validation and cleanup, so they cannot make stale media appear
            # current after a crash.
            _write_text(item_dir / "image-prompt.txt", still_prompt + "\n")
            _write_text(item_dir / "video-prompt.txt", motion_prompt + "\n")
            _log(log, f"Collage B-roll {index}/{len(specs)}: generating {frame.aspect_ratio} still for {scene_id}")
            if cached_still is not None:
                still = cached_still
                chatgpt_url = ""
                item["still_provider"] = "existing_verified_artifact"
                item["cache_reuse"]["still"] = True
                _log(
                    log,
                    f"Collage B-roll {index}/{len(specs)}: "
                    f"reusing semantic-verified still for {scene_id}",
                )
            else:
                try:
                    still, chatgpt_url = await _generate_still(still_prompt, item_dir)
                    item["still_provider"] = "chatgpt_web_via_opencli"
                except Exception as exc:  # noqa: BLE001 - local renderer preserves requested count
                    warning = f"Web still unavailable ({exc}); used deterministic local paper collage"
                    item["generation_warnings"].append(warning)
                    _log(log, f"Collage B-roll {index}/{len(specs)}: {warning}")
                    still = await _render_local_still(spec, item_dir, frame)
                    chatgpt_url = ""
                    item["still_provider"] = "deterministic_local_paper_collage"
                _assert_item_stage_paths_safe(item_dir)
                still = _materialize_still(still, item_dir)
            first, last = await _prepare_frames(still, item_dir, spec["background_hex"], frame)
            _assert_item_stage_paths_safe(item_dir)
            if cached_still is None:
                # Frame preparation is the still decode gate. Commit the proof
                # only after it succeeds; a partial download can never become
                # a permanently reusable, failing still.
                _write_still_contract(item_dir, still_bindings, still)
            _log(
                log,
                f"Collage B-roll {index}/{len(specs)}: animating {scene_id} once for "
                f"{target_duration:.3f}s in Gemini Web Create Video",
            )
            if gemini_video_upload_unavailable:
                warning = (
                    "Gemini video upload capability unavailable for this compose; "
                    "skipped web generation and used deterministic local paper assembly"
                )
                item["generation_warnings"].append(warning)
                _log(log, f"Collage B-roll {index}/{len(specs)}: {warning}")
                raw = await _animate_still_locally(
                    first, last, item_dir, frame, target_duration
                )
                gemini_url = ""
                item["video_provider"] = "deterministic_local_paper_assembly"
            else:
                try:
                    raw, gemini_url = await _generate_video(
                        motion_prompt, first, last, item_dir, frame
                    )
                    item["video_provider"] = "gemini_web_create_video_via_opencli"
                    manifest["gemini_video_upload_capability"]["status"] = "available"
                except Exception as exc:  # noqa: BLE001 - local animation preserves requested count
                    if isinstance(exc, GeminiVideoUploadCapabilityError):
                        gemini_video_upload_unavailable = True
                        manifest["gemini_video_upload_capability"].update(
                            {
                                "status": "unavailable",
                                "error_code": exc.error_code,
                                "detected_at_scene_id": scene_id,
                            }
                        )
                    warning = (
                        f"Web video unavailable ({exc}); "
                        "used deterministic local paper assembly"
                    )
                    item["generation_warnings"].append(warning)
                    _log(log, f"Collage B-roll {index}/{len(specs)}: {warning}")
                    raw = await _animate_still_locally(
                        first, last, item_dir, frame, target_duration
                    )
                    gemini_url = ""
                    item["video_provider"] = "deterministic_local_paper_assembly"
            _assert_item_stage_paths_safe(item_dir)
            final = await _normalize_video(raw, item_dir, frame, target_duration)
            qa = await probe_video(final, frame, target_duration)
            if not qa["passed"]:
                raise RuntimeError(f"normalized collage clip failed QA: {qa['checks']}")
            sheet = await _contact_sheet(final, item_dir, frame, target_duration)
            _assert_item_stage_paths_safe(item_dir)
            # Publish the reusable-final proof last. Until QA, hold-frame, and
            # contact-sheet work all succeed, an existing MP4 stays untrusted.
            _write_final_contract(item_dir, final_bindings, final, last)
            item.update(
                {
                    "status": "ready",
                    "still_path": str(last.relative_to(task_dir)),
                    "video_path": str(final.relative_to(task_dir)),
                    "contact_sheet": str(sheet.relative_to(task_dir)),
                    "chatgpt_url": chatgpt_url,
                    "gemini_url": gemini_url,
                    "qa": qa,
                }
            )
            _log(log, f"Collage B-roll {index}/{len(specs)}: ready for {scene_id}")
        except Exception as exc:  # noqa: BLE001 - preserve other clips and fallback scene
            item.update({"status": "failed", "error": str(exc)})
            manifest["errors"].append({"scene_id": scene_id, "message": str(exc)})
            _log(log, f"Collage B-roll {index}/{len(specs)} failed for {scene_id}: {exc}")
        _write_json(manifest_path, manifest)

    ready = sum(1 for item in manifest["items"] if item["status"] == "ready")
    manifest["status"] = "ready" if ready == len(specs) else ("partial" if ready else "failed")
    manifest["ready_count"] = ready
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    return manifest


def attach_collage(plans: list[dict], manifest: dict | None, task_dir: Path) -> int:
    """Promote successful generated items to clean, full-bleed scene plates."""
    by_id = {plan.get("id"): plan for plan in plans}
    plan_positions = {plan.get("id"): index for index, plan in enumerate(plans)}
    occupied = {
        str(plan.get("id"))
        for plan in plans
        if plan.get("archetype") == "footage" and plan.get("footage_src")
    }
    attached = 0
    for item in (manifest or {}).get("items") or []:
        if item.get("status") != "ready":
            continue
        preferred_id = str(item.get("scene_id") or "")
        plan = by_id.get(preferred_id)
        raw = str(item.get("video_path") or "")
        path = task_dir / raw
        if not plan or not raw or not path.is_file():
            continue
        already_attached = bool(
            plan.get("collage_broll")
            and str(plan.get("collage_source_scene_id") or "") == preferred_id
        )
        if preferred_id in occupied and not already_attached:
            preferred_position = plan_positions.get(preferred_id, 0)
            candidates = [
                candidate
                for candidate in plans
                if str(candidate.get("id") or "") not in occupied
            ]
            if not candidates:
                continue
            plan = min(
                candidates,
                key=lambda candidate: abs(
                    plan_positions.get(str(candidate.get("id") or ""), 0)
                    - preferred_position
                ),
            )
        placed_scene_id = str(plan.get("id") or "")
        hold_raw = str(item.get("still_path") or "")
        hold_path = task_dir / hold_raw
        script_duration = _seconds(item.get("script_duration_seconds"))
        target_duration = _seconds(
            item.get("target_duration_seconds"),
            _seconds((item.get("qa") or {}).get("duration")),
        )
        if not hold_raw or not hold_path.is_file():
            # HyperFrames does not reliably retain an ended video's frame.
            # Keep the ordinary deterministic scene instead of attaching a
            # collage that would turn black for the rest of the narration.
            continue
        plan.update(
            {
                "archetype": "footage",
                # Scene files live in compositions/, so media stored relative
                # to the render-project root needs one parent hop.
                "footage_src": f"../{raw}",
                "footage_kind": "video",
                "footage_credit": "",
                "collage_broll": True,
                "collage_source_scene_id": preferred_id,
                "collage_placed_scene_id": placed_scene_id,
                "collage_metaphor": item.get("spec", {}).get("visual_metaphor", ""),
                "collage_qa": item.get("qa", {}),
                "collage_hold_src": (
                    f"../{hold_raw}" if hold_raw and hold_path.is_file() else ""
                ),
                "collage_script_duration_seconds": script_duration,
                "collage_target_duration_seconds": target_duration,
                "collage_playback_policy": "play_once_then_hold_last_frame",
            }
        )
        item["placed_scene_id"] = placed_scene_id
        occupied.add(placed_scene_id)
        attached += 1
    if manifest is not None:
        manifest["placed_count"] = attached
        _write_json(task_dir / "collage_broll" / "manifest.json", manifest)
    return attached
