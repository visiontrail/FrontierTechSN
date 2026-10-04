"""Configured-provider 3D direction rendered locally with Remotion and R3F.

The model returns bounded scene data, never executable code. Existing narration,
editorial copy and shot intervals remain authoritative; acquired source media is
never replaced. When requested 3D cannot be produced, the stage fails explicitly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from PIL import Image

from backend import config
from backend.pipeline.process_logging import stream_subprocess
from backend.pipeline.timing import timed

CONTRACT = "three-d-explainer-v1"
RENDER_CONTRACT = "three-d-render-v1"
MAX_BATCH = 3
MAX_PLAN_ATTEMPTS = 3
_KINDS = {"chip", "rack", "satellite", "robot", "data", "object"}
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,47}$")
_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
# Clarifies the existing validator; changing creative direction still belongs in
# the versioned base prompt, whose hash invalidates saved model plans.
_ACTOR_CONTRACT = {
    "primitive": "Only kind=object may contain primitive (box, sphere, cylinder, torus). "
                 "Omit primitive entirely for chip, rack, satellite, robot and data; "
                 "these kinds already select a fixed mesh assembly. Choose object only "
                 "when generic geometry is the intended subject, preserving narration meaning.",
    "scale": "Every coordinate must be between 0.1 and 4 inclusive, including thin surfaces.",
    "explode": "Nonzero explode/end_explode is supported only by chip, rack and satellite.",
}
_REPAIR_CONTRACT = (
    "Repair the rejected scene using the exact field paths and values in validation_feedback. "
    "Keep valid fields unchanged, including camera, actors and motion, unless the reported "
    "failure requires changing them. Copy narration_excerpt verbatim from the supplied shot, "
    "including punctuation; do not normalize apostrophes or quotes. Recheck all numeric bounds "
    "before returning the complete repaired scene. Never clamp values automatically or change "
    "the narrated meaning to satisfy validation."
)
_TEXT_LIMITS = {"semantic_intent": 600, "narration_excerpt": 5000}
_TEXT_CONTRACT = {
    "maximum_characters": _TEXT_LIMITS,
    "requirements": "Both fields must be nonempty strings. Character counts include punctuation "
                    "and whitespace. Keep semantic_intent within its limit by summarizing the "
                    "narration-supported purpose and spatial action; do not repeat numeric "
                    "transforms or validation repairs there. Preserve narration_excerpt verbatim "
                    "from the supplied shot. Never silently truncate either field.",
}
logger = logging.getLogger(__name__)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _read(path: Path) -> dict:
    try:
        result = json.loads(path.read_text())
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


def _rejection(scene_id: str, raw, error: Exception) -> dict:
    # Invalid nonfinite numbers remain rejected. Only the diagnostic copy is
    # converted so the original failure can be persisted as strict JSON.
    diagnostic = json.loads(json.dumps(raw), parse_constant=str)
    return {"id": scene_id, "error": str(error), "spec": diagnostic}


def _number(value, low: float, high: float, *, field: str = "value") -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"3D {field}={value!r}: coordinates must be numeric")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"3D {field}={value!r}: numeric value must lie within {low}..{high}")
    return value


def _vector(value, low: float, high: float, *, field: str = "vector") -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"3D {field}={value!r}: vector needs exactly three coordinates")
    result, errors = [], []
    for index, item in enumerate(value):
        try:
            result.append(_number(item, low, high, field=f"{field}[{index}]"))
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("; ".join(errors))
    return result


def _color(value) -> str:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError("3D colors must be six-digit hex strings")
    return value


def _text(value, label: str) -> str:
    maximum = _TEXT_LIMITS[label]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"3D {label} must be nonempty text of at most {maximum} characters")
    if len(value) > maximum:
        raise ValueError(f"3D {label} length={len(value)}: text must be at most {maximum} characters")
    return value.strip()


def _route_receipt(endpoint: str, model: str) -> dict:
    """Describe a configured route without credentials or URL query secrets."""
    parsed = urlsplit(endpoint)
    return {"endpoint": urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1],
                                    parsed.path, "", "")), "model": model}


def validate_spec(raw: dict, context: dict, *, width: int, height: int, fps: int) -> dict:
    """Allowlist data and bind it to program-owned copy, duration and geometry."""
    if not isinstance(raw, dict):
        raise ValueError("3D scene spec must be an object")
    if raw.get("id") != context["id"]:
        raise ValueError("3D spec refers to a different editorial shot")
    if not isinstance(context["id"], str) or not _ID.fullmatch(context["id"]):
        raise ValueError("3D scene ID must be a safe identifier")
    for size in (width, height):
        if type(size) is not int or not 320 <= size <= 3840 or size % 2:
            raise ValueError("3D render dimensions must be even, within 320..3840")
    if type(fps) is not int or not 24 <= fps <= 60:
        raise ValueError("3D fps must be an integer within 24..60")
    duration = _number(context["duration"], 4, 60, field="duration")
    frames = math.ceil(duration * fps)
    camera_raw = raw.get("camera")
    if not isinstance(camera_raw, dict):
        raise ValueError("3D camera is required")
    camera = {"position": _vector(camera_raw.get("position"), -30, 30, field="camera.position"),
              "target": _vector(camera_raw.get("target"), -12, 12, field="camera.target"),
              "fov": _number(camera_raw.get("fov", 40), 20, 75, field="camera.fov")}
    camera["end_position"] = _vector(camera_raw.get("end_position", camera["position"]), -30, 30,
                                     field="camera.end_position")
    camera["end_target"] = _vector(camera_raw.get("end_target", camera["target"]), -12, 12,
                                   field="camera.end_target")
    for position, target in ((camera["position"], camera["target"]),
                             (camera["end_position"], camera["end_target"])):
        if math.dist(position, target) < 2:
            raise ValueError("3D camera must stay at least two units from its target")
    separation = [a - b for a, b in zip(camera["position"], camera["target"])]
    delta = [c - d - a for c, d, a in zip(camera["end_position"], camera["end_target"], separation)]
    norm = sum(value * value for value in delta)
    closest = max(0, min(1, -sum(a * b for a, b in zip(separation, delta)) / norm)) if norm else 0
    if math.sqrt(sum((a + closest * b) ** 2 for a, b in zip(separation, delta))) < 2:
        raise ValueError("3D camera path crosses its target")
    actors_raw = raw.get("actors")
    if not isinstance(actors_raw, list) or not 1 <= len(actors_raw) <= 12:
        raise ValueError("3D scene requires 1..12 substantive actors")
    # Report every invalid surface thickness/extent in the same repair turn.
    # Stopping at the first actor conceals later violations and can exhaust the
    # bounded model attempts even when it fixes every field we reported.
    scales, scale_errors = [], []
    for actor_index, actor_raw in enumerate(actors_raw):
        if not isinstance(actor_raw, dict):
            raise ValueError("3D actor must be an object")
        field = f"actors[{actor_index}]({actor_raw.get('id')}).scale"
        try:
            scales.append(_vector(actor_raw.get("scale", [1, 1, 1]), .1, 4, field=field))
        except ValueError as exc:
            scale_errors.append(str(exc))
    if scale_errors:
        raise ValueError("; ".join(scale_errors))
    actors, ids = [], set()
    for actor_index, actor_raw in enumerate(actors_raw):
        if not isinstance(actor_raw, dict):
            raise ValueError("3D actor must be an object")
        actor_id, kind = actor_raw.get("id"), actor_raw.get("kind")
        if not isinstance(actor_id, str) or not _ID.fullmatch(actor_id) or actor_id in ids:
            raise ValueError("3D actor IDs must be unique safe identifiers")
        if kind not in _KINDS:
            raise ValueError("unsupported 3D actor kind")
        ids.add(actor_id)
        field = f"actors[{actor_index}]({actor_id})"
        actor = {"id": actor_id, "kind": kind,
                 "position": _vector(actor_raw.get("position"), -12, 12, field=f"{field}.position"),
                 "scale": scales[actor_index],
                 "rotation": _vector(actor_raw.get("rotation", [0, 0, 0]), -12.566, 12.566,
                                     field=f"{field}.rotation"),
                 "color": _color(actor_raw.get("color", raw.get("accent", "#59C4D8"))),
                 "explode": _number(actor_raw.get("explode", 0), 0, 1, field=f"{field}.explode"),
                 "motion_start": _number(actor_raw.get("motion_start", .12), 0, 1, field=f"{field}.motion_start"),
                 "motion_end": _number(actor_raw.get("motion_end", .72), 0, 1, field=f"{field}.motion_end")}
        actor["end_position"] = _vector(actor_raw.get("end_position", actor["position"]), -12, 12,
                                        field=f"{field}.end_position")
        actor["end_rotation"] = _vector(actor_raw.get("end_rotation", actor["rotation"]), -12.566, 12.566,
                                        field=f"{field}.end_rotation")
        actor["end_explode"] = _number(actor_raw.get("end_explode", actor["explode"]), 0, 1,
                                       field=f"{field}.end_explode")
        if kind not in {"chip", "rack", "satellite"} and (actor["explode"] or actor["end_explode"]):
            raise ValueError("exploded views are only supported for chip, rack and satellite actors")
        if actor["motion_start"] >= actor["motion_end"]:
            raise ValueError("3D actor motion_start must precede motion_end")
        action = actor_raw.get("action", "reveal")
        if action not in {"reveal", "translate", "rotate", "explode", "hold"}:
            raise ValueError("unsupported 3D actor action")
        actor["action"] = action
        if kind == "object":
            primitive = actor_raw.get("primitive", "box")
            if primitive not in {"box", "sphere", "cylinder", "torus"}:
                raise ValueError("unsupported 3D object primitive")
            actor["primitive"] = primitive
        elif actor_raw.get("primitive") is not None:
            raise ValueError(f"actor {actor_id} (kind={kind}): primitives are only permitted "
                             "for generic object actors; omit primitive for this fixed mesh "
                             "kind, or explicitly choose object if generic geometry was intended")
        if kind not in {"chip", "rack", "satellite"}:
            actor.pop("explode")
            actor.pop("end_explode")
        actors.append(actor)
    links_raw = raw.get("links", [])
    if not isinstance(links_raw, list) or len(links_raw) > 24:
        raise ValueError("3D links must be a list of at most 24 connections")
    relation = str(context.get("relationship_excerpt") or "")
    if links_raw and not relation:
        raise ValueError("3D links require the shot's grounded relationship excerpt")
    links = []
    for link in links_raw:
        if (not isinstance(link, dict) or link.get("from") not in ids or link.get("to") not in ids
                or link["from"] == link["to"]):
            raise ValueError("3D links need distinct known actor endpoints")
        particles = link.get("particles", 5)
        if type(particles) is not int or not 0 <= particles <= 12:
            raise ValueError("3D link particles must be an integer within 0..12")
        links.append({"from": link["from"], "to": link["to"], "particles": particles,
                      "color": _color(link.get("color", raw.get("accent", "#59C4D8")))})
    moving = (camera["position"] != camera["end_position"] or camera["target"] != camera["end_target"]
              or any(actor["action"] != "hold" and (
                  actor["position"] != actor["end_position"] or actor["rotation"] != actor["end_rotation"]
                  or actor.get("explode") != actor.get("end_explode")) for actor in actors)
              or any(link["particles"] for link in links))
    if not moving:
        raise ValueError("3D shot must demonstrate spatial motion, not only a static object or text reveal")
    excerpt = _text(raw.get("narration_excerpt"), "narration_excerpt")
    if excerpt not in context["narration_excerpt"]:
        raise ValueError("3D narration excerpt is not from the selected shot: narration_excerpt "
                         "must preserve the supplied shot's exact characters, including punctuation")
    if raw.get("representation") != "illustration":
        raise ValueError("generated 3D must be explicitly classified as illustration")
    return {"contract": CONTRACT, "id": context["id"], "width": width, "height": height,
            "fps": fps, "duration_frames": frames, "background": _color(raw.get("background", "#101820")),
            "accent": _color(raw.get("accent", "#59C4D8")), "camera": camera, "actors": actors,
            "links": links, "copy": list(context["copy"]), "narration_excerpt": excerpt,
            "semantic_intent": _text(raw.get("semantic_intent"), "semantic_intent"),
            "representation": "illustration"}


def select_shots(plans: list[dict], board: dict) -> tuple[list[dict], list[dict]]:
    """Select one substantive editorial interval per story, keeping all sources."""
    by_id = {plan["id"]: plan for plan in plans}
    selected, skipped = [], []
    for scene in board.get("scenes", []):
        if scene.get("program_segment_kind") in {"opening", "closing"}:
            continue
        plan = by_id.get(scene["id"], {})
        if plan.get("archetype") in {"intro", "outro"}:
            continue
        candidates = [(index, shot) for index, shot in enumerate(plan.get("media_shots", []))
                      if shot.get("kind") in {"editorial", "three_d"} and 4 <= float(shot["duration"]) <= 60]
        if not candidates:
            skipped.append({"scene_id": scene["id"], "reason": "No editorial interval of 4..60 seconds; source media retained"})
            continue
        index, shot = max(candidates, key=lambda row: (
            row[1].get("layout") in {"flow", "comparison"}, min(float(row[1]["duration"]), 15), -row[0],
        ))
        excerpt = str(shot.get("script_excerpt") or "")
        if not excerpt or excerpt not in scene["text"]:
            raise ValueError("3D candidate must retain its source-grounded narration excerpt")
        selected.append({"id": f"{scene['id']}-shot-{index + 1:02d}", "scene_id": scene["id"],
                         "shot_index": index, "duration": float(shot["duration"]),
                         "copy": list(shot["copy"]), "narration_excerpt": excerpt,
                         "scene_text": str(scene["text"]), "headline": str(plan.get("headline") or ""),
                         "relationship_excerpt": shot.get("relationship_excerpt", ""),
                         "layout": shot.get("layout"), "direction": {
                             key: value for key, value in shot.get("direction", {}).items() if key != "evidence_kind"
                         }, "director_brief": board.get("director_brief", {}),
                         "visual_review_feedback": plan.get("visual_review_feedback", {}),
                         "review_repair_source_sha256": plan.get("review_repair_source_sha256", "")})
    return selected, skipped


def renderer_identity(root: Path) -> dict:
    paths = [root / "package.json", root / "package-lock.json"]
    paths.extend(path for folder in ("src", "scripts") for path in (root / folder).rglob("*") if path.is_file())
    if not all(path.is_file() for path in paths[:2]) or not (root / "scripts/render.mjs").is_file():
        raise RuntimeError("Local Remotion renderer and pinned dependencies are not installed")
    evidence = {str(path.relative_to(root)): _file_hash(path) for path in sorted(set(paths))}
    return {"files": evidence, "sha256": _digest(evidence)}


async def _command(name: str, command: list, *, log=None, cwd=None, timeout=300) -> str:
    returncode, output = await stream_subprocess(name=name, command=command, logger=logger, log=log,
                                               cwd=cwd, timeout=timeout, stall_timeout=min(timeout, 300))
    if returncode != 0:
        raise RuntimeError(f"{name} failed (exit {returncode}): {output[-1500:]}")
    return output


def _preview_evidence(sidecar: dict, output: Path, expected_frames: list[int], width: int, height: int) -> list[dict]:
    stills = sidecar.get("stills")
    if not isinstance(stills, list) or sorted(row.get("frame") for row in stills) != sorted(expected_frames):
        raise ValueError("3D renderer did not capture every required preview frame")
    evidence = []
    for row in stills:
        path = Path(row["path"])
        if not path.is_absolute():
            path = output.parent / path
        path = path.resolve()
        if not path.is_relative_to(output.parent.resolve()) or not path.is_file():
            raise ValueError("3D preview is missing or outside its output directory")
        digest = _file_hash(path)
        if digest != row.get("sha256"):
            raise ValueError("3D preview bytes changed")
        with Image.open(path) as frame:
            frame.load()
            if frame.size != (width, height):
                raise ValueError("3D preview dimensions do not match delivery")
            if all(high - low <= 1 for low, high in frame.convert("RGB").getextrema()):
                raise ValueError("3D preview is uniformly blank")
        evidence.append({"frame": row["frame"], "path": str(path), "sha256": digest})
    return evidence


async def validate_render(output: Path, spec: dict, *, log=None) -> dict:
    """Inspect the delivered silent clip and real representative 3D pixels."""
    sidecar = _read(Path(str(output) + ".render.json"))
    if sidecar.get("contract") != RENDER_CONTRACT or sidecar.get("passed") is not True:
        raise ValueError("3D renderer did not provide successful local render evidence")
    spec_path = output.parent / "scene.json"
    if _digest(_read(spec_path)) != _digest(spec):
        raise ValueError("3D rendered scene differs from the currently requested spec")
    if sidecar.get("input_sha256") != await asyncio.to_thread(_file_hash, spec_path):
        raise ValueError("3D renderer evidence belongs to a different scene spec")
    digest = await asyncio.to_thread(_file_hash, output)
    if sidecar.get("output_sha256") != digest:
        raise ValueError("3D rendered clip bytes differ from renderer evidence")
    data = json.loads(await _command("3D clip probe", [
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(output),
    ], log=log, timeout=max(60, spec["duration_frames"] / spec["fps"] * 3)))
    videos = [row for row in data.get("streams", []) if row.get("codec_type") == "video"]
    audios = [row for row in data.get("streams", []) if row.get("codec_type") == "audio"]
    if len(videos) != 1 or audios:
        raise ValueError("3D clip must contain exactly one video stream and no audio")
    video = videos[0]
    numerator, denominator = str(video.get("avg_frame_rate", "0/1")).split("/")
    rate = float(numerator) / float(denominator)
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    if (video.get("width"), video.get("height")) != (spec["width"], spec["height"]):
        raise ValueError("3D clip has incorrect dimensions")
    if (abs(rate - spec["fps"]) > .001 or int(video.get("nb_read_frames", -1)) != spec["duration_frames"]
            or not math.isfinite(duration) or abs(duration - spec["duration_frames"] / spec["fps"]) > .002):
        raise ValueError("3D clip frame count, duration or fps violates the shared timeline")
    await _command("3D clip decode", ["ffmpeg", "-v", "error", "-xerror", "-i", str(output),
                                      "-map", "0:v:0", "-f", "null", "-"], log=log,
                   timeout=max(60, duration * 3))
    frame_count = spec["duration_frames"]
    requested = [0, frame_count // 2, frame_count - 1]
    previews = await asyncio.to_thread(_preview_evidence, sidecar, output, requested, spec["width"], spec["height"])
    return {"passed": True, "sha256": digest, "duration": duration, "frames": frame_count,
            "width": spec["width"], "height": spec["height"], "fps": spec["fps"], "stills": previews,
            "engine": sidecar.get("engine", {}), "render_seconds": sidecar.get("render_seconds"),
            "semantic_review": "Final encoded program review remains required"}


@timed("three_d")
async def render_story_shots(plans: list[dict], board: dict, task_dir: Path, *,
                             width: int = 1920, height: int = 1080, fps: int = 30, provider_id=None,
                             ai_endpoint=None, ai_model=None, log=None) -> dict:
    """Plan and render one real 3D insert per eligible story, without 2D fallback."""
    from backend.pipeline.digester import _chat, _resolve_provider
    from backend.pipeline.visual_plan import _first_json_array

    root = config.PROJECT_ROOT / "remotion"
    task_dir = Path(task_dir).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    report_path, plan_path = task_dir / "three_d_render_report.json", task_dir / "three_d_plan.json"
    contexts, skipped = select_shots(plans, board)
    report = {"contract": CONTRACT, "enabled": True, "passed": False, "required_shots": len(contexts),
              "rendered_count": 0,
              "skipped_scenes": skipped, "shots": [], "status": "planning"}
    _write(report_path, report)
    if not contexts:
        report.update(status="no_eligible_shots", error="No eligible editorial interval for requested real 3D")
        _write(report_path, report)
        raise RuntimeError(report["error"])
    try:
        identity = await asyncio.to_thread(renderer_identity, root)
        prompt = (config.PROMPTS_DIR / "three_d.txt").read_text(encoding="utf-8")
        provider = await _resolve_provider(provider_id, ai_endpoint, ai_model)
        requested_route = _route_receipt(provider[0], provider[1])
        route_identity = _digest({"provider_id": provider_id, "endpoint": provider[0], "model": provider[1]})
        saved = _read(plan_path)
        previous = {row["id"]: row for row in saved.get("shots", [])}
        selected, errors = {}, {}
        for context in contexts:
            context["input_sha256"] = _digest({"contract": CONTRACT, "prompt": prompt,
                "context": context, "width": width, "height": height, "fps": fps,
                "requested_route_sha256": route_identity})
            old = previous.get(context["id"], {})
            if old.get("input_sha256") == context["input_sha256"]:
                try:
                    spec = validate_spec(old["spec"], context, width=width, height=height, fps=fps)
                    selected[context["id"]] = {"id": context["id"], "input_sha256": context["input_sha256"],
                        "spec": spec, "planner_model": old.get("planner_model"), "planner": "cache",
                        "planner_route": old.get("planner_route", {}), "requested_route": requested_route}
                    if log:
                        log(f"3D {context['id']}: reused validated configured-provider direction")
                except (KeyError, TypeError, ValueError):
                    pass

        def save_plans():
            _write(plan_path, {"contract": CONTRACT, "renderer": identity, "skipped_scenes": skipped,
                              "shots": [selected[row["id"]] for row in contexts if row["id"] in selected]})

        rejected = {}
        # Resume the exact rejected input, not a fresh creative request. The
        # context hash includes narration, review feedback/candidate, route and
        # delivery settings. Recompute the error with the current validator.
        unresolved = {row["id"]: row for row in contexts if row["id"] not in selected}
        for path in sorted((task_dir / "three_d_planning").glob("*.json"),
                           key=lambda item: item.stat().st_mtime_ns, reverse=True):
            if len(rejected) == len(unresolved):
                break
            evidence = _read(path)
            if evidence.get("status") != "validated":
                continue
            prior_request, prior_errors = evidence.get("request"), evidence.get("validation_errors")
            if (not isinstance(prior_request, dict) or not isinstance(prior_request.get("shots"), list)
                    or not isinstance(prior_errors, dict) or not isinstance(evidence.get("response"), str)):
                continue
            prior_contexts = {row.get("id"): row for row in prior_request["shots"]
                              if isinstance(row, dict) and isinstance(row.get("id"), str)}
            matched = [row for scene_id, row in unresolved.items()
                       if scene_id not in rejected and scene_id in prior_errors
                       and prior_contexts.get(scene_id, {}).get("input_sha256") == row["input_sha256"]]
            if not matched:
                continue
            prior_raw = _first_json_array(evidence.get("response", "")) or []
            indexed = {row["id"]: row for row in prior_raw
                       if isinstance(row, dict) and isinstance(row.get("id"), str)}
            for context in matched:
                raw_scene = indexed.get(context["id"])
                try:
                    validate_spec(raw_scene, context, width=width, height=height, fps=fps)
                except (TypeError, ValueError) as exc:
                    errors[context["id"]] = str(exc)
                    rejected[context["id"]] = _rejection(context["id"], raw_scene, exc)
                    if log:
                        log(f"3D {context['id']}: resuming rejected direction from {path.name}: {exc}")
        previous_attempt_rejections, exhausted = {}, set()
        for attempt in range(MAX_PLAN_ATTEMPTS):
            pending = [row for row in contexts if row["id"] not in selected and row["id"] not in exhausted]
            if not pending:
                break
            for offset in range(0, len(pending), MAX_BATCH):
                batch = pending[offset:offset + MAX_BATCH]
                actual_route = {}

                def remember_route(endpoint, model, _api_key):
                    actual_route.update(_route_receipt(endpoint, model))

                batch_ids = {row["id"] for row in batch}
                request = {"shots": batch, "actor_contract": _ACTOR_CONTRACT,
                           "text_contract": _TEXT_CONTRACT,
                           "repair_contract": _REPAIR_CONTRACT,
                           "validation_feedback": {key: value for key, value in errors.items() if key in batch_ids},
                           "rejected_scenes": [value for key, value in rejected.items() if key in batch_ids]}
                evidence_path = task_dir / "three_d_planning" / f"{uuid4().hex}.json"
                evidence = {"status": "requested", "attempt": attempt + 1,
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "requested_route": requested_route, "request": request}
                _write(evidence_path, evidence)
                answer = await _chat(prompt, json.dumps(request, ensure_ascii=False),
                    *provider, log, f"3D director {offset // MAX_BATCH + 1} ({attempt + 1}/{MAX_PLAN_ATTEMPTS})",
                    max_tokens=16384, enable_skills=False, disable_thinking=True, route_selected=remember_route)
                evidence.update(status="responded", response=answer, planner_route=actual_route.copy())
                _write(evidence_path, evidence)
                raw = _first_json_array(answer) or []
                indexed = {row["id"]: row for row in raw
                           if isinstance(row, dict) and isinstance(row.get("id"), str)}
                for context in batch:
                    try:
                        spec = validate_spec(indexed.get(context["id"]), context,
                                             width=width, height=height, fps=fps)
                        selected[context["id"]] = {"id": context["id"], "input_sha256": context["input_sha256"],
                            "spec": spec, "planner_model": actual_route.get("model"),
                            "planner_route": actual_route.copy(), "requested_route": requested_route,
                            "planner": "configured_provider"}
                        errors.pop(context["id"], None)
                        rejected.pop(context["id"], None)
                        save_plans()
                    except (TypeError, ValueError) as exc:
                        errors[context["id"]] = str(exc)
                        failure = _rejection(context["id"], indexed.get(context["id"]), exc)
                        if previous_attempt_rejections.get(context["id"]) == failure:
                            exhausted.add(context["id"])
                        previous_attempt_rejections[context["id"]] = failure
                        rejected[context["id"]] = failure
                evidence.update(status="validated", accepted_ids=sorted(batch_ids & selected.keys()),
                                validation_errors={key: value for key, value in errors.items() if key in batch_ids},
                                unchanged_rejected_ids=sorted(batch_ids & exhausted))
                _write(evidence_path, evidence)
        missing = [row["id"] for row in contexts if row["id"] not in selected]
        if missing:
            raise RuntimeError("Configured provider did not return valid 3D direction: "
                               + "; ".join(f"{key}: {errors.get(key, 'missing')}" for key in missing))
        save_plans()
        report.update(status="rendering", renderer_sha256=identity["sha256"])
        by_id = {plan["id"]: plan for plan in plans}
        for context in contexts:
            spec = selected[context["id"]]["spec"]
            render_hash = _digest({"contract": CONTRACT, "spec": spec, "renderer": identity["sha256"]})
            folder = task_dir / "three_d" / render_hash[:24]
            folder.mkdir(parents=True, exist_ok=True)
            output, spec_path = folder / "clip.mp4", folder / "scene.json"
            receipt = _read(folder / "validated.json")
            cached = (receipt.get("render_sha256") == render_hash and output.is_file()
                      and await asyncio.to_thread(_file_hash, output) == receipt.get("sha256"))
            _write(spec_path, spec)
            if cached:
                try:
                    evidence = await validate_render(output, spec, log=log)
                except (OSError, TypeError, ValueError, RuntimeError):
                    cached = False
            if not cached:
                count = spec["duration_frames"]
                command = ["node", str(root / "scripts/render.mjs"), "--spec", str(spec_path),
                           "--output", str(output), "--stills", f"0,{count // 2},{count - 1}"]
                browser = os.getenv("REMOTION_BROWSER_EXECUTABLE")
                if browser:
                    command += ["--browser", browser]
                await _command("Remotion R3F render", command, log=log, cwd=root,
                               timeout=max(600, min(7200, count * 3)))
                evidence = await validate_render(output, spec, log=log)
                _write(folder / "validated.json", {"render_sha256": render_hash, **evidence})
            shot = by_id[context["scene_id"]]["media_shots"][context["shot_index"]]
            shot.update(kind="three_d", src="../" + output.relative_to(task_dir).as_posix(),
                        sha256=evidence["sha256"], source_duration=evidence["duration"], source_start=0,
                        three_d_spec=spec, three_d_spec_path="../" + spec_path.relative_to(task_dir).as_posix(),
                        three_d_spec_sha256=_digest(spec), three_d_renderer_sha256=identity["sha256"],
                        three_d_render_sha256=render_hash, three_d_representation="illustration")
            span = (spec["duration_frames"] - 1) / spec["fps"]
            onset = min(actor["motion_start"] for actor in spec["actors"]) * span
            settled = max(actor["motion_end"] for actor in spec["actors"]) * span
            shot["three_d_motion_cues"] = {"motion_start": round(onset, 6),
                "action": round((onset + settled) / 2, 6), "settle": round(settled, 6)}
            shot["direction"] = {**shot.get("direction", {}), "evidence_kind": "illustration"}
            row = {"id": context["id"], "scene_id": context["scene_id"], "shot_index": context["shot_index"],
                   "render_sha256": render_hash, "path": str(output), "spec_path": str(spec_path),
                   "cached": cached, **evidence}
            report["shots"].append(row)
            report["rendered_count"] = len(report["shots"])
            _write(report_path, report)
            if log:
                log(f"3D {context['id']}: real Remotion/R3F clip validated ({evidence['frames']} frames; cache={cached})")
        report.update(passed=True, status="ready")
        _write(report_path, report)
        return report
    except asyncio.CancelledError:
        report.update(status="interrupted", error="3D generation cancelled; completed validated clips retained")
        _write(report_path, report)
        raise
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        _write(report_path, report)
        raise
