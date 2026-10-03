"""Director intent and delivery evidence derived from the production timeline.

The existing audio/HyperFrames centisecond timeline remains authoritative. Frame
indices below are a delivery projection of it, never a second editable schedule.
No model credentials, browser sessions or Codex runtime are needed here.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

CONTRACT = "directed-explainer-v1"
THREE_D_CONTRACT = "three-d-explainer-v1"


def three_d_review_context(shot: dict) -> dict:
    """Expose bounded declarative geometry, never classify generated pixels as footage."""
    spec = shot.get("three_d_spec")
    if (
        not isinstance(spec, dict) or spec.get("contract") != THREE_D_CONTRACT
        or spec.get("representation") != "illustration"
        or not isinstance(spec.get("camera"), dict)
        or not isinstance(spec.get("actors"), list) or not 1 <= len(spec["actors"]) <= 12
        or any(not isinstance(actor, dict) for actor in spec["actors"])
    ):
        raise ValueError("three_d shot lacks a valid procedural illustration spec")
    actor_keys = (
        "id", "kind", "primitive", "position", "end_position", "rotation", "end_rotation",
        "scale", "color", "explode", "end_explode", "motion_start", "motion_end", "action",
    )
    context = {
        "representation": "illustration",
        "provenance": "Procedural Three.js geometry rendered with React Three Fiber and Remotion",
        "contract": spec["contract"],
        "spec_sha256": shot.get("three_d_spec_sha256"),
        "renderer_sha256": shot.get("three_d_renderer_sha256"),
        "semantic_intent": spec.get("semantic_intent", ""),
        "camera": {key: spec["camera"][key] for key in (
            "position", "target", "end_position", "end_target", "fov",
        ) if key in spec["camera"]},
        "actors": [{key: actor[key] for key in actor_keys if key in actor} for actor in spec["actors"]],
        "links": spec.get("links", []),
    }
    # A copy avoids tying the immutable timeline to later mutations of a plan.
    encoded = json.dumps(context, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if len(encoded) > 18000:
        raise ValueError("three_d review context exceeds the bounded scene specification")
    return json.loads(encoded)


def three_d_sample_times(shot: dict) -> list[tuple[str, float]]:
    """Follow Remotion's frame-driven actor/camera phases, not overlay reveal cues."""
    three_d_review_context(shot)
    spec = shot["three_d_spec"]
    fps, count = spec.get("fps"), spec.get("duration_frames")
    if type(fps) is not int or fps <= 0 or type(count) is not int or count < 2:
        raise ValueError("three_d sampling requires a finite frame timeline")
    duration = float(shot["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("three_d shot duration must be finite and positive")
    span = (count - 1) / fps
    if span >= duration + .001 or count / fps < duration - .001:
        raise ValueError("three_d clip frame timeline differs from its shot duration")
    windows = []
    for actor in spec["actors"]:
        start, end = actor.get("motion_start", .12), actor.get("motion_end", .72)
        if (
            type(start) not in (float, int) or type(end) not in (float, int)
            or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= 1
        ):
            raise ValueError("three_d actor motion window is invalid")
        windows.append((start, end))
    # sceneState() holds its camera and connector flow at the latest actor's
    # motion_end, including hold actors; the render contract uses this same rule.
    first = min(start for start, _ in windows)
    last = max(end for _, end in windows)
    onset = min(.25, span * .05)
    action = span * (first + last) / 2
    settled = min(span, span * last + 1 / fps)
    declared = shot.get("three_d_motion_cues")
    if declared is not None:
        expected = {"motion_start": span * first, "action": action, "settle": span * last}
        if not isinstance(declared, dict) or any(
            type(declared.get(key)) not in (int, float) or not math.isfinite(declared[key])
            or abs(declared[key] - value) > .001 for key, value in expected.items()
        ):
            raise ValueError("three_d motion cues disagree with the rendered frame timeline")
    # Remotion may emit an IEEE-ceil tail at exactly the shot end; that instant
    # belongs to the next shot in the assembly, so select the preceding frame.
    limit = min(span, max(0, duration - 1 / fps)) if span >= duration else span
    return [(phase, round(min(value, limit), 6)) for phase, value in (
        ("onset", onset), ("action", action), ("settled", settled),
    )]


def director_brief(board: dict, *, theme: str, frame, render, captions: bool,
                   music_enabled: bool) -> dict:
    """Make the established task constraints explicit to every planning stage."""
    return {
        "contract": CONTRACT,
        "audience": "Viewers of a narrated technology explainer or news briefing",
        "title": board.get("title", ""),
        "takeaway": board.get("thesis") or board.get("title", ""),
        "narrative_method": "Establish the subject, show the supported change or relationship, then its consequence and limits",
        "source_of_truth": "Approved narration, supplied source excerpts and verified media inventory",
        "visual_language": {
            "theme": theme,
            "hierarchy": "One main subject and one explanatory action at a time",
            "continuity": "Keep the subject identifiable across cuts and scale changes",
            "motion": "Establish, reveal or compare, then hold for reading; no decorative perpetual motion",
            "representation": "Use captured evidence for real events and grounded diagrams for relationships; distinguish illustration from measurement",
        },
        "delivery": {"width": render.width, "height": render.height, "fps": render.fps,
                     "orientation": frame.orientation, "captions": captions,
                     "narration": "existing verified audio", "music_enabled": music_enabled},
        "timing": "Existing audio-derived seconds, zero based half-open intervals; motion cues are local to each shot",
        "sound": "Narration leads. Existing paced music/mix remains authoritative; do not invent independent sound or beat timings",
        "prohibitions": ["No invented facts, causal links, measurements or source footage",
                         "No changes to narration or measured asset in/out points",
                         "No repeated full-screen title cards as filler"],
        "acceptance": ["Every content shot has a purpose and grounded copy",
                       "Representative action and settled frames inspected before full export",
                       "Final encoded timeline and complete-shot semantic/A/V review pass"],
    }


def _number(value) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("timeline time cannot be boolean")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("timeline time must be finite and nonnegative")
    return result


def build_timeline(board: dict, plans: list[dict], *, fps: int) -> dict:
    """Export ordered picture ownership and named events from the actual plans."""
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("delivery fps must be a positive integer")
    total = _number(board["total_duration"])
    # HyperFrames' render probe uses Math.ceil(duration * fps), including a
    # partial final frame. Mirror it so fractional audio durations stay intact.
    frames = max(1, math.ceil(float(total) * fps))
    by_id = {p["id"]: p for p in plans}
    shots, events = [], []
    cursor = Decimal(0)
    for scene_index, scene in enumerate(board["scenes"]):
        start, duration = _number(scene["start"]), _number(scene["duration"])
        if abs(start - cursor) > Decimal("0.001") or duration <= 0:
            raise ValueError(f"scene timeline gap/overlap at {scene['id']}")
        plan = by_id[scene["id"]]
        scene_shots = plan.get("media_shots") or [{"start": 0, "duration": float(duration),
                                                   "kind": plan.get("archetype", "editorial")}]
        local_cursor = Decimal(0)
        for index, shot in enumerate(scene_shots, 1):
            local, length = _number(shot["start"]), _number(shot["duration"])
            if abs(local - local_cursor) > Decimal("0.001") or length <= 0:
                raise ValueError(f"shot timeline gap/overlap in {scene['id']}")
            absolute, end = start + local, start + local + length
            first = min(frames, int((absolute * fps).to_integral_value(rounding=ROUND_CEILING)))
            last = min(frames, int((end * fps).to_integral_value(rounding=ROUND_CEILING)))
            # JS/Python float multiplication can add a tiny positive residue
            # at an exact decimal ending (8.3 * 30). The renderer still emits
            # that final frame; include it in the final shot's ownership.
            if scene_index == len(board["scenes"]) - 1 and index == len(scene_shots):
                last = frames
            if last <= first:
                raise ValueError(f"shot has no delivery frames in {scene['id']}")
            shot_id = f"{scene['id']}-shot-{index:02d}"
            shots.append({"id": shot_id, "scene_id": scene["id"], "start": first, "end": last,
                          "start_seconds": float(absolute), "end_seconds": float(end),
                          "kind": shot["kind"], "direction": shot.get("direction", {}),
                          "source": shot.get("src"), "source_sha256": shot.get("sha256"),
                          "source_start": shot.get("source_start"),
                          "copy": shot.get("copy", []), "layout": shot.get("layout")})
            if shot["kind"] == "three_d":
                shots[-1]["three_d"] = three_d_review_context(shot)
            for name, offset in shot.get("motion_cues", {}).items():
                cue = _number(offset)
                if cue >= length:
                    raise ValueError(f"motion event {name} outside {shot_id}")
                event_time = absolute + cue
                events.append({"id": f"{shot_id}-{name}", "shot_id": shot_id, "name": name,
                               "time_seconds": float(event_time),
                               "frame": min(last - 1, int((event_time * fps).to_integral_value(rounding=ROUND_CEILING)))})
            if shot["kind"] == "three_d":
                for phase, offset in three_d_sample_times(shot):
                    event_time = absolute + _number(offset)
                    name = f"three_d_{phase}"
                    events.append({"id": f"{shot_id}-{name}", "shot_id": shot_id, "name": name,
                                   "time_seconds": float(event_time),
                                   "frame": min(last - 1, int((event_time * fps).to_integral_value(rounding=ROUND_CEILING)))})
            local_cursor = local + length
        if abs(local_cursor - duration) > Decimal("0.001"):
            raise ValueError(f"incomplete shot coverage in {scene['id']}")
        cursor = start + duration
    if not shots or abs(cursor - total) > Decimal("0.001"):
        raise ValueError("scenes do not cover the complete program")
    for previous, following in zip(shots, shots[1:]):
        if previous["end"] != following["start"]:
            raise ValueError("projected frame timeline is not contiguous")
    if shots[0]["start"] != 0 or shots[-1]["end"] != frames:
        raise ValueError("frame timeline does not cover the delivery")
    payload = {"contract": CONTRACT, "fps": fps, "duration_frames": frames,
               "duration_seconds": float(total), "timing_authority": "storyboard.json + media_shots.json",
               "projection": "ceil(seconds * fps), clipped to rendered frame count; half-open intervals",
               "shots": shots, "events": events,
               "three_d_shot_count": sum(shot["kind"] == "three_d" for shot in shots)}
    payload["input_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def preview_times(timeline: dict) -> list[float]:
    """Inspect every shot's settled composition plus purposeful action moments."""
    fps = timeline["fps"]
    selected = {(s["start"] + s["end"] - 1) // 2 for s in timeline["shots"]}
    editorial = {s["id"] for s in timeline["shots"] if s["kind"] == "editorial"}
    procedural = {s["id"] for s in timeline["shots"] if s["kind"] == "three_d"}
    selected.update(e["frame"] for e in timeline["events"]
                    if (e["shot_id"] in editorial and e["name"] in {"emphasis", "settle"})
                    or (e["shot_id"] in procedural and e["name"] in {
                        "three_d_onset", "three_d_action", "three_d_settled",
                    }))
    return [round(frame / fps, 6) for frame in sorted(selected)]


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def verify_delivery(video: Path, timeline: dict, report_path: Path) -> dict:
    """Check decoded frame count and EVERY presentation time, not just avg fps."""
    failures = []
    count, maximum_error = 0, 0.0
    origin, audio_start, origin_tolerance = None, None, None
    try:
        metadata = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,codec_name,sample_rate,start_time", "-of", "json", str(video),
        ], capture_output=True, text=True, timeout=30, check=True)
        audio_streams = [s for s in json.loads(metadata.stdout).get("streams", [])
                         if s.get("codec_type") == "audio"]
        if len(audio_streams) != 1:
            raise ValueError("expected one audio stream for timeline offset check")
        audio = audio_streams[0]
        audio_start = float(audio.get("start_time", 0))
        sample_rate = int(audio.get("sample_rate", 0))
        if not math.isfinite(audio_start) or sample_rate <= 0:
            raise ValueError("invalid audio timestamp/sample rate")
        # AAC muxing can shift picture by one encoder priming packet. Bound
        # that offset separately; cadence is measured relative to first video
        # PTS, as the shared-timeline checker requires.
        priming = 1024 / sample_rate if audio.get("codec_name") == "aac" else 0
        origin_tolerance = max(1 / timeline["fps"], priming) + 0.0001
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_streams",
            "-show_entries", "frame=best_effort_timestamp_time:stream=time_base", "-of", "json", str(video),
        ], capture_output=True, text=True, timeout=max(60, min(900, timeline["duration_seconds"] * 2 + 30)), check=True)
        parsed = json.loads(result.stdout)
        decoded = parsed.get("frames", [])
        count = len(decoded)
        if count != timeline["duration_frames"]:
            failures.append(f"decoded {count} frames, expected {timeline['duration_frames']}")
        numerator, denominator = parsed["streams"][0]["time_base"].split("/")
        tick = float(numerator) / float(denominator)
        if not math.isfinite(tick) or tick <= 0:
            raise ValueError("invalid stream time base")
        tolerance = tick + 0.000002
        if not decoded:
            raise ValueError("no decoded frame timestamps")
        origin = float(decoded[0]["best_effort_timestamp_time"])
        if not math.isfinite(origin) or abs(origin - audio_start) > origin_tolerance or abs(audio_start) > origin_tolerance:
            failures.append("video/audio start offset exceeds one frame or encoder priming packet")
        for index, frame in enumerate(decoded):
            timestamp = float(frame["best_effort_timestamp_time"])
            error = abs(timestamp - origin - index / timeline["fps"])
            if not math.isfinite(timestamp):
                raise ValueError("nonfinite presentation timestamp")
            maximum_error = max(maximum_error, error)
        if maximum_error > tolerance:
            failures.append(f"presentation cadence drift {maximum_error:.6f}s exceeds {tolerance:.6f}s")
    except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.SubprocessError) as exc:
        failures.append(f"decoded timeline unavailable: {exc}")
    with video.open("rb") as handle:
        video_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {"contract": CONTRACT, "passed": not failures, "failures": failures,
              "video_sha256": video_hash, "timeline_sha256": timeline["input_sha256"],
              "decoded_frames": count, "maximum_cadence_error_seconds": maximum_error,
              "first_video_pts": origin, "first_audio_pts": audio_start,
              "initial_offset_tolerance_seconds": origin_tolerance,
              "visual_quality": "separate rendered-frame review required",
              "audio_listening": "not established by timeline measurements"}
    write_json(report_path, report)
    return report
