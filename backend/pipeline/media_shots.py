"""AI-edited intra-story shots with measured assets and a complete timeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import subprocess
from decimal import Decimal, ROUND_FLOOR
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

from backend import config
from backend.pipeline.image_layout import image_layout
from backend.pipeline.timing import timed

CONTRACT_VERSION = 3
MAX_CONCURRENT_BATCHES = 2
_RELATIONSHIP_MARKERS = {
    "flow": re.compile(
        r"\b(?:then|before|after|first|next|finally|because|therefore|leads? to|results? in|"
        r"causes?|produces?|converts?|transforms?)\b|首先|然后|随后|最后|导致|因此|转化|转换", re.I),
    "comparison": re.compile(
        r"\b(?:versus|vs\.?|compared (?:with|to)|whereas|while|unlike|rather than|more than|"
        r"less than|higher than|lower than|from .+ to)\b|相比|对比|而非|高于|低于|从.+到", re.I),
}


def motion_cues(duration: float) -> dict[str, float]:
    """One shot-local, centisecond schedule shared by motion and review.

    Keep the existing audio-owned centisecond convention; do not round each
    consumer independently or retime narration to fit a visual beat.
    """
    total = ticks(duration)
    if total < 20:
        raise ValueError("shot shorter than 0.20 seconds")
    reveal = max(1, min(10, total // 12))
    revealed = max(reveal, min(total * 28 // 100, reveal + 55))
    emphasis = max(revealed, min(total * 42 // 100, revealed + 65))
    settled = max(emphasis, min(total * 55 // 100, emphasis + 45))
    return {"reveal_start": reveal / 100, "reveal_end": revealed / 100,
            "emphasis": emphasis / 100, "settle": settled / 100}


def _reading_units(text: str) -> int:
    # CJK glyphs carry more information than Latin letters; count the unspaced
    # characters individually so a Chinese paragraph cannot count as one word.
    return len(re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:['’.-][A-Za-z0-9]+)*", text))


def _reading_budget(duration: float) -> int:
    return max(0, int(float(duration) * 3))


def _copy_identity(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def _default_direction(kind: str, copy: list[str]) -> dict[str, str]:
    evidence = ("illustration" if kind == "paper_collage" else
                "editorial" if kind == "editorial" else "captured_evidence")
    subject = next((value for value in copy if value), "Narrated subject")[:400]
    return {"purpose": "Explain the supplied narration with its selected copy",
            "subject": subject, "action": "Reveal the selected information, then hold for reading",
            "continuity": subject, "evidence_kind": evidence, "motion": "reveal"}


def _direction(item: dict, kind: str, copy: list[str], *, required: bool) -> dict:
    defaults = _default_direction(kind, copy)
    supplied = item.get("direction")
    if supplied is None and not required:
        return defaults
    if not isinstance(supplied, dict):
        raise ValueError("each shot needs a direction object")
    result = {}
    for key in ("purpose", "subject", "action", "continuity"):
        value = supplied.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 400:
            raise ValueError(f"direction {key} must be concise nonempty text")
        result[key] = value.strip()
    if supplied.get("evidence_kind") != defaults["evidence_kind"]:
        raise ValueError("direction evidence_kind conflicts with the actual asset")
    result["evidence_kind"] = defaults["evidence_kind"]
    motion = supplied.get("motion")
    if motion not in {"reveal", "compare", "trace"}:
        raise ValueError("unknown direction motion")
    result["motion"] = motion
    return result


class _Elements(HTMLParser):
    def __init__(self, source: str):
        super().__init__()
        self.elements = {}
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            if attrs["id"] in self.elements:
                raise ValueError(f"duplicate rendered id: {attrs['id']}")
            self.elements[attrs["id"]] = (tag, attrs)


def assert_rendered_shots(plan: dict, source: str) -> None:
    """Audit actual emitted media intervals, not a source mentioned in a comment."""
    if not plan.get("media_shots"):
        return
    elements = _Elements(source).elements
    cursor = 0
    for i, shot in enumerate(plan["media_shots"]):
        if ticks(shot["start"]) != cursor:
            raise ValueError(f"shot gap or overlap in {plan['id']}")
        cursor += ticks(shot["duration"])
        prefix = f"{plan['id']}-shot-{i + 1}"
        expected = [(prefix + "-panel", "div")]
        if shot["kind"] != "editorial":
            expected.append((prefix + "-media", "video" if shot["kind"] in {"public_footage", "paper_collage"} else "img"))
        for element_id, tag in expected:
            actual_tag, attrs = elements.get(element_id, (None, {}))
            if actual_tag != tag or "clip" not in attrs.get("class", "").split():
                raise ValueError(f"missing rendered shot: {element_id}")
            if ticks(attrs.get("data-start")) != ticks(shot["start"]) or ticks(attrs.get("data-duration")) != ticks(shot["duration"]):
                raise ValueError(f"rendered shot timing changed: {element_id}")
            if tag in {"video", "img"} and (attrs.get("src") != shot["src"] or "loop" in attrs):
                raise ValueError(f"rendered source changed or looped: {element_id}")
            if tag == "img" and shot.get("fit") == "contain" and shot["kind"] == "image":
                style = dict(
                    (key.strip(), value.strip())
                    for declaration in attrs.get("style", "").split(";") if ":" in declaration
                    for key, value in [declaration.split(":", 1)]
                )
                if style.get("object-fit") != "contain":
                    raise ValueError(f"rendered image containment changed: {element_id}")
            if tag == "video":
                if ticks(attrs.get("data-media-start", 0)) != ticks(shot.get("source_start", 0)):
                    raise ValueError(f"rendered source in-point changed: {element_id}")
                if ticks(shot.get("source_start", 0)) + ticks(shot["duration"]) > ticks(shot["source_duration"]):
                    raise ValueError(f"rendered shot exceeds source: {element_id}")


def ticks(value) -> int:
    if isinstance(value, bool):
        raise ValueError("time must be a number, not a boolean")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("time must be finite and nonnegative")
    return int((Decimal(str(value)) * 100).to_integral_value(rounding=ROUND_FLOOR))


def editorial_copy(plan: dict, scene: dict) -> list[dict]:
    """Address existing, grounded copy by id; the editor cannot add new facts."""
    values = [plan.get("headline"), plan.get("body"), *plan.get("items", [])]
    values += [plan.get("quote"), plan.get("attribution")]
    if plan.get("stat"):
        values.append(f"{plan['stat']} {plan.get('stat_label') or ''}".strip())
    for side in ("left", "right"):
        panel = plan.get(side) or {}
        values.append(" — ".join(str(panel.get(k) or "") for k in ("label", "text")).strip(" —"))
    narration = str(scene.get("text") or "")
    boundaries = [0]
    for match in re.finditer(r"(?<=[.!?])\s+", narration):
        # A middle initial or title is not the end of a complete display unit.
        if re.search(r"\b(?:(?:[A-Z]\.)+|(?:Mr|Mrs|Ms|Dr|Prof|Gen|Lt|Col|Sr|Jr|St|vs|etc|e\.g|i\.e)\.)$", narration[:match.start()]):
            continue
        boundaries.append(match.end())
    boundaries.append(len(narration))
    values += [narration[start:end].strip() for start, end in zip(boundaries, boundaries[1:])]
    unique = []
    identities = set()
    for value in values:
        text = str(value or "").strip()
        identity = _copy_identity(text)
        if text and identity not in identities:
            unique.append(text)
            identities.add(identity)
    return [{"id": f"copy-{i + 1}", "text": text} for i, text in enumerate(unique)]


def _asset(task_dir: Path, raw: str, kind: str, **metadata) -> dict:
    root = task_dir.resolve()
    path = (task_dir / "compositions" / raw).resolve() if raw.startswith("../") else (task_dir / raw).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or out-of-project {kind} asset: {raw}")
    result = {"src": "../" + path.relative_to(root).as_posix(), "kind": kind, **metadata}
    if kind in {"public_footage", "paper_collage"}:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=duration:format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, check=True, timeout=30,
        )
        data = json.loads(probe.stdout)
        streams = data.get("streams") or []
        if not streams:
            raise ValueError(f"No video stream: {raw}")
        # Prefer the video stream, not a longer audio/container tail.
        duration = streams[0].get("duration")
        if duration in (None, "N/A"):
            duration = (data.get("format") or {}).get("duration")
        result["duration"] = ticks(duration) / 100
        if result["duration"] <= 0:
            raise ValueError(f"Empty video: {raw}")
    else:
        with Image.open(path) as image:
            image.verify()
        result.update(image_layout(path, fit=metadata.get("fit", "cover")))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result["sha256"] = digest.hexdigest()
    return result


def inventory(plan: dict, task_dir: Path) -> list[dict]:
    assets = []
    public = plan.get("footage_sequence") or []
    if not public and plan.get("footage_src") and (not plan.get("collage_broll") or (
        plan.get("collage_src") and plan["footage_src"] != plan["collage_src"]
    )):
        public = [{"src": plan["footage_src"], "credit": plan.get("footage_credit", "")}]
    for item in public:
        assets.append(_asset(task_dir, item["src"], "public_footage",
                             credit=item.get("credit", ""), script_excerpt=item.get("script_excerpt", "")))
    collage_src = plan.get("collage_src") or (plan.get("footage_src") if plan.get("collage_broll") else "")
    if collage_src:
        assets.append(_asset(task_dir, collage_src, "paper_collage", credit="Illustration · Paper-Collage"))
    images = plan.get("news_image_srcs") or ([plan["news_image_src"]] if plan.get("news_image_src") else [])
    credits = plan.get("news_image_credits") or []
    fits = plan.get("news_image_fits") or []
    for i, src in enumerate(images):
        assets.append(_asset(
            task_dir, src, "image",
            credit=credits[i] if i < len(credits) else plan.get("news_image_credit", ""),
            fit=fits[i] if i < len(fits) else plan.get("news_image_fit", "cover"),
        ))
    if plan.get("news_webpage_src"):
        assets.append(_asset(task_dir, plan["news_webpage_src"], "article", credit=plan.get("news_webpage_source", "")))
    rejected = {(item["src"], item["sha256"]) for item in plan.get("review_rejected_media", [])}
    assets = [asset for asset in assets if (asset["src"], asset["sha256"]) not in rejected]
    return [{"id": f"asset-{i + 1}", **asset} for i, asset in enumerate(assets)]


def review_rejected_sources(plan: dict, task_dir: Path) -> dict:
    """Exempt only the exact source bytes rejected by a rendered-frame review."""
    rejected = {}
    for item in plan.get("review_rejected_media", []):
        path = (task_dir / "compositions" / item["src"]).resolve()
        if not path.is_relative_to(task_dir.resolve()) or not path.is_file():
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest == item["sha256"]:
            rejected[item["src"]] = item
    return rejected


def validate(raw: dict, scene: dict, assets: list[dict], copy: list[dict], *,
             require_direction: bool = False) -> list[dict]:
    allowed = {asset["id"]: asset for asset in assets}
    words = {item["id"]: item["text"] for item in copy}
    seen = set()
    shots = []
    cursor = 0
    for item in raw.get("shots") or []:
        aid = item.get("asset_id")
        if aid != "editorial" and aid not in allowed:
            raise ValueError(f"unknown asset {aid}")
        duration = ticks(item.get("duration"))
        if duration < 20:
            raise ValueError("shot shorter than 0.20 seconds")
        asset = allowed.get(aid, {})
        if item.get("treatment") not in {None, "play", "hold"}:
            raise ValueError("unknown shot treatment")
        hold = item.get("treatment") == "hold"
        source_start = ticks(item.get("source_start", 0))
        if hold:
            if duration > 200 or not shots or shots[-1]["asset_id"] != aid or shots[-1]["kind"] not in {"public_footage", "paper_collage"}:
                raise ValueError("hold must follow its video and last at most 2 seconds")
            source_start = max(0, ticks(shots[-1]["source_start"]) + ticks(shots[-1]["duration"]) - 5)
        elif aid != "editorial":
            if aid in seen:
                raise ValueError(f"asset repeated: {aid}")
            seen.add(aid)
        if not hold and "duration" in asset and source_start + duration > ticks(asset["duration"]):
            raise ValueError(f"shot exceeds measured duration: {aid}")
        excerpt = str(item.get("script_excerpt") or "").strip()
        if not excerpt or excerpt not in scene["text"]:
            raise ValueError("shot excerpt is not from its narration")
        ids = item.get("copy_ids") or []
        if not isinstance(ids, list) or not ids or len(ids) > 5 or any(key not in words for key in ids):
            raise ValueError("each shot needs 1-5 supplied copy ids")
        selected_copy = [words[key] for key in ids]
        comparable_copy = [_copy_identity(text) for text in selected_copy]
        if len(comparable_copy) != len(set(comparable_copy)):
            raise ValueError("shot repeats the same title or copy")
        if require_direction and sum(_reading_units(text) for text in selected_copy) > _reading_budget(duration / 100):
            raise ValueError("shot copy exceeds its readable attention window; reduce copy or extend the shot")
        layout = item.get("layout", "cards")
        transition = item.get("transition", "wipe")
        if layout not in {"cards", "focus", "split", "flow", "comparison"} or transition not in {"wipe", "iris"}:
            raise ValueError("unknown layout or transition")
        relation = item.get("relationship_excerpt", "")
        if not isinstance(relation, str):
            raise ValueError("relationship_excerpt must be text")
        if layout in _RELATIONSHIP_MARKERS:
            if aid != "editorial":
                raise ValueError("relationship layouts require an editorial shot; preserve captured media framing")
            if len(ids) < 3 or (layout == "comparison" and len(ids) != 3):
                raise ValueError("relationship layout needs a title and two distinct supplied claims")
            if not relation or relation not in scene["text"] or not _RELATIONSHIP_MARKERS[layout].search(relation):
                raise ValueError("relationship layout lacks an explicit grounded relationship_excerpt")
        elif relation and relation not in scene["text"]:
            raise ValueError("relationship_excerpt is not from its narration")
        direction = _direction(item, asset.get("kind", "editorial"), selected_copy,
                               required=require_direction)
        expected_motion = {"flow": "trace", "comparison": "compare"}.get(layout, "reveal")
        if direction["motion"] != expected_motion:
            raise ValueError("direction motion must match the grounded layout")
        shots.append({
            **asset, "asset_id": aid, "kind": "hold" if hold else asset.get("kind", "editorial"),
            "start": cursor / 100, "duration": duration / 100,
            "source_start": source_start / 100, "treatment": "hold" if hold else "play",
            "source_duration": asset.get("duration"),
            "script_excerpt": excerpt, "copy_ids": ids,
            "copy": selected_copy, "layout": layout, "transition": transition,
            "direction": direction, "relationship_excerpt": relation,
            "motion_cues": motion_cues(duration / 100),
        })
        cursor += duration
    if cursor != ticks(scene["duration"]):
        raise ValueError(f"coverage ends at {cursor / 100:.2f}s; needs {ticks(scene['duration']) / 100:.2f}s")
    if seen != set(allowed):
        raise ValueError(f"required assets omitted: {sorted(set(allowed) - seen)}")
    return shots


def _prepare_holds(shots: list[dict], task_dir: Path) -> None:
    for shot in shots:
        if shot["kind"] != "hold":
            continue
        source = (task_dir / "compositions" / shot["src"]).resolve()
        directory = task_dir / "media_holds"
        directory.mkdir(exist_ok=True)
        name = hashlib.sha256(f"{shot['sha256']}:{shot['source_start']}".encode()).hexdigest()[:24]
        destination = directory / f"{name}.jpg"
        if not destination.is_file():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(shot["source_start"]),
                            "-i", str(source), "-frames:v", "1", "-q:v", "2", str(destination)],
                           capture_output=True, check=True, timeout=30)
        with Image.open(destination) as image:
            image.verify()
        shot["src"] = "../" + destination.relative_to(task_dir).as_posix()
        shot["credit"] = "Still frame · " + str(shot.get("credit") or "Source footage")


def fallback(scene: dict, assets: list[dict], copy: list[dict]) -> dict:
    """An explicit recovery only: retain assets and explain remaining narration."""
    remaining = ticks(scene["duration"])
    shots = []
    excerpt = scene["text"]
    copy_cursor = 0

    def readable_copy(duration, *, kind):
        nonlocal copy_cursor
        selected = []
        budget = _reading_budget(duration / 100)
        for offset in range(len(copy)):
            candidate = copy[(copy_cursor + offset) % len(copy)]
            units = _reading_units(candidate["text"])
            if units > budget:
                continue
            selected.append(candidate)
            budget -= units
            if len(selected) == (3 if kind == "editorial" else 1):
                break
        if not selected:
            raise ValueError("No supplied copy fits the shot's reading window")
        copy_cursor = (copy.index(selected[-1]) + 1) % len(copy)
        return {"copy_ids": [entry["id"] for entry in selected],
                "direction": _default_direction(kind, [entry["text"] for entry in selected])}

    for i, asset in enumerate(assets):
        reserve = 20 * (len(assets) - i - 1)
        duration = min(ticks(asset.get("duration", 6)), remaining - reserve)
        if duration < 20:
            raise ValueError("too many required assets for the scene duration")
        shots.append({"asset_id": asset["id"], "duration": duration / 100,
                      **readable_copy(duration, kind=asset["kind"]), "script_excerpt": excerpt})
        remaining -= duration
    # Leave a readable tail, not a one-frame information flash. When a video
    # almost fills the scene, shorten its trim rather than accelerating copy.
    shortest = min((_reading_units(entry["text"]) for entry in copy), default=0)
    minimum_reading_ticks = max(20, math.ceil(shortest / 3 * 100))
    tail_minimum = max(120, minimum_reading_ticks)
    if 0 < remaining < tail_minimum and shots:
        trim = min(tail_minimum - remaining, ticks(shots[-1]["duration"]) - minimum_reading_ticks)
        trim = max(0, trim)
        shots[-1]["duration"] = (ticks(shots[-1]["duration"]) - trim) / 100
        if trim:
            shots[-1].update(readable_copy(ticks(shots[-1]["duration"]), kind=assets[-1]["kind"]))
        remaining += trim
    while remaining:
        parts = max(1, min(math.ceil(remaining / 1200), remaining // minimum_reading_ticks))
        duration = remaining // parts
        selected = readable_copy(duration, kind="editorial")
        shots.append({"asset_id": "editorial", "duration": duration / 100,
                      **selected, "script_excerpt": excerpt,
                      "layout": "cards" if len(selected["copy_ids"]) > 1 else "focus"})
        remaining -= duration
    return {"id": scene["id"], "rationale": "Provider/plan recovery using verified assets and grounded editorial graphics", "shots": shots}


@timed("media_shots")
async def plan_media_shots(plans: list[dict], board: dict, task_dir: Path, *,
                           provider_id=None, ai_endpoint=None, ai_model=None, log=None) -> dict:
    from backend.pipeline.digester import _chat, _resolve_provider
    from backend.pipeline.visual_plan import _first_json_array

    def emit(message):
        if log:
            log(message)

    by_id = {plan["id"]: plan for plan in plans}
    contexts = []
    for scene in board.get("scenes", []):
        if scene.get("program_segment_kind") in {"opening", "closing"}:
            continue
        plan = by_id[scene["id"]]
        assets = inventory(plan, task_dir)
        contexts.append({"id": scene["id"], "duration": ticks(scene["duration"]) / 100,
                         "text": scene["text"], "lines": scene.get("lines", []),
                         "assets": assets, "editorial_copy": editorial_copy(plan, scene),
                         "visual_review_feedback": plan.get("visual_review_feedback", {}),
                         "director_brief": board.get("director_brief", {})})
    prompt = (config.PROMPTS_DIR / "media_shots.txt").read_text()
    fingerprint = hashlib.sha256(json.dumps([CONTRACT_VERSION, prompt, contexts], sort_keys=True).encode()).hexdigest()
    checkpoint = task_dir / "media_shots.json"
    cached = {}
    try:
        stored = json.loads(checkpoint.read_text())
        cached = {item["id"]: item for item in stored.get("scenes", [])}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    results = {}
    failures = {scene["id"]: [] for scene in contexts}
    for scene in contexts:
        scene["input_sha256"] = hashlib.sha256(
            json.dumps([CONTRACT_VERSION, prompt, scene], sort_keys=True).encode()
        ).hexdigest()

    def accept(scene, selected, planner):
        shots = validate(selected, scene, scene["assets"], scene["editorial_copy"], require_direction=True)
        _prepare_holds(shots, task_dir)
        by_id[scene["id"]]["media_shots"] = shots
        results[scene["id"]] = {
            **selected, "shots": shots, "planner": planner,
            "input_sha256": scene["input_sha256"], "validation_errors": failures[scene["id"]],
            "duration": scene["duration"], "coverage": 1.0, "passed": True,
            "review_rejected_media": by_id[scene["id"]].get("review_rejected_media", []),
        }
        checkpoint.write_text(json.dumps({"input_sha256": fingerprint, "passed": False,
                                          "scenes": list(results.values())}, indent=2, ensure_ascii=False))
        emit(f"Shot editor {scene['id']}: {planner}, "
             + " → ".join(shot["kind"] for shot in shots) + f"; {scene['duration']:.2f}s covered")

    for scene in contexts:
        selected = cached.get(scene["id"], {})
        if selected.get("input_sha256") == scene["input_sha256"]:
            try:
                accept(scene, selected, selected.get("planner", "ai"))
            except (ValueError, TypeError, KeyError, AttributeError, OSError, subprocess.SubprocessError):
                pass
    provider = None
    gate = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)

    async def plan_batch(batch, batch_index, attempt):
        async with gate:
            try:
                answer = await _chat(
                    prompt, json.dumps({"scenes": batch, "validation_feedback":
                                        {scene["id"]: failures[scene["id"]] for scene in batch}}, ensure_ascii=False),
                    *provider, log, f"Shot editor batch {batch_index} ({attempt + 1}/2)",
                    # Complete direction can exceed 14k; avoid a truncated
                    # first response before the provider doubles its budget.
                    max_tokens=28000, enable_skills=False, disable_thinking=True,
                )
                entries = _first_json_array(answer) or []
                indexed = {entry["id"]: entry for entry in entries}
            except Exception as exc:
                indexed = {}
                for scene in batch:
                    failures[scene["id"]].append(str(exc))
            # Validation and checkpoint writes are synchronous on this event
            # loop. Each batch owns disjoint scenes; completed work is retained
            # even while another batch is still awaiting its model response.
            for scene in batch:
                try:
                    accept(scene, indexed.get(scene["id"], {}), "ai")
                except (ValueError, TypeError, KeyError, AttributeError, OSError, subprocess.SubprocessError) as exc:
                    failures[scene["id"]].append(str(exc))
                    emit(f"Shot editor {scene['id']}: replanning after {exc}")

    # Batch normal direction; only rejected scenes return for the repair round.
    for attempt in range(2):
        pending = [scene for scene in contexts if scene["id"] not in results]
        if not pending:
            break
        try:
            if provider is None:
                provider = await _resolve_provider(provider_id, ai_endpoint, ai_model)
        except Exception as exc:
            for scene in pending:
                failures[scene["id"]].append(str(exc))
            continue
        # These are API-only editorial requests. _chat still applies the shared
        # provider pacing/cooldown; browser fact and final reviews stay unchanged.
        # TaskGroup cancels and drains siblings if the parent is interrupted.
        async with asyncio.TaskGroup() as group:
            for offset in range(0, len(pending), 4):
                group.create_task(plan_batch(pending[offset:offset + 4], offset // 4 + 1, attempt))
    for scene in contexts:
        if scene["id"] not in results:
            accept(scene, fallback(scene, scene["assets"], scene["editorial_copy"]), "recovery")
    report = {"contract_version": CONTRACT_VERSION, "input_sha256": fingerprint,
              "passed": True, "scenes": [results[scene["id"]] for scene in contexts]}
    checkpoint.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
