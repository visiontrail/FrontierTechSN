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
CLIP_SECONDS = 5
CLIP_FPS = 24
MOTION_SAMPLE_FPS = 4
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
_COLORS = ("#D96B35", "#D2A928", "#315F4C", "#594080", "#188C85", "#B73D3D")


def _log(log: LogCallback | None, message: str) -> None:
    if log:
        log(message)
    else:
        logger.info(message)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


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
    objects = ["central halftone subject", "paper mechanism", "connector pieces", "result card"]
    return {
        "scene_id": scene["id"],
        "script_meaning": meaning,
        "emotion": "clarity",
        "visual_metaphor": f"A paper mechanism physically reveals how {meaning.rstrip('.。')}.",
        "background_hex": _COLORS[index % len(_COLORS)],
        "accent_colors": ["warm cream", "cyan"],
        "elements": [
            {"what": item, "role": "metaphor", "motion": "slides and snaps into place", "placement": "center"}
            for item in objects
        ],
        "assembly_order": objects,
        "final_frame": "A concentrated completed paper mechanism with generous clear color field.",
        "planner": "deterministic_fallback",
    }


def _normalize_spec(raw: dict, scene: dict, index: int) -> dict[str, Any]:
    fallback = _fallback_spec(scene, index)
    spec = {**fallback, **raw, "scene_id": scene["id"], "planner": "claude_agent_sdk"}
    color = str(spec.get("background_hex") or "")
    spec["background_hex"] = color.upper() if _HEX.fullmatch(color) else fallback["background_hex"]
    accents = [str(value)[:40] for value in (spec.get("accent_colors") or []) if str(value).strip()]
    spec["accent_colors"] = accents[:3] or fallback["accent_colors"]
    elements = [item for item in (spec.get("elements") or []) if isinstance(item, dict)][:6]
    spec["elements"] = elements if len(elements) >= 3 else fallback["elements"]
    order = [str(value)[:100] for value in (spec.get("assembly_order") or []) if str(value).strip()]
    spec["assembly_order"] = order[:6] or [item["what"] for item in spec["elements"]]
    for key in ("script_meaning", "emotion", "visual_metaphor", "final_frame"):
        spec[key] = str(spec.get(key) or fallback[key])[:500]
    return spec


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
    accents = ", ".join(spec["accent_colors"])
    return f"""Use case: documentary B-roll.
Asset type: final still frame for a {frame.aspect_ratio} image-to-video clip.
Create a finished premium editorial paper-collage expressing this visual metaphor: {spec['visual_metaphor']}
Scene/backdrop: a perfectly flat {spec['background_hex']} paper field with subtle uncoated-paper fiber.
Style: sophisticated stop-motion editorial collage; black-and-white halftone photographic cut-outs mixed with selective {accents} colored cardstock.
Composition: locked {frame.aspect_ratio} frame; one concentrated subject inside the middle 70 percent; generous clean color-field negative space; three to six large separable groups for assemble-from-empty animation.
Required groups: {elements}.
Final relationship: {spec['final_frame']}
Materials: visible printed halftone dots, crisp machine-cut edges, thin warm-cream keylines, soft low-opacity physical shadows.
Avoid all typography, readable letters, numerals, logos, watermarks, UI, subtitles, glossy 3D, photoreal environments, clutter, frames, and borders."""


def video_prompt(spec: dict[str, Any], frame: FrameSpec) -> str:
    order = " → ".join(spec["assembly_order"])
    return f"""Paper-collage stop-motion assembly. Image 1 is the exact empty first frame and Image 2 is the exact completed last frame. In one continuous locked-off {frame.aspect_ratio} shot, open on the empty flat {spec['background_hex']} paper field.

Assemble the scene piece by piece with crisp physical stop-motion timing in this exact order: {order}. Pieces slide, drop, stamp, and snap into place. End by holding the supplied completed Image 2 composition.

Preserve the exact {frame.aspect_ratio} framing, {spec['background_hex']} color field, colored cardstock accents, uncoated paper grain, halftone dots, cream keylines, crisp cut edges, and soft paper shadows. Restrained tactile 2D paper craft only. Aim for five seconds.

No scene cuts, camera movement, zoom, morphing, new objects, text, letters, numbers, logos, watermark, UI, or sound."""


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
    args: list[str], *, timeout: int, label: str
):
    """Give every browser-backed collage operation ten independent chances."""
    last_error: Exception | None = None
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
            if attempt < maximum_attempts:
                await asyncio.sleep(min(45, 3 * 2 ** (attempt - 1)))
    raise OpenCLIError(
        f"{label} exhausted {maximum_attempts} OpenCLI attempts: {last_error}"
    ) from last_error


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
    first: Path, last: Path, item_dir: Path, frame: FrameSpec
) -> Path:
    """Animate a still as staggered paper tiles with motion across all five seconds."""
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
    total_frames = CLIP_SECONDS * CLIP_FPS
    try:
        for frame_index in range(total_frames):
            progress = frame_index / max(1, total_frames - 1)
            canvas = background.copy().convert("RGBA")
            for index, tile_spec in enumerate(tile_specs):
                entrance_start = 0.06 + index * 0.095
                entrance_progress = min(1.0, max(0.0, (progress - entrance_start) / 0.25))
                exit_start = 0.72 + (len(tile_specs) - index - 1) * 0.035
                exit_progress = min(1.0, max(0.0, (progress - exit_start) / 0.105))
                if entrance_progress <= 0 or exit_progress >= 1:
                    continue
                # Back-ease gives each paper piece a physical snap on arrival.
                shifted = entrance_progress - 1
                eased = 1 + 2.70158 * shifted**3 + 1.70158 * shifted**2
                exit_eased = exit_progress * exit_progress * (3 - 2 * exit_progress)
                direction_x, direction_y = tile_spec["direction"]
                target_x, target_y = tile_spec["target"]
                travel = 1 - eased + exit_eased
                travel_x = direction_x * width * 0.72 * travel
                travel_y = direction_y * height * 0.72 * travel
                ambient = max(0.0, entrance_progress - 0.75) / 0.25
                drift_x = math.sin(progress * math.tau * 1.15 + tile_spec["phase"]) * 4 * ambient
                drift_y = math.cos(progress * math.tau * 0.9 + tile_spec["phase"]) * 3 * ambient
                rotation = tile_spec["rotation"] + travel * direction_x * 10
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

            # A reversible camera push keeps the assembled hold alive and returns
            # to the exact starting scale at the seamless loop boundary.
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
    )
    if not raw.is_file() or raw.stat().st_size < 1024:
        raise OpenCLIError("Gemini Web Create Video produced no downloaded MP4")
    rows = _rows(first_json(result.stdout))
    conversation = next((_field(row, "link") for row in rows if _field(row, "link")), "")
    return raw, conversation


async def _normalize_video(raw: Path, item_dir: Path, frame: FrameSpec) -> Path:
    final = item_dir / "video" / "final-5s-noaudio.mp4"
    await _media_command(
        [
            "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(raw),
            "-t", str(CLIP_SECONDS), "-vf",
            f"scale={frame.media_width}:{frame.media_height}:force_original_aspect_ratio=increase,crop={frame.media_width}:{frame.media_height},fps={CLIP_FPS}",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final),
        ],
        timeout=300,
    )
    return final


async def probe_video(path: Path, frame: FrameSpec) -> dict[str, Any]:
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
    motion = await _probe_motion(path, width=int(stream.get("width") or 0), height=int(stream.get("height") or 0))
    checks = {
        "dimensions": (stream.get("width"), stream.get("height")) == (frame.media_width, frame.media_height),
        "duration": abs(duration - CLIP_SECONDS) <= 0.15,
        "fps": abs(fps - CLIP_FPS) <= 0.1,
        "no_audio": not audio,
        "sustained_motion": motion["active_seconds"] >= 4,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "width": stream.get("width"),
        "height": stream.get("height"),
        "duration": round(duration, 3),
        "fps": round(fps, 3),
        "audio_streams": len(audio),
        "motion": motion,
    }


async def _probe_motion(path: Path, *, width: int, height: int) -> dict[str, Any]:
    """Measure low-resolution frame deltas and reject clips that settle into a still."""
    if width <= 0 or height <= 0:
        return {"active_seconds": 0, "second_scores": []}
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
    for second in range(CLIP_SECONDS):
        start = second * MOTION_SAMPLE_FPS
        end = min(len(frames) - 1, (second + 1) * MOTION_SAMPLE_FPS)
        deltas: list[float] = []
        for index in range(start, end):
            before, after = frames[index], frames[index + 1]
            deltas.append(sum(abs(left - right) for left, right in zip(before, after, strict=True)) / frame_size)
        second_scores.append(round(sum(deltas) / len(deltas), 3) if deltas else 0.0)
    active_seconds = sum(score >= 0.35 for score in second_scores)
    return {"active_seconds": active_seconds, "second_scores": second_scores}


async def _contact_sheet(video: Path, item_dir: Path, frame: FrameSpec) -> Path:
    sheet = item_dir / "video" / "contact-sheet.jpg"
    thumb_w, thumb_h = ((256, 144) if not frame.is_portrait else (144, 256))
    await _media_command(
        [
            "ffmpeg", "-y", "-i", str(video), "-vf",
            f"fps=1,scale={thumb_w}:{thumb_h},tile=5x1", "-frames:v", "1", str(sheet),
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
    root = task_dir / "collage_broll"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest: dict[str, Any] = {
        "status": "planning",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "approval_gates": [],
        "approval_mode": "automatic",
        "planner": "claude_agent_sdk",
        "still_provider": "chatgpt_web_via_opencli",
        "video_provider": "gemini_web_create_video_via_opencli",
        "gemini_api_key_used": False,
        "orientation": frame.orientation,
        "aspect_ratio": frame.aspect_ratio,
        "requested_count": count,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "items": [],
        "errors": [],
    }
    specs_path = root / "visual-spec.json"
    cached_specs: list[dict[str, Any]] = []
    if specs_path.is_file():
        try:
            payload = json.loads(specs_path.read_text(encoding="utf-8"))
            valid_scene_ids = {
                str(scene.get("id") or "") for scene in storyboard.get("scenes") or []
            }
            if (
                isinstance(payload, list)
                and len(payload) == min(max(1, count), len(valid_scene_ids))
                and (
                    not force_opening
                    or (
                        bool(payload)
                        and str(payload[0].get("scene_id") or "")
                        == str((storyboard.get("scenes") or [{}])[0].get("id") or "")
                    )
                )
                and all(
                    isinstance(item, dict)
                    and str(item.get("scene_id") or "") in valid_scene_ids
                    for item in payload
                )
            ):
                cached_specs = payload
        except (OSError, json.JSONDecodeError):
            cached_specs = []
    _write_json(manifest_path, manifest)
    if cached_specs:
        specs = cached_specs
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
    _write_json(specs_path, specs)
    manifest["status"] = "generating"
    _write_json(manifest_path, manifest)

    for index, spec in enumerate(specs, start=1):
        scene_id = spec["scene_id"]
        item_dir = root / f"{index:02d}-{scene_id}"
        item_dir.mkdir(parents=True, exist_ok=True)
        still_prompt = image_prompt(spec, frame)
        motion_prompt = video_prompt(spec, frame)
        (item_dir / "image-prompt.txt").write_text(still_prompt + "\n", encoding="utf-8")
        (item_dir / "video-prompt.txt").write_text(motion_prompt + "\n", encoding="utf-8")
        item: dict[str, Any] = {
            "scene_id": scene_id,
            "status": "generating",
            "spec": spec,
            "still_path": "",
            "video_path": "",
            "contact_sheet": "",
            "chatgpt_url": "",
            "gemini_url": "",
            "qa": {},
            "generation_warnings": [],
            "error": None,
        }
        manifest["items"].append(item)
        _write_json(manifest_path, manifest)
        try:
            cached_final = item_dir / "video" / "final-5s-noaudio.mp4"
            if cached_final.is_file():
                cached_qa = await probe_video(cached_final, frame)
                if cached_qa["passed"]:
                    cached_sheet = item_dir / "video" / "contact-sheet.jpg"
                    if not cached_sheet.is_file():
                        cached_sheet = await _contact_sheet(cached_final, item_dir, frame)
                    cached_still = item_dir / "frames" / "last-frame.jpg"
                    item.update(
                        {
                            "status": "ready",
                            "still_path": (
                                str(cached_still.relative_to(task_dir))
                                if cached_still.is_file()
                                else ""
                            ),
                            "video_path": str(cached_final.relative_to(task_dir)),
                            "contact_sheet": str(cached_sheet.relative_to(task_dir)),
                            "still_provider": "existing_verified_artifact",
                            "video_provider": "existing_verified_artifact",
                            "qa": cached_qa,
                        }
                    )
                    _log(log, f"Collage B-roll {index}/{len(specs)}: reused verified clip for {scene_id}")
                    _write_json(manifest_path, manifest)
                    continue
            _log(log, f"Collage B-roll {index}/{len(specs)}: generating {frame.aspect_ratio} still for {scene_id}")
            still_dir = item_dir / "stills"
            reusable_stills = (
                sorted(
                    (
                        path
                        for path in still_dir.glob("*")
                        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
                    ),
                    key=lambda path: path.stat().st_mtime_ns,
                )
                if still_dir.is_dir()
                else []
            )
            if reusable_stills:
                still = reusable_stills[-1]
                chatgpt_url = ""
                item["still_provider"] = "existing_chatgpt_artifact"
                _log(
                    log,
                    f"Collage B-roll {index}/{len(specs)}: reusing existing still for {scene_id}",
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
            first, last = await _prepare_frames(still, item_dir, spec["background_hex"], frame)
            _log(log, f"Collage B-roll {index}/{len(specs)}: animating {scene_id} in Gemini Web Create Video")
            try:
                raw, gemini_url = await _generate_video(motion_prompt, first, last, item_dir, frame)
                item["video_provider"] = "gemini_web_create_video_via_opencli"
            except Exception as exc:  # noqa: BLE001 - local animation preserves requested count
                warning = f"Web video unavailable ({exc}); used deterministic local paper assembly"
                item["generation_warnings"].append(warning)
                _log(log, f"Collage B-roll {index}/{len(specs)}: {warning}")
                raw = await _animate_still_locally(first, last, item_dir, frame)
                gemini_url = ""
                item["video_provider"] = "deterministic_local_paper_assembly"
            final = await _normalize_video(raw, item_dir, frame)
            qa = await probe_video(final, frame)
            if not qa["passed"]:
                raise RuntimeError(f"normalized collage clip failed QA: {qa['checks']}")
            sheet = await _contact_sheet(final, item_dir, frame)
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
        if preferred_id in occupied:
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
            }
        )
        item["placed_scene_id"] = placed_scene_id
        occupied.add(placed_scene_id)
        attached += 1
    if manifest is not None:
        manifest["placed_count"] = attached
        _write_json(task_dir / "collage_broll" / "manifest.json", manifest)
    return attached
