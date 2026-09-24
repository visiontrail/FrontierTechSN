"""Scene-synchronous illustrated explainers, independent from news production.

Provider media is real or generation fails. Content-addressed stages preserve
successful audio and artwork through retries and revisions without mixing scripts.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from backend import config
from backend.pipeline.collage_broll import _generate_still, _generate_video
from backend.pipeline.opencli_session import owned_media_session
from backend.pipeline.tts import generate_tts
from backend.pipeline.video_format import LANDSCAPE

FPS = 30
WIDTH, HEIGHT = 1920, 1080
SAMPLE_RATE = 48000
CONTRACT_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
Log = Callable[[str], None]


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _safe(path: Path, root: Path) -> Path:
    if ".." in path.parts:
        raise ValueError("Explainer artifact cannot traverse parent directories")
    absolute = path.absolute()
    if not absolute.is_relative_to(root.absolute()):
        raise ValueError("Explainer artifact must remain inside its project directory")
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Explainer artifact path cannot contain a symlink: {current}")
    return absolute


def _write(path: Path, value: str, root: Path) -> None:
    _safe(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _safe(path.with_name(path.name + ".tmp"), root)
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _json(path: Path, value: Any, root: Path) -> None:
    _write(path, json.dumps(value, ensure_ascii=False, indent=2), root)


def _stage(root: Path, scene_id: str, kind: str, key: str) -> Path:
    path = _safe(root / "artifacts" / scene_id / kind / key[:24], root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cached(stage: Path, key: str, root: Path) -> dict | None:
    checkpoint = _safe(stage / "complete.json", root)
    if not checkpoint.is_file():
        return None
    try:
        record = json.loads(checkpoint.read_text(encoding="utf-8"))
        if record.get("input_hash") != key or not record.get("files"):
            return None
        for name, checksum in record["files"].items():
            path = _safe(stage / name, root)
            if not path.is_file() or _sha(path) != checksum:
                return None
        return record
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _complete(stage: Path, key: str, paths: list[Path], root: Path, **details: Any) -> dict:
    record = {"input_hash": key, "files": {str(p.relative_to(stage)): _sha(p) for p in paths}, **details}
    _json(stage / "complete.json", record, root)
    return record


async def _command(args: list[str], *, timeout: float = 300, cwd: Path | None = None) -> bytes:
    """Bound subprocess lifetime and reap it on cancellation as well as timeout."""
    process = await asyncio.create_subprocess_exec(
        *args, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except (TimeoutError, asyncio.CancelledError):
        if process.returncode is None:
            process.kill()
        await process.communicate()
        raise
    if process.returncode:
        raise RuntimeError(f"{args[0]} failed: {stderr.decode(errors='replace')[-2000:]}")
    return stdout


async def _probe(path: Path) -> dict:
    result = json.loads(await _command([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
    ], timeout=30))
    return result


def _duration(probe: dict, kind: str) -> float:
    stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == kind), None)
    if stream is None:
        raise RuntimeError(f"Generated media has no {kind} stream")
    value = float(stream.get("duration") or probe.get("format", {}).get("duration") or 0)
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"Generated {kind} duration is invalid")
    return value


async def _verify(path: Path, *, duration: float, video: bool, audio: bool) -> dict:
    probe = await _probe(path)
    for kind, required in (("video", video), ("audio", audio)):
        if required and abs(_duration(probe, kind) - duration) > max(0.08, 2 / FPS):
            raise RuntimeError(f"Rendered {kind} does not cover the full narration timeline")
    if video:
        stream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
        if (stream.get("width"), stream.get("height")) != (WIDTH, HEIGHT):
            raise RuntimeError("Explainer render dimensions do not match the delivery canvas")
    await _command([
        "ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-",
    ], timeout=max(120, duration * 4))
    return probe


def _image_prompt(scene: dict, style: dict) -> str:
    return f"""Create an original illustrated explainer frame, 16:9 landscape, 1920x1080 delivery.
Use this recurring character description consistently: {style.get('character_description', 'A simple rounded line-drawn person with expressive eyes and minimal facial features')}.
Visual direction: {style.get('visual_direction', 'Minimal expressive line drawings, sparse everyday props, clear emotional metaphors and generous negative space')}.
Background color {style.get('background_color', '#F4EBDD')}; ink color {style.get('ink_color', '#302A25')}; selective accent color {style.get('accent_color', '#C56545')}.
Depict this precise scene: {scene['visual_prompt']}
Narration context, for meaning only: {scene['narration']}
Compose a single readable scene. Keep the main character and meaningful action in the central safe area; leave the lower quarter quiet for captions. Keep recurring character proportions, ink weight and colors consistent across the series.
Draw original characters and imagery. Do not reproduce any channel logo, recognizable branded character, existing thumbnail or frame. No text, typography, letters, numerals, watermarks, UI or embedded subtitles. No photorealism or paper-cut collage unless expressly requested in the visual direction above."""


def _motion_prompt(scene: dict, style: dict) -> str:
    return f"""Animate the supplied original line illustration as a short 16:9 explainer shot.
Use this single reference frame to preserve the character identity, palette, proportions, background and line weight. Begin with this composition. Add a single gentle expressive gesture, blink or prop movement that supports this narration: {scene['narration']}
Visual meaning: {scene['visual_prompt']}
Art direction: {style.get('visual_direction', '')}
Keep the camera steady and movement subtle. Show a short, complete action with a clear settled endpoint; avoid a long buildup. Do not create a montage or new scene. No typography, subtitles, logos or audio. Play the action once, settle naturally, and hold the final pose. Never restart the action or loop."""


def _font(size: int, language: str) -> ImageFont.FreeTypeFont:
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    if language != "zh":
        paths += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    for path in paths:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    raise RuntimeError("Install a Unicode font (Arial Unicode or Noto Sans CJK) to render explainer captions")


def _wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    # Keep Latin words intact where possible; CJK naturally wraps by character.
    tokens = re.findall(r"[\u3400-\u9fff]|[^\u3400-\u9fff\s]+\s*|\s+", text.replace("\n", " "))
    lines: list[str] = []
    current = ""
    for token in tokens:
        if font.getlength(current + token) <= width:
            current += token
            continue
        if current.strip():
            lines.append(current.strip())
        current = ""
        for char in token:
            if current and font.getlength(current + char) > width:
                lines.append(current.strip())
                current = ""
            current += char
    if current.strip():
        lines.append(current.strip())
    return lines


def _overlay(scene: dict, project: dict, path: Path) -> None:
    """Rasterize text ourselves: Homebrew ffmpeg may have no libass filters."""
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    language = project.get("language", "en")
    style = project.get("style") or {}
    title = str(scene.get("on_screen_text") or "").strip()
    if title:
        font = _font(52, language)
        lines = _wrap(title, font, WIDTH - 240)
        if len(lines) > 3:
            raise ValueError("On-screen text is too long; shorten it to three lines")
        width = max(font.getlength(line) for line in lines) + 64
        left, right = (WIDTH - width) / 2, (WIDTH + width) / 2
        draw.rounded_rectangle((left, 56, right, 90 + len(lines) * 65), radius=18,
                               fill=style.get("background_color", "#F4EBDD"))
        draw.line((left + 24, 65, right - 24, 65), fill=style.get("accent_color", "#C56545"), width=5)
        for index, line in enumerate(lines):
            draw.text((WIDTH / 2, 86 + index * 65), line, font=font, anchor="mt",
                      fill=style.get("ink_color", "#302A25"))
    if project.get("captions_enabled", False):
        lines: list[str] = []
        for size in (48, 44, 40, 36, 32):
            font = _font(size, language)
            lines = _wrap(scene["narration"], font, WIDTH - 220)
            if len(lines) <= 5:
                break
        if len(lines) > 5:
            raise ValueError("Scene narration is too long for readable captions; split this scene")
        line_height = int(font.size * 1.35)
        top = HEIGHT - 76 - len(lines) * line_height
        draw.rounded_rectangle((75, top - 18, WIDTH - 75, HEIGHT - 48), radius=20, fill=(28, 25, 23, 218))
        for index, line in enumerate(lines):
            draw.text((WIDTH / 2, top + index * line_height), line, font=font, anchor="mt", fill="#FFFFFF")
    overlay.save(path)


def _srt_time(value: float) -> str:
    millis = round(value * 1000)
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    seconds, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


async def _audio(scene: dict, project: dict, root: Path, log: Log) -> tuple[Path, dict]:
    model = project.get("tts_model") or config.TTS_DEFAULT_MODEL
    voice = project.get("voice") or config.TTS_DEFAULT_VOICE_1
    provider_model = config.TTS_MODELS.get(model) or {}
    requires_labels = bool(provider_model.get("requires_speaker_labels"))
    inputs = {"version": CONTRACT_VERSION, "narration": scene["narration"], "model": model,
                 "voice": voice, "language": project.get("language"),
                 "provider": provider_model, "seed": config.TTS_RANDOM_SEED,
                 "endpoint": config.ORPHEUS_TTS_URL if model == "orpheus-en" else config.POCKET_TTS_URL if model == "pocket-tts-en" else None,
                 "provider_revision": config.POCKET_TTS_MODEL_REVISION if model == "pocket-tts-en" else None}
    if requires_labels:
        inputs["speaker_input_contract"] = 1
    key = _hash(inputs)
    stage = _stage(root, scene["id"], "audio", key)
    cached = _cached(stage, key, root)
    if cached:
        return stage / "narration.wav", cached
    _write(stage / "script.txt", scene["narration"], root)
    script_path = stage / "script.txt"
    if requires_labels:
        # Dialogue-capable models need metadata on every nonempty input line.
        # Keep the canonical script, captions and scene narration untouched.
        provider_input = "\n".join(
            f"Speaker 1: {line.strip()}" for line in scene["narration"].splitlines() if line.strip()
        )
        script_path = stage / "provider-script.txt"
        _write(script_path, provider_input, root)
    raw = Path(await asyncio.wait_for(generate_tts(
        str(script_path), str(stage / "provider"), voices=[voice], tts_model=model, log=log,
    ), timeout=max(60, min(1800, config.TTS_TIMEOUT))))
    _safe(raw, root)
    if not raw.is_file():
        raise RuntimeError("TTS returned no audio file")
    measured = _duration(await _probe(raw), "audio")
    if measured > 180:
        raise ValueError("A single explainer scene exceeds 180 seconds; split its narration")
    frames = max(1, math.ceil(measured * FPS - 1e-7))
    duration = frames / FPS
    temporary = stage / "narration.partial.wav"
    await _command([
        "ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af",
        f"aresample={SAMPLE_RATE},apad,atrim=end_sample={frames * (SAMPLE_RATE // FPS)}",
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(temporary),
    ])
    await _verify(temporary, duration=duration, video=False, audio=True)
    final = stage / "narration.wav"
    temporary.replace(final)
    input_paths = [stage / "script.txt", script_path] if requires_labels else [script_path]
    record = _complete(stage, key, [final, *input_paths], root,
                       narration_duration_seconds=measured, duration_seconds=duration, frames=frames,
                       provider="existing_generate_tts", model=model, voice=voice,
                       input_policy="speaker_1_metadata" if requires_labels else "plain_narration")
    return final, record


async def _image(scene: dict, style: dict, root: Path) -> tuple[Path, dict]:
    prompt = _image_prompt(scene, style)
    key = _hash({"version": CONTRACT_VERSION, "prompt": prompt})
    stage = _stage(root, scene["id"], "image", key)
    # Reclaim interrupted browser ownership even on a cache hit.
    async with owned_media_session(stage, "chatgpt", recover_only=True):
        pass
    cached = _cached(stage, key, root)
    if cached:
        return stage / "image.png", cached
    _write(stage / "prompt.txt", prompt, root)
    raw, conversation = await _generate_still(prompt, stage)
    _safe(raw, root)
    with Image.open(raw) as artwork:
        if artwork.width < 256 or artwork.height < 144:
            raise RuntimeError("Generated illustration is too small to render")
        artwork.convert("RGB").save(stage / "image.partial.png")
    final = stage / "image.png"
    (stage / "image.partial.png").replace(final)
    return final, _complete(stage, key, [final, stage / "prompt.txt"], root,
                            provider="chatgpt_web_opencli", conversation_url=conversation)


async def _motion(scene: dict, style: dict, image: Path, root: Path) -> tuple[Path, dict]:
    prompt = _motion_prompt(scene, style)
    key = _hash({"version": CONTRACT_VERSION, "prompt": prompt, "image": _sha(image)})
    stage = _stage(root, scene["id"], "motion", key)
    async with owned_media_session(stage, "gemini", recover_only=True):
        pass
    cached = _cached(stage, key, root)
    if cached:
        return stage / "motion.mp4", cached
    _write(stage / "prompt.txt", prompt, root)
    # Normalize a small upload keyframe, preserving the entire illustration.
    keyframe = stage / "keyframe.jpg"
    with Image.open(image) as artwork:
        artwork.thumbnail((1280, 720), Image.Resampling.LANCZOS)
        color = style.get("background_color", "#F4EBDD")
        canvas = Image.new("RGB", (1280, 720), color)
        canvas.paste(artwork, ((1280 - artwork.width) // 2, (720 - artwork.height) // 2))
        canvas.save(keyframe, quality=92)
    raw, conversation = await _generate_video(prompt, keyframe, None, stage, LANDSCAPE)
    _safe(raw, root)
    measured = _duration(await _probe(raw), "video")
    await _command(["ffmpeg", "-v", "error", "-xerror", "-i", str(raw), "-an", "-f", "null", "-"], timeout=120)
    final = stage / "motion.mp4"
    shutil.copyfile(raw, stage / "motion.partial.mp4")
    (stage / "motion.partial.mp4").replace(final)
    return final, _complete(stage, key, [final, stage / "prompt.txt", keyframe], root,
                            provider="gemini_web_opencli", conversation_url=conversation,
                            source_duration_seconds=measured,
                            playback_policy="play_once_fit_longer_or_hold_shorter")


async def _render_scene(scene: dict, project: dict, image: Path, motion: Path | None,
                        audio: dict, root: Path) -> tuple[Path, dict]:
    style = project.get("style") or {}
    key = _hash({"version": CONTRACT_VERSION, "image": _sha(image), "motion": _sha(motion) if motion else None,
                 "motion_playback_version": 2 if motion else 1,
                 "frames": audio["frames"], "mode": style.get("motion_mode", "none"),
                 "captions": project.get("captions_enabled", False), "text": scene["narration"],
                 "title": scene.get("on_screen_text"), "language": project.get("language"),
                 "background": style.get("background_color", "#F4EBDD"),
                 "ink": style.get("ink_color", "#302A25"),
                 "accent": style.get("accent_color", "#C56545")})
    stage = _stage(root, scene["id"], "render", key)
    cached = _cached(stage, key, root)
    if cached:
        return stage / "clip.mp4", cached
    _overlay(scene, project, stage / "overlay.png")
    color = style.get("background_color", "#F4EBDD").replace("#", "0x")
    base = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color={color},setsar=1"
    args = ["ffmpeg", "-y", "-v", "error"]
    motion_playback = None
    if motion:
        args += ["-i", str(motion)]
        source_duration = _duration(await _probe(motion), "video")
        target_duration = audio["duration_seconds"]
        rate = max(1.0, source_duration / target_duration)
        motion_playback = {
            "policy": "fit_full_clip" if rate > 1 else "play_once_then_hold",
            "playback_rate": rate,
            "source_duration_seconds": source_duration,
            "played_duration_seconds": min(source_duration, target_duration),
            "hold_seconds": max(0.0, target_duration - source_duration),
            "narration_playback_rate": 1.0,
        }
        # Only the silent visual may be accelerated. Narration remains the
        # exact generated waveform; short visual sources play once then hold.
        base += (f",setpts=(PTS-STARTPTS)/{rate:.12f},fps={FPS},"
                 f"tpad=stop_mode=clone:stop_duration={target_duration}")
    else:
        args += ["-loop", "1", "-framerate", str(FPS), "-i", str(image)]
        if style.get("motion_mode", "none") == "subtle":
            base += (f",zoompan=z='1+0.018*on/{max(1, audio['frames'] - 1)}':"
                     f"x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=1:s={WIDTH}x{HEIGHT}:fps={FPS}")
    args += ["-loop", "1", "-framerate", str(FPS), "-i", str(stage / "overlay.png"),
             "-filter_complex", f"[0:v]{base}[base];[base][1:v]overlay=0:0:shortest=1,format=yuv420p[v]",
             "-map", "[v]", "-an", "-frames:v", str(audio["frames"]), "-r", str(FPS),
             "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-threads", "4",
             "-movflags", "+faststart", str(stage / "clip.partial.mp4")]
    await _command(args, timeout=max(180, audio["duration_seconds"] * 30))
    await _verify(stage / "clip.partial.mp4", duration=audio["duration_seconds"], video=True, audio=False)
    final = stage / "clip.mp4"
    (stage / "clip.partial.mp4").replace(final)
    return final, _complete(stage, key, [final, stage / "overlay.png"], root,
                            motion_playback=motion_playback)


def _validate(project: dict, root: Path) -> list[dict]:
    _safe(root, root)
    # Existing descendants can later be touched by third-party provider helpers.
    if root.exists() and any(child.is_symlink() for child in root.rglob("*")):
        raise ValueError("Explainer project cannot contain symlink artifacts")
    scenes = project.get("scenes")
    if not isinstance(scenes, list) or not scenes or len(scenes) > 900:
        raise ValueError("An explainer needs 1 to 900 scripted scenes")
    ids = set()
    for scene in scenes:
        scene_id = str(scene.get("id", ""))
        if not _SAFE_ID.fullmatch(scene_id) or scene_id in ids:
            raise ValueError("Scene IDs must be unique, safe filenames")
        ids.add(scene_id)
        if not str(scene.get("narration", "")).strip() or not str(scene.get("visual_prompt", "")).strip():
            raise ValueError("Every scene needs narration and an illustration prompt")
    style = project.get("style") or {}
    for name in ("background_color", "ink_color", "accent_color"):
        if name in style and not _HEX.fullmatch(str(style[name])):
            raise ValueError(f"Invalid style color: {name}")
    if style.get("motion_mode", "none") not in ("none", "subtle", "gemini"):
        raise ValueError("Unsupported motion mode")
    return scenes


async def render_project(project: dict, project_dir: Path,
                         update: Callable[[dict], None], log: Log) -> dict:
    root = project_dir.absolute()
    scenes = _validate(project, root)
    root.mkdir(parents=True, exist_ok=True)
    style = project.get("style") or {}
    total = len(scenes) * 4 + 1
    completed = 0
    artifacts: list[dict] = []
    current_frame = 0

    def progress(status: str, message: str) -> None:
        update({"status": status, "progress": {"completed": completed, "total": total, "message": message}})
        log(message)

    def relative(path: Path) -> str:
        return str(_safe(path, root).relative_to(root))

    def save_manifest(status: str, error: str | None = None) -> None:
        _json(root / "manifest.json", {"version": CONTRACT_VERSION, "project_id": project.get("id"),
              "status": status, "error": error, "fps": FPS, "canvas": [WIDTH, HEIGHT],
              "scene_artifacts": artifacts}, root)

    try:
        for journal in (root / "artifacts").rglob("browser-session-*.json"):
            site = journal.stem.removeprefix("browser-session-")
            if site in ("chatgpt", "gemini"):
                async with owned_media_session(journal.parent, site, recover_only=True):
                    pass
        for index, scene in enumerate(scenes):
            prefix = f"Scene {index + 1}/{len(scenes)}"
            progress("generating_audio", f"{prefix}: generating or reusing exact narration")
            audio_path, audio = await _audio(scene, project, root, log)
            completed += 1
            progress("generating_images", f"{prefix}: generating or reusing ChatGPT illustration")
            image_path, illustration = await _image(scene, style, root)
            completed += 1
            motion_path, motion = None, None
            every = max(0, int(style.get("motion_every_n_scenes", 4)))
            selected = bool(scene.get("motion")) or (every > 0 and (index + 1) % every == 0)
            if style.get("motion_mode") == "gemini" and selected:
                progress("generating_motion", f"{prefix}: generating or reusing Gemini motion")
                motion_path, motion = await _motion(scene, style, image_path, root)
            completed += 1
            progress("rendering", f"{prefix}: composing measured narration and illustration")
            clip, render = await _render_scene(scene, project, image_path, motion_path, audio, root)
            completed += 1
            item = {"id": scene["id"], "chapter": scene.get("chapter", ""), "narration": scene["narration"],
                    "on_screen_text": scene.get("on_screen_text", ""),
                    "start_frame": current_frame, "end_frame": current_frame + audio["frames"],
                    "start_seconds": current_frame / FPS, "duration_seconds": audio["duration_seconds"],
                    "narration_duration_seconds": audio["narration_duration_seconds"],
                    "audio_path": relative(audio_path), "image_path": relative(image_path),
                    "motion_path": relative(motion_path) if motion_path else None, "clip_path": relative(clip),
                    "audio_sha256": _sha(audio_path), "clip_sha256": _sha(clip),
                    "audio_hash": audio["input_hash"], "image_hash": illustration["input_hash"],
                    "motion_hash": motion["input_hash"] if motion else None, "render_hash": render["input_hash"],
                    "motion_playback": render.get("motion_playback"),
                    "image_source_url": illustration.get("conversation_url"),
                    "motion_source_url": motion.get("conversation_url") if motion else None}
            artifacts.append(item)
            current_frame += audio["frames"]
            save_manifest("rendering")
            update({"scene_artifacts": artifacts, "progress": {"completed": completed, "total": total, "message": f"{prefix}: ready"}})
        duration = current_frame / FPS
        timeline = {"version": CONTRACT_VERSION, "fps": FPS, "duration_seconds": duration,
                    "total_frames": current_frame, "caption_alignment": "scene_narration",
                    "playback_policy": "play_once_fit_longer_or_hold_shorter", "scenes": artifacts}
        _json(root / "timeline.json", timeline, root)
        captions = []
        for index, item in enumerate(artifacts, 1):
            start = item["start_seconds"]
            text = re.sub(r"\s+", " ", item["narration"]).strip()
            captions.append(f"{index}\n{_srt_time(start)} --> {_srt_time(start + item['narration_duration_seconds'])}\n{text}\n")
        _write(root / "captions.srt", "\n".join(captions), root)
        delivery_key = _hash({"version": CONTRACT_VERSION, "timeline": timeline})
        delivery = _safe(root / "renders" / delivery_key[:24], root)
        delivery.mkdir(parents=True, exist_ok=True)
        cached = _cached(delivery, delivery_key, root)
        progress("rendering", "Joining all scenes and verifying complete audio/video coverage")
        if not cached:
            for kind, field in (("video", "clip_path"), ("audio", "audio_path")):
                # Internally generated relative paths contain no quote or escape characters.
                entries = ["file '" + os.path.relpath(root / item[field], delivery) + "'" for item in artifacts]
                _write(delivery / f"{kind}.txt", "\n".join(entries) + "\n", root)
            await _command(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", "audio.txt",
                            "-c:a", "pcm_s16le", "narration.partial.wav"], cwd=delivery,
                           timeout=max(120, duration * 4))
            await _verify(delivery / "narration.partial.wav", duration=duration, video=False, audio=True)
            (delivery / "narration.partial.wav").replace(delivery / "narration.wav")
            await _command(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", "video.txt",
                            "-i", "narration.wav", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "video.partial.mp4"],
                           cwd=delivery, timeout=max(180, duration * 8))
            await _verify(delivery / "video.partial.mp4", duration=duration, video=True, audio=True)
            (delivery / "video.partial.mp4").replace(delivery / "video.mp4")
            _complete(delivery, delivery_key, [delivery / "video.mp4", delivery / "narration.wav"], root,
                      duration_seconds=duration, verified_full_decode=True)
        result = {"video_path": relative(delivery / "video.mp4"), "audio_path": relative(delivery / "narration.wav"),
                  "timeline_path": "timeline.json", "captions_path": "captions.srt", "manifest_path": "manifest.json",
                  "duration_seconds": duration, "scene_artifacts": artifacts}
        completed = total
        save_manifest("completed")
        update({"progress": {"completed": completed, "total": total, "message": "Video and narration verified"}})
        return result
    except asyncio.CancelledError:
        save_manifest("cancelled")
        raise
    except Exception as exc:
        save_manifest("failed", str(exc))
        raise
