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
from backend.explainer_series import BRAND_VERSION, SERIES_NAME, motion_indices, validate_motion_budget, thumbnail_prompt
from backend.explainer_availability import GeminiVideoDeferred, availability_reason
from backend.explainer_pacing import pacing_plan, write_paced_audio
from backend.explainer_visual_review import (
    MAX_IMAGE_ATTEMPTS,
    REVIEW_VERSION,
    review_image,
    review_prompt,
    visual_logic_rules,
)
from backend.pipeline.collage_broll import _generate_still, _generate_video
from backend.pipeline.gemini_watermark import repair_plan
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


def _legacy_image_prompt(scene: dict, style: dict) -> str:
    """Exact v1 base, retained to locate and review existing task artifacts."""
    return f"""Create an original illustrated explainer frame, 16:9 landscape, 1920x1080 delivery.
Use this recurring character description consistently: {style.get('character_description', 'A simple rounded line-drawn person with expressive eyes and minimal facial features')}.
Visual direction: {style.get('visual_direction', 'Minimal expressive line drawings, sparse everyday props, clear emotional metaphors and generous negative space')}.
Background color {style.get('background_color', '#F4EBDD')}; ink color {style.get('ink_color', '#302A25')}; selective accent color {style.get('accent_color', '#C56545')}.
Depict this precise scene: {scene['visual_prompt']}
Narration context, for meaning only: {scene['narration']}
Compose a single readable scene. Keep the main character and meaningful action in the central safe area; leave the lower quarter quiet for captions. Keep recurring character proportions, ink weight and colors consistent across the series.
Draw original characters and imagery. Do not reproduce any channel logo, recognizable branded character, existing thumbnail or frame. No text, typography, letters, numerals, watermarks, UI or embedded subtitles. No photorealism or paper-cut collage unless expressly requested in the visual direction above."""


def _image_prompt(scene: dict, style: dict) -> str:
    base = _legacy_image_prompt(scene, style).replace(
        "No text, typography, letters, numerals, watermarks, UI or embedded subtitles.",
        "No readable text, typography, letters, numerals, watermarks or embedded subtitles. "
        "Information-bearing objects may contain simplified unlabeled diagrams or pictograms.")
    return base + "\n\nVisual logic requirements (apply to every depicted object and relationship):\n" + visual_logic_rules()


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
        # Illustration prompts reserve the lower quarter. Use that space for
        # closing beats when captions do not need it, keeping faces unobscured.
        top = (HEIGHT - 90 - len(lines) * 65
               if (scene.get("ending_role") or scene.get("_brand_role")) and not project.get("captions_enabled", False) else 188 if scene.get("_brand_role") else 56)
        draw.rounded_rectangle((left, top, right, top + 34 + len(lines) * 65), radius=18,
                               fill=style.get("background_color", "#F4EBDD"))
        draw.line((left + 24, top + 9, right - 24, top + 9), fill=style.get("accent_color", "#C56545"), width=5)
        for index, line in enumerate(lines):
            draw.text((WIDTH / 2, top + 30 + index * 65), line, font=font, anchor="mt",
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
    model = project.get("tts_model") or config.EXPLAINER_TTS_DEFAULT_MODEL
    voice = project.get("voice") or config.EXPLAINER_TTS_DEFAULT_VOICE
    provider_model = config.TTS_MODELS.get(model) or {}
    requires_labels = bool(provider_model.get("requires_speaker_labels"))
    inputs = {"version": CONTRACT_VERSION, "narration": scene["narration"], "model": model,
                 "voice": voice, "language": project.get("language"),
                 "provider": provider_model, "seed": config.TTS_RANDOM_SEED,
                 "endpoint": config.ORPHEUS_TTS_URL if model == "orpheus-en" else config.POCKET_TTS_URL if model == "pocket-tts-en" else None,
                 "provider_revision": config.POCKET_TTS_MODEL_REVISION if model == "pocket-tts-en" else None}
    if provider_model.get("kind") == "chatterbox_http":
        from backend.pipeline.chatterbox import cache_contract

        inputs["chatterbox_contract"] = cache_contract()
    if requires_labels:
        inputs["speaker_input_contract"] = 1
    key = _hash(inputs)
    stage = _stage(root, scene["id"], "audio", key)
    cached = _cached(stage, key, root)
    if cached:
        return stage / "narration.wav", cached
    if "chatterbox_contract" in inputs and project.get("language") == "en":
        # The added independent audit/tail checks need new evidence, not new
        # synthesis. Reuse only the exact preceding contract, with every other
        # synthesis, pronunciation and verifier input still identical.
        previous_contract = {k: v for k, v in inputs["chatterbox_contract"].items()
                             if k not in {"independent_audit_version", "waveform_tail_guard_version"}}
        previous_key = _hash({**inputs, "chatterbox_contract": previous_contract})
        previous = _safe(stage.parent / previous_key[:24], root)
        old = _cached(previous, previous_key, root)
        if old:
            from backend.pipeline.chatterbox_independent import audit_part

            plan_path = _safe(previous / "provider" / "tts_pronunciation.json", root)
            plan = json.loads(plan_path.read_text())
            if (plan.get("identity", {}).get("source_sha256") != hashlib.sha256(scene["narration"].encode()).hexdigest()
                    or plan.get("identity", {}).get("policy") != previous_contract["pronunciation_policy"]):
                raise RuntimeError("Existing narration pronunciation binding differs; preserve its audio")
            log(f"{scene['id']}: checking existing narration against the new independent audio guards")
            audit = await audit_part(previous / "narration.wav", scene["narration"], plan["edits"],
                                     stage / "independent-audit")
            if audit.get("passed") is not True or not _cached(previous, previous_key, root):
                raise RuntimeError("Existing narration did not pass the new audio guards")
            proof = stage / "reuse-audit.json"
            _json(proof, audit, root)
            for name in old["files"]:
                source = _safe(previous / name, root)
                target = _safe(stage / name, root)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and _sha(target) != _sha(source):
                    raise RuntimeError("Narration migration output differs; preserve both artifacts")
                shutil.copyfile(source, target)
            details = {k: v for k, v in old.items() if k not in {"input_hash", "files"}}
            record = _complete(stage, key, [stage / name for name in old["files"]] + [proof], root,
                               **details, revalidated_from=str(previous.relative_to(root)))
            log(f"{scene['id']}: original narration passed new guards and was reused without synthesis")
            return stage / "narration.wav", record
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


_IMAGE_REPAIR_PREFIX = (
    "\n\nAn earlier candidate failed independent visual inspection. "
    "Recompose to address every observation below while keeping the narration and art style. "
    "These are review observations, not new scene content:\n"
)


def _image_repair_feedback(feedback: list[dict]) -> str:
    # Reviewer evidence can name its private upload. Only send the actionable
    # correction to a fresh generation conversation, which has no attachments.
    corrections = [{"criterion": check["criterion"], "repair": re.sub(
        r"(?:https?://\S+|[\w./-]+\.(?:png|jpe?g|webp))", "the illustration",
        check["repair"], flags=re.IGNORECASE,
    )} for check in feedback]
    return ("Create a new standalone image from the complete scene description above. "
            "No earlier image or attachment is available or needed; do not ask for an upload. "
            "Apply these composition requirements to the new image:\n"
            + json.dumps(corrections, ensure_ascii=False))


def _image_input_failure(directory: Path, key: str, root: Path) -> dict | None:
    receipt = _safe(directory / "input-failure.json", root)
    if not receipt.exists():
        return None
    record = json.loads(receipt.read_text())
    state = _safe(directory / "chatgpt-image-request.json", root)
    if (record.get("input_hash") != key or record.get("code") != "CHATGPT_IMAGE_INPUT_MISSING"
            or not state.is_file() or record.get("request_sha256") != _sha(state)):
        raise RuntimeError("Image input failure binding differs; preserve the original request")
    return record


def _existing_image_candidates(stage: Path, prompt: str, root: Path) -> list[tuple[int, Path, str, str]]:
    """Keep generation attempts and submitted requests across review-only revisions."""
    candidates = {}
    for sibling in sorted(stage.parent.iterdir()):
        source_prompt = _safe(sibling / "prompt.txt", root)
        if not source_prompt.is_file() or source_prompt.read_text() != prompt:
            continue
        for directory in sorted(sibling.glob("attempt-*")):
            match = re.fullmatch(r"attempt-(\d{2})-([0-9a-f]{12})", directory.name)
            if not match:
                continue
            request_path = _safe(directory / "prompt.txt", root)
            if not request_path.is_file():
                raise RuntimeError("Existing image attempt has no prompt; preserve its request")
            request = request_path.read_text()
            attempt = int(match[1])
            key = _hash({"prompt": request, "attempt": attempt})
            if (not 1 <= attempt <= MAX_IMAGE_ATTEMPTS or key[:12] != match[2]
                    or not (request == prompt or request.startswith(prompt + _IMAGE_REPAIR_PREFIX))):
                raise RuntimeError("Existing image attempt inputs differ; preserve its request")
            if attempt in candidates:
                raise RuntimeError("Multiple image histories share an attempt; reconcile before generation")
            candidates[attempt] = (attempt, directory, request, key)
    if candidates and sorted(candidates) != list(range(1, max(candidates) + 1)):
        raise RuntimeError("Existing image attempt history has gaps; reconcile before generation")
    # Do not abandon an unknown outcome from an earlier attempt.
    for _, directory, _, key in list(sorted(candidates.values()))[:-1]:
        if not _cached(directory, key, root) and not _image_input_failure(directory, key, root):
            raise RuntimeError("Earlier image attempt is incomplete; recover its original request first")
    return list(sorted(candidates.values()))


async def _image(scene: dict, style: dict, root: Path, log: Log = lambda _: None) -> tuple[Path, dict]:
    prompt = _image_prompt(scene, style)
    key = _hash({"version": CONTRACT_VERSION, "prompt": prompt,
                 "review_version": REVIEW_VERSION, "review": review_prompt(scene, style)})
    stage = _stage(root, scene["id"], "image", key)
    # Reclaim interrupted browser ownership even on a cache hit.
    async with owned_media_session(stage, "chatgpt", recover_only=True):
        pass
    cached = _cached(stage, key, root)
    if cached:
        return stage / "image.png", cached
    _write(stage / "prompt.txt", prompt, root)

    async def accept(candidate: Path, conversation: str, provenance: dict) -> tuple[tuple[Path, dict] | None, dict]:
        report, report_path = await review_image(candidate, scene, style, candidate.parent, log)
        if report.get("passed") is not True:
            return None, report
        final = stage / "image.png"
        shutil.copyfile(candidate, stage / "image.partial.png")
        (stage / "image.partial.png").replace(final)
        # Keep the report with the accepted pixels; old provider originals stay untouched.
        proof = stage / "visual-review.json"
        shutil.copyfile(report_path, proof)
        log(f"{scene['id']}: illustration passed all visual logic checks")
        return (final, _complete(stage, key, [final, stage / "prompt.txt", proof], root,
                                provider="chatgpt_web_opencli", conversation_url=conversation,
                                visual_review_path=str(proof.relative_to(root)),
                                provenance=provenance)), report

    feedback = []
    existing = _existing_image_candidates(stage, prompt, root)
    legacy_key = _hash({"version": CONTRACT_VERSION, "prompt": _legacy_image_prompt(scene, style)})
    legacy_stage = _stage(root, scene["id"], "image", legacy_key)
    legacy = _cached(legacy_stage, legacy_key, root)
    if not existing and not legacy and (legacy_stage / "chatgpt-image-request.json").exists():
        # A prompt-policy upgrade must not abandon an already submitted turn.
        # The provider helper resumes the exact state-file request, never a new one.
        legacy_prompt = _legacy_image_prompt(scene, style)
        if (legacy_stage / "prompt.txt").read_text() != legacy_prompt:
            raise RuntimeError("Pending legacy image prompt differs; preserve the submitted request")
        log(f"{scene['id']}: recovering submitted legacy illustration before visual review")
        raw, conversation = await _generate_still(legacy_prompt, legacy_stage)
        _safe(raw, root)
        with Image.open(raw) as artwork:
            if artwork.width < 256 or artwork.height < 144:
                raise RuntimeError("Generated illustration is too small to render")
            artwork.convert("RGB").save(legacy_stage / "image.partial.png")
        (legacy_stage / "image.partial.png").replace(legacy_stage / "image.png")
        legacy = _complete(legacy_stage, legacy_key,
                           [legacy_stage / "image.png", legacy_stage / "prompt.txt"], root,
                           provider="chatgpt_web_opencli", conversation_url=conversation)
    if not existing and legacy:
        # Reuse only exact-input, hash-verified legacy artwork after the new pixel review.
        # Copy it for review so immutable provider evidence is never edited.
        candidate = stage / "legacy" / "image.png"
        candidate.parent.mkdir(exist_ok=True)
        shutil.copyfile(legacy_stage / "image.png", candidate)
        accepted, report = await accept(candidate, legacy.get("conversation_url", ""),
                                        {"kind": "reviewed_legacy", "source": str(legacy_stage.relative_to(root))})
        if accepted:
            return accepted
        feedback = [check for check in report["checks"] if check["verdict"] != "pass"]
        log(f"{scene['id']}: existing illustration rejected by visual logic review")

    # Re-review the latest exact-input candidate under the current review contract.
    # Older rejections stay immutable; a reviewer change never grants extra images.
    first_attempt = existing[-1][0] if existing else 1
    for attempt in range(first_attempt, MAX_IMAGE_ATTEMPTS + 1):
        request = prompt
        if feedback:
            request += _IMAGE_REPAIR_PREFIX + _image_repair_feedback(feedback)
        candidate_key = _hash({"prompt": request, "attempt": attempt})
        candidate_stage = _safe(stage / f"attempt-{attempt:02d}-{candidate_key[:12]}", root)
        if existing and attempt == first_attempt:
            _, candidate_stage, request, candidate_key = existing[-1]
            log(f"{scene['id']}: retaining illustration candidate {attempt}/{MAX_IMAGE_ATTEMPTS}")
        candidate_stage.mkdir(exist_ok=True)
        candidate = candidate_stage / "image.png"
        async with owned_media_session(candidate_stage, "chatgpt", recover_only=True):
            pass
        record = _cached(candidate_stage, candidate_key, root)
        if not record and (candidate_stage / "complete.json").exists():
            raise RuntimeError("Existing image candidate failed hash verification; preserve its evidence")
        if not record:
            failure = _image_input_failure(candidate_stage, candidate_key, root)
            if failure:
                feedback = failure["feedback"]
                continue
            log(f"{scene['id']}: generating illustration candidate {attempt}/{MAX_IMAGE_ATTEMPTS}")
            _write(candidate_stage / "prompt.txt", request, root)
            try:
                raw, conversation = await _generate_still(request, candidate_stage)
            except Exception as exc:
                # Only a confirmed, owned terminal missing-input response can
                # consume a candidate. Timeouts and unknown outcomes still stop.
                if "CHATGPT_IMAGE_INPUT_MISSING:" not in str(exc):
                    raise
                state_path = _safe(candidate_stage / "chatgpt-image-request.json", root)
                state = json.loads(state_path.read_text())
                if (state.get("prompt") != "Generate an image of: " + request
                        or not re.fullmatch(r"https://chatgpt\.com/c/[\w-]+", state.get("conversationUrl", ""))):
                    raise RuntimeError("Missing-input response is not bound to the owned image request") from exc
                # A legacy pending candidate may contain raw review JSON. Keep
                # that evidence unchanged and extract only its actionable repairs.
                if not feedback and request.startswith(prompt + _IMAGE_REPAIR_PREFIX):
                    suffix = request[len(prompt + _IMAGE_REPAIR_PREFIX):]
                    if suffix.startswith("["):
                        feedback = json.loads(suffix)
                _json(candidate_stage / "input-failure.json", {
                    "input_hash": candidate_key, "code": "CHATGPT_IMAGE_INPUT_MISSING",
                    "request_sha256": _sha(state_path), "error": str(exc), "feedback": feedback,
                }, root)
                log(f"{scene['id']}: candidate {attempt}/{MAX_IMAGE_ATTEMPTS} requires unavailable input; "
                    "terminal request retained, continuing within the same candidate budget")
                continue
            _safe(raw, root)
            with Image.open(raw) as artwork:
                if artwork.width < 256 or artwork.height < 144:
                    raise RuntimeError("Generated illustration is too small to render")
                artwork.convert("RGB").save(candidate_stage / "image.partial.png")
            (candidate_stage / "image.partial.png").replace(candidate)
            record = _complete(candidate_stage, candidate_key,
                               list(dict.fromkeys([candidate, raw, candidate_stage / "prompt.txt"])), root,
                               conversation_url=conversation)
        accepted, report = await accept(candidate, record.get("conversation_url", ""),
                                        {"kind": "generated", "candidate": str(candidate_stage.relative_to(root))})
        if accepted:
            return accepted
        feedback = [check for check in report["checks"] if check["verdict"] != "pass"]
        log(f"{scene['id']}: visual logic rejection: " + "; ".join(check["evidence"] for check in feedback))
    raise RuntimeError(f"{scene['id']}: visual logic failed after {MAX_IMAGE_ATTEMPTS} image candidates; "
                       "rejected images retained, scene not rendered")


def _brand_overlay(path: Path, role: str) -> None:
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    left = WIDTH - 710 if role == "outro" else 64
    draw.rounded_rectangle((left, 40, left + 646, 168), radius=12, fill=(10, 31, 48, 242))
    draw.line((left + 24, 62, left + 24, 144), fill="#C78C54", width=5)
    draw.text((left + 48, 56), "BYTEFRONT", font=_font(22, "en"), fill="#C78C54")
    draw.text((left + 44, 84), SERIES_NAME, font=_font(54, "en"), fill="#F4EBDD")
    overlay.save(path)


async def _thumbnail(project: dict, root: Path, log: Log) -> dict:
    prompt = thumbnail_prompt(project)
    key = _hash({"brand_version": BRAND_VERSION, "prompt": prompt})
    stage = _stage(root, "series", "thumbnail", key)
    async with owned_media_session(stage, "chatgpt", recover_only=True):
        pass
    record = _cached(stage, key, root)
    if not record:
        _write(stage / "prompt.txt", prompt, root)
        source, conversation = await _generate_still(prompt, stage)
        _safe(source, root)
        with Image.open(source) as artwork:
            artwork.load()
            if artwork.width < 1000 or abs(artwork.width / artwork.height - 16 / 9) > 0.08:
                raise ValueError("Byte By Byte thumbnail must be a large landscape 16:9 image")
            artwork.convert("RGB").save(stage / "thumbnail.png")
        record = _complete(stage, key, [stage / "thumbnail.png", stage / "prompt.txt"], root,
                           provider="chatgpt_web_opencli", conversation_url=conversation)
        log("Byte By Byte episode thumbnail is ready")
    return {"thumbnail_path": str((stage / "thumbnail.png").relative_to(root)),
            "thumbnail_sha256": record["files"]["thumbnail.png"],
            "thumbnail_prompt_path": str((stage / "prompt.txt").relative_to(root))}


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
    from backend.explainers import pending_gemini_wait
    wait = pending_gemini_wait()
    if wait:
        raise GeminiVideoDeferred(wait["reason"], wait["evidence"], scene["id"])
    try:
        raw, conversation = await _generate_video(prompt, keyframe, None, stage, LANDSCAPE)
    except Exception as exc:
        reason = availability_reason(exc)
        if reason:
            raise GeminiVideoDeferred(reason, str(exc), scene["id"]) from exc
        raise
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


async def _verified_speech_edge(scene: dict, path: Path, audio: dict, root: Path) -> dict | None:
    """Reuse a passed speech audit only after binding it to this exact WAV."""
    source_sha = hashlib.sha256(scene["narration"].encode()).hexdigest()
    path_sha = _sha(path)

    def accepted(report: dict, expected_audio_sha: str) -> bool:
        identity = report.get("identity") or {}
        tail = report.get("waveform_tail") or {}
        return (report.get("passed") is True and tail.get("passed") is True
                and identity.get("audio_sha256") == expected_audio_sha
                and identity.get("source_sha256") == source_sha
                and isinstance(tail.get("speech_end_seconds"), (int, float)))

    reused = _safe(path.parent / "reuse-audit.json", root)
    if reused.is_file() and audio.get("files", {}).get("reuse-audit.json") == _sha(reused):
        report = json.loads(reused.read_text())
        if accepted(report, path_sha):
            return {"speech_end_seconds": report["waveform_tail"]["speech_end_seconds"],
                    "audit_path": str(reused.relative_to(root)), "audit_sha256": _sha(reused),
                    "audio_sha256": path_sha, "binding": "exact_narration_waveform"}
    manifest_path = _safe(path.parent / "provider" / "tts_manifest.json", root)
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("chunk_count") != 1 or manifest.get("source_text_sha256") != source_sha:
        return None
    raw = _safe(manifest_path.parent / str(manifest.get("output", "")), root)
    raw_sha = manifest.get("output_audio_sha256")
    if not raw.is_file() or _sha(raw) != raw_sha:
        return None
    for proof in sorted((manifest_path.parent / "independent-production").rglob("audit.json")):
        _safe(proof, root)
        report = json.loads(proof.read_text())
        if not accepted(report, raw_sha):
            continue
        # Reproduce precisely the existing resample/pad operation. Matching
        # duration alone cannot establish that an audit belongs to this audio.
        pcm = await _command([
            "ffmpeg", "-v", "error", "-i", str(raw), "-af",
            f"aresample={SAMPLE_RATE},apad,atrim=end_sample={audio['frames'] * (SAMPLE_RATE // FPS)}",
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", "-f", "s16le", "-",
        ])
        import wave
        with wave.open(str(path), "rb") as source:
            if pcm != source.readframes(source.getnframes()):
                return None
        return {"speech_end_seconds": report["waveform_tail"]["speech_end_seconds"],
                "audit_path": str(proof.relative_to(root)), "audit_sha256": _sha(proof),
                "audio_sha256": path_sha, "provider_audio_sha256": raw_sha,
                "binding": "sample_identical_resample_and_padding"}
    return None


async def _paced_audio(scene: dict, path: Path, audio: dict, root: Path, *,
                       chapter_end: bool, closing_hold: float) -> tuple[Path, dict]:
    proof = await _verified_speech_edge(scene, path, audio, root)
    plan = await asyncio.to_thread(pacing_plan, path, chapter_end=chapter_end, closing_hold=closing_hold,
                                  verified_speech_end=proof["speech_end_seconds"] if proof else None)
    plan["speech_edge_evidence"] = proof
    plan["narration_duration_seconds"] = min(
        plan["narration_duration_seconds"],
        audio["narration_duration_seconds"] - plan["removed_leading_seconds"],
    )
    plan["speech_end_seconds"] = min(plan["speech_end_seconds"], plan["narration_duration_seconds"])
    key = _hash({"source_sha256": _sha(path), "source_audio_hash": audio["input_hash"], "pacing": plan})
    stage = _stage(root, scene["id"], "paced-audio", key)
    cached = _cached(stage, key, root)
    if cached:
        return stage / "narration.wav", cached
    temporary = stage / "narration.partial.wav"
    await asyncio.to_thread(write_paced_audio, path, temporary, plan)
    await _verify(temporary, duration=plan["duration_seconds"], video=False, audio=True)
    final = stage / "narration.wav"
    temporary.replace(final)
    return final, _complete(stage, key, [final], root,
                           source_path=str(path.relative_to(root)), source_sha256=_sha(path),
                           source_audio_hash=audio["input_hash"], pacing=plan,
                           narration_duration_seconds=plan["narration_duration_seconds"],
                           duration_seconds=plan["duration_seconds"], frames=plan["frames"])


async def _render_scene(scene: dict, project: dict, image: Path, motion: Path | None,
                        audio: dict, root: Path) -> tuple[Path, dict]:
    style = project.get("style") or {}
    motion_probe = await _probe(motion) if motion else None
    watermark_repair = None
    if motion and style.get("gemini_watermark_removal", True):
        stream = next(s for s in motion_probe["streams"] if s.get("codec_type") == "video")
        watermark_repair = repair_plan(stream["width"], stream["height"], style.get("gemini_watermark_region"))
    render_inputs = {"version": CONTRACT_VERSION, "image": _sha(image), "motion": _sha(motion) if motion else None,
                 "motion_playback_version": 2 if motion else 1,
                 "frames": audio["frames"], "mode": style.get("motion_mode", "none"),
                 "captions": project.get("captions_enabled", False), "text": scene["narration"],
                 "title": scene.get("on_screen_text"), "language": project.get("language"),
                 "background": style.get("background_color", "#F4EBDD"),
                 "ink": style.get("ink_color", "#302A25"),
                 "accent": style.get("accent_color", "#C56545")}
    if scene.get("ending_role") and not project.get("captions_enabled", False):
        render_inputs["closing_title_position"] = "lower_safe_area"
    if scene.get("_brand_role"):
        render_inputs["series_brand"] = {"version": BRAND_VERSION, "role": scene["_brand_role"]}
    closing_hold = (audio.get("pacing") or {}).get("closing_hold_seconds", 0)
    fade_seconds = min(0.8, closing_hold)
    if fade_seconds:
        render_inputs["closing_fade_seconds"] = fade_seconds
    if watermark_repair:
        render_inputs["watermark_repair"] = watermark_repair
    key = _hash(render_inputs)
    stage = _stage(root, scene["id"], "render", key)
    cached = _cached(stage, key, root)
    if cached:
        return stage / "clip.mp4", cached
    _overlay(scene, project, stage / "overlay.png")
    if scene.get("_brand_role"):
        _brand_overlay(stage / "brand.png", scene["_brand_role"])
    color = style.get("background_color", "#F4EBDD").replace("#", "0x")
    base = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color={color},setsar=1"
    if watermark_repair:
        # Repair at native resolution before scaling, padding, titles or captions.
        # Fold it into the existing encode; keep the downloaded motion untouched.
        base = watermark_repair["filter"] + "," + base
    args = ["ffmpeg", "-y", "-v", "error"]
    motion_playback = None
    if motion:
        args += ["-i", str(motion)]
        source_duration = _duration(motion_probe, "video")
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
    finish = "format=yuv420p"
    if fade_seconds:
        finish += f",fade=t=out:st={audio['duration_seconds'] - fade_seconds:.9f}:d={fade_seconds}:color={color}"
    args += ["-loop", "1", "-framerate", str(FPS), "-i", str(stage / "overlay.png")]
    filters = f"[0:v]{base}[base];[base][1:v]overlay=0:0:shortest=1[text]"
    if scene.get("_brand_role"):
        args += ["-loop", "1", "-framerate", str(FPS), "-i", str(stage / "brand.png")]
        enable = ":enable='lt(t,2.2)'" if scene["_brand_role"] == "intro" else ""
        filters += f";[text][2:v]overlay=0:0:shortest=1{enable}[branded];[branded]{finish}[v]"
    else:
        filters += f";[text]{finish}[v]"
    args += ["-filter_complex", filters,
             "-map", "[v]", "-an", "-frames:v", str(audio["frames"]), "-r", str(FPS),
             "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-threads", "4",
             "-movflags", "+faststart", str(stage / "clip.partial.mp4")]
    await _command(args, timeout=max(180, audio["duration_seconds"] * 30))
    await _verify(stage / "clip.partial.mp4", duration=audio["duration_seconds"], video=True, audio=False)
    final = stage / "clip.mp4"
    (stage / "clip.partial.mp4").replace(final)
    return final, _complete(stage, key, [final, stage / "overlay.png"], root,
                            motion_playback=motion_playback, watermark_repair=watermark_repair)


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
    validate_motion_budget(scenes, style)
    if style.get("motion_mode") == "gemini":
        every = max(0, int(style.get("motion_every_n_scenes", 0)))
        if not any(s.get("motion") or (every and (i + 1) % every == 0) for i, s in enumerate(scenes)):
            raise ValueError("Gemini animation is enabled but no scenes are selected; select a motion scene or set an interval")
    return scenes


async def render_project(project: dict, project_dir: Path,
                         update: Callable[[dict], None], log: Log) -> dict:
    root = project_dir.absolute()
    scenes = _validate(project, root)
    root.mkdir(parents=True, exist_ok=True)
    style = project.get("style") or {}
    total = len(scenes) * 4 + 2
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
        progress("generating_images", "Generating or reusing the Byte By Byte episode thumbnail")
        thumbnail = await _thumbnail(project, root, log)
        update({"artifacts": thumbnail})
        completed += 1
        selected_motion = set(motion_indices(scenes, style))
        for index, saved_scene in enumerate(scenes):
            scene = {**saved_scene, "_brand_role": "intro" if index == 0 else "outro" if index == len(scenes) - 1 else ""}
            prefix = f"Scene {index + 1}/{len(scenes)}"
            progress("generating_audio", f"{prefix}: generating or reusing exact narration")
            source_audio_path, source_audio = await _audio(scene, project, root, log)
            last_scene = index == len(scenes) - 1
            chapter_end = last_scene or scene.get("chapter", "") != scenes[index + 1].get("chapter", "")
            audio_path, audio = await _paced_audio(
                scene, source_audio_path, source_audio, root, chapter_end=chapter_end,
                closing_hold=float(style.get("closing_hold_seconds", 2.4)) if last_scene else 0,
            )
            completed += 1
            progress("generating_images", f"{prefix}: generating or reusing ChatGPT illustration")
            image_path, illustration = await _image(scene, style, root, log)
            completed += 1
            motion_path, motion = None, None
            if index in selected_motion:
                progress("generating_motion", f"{prefix}: generating or reusing Gemini motion")
                motion_path, motion = await _motion(scene, style, image_path, root)
            completed += 1
            progress("rendering", f"{prefix}: composing measured narration and illustration")
            if motion_path and style.get("gemini_watermark_removal", True):
                log(f"{prefix}: repairing visible Gemini watermark locally before titles and captions")
            clip, render = await _render_scene(scene, project, image_path, motion_path, audio, root)
            completed += 1
            item = {"id": scene["id"], "chapter": scene.get("chapter", ""), "narration": scene["narration"],
                    "on_screen_text": scene.get("on_screen_text", ""),
                    "start_frame": current_frame, "end_frame": current_frame + audio["frames"],
                    "start_seconds": current_frame / FPS, "duration_seconds": audio["duration_seconds"],
                    "narration_duration_seconds": audio["narration_duration_seconds"],
                    "audio_path": relative(audio_path), "image_path": relative(image_path),
                    "source_audio_path": relative(source_audio_path), "source_audio_sha256": _sha(source_audio_path),
                    "pacing": audio["pacing"],
                    "motion_path": relative(motion_path) if motion_path else None, "clip_path": relative(clip),
                    "audio_sha256": _sha(audio_path), "clip_sha256": _sha(clip),
                    "audio_hash": audio["input_hash"], "image_hash": illustration["input_hash"],
                    "motion_hash": motion["input_hash"] if motion else None, "render_hash": render["input_hash"],
                    "motion_playback": render.get("motion_playback"),
                    "image_source_url": illustration.get("conversation_url"),
                    "image_visual_review_path": illustration["visual_review_path"],
                    "motion_source_url": motion.get("conversation_url") if motion else None}
            if render.get("watermark_repair"):
                item["watermark_repair"] = render["watermark_repair"]
            artifacts.append(item)
            current_frame += audio["frames"]
            save_manifest("rendering")
            update({"scene_artifacts": artifacts, "progress": {"completed": completed, "total": total, "message": f"{prefix}: ready"}})
        duration = current_frame / FPS
        timeline = {"version": CONTRACT_VERSION, "fps": FPS, "duration_seconds": duration,
                    "total_frames": current_frame, "caption_alignment": "scene_narration",
                    "pacing_policy": "sample_preserving_edge_trim_with_closing_hold",
                    "gemini_scene_count": sum(bool(item["motion_path"]) for item in artifacts),
                    "playback_policy": "play_once_fit_longer_or_hold_shorter", "scenes": artifacts}
        _json(root / "timeline.json", timeline, root)
        captions = []
        for index, item in enumerate(artifacts, 1):
            start = item["start_seconds"] + item["pacing"]["speech_start_seconds"]
            end = item["start_seconds"] + item["pacing"]["speech_end_seconds"]
            text = re.sub(r"\s+", " ", item["narration"]).strip()
            captions.append(f"{index}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
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
        result = {**thumbnail, "series_name": SERIES_NAME, "video_path": relative(delivery / "video.mp4"), "audio_path": relative(delivery / "narration.wav"),
                  "timeline_path": "timeline.json", "captions_path": "captions.srt", "manifest_path": "manifest.json",
                  "duration_seconds": duration, "scene_artifacts": artifacts}
        completed = total
        save_manifest("completed")
        update({"progress": {"completed": completed, "total": total, "message": "Video and narration verified"}})
        return result
    except GeminiVideoDeferred as exc:
        save_manifest("waiting_provider", str(exc))
        raise
    except asyncio.CancelledError:
        save_manifest("cancelled")
        raise
    except Exception as exc:
        save_manifest("failed", str(exc))
        raise
