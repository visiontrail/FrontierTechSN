from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import config
from backend.pipeline.opencli import OpenCLIError, run_opencli

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]
MEDIA_SUFFIXES = {".mp4", ".m4a", ".mp3", ".wav", ".webm"}


@dataclass(frozen=True)
class MusicArtifact:
    provider: str
    prompt: str
    conversation_url: str
    original_path: str
    audio_path: str
    manifest_path: str
    fallback_reason: str = ""


def _log(log: LogCallback | None, message: str) -> None:
    if log:
        log(message)
    else:
        logger.info(message)


async def _media_command(args: Sequence[str], timeout: int = 300) -> tuple[str, str]:
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
    out = stdout.decode(errors="replace")
    err = stderr.decode(errors="replace")
    if process.returncode:
        raise RuntimeError(f"Media command failed ({process.returncode}): {(err or out)[-1600:]}")
    return out, err


def build_music_prompt(title: str, summary: dict | None) -> str:
    talking_points = (summary or {}).get("talking_points") or []
    categories = []
    headlines = []
    for item in talking_points[:6]:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip()
        headline = str(item.get("headline") or "").strip()
        if category and category not in categories:
            categories.append(category)
        if headline:
            headlines.append(headline)
    palette = ", ".join(categories) or "AI, engineering and frontier science"
    context = "; ".join(headlines[:3]) or title
    return (
        "Create an instrumental 60-second seamless-loop morning technology news podcast bed. "
        "Restrained analog pulse, warm marimba, subtle glass textures, soft bass, 92 BPM; "
        "optimistic, precise and serious rather than cinematic. No vocals, speech, slogans, "
        "dramatic drops, sirens or dominant lead melody. Keep the midrange sparse so narration "
        "remains intelligible. The editorial palette is "
        f"{palette}. Context: {context}. End on a loop-compatible cadence."
    )


async def _browser(
    session: str,
    args: list[str],
    *,
    timeout: int = 90,
    check: bool = True,
):
    return await run_opencli(
        ["browser", session, *args, "--window", "background"],
        timeout=timeout,
        check=check,
    )


def _downloads_snapshot() -> dict[Path, tuple[int, int]]:
    directory = Path.home() / "Downloads"
    if not directory.is_dir():
        return {}
    return {
        path.resolve(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES
    }


async def _wait_for_new_download(before: dict[Path, tuple[int, int]], timeout: int) -> Path:
    deadline = time.monotonic() + timeout
    last: tuple[Path, int] | None = None
    stable = 0
    while time.monotonic() < deadline:
        current = _downloads_snapshot()
        changed = [
            path
            for path, stat in current.items()
            if path not in before or before[path] != stat
        ]
        if changed:
            candidate = max(changed, key=lambda path: path.stat().st_mtime_ns)
            size = candidate.stat().st_size
            if last == (candidate, size) and size > 4096:
                stable += 1
            else:
                last = (candidate, size)
                stable = 0
            if stable >= 2:
                return candidate
        await asyncio.sleep(2)
    raise TimeoutError(f"Gemini music download did not finish within {timeout}s")


async def _gemini_create_music_once(
    prompt: str,
    music_dir: Path,
    session: str,
    *,
    timeout: int,
) -> tuple[Path, str]:
    before = _downloads_snapshot()
    await _browser(session, ["open", "https://gemini.google.com/app"], timeout=60)
    await _browser(session, ["wait", "time", "2", "--timeout", "10000"], timeout=20)
    await _browser(
        session,
        ["click", "--role", "button", "--name", "Upload & tools"],
        timeout=30,
    )
    await _browser(
        session,
        ["click", "--role", "menuitemcheckbox", "--name", "Music"],
        timeout=30,
    )
    await _browser(
        session,
        [
            "fill",
            "--role",
            "textbox",
            "--name",
            "Enter a prompt for Gemini",
            prompt,
        ],
        timeout=30,
    )
    await _browser(
        session,
        ["click", "--role", "button", "--name", "Send message"],
        timeout=30,
    )
    deadline = time.monotonic() + timeout
    state = ""
    while time.monotonic() < deadline:
        await asyncio.sleep(8)
        result = await _browser(session, ["state"], timeout=45, check=False)
        state = f"{result.stdout}\n{result.stderr}"
        if "Play music" in state and "Generating your track" not in state:
            break
    else:
        raise TimeoutError(f"Gemini Create Music did not finish within {timeout}s")

    url_result = await _browser(session, ["get", "url"], timeout=30)
    conversation_url = url_result.stdout.strip().splitlines()[-1].strip().strip('"')
    script = (
        "(() => { const media=document.querySelector('audio,video'); "
        "if(!media) return 'no-media'; const a=document.createElement('a'); "
        "a.href=media.currentSrc||media.src; a.download='frontier-tech-music.mp4'; "
        "document.body.appendChild(a); a.click(); a.remove(); return 'download-triggered'; })()"
    )
    result = await _browser(session, ["eval", script], timeout=45)
    if "download-triggered" not in result.stdout:
        raise OpenCLIError(f"Gemini music media could not be downloaded: {result.stdout[-500:]}")
    downloaded = await _wait_for_new_download(before, min(180, timeout))
    original = music_dir / f"gemini-original{downloaded.suffix.lower()}"
    shutil.copyfile(downloaded, original)
    return original, conversation_url


async def _local_music(music_dir: Path, duration: float = 60.0) -> Path:
    output = music_dir / "local-newsroom-bed.wav"
    # Harmonic pulses plus a sparse bell envelope make a deterministic musical
    # bed, not white-noise filler. The expression is intentionally quiet and
    # leaves the voice-frequency band mostly open.
    expression = (
        "0.055*sin(2*PI*110*t)+"
        "0.028*sin(2*PI*220*t)*(0.35+0.65*pow(sin(PI*1.533*t)\\,8))+"
        "0.018*sin(2*PI*440*t)*(0.25+0.75*pow(sin(PI*0.383*t)\\,18))+"
        "0.012*sin(2*PI*659.25*t)*(0.20+0.80*pow(sin(PI*0.1916*t)\\,24))"
    )
    await _media_command(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"aevalsrc={expression}:s=48000:d={duration}",
            "-af",
            "highpass=f=70,lowpass=f=5000,aecho=0.8:0.7:120|240:0.10|0.06,alimiter=limit=0.65",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        timeout=180,
    )
    return output


async def generate_background_music(
    task_id: str,
    task_dir: Path,
    *,
    title: str,
    summary: dict | None,
    provider: str,
    log: LogCallback | None = None,
) -> MusicArtifact:
    music_dir = task_dir / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_music_prompt(title, summary)
    (music_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    provider_used = provider
    conversation_url = ""
    fallback_reason = ""
    original: Path
    if provider == "gemini_create_music":
        session_base = f"ftsn-music-{re.sub(r'[^a-zA-Z0-9-]', '-', task_id)[:32]}"
        active_session = ""
        successful_session = ""
        last_error: Exception | None = None
        for attempt in range(1, config.OPENCLI_MAX_ATTEMPTS + 1):
            # A killed Browser Bridge command can leave its tab lease blocked in
            # the daemon.  Reusing that session would turn ten configured
            # retries into ten waits behind the same stale lease, so every
            # attempt gets an independent session and cleanup pass.
            active_session = f"{session_base}-{attempt}"
            try:
                _log(log, f"Gemini Create Music attempt {attempt}/{config.OPENCLI_MAX_ATTEMPTS}")
                original, conversation_url = await _gemini_create_music_once(
                    prompt,
                    music_dir,
                    active_session,
                    timeout=max(240, config.OPENCLI_TIMEOUT * 2),
                )
                successful_session = active_session
                break
            except Exception as exc:  # noqa: BLE001 - deterministic fallback below
                last_error = exc
                _log(log, f"Gemini Create Music attempt {attempt} failed: {exc}")
                await _browser(active_session, ["close"], timeout=30, check=False)
                if attempt < config.OPENCLI_MAX_ATTEMPTS:
                    await asyncio.sleep(min(45, 3 * 2 ** (attempt - 1)))
        else:
            fallback_reason = f"Gemini Create Music exhausted retries: {last_error}"
            provider_used = "local_deterministic"
            original = await _local_music(music_dir)
        if successful_session:
            await _browser(successful_session, ["close"], timeout=30, check=False)
    else:
        provider_used = "local_deterministic"
        original = await _local_music(music_dir)

    audio_path = music_dir / "background.wav"
    if original.suffix.lower() == ".wav" and original != audio_path:
        shutil.copyfile(original, audio_path)
    else:
        await _media_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(original),
                "-vn",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                str(audio_path),
            ],
            timeout=180,
        )
    if not audio_path.is_file() or audio_path.stat().st_size < 4096:
        raise RuntimeError("Background-music generation produced no usable audio")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider_requested": provider,
        "provider_used": provider_used,
        "prompt": prompt,
        "conversation_url": conversation_url,
        "original_path": str(original.resolve()),
        "audio_path": str(audio_path.resolve()),
        "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "fallback_reason": fallback_reason,
    }
    manifest_path = music_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(log, f"Background music ready via {provider_used}: {audio_path.name}")
    return MusicArtifact(
        provider_used,
        prompt,
        conversation_url,
        str(original),
        str(audio_path),
        str(manifest_path),
        fallback_reason,
    )


async def _probe_duration(path: Path) -> float:
    stdout, _ = await _media_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        timeout=60,
    )
    return float(stdout.strip())


async def _integrated_loudness(path: Path) -> float | None:
    _, stderr = await _media_command(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
        timeout=180,
    )
    matches = re.findall(r"I:\s*(-?[0-9.]+) LUFS", stderr)
    return float(matches[-1]) if matches else None


async def mix_narration_and_music(
    narration_path: str | Path,
    music_path: str | Path,
    output_dir: Path,
    *,
    bed_db: float,
    duck_db: float,
    log: LogCallback | None = None,
) -> Path:
    narration = Path(narration_path)
    music = Path(music_path)
    mix_dir = output_dir / "audio"
    mix_dir.mkdir(parents=True, exist_ok=True)
    output = mix_dir / "program_mix.wav"
    duration = await _probe_duration(narration)
    intro_db = min(-16.0, bed_db + 7.0)
    ratio = max(4.0, min(20.0, abs(duck_db) * 1.2))
    volume_expression = (
        f"if(lt(t,12),pow(10\\,{intro_db}/20),pow(10\\,{bed_db}/20))"
    )
    filter_graph = (
        f"[1:a]aloop=loop=-1:size=2147483647,atrim=0:{duration:.6f},"
        f"asetpts=N/SR/TB,volume='{volume_expression}':eval=frame[music];"
        f"[music][0:a]sidechaincompress=threshold=0.018:ratio={ratio:.2f}:"
        "attack=18:release=650:makeup=1[ducked];"
        "[0:a][ducked]amix=inputs=2:duration=first:dropout_transition=0,"
        "alimiter=limit=0.92[out]"
    )
    await _media_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(narration),
            "-i",
            str(music),
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        timeout=max(300, math.ceil(duration * 2)),
    )
    mixed_duration = await _probe_duration(output)
    if abs(mixed_duration - duration) > 0.25:
        raise RuntimeError(
            f"Program mix duration drifted from narration: {mixed_duration:.3f}s vs {duration:.3f}s"
        )
    narration_lufs, music_lufs, mix_lufs = await asyncio.gather(
        _integrated_loudness(narration),
        _integrated_loudness(music),
        _integrated_loudness(output),
    )
    report = {
        "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "narration_path": str(narration.resolve()),
        "music_path": str(music.resolve()),
        "program_mix_path": str(output.resolve()),
        "narration_duration_seconds": duration,
        "program_mix_duration_seconds": mixed_duration,
        "intro_music_db": intro_db,
        "content_music_db": bed_db,
        "sidechain_duck_target_db": duck_db,
        "sidechain_ratio": ratio,
        "narration_integrated_lufs": narration_lufs,
        "source_music_integrated_lufs": music_lufs,
        "program_mix_integrated_lufs": mix_lufs,
        "program_mix_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (mix_dir / "music_mix_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _log(
        log,
        f"Music mix passed: intro {intro_db:.1f} dB, news bed {bed_db:.1f} dB, "
        f"sidechain duck target {duck_db:.1f} dB",
    )
    return output
