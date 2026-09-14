from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import shutil
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import config
from backend.pipeline.music_loop import render_music_bed
from backend.pipeline.opencli import OpenCLIError, run_opencli

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]
MEDIA_SUFFIXES = {".mp4", ".m4a", ".mp3", ".wav", ".webm"}
PROGRAM_MUSIC_DIR = config.PROJECT_ROOT / "data" / "program_music"
PROGRAM_MUSIC_CATALOG = PROGRAM_MUSIC_DIR / "catalog.json"
DEFAULT_PROGRAM_MUSIC_TRACK_ID = "morning-blueprint"


@dataclass(frozen=True)
class MusicArtifact:
    provider: str
    prompt: str
    conversation_url: str
    original_path: str
    audio_path: str
    manifest_path: str
    fallback_reason: str = ""


def _program_music_catalog() -> tuple[str, list[dict[str, Any]]]:
    """Load and verify the operator-managed local program-music catalog."""
    if not PROGRAM_MUSIC_CATALOG.is_file():
        return DEFAULT_PROGRAM_MUSIC_TRACK_ID, []
    try:
        payload = json.loads(PROGRAM_MUSIC_CATALOG.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Program-music catalog is unreadable: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("tracks"), list):
        raise RuntimeError("Program-music catalog must contain a tracks array")

    root = PROGRAM_MUSIC_DIR.resolve()
    default_id = str(payload.get("default_track_id") or DEFAULT_PROGRAM_MUSIC_TRACK_ID)
    tracks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in payload["tracks"]:
        if not isinstance(raw, dict):
            raise RuntimeError("Program-music catalog entries must be objects")
        track_id = str(raw.get("id") or "").strip()
        filename = str(raw.get("filename") or "").strip()
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", track_id):
            raise RuntimeError(f"Invalid program-music track id: {track_id!r}")
        if track_id in seen:
            raise RuntimeError(f"Duplicate program-music track id: {track_id}")
        seen.add(track_id)
        path = (PROGRAM_MUSIC_DIR / filename).resolve()
        if path.parent != root or path.suffix.lower() not in MEDIA_SUFFIXES:
            raise RuntimeError(f"Unsafe program-music filename for {track_id}: {filename!r}")
        if not path.is_file() or path.stat().st_size < 4096:
            raise RuntimeError(f"Program-music file is missing or empty: {path}")
        expected_hash = str(raw.get("sha256") or "").strip().lower()
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_hash and expected_hash != actual_hash:
            raise RuntimeError(f"Program-music checksum mismatch for {track_id}")
        tracks.append(
            {
                "id": track_id,
                "title": str(raw.get("title") or track_id).strip(),
                "description": str(raw.get("description") or "").strip(),
                "source": str(raw.get("source") or "local").strip(),
                "duration_seconds": float(raw.get("duration_seconds") or 0),
                "sha256": actual_hash,
                "filename": filename,
                "path": path,
                "is_default": track_id == default_id,
            }
        )
    if tracks and default_id not in seen:
        raise RuntimeError(f"Program-music default track is absent: {default_id}")
    return default_id, tracks


def list_program_music_tracks() -> list[dict[str, Any]]:
    """Return browser-safe metadata for every locally available music bed."""
    _, tracks = _program_music_catalog()
    return [
        {key: value for key, value in track.items() if key not in {"path", "filename"}}
        for track in tracks
    ]


def resolve_program_music_track(track_id: str | None) -> dict[str, Any]:
    default_id, tracks = _program_music_catalog()
    selected = (track_id or default_id).strip()
    for track in tracks:
        if track["id"] == selected:
            return track
    available = ", ".join(track["id"] for track in tracks) or "none"
    raise ValueError(f"Unknown program-music track {selected!r}; available: {available}")


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


async def _gemini_dom_action(
    session: str,
    script: str,
    expected: str,
    *,
    timeout: int = 30,
) -> str:
    """Run one Gemini DOM action when OpenCLI's semantic tree is unavailable."""
    result = await _browser(session, ["eval", script], timeout=timeout)
    output = result.stdout.strip()
    if expected not in output:
        raise OpenCLIError(
            f"Gemini DOM action did not report {expected!r}: {output[-500:]}"
        )
    return output


async def _enable_gemini_music_tool(session: str) -> None:
    await _gemini_dom_action(
        session,
        """(() => {
          const button = document.querySelector('button[aria-label="Upload & tools"]');
          if (!button) return 'missing-upload-tools';
          button.click();
          return 'clicked-upload-tools';
        })()""",
        "clicked-upload-tools",
    )
    await _browser(session, ["wait", "time", "1", "--timeout", "10000"], timeout=20)

    # Gemini moved Music beneath a second-level "More tools" menu in August
    # 2026. Keep the old direct-menu shape working when that button is absent.
    result = await _browser(
        session,
        [
            "eval",
            """(() => {
              const buttons = [...document.querySelectorAll('button')];
              const more = buttons.find((button) =>
                button.getAttribute('aria-label') === 'More tools' ||
                (button.innerText || '').trim() === 'More tools'
              );
              if (!more) return 'no-more-tools';
              more.click();
              return 'opened-more-tools';
            })()""",
        ],
        timeout=30,
    )
    if "opened-more-tools" in result.stdout:
        await _browser(session, ["wait", "time", "1", "--timeout", "10000"], timeout=20)
    elif "no-more-tools" not in result.stdout:
        raise OpenCLIError(
            f"Gemini More tools action returned an unexpected result: {result.stdout[-500:]}"
        )

    await _gemini_dom_action(
        session,
        """(() => {
          const items = [...document.querySelectorAll('[role="menuitemcheckbox"]')];
          const music = items.find((item) => {
            const text = (item.innerText || item.textContent || '').trim();
            return text === 'Create music' || text === 'Music';
          });
          if (!music) return 'missing-music-tool';
          music.click();
          return 'selected-music-tool';
        })()""",
        "selected-music-tool",
    )


async def _fill_gemini_prompt(session: str, prompt: str) -> None:
    prompt_json = json.dumps(prompt, ensure_ascii=False)
    await _gemini_dom_action(
        session,
        f"""(() => {{
          const editor = document.querySelector(
            '[role="textbox"][aria-label="Enter a prompt for Gemini"]'
          );
          if (!editor) return 'missing-prompt-editor';
          const prompt = {prompt_json};
          editor.focus();
          editor.textContent = prompt;
          editor.dispatchEvent(new InputEvent('input', {{
            bubbles: true,
            inputType: 'insertText',
            data: prompt,
          }}));
          return editor.innerText === prompt ? 'filled-prompt' : 'prompt-fill-mismatch';
        }})()""",
        "filled-prompt",
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
    await _enable_gemini_music_tool(session)
    await _fill_gemini_prompt(session, prompt)
    await _gemini_dom_action(
        session,
        """(() => {
          const button = document.querySelector('button[aria-label="Send message"]');
          if (!button) return 'missing-send-message';
          button.click();
          return 'sent-message';
        })()""",
        "sent-message",
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
    track_id: str | None = None,
    log: LogCallback | None = None,
) -> MusicArtifact:
    music_dir = task_dir / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_music_prompt(title, summary)
    (music_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    provider_used = provider
    conversation_url = ""
    fallback_reason = ""
    selected_track: dict[str, Any] | None = None
    original: Path
    if provider == "local_library":
        selected_track = resolve_program_music_track(track_id)
        provider_used = "local_library"
        original = selected_track["path"]
        prompt = f"Local program-music selection: {selected_track['title']}"
        (music_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        _log(
            log,
            "Program music: using local library track "
            f"{selected_track['title']} ({selected_track['id']}); Gemini generation bypassed",
        )
    elif provider == "gemini_create_music":
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
        "selected_track_id": selected_track["id"] if selected_track else None,
        "selected_track_title": selected_track["title"] if selected_track else None,
        "selected_track_source_sha256": selected_track["sha256"] if selected_track else None,
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


async def _integrated_loudness(path: Path, *, end: float | None = None) -> float | None:
    filters = f"atrim=end={end:.6f},ebur128" if end is not None else "ebur128"
    _, stderr = await _media_command(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", filters, "-f", "null", "-"],
        timeout=180,
    )
    matches = re.findall(r"I:\s*(-?[0-9.]+) LUFS", stderr)
    return float(matches[-1]) if matches else None


async def _mean_volume(path: Path, start: float, end: float) -> float | None:
    duration = max(0.0, end - start)
    if duration < 0.2:
        return None
    _, stderr = await _media_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        timeout=max(60, math.ceil(duration * 3)),
    )
    match = re.findall(r"mean_volume:\s*(-?[0-9.]+) dB", stderr)
    return float(match[-1]) if match else None


async def create_paced_narration(
    narration_path: str | Path,
    output_dir: Path,
    aligned_segments: list[dict[str, Any]],
    *,
    intro_seconds: float = 2.0,
    opening_gap_seconds: float = 3.0,
    story_gap_seconds: float = 1.5,
    log: LogCallback | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Insert deterministic full-music windows around daily-news narration.

    ``aligned_segments`` contains one timing row per physical script line: the
    opening, each story, and the closing. The output keeps every source sample,
    adding silence only at the requested program boundaries.
    """
    source = Path(narration_path)
    if not source.is_file():
        raise FileNotFoundError(f"Narration is missing: {source}")
    if not aligned_segments:
        raise RuntimeError("Program pacing requires at least one aligned script segment")

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    output = audio_dir / "paced_narration.wav"
    report_path = audio_dir / "program_pacing_report.json"
    source_duration = await _probe_duration(source)
    intro = max(0.0, float(intro_seconds))
    opening_gap = max(0.0, float(opening_gap_seconds))
    story_gap = max(0.0, float(story_gap_seconds))

    filter_parts: list[str] = []
    concat_labels: list[str] = []
    if intro:
        filter_parts.append(f"anullsrc=r=48000:cl=stereo:d={intro:.6f}[intro]")
        concat_labels.append("[intro]")

    program_cursor = intro
    source_cursor = 0.0
    segments: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for index, segment in enumerate(aligned_segments):
        proposed_end = float(segment.get("start") or 0) + float(segment.get("duration") or 0)
        source_end = source_duration if index == len(aligned_segments) - 1 else proposed_end
        source_end = min(source_duration, max(source_cursor + 0.001, source_end))
        label = f"segment{index}"
        filter_parts.append(
            f"[0:a]atrim=start={source_cursor:.6f}:end={source_end:.6f},"
            "asetpts=PTS-STARTPTS,aresample=48000,"
            f"aformat=sample_fmts=s16:channel_layouts=stereo[{label}]"
        )
        concat_labels.append(f"[{label}]")
        segment_duration = source_end - source_cursor
        kind = (
            "opening"
            if index == 0
            else "closing" if index == len(aligned_segments) - 1 else "news"
        )
        segment_report = {
            "index": index,
            "kind": kind,
            "text": str(segment.get("text") or ""),
            "source_start": round(source_cursor, 3),
            "source_end": round(source_end, 3),
            "program_start": round(program_cursor, 3),
            "program_end": round(program_cursor + segment_duration, 3),
        }
        segments.append(segment_report)
        program_cursor += segment_duration
        source_cursor = source_end

        gap_duration = 0.0
        gap_kind = ""
        if index == 0 and len(aligned_segments) > 1:
            gap_duration = opening_gap
            gap_kind = "after_opening"
        elif 0 < index < len(aligned_segments) - 1:
            gap_duration = story_gap
            gap_kind = "after_news"
        if gap_duration:
            gap_label = f"gap{index}"
            filter_parts.append(
                f"anullsrc=r=48000:cl=stereo:d={gap_duration:.6f}[{gap_label}]"
            )
            concat_labels.append(f"[{gap_label}]")
            gaps.append(
                {
                    "after_segment_index": index,
                    "kind": gap_kind,
                    "source_boundary": round(source_end, 3),
                    "program_start": round(program_cursor, 3),
                    "program_end": round(program_cursor + gap_duration, 3),
                    "duration_seconds": round(gap_duration, 3),
                }
            )
            program_cursor += gap_duration

    filter_parts.append(
        "".join(concat_labels)
        + f"concat=n={len(concat_labels)}:v=0:a=1[out]"
    )
    await _media_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            ";".join(filter_parts),
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
        timeout=max(300, math.ceil(source_duration * 2)),
    )
    actual_duration = await _probe_duration(output)
    expected_duration = source_duration + intro + sum(
        float(gap["duration_seconds"]) for gap in gaps
    )
    drift = abs(actual_duration - expected_duration)
    report = {
        "passed": drift <= 0.15,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_narration_path": str(source.resolve()),
        "paced_narration_path": str(output.resolve()),
        "source_duration_seconds": round(source_duration, 3),
        "paced_duration_seconds": round(actual_duration, 3),
        "expected_duration_seconds": round(expected_duration, 3),
        "duration_drift_seconds": round(drift, 4),
        "intro_seconds": round(intro, 3),
        "opening_gap_seconds": round(opening_gap, 3),
        "story_gap_seconds": round(story_gap, 3),
        "segments": segments,
        "gaps": gaps,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(
            "Program narration pacing drifted from its deterministic timeline: "
            f"{actual_duration:.3f}s vs {expected_duration:.3f}s"
        )
    _log(
        log,
        "Program pacing ready: "
        f"{intro:.1f}s music intro, {opening_gap:.1f}s after opening, "
        f"{story_gap:.1f}s after each news segment",
    )
    return output, report


def shift_word_transcript_for_pacing(
    word_transcript: list[dict[str, Any]],
    pacing_report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Shift raw word timestamps onto the silence-extended program clock."""
    intro = float(pacing_report.get("intro_seconds") or 0)
    gaps = list(pacing_report.get("gaps") or [])
    shifted: list[dict[str, Any]] = []
    for word in word_transcript:
        try:
            start = float(word["start"])
            end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            shifted.append(dict(word))
            continue
        midpoint = (start + end) / 2
        offset = intro + sum(
            float(gap.get("duration_seconds") or 0)
            for gap in gaps
            if midpoint >= float(gap.get("source_boundary") or 0)
        )
        row = dict(word)
        row["start"] = round(start + offset, 4)
        row["end"] = round(end + offset, 4)
        shifted.append(row)
    return shifted


async def mix_narration_and_music(
    narration_path: str | Path,
    music_path: str | Path,
    output_dir: Path,
    *,
    bed_db: float,
    duck_db: float,
    log: LogCallback | None = None,
    original_narration_path: str | Path | None = None,
) -> Path:
    narration = Path(narration_path)
    music = Path(music_path)
    mix_dir = output_dir / "audio"
    mix_dir.mkdir(parents=True, exist_ok=True)
    output = mix_dir / "program_mix.wav"
    music_probe = mix_dir / ".music_bed_probe.wav"
    duration = await _probe_duration(narration)
    # Decode once, find recurring interior passages, then construct the complete
    # bed before loudness measurement and either narration-ducking mode.
    continuous_music = mix_dir / "continuous_music.wav"
    decoded_music = mix_dir / ".music_source.f32"
    try:
        await _media_command(
            ["ffmpeg", "-y", "-i", str(music), "-vn", "-ar", "48000",
             "-ac", "2", "-f", "f32le", str(decoded_music)], timeout=180,
        )
        loop_report = await asyncio.to_thread(
            render_music_bed, decoded_music, continuous_music, duration,
        )
    finally:
        decoded_music.unlink(missing_ok=True)
    intro_db = bed_db
    speech_music_db = bed_db + duck_db
    ratio = max(4.0, min(20.0, abs(duck_db) * 1.2))
    narration_source = Path(original_narration_path) if original_narration_path else narration
    narration_end = await _probe_duration(narration_source) if original_narration_path else duration
    narration_lufs, music_lufs = await asyncio.gather(
        _integrated_loudness(narration_source),
        _integrated_loudness(continuous_music),
    )
    if music_lufs is None or not math.isfinite(music_lufs) or music_lufs <= -60.0:
        raise RuntimeError("Background music is silent or has no measurable loudness")

    # The targets are source-independent loudness goals, not extra attenuation.
    # Generated sources vary widely: the deterministic fallback is deliberately
    # sparse and can measure near -31 LUFS, while downloaded tracks are often
    # much louder. Convert both targets into source-relative gains so either
    # provider lands at the same audible level before narration ducking.
    intro_gain_db = intro_db - music_lufs
    content_gain_db = bed_db - music_lufs
    speech_gain_db = speech_music_db - music_lufs
    pacing_path = mix_dir / "program_pacing_report.json"
    pacing_report: dict[str, Any] | None = None
    restored_intervals: list[dict[str, Any]] = []
    speech_intervals: list[dict[str, Any]] = []
    if pacing_path.is_file():
        try:
            candidate = json.loads(pacing_path.read_text(encoding="utf-8"))
            paced_path = Path(str(candidate.get("paced_narration_path") or ""))
            pacing_source = Path(original_narration_path) if original_narration_path else narration
            if candidate.get("passed") and paced_path.resolve() == pacing_source.resolve():
                pacing_report = candidate
        except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError):
            pacing_report = None
    if pacing_report:
        intro_seconds = float(pacing_report.get("intro_seconds") or 0)
        if intro_seconds > 0:
            restored_intervals.append(
                {"kind": "intro", "start": 0.0, "end": min(duration, intro_seconds)}
            )
        restored_intervals.extend(
            {
                "kind": str(gap.get("kind") or "gap"),
                "start": float(gap["program_start"]),
                "end": float(gap["program_end"]),
            }
            for gap in pacing_report.get("gaps") or []
        )
        speech_intervals = [
            {
                "kind": str(segment.get("kind") or "speech"),
                "start": float(segment["program_start"]),
                "end": float(segment["program_end"]),
            }
            for segment in pacing_report.get("segments") or []
        ]
        if original_narration_path:
            credits_start = narration_end
            if duration > credits_start:
                restored_intervals.append({"kind": "credits", "start": credits_start, "end": duration})

    if restored_intervals:
        condition = "+".join(
            f"between(t\\,{interval['start']:.6f}\\,{interval['end']:.6f})"
            for interval in restored_intervals
        )
        volume_expression = (
            f"if(gt({condition}\\,0)\\,pow(10\\,{content_gain_db}/20)\\,"
            f"pow(10\\,{speech_gain_db}/20))"
        )
        music_filter = (
            f"[1:a]atrim=0:{duration:.6f},"
            f"asetpts=N/SR/TB,volume='{volume_expression}':eval=frame[ducked];"
        )
        ducking_mode = "program_timeline_envelope"
    else:
        volume_expression = f"pow(10\\,{content_gain_db}/20)"
        music_filter = (
            f"[1:a]atrim=0:{duration:.6f},"
            f"asetpts=N/SR/TB,volume='{volume_expression}':eval=frame[music];"
            f"[music][0:a]sidechaincompress=threshold=0.018:ratio={ratio:.2f}:"
            "attack=18:release=650:makeup=1[ducked];"
        )
        ducking_mode = "sidechain"
    filter_graph = (
        music_filter
        + "[ducked]asplit=2[ducked_mix][ducked_probe];"
        "[0:a][ducked_mix]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
        "alimiter=limit=0.92[out]"
    )
    music_probe.unlink(missing_ok=True)
    window_measurements: dict[str, list[dict[str, Any]]] = {
        "restored_music_windows": [],
        "speech_ducked_windows": [],
    }
    try:
        await _media_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(narration),
                "-i",
                str(continuous_music),
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
                "-map",
                "[ducked_probe]",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                str(music_probe),
            ],
            timeout=max(300, math.ceil(duration * 2)),
        )
        mixed_duration = await _probe_duration(output)
        if abs(mixed_duration - duration) > 0.25:
            raise RuntimeError(
                "Program mix duration drifted from narration: "
                f"{mixed_duration:.3f}s vs {duration:.3f}s"
            )
        ducked_music_lufs, mix_lufs = await asyncio.gather(
            _integrated_loudness(music_probe),
            _integrated_loudness(output),
        )
        # A long music-only tail changes integrated LUFS without changing any
        # spoken samples. Compare the same content interval on both sides.
        content_mix_lufs = (
            await _integrated_loudness(output, end=narration_end)
            if original_narration_path else mix_lufs
        )
        if restored_intervals:
            measurement_rows: list[tuple[str, dict[str, Any], float, float]] = []
            for group, intervals in (
                ("restored_music_windows", restored_intervals),
                ("speech_ducked_windows", speech_intervals),
            ):
                for interval in intervals:
                    start = float(interval["start"])
                    end = min(duration, float(interval["end"]))
                    padding = min(0.2, max(0.0, (end - start) * 0.1))
                    measurement_rows.append((group, interval, start + padding, end - padding))
            values = await asyncio.gather(
                *(
                    _mean_volume(music_probe, start, end)
                    for _, _, start, end in measurement_rows
                )
            )
            for (group, interval, start, end), value in zip(measurement_rows, values):
                window_measurements[group].append(
                    {
                        "kind": interval["kind"],
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "mean_volume_db": value,
                    }
                )
    finally:
        music_probe.unlink(missing_ok=True)

    failure_reasons = []
    if narration_lufs is None or mix_lufs is None or content_mix_lufs is None:
        failure_reasons.append("narration/program loudness could not be measured")
    if ducked_music_lufs is None:
        failure_reasons.append("ducked music loudness could not be measured")
    mix_vs_narration = (
        content_mix_lufs - narration_lufs
        if content_mix_lufs is not None and narration_lufs is not None
        else None
    )
    music_vs_narration = (
        ducked_music_lufs - narration_lufs
        if ducked_music_lufs is not None and narration_lufs is not None
        else None
    )
    minimum_ducked_music_lufs = bed_db + duck_db - 6.0
    restored_values = [
        float(row["mean_volume_db"])
        for row in window_measurements["restored_music_windows"]
        if row.get("mean_volume_db") is not None
    ]
    speech_values = [
        float(row["mean_volume_db"])
        for row in window_measurements["speech_ducked_windows"]
        if row.get("mean_volume_db") is not None
    ]
    measured_gap_lift_db = (
        statistics.median(restored_values) - statistics.median(speech_values)
        if restored_values and speech_values
        else None
    )
    if mix_vs_narration is not None and mix_vs_narration < -1.5:
        failure_reasons.append(
            f"program mix lowered narration by {abs(mix_vs_narration):.1f} LU"
        )
    if (
        ducked_music_lufs is not None
        and ducked_music_lufs < minimum_ducked_music_lufs
    ):
        failure_reasons.append(
            "ducked music is below the configured audibility floor "
            f"({ducked_music_lufs:.1f} < {minimum_ducked_music_lufs:.1f} LUFS)"
        )
    if restored_intervals and measured_gap_lift_db is None:
        failure_reasons.append("program music restoration windows could not be measured")
    elif measured_gap_lift_db is not None and measured_gap_lift_db < 3.0:
        failure_reasons.append(
            "program music did not recover clearly during narration gaps "
            f"({measured_gap_lift_db:.1f} dB measured lift)"
        )
    report = {
        "passed": not failure_reasons,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "narration_path": str(narration.resolve()),
        "music_path": str(music.resolve()),
        "program_mix_path": str(output.resolve()),
        "continuous_music_path": str(continuous_music.resolve()),
        "music_loop": loop_report,
        "narration_duration_seconds": duration,
        "program_mix_duration_seconds": mixed_duration,
        "intro_music_db": intro_db,
        "content_music_db": bed_db,
        "intro_music_gain_db": round(intro_gain_db, 3),
        "content_music_gain_db": round(content_gain_db, 3),
        "speech_music_db": speech_music_db,
        "speech_music_gain_db": round(speech_gain_db, 3),
        "sidechain_duck_target_db": duck_db,
        "ducking_mode": ducking_mode,
        "sidechain_ratio": ratio if ducking_mode == "sidechain" else None,
        "narration_integrated_lufs": narration_lufs,
        "source_music_integrated_lufs": music_lufs,
        "ducked_music_integrated_lufs": ducked_music_lufs,
        "ducked_music_vs_narration_lu": music_vs_narration,
        "minimum_ducked_music_lufs": minimum_ducked_music_lufs,
        "program_mix_integrated_lufs": mix_lufs,
        "content_mix_integrated_lufs": content_mix_lufs,
        "narration_comparison_end_seconds": narration_end,
        "program_mix_vs_narration_lu": mix_vs_narration,
        "restored_music_median_volume_db": (
            round(statistics.median(restored_values), 2) if restored_values else None
        ),
        "speech_ducked_median_volume_db": (
            round(statistics.median(speech_values), 2) if speech_values else None
        ),
        "measured_gap_lift_db": (
            round(measured_gap_lift_db, 2) if measured_gap_lift_db is not None else None
        ),
        "window_measurements": window_measurements,
        "program_mix_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "failure_reasons": failure_reasons,
    }
    (mix_dir / "music_mix_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if failure_reasons:
        raise RuntimeError(f"Background-music mix quality failed: {'; '.join(failure_reasons)}")
    _log(
        log,
        f"Music mix passed ({ducking_mode}): restored target {bed_db:.1f} LUFS, "
        f"speech target {speech_music_db:.1f} LUFS, ducked bed {ducked_music_lufs:.1f} LUFS, "
        f"program {mix_lufs:.1f} LUFS",
    )
    return output
