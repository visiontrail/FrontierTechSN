"""YouTube discovery, web-model trim analysis, and FFmpeg editing.

The source ledger is intentionally stricter than the Wikimedia path: a file
being downloadable does not imply reuse rights. Platform-hosted clips are
marked ``review_required`` in the manifest and remain traceable to their source.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from backend import config
from backend.pipeline.extractors.youtube import _yt_dlp_common_args
from backend.pipeline.opencli import OpenCLIError, first_json, run_opencli

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
YOUTUBE_URL_RE = re.compile(r"https?://(?:www\.)?(?:youtube\.com/watch|youtu\.be/)")
GEMINI_RECOVERY_TIMEOUT_SECONDS = 60.0
GEMINI_RECOVERY_POLL_SECONDS = 5.0


class WebFootageError(RuntimeError):
    """One candidate failed without invalidating the rest of the scout."""


def _emit(log: LogCallback | None, message: str) -> None:
    if log:
        log(message)
    else:
        logger.info(message)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _tool_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [str(config.PROJECT_ROOT / ".venv" / "bin"), env.get("PATH", "")]
    )
    return env


async def _run_command(
    command: list[str],
    *,
    timeout: int,
    check: bool = True,
) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(config.PROJECT_ROOT),
        env=_tool_environment(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise WebFootageError(
            f"Command timed out after {timeout}s: {Path(command[0]).name}"
        ) from exc
    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    returncode = process.returncode or 0
    if check and returncode != 0:
        raise WebFootageError(
            f"{Path(command[0]).name} failed with exit {returncode}: "
            f"{(stderr or stdout)[-1200:]}"
        )
    return returncode, stdout, stderr


def _yt_dlp_bin() -> str:
    project_binary = config.PROJECT_ROOT / ".venv" / "bin" / "yt-dlp"
    if project_binary.is_file():
        return str(project_binary)
    binary = shutil.which("yt-dlp")
    if binary:
        return binary
    raise WebFootageError("yt-dlp is not installed in the project virtualenv")


async def search_youtube(query: str, *, limit: int = 4) -> list[dict]:
    command = [
        _yt_dlp_bin(),
        *_yt_dlp_common_args(include_cookies=False),
        "--flat-playlist",
        "--dump-json",
        "--playlist-end",
        str(limit),
        f"ytsearch{limit}:{query}",
    ]
    _, stdout, _ = await _run_command(command, timeout=90)
    candidates = []
    for line in stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        url = row.get("webpage_url") or row.get("url")
        if not str(url).startswith("http"):
            continue
        candidates.append(
            {
                "platform": "youtube",
                "provider": "YouTube",
                "provider_id": "youtube-ytdlp",
                "title": str(row.get("title") or "YouTube video"),
                "creator": str(row.get("channel") or row.get("uploader") or "Unknown"),
                "source_page_url": str(url),
                "duration_seconds": float(row.get("duration") or 0),
                "description": str(row.get("description") or "")[:500],
                "search_score": int(row.get("view_count") or 0),
            }
        )
    return candidates


def matching_script_excerpt(script: str, query: str, *, limit: int = 1200) -> str:
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n|(?<=[.!?])\s+", script) if chunk.strip()]
    if not chunks:
        return script[:limit]
    terms = {word.lower() for word in WORD_RE.findall(query)}

    def score(chunk: str) -> tuple[int, int]:
        words = {word.lower() for word in WORD_RE.findall(chunk)}
        return len(terms & words), min(len(chunk), limit)

    best = max(chunks, key=score)
    return best[:limit]


def _fallback_analysis(candidate: dict, *, reason: str) -> dict:
    duration = float(candidate.get("duration_seconds") or 0)
    clip_seconds = float(config.WEB_FOOTAGE_CLIP_SECONDS)
    start = min(3.0, max(0.0, duration * 0.1)) if duration else 0.0
    end = min(duration, start + clip_seconds) if duration else start + clip_seconds
    return {
        "start_seconds": round(start, 3),
        "end_seconds": round(max(start + 1.0, end), 3),
        "confidence": 0.25,
        "reason": reason[:300],
        "analyzer": "deterministic-safe-offset",
        "status": "fallback",
    }


def _normalise_analysis(parsed: dict, candidate: dict) -> dict:
    duration = float(candidate.get("duration_seconds") or 0)
    maximum = float(config.WEB_FOOTAGE_CLIP_SECONDS)
    minimum = float(config.WEB_FOOTAGE_CLIP_MIN_SECONDS)
    start = max(0.0, float(parsed.get("start_seconds") or 0))
    end = float(parsed.get("end_seconds") or (start + maximum))
    if end <= start:
        end = start + maximum
    # Extend short intervals toward the target maximum so Gemini's tendency to
    # return ~6-second fragments does not produce unusably brief B-roll.
    target_length = max(minimum, min(maximum, end - start))
    end = start + target_length
    end = min(end, start + maximum)
    if duration:
        start = min(start, max(0.0, duration - 1.0))
        end = min(duration, max(start + 1.0, end))
        # Re-extend after duration clamping when there is room.
        if end - start < minimum and duration > start + minimum:
            end = min(duration, start + minimum)
    confidence = min(1.0, max(0.0, float(parsed.get("confidence") or 0.5)))
    return {
        "start_seconds": round(start, 3),
        "end_seconds": round(end, 3),
        "confidence": round(confidence, 3),
        "reason": str(parsed.get("reason") or "Gemini selected this interval")[:300],
        "analyzer": "gemini-web-via-opencli",
        "status": "analyzed",
    }


def _analysis_from_gemini_turns(value: str, source_page_url: str) -> dict | None:
    """Find the assistant JSON belonging to ``source_page_url`` in Gemini read output."""
    rows = first_json(value)
    if not isinstance(rows, list):
        return None

    anchor = -1
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        role = str(row.get("Role") or row.get("role") or "").lower()
        text = str(row.get("Text") or row.get("text") or "")
        if role == "user" and source_page_url in text:
            anchor = index
    if anchor < 0:
        return None

    for row in rows[anchor + 1 :]:
        if not isinstance(row, dict):
            continue
        role = str(row.get("Role") or row.get("role") or "").lower()
        text = str(row.get("Text") or row.get("text") or "")
        # Do not attribute an answer to this request after a different video
        # request has begun in the same persistent browser session.
        if role == "user" and YOUTUBE_URL_RE.search(text) and source_page_url not in text:
            return None
        if role != "assistant":
            continue
        try:
            parsed = first_json(text)
        except OpenCLIError:
            continue
        if isinstance(parsed, dict) and "start_seconds" in parsed and "end_seconds" in parsed:
            return parsed
    return None


async def _recover_late_gemini_analysis(candidate: dict) -> dict:
    """Poll the current Gemini conversation after ``gemini ask`` times out."""
    source_page_url = str(candidate["source_page_url"])
    loop = asyncio.get_running_loop()
    deadline = loop.time() + GEMINI_RECOVERY_TIMEOUT_SECONDS
    last_error = "assistant response was not visible"

    while True:
        remaining = deadline - loop.time()
        if remaining < 0:
            break
        try:
            result = await run_opencli(
                ["gemini", "read", "-f", "json"],
                timeout=max(5, min(30, int(remaining) + 5)),
            )
            parsed = _analysis_from_gemini_turns(result.stdout, source_page_url)
            if parsed is not None:
                return parsed
        except Exception as exc:  # noqa: BLE001 - keep polling within the grace window
            last_error = str(exc)

        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        await asyncio.sleep(min(GEMINI_RECOVERY_POLL_SECONDS, remaining))

    raise OpenCLIError(
        "Gemini returned no detectable response during the recovery window: "
        f"{last_error}"
    )


async def analyze_candidate_link(candidate: dict, script_excerpt: str) -> dict:
    if not config.WEB_FOOTAGE_GEMINI_ENABLED:
        return _fallback_analysis(candidate, reason="Gemini web analysis is disabled")
    if candidate.get("platform") != "youtube":
        raise WebFootageError("Web footage only accepts YouTube candidates")

    duration = float(candidate.get("duration_seconds") or 0)
    max_seconds = int(config.WEB_FOOTAGE_CLIP_SECONDS)
    min_seconds = int(config.WEB_FOOTAGE_CLIP_MIN_SECONDS)
    prompt = (
        "You are a film editor selecting B-roll for narration. Analyze the actual visuals in "
        f"this public YouTube video: {candidate['source_page_url']}\n"
        f"Candidate duration: {duration:.1f} seconds.\n"
        f"Narration excerpt:\n{script_excerpt}\n\n"
        "Return ONLY one compact JSON object with numeric start_seconds, end_seconds, "
        "confidence (0 to 1), and a short reason. "
        f"Select a visually coherent interval of AT LEAST {min_seconds} seconds and AT MOST "
        f"{max_seconds} seconds — aim for close to {max_seconds} seconds so the clip can "
        "accompany the full narration excerpt. The interval must be long enough to cover the "
        "spoken content meaningfully; do not return a short 5-6 second fragment. "
        "Avoid intros, logos, subtitles, talking-head filler, and end cards."
    )
    try:
        result = await run_opencli(
            [
                "gemini",
                "ask",
                prompt,
                "--new",
                "true",
                "--timeout",
                str(config.WEB_FOOTAGE_GEMINI_TIMEOUT),
            ],
            timeout=config.WEB_FOOTAGE_GEMINI_TIMEOUT + 30,
        )
        recovered = "[NO RESPONSE]" in result.stdout
        parsed = (
            await _recover_late_gemini_analysis(candidate)
            if recovered
            else first_json(result.stdout)
        )
        if not isinstance(parsed, dict):
            raise OpenCLIError("Gemini trim analysis was not a JSON object")
        analysis = _normalise_analysis(parsed, candidate)
        if recovered:
            analysis["status"] = "analyzed_after_timeout"
        return analysis
    except Exception as exc:  # noqa: BLE001 - trim fallback must keep the scout moving
        return _fallback_analysis(candidate, reason=f"Gemini analysis fallback: {exc}")


async def _download_youtube(
    candidate: dict,
    raw_dir: Path,
    analysis: dict,
) -> tuple[Path, bool]:
    template = raw_dir / "%(id)s.%(ext)s"
    sectioned = float(candidate.get("duration_seconds") or 0) > 0
    command = [
        _yt_dlp_bin(),
        *_yt_dlp_common_args(include_cookies=False),
        "--no-playlist",
        "-f",
        "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]",
        "-o",
        str(template),
    ]
    if not sectioned:
        command.extend(["--max-filesize", str(config.FOOTAGE_MAX_BYTES)])
    if sectioned:
        start = float(analysis["start_seconds"])
        end = float(analysis["end_seconds"])
        command.extend(
            [
                "--download-sections",
                f"*{start:.3f}-{end:.3f}",
                "--force-keyframes-at-cuts",
            ]
        )
    command.append(candidate["source_page_url"])
    before = {path.resolve() for path in raw_dir.glob("*") if path.is_file()}
    await _run_command(command, timeout=config.WEB_FOOTAGE_DOWNLOAD_TIMEOUT)
    after = [
        path for path in raw_dir.glob("*")
        if path.is_file() and path.resolve() not in before and path.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}
    ]
    if not after:
        raise WebFootageError("yt-dlp YouTube download produced no media file")
    media_path = max(after, key=lambda path: path.stat().st_mtime_ns)
    if media_path.stat().st_size > config.FOOTAGE_MAX_BYTES:
        raise WebFootageError(
            f"YouTube download exceeds the {config.FOOTAGE_MAX_BYTES}-byte footage limit"
        )
    return media_path, sectioned


async def _probe(path: Path) -> dict:
    _, stdout, _ = await _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(path),
        ],
        timeout=60,
    )
    data = json.loads(stdout)
    stream = (data.get("streams") or [{}])[0]
    return {
        "duration_seconds": float((data.get("format") or {}).get("duration") or 0),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
    }


def _fit_analysis_to_media(analysis: dict, duration: float) -> dict:
    enriched = dict(analysis)
    maximum = float(config.WEB_FOOTAGE_CLIP_SECONDS)
    minimum = float(config.WEB_FOOTAGE_CLIP_MIN_SECONDS)
    requested_start = float(enriched.get("start_seconds") or 0)
    requested_end = float(enriched.get("end_seconds") or (requested_start + maximum))
    requested_length = requested_end - requested_start
    # Clamp to [minimum, maximum] so a short Gemini interval is extended and an
    # over-long one is capped, rather than preserving whatever was returned.
    requested_length = max(minimum, min(maximum, max(1.0, requested_length)))
    if (
        enriched.get("analyzer") == "deterministic-safe-offset"
        and requested_start == 0
        and duration > maximum * 2
    ):
        # If discovery did not expose duration, use the probed duration to skip
        # the likely channel bumper instead of blindly trimming from frame zero.
        requested_start = min(max(3.0, duration * 0.1), duration - maximum)
    start = max(0.0, min(requested_start, max(0.0, duration - 1.0)))
    end = min(duration, start + requested_length)
    # Final guard: if the source is long enough, never ship a sub-minimum clip.
    if end - start < minimum and duration >= start + minimum:
        end = min(duration, start + minimum)
    enriched["start_seconds"] = round(start, 3)
    enriched["end_seconds"] = round(end, 3)
    return enriched


async def _trim(raw_path: Path, destination: Path, analysis: dict, orientation: str) -> None:
    start = float(analysis["start_seconds"])
    length = max(1.0, float(analysis["end_seconds"]) - start)
    if orientation == "portrait":
        width, height = 720, 1280
    else:
        width, height = 1280, 720
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1"
    )
    await _run_command(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(raw_path),
            "-t",
            f"{length:.3f}",
            "-an",
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        timeout=180,
    )


async def _evidence_frames(video_path: Path, evidence_dir: Path, duration: float) -> list[str]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    timestamps = sorted({0.0, max(0.0, duration / 2), max(0.0, duration - 0.15)})
    output = []
    for index, timestamp in enumerate(timestamps, start=1):
        path = evidence_dir / f"frame-{index:02d}-{timestamp:.2f}s.jpg"
        await _run_command(
            [
                "ffmpeg", "-y", "-v", "error", "-ss", f"{timestamp:.3f}",
                "-i", str(video_path), "-frames:v", "1", "-vf", "scale=640:-2", str(path),
            ],
            timeout=60,
        )
        output.append(path.name)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def supplement_web_footage(
    *,
    task_dir: Path,
    manifest: dict,
    query_plan: list[dict[str, str]],
    target_total: int,
    orientation: str,
    script: str,
    log: LogCallback | None = None,
) -> dict:
    """Append rights-ledgered web clips until ``target_total`` is reached."""
    manifest_file = task_dir / "footage" / "manifest.json"
    raw_dir = task_dir / "footage" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    evidence_root = task_dir / "footage" / "evidence"
    used_sources = {str(clip.get("source_page_url") or "") for clip in manifest.get("clips", [])}

    if manifest.get("provider_id") in {"opencli-web", "youtube-web"}:
        manifest["provider"] = "YouTube"
        manifest["provider_id"] = "youtube-web"
    else:
        manifest["provider"] = "Hybrid: Wikimedia Commons + YouTube"
        manifest["provider_id"] = "hybrid-youtube"
    manifest["requested_clip_count"] = target_total
    manifest["rights_review_required"] = True
    manifest.setdefault("publication_blockers", [])
    legacy_blocker = "Review reuse rights for every Bilibili/YouTube clip before publication"
    manifest["publication_blockers"] = [
        item for item in manifest["publication_blockers"] if item != legacy_blocker
    ]
    blocker = "Review reuse rights for every YouTube clip before publication"
    if blocker not in manifest["publication_blockers"]:
        manifest["publication_blockers"].append(blocker)
    manifest["status"] = "searching"
    manifest["updated_at"] = _now()
    _write_manifest(manifest_file, manifest)

    for shot in query_plan:
        if len(manifest.get("clips", [])) >= target_total:
            break
        query = str(shot.get("query") or "").strip()
        if not query:
            continue
        _emit(log, f"Web footage: searching YouTube for '{query}'")
        try:
            results = await search_youtube(query)
        except Exception as exc:  # noqa: BLE001 - record and continue with the next shot
            manifest.setdefault("errors", []).append(
                {"query": query, "stage": "youtube-search", "message": str(exc)}
            )
            continue
        candidate = next(
            (
                item
                for item in results
                if item["source_page_url"] not in used_sources
            ),
            None,
        )
        if candidate is None:
            manifest.setdefault("errors", []).append(
                {"query": query, "stage": "web-selection", "message": "No unique web candidate found"}
            )
            continue

        excerpt = matching_script_excerpt(script, query)
        _emit(log, f"Web footage: asking Gemini Web to select a trim for {candidate['source_page_url']}")
        analysis = await analyze_candidate_link(candidate, excerpt)
        raw_path: Path | None = None
        try:
            source_duration = float(candidate.get("duration_seconds") or 0)
            if source_duration:
                analysis = _fit_analysis_to_media(analysis, source_duration)
            raw_path, sectioned = await _download_youtube(candidate, raw_dir, analysis)
            media = await _probe(raw_path)
            if not source_duration:
                source_duration = media["duration_seconds"]
                analysis = _fit_analysis_to_media(analysis, source_duration)
            edit_analysis = analysis
            if sectioned:
                requested_length = max(
                    1.0,
                    float(analysis["end_seconds"]) - float(analysis["start_seconds"]),
                )
                edit_analysis = {
                    "start_seconds": 0.0,
                    "end_seconds": min(media["duration_seconds"], requested_length),
                }
            clip_id = f"clip-{len(manifest.get('clips', [])) + 1:02d}"
            destination = task_dir / "footage" / f"{clip_id}.mp4"
            await _trim(raw_path, destination, edit_analysis, orientation)
            trimmed = await _probe(destination)
            evidence_names = await _evidence_frames(
                destination,
                evidence_root / clip_id,
                trimmed["duration_seconds"],
            )
        except Exception as exc:  # noqa: BLE001 - try the next query/candidate
            manifest.setdefault("errors", []).append(
                {
                    "query": query,
                    "stage": "web-download-edit",
                    "source_page_url": candidate["source_page_url"],
                    "message": str(exc),
                }
            )
            _emit(log, f"Web footage candidate failed: {exc}")
            continue

        # Keep failed downloads for diagnosis. Remove the exact raw file only
        # after the normalized derivative and evidence frames are verified.
        if raw_path and raw_path.exists():
            raw_path.unlink()

        entry = {
            "id": clip_id,
            "query": query,
            "purpose": str(shot.get("purpose") or ""),
            **candidate,
            "duration_seconds": round(trimmed["duration_seconds"], 3),
            "source_duration_seconds": round(source_duration, 3),
            "width": trimmed["width"],
            "height": trimmed["height"],
            "bytes": destination.stat().st_size,
            "mime_type": "video/mp4",
            "sha256": _sha256(destination),
            "local_path": destination.relative_to(task_dir).as_posix(),
            "status": "downloaded_and_trimmed",
            "analysis": analysis,
            "script_excerpt": excerpt,
            "evidence_frames": [
                f"footage/evidence/{clip_id}/{name}" for name in evidence_names
            ],
            "license": "Rights not verified — human review required",
            "license_code": "rights-review-required",
            "license_url": "",
            "attribution_required": True,
            "review_required": True,
            "rights_status": "review_required",
        }
        manifest.setdefault("clips", []).append(entry)
        used_sources.add(candidate["source_page_url"])
        manifest["updated_at"] = _now()
        _write_manifest(manifest_file, manifest)
        _emit(
            log,
            f"Web footage: {clip_id} trimmed to {analysis['start_seconds']:.1f}–"
            f"{analysis['end_seconds']:.1f}s via {analysis['analyzer']}",
        )

    if raw_dir.exists() and not any(raw_dir.iterdir()):
        raw_dir.rmdir()
    acquired = len(manifest.get("clips", []))
    manifest["status"] = "ready" if acquired >= target_total else ("partial" if acquired else "no_results")
    manifest["updated_at"] = _now()
    _write_manifest(manifest_file, manifest)
    return manifest
