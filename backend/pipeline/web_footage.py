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
import math
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
SEARCH_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
YOUTUBE_URL_RE = re.compile(r"https?://(?:www\.)?(?:youtube\.com/watch|youtu\.be/)")
GEMINI_RECOVERY_TIMEOUT_SECONDS = 60.0
GEMINI_RECOVERY_POLL_SECONDS = 5.0
MAX_CANDIDATE_ATTEMPTS_PER_QUERY = 3
SEARCH_STOPWORDS = frozenset(
    "a an and are as at be by for from how in into is it of on or the this to use with".split()
)
YOUTUBE_EMBEDDED_PLAYER_ARGS = (
    "--extractor-args",
    "youtube:player_client=web_embedded",
)


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


def _search_match_terms(value: str) -> set[str]:
    """Normalize discovery text while retaining product and entity names."""
    words = [_normalize_search_word(word) for word in SEARCH_WORD_RE.findall(value)]
    terms: set[str] = set()
    for word in words:
        if word in SEARCH_STOPWORDS:
            continue
        terms.add(word)
    # YouTube titles alternate freely between "robotaxi", "robo-taxi", and
    # "robo taxi". Adjacent compact forms make those spellings equivalent
    # without adding a product-specific alias table.
    terms.update(
        left + right
        for left, right in zip(words, words[1:])
        if left not in SEARCH_STOPWORDS and right not in SEARCH_STOPWORDS
    )
    return terms


def _normalize_search_word(word: str) -> str:
    value = word.casefold()
    if value.endswith("ies") and len(value) > 5:
        return value[:-3] + "y"
    if value.endswith("s") and not value.endswith("ss") and len(value) > 4:
        return value[:-1]
    return value


def _rank_youtube_candidates(candidates: list[dict], query: str) -> list[dict]:
    """Prefer query-specific results while preserving YouTube order for ties."""
    query_terms = _search_match_terms(query)
    ordered_query_terms = [
        _normalize_search_word(word)
        for word in SEARCH_WORD_RE.findall(query)
        if word.casefold() not in SEARCH_STOPWORDS
    ]
    query_weights = {
        term: len(ordered_query_terms) - index
        for index, term in enumerate(ordered_query_terms)
    }
    ranked: list[tuple[tuple[int, int, int, int], int, dict]] = []
    for index, candidate in enumerate(candidates):
        title_terms = _search_match_terms(str(candidate.get("title") or ""))
        metadata_terms = _search_match_terms(
            " ".join(
                [
                    str(candidate.get("title") or ""),
                    str(candidate.get("creator") or ""),
                    str(candidate.get("description") or ""),
                ]
            )
        )
        title_overlap = len(query_terms & title_terms)
        metadata_overlap = len(query_terms & metadata_terms)
        weighted_title_overlap = sum(
            query_weights.get(term, 1) for term in query_terms & title_terms
        )
        # Distinct title-term coverage is the primary signal. Query order then
        # breaks ties so a subject/object pair ("Didi Robotaxi") beats an
        # object/location pair ("Robotaxi Beijing") without letting one early
        # but ambiguous word ("Faraday" in "Faraday motor") outrank two later
        # exact concepts ("Volta battery").
        relevance = (
            title_overlap,
            weighted_title_overlap,
            metadata_overlap,
            -index,
        )
        enriched = {
            **candidate,
            "query_relevance_score": (
                title_overlap * 100 + weighted_title_overlap * 10 + metadata_overlap
            ),
        }
        ranked.append((relevance, index, enriched))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _score, _index, candidate in ranked]


async def search_youtube(query: str, *, limit: int = 8) -> list[dict]:
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
    return _rank_youtube_candidates(candidates, query)


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


def analysis_rejection(analysis: dict) -> str:
    """Use the same minimum suitability as placement, before downloading."""
    verdict = str(analysis.get("suitable", "")).strip().casefold()
    reason = str(analysis.get("reason") or "")
    if verdict in {"false", "no", "0", "unsuitable"}:
        return reason or "Gemini rejected this candidate"
    if re.search(
        r"no (?:visual )?connection|unrelated|not suitable|does not (?:match|depict)"
        r"|unavailable|cannot (?:view|access|verify)", reason, re.I,
    ):
        return reason
    confidence = float(analysis.get("confidence") or 0)
    if not math.isfinite(confidence) or confidence < 0.65:
        return reason or "Visual suitability confidence is below 0.65"
    return ""


def _normalise_analysis(parsed: dict, candidate: dict) -> dict:
    rejection = analysis_rejection(parsed)
    if rejection:
        return {
            "suitable": False, "status": "rejected",
            "confidence": float(parsed.get("confidence") or 0),
            "reason": rejection, "analyzer": "gemini-web-via-opencli",
        }
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
        if isinstance(parsed, dict) and ("suitable" in parsed or ("start_seconds" in parsed and "end_seconds" in parsed)):
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
        "First determine whether you can inspect the actual video and whether it depicts "
        "the narrated subject. Do not infer visuals from the narration or invent imagery. "
        f"Source title: {candidate.get('title', '')}\n"
        "If unrelated or unviewable, return suitable:false, confidence and reason, without timestamps. "
        "Otherwise return ONLY one compact JSON object with suitable:true, numeric start_seconds, end_seconds, "
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
            analysis["status"] = "rejected_after_timeout" if analysis.get("suitable") is False else "analyzed_after_timeout"
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
    try:
        await _run_command(command, timeout=config.WEB_FOOTAGE_DOWNLOAD_TIMEOUT)
    except WebFootageError as section_error:
        if not sectioned:
            embedded_command = [
                _yt_dlp_bin(),
                *_yt_dlp_common_args(include_cookies=False),
                *YOUTUBE_EMBEDDED_PLAYER_ARGS,
                "--no-playlist",
                "-f",
                "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]",
                "--max-filesize",
                str(config.FOOTAGE_MAX_BYTES),
                "-o",
                str(template),
                candidate["source_page_url"],
            ]
            try:
                await _run_command(
                    embedded_command,
                    timeout=config.WEB_FOOTAGE_DOWNLOAD_TIMEOUT,
                )
            except WebFootageError as embedded_error:
                raise WebFootageError(
                    "YouTube default and embedded-client downloads both failed: "
                    f"default={str(section_error)[-420:]}; "
                    f"embedded={str(embedded_error)[-420:]}"
                ) from embedded_error
        else:
            # Current YouTube GVS enforcement can return signed Android/VR
            # googlevideo URLs that list normally but reject both FFmpeg range
            # requests and yt-dlp's native downloader with HTTP 403.  The
            # unauthenticated web-embedded player uses a separately attested URL
            # for embeddable public videos, so retry the same bounded interval
            # before paying the cost of downloading the complete source.
            embedded_section_command = [
                _yt_dlp_bin(),
                *_yt_dlp_common_args(include_cookies=False),
                *YOUTUBE_EMBEDDED_PLAYER_ARGS,
                "--no-playlist",
                "-f",
                "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]",
                "-o",
                str(template),
                "--download-sections",
                f"*{start:.3f}-{end:.3f}",
                "--force-keyframes-at-cuts",
                candidate["source_page_url"],
            ]
            try:
                await _run_command(
                    embedded_section_command,
                    timeout=config.WEB_FOOTAGE_DOWNLOAD_TIMEOUT,
                )
            except WebFootageError as embedded_section_error:
                # Some videos cannot be embedded. Fetch a bounded 360p source
                # with yt-dlp's native downloader, then trim it locally. The
                # byte ceiling prevents an unexpectedly long source from
                # filling the task workspace.
                fallback_template = raw_dir / "%(id)s-full.%(ext)s"
                fallback_command = [
                    _yt_dlp_bin(),
                    *_yt_dlp_common_args(include_cookies=False),
                    "--no-playlist",
                    "-f",
                    (
                        "bestvideo[height<=360][ext=mp4]/"
                        "bestvideo[height<=360]/best[height<=360]"
                    ),
                    "--max-filesize",
                    str(config.FOOTAGE_MAX_BYTES),
                    "-o",
                    str(fallback_template),
                    candidate["source_page_url"],
                ]
                try:
                    await _run_command(
                        fallback_command,
                        timeout=config.WEB_FOOTAGE_DOWNLOAD_TIMEOUT,
                    )
                except WebFootageError as fallback_error:
                    raise WebFootageError(
                        "YouTube section, embedded-section, and bounded "
                        "full-download fallbacks all failed: "
                        f"section={str(section_error)[-300:]}; "
                        f"embedded={str(embedded_section_error)[-300:]}; "
                        f"fallback={str(fallback_error)[-300:]}"
                    ) from fallback_error
                sectioned = False
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


async def _analyze_candidate_preview(candidate: dict, excerpt: str, task_dir: Path) -> dict:
    """Judge actual downloaded pixels when the model cannot inspect a URL.

    A deterministic offset is only a preview proposal. It becomes eligible
    footage only after the attached contact sheet receives an explicit verdict.
    """
    key = hashlib.sha256(candidate["source_page_url"].encode()).hexdigest()[:16]
    folder = task_dir / "footage" / "evidence" / "previews" / key
    folder.mkdir(parents=True, exist_ok=True)
    interval = _fallback_analysis(candidate, reason="Unreviewed preview proposal")
    raw, sectioned = await _download_youtube(candidate, folder, interval)
    media = await _probe(raw)
    if not sectioned:
        interval = _fit_analysis_to_media(interval, media["duration_seconds"])
    edit = {"start_seconds": 0, "end_seconds": interval["end_seconds"] - interval["start_seconds"]} if sectioned else interval
    preview = folder / "preview.mp4"
    await _trim(raw, preview, edit, "landscape")
    sheet = folder / "contact-sheet.jpg"
    await _run_command([
        "ffmpeg", "-y", "-v", "error", "-i", str(preview),
        "-vf", "fps=1,scale=320:180:force_original_aspect_ratio=decrease,pad=320:180:(ow-iw)/2:(oh-ih)/2,tile=4x4",
        "-frames:v", "1", str(sheet),
    ], timeout=60)
    prompt = (
        "Evaluate the attached contact sheet of REAL downloaded video frames, "
        "sampled at one frame per second in row-major order. Black final cells are padding. "
        "Judge only visible pixels, not what the source title or narration suggests. "
        f"Narration: {excerpt}\nSource title: {candidate.get('title', '')}\n"
        "Does this preview depict the narrated people, product or action, or provide "
        "clearly relevant contextual B-roll without misrepresenting a different event? "
        "Reject unrelated footage and text-only/talking-head filler. "
        "Return JSON with image_received (boolean), suitable (boolean), confidence (0..1), "
        "visible_content (concrete visual description), reason. If the image is absent, "
        "unreadable or insufficient to establish relevance, suitable must be false."
    )
    from backend.pipeline.multimodal_review import _response_payload, _review_command

    result = await run_opencli(
        _review_command("gemini", prompt, sheet, config.WEB_FOOTAGE_GEMINI_TIMEOUT),
        timeout=config.WEB_FOOTAGE_GEMINI_TIMEOUT + 60,
    )
    verdict = _response_payload(result.stdout)
    (folder / "review.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    if verdict.get("image_received") is not True or verdict.get("suitable") is not True:
        raise WebFootageError(f"Preview visual review rejected: {verdict.get('reason', 'No explicit image verdict')}")
    rejection = analysis_rejection(verdict)
    if rejection or not str(verdict.get("visible_content") or "").strip():
        raise WebFootageError(f"Preview visual review rejected: {rejection or 'Missing visible-content evidence'}")
    return {
        **interval, **verdict, "analyzer": "gemini-web-contact-sheet",
        "status": "analyzed", "preview_evidence": sheet.relative_to(task_dir).as_posix(),
    }


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

    pending_shots = [dict(shot) for shot in query_plan]
    for shot in pending_shots:
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
            # Long entity lists can overconstrain YouTube. Broaden only the
            # query, retaining the same narration binding and suitability gate.
            words = query.split()
            if len(words) > 3 and not shot.get("_short_query"):
                shorter = " ".join(words[:3])
                pending_shots.append({**shot, "query": shorter, "_short_query": True})
                _emit(log, f"Web footage: no results; retrying subject query '{shorter}'")
            continue

        excerpt = str(shot.get("script_excerpt") or "").strip()
        if not excerpt:
            # Backward-compatible recovery for an older/manual query plan.  A
            # descriptive purpose often contains the story identity even when
            # the short visual query uses a synonym (Taipei vs Taiwan).
            excerpt = matching_script_excerpt(
                script,
                f"{query} {str(shot.get('purpose') or '')}",
            )
        candidate_attempt = int(shot.get("_candidate_attempt") or 1)
        _emit(
            log,
            "Web footage: asking Gemini Web to select a trim for "
            f"{candidate['source_page_url']} (candidate {candidate_attempt}/"
            f"{MAX_CANDIDATE_ATTEMPTS_PER_QUERY})",
        )
        raw_path: Path | None = None
        try:
            if manifest.get("url_inspection_unavailable"):
                analysis = await _analyze_candidate_preview(candidate, excerpt, task_dir)
            else:
                analysis = await analyze_candidate_link(candidate, excerpt)
            rejection = analysis_rejection(analysis)
            if rejection and (
                analysis.get("status") == "fallback"
                or re.search(r"unavailable|unviewable|not possible|cannot (?:view|access|verify)|unable to", rejection, re.I)
            ):
                _emit(log, "Web footage: URL inspection unavailable; reviewing actual preview frames")
                manifest["url_inspection_unavailable"] = True
                _write_manifest(manifest_file, manifest)
                analysis = await _analyze_candidate_preview(candidate, excerpt, task_dir)
                rejection = analysis_rejection(analysis)
            if rejection:
                manifest.setdefault("rejected_candidates", []).append({
                    "query": query, "source_page_url": candidate["source_page_url"],
                    "script_excerpt": excerpt, "analysis": analysis,
                })
                raise WebFootageError(f"Visual suitability rejected: {rejection}")
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
            from backend.pipeline.footage import _next_clip_id

            clip_id = _next_clip_id(task_dir / "footage", manifest.get("clips", []))
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
                    "candidate_attempt": candidate_attempt,
                    "message": str(exc),
                }
            )
            used_sources.add(candidate["source_page_url"])
            manifest["updated_at"] = _now()
            _write_manifest(manifest_file, manifest)
            if candidate_attempt < MAX_CANDIDATE_ATTEMPTS_PER_QUERY:
                pending_shots.append(
                    {
                        **shot,
                        "_candidate_attempt": candidate_attempt + 1,
                    }
                )
                _emit(
                    log,
                    "Web footage candidate failed; queued the next unique result for "
                    f"'{query}': {str(exc)[-520:]}",
                )
            else:
                if not shot.get("_stock_variant") and "stock footage" not in query.casefold():
                    variant_query = f"{query} stock footage"
                    pending_shots.append(
                        {
                            **shot,
                            "query": variant_query,
                            "_candidate_attempt": 1,
                            "_stock_variant": True,
                        }
                    )
                    _emit(
                        log,
                        f"Web footage exhausted {candidate_attempt} candidates for '{query}'; "
                        f"queued focused short-form fallback '{variant_query}'",
                    )
                else:
                    _emit(
                        log,
                        f"Web footage exhausted {candidate_attempt} candidates for '{query}': "
                        f"{str(exc)[-520:]}",
                    )
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
