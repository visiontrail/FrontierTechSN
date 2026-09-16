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
import signal
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from backend import config
from backend.pipeline.extractors.youtube import _yt_dlp_common_args
from backend.pipeline.opencli import OpenCLIError, first_json, run_opencli
from backend.pipeline.review_response import ReviewResponseError, parse_review_response

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
SEARCH_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
YOUTUBE_URL_RE = re.compile(r"https?://(?:www\.)?(?:youtube\.com/watch|youtu\.be/)")
GEMINI_RECOVERY_TIMEOUT_SECONDS = 60.0
GEMINI_RECOVERY_POLL_SECONDS = 5.0
MAX_CANDIDATE_ATTEMPTS_PER_QUERY = 3
MAX_PREVIEW_REVIEW_ATTEMPTS = 3
SEARCH_REPAIR_TIMEOUT_SECONDS = 60
SEARCH_STOPWORDS = frozenset(
    "a an and are as at be by for from how in into is it of on or the this to use with".split()
)
YOUTUBE_EMBEDDED_PLAYER_ARGS = (
    "--extractor-args",
    "youtube:player_client=web_embedded",
)


class WebFootageError(RuntimeError):
    """One candidate failed without invalidating the rest of the scout."""


class WebFootageReviewUnavailable(WebFootageError):
    """The review did not execute; candidate suitability remains undecided."""

    def __init__(self, message: str, *, failure_kind: str = "transport") -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


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
        start_new_session=True,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except (TimeoutError, asyncio.CancelledError) as exc:
        try:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        await process.communicate()
        if isinstance(exc, asyncio.CancelledError):
            raise
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


def _metadata_matches_query(candidate: dict, query: str) -> bool:
    """Avoid a paced visual review of search results with no concrete anchors."""
    if "query_relevance_score" not in candidate:
        return True  # Older/manual candidates still require actual pixel review.
    terms = {_normalize_search_word(word) for word in SEARCH_WORD_RE.findall(query)
             if len(word) > 1 and word.casefold() not in SEARCH_STOPWORDS | {"stock", "footage", "video"}}
    metadata = _search_match_terms(" ".join(str(candidate.get(key) or "") for key in ("title", "creator", "description")))
    return len(terms & metadata) >= min(2, len(terms))


def _metadata_rejection(candidate: dict, query: str) -> str | None:
    if re.search(r"(?:\(|\[|\|)\s*audio[\s-]+only\s*(?:\)|\]|\||$)",
                 str(candidate.get("title") or ""), re.I):
        return "Source title explicitly labels the recording Audio Only; no B-roll candidate"
    if not _metadata_matches_query(candidate, query):
        return "Search metadata lacks two concrete query anchors"
    return None


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
        "--force-overwrites", "--no-mtime",
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
    before = {path.resolve(): path.stat().st_mtime_ns for path in raw_dir.glob("*") if path.is_file()}
    try:
        await _run_command(command, timeout=config.WEB_FOOTAGE_DOWNLOAD_TIMEOUT)
    except WebFootageError as section_error:
        if not sectioned:
            embedded_command = [
                _yt_dlp_bin(),
                *_yt_dlp_common_args(include_cookies=False),
                "--force-overwrites", "--no-mtime",
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
                "--force-overwrites", "--no-mtime",
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
                    "--force-overwrites", "--no-mtime",
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
        if path.is_file() and before.get(path.resolve()) != path.stat().st_mtime_ns and path.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}
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


def _preview_cache_identity(candidate: dict) -> dict:
    return {"version": 1, "source_page_url": candidate["source_page_url"],
            "duration_seconds": float(candidate.get("duration_seconds") or 0)}


def _save_prepared_preview(candidate: dict, prepared: dict, task_dir: Path) -> None:
    folder = prepared["folder"]
    files = [prepared["sheet"], *[
        folder / f"window-{index}" / filename
        for index in range(len(prepared["intervals"]))
        for filename in ("preview.mp4", "frames.jpg")
    ]]
    contract = {"identity": _preview_cache_identity(candidate), "candidate": candidate,
                "intervals": prepared["intervals"],
                "files": {path.relative_to(task_dir).as_posix(): _sha256(path) for path in files}}
    _write_manifest(folder / "preview-cache.json", contract)


def _load_prepared_preview(candidate: dict, folder: Path, task_dir: Path) -> dict | None:
    try:
        contract = json.loads((folder / "preview-cache.json").read_text())
        if contract.get("identity") != _preview_cache_identity(candidate):
            return None
        intervals = contract["intervals"]
        if not intervals or not isinstance(intervals, list):
            return None
        expected = [folder / "contact-sheet.jpg", *[
            folder / f"window-{index}" / filename for index in range(len(intervals))
            for filename in ("preview.mp4", "frames.jpg")
        ]]
        for path in expected:
            if (task_dir.resolve() not in path.resolve().parents or not path.is_file()
                    or _sha256(path) != contract["files"].get(path.relative_to(task_dir).as_posix())):
                return None
        return {"folder": folder, "sheet": expected[0], "intervals": intervals}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


async def _prepare_candidate_preview(
    candidate: dict, excerpt: str, task_dir: Path, *, source_path: Path | None = None,
) -> dict:
    """Judge actual downloaded pixels when the model cannot inspect a URL.

    A deterministic offset is only a preview proposal. It becomes eligible
    footage only after the attached contact sheet receives an explicit verdict.
    """
    if source_path is not None:
        source_path = source_path.resolve()
        source_path.relative_to(task_dir.resolve())
        if not source_path.is_file():
            raise WebFootageError("Publisher source is not a local task artifact")
    key = hashlib.sha256(candidate["source_page_url"].encode()).hexdigest()[:16]
    folder = task_dir / "footage" / "evidence" / "previews" / key
    folder.mkdir(parents=True, exist_ok=True)
    if source_path is None:
        cached = _load_prepared_preview(candidate, folder, task_dir)
        if cached is not None:
            return cached
    proposal = _fallback_analysis(candidate, reason="Unreviewed preview proposal")
    duration = float(candidate.get("duration_seconds") or 0)
    length = proposal["end_seconds"] - proposal["start_seconds"]
    starts = sorted({
        proposal["start_seconds"],
        *([min(max(0, duration - length), duration * fraction) for fraction in (0.33, 0.67)]
          if duration > length * 2 else []),
    })
    intervals: list[dict] = []
    sheets: list[Path] = []
    for index, start in enumerate(starts):
        window = folder / f"window-{index}"
        window.mkdir(exist_ok=True)
        interval = {**proposal, "start_seconds": start, "end_seconds": start + length}
        raw, sectioned = (
            (source_path, False) if source_path is not None
            else await _download_youtube(candidate, window, interval)
        )
        media = await _probe(raw)
        if not sectioned:
            interval = _fit_analysis_to_media(interval, media["duration_seconds"])
        edit = ({"start_seconds": 0, "end_seconds": interval["end_seconds"] - interval["start_seconds"]}
                if sectioned else interval)
        preview = window / "preview.mp4"
        await _trim(raw, preview, edit, "landscape")
        row = window / "frames.jpg"
        await _run_command([
            "ffmpeg", "-y", "-v", "error", "-i", str(preview),
            "-vf", "fps=1/3,scale=320:180:force_original_aspect_ratio=decrease,pad=320:180:(ow-iw)/2:(oh-ih)/2,tile=5x1",
            "-frames:v", "1", str(row),
        ], timeout=60)
        intervals.append(interval)
        sheets.append(row)
    sheet = folder / "contact-sheet.jpg"
    inputs = [arg for row in sheets for arg in ("-i", str(row))]
    await _run_command([
        "ffmpeg", "-y", "-v", "error", *inputs,
        "-filter_complex", f"vstack=inputs={len(sheets)}" if len(sheets) > 1 else "null",
        "-frames:v", "1", str(sheet),
    ], timeout=60)
    prepared = {"folder": folder, "sheet": sheet, "intervals": intervals}
    if source_path is None:
        _save_prepared_preview(candidate, prepared, task_dir)
    return prepared


async def _analyze_candidate_preview(
    candidate: dict, excerpt: str, task_dir: Path, *, source_path: Path | None = None,
    log: LogCallback | None = None,
) -> dict:
    prepared = await _prepare_candidate_preview(candidate, excerpt, task_dir, source_path=source_path)
    folder, sheet, intervals = (prepared[key] for key in ("folder", "sheet", "intervals"))
    prompt = (
        "Evaluate the attached contact sheet of REAL downloaded video frames, "
        "Each row is a different 15-second candidate interval with five frames sampled "
        "three seconds apart. Rows are numbered from zero, top to bottom. Black final cells are padding. "
        f"Available row indices: {list(range(len(intervals)))}. "
        "Judge only visible pixels, not what the source title or narration suggests. "
        f"Narration: {excerpt}\nSource title: {candidate.get('title', '')}\n"
        f"Planned visual subject: {candidate.get('visual_query', '')}\n"
        f"Visual purpose: {candidate.get('visual_purpose', '')}\n"
        "Does this preview depict the narrated people, product or action, or provide "
        "clearly relevant contextual B-roll without misrepresenting a different event? "
        "Contextual B-roll illustrates a narrated object or action; it need not visually prove "
        "spoken statistics or funding claims. It must still match the planned visual subject "
        "and the narration, and must not substitute a different named product or event. "
        "Mere theme or keyword overlap is insufficient: for example, synthetic political "
        "or religious memes do not depict government AI policy. Do not treat invented "
        "events as real-world context. Reject unrelated footage and text-only/talking-head filler. "
        "Return ONLY one complete JSON object with image_received (boolean), suitable (boolean), confidence (0..1), "
        "selected_window (integer row index of the best suitable interval), "
        "visible_content (concrete visual description of THAT row), reason. If the image is absent, "
        "unreadable or insufficient to establish relevance, suitable must be false."
    )
    verdict = await _request_preview_review(
        prompt, sheet, {0: prepared}, batch=False, log=log,
    )
    (folder / "review.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return _validated_preview(verdict, prepared, task_dir)


def _preview_response_payload(output: str, *, batch: bool) -> dict:
    return parse_review_response(
        output, required_fields={"results"} if batch else {"image_received", "suitable"},
        label="Gemini batch footage preview" if batch else "Gemini footage preview",
    )


def _validate_preview_response(payload: dict, prepared: dict[int, dict], *, batch: bool) -> None:
    """Separate an unusable answer from an explicit rejection of the pixels."""
    if batch:
        rows = payload.get("results")
        if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
            raise ReviewResponseError("Footage preview results must be an array of verdicts")
        # Map and validate each candidate independently below. Preserve valid
        # reviews/rejections when another row is missing or ambiguous.
        return
    if payload.get("image_received") is not True:
        raise ReviewResponseError("Footage preview image was not received or readable")
    if type(payload.get("suitable")) is not bool:
        raise ReviewResponseError("Footage preview requires an explicit boolean suitability verdict")
    if payload["suitable"] is False:
        # A real rejection is terminal for this candidate, never retried
        # in search of a more favorable answer.
        return
    confidence = payload.get("confidence")
    if (type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not 0 <= confidence <= 1):
        raise ReviewResponseError("Footage preview confidence must be a finite number from 0 to 1")
    selected = payload.get("selected_window")
    if type(selected) is not int or not 0 <= selected < len(prepared[0]["intervals"]):
        raise ReviewResponseError("Footage preview did not select a valid sampled interval")
    if not isinstance(payload.get("visible_content"), str) or not payload["visible_content"].strip():
        raise ReviewResponseError("Footage preview omitted visible-content evidence")


async def _request_preview_review(
    prompt: str, sheet: Path, prepared: dict[int, dict], *, batch: bool,
    log: LogCallback | None,
) -> dict:
    from backend.pipeline.multimodal_review import _review_command

    # Each retry invocation gets its own directory so resuming a failed task
    # cannot overwrite the response that explains the original failure.
    evidence = sheet.parent / "review-attempts" / uuid.uuid4().hex
    evidence.mkdir(parents=True, exist_ok=True)
    _write_manifest(evidence / "request.json", {
        "prompt": prompt, "sheet_sha256": _sha256(sheet) if sheet.is_file() else None,
        "contract": "footage-preview-batch" if batch else "footage-preview",
        "candidate_ids": list(prepared),
    })
    last_error: Exception | None = None
    failure_kind = "transport"
    for attempt in range(1, MAX_PREVIEW_REVIEW_ATTEMPTS + 1):
        result = None
        record = {"provider": "gemini", "attempt": attempt}
        try:
            result = await run_opencli(
                _review_command("gemini", prompt, sheet, config.WEB_FOOTAGE_GEMINI_TIMEOUT),
                timeout=config.WEB_FOOTAGE_GEMINI_TIMEOUT + 60,
                check=False,
            )
            record.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
            if result.returncode:
                raise OpenCLIError(
                    f"OpenCLI Gemini preview failed with exit {result.returncode}: "
                    f"{(result.stderr or result.stdout)[-1200:]}"
                )
            payload = _preview_response_payload(f"{result.stdout}\n{result.stderr}", batch=batch)
            _validate_preview_response(payload, prepared, batch=batch)
            record["status"] = "reviewed"
            return payload
        except OpenCLIError as exc:
            last_error = exc
            failure_kind = "response_contract" if isinstance(exc, ReviewResponseError) else "transport"
            record.update(status="unavailable", failure_kind=failure_kind, error=str(exc))
            _emit(log, f"Web footage: Gemini preview attempt {attempt}/{MAX_PREVIEW_REVIEW_ATTEMPTS} "
                  f"unavailable ({failure_kind}): {exc}")
        finally:
            _write_manifest(evidence / f"attempt-{attempt:02d}.json", record)
    raise WebFootageReviewUnavailable(
        f"Gemini footage preview {failure_kind} failure after "
        f"{MAX_PREVIEW_REVIEW_ATTEMPTS} attempts: {last_error}", failure_kind=failure_kind,
    ) from last_error


def _validated_preview(verdict: dict, prepared: dict, task_dir: Path) -> dict:
    intervals, sheet = prepared["intervals"], prepared["sheet"]
    if verdict.get("image_received") is not True:
        raise WebFootageReviewUnavailable("Preview image was not received or readable", failure_kind="response_contract")
    if verdict.get("suitable") is not True:
        raise WebFootageError(f"Preview visual review rejected: {verdict.get('reason', 'No explicit image verdict')}")
    rejection = analysis_rejection(verdict)
    if rejection or not str(verdict.get("visible_content") or "").strip():
        raise WebFootageError(f"Preview visual review rejected: {rejection or 'Missing visible-content evidence'}")
    selected = verdict.get("selected_window")
    if type(selected) is not int or not 0 <= selected < len(intervals):
        raise WebFootageError("Preview visual review did not select a valid sampled interval")
    reviewed = prepared["folder"] / f"window-{selected}" / "preview.mp4"
    artifact = ({"reviewed_preview_path": reviewed.relative_to(task_dir).as_posix(),
                 "reviewed_preview_orientation": "landscape",
                 "reviewed_preview_sha256": _sha256(reviewed)} if reviewed.is_file() else {})
    return {
        **intervals[selected], **verdict, "analyzer": "gemini-web-contact-sheet",
        "status": "analyzed", "preview_evidence": sheet.relative_to(task_dir).as_posix(),
        **artifact,
    }


async def _analyze_preview_batch(
    requests: list[tuple[dict, str]], task_dir: Path, *, log: LogCallback | None = None,
) -> list[dict | Exception]:
    """Review up to four separately labelled candidates in one paced request."""
    from PIL import Image, ImageDraw, ImageFont

    results: list[dict | Exception] = [
        WebFootageReviewUnavailable("Missing or ambiguous batch verdict", failure_kind="response_contract")
        for _ in requests
    ]
    slots = asyncio.Semaphore(2)
    source_locks = {candidate['source_page_url']: asyncio.Lock() for candidate, _ in requests}

    async def prepare(index, candidate, excerpt):
        try:
            # Candidates have separate cache directories. Serialize duplicate
            # URLs so two narration bindings never write the same preview.
            async with source_locks[candidate['source_page_url']], slots:
                _emit(log, f"Web footage preview {index + 1}/{len(requests)}: sampling {candidate.get('title') or candidate['source_page_url']}")
                return index, await _prepare_candidate_preview(candidate, excerpt, task_dir)
        except Exception as exc:
            results[index] = exc
            return None

    jobs = [asyncio.create_task(prepare(index, candidate, excerpt))
            for index, (candidate, excerpt) in enumerate(requests)]
    try:
        prepared = [item for item in await asyncio.gather(*jobs) if item is not None]
    except BaseException:
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        raise
    if not prepared:
        return results
    key = hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()[:16]
    folder = task_dir / "footage" / "evidence" / "batches" / key
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    descriptions = []
    for index, item in prepared:
        with Image.open(item["sheet"]) as source:
            row = Image.new("RGB", (source.width, source.height + 40), "white")
            row.paste(source, (0, 40))
        ImageDraw.Draw(row).text(
            (12, 7), f"CANDIDATE {index} | local rows 0 to {len(item['intervals']) - 1}",
            fill="black", font=ImageFont.load_default(size=24),
        )
        rows.append(row)
        candidate, excerpt = requests[index]
        descriptions.append({"candidate_id": index, "narration": excerpt,
                             "title": candidate.get("title"), "rows": len(item["intervals"]),
                             "visual_subject": candidate.get("visual_query", ""),
                             "visual_purpose": candidate.get("visual_purpose", "")})
    sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
    top = 0
    for row in rows:
        sheet.paste(row, (0, top))
        top += row.height
    sheet_path = folder / "contact-sheet.jpg"
    sheet.save(sheet_path, quality=92)
    prompt = (
        "Review REAL downloaded video frames. Each labelled CANDIDATE has its own narration below. "
        "Within each candidate, rows are local indices starting at zero; each row samples one interval. "
        "Judge each candidate INDEPENDENTLY against ONLY its assigned narration. Judge visible pixels, "
        "not what titles imply. Accept only depictions of narrated subjects/actions or clearly relevant "
        "contextual B-roll that does not misrepresent a different event. Mere theme/keyword overlap "
        "is insufficient. Contextual B-roll illustrates a narrated object or action and the planned "
        "visual subject; it need not prove spoken metrics or funding claims. Never substitute a "
        "different named product or event. "
        "Unrelated visuals, invented events, text-only and talking-head filler must be rejected. "
        "If an image is missing, unreadable or insufficient to establish relevance, suitable must be false. "
        "Return ONLY one complete JSON object {\"results\":[{\"candidate_id\":0,\"image_received\":true,\"suitable\":false,"
        "\"confidence\":0.0,\"selected_window\":0,\"visible_content\":\"concrete description of that row\","
        "\"reason\":\"reason\"}]}. Include one result for every candidate, even rejections. "
        + json.dumps(descriptions, ensure_ascii=False)
    )
    (folder / "request.json").write_text(json.dumps(descriptions, indent=2), encoding="utf-8")
    try:
        payload = await _request_preview_review(
            prompt, sheet_path, dict(prepared), batch=True, log=log,
        )
        (folder / "review.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        verdicts = payload.get("results", [])
        for index, item in prepared:
            matched = [v for v in verdicts if isinstance(v, dict) and type(v.get("candidate_id")) is int and v["candidate_id"] == index]
            if len(matched) != 1:
                continue
            verdict = matched[0]
            (item["folder"] / "review.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
            try:
                _validate_preview_response(verdict, {0: item}, batch=False)
                results[index] = {**_validated_preview(verdict, item, task_dir),
                                  "batch_evidence": sheet_path.relative_to(task_dir).as_posix()}
            except ReviewResponseError as exc:
                results[index] = WebFootageReviewUnavailable(str(exc), failure_kind="response_contract")
            except Exception as exc:
                results[index] = exc
    except Exception as exc:
        for index, _ in prepared:
            results[index] = WebFootageReviewUnavailable(
                str(exc), failure_kind=getattr(exc, "failure_kind", "transport"),
            )
    return results


async def supplement_web_footage(
    *,
    task_dir: Path,
    manifest: dict,
    query_plan: list[dict[str, str]],
    target_total: int,
    orientation: str,
    script: str,
    log: LogCallback | None = None,
    provider_id: int | None = None,
    ai_endpoint: str | None = None,
    ai_model: str | None = None,
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

    replanned_this_run: set[str] = set()
    repair_provider_unavailable = False

    async def replan(shot: dict) -> list[dict]:
        """Change discovery direction using rejection evidence, not a suffix."""
        nonlocal repair_provider_unavailable
        from backend.pipeline.footage import (
            _chat, _fallback_search_queries, _resolve_provider, _sanitize_query,
        )

        original = str(shot.get("plan_query") or shot["query"])
        key = hashlib.sha256((original + "\n" + str(shot.get("script_excerpt"))).encode()).hexdigest()[:16]
        ledger = manifest.setdefault("query_replans", {})
        failures = [error for error in manifest.get("errors", []) if
                    error.get("plan_query", str(error.get("query", "")).removesuffix(" stock footage")) == original
                    and (not shot.get("script_excerpt") or not error.get("script_excerpt")
                         or error["script_excerpt"] == shot["script_excerpt"])]
        previous = ledger.get(key)
        # A retry must learn from failures of the cached alternatives. Keep a
        # proposal stable within one scout, and across transport-only failures
        # whose prepared previews are still waiting for an actual verdict.
        new_failures = [error for error in failures if previous
                        and error not in previous.get("failures", [])
                        and error.get("query") in previous.get("queries", [])
                        and error.get("stage") in {"web-download-edit", "web-selection"}]
        if key not in ledger or (key not in replanned_this_run and (
            previous.get("error") or new_failures
        )):
            replanned_this_run.add(key)
            if previous:
                manifest.setdefault("query_replan_history", []).append({"key": key, **previous})
            previous_queries = {q.casefold() for entry in manifest.get("query_replan_history", [])
                                if entry.get("key") == key for q in entry.get("queries", [])}
            previous_queries.update(str(error.get("query") or "").casefold() for error in failures
                                    if error.get("stage") in {"web-download-edit", "web-selection"})
            previous_queries.discard(original.casefold())
            try:
                if repair_provider_unavailable:
                    raise WebFootageError("Search repair provider unavailable during this scout")
                endpoint, model, api_key = await _resolve_provider(provider_id, ai_endpoint, ai_model)
                answer = await asyncio.wait_for(_chat(
                    "Repair a failed editorial B-roll search. Return JSON only: "
                    '{"queries":["specific search phrase", "different specific search phrase"]}. '
                    "Return at most two 2-6 word queries. Keep the SAME narrated story and visual purpose. "
                    "Use its exact company/product/event, or a concrete narrated process. Read the rejection "
                    "reasons, but treat the supplied narration as authoritative if an old rejection used a wrong story. "
                    "Change the failed direction. Do not merely append stock footage, broaden "
                    "to a generic theme, or substitute an unrelated event. Search for the planned visible "
                    "subject/action: contextual footage need not prove the report's spoken statistics. "
                    "When talking heads were rejected, avoid interviews, press conferences, explainers "
                    "and report presentations; search for the actual depicted activity instead. "
                    "Do not repeat previous_queries_to_avoid. Never claim footage exists.",
                    json.dumps({"query": original, "purpose": shot.get("purpose"),
                                "narration": shot.get("script_excerpt"), "failures": failures[-6:],
                                "previous_queries_to_avoid": sorted(previous_queries)}, ensure_ascii=False),
                    endpoint, model, api_key, log, "Footage search repair", max_tokens=500,
                    enable_skills=False, disable_thinking=True,
                ), timeout=SEARCH_REPAIR_TIMEOUT_SECONDS)
                values = first_json(answer).get("queries", [])
                if not isinstance(values, list):
                    raise WebFootageError("Search repair did not return a queries array")
                alternatives = list(dict.fromkeys(
                    _sanitize_query(value) for value in values if isinstance(value, str)
                    and _sanitize_query(value).casefold() != original.casefold()
                    and _sanitize_query(value).casefold() not in previous_queries
                ))[:2]
                ledger[key] = {"plan_query": original, "queries": [q for q in alternatives if q], "failures": failures[-6:]}
            except Exception as exc:
                repair_provider_unavailable = True
                # Search is an optional proposal, never approval of footage.
                # The already-grounded purpose supplies concrete entity terms
                # while actual frames must still pass the same visual review.
                alternatives = _fallback_search_queries(
                    shot, excluded=previous_queries | {original.casefold()},
                )
                ledger[key] = {"plan_query": original, "queries": alternatives,
                               "error": str(exc) or type(exc).__name__, "fallback": "grounded-purpose"}
                _emit(log, f"Footage search repair unavailable; using {len(alternatives)} grounded purpose queries")
            _write_manifest(manifest_file, manifest)
        return [{**shot, "query": query, "plan_query": original, "_candidate_attempt": 1,
                 "_replanned": True} for query in ledger[key]["queries"]]

    fulfilled = {str(clip.get("plan_query") or clip.get("query") or "") for clip in manifest.get("clips", [])}
    pending_shots = []
    for shot in query_plan:
        shot = {**shot, "plan_query": shot.get("plan_query") or shot["query"]}
        if shot["plan_query"] in fulfilled:
            continue
        prior = [error for error in manifest.get("errors", []) if
                 error.get("plan_query", str(error.get("query", "")).removesuffix(" stock footage")) == shot["plan_query"]
                 and (not error.get("script_excerpt") or error["script_excerpt"] == shot.get("script_excerpt"))
                 and error.get("source_page_url") and error.get("stage") != "web-metadata"]
        # An exhausted legacy search already tried its stock suffix. Resume
        # with corrected discovery, without repeating those paid web reviews.
        if len(prior) >= MAX_CANDIDATE_ATTEMPTS_PER_QUERY:
            pending_shots.extend(await replan(shot))
        else:
            pending_shots.append({**shot, "_candidate_attempt": len(prior) + 1})
    async def choose_candidate(shot: dict, reserved: set[str] | None = None) -> dict | None:
        cached_candidates = []
        for cache_path in (task_dir / "footage" / "evidence" / "previews").glob("*/preview-cache.json"):
            try:
                cached = json.loads(cache_path.read_text()).get("candidate")
                if (isinstance(cached, dict)
                        and cached.get("visual_plan_query", cached.get("visual_query")) == shot["plan_query"]
                        and str(cached.get("visual_purpose") or "") == str(shot.get("purpose") or "")
                        and _load_prepared_preview(cached, cache_path.parent, task_dir) is not None):
                    cached_candidates.append(cached)
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
        # A completed preview is a retained discovery result even if its
        # source falls out of the next search page. It still needs a fresh
        # context-bound visual verdict before it can become a clip.
        results = [*cached_candidates, *await search_youtube(shot["query"])]
        eligible = []
        for candidate in results:
            reason = _metadata_rejection(candidate, shot["query"])
            if reason:
                record = {"query": shot["query"], "plan_query": shot["plan_query"],
                          "script_excerpt": shot.get("script_excerpt", ""),
                          "stage": "web-metadata", "source_page_url": candidate["source_page_url"],
                          "message": reason}
                if record not in manifest.setdefault("errors", []):
                    manifest["errors"].append(record)
                continue
            # A repaired search changes the proposed visible subject, while
            # plan_query remains the stable shot identity used for recovery.
            # Reviewing the old discovery phrase defeats that repair.
            eligible.append({**candidate, "visual_query": shot["query"],
                             "visual_plan_query": shot["plan_query"],
                             "visual_purpose": shot.get("purpose", "")})
        failed = {
            error.get("source_page_url") for error in manifest.get("errors", [])
            if error.get("plan_query", str(error.get("query", "")).removesuffix(" stock footage")) == shot["plan_query"]
            and (not error.get("script_excerpt") or error["script_excerpt"] == shot.get("script_excerpt"))
            and (error.get("stage") != "web-metadata" or error.get("query") == shot["query"])
        }
        available = [item for item in eligible if item["source_page_url"] not in
                     used_sources | failed | (reserved or set())]
        def needs_preview(item: dict) -> bool:
            key = hashlib.sha256(item["source_page_url"].encode()).hexdigest()[:16]
            folder = task_dir / "footage" / "evidence" / "previews" / key
            return _load_prepared_preview(item, folder, task_dir) is None
        # Search ranking can change between retries. Prefer intact evidence
        # anywhere in the eligible results before starting another download.
        return min(available, key=needs_preview, default=None)

    for shot_index, shot in enumerate(pending_shots):
        if len(manifest.get("clips", [])) >= target_total:
            break
        query = str(shot.get("query") or "").strip()
        if shot["plan_query"] in fulfilled:
            continue
        if not query:
            continue
        if manifest.get("url_inspection_unavailable") and "_preview_result" not in shot:
            # Consume every result already paid for before reviewing an
            # alternate query. Otherwise an interleaved alternate can trigger
            # another paced request before its first candidate is committed.
            if any("_preview_result" in item for item in pending_shots[shot_index + 1:]):
                pending_shots.append(shot)
                continue
            batch = []
            reserved_sources: set[str] = set()
            reserved_plans: set[str] = set()
            for upcoming in pending_shots[shot_index:]:
                if upcoming["plan_query"] in fulfilled | reserved_plans or "_preview_result" in upcoming:
                    continue
                try:
                    candidate = await choose_candidate(upcoming, reserved_sources)
                except Exception:
                    continue  # The normal search path records the exact error.
                if candidate is None:
                    continue
                upcoming["_candidate"] = candidate
                upcoming["_excerpt"] = str(upcoming.get("script_excerpt") or matching_script_excerpt(
                    script, f"{upcoming['query']} {upcoming.get('purpose', '')}",
                ))
                batch.append(upcoming)
                reserved_sources.add(candidate["source_page_url"])
                reserved_plans.add(upcoming["plan_query"])
                if len(batch) >= min(4, target_total - len(manifest.get("clips", []))):
                    break
            # Use spare review capacity for alternative candidates of the same
            # missing shot. A single remaining slot must not cost three paced
            # requests simply because its first search result is unsuitable.
            for seed in list(batch):
                preview_key = hashlib.sha256(seed["_candidate"]["source_page_url"].encode()).hexdigest()[:16]
                preview_folder = task_dir / "footage" / "evidence" / "previews" / preview_key
                if _load_prepared_preview(seed["_candidate"], preview_folder, task_dir) is not None:
                    # Resume the ready evidence before downloading speculative
                    # alternatives. Filling a batch must not make one cached
                    # final shot wait for several new source downloads.
                    continue
                previous = seed
                for attempt in range(int(seed.get("_candidate_attempt") or 1) + 1,
                                     MAX_CANDIDATE_ATTEMPTS_PER_QUERY + 1):
                    if len(batch) >= 4:
                        break
                    alternative = {key: value for key, value in seed.items()
                                   if key not in {"_candidate", "_preview_result", "_batch_followup"}}
                    alternative["_candidate_attempt"] = attempt
                    try:
                        candidate = await choose_candidate(alternative, reserved_sources)
                    except Exception:
                        break
                    if candidate is None:
                        break
                    alternative["_candidate"] = candidate
                    previous["_batch_followup"] = True
                    batch.append(alternative)
                    pending_shots.append(alternative)
                    reserved_sources.add(candidate["source_page_url"])
                    previous = alternative
            if batch:
                _emit(log, f"Web footage: reviewing actual preview frames for {len(batch)} candidates in one Gemini request")
                reviews = await _analyze_preview_batch(
                    [(item["_candidate"], item["_excerpt"]) for item in batch], task_dir, log=log,
                )
                for item, review in zip(batch, reviews):
                    item["_preview_result"] = review
        _emit(log, f"Web footage: searching YouTube for '{query}'")
        try:
            candidate = shot.pop("_candidate", None) or await choose_candidate(shot)
        except Exception as exc:  # noqa: BLE001 - record and continue with the next shot
            manifest.setdefault("errors", []).append(
                {"query": query, "stage": "youtube-search", "message": str(exc)}
            )
            continue
        if candidate is None:
            manifest.setdefault("errors", []).append(
                {"query": query, "plan_query": shot["plan_query"],
                 "script_excerpt": shot.get("script_excerpt", ""),
                 "stage": "web-selection", "message": "No unique web candidate found"}
            )
            # Long entity lists can overconstrain YouTube. Broaden only the
            # query, retaining the same narration binding and suitability gate.
            words = query.split()
            if len(words) > 3 and not shot.get("_short_query"):
                shorter = " ".join(words[:3])
                pending_shots.append({**shot, "query": shorter, "_short_query": True})
                _emit(log, f"Web footage: no results; retrying subject query '{shorter}'")
            elif not shot.get("_replanned"):
                pending_shots.extend(await replan(shot))
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
        using_reviewed_preview = False
        preview_result = shot.pop("_preview_result", None)
        shot.pop("_excerpt", None)
        try:
            if isinstance(preview_result, Exception):
                raise preview_result
            if preview_result is not None:
                analysis = preview_result
            elif manifest.get("url_inspection_unavailable"):
                analysis = await _analyze_candidate_preview(candidate, excerpt, task_dir, log=log)
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
                analysis = await _analyze_candidate_preview(candidate, excerpt, task_dir, log=log)
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
            if (analysis.get("reviewed_preview_path")
                    and analysis.get("reviewed_preview_orientation", "landscape") == orientation):
                raw_path = (task_dir / analysis["reviewed_preview_path"]).resolve()
                if (task_dir.resolve() not in raw_path.parents or not raw_path.is_file()
                        or _sha256(raw_path) != analysis.get("reviewed_preview_sha256")):
                    raise WebFootageError("Reviewed preview artifact failed its path/checksum check")
                sectioned = using_reviewed_preview = True
            else:
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
        except (WebFootageReviewUnavailable, OpenCLIError) as exc:
            manifest.setdefault("errors", []).append({
                "query": query, "plan_query": shot["plan_query"],
                "stage": "web-review", "message": str(exc),
                "failure_kind": getattr(exc, "failure_kind", "transport"),
                "pending_source_page_url": candidate["source_page_url"],
            })
            manifest.update(status="review_unavailable", updated_at=_now())
            _write_manifest(manifest_file, manifest)
            raise WebFootageReviewUnavailable(
                "Public-footage visual review unavailable; downloaded previews and verified clips "
                f"were retained. Retry acquisition to resume from saved evidence. {exc}",
                failure_kind=getattr(exc, "failure_kind", "transport"),
            ) from exc
        except Exception as exc:  # noqa: BLE001 - try the next query/candidate
            if "review rejected" in str(exc).casefold():
                manifest.setdefault("rejected_candidates", []).append({
                    "query": query, "plan_query": shot["plan_query"],
                    "source_page_url": candidate["source_page_url"],
                    "script_excerpt": excerpt, "reason": str(exc),
                })
            manifest.setdefault("errors", []).append(
                {
                    "query": query,
                    "plan_query": shot["plan_query"],
                    "script_excerpt": excerpt,
                    "stage": "web-download-edit",
                    "source_page_url": candidate["source_page_url"],
                    "candidate_attempt": candidate_attempt,
                    "message": str(exc),
                }
            )
            manifest["updated_at"] = _now()
            _write_manifest(manifest_file, manifest)
            if shot.get("_batch_followup"):
                _emit(log, f"Web footage candidate failed; consuming the next already-reviewed candidate for '{query}'")
            elif candidate_attempt < MAX_CANDIDATE_ATTEMPTS_PER_QUERY:
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
                if not shot.get("_replanned"):
                    alternatives = await replan(shot)
                    pending_shots.extend(alternatives)
                    _emit(
                        log,
                        f"Web footage exhausted {candidate_attempt} candidates for '{query}'; "
                        f"queued {len(alternatives)} rejection-informed search directions",
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
        if raw_path and raw_path.exists() and not using_reviewed_preview:
            raw_path.unlink()

        entry = {
            "id": clip_id,
            "query": query,
            "plan_query": shot["plan_query"],
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
        fulfilled.add(shot["plan_query"])
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
    manifest["missing_queries"] = [
        {"query": item["query"], "purpose": item.get("purpose", ""),
         "script_excerpt": item.get("script_excerpt", "")}
        for item in query_plan
        if str(item.get("plan_query") or item["query"]) not in fulfilled
    ] if acquired < target_total else []
    manifest["status"] = "ready" if acquired >= target_total else ("partial" if acquired else "no_results")
    manifest["updated_at"] = _now()
    _write_manifest(manifest_file, manifest)
    return manifest
