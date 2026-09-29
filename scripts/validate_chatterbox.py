#!/usr/bin/env python3
"""Prepare, run, or audit an isolated production TTS-to-video acceptance sample.

No task/default/publication records are changed. Run uses the existing configured
AI providers and browser review, so ordinary inference usage still applies.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.pipeline import composer, music, outros, tts  # noqa: E402
from backend.pipeline.video_format import LANDSCAPE, resolve_render_spec  # noqa: E402

SCRIPT = """Good morning. This is ByteFront Espresso, with a short technology update.
Voice synthesis turns written text into speech. Each sentence keeps its natural pace, while the video follows the measured audio. Clear captions and carefully timed visuals help the story remain easy to follow.
Thanks for watching. We will see you tomorrow.
"""
TITLE = "Natural speech, measured video timing"
CONTRACT_VERSION = 1
SAMPLE_RATE = 24000
TERMINAL = {"complete", "failed", "awaiting_review", "cancelled"}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def output_directory(value: str | None) -> Path:
    base = (config.OUTPUTS_DIR / "chatterbox-validation").resolve()
    candidate = Path(value).absolute() if value else base / (
        datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    )
    if any(item.is_symlink() for item in (candidate, *candidate.parents)):
        raise ValueError("Validation output must not use symlinks")
    directory = candidate.resolve()
    if not directory.is_relative_to(base) or directory == base:
        raise ValueError(f"Use a new child directory under {base}")
    return directory


def require_profile() -> dict:
    profile = resolve_render_spec(LANDSCAPE).to_dict()
    if (profile["width"], profile["height"], profile["fps"]) != (3840, 2160, 30):
        raise ValueError("The configured delivery must be 3840x2160 at 30 fps; settings were not changed")
    if not config.AV_SYNC_GEMINI_REVIEW_ENABLED:
        raise ValueError("Production multimodal review must remain enabled")
    if config.AV_SYNC_LANGUAGE != "en":
        raise ValueError("This English validation requires the existing English ASR configuration")
    return profile


def prepare(directory: Path, *, model: str, voice: str, track: str, captions: bool) -> dict:
    request = {
        "contract_version": CONTRACT_VERSION, "kind": "isolated_tts_video_acceptance",
        "script": SCRIPT, "title": TITLE, "tts_model": model, "voice": voice,
        "music_track_id": track, "captions_enabled": captions,
        "video_template": "swiss", "outro_style": "morning-brief",
        "render_profile": require_profile(), "publication": "none",
    }
    if voice not in config.voices_for_model(model):
        raise ValueError(f"Voice {voice!r} is unavailable for {model!r}")
    music.resolve_program_music_track(track)
    outros.resolve_outro_preset("morning-brief")
    marker = directory / "validation_request.json"
    if directory.exists() and any(directory.iterdir()):
        if not marker.is_file() or read_json(marker) != request:
            raise ValueError("Existing output belongs to another request; use a new directory")
        if (directory / "script.txt").read_text(encoding="utf-8") != SCRIPT:
            raise ValueError("The prepared script was changed; use a new validation directory")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "script.txt").write_text(SCRIPT, encoding="utf-8")
        write_json(marker, request)
    return request


async def require_idle(app_url: str) -> dict:
    import httpx

    async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
        tasks_response, queue_response = await asyncio.gather(
            client.get(f"{app_url.rstrip('/')}/api/tasks"),
            client.get(f"{app_url.rstrip('/')}/api/explainers/queue"),
        )
        tasks_response.raise_for_status()
        queue_response.raise_for_status()
    tasks = tasks_response.json()["tasks"]
    queue = queue_response.json()
    busy = [{"id": task["id"], "status": task["status"]} for task in tasks if task["status"] not in TERMINAL]
    if busy or queue.get("active") or queue.get("items"):
        raise RuntimeError("The shared production queue is active; retry validation after it is idle")
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "active_tasks": busy,
            "explainer_active": queue.get("active"), "queued_explainers": len(queue.get("items") or [])}


def command(args: list[str], *, timeout: int = 180) -> bytes:
    result = subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return result.stdout


def probe(path: Path) -> dict:
    return json.loads(command(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))


def decode_audio(path: Path):
    import numpy as np

    data = command(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1",
                    "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"])
    return np.frombuffer(data, dtype="<f4").astype(np.float64)


def compare_audio(reference, observed, *, maximum_lag_seconds: float = 0.05) -> dict:
    """Find a small codec/boundary offset, then compare without changing speed."""
    import numpy as np

    if min(len(reference), len(observed)) < SAMPLE_RATE // 10:
        raise ValueError("Audio comparison needs at least 100 milliseconds")
    count = min(len(reference), len(observed))
    # FFT correlation avoids an expensive sample-by-sample lag search. No
    # resampling or time warping is performed during alignment.
    size = 1 << (2 * count - 1).bit_length()
    correlation = np.fft.irfft(np.fft.rfft(observed[:count], size) *
                              np.conj(np.fft.rfft(reference[:count], size)), size)
    maximum = round(maximum_lag_seconds * SAMPLE_RATE)
    lags = np.arange(-maximum, maximum + 1)
    lag = int(lags[np.argmax(correlation[lags % size])])
    left = reference[max(0, -lag):]
    right = observed[max(0, lag):]
    count = min(len(left), len(right))
    # Segment resampling can affect boundary filters; inspect unchanged speech
    # away from a 20 ms boundary margin and report the excluded margin.
    margin = round(0.02 * SAMPLE_RATE)
    left = left[margin:count - margin]
    right = right[margin:count - margin]
    power = float(np.dot(left, left))
    observed_power = float(np.dot(right, right))
    if power <= 0 or observed_power <= 0:
        raise ValueError("Silent audio cannot prove narration preservation")
    dot = float(np.dot(left, right))
    gain = dot / power
    return {
        "reference_frames": len(reference), "observed_frames": len(observed),
        "frame_ratio": len(observed) / len(reference), "sample_rate": SAMPLE_RATE,
        "alignment_lag_frames": lag, "alignment_lag_seconds": lag / SAMPLE_RATE,
        "correlation": dot / (power * observed_power) ** 0.5,
        "best_fit_gain": gain,
        "relative_residual_rms": float(np.linalg.norm(right - gain * left) / np.linalg.norm(right)),
        "boundary_margin_seconds": margin / SAMPLE_RATE,
        "comparison_policy": "offset_only_no_time_warp",
    }


def _inside(directory: Path, value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Validation media reference escaped its own output directory")
    return path


def audit(directory: Path) -> dict:
    request = read_json(directory / "validation_request.json")
    delivery = read_json(directory / "validation_delivery.json")
    video = _inside(directory, delivery["video_path"])
    raw_audio = _inside(directory, delivery["audio_path"])
    manifest = read_json(raw_audio.parent / "tts_manifest.json")
    quality = read_json(directory / "av_sync_report.json")
    board = read_json(directory / "storyboard.json")
    pacing = read_json(directory / "audio" / "program_pacing_report.json")
    mix = read_json(directory / "audio" / "music_mix_report.json")
    failures = composer._narration_manifest_failures(directory / "script.txt", raw_audio, request["tts_model"])
    raw_info = tts._read_pcm_wav(raw_audio)
    if (raw_info.sample_rate, raw_info.channels, raw_info.sample_width) != (24000, 1, 2):
        failures.append("source narration is not 24 kHz mono PCM16")
    if quality.get("passed") is not True or quality.get("rendered_video_sha256") != sha256(video):
        failures.append("final quality report does not pass or does not match the video")
    if quality.get("quality_status") != "passed" or quality.get("delivery_status") != "completed":
        failures.append("production quality and delivery states are not final")
    if quality.get("multimodal", {}).get("passed") is not True:
        failures.append("production multimodal review has not passed")
    if not pacing.get("passed") or not mix.get("passed"):
        failures.append("program pacing or music mix did not pass")
    expected_filename = f"video-{sha256(video)[:16]}.mp4"
    if video.name != expected_filename:
        failures.append("video was not promoted to the production hash-versioned filename")
    render = resolve_render_spec(LANDSCAPE)
    failures.extend(composer._rendered_video_failures(
        video, frame=LANDSCAPE, expected_duration=float(board["total_duration"]), render=render,
    ))
    subtitle = video.with_suffix(".srt")
    if not subtitle.is_file() or " --> " not in subtitle.read_text(encoding="utf-8"):
        failures.append("exported subtitle track is missing or malformed")

    source = decode_audio(raw_audio)
    paced = decode_audio(_inside(directory, pacing["paced_narration_path"]))
    segment_reports = []
    for segment in pacing["segments"]:
        start, end = [round(float(segment[key]) * SAMPLE_RATE) for key in ("source_start", "source_end")]
        program_start, program_end = [round(float(segment[key]) * SAMPLE_RATE) for key in ("program_start", "program_end")]
        result = compare_audio(source[start:end], paced[program_start:program_end], maximum_lag_seconds=0.005)
        result.update(index=segment["index"], kind=segment["kind"])
        segment_reports.append(result)
        if result["correlation"] < 0.995 or abs((program_end - program_start) - (end - start)) > 2:
            failures.append(f"source-to-paced narration did not preserve segment {segment['index']} at 1.0x")
    program = decode_audio(_inside(directory, mix["program_mix_path"]))
    rendered = decode_audio(video)
    offset = round(float(board.get("content_start") or 0) * SAMPLE_RATE)
    rendered = rendered[offset:]
    final_audio = compare_audio(program, rendered)
    # AAC framing and the 30 fps final-frame boundary can add a short tail.
    if final_audio["correlation"] < 0.97 or abs(len(program) - len(rendered)) / SAMPLE_RATE > 0.09:
        failures.append("rendered AAC does not retain the exact program mix at its original duration")
    report = {
        "contract_version": CONTRACT_VERSION, "kind": "isolated_tts_video_acceptance",
        "passed": not failures, "failure_reasons": failures,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "video_path": str(video), "video_sha256": sha256(video), "audio_path": str(raw_audio),
        "tts_model": request["tts_model"], "voice": request["voice"],
        "captions_enabled": request["captions_enabled"], "source_word_count": len(SCRIPT.split()),
        "render_profile": quality.get("render_profile"), "ffprobe": probe(video),
        "source_narration": {"sha256": sha256(raw_audio), "frames": raw_info.frame_count,
                             "sample_rate": raw_info.sample_rate, "duration_seconds": raw_info.duration_seconds,
                             "synthesis_speed_ratio": manifest.get("synthesis_speed_ratio"),
                             "integrity": {key: value for key, value in manifest.get("integrity", {}).items()
                                           if key != "part_reports"}},
        "source_to_paced_narration": segment_reports, "program_mix_to_rendered_aac": final_audio,
        "program_pacing_report": str(directory / "audio" / "program_pacing_report.json"),
        "music_mix_report": str(directory / "audio" / "music_mix_report.json"),
        "av_sync_report": str(directory / "av_sync_report.json"), "subtitle_path": str(subtitle),
        "scope": "Production provider, composer, rendering and quality gates; no application task or publication created",
        "browser_playback": "pending separate Ego Lite playback verification",
    }
    write_json(directory / "validation_report.json", report)
    return report


async def run(directory: Path, request: dict, *, app_url: str, attempts: int) -> dict:
    def log(message: str) -> None:
        print(message, flush=True)
        with (directory / "validation.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    write_json(directory / "idle_before_tts.json", await require_idle(app_url))
    require_profile()
    audio = await tts.generate_tts(str(directory / "script.txt"), str(directory / "audio"),
                                   voices=[request["voice"]], tts_model=request["tts_model"], log=log)
    background = await music.generate_background_music(
        directory.name, directory, title=request["title"], summary=None,
        provider="local_library", track_id=request["music_track_id"], log=log,
    )
    video = None
    for attempt in range(1, attempts + 1):
        write_json(directory / "idle_before_compose.json", await require_idle(app_url))
        try:
            video = await composer.compose_video(
                script_path=str(directory / "script.txt"), audio_path=audio, output_dir=str(directory),
                title=request["title"], tts_model=request["tts_model"], is_monologue=True,
                video_template=request["video_template"], video_orientation="landscape",
                captions_enabled=request["captions_enabled"], opening_style="editorial_motion",
                outro_style=request["outro_style"], collage_broll_enabled=False,
                news_images_enabled=False, footage_enabled=False,
                background_music_path=background.audio_path, background_music_bed_db=-25.0,
                background_music_duck_db=-11.0, program_music_pacing_enabled=True,
                program_music_intro_seconds=2.0, program_music_opening_gap_seconds=3.0,
                program_music_story_gap_seconds=1.5, log=log,
            )
            break
        except composer.QualityGateRetry:
            if attempt == attempts:
                raise
            log(f"Production quality gate requested compose retry {attempt + 1}/{attempts}; retained evidence stays in this directory")
    write_json(directory / "validation_delivery.json", {"audio_path": audio, "video_path": video})
    return await asyncio.to_thread(audit, directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "audit"))
    parser.add_argument("--output-dir", help="Owned child of outputs/chatterbox-validation; required for run/audit")
    parser.add_argument("--model", choices=("chatterbox-multilingual", "pocket-tts-en"), default="chatterbox-multilingual")
    parser.add_argument("--voice", default="jane")
    parser.add_argument("--music-track", default="morning-briefing")
    parser.add_argument("--no-captions", action="store_true", help="Match the latest Daily News burned-caption setting; SRT still exported")
    parser.add_argument("--app-url", default="http://localhost:8101")
    parser.add_argument("--deployment-fingerprint", help="Validate a candidate fingerprint in this process without changing saved application settings")
    parser.add_argument("--compose-attempts", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    if args.deployment_fingerprint:
        fingerprint = args.deployment_fingerprint
        if args.mode != "run" or args.model != "chatterbox-multilingual":
            parser.error("--deployment-fingerprint is only for a Chatterbox run")
        if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
            parser.error("--deployment-fingerprint must be a complete lowercase SHA-256")
        config.CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT = fingerprint
    if args.mode != "prepare" and not args.output_dir:
        parser.error("--output-dir is required for run/audit so an existing owned run is resumed")
    directory = output_directory(args.output_dir)
    if args.mode == "audit":
        report = audit(directory)
    else:
        request = prepare(directory, model=args.model, voice=args.voice,
                          track=args.music_track, captions=not args.no_captions)
        if args.mode == "prepare":
            print(json.dumps({"output_dir": str(directory), "source_words": len(SCRIPT.split()),
                              "request": request}, indent=2))
            return 0
        # Prevent two validators from competing. This does not claim or mutate
        # the production queue; idle checks run before TTS and every compose.
        with (directory.parent / ".runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report = asyncio.run(run(directory, request, app_url=args.app_url, attempts=args.compose_attempts))
    print(json.dumps({"passed": report["passed"], "video_path": report["video_path"],
                      "report": str(directory / "validation_report.json"),
                      "failure_reasons": report["failure_reasons"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
