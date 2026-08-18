"""Acoustic timing and audio/visual quality gates.

The script is the semantic source of truth, but it is not a clock. TTS delivery
changes with punctuation, numbers, names and pauses, so word-count timing and a
raw silence list both drift on long narration. This module transcribes the
rendered WAV to word timestamps, caches the result next to the task, and lets
``storyboard`` force-align the canonical script to that acoustic clock.

MLX Whisper is used on Apple Silicon because it is fast and can reuse the
operator's Hugging Face cache. It is a post-TTS analyzer only: the selected TTS
model remains the narration source. If bounded transcription
retries are exhausted, the caller receives a warning result and continues with
estimated timing. If a transcript is available but covers too little of the
script/audio, the composer blocks rendering as an incomplete narration;
HyperFrames is not used for transcription.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import wave
from collections.abc import Callable
from pathlib import Path

from backend import config
from backend.pipeline.process_logging import stream_subprocess

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

TRANSCRIPT_NAME = "transcript.json"
TRANSCRIPT_META_NAME = "transcript.meta.json"
TRANSCRIBE_TIMEOUT = 1800
TRANSCRIBE_STALL_TIMEOUT = 900
NON_SPEECH_TOKENS = frozenset({"♪", "♫", "♬", "�"})
TAIL_RESCAN_SECONDS = 15.0


def _emit(log: LogCallback | None, message: str) -> None:
    if log:
        log(message)
    else:
        logger.info(message)


def _audio_signature(audio_path: Path) -> dict[str, int]:
    stat = audio_path.stat()
    return {"audio_size": stat.st_size, "audio_mtime_ns": stat.st_mtime_ns}


def _load_words(path: Path) -> list[dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(value, list):
        return []

    words: list[dict] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or entry.get("word") or "").strip()
        try:
            start = float(entry["start"])
            end = float(entry["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not text or end <= start:
            continue
        words.append({"text": text, "start": round(max(0.0, start), 3), "end": round(end, 3)})
    words.sort(key=lambda word: (word["start"], word["end"]))
    return words


def _transcript_quality(words: list[dict], *, minimum_words: int = 3) -> tuple[bool, str]:
    # Short promos can legitimately contain fewer than ten spoken words. The
    # later forced-alignment gate compares transcript coverage with the actual
    # script, so this preflight only needs enough samples to reject empty or
    # obviously broken Whisper output without rejecting complete short clips.
    if len(words) < minimum_words:
        return False, f"only {len(words)} timestamped words"
    non_speech = sum(1 for word in words if word["text"].strip() in NON_SPEECH_TOKENS)
    ratio = non_speech / len(words)
    if ratio > 0.20:
        return False, f"{ratio:.1%} non-speech/music tokens"
    return True, ""


def _cache_is_current(task_dir: Path, audio_path: Path) -> bool:
    transcript_path = task_dir / TRANSCRIPT_NAME
    meta_path = task_dir / TRANSCRIPT_META_NAME
    if not transcript_path.is_file():
        return False
    signature = _audio_signature(audio_path)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Backwards-compatible reuse of a transcript made after the WAV.
        return transcript_path.stat().st_mtime_ns >= audio_path.stat().st_mtime_ns
    return all(meta.get(key) == value for key, value in signature.items())


def _python_can_import(python: Path, module: str) -> bool:
    if not python.is_file():
        return False
    try:
        return (
            subprocess.run(
                [str(python), "-c", f"import {module}"],
                capture_output=True,
                timeout=20,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def _mlx_command(audio_path: Path, task_dir: Path) -> list[str] | None:
    python = Path(sys.executable)
    if importlib.util.find_spec("mlx_whisper") is None:
        # The TTS installation already carries torch/numpy/scipy, so setup may
        # install the small MLX-specific layer there to avoid duplicating
        # several gigabytes in the application venv.
        candidates = [
            config.AIWORK_ROOT / "venvs" / "vibevoice" / "bin" / "python",
            config.AIWORK_ROOT / "venvs" / "vibevoice-1.5b" / "bin" / "python",
        ]
        python = next(
            (candidate for candidate in candidates if _python_can_import(candidate, "mlx_whisper")),
            Path(),
        )
    if not python.is_file():
        return None
    return [
        str(python),
        "-m",
        "backend.pipeline.av_sync",
        "transcribe-mlx",
        str(audio_path),
        str(task_dir / TRANSCRIPT_NAME),
        "--model",
        config.AV_SYNC_MLX_MODEL,
        "--language",
        config.AV_SYNC_LANGUAGE,
    ]


async def ensure_word_transcript(
    audio_path: str | Path,
    task_dir: str | Path,
    *,
    log: LogCallback | None = None,
    minimum_words: int = 3,
) -> tuple[list[dict], dict]:
    """Return MLX word timestamps or a non-fatal warning result."""
    audio = Path(audio_path).resolve()
    directory = Path(task_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    transcript_path = directory / TRANSCRIPT_NAME

    if _cache_is_current(directory, audio):
        words = _load_words(transcript_path)
        good, reason = _transcript_quality(words, minimum_words=minimum_words)
        if good:
            _emit(log, f"A/V sync: reusing {len(words)} cached word timestamps")
            return words, {"backend": "cache", "word_count": len(words), "passed": True}
        _emit(log, f"A/V sync: cached transcript rejected ({reason}); regenerating")

    command = _mlx_command(audio, directory)
    failures: list[str] = []
    if command is None:
        failures.append("mlx-whisper: runtime unavailable")
        _emit(
            log,
            "A/V sync warning: MLX Whisper is unavailable; continuing with "
            "estimated timing. The generated narration is unchanged.",
        )
        return [], {
            "backend": "unavailable",
            "passed": False,
            "attempts": 0,
            "failure_reasons": failures,
        }

    maximum_attempts = max(1, int(config.AV_SYNC_TRANSCRIBE_MAX_RETRIES) + 1)
    for attempt in range(1, maximum_attempts + 1):
        _emit(
            log,
            f"A/V sync: transcribing finished narration with MLX Whisper "
            f"(attempt {attempt}/{maximum_attempts})",
        )
        try:
            returncode, output = await stream_subprocess(
                name="A/V transcription (mlx-whisper)",
                command=command,
                logger=logger,
                log=log,
                cwd=config.PROJECT_ROOT,
                timeout=TRANSCRIBE_TIMEOUT,
                stall_timeout=TRANSCRIBE_STALL_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001 - retry, then continue with warning
            failures.append(f"mlx-whisper attempt {attempt}: {exc}")
            continue
        if returncode != 0:
            failures.append(
                f"mlx-whisper attempt {attempt}: exit {returncode}: {output[-300:]}"
            )
            continue
        words = _load_words(transcript_path)
        good, reason = _transcript_quality(words, minimum_words=minimum_words)
        if not good:
            failures.append(f"mlx-whisper attempt {attempt}: {reason}")
            continue
        meta = {
            **_audio_signature(audio),
            "backend": "mlx-whisper",
            "word_count": len(words),
            "model": config.AV_SYNC_MLX_MODEL,
            "attempts": attempt,
        }
        (directory / TRANSCRIPT_META_NAME).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _emit(log, f"A/V sync: {len(words)} word timestamps ready via mlx-whisper")
        return words, {**meta, "passed": True}

    detail = "; ".join(failures)
    _emit(
        log,
        "A/V sync warning: MLX Whisper retries were exhausted; continuing with "
        "estimated timing. The generated narration is unchanged. " + detail,
    )
    return [], {
        "backend": "unavailable",
        "passed": False,
        "attempts": maximum_attempts,
        "failure_reasons": failures,
    }


def _result_words(result: dict) -> list[dict]:
    words: list[dict] = []
    for segment in result.get("segments") or []:
        for word in segment.get("words") or []:
            text = str(word.get("word") or word.get("text") or "").strip()
            if not text:
                continue
            words.append(
                {
                    "text": text,
                    "start": round(float(word["start"]), 3),
                    "end": round(float(word["end"]), 3),
                }
            )
    return words


def _replace_tail_window(
    primary_words: list[dict], tail_words: list[dict], tail_start: float
) -> list[dict]:
    """Replace the long-pass tail with a focused tail scan without duplicates."""
    if not tail_words:
        return primary_words
    head = [
        word
        for word in primary_words
        if (float(word["start"]) + float(word["end"])) / 2 < tail_start
    ]
    return sorted([*head, *tail_words], key=lambda word: (word["start"], word["end"]))


def _write_mlx_transcript(audio_path: Path, output_path: Path, model: str, language: str) -> None:
    """Subprocess entry point so model memory is released before rendering.

    Whisper can classify a short sentence after a pause at the very end of a
    long recording as silence. A focused final-window pass catches that case;
    it replaces the overlapping long-pass words, so the transcript remains
    monotonic and duplicate-free.
    """
    import mlx_whisper

    print(f"Loading MLX Whisper model {model}", flush=True)
    options = {
        "path_or_hf_repo": model,
        "word_timestamps": True,
        "language": language,
        "verbose": False,
        "condition_on_previous_text": False,
    }
    result = mlx_whisper.transcribe(str(audio_path), **options)
    words = _result_words(result)

    with wave.open(str(audio_path), "rb") as handle:
        duration = handle.getnframes() / max(1, handle.getframerate())
    if duration > TAIL_RESCAN_SECONDS * 2:
        tail_start = max(0.0, duration - TAIL_RESCAN_SECONDS)
        print(
            f"Rescanning final {TAIL_RESCAN_SECONDS:.0f}s for post-pause closing speech",
            flush=True,
        )
        tail_result = mlx_whisper.transcribe(
            str(audio_path), clip_timestamps=[tail_start], **options
        )
        words = _replace_tail_window(words, _result_words(tail_result), tail_start)

    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(words, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, output_path)
    print(f"Wrote {len(words)} word timestamps to {output_path}", flush=True)


def _main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    mlx = subparsers.add_parser("transcribe-mlx")
    mlx.add_argument("audio")
    mlx.add_argument("output")
    mlx.add_argument("--model", required=True)
    mlx.add_argument("--language", default="en")
    args = parser.parse_args()
    if args.command == "transcribe-mlx":
        _write_mlx_transcript(
            Path(args.audio).resolve(),
            Path(args.output).resolve(),
            args.model,
            args.language,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
