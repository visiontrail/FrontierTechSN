"""Plain UTF-8 SubRip captions using the final composition's caption timing."""

from __future__ import annotations

import math
from pathlib import Path

from backend.pipeline.assembler import caption_segments


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Subtitle timing must contain finite numbers")
    return float(value)


def _timestamp(milliseconds: int) -> str:
    seconds, millis = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def render_srt(board: dict) -> str:
    """Keep music gaps/intro offsets and exclude the silent source-roll tail.

    These are the same readable groups used by burned-in captions. Within a
    spoken line their times are proportional estimates, not word-level ASR.
    """
    audio_end = _number(board["audio_duration"])
    lines = [line for scene in board["scenes"] for line in scene.get("lines", [])]
    for line in lines:
        start = _number(line["start"])
        duration = _number(line["duration"])
        if start < 0 or duration <= 0 or start + duration > audio_end + 0.02:
            raise ValueError("Subtitle line falls outside the narration timeline")
        if not isinstance(line.get("text"), str) or not line["text"].strip():
            raise ValueError("Subtitle lines must contain narration text")

    cues = caption_segments(board)
    if not cues:
        raise ValueError("No timed narration is available for subtitles")
    blocks = []
    previous_end = 0
    for index, cue in enumerate(cues, 1):
        start = round(cue["start"] * 1000)
        end = round((cue["start"] + cue["duration"]) * 1000)
        if start < previous_end or end <= start or end > round(audio_end * 1000) + 20:
            raise ValueError("Subtitle cues overlap or exceed the narration timeline")
        blocks.append(f"{index}\n{_timestamp(start)} --> {_timestamp(end)}\n{cue['text']}\n")
        previous_end = end
    return "\n".join(blocks) + "\n"


def write_subtitles(board: dict, video_path: Path) -> Path:
    """Bind each sidecar to its versioned video; never use a mutable task alias."""
    content = render_srt(board)
    path = video_path.with_suffix(".srt")
    staged = path.with_suffix(".srt.tmp")
    staged.write_text(content, encoding="utf-8")
    staged.replace(path)
    return path
