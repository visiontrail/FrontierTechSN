"""Sample-preserving editorial trims for independently synthesized scenes.

Only contiguous, near-silent edge windows can be removed. Pauses inside speech
and the provider's original recording are never rewritten.
"""
from __future__ import annotations

import math
import sys
import wave
from array import array
from pathlib import Path

VERSION = 1
SAMPLE_RATE = 48000
FPS = 30
WINDOW_SAMPLES = 480  # 10 ms; peak, not RMS, protects quiet consonants.
THRESHOLD_DBFS = -60


def pacing_plan(path: Path, *, chapter_end: bool = False, closing_hold: float = 0,
                verified_speech_end: float | None = None) -> dict:
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (1, 2, SAMPLE_RATE, "NONE"):
            raise ValueError("Explainer pacing requires mono 48 kHz PCM16 narration")
        count = source.getnframes()
        samples = array("h", source.readframes(count))
    if sys.byteorder != "little":
        samples.byteswap()
    threshold = 32767 * 10 ** (THRESHOLD_DBFS / 20)
    audible = [start for start in range(0, count, WINDOW_SAMPLES)
               if max((abs(v) for v in samples[start:start + WINDOW_SAMPLES]), default=0) >= threshold]
    if not audible:
        raise ValueError("Explainer narration is silent; refusing an empty scene")
    first, last = audible[0], min(count, audible[-1] + WINDOW_SAMPLES)
    if verified_speech_end is not None:
        if not math.isfinite(verified_speech_end) or not 0 < verified_speech_end <= count / SAMPLE_RATE:
            raise ValueError("Verified speech edge is outside narration")
        # The independent source-bound speech audit protects the last word.
        # Beyond that guard only near-silent samples may be discarded; a later
        # audible sound remains in the recording for inspection.
        guard = min(count, math.ceil((verified_speech_end + 0.12) * SAMPLE_RATE / WINDOW_SAMPLES) * WINDOW_SAMPLES)
        threshold_after_speech = 32767 * 10 ** (-42 / 20)
        later = [start for start in range(guard, count, WINDOW_SAMPLES)
                 if max((abs(v) for v in samples[start:start + WINDOW_SAMPLES]), default=0) >= threshold_after_speech]
        last = min(last, max(guard, min(count, later[-1] + WINDOW_SAMPLES) if later else guard))
    # Preserve breathing room around the first/last audible window. Never pad
    # an ordinary boundary simply to meet a target pause.
    start = max(0, first - round(0.08 * SAMPLE_RATE))
    tail = 0.45 if chapter_end or closing_hold else 0.22
    end = min(count, last + round(tail * SAMPLE_RATE))
    kept = end - start
    hold_samples = round(closing_hold * SAMPLE_RATE)
    frame_samples = SAMPLE_RATE // FPS
    frames = math.ceil((kept + hold_samples) / frame_samples)
    return {
        "version": VERSION, "threshold_dbfs": THRESHOLD_DBFS,
        "method": "contiguous_edge_peak_windows_only", "window_samples": WINDOW_SAMPLES,
        "source_samples": count, "start_sample": start, "end_sample": end,
        "removed_leading_seconds": start / SAMPLE_RATE,
        "removed_trailing_seconds": (count - end) / SAMPLE_RATE,
        "speech_start_seconds": (first - start) / SAMPLE_RATE,
        "speech_end_seconds": (last - start) / SAMPLE_RATE,
        "narration_duration_seconds": kept / SAMPLE_RATE,
        "closing_hold_seconds": closing_hold, "chapter_end": chapter_end,
        "verified_speech_end_seconds": verified_speech_end,
        "post_speech_threshold_dbfs": -42 if verified_speech_end is not None else None,
        "frames": frames, "duration_seconds": frames / FPS,
        "output_samples": frames * frame_samples,
    }


def write_paced_audio(source_path: Path, destination: Path, plan: dict) -> None:
    with wave.open(str(source_path), "rb") as source:
        source.setpos(plan["start_sample"])
        pcm = source.readframes(plan["end_sample"] - plan["start_sample"])
    with wave.open(str(destination), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(pcm)
        target.writeframes(b"\0\0" * (plan["output_samples"] - len(pcm) // 2))
