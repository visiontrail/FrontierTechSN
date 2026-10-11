"""Detect sound that ASR omitted after a separated speech ending.

This read-only gate never authorizes cutting speech. Suspicious provider audio
must be regenerated or repaired with separately verified delivery evidence.
"""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

VERSION = 2


def inspect_tail(path: Path, words: list[dict]) -> dict:
    ends = [float(word["end"]) for word in words
            if isinstance(word.get("end"), (int, float))
            and math.isfinite(word["end"]) and word["end"] >= 0]
    # Whisper can place an opening token at [0, 0]. Tail inspection only
    # needs the last observed speech edge; retain all lexical observations
    # and require a positive edge elsewhere instead of rejecting that token.
    if not ends or len(ends) != len(words) or max(ends) <= 0:
        raise RuntimeError("Tail waveform observations unavailable; preserving the completed job")
    with wave.open(str(path), "rb") as reader:
        if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
            raise RuntimeError("Tail waveform requires PCM16 audio")
        rate, channels, count = reader.getframerate(), reader.getnchannels(), reader.getnframes()
        raw = reader.readframes(count)
    if len(raw) != count * channels * 2:
        raise RuntimeError("Incomplete tail waveform; preserving the completed job")
    speech_end = max(ends)
    if speech_end > count / rate + 0.05:
        raise RuntimeError("Tail timestamps exceed the waveform; preserving the completed job")
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
    window = max(1, round(rate * 0.01))
    start = math.ceil((speech_end + 0.12) * rate / window) * window
    stop = start + max(0, (count - start) // window) * window
    report = {"version": VERSION, "passed": True, "speech_end_seconds": speech_end,
              "guard_seconds": 0.12, "minimum_quiet_seconds": 0.12,
              "minimum_residual_seconds": 0.06, "quiet_rms_dbfs": -42,
              "quiet_peak_dbfs": -30, "residual_rms_dbfs": -35,
              "residual_start_seconds": None}
    if stop <= start:
        return report
    windows = samples[start:stop].astype(float).reshape(-1, window, channels)
    rms = np.sqrt(np.mean(windows ** 2, axis=1)).max(axis=1)
    peak = np.max(np.abs(windows), axis=(1, 2))
    quiet = (rms <= 32768 * 10 ** (-42 / 20)) & (peak <= 32768 * 10 ** (-30 / 20))
    active = rms > 32768 * 10 ** (-35 / 20)
    quiet_run = active_run = 0
    separated = False
    for index in range(len(windows)):
        quiet_run = quiet_run + 1 if quiet[index] else 0
        if quiet_run >= 12:
            separated = True
        active_run = active_run + 1 if separated and active[index] else 0
        if active_run >= 6:
            report.update(passed=False, residual_start_seconds=(start + (index - 5) * window) / rate)
            break
    return report
