"""Build a bounded-memory music bed from recurring interior audio, never file seams."""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np

RATE = 48_000
HOP = 480


def _loop_points(samples: np.ndarray) -> tuple[int, int, int, float]:
    # Compare entire overlapping passages, including their transient envelopes.
    # This is recurrence matching, not a claim to infer meter or musical quality.
    mono = samples.mean(axis=1)
    frames = np.lib.stride_tricks.sliding_window_view(mono, 2048)[::HOP]
    power = abs(np.fft.rfft(frames * np.hanning(2048), axis=1)) ** 2
    edges = np.unique(np.geomspace(1, power.shape[1], 25).astype(int))
    bands = np.stack([power[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    level = np.sqrt((frames ** 2).mean(axis=1))
    active = np.flatnonzero(level > max(float(np.percentile(level, 80)) * 0.18, 1e-5))
    if len(active) < 20:
        raise ValueError("Music has too little active audio to loop")
    first, last = int(active[0]), int(active[-1])
    span = last - first
    overlap = min(200, max(1, span // 8))
    # Log spectra capture recurring harmony/timbre; explicit level comparison
    # prevents matching a fade-out to a quiet introduction.
    features = np.log(np.maximum(bands, 1e-8))
    features -= features.mean(axis=1, keepdims=True)
    min_period = max(overlap * 2, int(span * 0.45))
    step = max(10, span // 300)
    starts = range(first, max(first + 1, first + span // 3 - overlap), step)
    best = (float("inf"), first, max(first + min_period, last - overlap))

    def score(a: int, b: int) -> float:
        spectral = float(np.mean((features[a:a + overlap] - features[b:b + overlap]) ** 2))
        envelope = float(np.mean(np.log((level[a:a + overlap] + 1e-7) / (level[b:b + overlap] + 1e-7)) ** 2))
        return spectral + 4 * envelope

    for a in starts:
        for b in range(a + min_period, last - overlap + 1, step):
            value = score(a, b)
            if value < best[0]:
                best = (value, a, b)
    while step > 1:
        _, a0, b0 = best
        radius, step = step, max(1, step // 10)
        for a in range(max(first, a0 - radius), min(a0 + radius + 1, last - overlap), step):
            for b in range(max(a + min_period, b0 - radius), min(b0 + radius + 1, last - overlap + 1), step):
                value = score(a, b)
                if value < best[0]:
                    best = (value, a, b)
    value, a, b = best
    return a * HOP, b * HOP, overlap * HOP, value


def render_music_bed(decoded: Path, output: Path, duration: float) -> dict:
    """Read short float PCM source, write exact-length PCM in cycle-sized blocks."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Music bed duration must be positive and finite")
    samples = np.fromfile(decoded, dtype='<f4').reshape(-1, 2)
    count = round(duration * RATE)
    if len(samples) < RATE // 4 or not np.isfinite(samples).all():
        raise ValueError("Music source is too short or contains invalid samples")
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < 1e-6:
        raise ValueError("Music source is silent")
    report = {"source_duration_seconds": len(samples) / RATE, "duration_seconds": count / RATE}
    if count <= len(samples):
        cycle = samples[:count].copy()
        report.update(mode="single_pass", seam_seconds=[])
    else:
        start, end, overlap, score = _loop_points(samples)
        head, tail = samples[start:start + overlap], samples[end:end + overlap]
        t = np.linspace(0, np.pi / 2, overlap, dtype=np.float64)[:, None]
        a, b = np.cos(t), np.sin(t)
        # Equal-power for unrelated audio; correlated material needs less gain.
        # Negative correlation is not boosted (avoid amplifying cancellation).
        correlation = float(np.sum(head * tail) / max(np.linalg.norm(head) * np.linalg.norm(tail), 1e-12))
        blend = (tail * a + head * b) / np.sqrt(1 + 2 * max(0, correlation) * a * b)
        cycle = np.concatenate((samples[start + overlap:end], blend)).astype(np.float32)
        period = len(cycle) / RATE
        first_seam = (end - start - overlap) / RATE
        report.update(mode="interior_recurrence_crossfade", loop_start_seconds=start / RATE,
                      loop_end_seconds=end / RATE, crossfade_seconds=overlap / RATE,
                      period_seconds=period, recurrence_score=score, correlation=correlation,
                      seam_seconds=[round(first_seam + i * period, 6)
                                    for i in range(math.ceil(duration / period))
                                    if first_seam + i * period < duration])
    # Headroom is global, not a pump at each seam. Mixer measures this rendered
    # bed's LUFS, so source-independent targets continue to hold.
    peak = float(np.max(np.abs(cycle)))
    gain = min(1.0, 0.95 / max(peak, 1e-12))
    cycle *= gain
    report.update(peak_before_headroom=peak, headroom_gain=gain)
    fade_in, fade_out = min(RATE // 50, count), min(RATE, count // 4)
    with wave.open(str(output), 'wb') as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        for offset in range(0, count, len(cycle)):
            block = cycle[:min(len(cycle), count - offset)].copy()
            positions = np.arange(offset, offset + len(block))
            envelope = np.minimum(1.0, positions / max(1, fade_in))
            envelope *= np.minimum(1.0, (count - 1 - positions) / max(1, fade_out))
            block *= envelope[:, None]
            stream.writeframes((np.clip(block, -1, 1) * 32767).astype('<i2').tobytes())
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
