"""Build a bounded-memory music bed from recurring interior audio, never file seams."""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np

RATE = 48_000
HOP = 240


def _metered_period(power: np.ndarray) -> tuple[int, float] | None:
    """Estimate a pulse from agreement at beat, bar, and phrase scales.

    Restricted to 75--150 BPM newsroom beds in common time. A weak grid is
    explicitly rejected; spectral similarity alone cannot establish a beat.
    """
    flux = np.maximum(0, np.diff(np.log1p(np.sqrt(power) * 10), axis=0)).sum(axis=1)
    flux = flux[:round(60 * RATE / HOP)]
    flux -= flux.mean()
    if np.linalg.norm(flux) < 1e-7 or len(flux) * HOP / RATE < 8:
        return None

    def correlation(lag: int) -> float:
        a, b = flux[:-lag], flux[lag:]
        return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

    max_lag = len(flux) - round(3 * RATE / HOP)
    correlations = {lag: correlation(lag) for lag in range(1, max_lag)}
    best = (-1.0, 0.0)
    for beat in np.arange(0.4, 0.8001, 0.0005):
        lags = [round(beat * n * RATE / HOP) for n in (1, 2, 4, 8, 16, 32)
                if beat * n * RATE / HOP < max_lag]
        score = sum(correlations[lag] for lag in lags) / len(lags)
        if score > best[0]:
            best = (score, float(beat))
    score, beat = best
    if score < 0.35:
        return None
    # Preserve long sources in whole four-bar units; the 60-second analysis
    # window must not truncate a three-minute track to a short loop.
    available = (len(power) * HOP / RATE) - 3.0
    beats = int(available / beat) // 16 * 16
    beats = beats if beats >= 16 else None
    return (round(beats * beat * RATE), score) if beats else None


def _loop_points(samples: np.ndarray) -> tuple[int, int, int, float, bool]:
    # Compare entire overlapping passages, including their transient envelopes.
    # This is recurrence matching, not a claim to infer meter or musical quality.
    mono = samples.mean(axis=1)[::4]
    frames = np.lib.stride_tricks.sliding_window_view(mono, 512)[::HOP // 4]
    power = abs(np.fft.rfft(frames * np.hanning(512), axis=1)) ** 2
    edges = np.unique(np.geomspace(1, power.shape[1], 25).astype(int))
    bands = np.stack([power[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    level = np.sqrt((frames ** 2).mean(axis=1))
    active = np.flatnonzero(level > max(float(np.percentile(level, 80)) * 0.18, 1e-5))
    if len(active) < 20:
        raise ValueError("Music has too little active audio to loop")
    first, last = int(active[0]), int(active[-1])
    span = last - first
    overlap = min(round(2 * RATE / HOP), max(1, span // 8))
    # Log spectra capture recurring harmony/timbre; explicit level comparison
    # prevents matching a fade-out to a quiet introduction.
    features = np.log(np.maximum(bands, 1e-8))
    features -= features.mean(axis=1, keepdims=True)
    metered = _metered_period(power)
    if metered:
        period, grid_score = metered
        lag = round(period / HOP)
        match = min(round(RATE / HOP), max(1, span // 10))
        candidates = range(first, last - lag - match, 2)
        # A one-second musical context selects compatible harmony. Prefer a
        # low-energy edit between attacks; the overlap itself is only 10 ms.
        def boundary_score(a: int) -> float:
            spectral = np.mean((features[a:a + match] - features[a + lag:a + lag + match]) ** 2)
            energy = (level[a] + level[a + lag]) / max(float(np.median(level)), 1e-7)
            return float(spectral + 0.5 * energy)
        if candidates:
            start = min(candidates, key=boundary_score) * HOP
            # Fine phase alignment within 5 ms does not move a beat into a new
            # musical position. Linear blending avoids doubling correlated audio.
            overlap_samples = RATE // 100
            head = samples[start:start + overlap_samples]
            def error(delta: int) -> float:
                tail = samples[start + period + delta:start + period + delta + overlap_samples]
                return float(np.mean((tail - head) ** 2)) if len(tail) == len(head) else float('inf')
            delta = min(range(-RATE // 200, RATE // 200 + 1, 8), key=error)
            return start, start + period + delta, overlap_samples, grid_score, True
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
    return a * HOP, b * HOP, overlap * HOP, value, False


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
        start, end, overlap, score, rhythmic = _loop_points(samples)
        head, tail = samples[start:start + overlap], samples[end:end + overlap]
        t = np.linspace(0, 1, overlap, dtype=np.float64)[:, None]
        a, b = (1 - t, t) if rhythmic else (np.cos(t * np.pi / 2), np.sin(t * np.pi / 2))
        # Equal-power for unrelated audio; correlated material needs less gain.
        # Negative correlation is not boosted (avoid amplifying cancellation).
        correlation = float(np.sum(head * tail) / max(np.linalg.norm(head) * np.linalg.norm(tail), 1e-12))
        blend = tail * a + head * b
        if not rhythmic:
            blend /= np.sqrt(1 + 2 * max(0, correlation) * a * b)
        cycle = np.concatenate((samples[start + overlap:end], blend)).astype(np.float32)
        period = len(cycle) / RATE
        first_seam = (end - start - overlap) / RATE
        report.update(mode="metered_phrase_microfade" if rhythmic else "interior_recurrence_crossfade", loop_start_seconds=start / RATE,
                      loop_end_seconds=end / RATE, crossfade_seconds=overlap / RATE,
                      period_seconds=period, recurrence_score=None if rhythmic else score,
                      beat_grid_score=score if rhythmic else None, correlation=correlation,
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
