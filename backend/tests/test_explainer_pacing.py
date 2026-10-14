import math
import struct
import wave

import pytest

from backend.explainer_pacing import SAMPLE_RATE, pacing_plan, write_paced_audio


def write_wave(path, samples):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        output.writeframes(b"".join(struct.pack("<h", value) for value in samples))


def test_only_edge_silence_is_removed_and_retained_samples_are_exact(tmp_path):
    source = tmp_path / "source.wav"
    tone = [int(2000 * math.sin(i * 2 * math.pi * 440 / SAMPLE_RATE)) for i in range(SAMPLE_RATE)]
    # An internal half-second pause and quiet final consonant are retained.
    samples = [0] * 24000 + tone + [0] * 24000 + [v // 30 for v in tone] + [0] * 144000
    write_wave(source, samples)
    original = source.read_bytes()
    plan = pacing_plan(source)
    assert plan["removed_leading_seconds"] == pytest.approx(0.42)
    assert plan["removed_trailing_seconds"] == pytest.approx(2.78)
    assert plan["end_sample"] > 144000
    output = tmp_path / "paced.wav"
    write_paced_audio(source, output, plan)
    with wave.open(str(output), "rb") as result:
        kept = result.readframes(plan["end_sample"] - plan["start_sample"])
        assert kept == b"".join(struct.pack("<h", v) for v in samples[plan["start_sample"]:plan["end_sample"]])
        assert result.getnframes() == plan["frames"] * 1600
    assert source.read_bytes() == original


def test_chapter_pause_and_closing_hold_are_explicit(tmp_path):
    source = tmp_path / "source.wav"
    write_wave(source, [500] * SAMPLE_RATE + [0] * (3 * SAMPLE_RATE))
    ordinary = pacing_plan(source)
    chapter = pacing_plan(source, chapter_end=True)
    ending = pacing_plan(source, closing_hold=2.4)
    assert ordinary["narration_duration_seconds"] == 1.22
    assert chapter["narration_duration_seconds"] == 1.45
    assert ending["duration_seconds"] == pytest.approx(math.ceil(3.85 * 30) / 30)
    assert ending["speech_end_seconds"] == 1
    assert ending["closing_hold_seconds"] == 2.4


def test_short_edges_are_not_lengthened_and_silence_is_rejected(tmp_path):
    source = tmp_path / "source.wav"
    write_wave(source, [0] * 2400 + [500] * SAMPLE_RATE + [0] * 2400)
    plan = pacing_plan(source)
    assert plan["removed_leading_seconds"] == plan["removed_trailing_seconds"] == 0
    write_wave(source, [0] * SAMPLE_RATE)
    with pytest.raises(ValueError, match="silent"):
        pacing_plan(source)
