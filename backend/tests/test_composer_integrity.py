import hashlib
import json
import wave

from backend import config
from backend.pipeline import composer, tts


def test_narration_completeness_blocks_missing_script_audio(monkeypatch):
    monkeypatch.setattr(config, "AV_SYNC_MIN_WORD_COVERAGE_PERCENT", 65)
    failures = composer._narration_completeness_failures(
        {
            "method": "whisper_script_forced_alignment",
            "word_coverage": 0.20,
            "line_coverage": 0.35,
            "audio_coverage": 0.99,
        }
    )

    assert any("matched-word coverage" in failure for failure in failures)
    assert any("matched-line coverage" in failure for failure in failures)
    assert not any("transcript covers" in failure for failure in failures)


def test_narration_completeness_does_not_block_estimated_timing():
    assert composer._narration_completeness_failures(
        {
            "method": "word_count_estimate",
            "passed": False,
            "failure_reasons": ["word-level transcript unavailable"],
        }
    ) == []


def test_orpheus_manifest_requires_unchanged_script_and_audio(tmp_path):
    script = tmp_path / "script.txt"
    script.write_text("Speaker 1: Every word is verified.")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "tts_input_generated.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(b"\0\0" * 100)
    canonical = tts._strip_speaker_labels(script.read_text())
    (audio_dir / "tts_manifest.json").write_text(
        json.dumps(
            {
                "source_text_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                "output_audio_sha256": tts._file_sha256(audio),
                "integrity": {"passed": True, "verified_source_coverage": 1.0},
            }
        )
    )

    assert composer._orpheus_manifest_failures(script, audio, "orpheus-en") == []

    script.write_text("Speaker 1: The script was edited afterward.")
    failures = composer._orpheus_manifest_failures(script, audio, "orpheus-en")
    assert any("script changed" in failure for failure in failures)
