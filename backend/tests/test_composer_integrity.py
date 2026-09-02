import hashlib
import json
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend import config
from backend.pipeline import composer, tts
from backend.pipeline.video_format import LANDSCAPE


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


def test_verified_orpheus_source_contract_outweighs_paced_whisper_false_negative():
    alignment = {
        "method": "whisper_script_forced_alignment",
        "word_coverage": 0.67,
        "line_coverage": 0.70,
        "audio_coverage": 0.79,
    }

    assert composer._narration_completeness_failures(alignment)
    assert composer._narration_completeness_failures(
        alignment,
        source_contract_verified=True,
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
                "model": "orpheus-en",
                "pacing_policy": tts.NARRATION_PACING_POLICY,
                "synthesis_speed_ratio": tts.NARRATION_SYNTHESIS_SPEED_RATIO,
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


def test_pocket_manifest_has_same_verified_source_contract_without_vibe_rewrites(tmp_path):
    script = tmp_path / "script.txt"
    script.write_text("Speaker 1: Qwen remains canonical in Pocket TTS.")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "tts_input_generated.wav"
    audio.write_bytes(b"pocket narration")
    canonical = tts._strip_speaker_labels(script.read_text())
    manifest = {
        "model": "pocket-tts-en",
        "pacing_policy": tts.NARRATION_PACING_POLICY,
        "synthesis_speed_ratio": tts.NARRATION_SYNTHESIS_SPEED_RATIO,
        "source_text_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "output_audio_sha256": tts._file_sha256(audio),
        "integrity": {"passed": True, "verified_source_coverage": 1.0},
    }
    (audio_dir / "tts_manifest.json").write_text(json.dumps(manifest))

    assert composer._narration_manifest_failures(
        script, audio, "pocket-tts-en"
    ) == []

    manifest["integrity"]["passed"] = False
    (audio_dir / "tts_manifest.json").write_text(json.dumps(manifest))
    failures = composer._narration_manifest_failures(script, audio, "pocket-tts-en")
    assert any("Kyutai Pocket TTS" in failure for failure in failures)


def test_vibevoice_manifest_also_binds_script_and_audio(tmp_path):
    script = tmp_path / "script.txt"
    script.write_text("A stable narration contract.", encoding="utf-8")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "tts_input_generated.wav"
    audio.write_bytes(b"generated narration")
    (audio_dir / "tts_manifest.json").write_text(
        json.dumps(
            {
                "model": "vibevoice-0.5b",
                "pacing_policy": tts.NARRATION_PACING_POLICY,
                "synthesis_speed_ratio": tts.NARRATION_SYNTHESIS_SPEED_RATIO,
                "source_text_sha256": hashlib.sha256(
                    script.read_text(encoding="utf-8").encode()
                ).hexdigest(),
                "output_audio_sha256": tts._file_sha256(audio),
            }
        ),
        encoding="utf-8",
    )

    assert composer._narration_manifest_failures(
        script, audio, "vibevoice-0.5b"
    ) == []
    script.write_text("Edited after audio generation.", encoding="utf-8")
    assert any(
        "script changed" in failure
        for failure in composer._narration_manifest_failures(
            script, audio, "vibevoice-0.5b"
        )
    )


def test_narration_manifest_rejects_retimed_or_unproven_speed(tmp_path):
    script = tmp_path / "script.txt"
    script.write_text("Natural narration owns the timeline.", encoding="utf-8")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "tts_input_generated.wav"
    audio.write_bytes(b"generated narration")
    base = {
        "model": "vibevoice-0.5b",
        "pacing_policy": tts.NARRATION_PACING_POLICY,
        "synthesis_speed_ratio": 1.0,
        "source_text_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "output_audio_sha256": tts._file_sha256(audio),
    }

    for speed_ratio in (None, 1.4):
        manifest = dict(base)
        if speed_ratio is None:
            manifest.pop("synthesis_speed_ratio")
        else:
            manifest["synthesis_speed_ratio"] = speed_ratio
        (audio_dir / "tts_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        failures = composer._narration_manifest_failures(
            script, audio, "vibevoice-0.5b"
        )

        assert any("natural 1.0x" in failure for failure in failures)


def test_narration_manifest_rejects_non_object_root(tmp_path):
    script = tmp_path / "script.txt"
    script.write_text("Narration", encoding="utf-8")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "audio.wav"
    audio.write_bytes(b"audio")
    (audio_dir / "tts_manifest.json").write_text("[]", encoding="utf-8")

    assert composer._narration_manifest_failures(
        script, audio, "orpheus-en"
    ) == ["narration integrity manifest must be a JSON object"]


def test_rendered_video_probe_requires_expected_streams_dimensions_and_duration():
    good = SimpleNamespace(
        returncode=0,
        stderr="",
        stdout=json.dumps(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1920,
                        "height": 1080,
                    },
                    {"codec_type": "audio"},
                ],
                "format": {"duration": "12.25"},
            }
        ),
    )
    decoded = SimpleNamespace(returncode=0, stderr="", stdout="")
    staged = Path("video.next.mp4")
    with patch.object(
        composer,
        "run_capture_logged",
        side_effect=[good, decoded],
    ) as runner:
        assert composer._rendered_video_failures(
            staged,
            frame=LANDSCAPE,
            expected_duration=12.0,
        ) == []
    assert runner.call_count == 2
    assert runner.call_args_list[1].kwargs["command"][:5] == [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
    ]
    assert "explode" in runner.call_args_list[1].kwargs["command"]

    bad = SimpleNamespace(
        returncode=0,
        stderr="",
        stdout=json.dumps(
            {
                "streams": [
                    {"codec_type": "video", "width": 1280, "height": 720}
                ],
                "format": {"duration": "3.0"},
            }
        ),
    )
    with patch.object(composer, "run_capture_logged", return_value=bad):
        failures = composer._rendered_video_failures(
            staged,
            frame=LANDSCAPE,
            expected_duration=12.0,
        )
    assert any("dimensions" in failure for failure in failures)
    assert any("audio stream" in failure for failure in failures)
    assert any("duration" in failure for failure in failures)


def test_rendered_video_probe_rejects_a_stream_that_cannot_fully_decode():
    probe = SimpleNamespace(
        returncode=0,
        stderr="",
        stdout=json.dumps(
            {
                "streams": [
                    {"codec_type": "video", "width": 1920, "height": 1080},
                    {"codec_type": "audio"},
                ],
                "format": {"duration": "12.0"},
            }
        ),
    )
    decode = SimpleNamespace(
        returncode=1,
        stderr="Invalid data found when processing input",
        stdout="",
    )

    with patch.object(
        composer,
        "run_capture_logged",
        side_effect=[probe, decode],
    ):
        failures = composer._rendered_video_failures(
            Path("video.next.mp4"),
            frame=LANDSCAPE,
            expected_duration=12.0,
        )

    assert failures == [
        "full render decode failed with ffmpeg exit 1: "
        "Invalid data found when processing input"
    ]


def test_render_promotion_keeps_the_previous_completed_cut(tmp_path):
    previous = tmp_path / "video.mp4"
    staged = tmp_path / "video.next.mp4"
    staged_report = tmp_path / "av_sync_report.next.json"
    previous.write_bytes(b"previous completed cut")
    staged.write_bytes(b"new verified cut")
    staged_report.write_text('{"passed": true}', encoding="utf-8")
    digest = composer._sha256_path(staged)

    promoted = composer._promote_render_candidate(
        staged,
        staged_report,
        tmp_path,
        digest,
    )

    assert previous.read_bytes() == b"previous completed cut"
    assert promoted.name == f"video-{digest[:16]}.mp4"
    assert promoted.read_bytes() == b"new verified cut"
    assert (tmp_path / "av_sync_report.json").read_text(encoding="utf-8") == (
        '{"passed": true}'
    )


def test_report_promotion_failure_cannot_overwrite_the_previous_cut(tmp_path):
    previous = tmp_path / "video.mp4"
    staged = tmp_path / "video.next.mp4"
    staged_report = tmp_path / "av_sync_report.next.json"
    previous.write_bytes(b"previous completed cut")
    staged.write_bytes(b"new verified cut")
    staged_report.write_text('{"passed": true}', encoding="utf-8")
    digest = composer._sha256_path(staged)
    real_replace = composer.os.replace

    def replace_then_fail(source, destination):
        if Path(source) == staged_report:
            raise OSError("simulated report promotion failure")
        return real_replace(source, destination)

    with patch.object(composer.os, "replace", side_effect=replace_then_fail):
        with pytest.raises(OSError, match="report promotion failure"):
            composer._promote_render_candidate(
                staged,
                staged_report,
                tmp_path,
                digest,
            )

    assert previous.read_bytes() == b"previous completed cut"
    assert (tmp_path / f"video-{digest[:16]}.mp4").read_bytes() == (
        b"new verified cut"
    )
