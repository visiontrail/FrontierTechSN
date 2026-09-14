from __future__ import annotations

import json
import hashlib
import math
import struct
import wave
from pathlib import Path

import pytest

from backend import config
from backend.pipeline import music
from backend.pipeline.opencli import OpenCLIResult


def _tone(path: Path, seconds: float, frequency: float, *, amplitude: int) -> None:
    rate = 48_000
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        samples = bytearray()
        for index in range(int(rate * seconds)):
            value = int(amplitude * math.sin(2 * math.pi * frequency * index / rate))
            samples.extend(struct.pack("<h", value))
        stream.writeframes(samples)


@pytest.mark.asyncio
@pytest.mark.parametrize("more_tools_result", ["opened-more-tools", "no-more-tools"])
async def test_gemini_music_tool_supports_current_and_legacy_menus(
    monkeypatch: pytest.MonkeyPatch,
    more_tools_result: str,
) -> None:
    calls: list[list[str]] = []
    eval_outputs = iter(["clicked-upload-tools", more_tools_result, "selected-music-tool"])

    async def browser(session, args, *, timeout=90, check=True):
        calls.append(args)
        output = next(eval_outputs) if args[0] == "eval" else "waited"
        return OpenCLIResult(tuple(args), 0, output, "")

    monkeypatch.setattr(music, "_browser", browser)

    await music._enable_gemini_music_tool("music-session")

    eval_calls = [args for args in calls if args[0] == "eval"]
    assert len(eval_calls) == 3
    wait_calls = [args for args in calls if args[0] == "wait"]
    assert len(wait_calls) == (2 if more_tools_result == "opened-more-tools" else 1)


@pytest.mark.asyncio
async def test_gemini_prompt_fill_escapes_javascript_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts: list[str] = []

    async def browser(session, args, *, timeout=90, check=True):
        scripts.append(args[1])
        return OpenCLIResult(tuple(args), 0, "filled-prompt", "")

    monkeypatch.setattr(music, "_browser", browser)

    await music._fill_gemini_prompt("music-session", "Leo's 音乐\nline two")

    assert "Leo's 音乐" in scripts[0]
    assert "\\nline two" in scripts[0]


@pytest.mark.asyncio
async def test_gemini_music_retries_use_independent_browser_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions: list[str] = []
    closed: list[str] = []

    async def fail_once(prompt, music_dir, session, *, timeout):
        sessions.append(session)
        raise TimeoutError("stale browser lease")

    async def close_session(session, args, *, timeout=90, check=True):
        if args == ["close"]:
            closed.append(session)

    async def local_fallback(music_dir, duration=60.0):
        path = music_dir / "fallback.wav"
        path.write_bytes(b"\0" * 5000)
        return path

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(config, "OPENCLI_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(music, "_gemini_create_music_once", fail_once)
    monkeypatch.setattr(music, "_browser", close_session)
    monkeypatch.setattr(music, "_local_music", local_fallback)
    monkeypatch.setattr(music.asyncio, "sleep", no_wait)

    artifact = await music.generate_background_music(
        "daily-task",
        tmp_path,
        title="Morning briefing",
        summary={},
        provider="gemini_create_music",
    )

    assert sessions == [
        "ftsn-music-daily-task-1",
        "ftsn-music-daily-task-2",
        "ftsn-music-daily-task-3",
    ]
    assert closed == sessions
    assert artifact.provider == "local_deterministic"
    assert Path(artifact.audio_path).stat().st_size == 5000


@pytest.mark.asyncio
async def test_music_mix_targets_source_loudness_and_preserves_narration(
    tmp_path: Path,
) -> None:
    narration = tmp_path / "narration.wav"
    quiet_music = tmp_path / "quiet-music.wav"
    _tone(narration, 3.0, 220, amplitude=5000)
    _tone(quiet_music, 3.0, 440, amplitude=700)

    output = await music.mix_narration_and_music(
        narration,
        quiet_music,
        tmp_path,
        bed_db=-25,
        duck_db=-11,
    )

    report = json.loads((tmp_path / "audio" / "music_mix_report.json").read_text())
    assert output.is_file()
    assert report["passed"] is True
    assert report["content_music_gain_db"] > 0
    assert report["program_mix_vs_narration_lu"] >= -1.5
    assert report["ducked_music_integrated_lufs"] >= report["minimum_ducked_music_lufs"]
    assert report["failure_reasons"] == []


@pytest.mark.asyncio
async def test_local_library_track_bypasses_gemini_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = tmp_path / "library"
    library.mkdir()
    source = library / "desk-theme.wav"
    _tone(source, 2.0, 330, amplitude=1200)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    catalog = library / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "default_track_id": "desk-theme",
                "tracks": [
                    {
                        "id": "desk-theme",
                        "title": "Desk Theme",
                        "filename": source.name,
                        "source": "user",
                        "duration_seconds": 2.0,
                        "sha256": digest,
                    }
                ],
            }
        )
    )

    async def forbidden(*args, **kwargs):
        raise AssertionError("Gemini must not run for a local library track")

    monkeypatch.setattr(music, "PROGRAM_MUSIC_DIR", library)
    monkeypatch.setattr(music, "PROGRAM_MUSIC_CATALOG", catalog)
    monkeypatch.setattr(music, "_gemini_create_music_once", forbidden)

    artifact = await music.generate_background_music(
        "local-library-task",
        tmp_path / "task",
        title="Desk edition",
        summary={},
        provider="local_library",
        track_id="desk-theme",
    )

    manifest = json.loads(Path(artifact.manifest_path).read_text())
    assert artifact.provider == "local_library"
    assert manifest["provider_used"] == "local_library"
    assert manifest["selected_track_id"] == "desk-theme"
    assert Path(artifact.audio_path).is_file()


@pytest.mark.asyncio
async def test_program_pacing_inserts_exact_gaps_and_restores_music(
    tmp_path: Path,
) -> None:
    narration = tmp_path / "narration.wav"
    bed = tmp_path / "bed.wav"
    _tone(narration, 6.0, 220, amplitude=5000)
    _tone(bed, 2.0, 440, amplitude=1800)
    aligned = [
        {"text": "Opening", "start": 0.0, "duration": 1.5},
        {"text": "Story one", "start": 1.5, "duration": 1.5},
        {"text": "Story two", "start": 3.0, "duration": 1.5},
        {"text": "Closing", "start": 4.5, "duration": 1.5},
    ]

    paced, pacing = await music.create_paced_narration(
        narration,
        tmp_path,
        aligned,
        intro_seconds=2.0,
        opening_gap_seconds=3.0,
        story_gap_seconds=1.5,
    )

    assert pacing["passed"] is True
    assert pacing["paced_duration_seconds"] == pytest.approx(14.0, abs=0.02)
    assert [gap["duration_seconds"] for gap in pacing["gaps"]] == [3.0, 1.5, 1.5]
    assert pacing["segments"][0]["program_start"] == pytest.approx(2.0)
    assert pacing["segments"][1]["program_start"] == pytest.approx(6.5)

    output = await music.mix_narration_and_music(
        paced,
        bed,
        tmp_path,
        bed_db=-25,
        duck_db=-11,
    )
    report = json.loads((tmp_path / "audio" / "music_mix_report.json").read_text())
    assert output.is_file()
    assert report["passed"] is True
    assert report["ducking_mode"] == "program_timeline_envelope"
    assert report["measured_gap_lift_db"] >= 8.0
    assert len(report["window_measurements"]["restored_music_windows"]) == 4


@pytest.mark.asyncio
async def test_credit_tail_preserves_pacing_envelope_and_compares_speech_interval(tmp_path):
    import subprocess
    narration = tmp_path / 'narration.wav'
    bed = tmp_path / 'bed.wav'
    _tone(narration, 6.0, 220, amplitude=5000)
    _tone(bed, 2.0, 440, amplitude=1800)
    paced, pacing = await music.create_paced_narration(
        narration, tmp_path,
        [{'text': 'Opening', 'start': 0, 'duration': 3},
         {'text': 'Closing', 'start': 3, 'duration': 3}],
        intro_seconds=2, opening_gap_seconds=3, story_gap_seconds=1.5,
    )
    padded = tmp_path / 'padded.wav'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(paced),
                    '-af', 'apad=whole_dur=45', str(padded)], check=True)
    await music.mix_narration_and_music(padded, bed, tmp_path, bed_db=-25, duck_db=-11,
                                       original_narration_path=paced)
    report = json.loads((tmp_path / 'audio/music_mix_report.json').read_text())
    assert report['passed'] is True
    assert report['ducking_mode'] == 'program_timeline_envelope'
    assert report['program_mix_duration_seconds'] == pytest.approx(45, abs=.02)
    assert report['narration_comparison_end_seconds'] == pytest.approx(pacing['paced_duration_seconds'])
    tail = report['window_measurements']['restored_music_windows'][-1]
    assert tail['kind'] == 'credits'
    assert tail['end'] > 44
    assert report['program_mix_vs_narration_lu'] >= -1.5
