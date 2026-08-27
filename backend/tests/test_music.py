from __future__ import annotations

from pathlib import Path

import pytest

from backend import config
from backend.pipeline import music
from backend.pipeline.opencli import OpenCLIResult


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
