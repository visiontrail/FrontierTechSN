import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.pipeline.opencli import OpenCLIError
from backend.publishing import _browser, _description, _upload_local_media


def test_description_includes_selected_research_sources(tmp_path):
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    (research_dir / "dossier.json").write_text(
        json.dumps({
            "selected": [
                {"url": "https://example.com/selected-one"},
                {"url": "https://example.com/selected-two"},
            ],
            "stories": [{"url": "https://example.com/legacy"}],
        }),
        encoding="utf-8",
    )

    description = _description(SimpleNamespace(output_dir=str(tmp_path)), "FTSN-TEST")

    assert "Test run: FTSN-TEST" in description
    assert "https://example.com/selected-one" in description
    assert "https://example.com/selected-two" in description
    assert "https://example.com/legacy" not in description


def test_description_supports_legacy_stories_key(tmp_path):
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    (research_dir / "dossier.json").write_text(
        json.dumps({"stories": [{"url": "https://example.com/legacy"}]}),
        encoding="utf-8",
    )

    description = _description(SimpleNamespace(output_dir=str(tmp_path)))

    assert "https://example.com/legacy" in description


def test_youtube_browser_sessions_are_foregrounded():
    runner = AsyncMock(return_value=SimpleNamespace(stdout="ok"))

    with patch("backend.publishing.run_opencli_with_retries", runner):
        result = asyncio.run(
            _browser(
                "ftsn-yt-delete-20260818",
                "open",
                "https://studio.youtube.com/",
            )
        )

    assert result == "ok"
    assert runner.await_args.args[0][-2:] == ["--window", "foreground"]


def test_non_youtube_browser_sessions_remain_backgrounded():
    runner = AsyncMock(return_value=SimpleNamespace(stdout="ok"))

    with patch("backend.publishing.run_opencli_with_retries", runner):
        result = asyncio.run(
            _browser("ftsn-x-20260818", "open", "https://x.com/compose/post")
        )

    assert result == "ok"
    assert runner.await_args.args[0][-2:] == ["--window", "background"]


def test_upload_permission_failure_is_actionable_without_retry():
    browser = AsyncMock(side_effect=OpenCLIError(
        'OpenCLI browser failed: {"code":-32000,"message":"Not allowed"}'
    ))

    with patch("backend.publishing._browser", browser):
        with pytest.raises(OpenCLIError, match="Allow access to file URLs"):
            asyncio.run(
                _upload_local_media(
                    "test-session",
                    "input[name=Filedata]",
                    "/tmp/video.mp4",
                    timeout=300,
                )
            )

    assert browser.await_count == 1
    assert browser.await_args.kwargs["attempts"] == 1


def test_upload_transient_failure_uses_remaining_retry_budget():
    browser = AsyncMock(side_effect=[
        OpenCLIError("temporary browser lease failure"),
        "uploaded",
    ])

    with patch("backend.publishing._browser", browser):
        result = asyncio.run(
            _upload_local_media(
                "test-session",
                "input[name=Filedata]",
                "/tmp/video.mp4",
                timeout=300,
            )
        )

    assert result == "uploaded"
    assert browser.await_count == 2
    assert browser.await_args_list[0].kwargs["attempts"] == 1
    assert browser.await_args_list[1].kwargs["attempts"] == 9
