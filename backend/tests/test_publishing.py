import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.pipeline.opencli import OpenCLIError
from backend.publishing import (
    _browser,
    _browser_json,
    _description,
    _upload_local_media,
    delete_youtube,
)


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


def test_browser_json_accepts_boolean_eval_results():
    browser = AsyncMock(return_value="true")

    with patch("backend.publishing._browser", browser):
        result = asyncio.run(_browser_json("ftsn-yt-test", "eval", "(()=>true)()"))

    assert result is True


def test_browser_json_recovers_prefixed_object_results():
    browser = AsyncMock(return_value='OpenCLI result: {"ready": true}')

    with patch("backend.publishing._browser", browser):
        result = asyncio.run(_browser_json("ftsn-yt-test", "eval", "some-js"))

    assert result == {"ready": True}


def test_delete_youtube_supports_current_studio_confirmation_flow():
    task = SimpleNamespace(id="20260818-065410-6bf4df")
    published = {
        "status": "published",
        "identity": {"channel_id": "channel-123"},
        "external_id": "ZGFa-jK9HbQ",
    }
    browser = AsyncMock(return_value="")
    wait_for = AsyncMock(side_effect=[
        "Frontier Tech Daily [TEST]",
        True,
        True,
        True,
        True,
    ])
    browser_json = AsyncMock(side_effect=[True, True, True])

    def record(_task, platform, payload):
        return {"platforms": {platform: payload}}

    with (
        patch(
            "backend.publishing.read_publication_manifest",
            return_value={"platforms": {"youtube": published}},
        ),
        patch(
            "backend.publishing._youtube_identity",
            AsyncMock(return_value={"channel_id": "channel-123"}),
        ),
        patch("backend.publishing._browser", browser),
        patch("backend.publishing._wait_for", wait_for),
        patch("backend.publishing._browser_json", browser_json),
        patch("backend.publishing._record", side_effect=record),
    ):
        result = asyncio.run(delete_youtube(task))

    assert result["status"] == "deleted"
    calls = [call.args for call in browser.await_args_list]
    assert any(("--name", "Delete", "--role", "menuitem") == args[2:6] for args in calls)
    assert any(
        "ytcp-video-delete-dialog #confirm-input textarea" in args
        and "Frontier Tech Daily [TEST]" in args
        for args in calls
    )
    assert wait_for.await_args_list[-1].args[1] == (
        "(()=>!location.href.includes('/video/ZGFa-jK9HbQ/edit'))()"
    )


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
