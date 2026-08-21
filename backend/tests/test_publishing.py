import asyncio
import hashlib
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.pipeline.opencli import OpenCLIError
from backend import config
from backend.models import TaskStatus
from backend.publishing import (
    PublicationPlan,
    PublicationLedgerError,
    _assert_expected_x_identity,
    _assert_expected_youtube_identity,
    _browser,
    _browser_json,
    _description,
    _publication_marker,
    _upload_local_media,
    automatic_publication_configuration_errors,
    delete_youtube,
    execute_publication_plan,
    plan_publication,
    prepare_apple_podcast,
    publication_manifest_view,
    publish_task,
    read_publication_manifest,
    reconcile_deleted_test_receipts,
    run_auto_publish_pipeline,
)


def _publication_task(tmp_path, task_id="task-publication"):
    task_dir = tmp_path / task_id
    task_dir.mkdir(parents=True)
    video = task_dir / "video.mp4"
    audio = task_dir / "audio.wav"
    video.write_bytes(b"current immutable video")
    audio.write_bytes(b"current immutable audio")
    return SimpleNamespace(
        id=task_id,
        status=TaskStatus.COMPLETE,
        output_dir=str(task_dir),
        video_path=str(video),
        audio_path=str(audio),
        generated_title="Episode title",
        source_title=None,
        origin_id=None,
        duration_seconds=42,
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_publication_markers_are_unique_for_tasks_created_on_the_same_day():
    first = _publication_marker("20260821-010101-aaaaaaaa")
    second = _publication_marker("20260821-020202-bbbbbbbb")

    assert first.startswith("FTSN-")
    assert second.startswith("FTSN-")
    assert first != second


def test_publication_execution_serializes_same_platform_across_tasks(tmp_path):
    first = _publication_task(tmp_path, "task-first")
    second = _publication_task(tmp_path, "task-second")
    active = 0
    maximum_active = 0

    async def fake_publish_x(task, **_kwargs):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"task_id": task.id}

    async def scenario():
        plan = PublicationPlan(
            targets=("x",),
            pending_targets=("x",),
            visibility="private",
            manifest={"platforms": {}},
        )
        with (
            patch("backend.publishing._assert_publication_dispatch_allowed"),
            patch("backend.publishing.publish_x", side_effect=fake_publish_x),
            patch(
                "backend.publishing.read_publication_manifest",
                return_value={"platforms": {}},
            ),
        ):
            await asyncio.gather(
                execute_publication_plan(first, plan=plan, test_mode=True),
                execute_publication_plan(second, plan=plan, test_mode=True),
            )

    asyncio.run(scenario())

    assert maximum_active == 1


def test_missing_publication_manifest_is_an_empty_ledger(tmp_path):
    task = _publication_task(tmp_path)

    assert read_publication_manifest(task) == {
        "task_id": task.id,
        "platforms": {},
    }


@pytest.mark.parametrize(
    "contents",
    [
        '{"platforms":',
        "[]",
        '{"task_id": "task-publication", "platforms": []}',
    ],
    ids=["malformed-json", "non-object-root", "non-object-platforms"],
)
def test_existing_invalid_publication_manifest_fails_closed(tmp_path, contents):
    task = _publication_task(tmp_path)
    publication_dir = tmp_path / task.id / "publications"
    publication_dir.mkdir()
    (publication_dir / "manifest.json").write_text(contents, encoding="utf-8")

    with pytest.raises(PublicationLedgerError):
        read_publication_manifest(task)


@pytest.mark.parametrize(
    "contents",
    [
        {},
        {"task_id": "task-publication"},
        {"platforms": {}},
    ],
    ids=["missing-task-and-platforms", "missing-platforms", "missing-task-id"],
)
def test_existing_incomplete_publication_manifest_fails_closed(tmp_path, contents):
    task = _publication_task(tmp_path)
    publication_dir = tmp_path / task.id / "publications"
    publication_dir.mkdir()
    (publication_dir / "manifest.json").write_text(
        json.dumps(contents),
        encoding="utf-8",
    )

    with pytest.raises(PublicationLedgerError):
        read_publication_manifest(task)


@pytest.mark.parametrize("canonical_exists", [False, True])
def test_unfinished_publication_manifest_write_fails_closed(
    tmp_path,
    canonical_exists,
):
    task = _publication_task(tmp_path)
    publication_dir = tmp_path / task.id / "publications"
    publication_dir.mkdir()
    if canonical_exists:
        (publication_dir / "manifest.json").write_text(
            json.dumps({"task_id": task.id, "platforms": {}}),
            encoding="utf-8",
        )
    (publication_dir / "manifest.tmp").write_text(
        json.dumps(
            {
                "task_id": task.id,
                "platforms": {"x": {"status": "published"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PublicationLedgerError, match="unfinished atomic write"):
        read_publication_manifest(task)


def test_publication_manifest_rejects_a_different_task_id(tmp_path):
    task = _publication_task(tmp_path)
    publication_dir = tmp_path / task.id / "publications"
    publication_dir.mkdir()
    (publication_dir / "manifest.json").write_text(
        json.dumps({"task_id": "another-task", "platforms": {}}),
        encoding="utf-8",
    )

    with pytest.raises(PublicationLedgerError, match="different task"):
        read_publication_manifest(task)


@pytest.mark.parametrize(
    ("platform", "status"),
    [
        ("youtube", "publshed"),
        ("youtube", None),
        ("x", "publshed"),
        ("x", None),
    ],
    ids=["youtube-typo", "youtube-missing", "x-typo", "x-missing"],
)
def test_publication_manifest_rejects_invalid_or_missing_receipt_status(
    tmp_path,
    platform,
    status,
):
    task = _publication_task(tmp_path)
    publication_dir = tmp_path / task.id / "publications"
    publication_dir.mkdir()
    receipt = {} if status is None else {"status": status}
    (publication_dir / "manifest.json").write_text(
        json.dumps({"task_id": task.id, "platforms": {platform: receipt}}),
        encoding="utf-8",
    )

    with pytest.raises(PublicationLedgerError, match=f"invalid {platform} status"):
        read_publication_manifest(task)


def test_publication_manifest_rejects_unknown_platform(tmp_path):
    task = _publication_task(tmp_path)
    publication_dir = tmp_path / task.id / "publications"
    publication_dir.mkdir()
    (publication_dir / "manifest.json").write_text(
        json.dumps(
            {
                "task_id": task.id,
                "platforms": {"tiktok": {"status": "published"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PublicationLedgerError, match="unsupported platform"):
        read_publication_manifest(task)


def test_deleted_social_receipts_are_valid_and_become_pending(tmp_path):
    task = _publication_task(tmp_path)
    publication_dir = tmp_path / task.id / "publications"
    publication_dir.mkdir()
    manifest = {
        "task_id": task.id,
        "platforms": {
            "youtube": {
                "status": "deleted",
                "published_at": "2026-08-20T00:00:00+00:00",
                "deleted_at": "2026-08-20T01:00:00+00:00",
                "identity": {"channel_id": "UC123", "channel_name": "Channel"},
                "url": "https://youtu.be/abc1234",
                "external_id": "abc1234",
                "visibility": "private",
                "marker": "FTSN-test",
                "test_mode": True,
                "sha256": _sha256(Path(task.video_path)),
            },
            "x": {
                "status": "deleted",
                "published_at": "2026-08-20T00:00:00+00:00",
                "deleted_at": "2026-08-20T01:00:00+00:00",
                "identity": {"username": "frontier"},
                "url": "https://x.com/frontier/status/123456",
                "external_id": "123456",
                "marker": "FTSN-test",
                "test_mode": True,
                "sha256": _sha256(Path(task.video_path)),
            },
        },
    }
    (publication_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    assert read_publication_manifest(task) == manifest
    with patch(
        "backend.publishing.automatic_publication_configuration_errors",
        return_value=[],
    ):
        plan = plan_publication(
            task,
            targets=["youtube", "x"],
            test_mode=True,
            visibility="private",
        )

    assert plan.pending_targets == ("youtube", "x")


def test_operator_can_reconcile_an_indeterminate_test_delete(tmp_path):
    task = _publication_task(tmp_path)
    publication_dir = Path(task.output_dir) / "publications"
    publication_dir.mkdir()
    manifest = {
        "task_id": task.id,
        "platforms": {
            "x": {
                "status": "published",
                "published_at": "2026-08-20T00:00:00+00:00",
                "identity": {"username": "frontier"},
                "url": "https://x.com/frontier/status/123456",
                "external_id": "123456",
                "marker": "FTSN-test",
                "test_mode": True,
                "sha256": _sha256(Path(task.video_path)),
            }
        },
    }
    (publication_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    reconciled = reconcile_deleted_test_receipts(task, ["x"])

    entry = reconciled["platforms"]["x"]
    assert entry["status"] == "deleted"
    assert entry["deleted_at"] == entry["deletion_reconciled_at"]
    assert read_publication_manifest(task)["platforms"]["x"]["status"] == "deleted"


def test_publication_manifest_view_marks_current_and_previous_media_revisions(tmp_path):
    task = _publication_task(tmp_path)
    video_sha = _sha256(tmp_path / task.id / "video.mp4")
    audio_sha = _sha256(tmp_path / task.id / "audio.wav")
    previous_video_sha = hashlib.sha256(b"previous video").hexdigest()
    manifest = {
        "task_id": task.id,
        "platforms": {
            "youtube": {"status": "published", "sha256": previous_video_sha},
            "x": {"status": "published", "sha256": video_sha},
            "apple_podcast": {"status": "feed_ready", "sha256": audio_sha},
        },
    }

    view = publication_manifest_view(task, manifest)

    assert view["current_video_sha256"] == video_sha
    assert view["current_audio_sha256"] == audio_sha
    assert view["platforms"]["youtube"]["sha256"] == previous_video_sha
    assert view["platforms"]["youtube"]["matches_current_media"] is False
    assert view["platforms"]["x"]["matches_current_media"] is True
    assert view["platforms"]["apple_podcast"]["matches_current_media"] is True
    assert "matches_current_media" not in manifest["platforms"]["youtube"]


def test_publish_task_skips_every_active_same_revision_receipt(tmp_path):
    task = _publication_task(tmp_path)
    video_sha = _sha256(tmp_path / task.id / "video.mp4")
    audio_sha = _sha256(tmp_path / task.id / "audio.wav")
    manifest = {
        "task_id": task.id,
        "platforms": {
            "youtube": {
                "status": "published",
                "sha256": video_sha,
                "test_mode": True,
                "visibility": "private",
                "identity": {
                    "channel_id": config.VIDEO_PUBLISH_YOUTUBE_CHANNEL_ID,
                    "channel_name": config.VIDEO_PUBLISH_YOUTUBE_CHANNEL_NAME,
                },
            },
            "x": {
                "status": "published",
                "sha256": video_sha,
                "test_mode": True,
                "identity": {"username": config.VIDEO_PUBLISH_X_HANDLE},
            },
            "apple_podcast": {
                "status": "feed_ready",
                "sha256": audio_sha,
                "identity": {
                    "feed_title": config.VIDEO_PUBLISH_APPLE_FEED_TITLE,
                    "author": config.VIDEO_PUBLISH_APPLE_FEED_AUTHOR,
                },
            },
        },
    }

    with (
        patch(
            "backend.publishing.automatic_publication_configuration_errors",
            return_value=[],
        ),
        patch("backend.publishing.read_publication_manifest", return_value=manifest),
        patch("backend.publishing.publish_youtube", AsyncMock()) as youtube,
        patch("backend.publishing.publish_x", AsyncMock()) as x,
        patch("backend.publishing.prepare_apple_podcast", Mock()) as apple,
    ):
        result = asyncio.run(
            publish_task(
                task,
                targets=["youtube", "x", "apple_podcast"],
                test_mode=True,
                visibility="private",
            )
        )

    assert result == manifest
    youtube.assert_not_awaited()
    x.assert_not_awaited()
    apple.assert_not_called()


def test_publish_task_with_current_youtube_receipt_only_dispatches_x(tmp_path):
    task = _publication_task(tmp_path)
    video_sha = _sha256(tmp_path / task.id / "video.mp4")
    manifest = {
        "task_id": task.id,
        "platforms": {
            "youtube": {
                "status": "published",
                "sha256": video_sha,
                "test_mode": True,
                "visibility": "private",
                "identity": {
                    "channel_id": config.VIDEO_PUBLISH_YOUTUBE_CHANNEL_ID,
                    "channel_name": config.VIDEO_PUBLISH_YOUTUBE_CHANNEL_NAME,
                },
            },
        },
    }

    with (
        patch(
            "backend.publishing.automatic_publication_configuration_errors",
            return_value=[],
        ),
        patch("backend.publishing.read_publication_manifest", return_value=manifest),
        patch("backend.publishing.publish_youtube", AsyncMock()) as youtube,
        patch("backend.publishing.publish_x", AsyncMock()) as x,
        patch("backend.publishing.prepare_apple_podcast", Mock()) as apple,
    ):
        asyncio.run(
            publish_task(
                task,
                targets=["youtube", "x"],
                test_mode=True,
                visibility="private",
            )
        )

    youtube.assert_not_awaited()
    x.assert_awaited_once_with(
        task,
        test_mode=True,
        require_auto_publish_enabled=False,
    )
    apple.assert_not_called()


@pytest.mark.parametrize(
    "conflict",
    ["previous-sha", "test-mode", "visibility"],
)
def test_publish_task_rejects_active_receipt_conflicts_before_dispatch(
    tmp_path,
    conflict,
):
    task = _publication_task(tmp_path)
    video_sha = _sha256(tmp_path / task.id / "video.mp4")
    youtube_receipt = {
        "status": "published",
        "sha256": video_sha,
        "test_mode": True,
        "visibility": "private",
        "identity": {
            "channel_id": config.VIDEO_PUBLISH_YOUTUBE_CHANNEL_ID,
            "channel_name": config.VIDEO_PUBLISH_YOUTUBE_CHANNEL_NAME,
        },
    }
    if conflict == "previous-sha":
        youtube_receipt["sha256"] = hashlib.sha256(b"previous video").hexdigest()
    elif conflict == "test-mode":
        youtube_receipt["test_mode"] = False
    else:
        youtube_receipt["visibility"] = "public"
    manifest = {
        "task_id": task.id,
        # X is deliberately pending and ordered first in the request. The
        # later YouTube conflict must still abort the whole batch preflight.
        "platforms": {"youtube": youtube_receipt},
    }

    with (
        patch(
            "backend.publishing.automatic_publication_configuration_errors",
            return_value=[],
        ),
        patch("backend.publishing.read_publication_manifest", return_value=manifest),
        patch("backend.publishing.publish_youtube", AsyncMock()) as youtube,
        patch("backend.publishing.publish_x", AsyncMock()) as x,
        patch("backend.publishing.prepare_apple_podcast", Mock()) as apple,
        pytest.raises(ValueError, match="already has an active publication"),
    ):
        asyncio.run(
            publish_task(
                task,
                targets=["x", "youtube"],
                test_mode=True,
                visibility="private",
            )
        )

    youtube.assert_not_awaited()
    x.assert_not_awaited()
    apple.assert_not_called()


@pytest.mark.parametrize(
    ("target", "identity", "config_changes"),
    [
        (
            "youtube",
            {"channel_id": "UC-old", "channel_name": "Old Channel"},
            {
                "VIDEO_PUBLISH_YOUTUBE_CHANNEL_ID": "UC-new",
                "VIDEO_PUBLISH_YOUTUBE_CHANNEL_NAME": "New Channel",
            },
        ),
        (
            "x",
            {"username": "old_handle"},
            {"VIDEO_PUBLISH_X_HANDLE": "new_handle"},
        ),
        (
            "apple_podcast",
            {"feed_title": "Old Feed", "author": "Old Author"},
            {
                "VIDEO_PUBLISH_APPLE_FEED_TITLE": "New Feed",
                "VIDEO_PUBLISH_APPLE_FEED_AUTHOR": "New Author",
            },
        ),
    ],
)
def test_publication_plan_refuses_receipt_from_a_previous_destination_identity(
    tmp_path,
    target,
    identity,
    config_changes,
):
    task = _publication_task(tmp_path)
    asset = Path(task.audio_path if target == "apple_podcast" else task.video_path)
    receipt = {
        "status": "feed_ready" if target == "apple_podcast" else "published",
        "sha256": _sha256(asset),
        "identity": identity,
    }
    if target in {"youtube", "x"}:
        receipt["test_mode"] = True
    if target == "youtube":
        receipt["visibility"] = "private"
    manifest = {"task_id": task.id, "platforms": {target: receipt}}

    with (
        patch.multiple(config, **config_changes),
        patch(
            "backend.publishing.automatic_publication_configuration_errors",
            return_value=[],
        ),
        patch("backend.publishing.read_publication_manifest", return_value=manifest),
        pytest.raises(ValueError, match="already has an active publication"),
    ):
        plan_publication(
            task,
            targets=[target],
            test_mode=True,
            visibility="private",
        )


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


def test_x_identity_gate_refuses_the_wrong_signed_in_account():
    with patch.object(config, "VIDEO_PUBLISH_X_HANDLE", "expected_handle"):
        with pytest.raises(OpenCLIError, match="configured @expected_handle"):
            _assert_expected_x_identity({"username": "other_handle"})


def test_youtube_identity_gate_checks_name_and_stable_channel_id():
    identity = {"channel_name": "Frontier Tech Daily", "channel_id": "UC-wrong"}
    with (
        patch.object(config, "VIDEO_PUBLISH_YOUTUBE_CHANNEL_NAME", "Frontier Tech Daily"),
        patch.object(config, "VIDEO_PUBLISH_YOUTUBE_CHANNEL_ID", "UC-expected"),
    ):
        with pytest.raises(OpenCLIError, match="configured channel ID UC-expected"):
            _assert_expected_youtube_identity(identity)


def test_automatic_publication_requires_explicit_target_identities():
    with (
        patch.object(config, "VIDEO_PUBLISH_YOUTUBE_ENABLED", True),
        patch.object(config, "VIDEO_PUBLISH_YOUTUBE_CHANNEL_NAME", ""),
        patch.object(config, "VIDEO_PUBLISH_YOUTUBE_CHANNEL_ID", ""),
        patch.object(config, "VIDEO_PUBLISH_X_ENABLED", True),
        patch.object(config, "VIDEO_PUBLISH_X_HANDLE", ""),
    ):
        errors = automatic_publication_configuration_errors(["youtube", "x"])

    assert "YouTube channel name is not configured" in errors
    assert "YouTube channel ID is not configured" in errors
    assert "X account handle is not configured" in errors


def test_global_kill_switch_blocks_daily_automatic_publication():
    task = SimpleNamespace(origin_type="daily_news")

    with patch.object(config, "VIDEO_AUTO_PUBLISH_ENABLED", False):
        result = asyncio.run(run_auto_publish_pipeline(task))

    assert result.action == "awaiting_review"
    assert result.reason == "Global automatic publication is disabled"


def test_manual_task_is_not_reclassified_when_global_switch_is_off():
    task = SimpleNamespace(origin_type="manual")

    with patch.object(config, "VIDEO_AUTO_PUBLISH_ENABLED", False):
        result = asyncio.run(run_auto_publish_pipeline(task))

    assert result.action == "not_applicable"


def test_apple_feed_copies_audio_from_persisted_task_root(tmp_path):
    configured_root = tmp_path / "current-output-root"
    old_task_root = tmp_path / "persisted-old-root" / "task-1"
    old_task_root.mkdir(parents=True)
    audio = old_task_root / "combined.wav"
    audio.write_bytes(b"immutable narration")
    task = SimpleNamespace(
        id="task-1",
        audio_path=str(audio),
        output_dir=str(old_task_root),
        generated_title="Episode title",
        source_title=None,
        origin_id=None,
        duration_seconds=42,
    )

    with (
        patch.object(config, "OUTPUTS_DIR", configured_root),
        patch.object(config, "VIDEO_PUBLISH_APPLE_FEED_PUBLIC_BASE_URL", ""),
        patch(
            "backend.settings_store.restart_required_change_pending",
            return_value=False,
        ),
    ):
        result = prepare_apple_podcast(task)

    copied = configured_root / "podcast" / "media" / "task-1.wav"
    assert copied.read_bytes() == b"immutable narration"
    episodes = json.loads(
        (configured_root / "podcast" / "episodes.json").read_text(encoding="utf-8")
    )
    assert episodes[0]["audio_url"] == "/outputs/podcast/media/task-1.wav"
    assert result["status"] == "feed_ready"


@pytest.mark.parametrize(
    "episodes_payload",
    [
        b"{broken",
        json.dumps({"not": "a list"}).encode(),
        json.dumps([{"guid": "old-but-incomplete"}]).encode(),
    ],
    ids=["malformed-json", "non-list", "invalid-item"],
)
def test_apple_feed_refuses_to_replace_a_corrupt_episode_ledger(
    tmp_path,
    episodes_payload,
):
    configured_root = tmp_path / "current-output-root"
    podcast_dir = configured_root / "podcast"
    media_dir = podcast_dir / "media"
    media_dir.mkdir(parents=True)
    episodes_path = podcast_dir / "episodes.json"
    feed_path = podcast_dir / "feed.xml"
    old_media = media_dir / "old.wav"
    episodes_path.write_bytes(episodes_payload)
    feed_path.write_bytes(b"<rss>old feed evidence</rss>")
    old_media.write_bytes(b"old episode audio")
    task = _publication_task(tmp_path / "tasks", "task-apple-corrupt")

    with (
        patch.object(config, "OUTPUTS_DIR", configured_root),
        patch(
            "backend.settings_store.restart_required_change_pending",
            return_value=False,
        ),
    ):
        with pytest.raises(PublicationLedgerError):
            prepare_apple_podcast(task)

    assert episodes_path.read_bytes() == episodes_payload
    assert feed_path.read_bytes() == b"<rss>old feed evidence</rss>"
    assert old_media.read_bytes() == b"old episode audio"
    assert not (media_dir / f"{task.id}.wav").exists()
    assert not (Path(task.output_dir) / "publications" / "manifest.json").exists()


def test_apple_feed_refuses_existing_feed_without_episode_ledger(tmp_path):
    configured_root = tmp_path / "current-output-root"
    podcast_dir = configured_root / "podcast"
    podcast_dir.mkdir(parents=True)
    feed_path = podcast_dir / "feed.xml"
    feed_path.write_bytes(b"<rss><guid>historical-episode</guid></rss>")
    task = _publication_task(tmp_path / "tasks", "task-apple-orphaned-feed")

    with (
        patch.object(config, "OUTPUTS_DIR", configured_root),
        patch(
            "backend.settings_store.restart_required_change_pending",
            return_value=False,
        ),
    ):
        with pytest.raises(PublicationLedgerError, match="without its episode ledger"):
            prepare_apple_podcast(task)

    assert feed_path.read_bytes() == b"<rss><guid>historical-episode</guid></rss>"
    assert not (podcast_dir / "media" / f"{task.id}.wav").exists()
    assert not (Path(task.output_dir) / "publications" / "manifest.json").exists()


def test_apple_feed_refuses_empty_ledger_with_historical_feed_and_media(tmp_path):
    configured_root = tmp_path / "current-output-root"
    podcast_dir = configured_root / "podcast"
    media_dir = podcast_dir / "media"
    media_dir.mkdir(parents=True)
    episodes_path = podcast_dir / "episodes.json"
    feed_path = podcast_dir / "feed.xml"
    old_media = media_dir / "historical.wav"
    episodes_path.write_text("[]", encoding="utf-8")
    feed_path.write_bytes(b"<rss><guid>historical-episode</guid></rss>")
    old_media.write_bytes(b"historical audio")
    task = _publication_task(tmp_path / "tasks", "task-apple-empty-ledger")

    with (
        patch.object(config, "OUTPUTS_DIR", configured_root),
        patch(
            "backend.settings_store.restart_required_change_pending",
            return_value=False,
        ),
    ):
        with pytest.raises(PublicationLedgerError, match="ledger is empty"):
            prepare_apple_podcast(task)

    assert episodes_path.read_text(encoding="utf-8") == "[]"
    assert feed_path.read_bytes() == b"<rss><guid>historical-episode</guid></rss>"
    assert old_media.read_bytes() == b"historical audio"
    assert not (media_dir / f"{task.id}.wav").exists()


def test_apple_feed_concurrent_and_repeated_updates_preserve_other_episodes(tmp_path):
    configured_root = tmp_path / "current-output-root"

    def apple_task(task_id, title, audio_bytes):
        task_root = tmp_path / "persisted" / task_id
        task_root.mkdir(parents=True)
        audio = task_root / "combined.wav"
        audio.write_bytes(audio_bytes)
        return SimpleNamespace(
            id=task_id,
            audio_path=str(audio),
            output_dir=str(task_root),
            generated_title=title,
            source_title=None,
            origin_id=None,
            duration_seconds=42,
        )

    first = apple_task("task-1", "First episode", b"first narration")
    second = apple_task("task-2", "Second episode", b"second narration")

    with (
        patch.object(config, "OUTPUTS_DIR", configured_root),
        patch.object(config, "VIDEO_PUBLISH_APPLE_FEED_PUBLIC_BASE_URL", ""),
        patch(
            "backend.settings_store.restart_required_change_pending",
            return_value=False,
        ),
    ):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(prepare_apple_podcast, [first, second]))
        first.generated_title = "First episode, corrected"
        prepare_apple_podcast(first)

    podcast_dir = configured_root / "podcast"
    episodes = json.loads(
        (podcast_dir / "episodes.json").read_text(encoding="utf-8")
    )
    assert [item["guid"] for item in episodes] == ["task-1", "task-2"]
    assert episodes[0]["title"] == "First episode, corrected"
    assert len([item for item in episodes if item["guid"] == "task-1"]) == 1

    feed_guids = [
        node.text
        for node in ET.parse(podcast_dir / "feed.xml").findall("./channel/item/guid")
    ]
    assert feed_guids == ["task-1", "task-2"]
    assert not list(podcast_dir.rglob("*.tmp"))


def test_browser_json_accepts_boolean_eval_results():
    browser = AsyncMock(return_value="true")

    with patch("backend.publishing._browser", browser):
        result = asyncio.run(_browser_json("ftsn-yt-test", "eval", "(()=>true)()"))

    assert result is True


def test_browser_json_forwards_single_attempt_for_irreversible_actions():
    browser = AsyncMock(return_value="true")

    with patch("backend.publishing._browser", browser):
        result = asyncio.run(
            _browser_json(
                "ftsn-yt-test",
                "eval",
                "commit-once",
                attempts=1,
            )
        )

    assert result is True
    browser.assert_awaited_once_with(
        "ftsn-yt-test",
        "eval",
        "commit-once",
        timeout=180,
        attempts=1,
    )


def test_browser_json_recovers_prefixed_object_results():
    browser = AsyncMock(return_value='OpenCLI result: {"ready": true}')

    with patch("backend.publishing._browser", browser):
        result = asyncio.run(_browser_json("ftsn-yt-test", "eval", "some-js"))

    assert result == {"ready": True}


def test_browser_json_accepts_plain_string_eval_results():
    browser = AsyncMock(return_value="Frontier Tech Daily [TEST]")

    with patch("backend.publishing._browser", browser):
        result = asyncio.run(_browser_json("ftsn-yt-test", "eval", "title-js"))

    assert result == "Frontier Tech Daily [TEST]"


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
    assert browser_json.await_args_list[-1].kwargs["attempts"] == 1


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
