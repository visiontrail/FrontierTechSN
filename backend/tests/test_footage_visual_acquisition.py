"""Do visual rejection before composition, without discarding usable downloads."""
import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from backend.pipeline import footage, web_footage


def commons_candidate(name):
    return {
        "provider_id": "wikimedia", "provider": "Wikimedia Commons",
        "source_page_url": f"https://commons.wikimedia.org/wiki/File:{name}.webm",
        "download_url": f"https://upload.wikimedia.org/{name}.webm",
        "title": f"City cyclists {name}", "description": "City cyclists crossing streets",
        "license": "CC0", "mime_type": "video/webm", "duration_seconds": 90,
    }


def setup_commons(tmp_path, monkeypatch, candidates):
    script = tmp_path / "script.txt"
    script.write_text("City cyclists cross a changing street.")

    async def download(client, *, candidate, destination):
        content = candidate["source_page_url"].encode()
        destination.write_bytes(content)
        return len(content), hashlib.sha256(content).hexdigest()

    async def normalize(source, destination):
        content = source.read_bytes()
        destination.write_bytes(content)
        return {
            "duration_seconds": 15, "width": 1920, "height": 1080,
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "render_metadata_version": 1, "render_profile": {},
            "acquisition_profile": footage.acquisition_profile(),
        }

    downloader = AsyncMock(side_effect=download)
    normalizer = AsyncMock(side_effect=normalize)
    monkeypatch.setattr(footage, "search_wikimedia", AsyncMock(return_value=candidates))
    monkeypatch.setattr(footage, "_download_candidate", downloader)
    monkeypatch.setattr(footage, "_normalize_render_clip", normalizer)
    kwargs = dict(
        task_id="test", task_dir=tmp_path, title="City cycling", script_path=script,
        clip_count=1, orientation="landscape", license_policy="open_only",
        provider_id=None, ai_endpoint=None, ai_model=None, supplied_queries=["city cyclists"],
    )
    return kwargs, downloader, normalizer


def accepted_preview(task_dir, name="approved.mp4"):
    preview = task_dir / name
    preview.write_bytes(b"reviewed interval")
    return {
        "analyzer": "gemini-web-contact-sheet", "image_received": True,
        "suitable": True, "confidence": .95, "visible_content": "Cyclists crossing a city street",
        "reviewed_preview_path": name,
        "reviewed_preview_sha256": hashlib.sha256(preview.read_bytes()).hexdigest(),
    }


def test_commons_visual_rejection_tries_next_candidate_and_retains_evidence(tmp_path, monkeypatch):
    candidates = [commons_candidate("wrong-event"), commons_candidate("actual-cycling")]
    kwargs, download, normalize = setup_commons(tmp_path, monkeypatch, candidates)
    review = AsyncMock(side_effect=[
        web_footage.WebFootageError("Preview visual review rejected: unrelated political visit"),
        accepted_preview(tmp_path),
    ])
    monkeypatch.setattr(web_footage, "_analyze_candidate_preview", review)

    manifest = asyncio.run(footage.acquire_public_footage(**kwargs))

    assert manifest["status"] == "ready"
    assert download.await_count == review.await_count == 2
    assert normalize.await_count == 1
    assert normalize.await_args.args[0] == tmp_path / "approved.mp4"
    clip = manifest["clips"][0]
    assert clip["source_page_url"] == candidates[1]["source_page_url"]
    assert footage._commons_visual_rejection(clip) == ""
    assert len(manifest["rejected_candidates"]) == 1
    assert "political visit" in manifest["rejected_candidates"][0]["reason"]
    assert len(list((tmp_path / "footage").glob("source-clip-*.webm"))) == 2
    assert manifest["pending_reviews"] == []


def test_commons_review_unavailable_retains_source_and_retry_does_not_download_again(tmp_path, monkeypatch):
    kwargs, download, normalize = setup_commons(tmp_path, monkeypatch, [commons_candidate("cycling")])
    review = AsyncMock(side_effect=web_footage.WebFootageReviewUnavailable("browser unavailable"))
    monkeypatch.setattr(web_footage, "_analyze_candidate_preview", review)

    with pytest.raises(web_footage.WebFootageReviewUnavailable):
        asyncio.run(footage.acquire_public_footage(**kwargs))
    saved = footage.read_manifest(tmp_path)
    assert saved["status"] == "review_unavailable"
    assert saved["clips"] == saved["rejected_candidates"] == []
    assert len(saved["pending_reviews"]) == 1
    normalize.assert_not_awaited()

    review.side_effect = None
    review.return_value = accepted_preview(tmp_path)
    manifest = asyncio.run(footage.acquire_public_footage(**kwargs))
    assert manifest["status"] == "ready"
    assert download.await_count == 1
    assert review.await_count == 2
    assert manifest["pending_reviews"] == []


def save_legacy_commons(tmp_path, candidate, *, provider="wikimedia"):
    folder = tmp_path / "footage"
    folder.mkdir()
    source = folder / "source-clip-01.webm"
    source.write_bytes(b"previous complete original video")
    render = folder / "clip-01-render.mp4"
    render.write_bytes(b"previous unreviewed render")
    excerpt = "City cyclists cross a changing street."
    shot = {"query": "city cyclists", "purpose": excerpt, "script_excerpt": excerpt}
    clip = {
        **candidate, **shot, "id": "clip-01", "source_duration_seconds": 90,
        "duration_seconds": 20, "source_local_path": source.relative_to(tmp_path).as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "local_path": render.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(render.read_bytes()).hexdigest(),
        "acquisition_profile": footage.acquisition_profile(), "render_safe": True,
        "render_metadata_version": 1,
    }
    manifest = {
        "provider_id": provider, "orientation": "landscape", "clips": [clip], "queries": [shot],
        "requested_clip_count": 1, "selection_mode": "manual", "binding_version": 2,
        "script_sha256": hashlib.sha256(excerpt.encode()).hexdigest(), "status": "ready",
    }
    footage._write_manifest(tmp_path, manifest)
    return manifest, source


@pytest.mark.parametrize("hybrid", [False, True])
def test_legacy_commons_source_is_reused_but_gets_current_visual_review(tmp_path, monkeypatch, hybrid):
    candidate = commons_candidate("cycling")
    kwargs, download, _ = setup_commons(tmp_path, monkeypatch, [candidate])
    _, source = save_legacy_commons(tmp_path, candidate, provider="hybrid-youtube" if hybrid else "wikimedia")
    review = AsyncMock(return_value=accepted_preview(tmp_path))
    monkeypatch.setattr(web_footage, "_analyze_candidate_preview", review)
    supplement = AsyncMock(side_effect=lambda **kw: kw["manifest"])
    monkeypatch.setattr(web_footage, "supplement_web_footage", supplement)

    if hybrid:
        result = asyncio.run(footage.acquire_footage(**{**kwargs, "media_provider": "hybrid", "supplied_queries": None}))
        assert supplement.call_args.kwargs["query_plan"] == []
    else:
        result = asyncio.run(footage.acquire_public_footage(**kwargs))
    download.assert_not_awaited()
    review.assert_awaited_once()
    assert review.await_args.kwargs["source_path"] == source.resolve()
    assert review.await_args.args[0]["duration_seconds"] == 90
    assert len(result["clips"]) == 1
    assert footage._commons_visual_rejection(result["clips"][0]) == ""
    assert source.read_bytes() == b"previous complete original video"


@pytest.mark.parametrize("defect", ["changed", "outside", "oversized"])
def test_legacy_source_migration_requires_task_path_hash_and_size(tmp_path, monkeypatch, defect):
    manifest, source = save_legacy_commons(tmp_path, commons_candidate("cycling"))
    clip = manifest["clips"][0]
    if defect == "changed":
        source.write_bytes(b"changed bytes")
    elif defect == "outside":
        clip["source_local_path"] = str(tmp_path.parent / "not-in-task.webm")
    else:
        monkeypatch.setattr(footage.config, "FOOTAGE_MAX_BYTES", 1)
    assert footage._pending_commons_sources(tmp_path, manifest) == []


@pytest.mark.parametrize("skip", ["occupied_query", "occupied_source", "rejected"])
def test_retained_commons_does_not_overfill_or_repeat_rejection(tmp_path, skip):
    manifest, _ = save_legacy_commons(tmp_path, commons_candidate("cycling"))
    pending = footage._pending_commons_sources(tmp_path, manifest)
    old_clip = manifest["clips"][0]
    manifest["pending_reviews"] = pending
    if skip == "occupied_source":
        manifest["clips"] = [{**old_clip, "query": "another subject"}]
    elif skip == "rejected":
        manifest["clips"] = []
        manifest["rejected_candidates"] = [{
            "source_page_url": old_clip["source_page_url"], "query": old_clip["query"],
            "script_excerpt": old_clip["script_excerpt"], "reason": "wrong event",
        }]
    with patch.object(footage, "_review_commons_source", AsyncMock()) as review:
        asyncio.run(footage._review_retained_commons(
            tmp_path, manifest, manifest["queries"], old_clip["script_excerpt"], log=None,
        ))
    review.assert_not_awaited()


def test_incomplete_commons_download_is_removed_but_not_marked_pending(tmp_path, monkeypatch):
    kwargs, _, _ = setup_commons(tmp_path, monkeypatch, [commons_candidate("cycling")])

    async def fail_download(client, *, candidate, destination):
        destination.write_bytes(b"truncated transfer")
        raise RuntimeError("network interrupted")

    monkeypatch.setattr(footage, "_download_candidate", fail_download)
    manifest = asyncio.run(footage.acquire_public_footage(**kwargs))
    assert manifest["pending_reviews"] == []
    assert list((tmp_path / "footage").glob("source-clip-*")) == []


@pytest.mark.parametrize("defect", ["missing", "rejected", "wrong_artifact", "wrong_narration"])
def test_commons_cached_clip_requires_review_of_current_artifact_and_narration(tmp_path, defect):
    clip = {**commons_candidate("cycling"), "query": "city cyclists", "purpose": "cycling",
            "script_excerpt": "City cyclists cross a changing street.", "sha256": "approved-sha"}
    clip["analysis"] = {
        **accepted_preview(tmp_path), "approved_clip_sha256": clip["sha256"],
        "review_binding": {key: clip[key] for key in ("query", "purpose", "script_excerpt")},
    }
    assert footage._commons_visual_rejection(clip) == ""
    if defect == "missing":
        clip.pop("analysis")
    elif defect == "rejected":
        clip["analysis"]["suitable"] = False
    elif defect == "wrong_artifact":
        clip["sha256"] = "new-sha"
    else:
        clip["script_excerpt"] = "Different event."
    assert footage._commons_visual_rejection(clip)


@pytest.mark.parametrize("message", [
    "ERROR: [youtube] sample: This video is not available",
    "ERROR: [youtube] sample: Private video",
    "ERROR: This video has been removed by the uploader",
])
def test_unavailable_youtube_source_does_not_repeat_download_fallbacks(tmp_path, message):
    runner = AsyncMock(side_effect=web_footage.WebFootageError(message))
    with patch.object(web_footage, "_run_command", runner), patch.object(web_footage, "_yt_dlp_bin", return_value="yt-dlp"):
        with pytest.raises(web_footage.WebFootageError, match="ERROR"):
            asyncio.run(web_footage._download_youtube(
                {"source_page_url": "https://www.youtube.com/watch?v=sample", "duration_seconds": 900},
                tmp_path, {"start_seconds": 30, "end_seconds": 45},
            ))
    assert runner.await_count == 1


def test_short_youtube_source_uses_size_bounded_native_download(tmp_path):
    async def run(command, **kwargs):
        (tmp_path / "sample.webm").write_bytes(b"complete nine-second source")
        return 0, "", ""

    with patch.object(web_footage, "_run_command", AsyncMock(side_effect=run)) as runner:
        path, sectioned = asyncio.run(web_footage._download_youtube(
            {"source_page_url": "https://www.youtube.com/watch?v=sample", "duration_seconds": 9},
            tmp_path, {"start_seconds": .9, "end_seconds": 9},
        ))
    assert path.name == "sample.webm"
    assert sectioned is False
    assert runner.await_count == 1
    command = runner.await_args.args[0]
    assert "--download-sections" not in command
    assert "--max-filesize" in command


def test_full_download_fallback_is_reused_for_later_preview_windows(tmp_path):
    source = tmp_path / "full.webm"
    source.write_bytes(b"full source")
    with (
        patch.object(web_footage, "_download_youtube", AsyncMock(return_value=(source, False))) as download,
        patch.object(web_footage, "_probe", AsyncMock(return_value={"duration_seconds": 100})),
        patch.object(web_footage, "_trim", AsyncMock()) as trim,
        patch.object(web_footage, "_run_command", AsyncMock()),
        patch.object(web_footage, "_save_prepared_preview"),
    ):
        prepared = asyncio.run(web_footage._prepare_candidate_preview(
            {"source_page_url": "https://www.youtube.com/watch?v=sample", "duration_seconds": 100},
            "Actual story", tmp_path,
        ))
    assert download.await_count == 1
    assert len(prepared["intervals"]) == trim.await_count == 3
    assert [call.args[0] for call in trim.await_args_list] == [source] * 3
    assert trim.await_args_list[2].args[2]["start_seconds"] == 67
