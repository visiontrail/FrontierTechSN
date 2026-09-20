"""UHD output contracts, source fidelity, and stale-cache regression coverage."""

import asyncio
import json
import os
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend import config
from backend.pipeline import collage_broll, composer, footage, web_footage
from backend.pipeline.video_format import resolve_frame_spec, resolve_render_spec


@pytest.mark.parametrize("orientation,size", [
    ("landscape", (3840, 2160)), ("portrait", (2160, 3840)),
])
def test_uhd_keeps_layout_and_supersamples_delivery(monkeypatch, tmp_path, orientation, size):
    monkeypatch.setattr(config, "RENDER_RESOLUTION", "4k")
    monkeypatch.setattr(config, "RENDER_FPS", 30)
    frame = resolve_frame_spec(orientation)
    render = resolve_render_spec(frame)
    assert (frame.width * 2, frame.height * 2) == size
    assert (frame.media_width, frame.media_height) == size
    assert (render.width, render.height) == size
    command = composer._build_render_command(tmp_path, tmp_path / "out.mp4", frame, render)
    assert command[command.index("--resolution") + 1] == f"{orientation}-4k"
    assert command[command.index("--fps") + 1] == "30"
    assert "--sdr" in command
    # Mid-render Admin changes cannot change capture/validation expectations.
    monkeypatch.setattr(config, "RENDER_RESOLUTION", "1080p")
    monkeypatch.setattr(config, "RENDER_FPS", 15)
    assert composer._build_render_command(tmp_path, tmp_path / "out.mp4", frame, render) == command


@pytest.mark.parametrize("legacy", ["1080p", "landscape", "portrait", "square"])
def test_legacy_resolution_preserves_task_orientation(monkeypatch, legacy):
    monkeypatch.setattr(config, "RENDER_RESOLUTION", legacy)
    render = resolve_render_spec(resolve_frame_spec("portrait"))
    assert (render.width, render.height, render.resolution) == (1080, 1920, "portrait")


def test_resolution_change_invalidates_review_checkpoint_and_extends_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RENDER_RESOLUTION", "1080p")
    request = {"video_orientation": "landscape"}
    old_hash = composer._review_retry_fingerprint(tmp_path, request)
    hd = resolve_render_spec()
    monkeypatch.setattr(config, "RENDER_RESOLUTION", "4k")
    uhd = resolve_render_spec()
    assert old_hash != composer._review_retry_fingerprint(tmp_path, request)
    assert composer._render_timeout(300, uhd) > composer._render_timeout(300, hd)


def test_uhd_encoder_and_quiet_deadlines_cover_long_episodes(monkeypatch):
    monkeypatch.setenv("FFMPEG_ENCODE_TIMEOUT_MS", "600000")
    monkeypatch.setenv("RENDER_ENV_INHERITANCE_TEST", "preserved")
    hd = replace(resolve_render_spec(), width=1920, height=1080, fps=30)
    uhd = replace(hd, width=3840, height=2160)
    timeout = composer._render_timeout(600, uhd)
    stall_timeout = composer._render_stall_timeout(600, uhd)
    assert 960 < stall_timeout < timeout
    assert stall_timeout > composer._render_stall_timeout(600, hd)
    assert composer._render_stall_timeout(600, replace(uhd, fps=60)) > stall_timeout
    env = composer._render_environment(timeout, stall_timeout)
    assert int(env["FFMPEG_ENCODE_TIMEOUT_MS"]) == stall_timeout * 1000
    assert int(env["FFMPEG_PROCESS_TIMEOUT_MS"]) == stall_timeout * 1000
    assert int(env["FFMPEG_STREAMING_TIMEOUT_MS"]) == timeout * 1000
    assert env["RENDER_ENV_INHERITANCE_TEST"] == "preserved"
    assert os.environ["FFMPEG_ENCODE_TIMEOUT_MS"] == "600000"


def test_quiet_deadline_covers_frozen_browser_protocol_limit():
    render = replace(resolve_render_spec(), protocol_timeout_ms=1800000)
    assert composer._render_stall_timeout(1, render) >= 1860


@pytest.mark.parametrize("rate", ["15/1", "0/0", "bad", "nan/1", None])
def test_delivery_rejects_wrong_or_missing_frame_rate(tmp_path, rate):
    render = replace(resolve_render_spec(), width=3840, height=2160, fps=30)
    result = SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
        "streams": [
            {"codec_type": "video", "width": 3840, "height": 2160,
             "avg_frame_rate": rate, "r_frame_rate": rate},
            {"codec_type": "audio"},
        ], "format": {"duration": "5"},
    }))
    with patch.object(composer, "run_capture_logged", return_value=result) as run:
        failures = composer._rendered_video_failures(
            tmp_path / "out.mp4", frame=resolve_frame_spec("landscape"),
            expected_duration=5, render=render,
        )
    assert any("fps" in failure for failure in failures)
    assert run.call_count == 1  # Never attempt promotion/full decode on bad headers.


def test_youtube_prefers_uhd_webm_over_lower_resolution_mp4():
    from yt_dlp import YoutubeDL

    with YoutubeDL({"quiet": True, "format": web_footage._youtube_format(),
                    "format_sort": ["res:2160"]}) as downloader:
        result = downloader.process_ie_result({
            "id": "resolution-test", "title": "Resolution test",
            "formats": [
                {"format_id": "hd-mp4", "url": "https://example.test/hd.mp4",
                 "width": 1280, "height": 720, "ext": "mp4", "vcodec": "avc1", "acodec": "none"},
                {"format_id": "uhd-webm", "url": "https://example.test/uhd.webm",
                 "width": 3840, "height": 2160, "ext": "webm", "vcodec": "vp9", "acodec": "none"},
            ],
        }, download=False)
    assert result["format_id"] == "uhd-webm"


@pytest.mark.parametrize("size,orientation", [
    ((3840, 2160), "landscape"), ((1280, 720), "landscape"),
    ((2160, 3840), "portrait"),
])
def test_real_ffmpeg_preserves_native_resolution_across_media_paths(tmp_path, size, orientation):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for source-fidelity integration coverage")
    width, height = size
    source = tmp_path / "source.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        f"color=c=navy:s={width}x{height}:r=30:d=0.3", "-c:v", "libx264",
        "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(source),
    ], check=True, timeout=60)

    async def run():
        normalized = await footage._normalize_render_clip(source, tmp_path / "normalized.mp4")
        assert (normalized["width"], normalized["height"]) == size
        trimmed = tmp_path / "trimmed.mp4"
        await web_footage._trim(source, trimmed, {"start_seconds": 0, "end_seconds": .3}, orientation)
        probe = await web_footage._probe(trimmed)
        assert (probe["width"], probe["height"]) == size
        (tmp_path / "video").mkdir()
        generated = await collage_broll._normalize_video(source, tmp_path, resolve_frame_spec(orientation), .3)
        probe = await web_footage._probe(generated)
        assert (probe["width"], probe["height"]) == size

    asyncio.run(run())
