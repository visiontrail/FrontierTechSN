import json
import re
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.models import TaskConfig, TaskResponse, TaskStatus
from backend.pipeline import assembler, composer, subtitles
from backend.routers import tasks


def storyboard():
    return {
        "audio_duration": 12.0,
        "total_duration": 22.0,
        "alignment": {"passed": True},
        "scenes": [
            {"lines": [{"start": 2.0, "duration": 3.0, "text": "Opening words."}]},
            {"lines": [{"start": 7.0, "duration": 5.0, "text": "你好，世界。This is the closing narration."}]},
            {"lines": []},
        ],
    }


def test_srt_uses_the_render_caption_groups_and_keeps_music_gaps(tmp_path):
    board = storyboard()
    path = subtitles.write_subtitles(board, tmp_path / "video-abc.mp4")
    content = path.read_text(encoding="utf-8")
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "00:00:02,000 --> 00:00:05,000" in content
    assert "00:00:07,000 -->" in content
    assert "00:00:12,000" in content
    assert "00:00:22,000" not in content
    cues = content.strip().split("\n\n")
    assert [int(cue.splitlines()[0]) for cue in cues] == list(range(1, len(cues) + 1))
    expected = assembler.caption_segments(board)
    assert [cue.splitlines()[2] for cue in cues] == [cue["text"] for cue in expected]
    html = "\n".join(assembler._caption_clips(board))
    assert len(re.findall('class="clip caption"', html)) == len(cues)


def test_timestamp_carries_milliseconds_and_supports_hours():
    assert subtitles._timestamp(round(59.9996 * 1000)) == "00:01:00,000"
    assert subtitles._timestamp(3_601_234) == "01:00:01,234"


@pytest.mark.parametrize("changes", [
    {"start": -1}, {"start": float("nan")}, {"duration": float("inf")},
    {"start": True}, {"duration": 0}, {"duration": 50}, {"text": " "},
])
def test_rejects_invalid_timing_before_replacing_an_existing_file(tmp_path, changes):
    board = storyboard()
    board["scenes"][0]["lines"][0].update(changes)
    path = tmp_path / "video-abc.srt"
    path.write_text("Previous good subtitles")
    with pytest.raises(ValueError):
        subtitles.write_subtitles(board, path.with_suffix(".mp4"))
    assert path.read_text() == "Previous good subtitles"


def test_empty_timeline_does_not_create_a_blank_upload():
    with pytest.raises(ValueError, match="No timed narration"):
        subtitles.render_srt({"audio_duration": 10, "scenes": []})


def test_successful_promotion_exports_subtitles_for_the_specific_cut(tmp_path):
    previous = tmp_path / "video-old.srt"
    previous.write_text("Old captions")
    staged = tmp_path / "video.next.mp4"
    staged.write_bytes(b"validated video")
    report = tmp_path / "av_sync_report.next.json"
    report.write_text('{"passed": true}')
    (tmp_path / "storyboard.json").write_text(json.dumps(storyboard()))
    digest = composer._sha256_path(staged)
    video = composer._promote_quality_gated_candidate(staged, report, tmp_path, digest, {"passed": True})
    assert video.with_suffix(".srt").read_text() == subtitles.render_srt(storyboard())
    assert previous.read_text() == "Old captions"


def test_rejected_video_never_exports_candidate_subtitles(tmp_path):
    (tmp_path / "storyboard.json").write_text(json.dumps(storyboard()))
    with pytest.raises(composer.QualityGateRetry):
        composer._promote_quality_gated_candidate(
            tmp_path / "video.next.mp4", tmp_path / "report.json", tmp_path, "a" * 64, {"passed": False},
        )
    assert not list(tmp_path.glob("*.srt"))


@pytest.mark.asyncio
async def test_download_is_utf8_and_tracks_the_video_state(tmp_path):
    video = tmp_path / "video-old.mp4"
    video.write_bytes(b"old video")
    path = subtitles.write_subtitles(storyboard(), video)
    task = TaskResponse(id="caption-test", created_at="2026-09-20", updated_at="2026-09-20",
                        source_type="topic", status="complete", config=TaskConfig(captions_enabled=False),
                        video_path=str(video))
    assert task.subtitles_available  # Independent of burned-in caption setting.
    app = FastAPI()
    app.include_router(tasks.router)
    with patch.object(tasks.db, "get_task", AsyncMock(return_value=task)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/tasks/caption-test/subtitles")
            assert response.status_code == 200
            assert response.content == path.read_bytes()
            assert response.headers["content-type"] == "application/x-subrip; charset=utf-8"
            assert 'filename="caption-test.srt"' in response.headers["content-disposition"]
            task.status = TaskStatus.FAILED
            assert (await client.get("/api/tasks/caption-test/subtitles")).status_code == 409
            response = await client.get("/api/tasks/caption-test/subtitles?artifact=retained")
            assert response.status_code == 200
            assert response.headers["x-video-artifact-state"] == "retained"
            assert 'filename="caption-test-retained.srt"' in response.headers["content-disposition"]
            # A new working storyboard must not change subtitles of the old cut.
            (tmp_path / "storyboard.json").write_text("{}")
            assert response.content == path.read_bytes()
            path.unlink()
            assert not task.subtitles_available
            assert (await client.get("/api/tasks/caption-test/subtitles?artifact=retained")).status_code == 404
