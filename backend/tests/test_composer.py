import json

from backend.pipeline import composer, visual_plan
from backend.pipeline.video_format import PORTRAIT


def test_mount_list_starts_with_content_and_has_no_title_card():
    board = {
        "content_start": 0.0,
        "outro_start": 12.0,
        "outro_duration": 5.0,
        "scenes": [
            {"id": "scene-01", "start": 0.0, "duration": 7.0},
            {"id": "scene-02", "start": 7.0, "duration": 5.0},
        ],
    }

    mounts = composer._mount_list(board)

    assert mounts[0] == {"id": "scene-01", "start": 0.0, "duration": 7.0}
    assert [mount["id"] for mount in mounts] == [
        "scene-01",
        "scene-02",
        visual_plan.OUTRO_SCENE_ID,
    ]


def test_cached_scene_plans_require_every_current_storyboard_scene(tmp_path):
    board = {"scenes": [{"id": "scene-01"}, {"id": "scene-02"}]}
    (tmp_path / "visual_plan.json").write_text(
        json.dumps(
            [
                {"id": "scene-02", "headline": "Second"},
                {"id": "scene-99-outro", "headline": "Outro"},
                {
                    "id": "scene-01",
                    "headline": "First",
                    "archetype": "footage",
                    "footage_src": "../collage.mp4",
                    "collage_broll": True,
                },
            ]
        ),
        encoding="utf-8",
    )

    cached = composer._load_cached_scene_plans(tmp_path, board)

    assert [item["id"] for item in cached] == ["scene-01", "scene-02"]
    assert cached[0]["archetype"] == "topic"
    assert "collage_broll" not in cached[0]
    assert "footage_src" not in cached[0]

    (tmp_path / "visual_plan.json").write_text(
        json.dumps([{"id": "scene-01"}]), encoding="utf-8"
    )
    assert composer._load_cached_scene_plans(tmp_path, board) is None


def test_quality_failures_become_final_delivery_warnings():
    warnings = composer._quality_warnings(
        {
            "passed": False,
            "failure_reasons": ["word-level transcript unavailable"],
            "transcription_failures": ["mlx-whisper attempt 3: timeout"],
        },
        {
            "passed": False,
            "scenes": [
                {"id": "scene-01", "grounded": True},
                {"id": "scene-02", "grounded": False},
            ],
        },
        {
            "status": "failed",
            "passed": False,
            "failed_scene_ids": ["scene-03"],
            "errors": ["batch 1: malformed response"],
        },
    )

    assert len(warnings) == 3
    assert "estimated timing was used" in warnings[0]
    assert "scene-02" in warnings[1]
    assert "scene-03" in warnings[2]
    assert "malformed response" in warnings[2]


def test_clean_quality_report_has_no_warnings():
    assert composer._quality_warnings(
        {"passed": True},
        {"passed": True, "scenes": []},
        {"status": "passed", "passed": True},
    ) == []


def test_failed_review_completes_delivery_with_warning():
    report = composer._finalize_quality_report(
        {},
        {"passed": True},
        {"passed": True, "scenes": []},
        {
            "status": "failed",
            "passed": False,
            "average_score": 81.42,
            "failed_scene_ids": [],
            "errors": [],
        },
        multimodal_enabled=True,
    )

    assert report["passed"] is False
    assert report["quality_status"] == "warning"
    assert report["delivery_status"] == "completed_with_warnings"
    assert "81.42" in report["warnings"][0]


def test_portrait_render_command_uses_task_resolution(tmp_path):
    command = composer._build_render_command(tmp_path, tmp_path / "video.mp4", PORTRAIT)

    assert command[command.index("--resolution") + 1] == "portrait"
