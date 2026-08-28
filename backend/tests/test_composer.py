import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest

from backend.pipeline import composer, visual_plan
from backend.pipeline.video_format import PORTRAIT


def test_verified_orpheus_chunks_define_exact_program_segments(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "tts_input_generated.wav"
    audio.write_bytes(b"wav-placeholder")
    (audio_dir / "part-1.txt").write_text("Good morning.")
    (audio_dir / "part-2.txt").write_text("First story.")
    (audio_dir / "part-3.txt").write_text("Thanks for watching.")
    (audio_dir / "tts_manifest.json").write_text(
        json.dumps(
            {
                "integrity": {"passed": True, "verified_source_coverage": 1.0},
                "parts": [
                    {"input": "part-1.txt", "duration_seconds": 1.25},
                    {"input": "part-2.txt", "duration_seconds": 2.5},
                    {"input": "part-3.txt", "duration_seconds": 1.75},
                ],
            }
        )
    )
    physical = [
        {"text": "Good morning.", "word_count": 2},
        {"text": "First story.", "word_count": 2},
        {"text": "Thanks for watching.", "word_count": 3},
    ]

    segments = composer._manifest_program_segments(audio, physical)
    alignment = composer._apply_manifest_program_alignment(
        {"passed": False, "word_coverage": 0.5, "failure_reasons": ["missed"]},
        {"passed": True},
        segments,
    )

    assert [row["start"] for row in segments] == [0.0, 1.25, 3.75]
    assert [row["duration"] for row in segments] == [1.25, 2.5, 1.75]
    assert alignment["passed"] is True
    assert alignment["method"] == "orpheus_manifest_program_timeline"
    assert alignment["whole_file_asr_observation"]["passed"] is False


def test_program_opening_copy_keeps_exact_show_title_and_date_visible():
    plans = [{"id": "scene-01", "kicker": "MORNING", "headline": "Generic"}]
    board = {
        "program_timeline": True,
        "scenes": [
            {
                "text": (
                    "It's Friday, August 28, 2026, and this is Frontier Tech Daily—"
                    "your concise morning briefing."
                )
            }
        ],
    }

    composer._enforce_program_opening_copy(plans, board)

    assert plans[0]["kicker"] == "FRIDAY, AUGUST 28, 2026"
    assert plans[0]["headline"] == "Frontier Tech Daily"


def _cache_board() -> dict:
    return {
        "thesis": "Frontier systems are crossing into daily work.",
        "scenes": [
            {
                "id": "scene-01",
                "index": 0,
                "text": "Robots learn from government-backed training centers.",
                "start": 0.0,
                "duration": 7.25,
                "keywords": ["robots", "training"],
            },
            {
                "id": "scene-02",
                "index": 1,
                "text": "Bacteria can form circuit-like computing systems.",
                "start": 7.25,
                "duration": 5.5,
                "keywords": ["bacteria", "circuits"],
            },
        ],
    }


def _cache_plans() -> list[dict]:
    return [
        {
            "id": "scene-01",
            "headline": "Robot training loop",
            "archetype": "footage",
            "footage_src": "../collage.mp4",
            "collage_broll": True,
        },
        {
            "id": "scene-02",
            "archetype": "news_image",
            "headline": "Living circuit boards",
            "news_image": True,
            "news_image_src": "../news_images/bacteria.jpg",
            "news_image_mode": "fullscreen",
            "news_image_original_archetype": "statement",
        },
        {"id": visual_plan.OUTRO_SCENE_ID, "headline": "Outro"},
    ]


def _write_cache(tmp_path, board: dict | None = None, plans: list[dict] | None = None) -> None:
    composer._write_visual_plan_checkpoint(
        tmp_path,
        board or _cache_board(),
        plans or _cache_plans(),
    )


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


def test_kit_plans_preserve_the_fifth_narrative_item():
    items = [
        "Summarize annual reports across multiple documents",
        "Search and compare company operational data online",
        "Control a desktop browser for retrieval and document generation",
        "Build English-language presentations from data",
        "Generate marketing posters from reference images",
        "A sixth unsupported task",
    ]

    plans = composer._kit_plans(
        [{"id": "scene-11", "archetype": "list", "items": items}],
        [{"id": "scene-11", "duration": 12.0}],
        composer.scene_kit.DEFAULT_THEME,
    )

    assert plans[0].items == tuple(items[: composer.scene_kit.MAX_NARRATIVE_ITEMS])
    assert plans[0].items[-1] == "Generate marketing posters from reference images"


def test_cached_scene_plans_require_exact_current_checkpoint_and_strip_placements(tmp_path):
    board = _cache_board()
    _write_cache(tmp_path, board)

    cached = composer._load_cached_scene_plans(tmp_path, board)

    assert cached is not None
    assert [item["id"] for item in cached] == ["scene-01", "scene-02"]
    assert cached[0]["archetype"] == "topic"
    assert "collage_broll" not in cached[0]
    assert "footage_src" not in cached[0]
    assert cached[1] == {
        "id": "scene-02",
        "archetype": "statement",
        "headline": "Living circuit boards",
    }
    assert isinstance(json.loads((tmp_path / "visual_plan.json").read_text()), list)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("thesis", "A different thesis"),
        ("thesis", None),
        ("index", 4),
        ("text", "Unrelated narration with the same scene id"),
        ("start", 0.125),
        ("duration", 8.0),
        ("keywords", ["different", "anchors"]),
    ],
)
def test_cached_scene_plans_reject_changed_storyboard_semantics(
    tmp_path,
    field,
    replacement,
):
    board = _cache_board()
    _write_cache(tmp_path, board)
    changed = copy.deepcopy(board)
    if field == "thesis":
        changed[field] = replacement
    else:
        changed["scenes"][0][field] = replacement

    assert composer._load_cached_scene_plans(tmp_path, changed) is None


def test_cached_scene_plans_reject_missing_and_legacy_sidecars(tmp_path):
    board = _cache_board()
    (tmp_path / "visual_plan.json").write_text(
        json.dumps(_cache_plans()), encoding="utf-8"
    )
    assert composer._load_cached_scene_plans(tmp_path, board) is None

    _write_cache(tmp_path, board)
    (tmp_path / composer.VISUAL_PLAN_CACHE_FILENAME).unlink()
    assert composer._load_cached_scene_plans(tmp_path, board) is None


@pytest.mark.parametrize("sidecar", [b"not-json", b'{"cache_version": NaN}'])
def test_cached_scene_plans_reject_corrupt_or_nonfinite_sidecars(tmp_path, sidecar):
    board = _cache_board()
    _write_cache(tmp_path, board)
    (tmp_path / composer.VISUAL_PLAN_CACHE_FILENAME).write_bytes(sidecar)

    assert composer._load_cached_scene_plans(tmp_path, board) is None


def test_cached_scene_plans_reject_tampered_plan_and_sidecar(tmp_path):
    board = _cache_board()
    _write_cache(tmp_path, board)
    plan_path = tmp_path / "visual_plan.json"
    plan_path.write_bytes(plan_path.read_bytes() + b"\n")
    assert composer._load_cached_scene_plans(tmp_path, board) is None

    _write_cache(tmp_path, board)
    cache_path = tmp_path / composer.VISUAL_PLAN_CACHE_FILENAME
    cache = json.loads(cache_path.read_text())
    cache["visual_plan_sha256"] = "0" * 64
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    assert composer._load_cached_scene_plans(tmp_path, board) is None


@pytest.mark.parametrize("nonfinite", ["NaN", "Infinity", "1e999"])
def test_cached_scene_plans_reject_nonfinite_plan_even_with_matching_sha(
    tmp_path,
    nonfinite,
):
    board = _cache_board()
    _write_cache(tmp_path, board)
    plan_bytes = (
        '[{"id":"scene-01","stat":'
        + nonfinite
        + '},{"id":"scene-02"},{"id":"scene-99-outro"}]'
    ).encode()
    plan_path = tmp_path / "visual_plan.json"
    plan_path.write_bytes(plan_bytes)
    cache_path = tmp_path / composer.VISUAL_PLAN_CACHE_FILENAME
    cache = json.loads(cache_path.read_text())
    cache["visual_plan_bytes"] = len(plan_bytes)
    cache["visual_plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    assert composer._load_cached_scene_plans(tmp_path, board) is None


def test_cached_scene_plans_reject_prompt_contract_version_change(tmp_path, monkeypatch):
    board = _cache_board()
    _write_cache(tmp_path, board)
    monkeypatch.setattr(
        composer,
        "VISUAL_PLAN_PROMPT_CONTRACT_VERSION",
        composer.VISUAL_PLAN_PROMPT_CONTRACT_VERSION + 1,
    )

    assert composer._load_cached_scene_plans(tmp_path, board) is None


def test_cached_scene_plans_reject_prompt_bytes_change(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    source_prompt = composer.config.PROMPTS_DIR / "visual_plan.txt"
    prompt_path = prompt_dir / "visual_plan.txt"
    prompt_path.write_bytes(source_prompt.read_bytes())
    monkeypatch.setattr(composer.config, "PROMPTS_DIR", prompt_dir)
    board = _cache_board()
    _write_cache(tmp_path, board)

    prompt_path.write_text("changed visual planner contract", encoding="utf-8")

    assert composer._load_cached_scene_plans(tmp_path, board) is None


def test_direction_checkpoint_survives_a_downstream_failure_without_attachments(
    tmp_path,
    monkeypatch,
):
    board = {**_cache_board(), "scene_count": 2}
    direction = [
        {"id": "scene-01", "archetype": "topic", "headline": "Robot training loop"},
        {"id": "scene-02", "archetype": "statement", "headline": "Living circuits"},
    ]
    planner_calls = 0

    async def plan_scene_visuals(*_args, **_kwargs):
        nonlocal planner_calls
        planner_calls += 1
        return copy.deepcopy(direction)

    monkeypatch.setattr(composer.visual_plan, "plan_scene_visuals", plan_scene_visuals)

    async def fail_after_direction():
        plans = await composer._load_or_plan_scene_visuals(
            tmp_path,
            board,
            ai_endpoint=None,
            ai_model=None,
            provider_id=None,
            log=lambda _message: None,
        )
        plans[0].update(
            {
                "archetype": "footage",
                "collage_broll": True,
                "footage_src": "../late-stage-collage.mp4",
            }
        )
        raise RuntimeError("downstream collage failed")

    with pytest.raises(RuntimeError, match="downstream collage failed"):
        asyncio.run(fail_after_direction())

    assert planner_calls == 1
    assert (tmp_path / composer.VISUAL_PLAN_CACHE_FILENAME).is_file()
    cached = composer._load_cached_scene_plans(tmp_path, board)
    assert cached == direction

    async def planner_must_not_repeat(*_args, **_kwargs):
        raise AssertionError("planner repeated after downstream failure")

    monkeypatch.setattr(
        composer.visual_plan,
        "plan_scene_visuals",
        planner_must_not_repeat,
    )
    retried = asyncio.run(
        composer._load_or_plan_scene_visuals(
            tmp_path,
            board,
            ai_endpoint=None,
            ai_model=None,
            provider_id=None,
            log=lambda _message: None,
        )
    )
    assert retried == direction


def test_visual_plan_checkpoint_commits_sidecar_last(tmp_path, monkeypatch):
    board = _cache_board()
    _write_cache(tmp_path, board)
    replacement = _cache_plans()
    replacement[0]["headline"] = "Replacement direction"
    real_atomic_write = composer._atomic_write_bytes

    def fail_sidecar(path, payload):
        if path.name == composer.VISUAL_PLAN_CACHE_FILENAME:
            raise RuntimeError("simulated crash before sidecar commit")
        real_atomic_write(path, payload)

    monkeypatch.setattr(composer, "_atomic_write_bytes", fail_sidecar)
    with pytest.raises(RuntimeError, match="simulated crash"):
        composer._write_visual_plan_checkpoint(tmp_path, board, replacement)

    assert not (tmp_path / composer.VISUAL_PLAN_CACHE_FILENAME).exists()
    assert json.loads((tmp_path / "visual_plan.json").read_text())[0]["headline"] == (
        "Replacement direction"
    )
    assert composer._load_cached_scene_plans(tmp_path, board) is None


def test_partial_news_image_error_reports_acquired_attached_and_missing_scenes():
    message = composer._news_image_placement_error(
        {
            "status": "partial",
            "images": [{"scene_id": "scene-02"}, {"scene_id": "scene-04"}, {"scene_id": "scene-05"}],
            "missing_scene_ids": ["scene-03", "scene-07", "scene-09", "scene-10"],
        },
        attached_images=0,
        required_count=7,
    )

    assert "scout acquired 3/7" in message
    assert "manifest status=partial" in message
    assert "missing scenes=scene-03,scene-07,scene-09,scene-10" in message
    assert "0/7 reached final scenes" in message


def test_partial_news_images_are_reported_as_delivery_warning():
    warnings = composer._quality_warnings(
        {"passed": True},
        {"passed": True, "scenes": []},
        news_image_inventory={
            "enabled": True,
            "complete": False,
            "warning": "Licensed image shortfall: 5/8 attached.",
        },
    )

    assert warnings == ["Licensed image shortfall: 5/8 attached."]


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


def test_failed_review_blocks_delivery():
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
    assert report["quality_status"] == "failed"
    assert report["delivery_status"] == "blocked"
    assert "81.42" in report["warnings"][0]


def test_failed_quality_report_never_promotes_and_preserves_candidate(tmp_path, monkeypatch):
    previous = tmp_path / "video.mp4"
    staged = tmp_path / "video.next.mp4"
    staged_report = tmp_path / "av_sync_report.next.json"
    previous.write_bytes(b"previous completed cut")
    staged.write_bytes(b"new rejected candidate")
    staged_report.write_text('{"passed": false}', encoding="utf-8")
    promoted = False

    def should_not_promote(*_args, **_kwargs):
        nonlocal promoted
        promoted = True
        raise AssertionError("quality-failed candidate was promoted")

    monkeypatch.setattr(composer, "_promote_render_candidate", should_not_promote)

    with pytest.raises(RuntimeError, match="failed the final A/V quality gate"):
        composer._promote_quality_gated_candidate(
            staged,
            staged_report,
            tmp_path,
            composer._sha256_path(staged),
            {
                "passed": False,
                "warnings": ["Gemini rendered-frame review average stayed below 82"],
            },
        )

    assert promoted is False
    assert previous.read_bytes() == b"previous completed cut"
    assert staged.read_bytes() == b"new rejected candidate"
    assert staged_report.read_text(encoding="utf-8") == '{"passed": false}'


def test_portrait_render_command_uses_task_resolution(tmp_path):
    command = composer._build_render_command(tmp_path, tmp_path / "video.mp4", PORTRAIT)

    assert command[command.index("--resolution") + 1] == "portrait"
    assert command[command.index("--protocol-timeout") + 1] == str(
        composer.config.RENDER_PROTOCOL_TIMEOUT_MS
    )
