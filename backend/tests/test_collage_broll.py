import asyncio
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from backend.pipeline import collage_broll
from backend.pipeline.video_format import FrameSpec, LANDSCAPE, PORTRAIT


def _board(count: int = 6) -> dict:
    return {
        "thesis": "Systems preserve judgment",
        "scenes": [
            {
                "id": f"scene-{index + 1:02d}",
                "start": index * 6.0,
                "duration": 6.0,
                "text": f"Narration idea {index + 1} becomes a visible system.",
                "keywords": ["system", f"idea-{index + 1}"],
            }
            for index in range(count)
        ],
    }


class _CollageCacheHarness:
    """Deterministic media boundaries for cache-contract regression tests."""

    def __init__(self, task_dir: Path) -> None:
        self.task_dir = task_dir
        self.calls = {
            "plan": 0,
            "still": 0,
            "frames": 0,
            "video": 0,
            "normalize": 0,
            "probe": 0,
            "sheet": 0,
        }
        self._stack = ExitStack()

    def __enter__(self):
        for name, replacement in (
            ("plan_specs", self.plan_specs),
            ("_generate_still", self.generate_still),
            ("_prepare_frames", self.prepare_frames),
            ("_generate_video", self.generate_video),
            ("_normalize_video", self.normalize_video),
            ("probe_video", self.probe_video),
            ("_contact_sheet", self.contact_sheet),
        ):
            self._stack.enter_context(patch.object(collage_broll, name, replacement))
        return self

    def __exit__(self, *args):
        return self._stack.__exit__(*args)

    async def plan_specs(
        self,
        storyboard,
        *,
        count,
        force_opening,
        **_kwargs,
    ):
        self.calls["plan"] += 1
        choices = collage_broll._scene_choices(storyboard, count, force_opening)
        return [
            collage_broll._fallback_spec(scene, index)
            for index, scene in enumerate(choices)
        ]

    async def generate_still(self, prompt, item_dir):
        self.calls["still"] += 1
        output = item_dir / "stills" / "generated.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"still:" + prompt.encode("utf-8"))
        return output, "https://chatgpt.test/collage"

    async def prepare_frames(self, source, item_dir, _color, _frame):
        self.calls["frames"] += 1
        frames = item_dir / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        first = frames / "first-frame.png"
        last = frames / "last-frame.jpg"
        first.write_bytes(b"first")
        last.write_bytes(b"hold:" + source.read_bytes())
        return first, last

    async def generate_video(self, prompt, _first, last, item_dir, _frame):
        self.calls["video"] += 1
        raw = item_dir / "video" / "raw.mp4"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"raw:" + prompt.encode("utf-8") + last.read_bytes())
        return raw, "https://gemini.test/collage"

    async def normalize_video(self, raw, item_dir, _frame, target_duration):
        self.calls["normalize"] += 1
        final = collage_broll._final_clip_path(item_dir, target_duration)
        final.write_bytes(b"final:" + raw.read_bytes())
        return final

    async def probe_video(self, _video, _frame, _target_duration):
        self.calls["probe"] += 1
        return {"passed": True, "checks": {"deterministic": True}}

    async def contact_sheet(self, _video, item_dir, _frame, _target_duration):
        self.calls["sheet"] += 1
        sheet = item_dir / "video" / "contact-sheet.jpg"
        sheet.write_bytes(b"sheet")
        return sheet

    def run(self, storyboard: dict, *, count: int | None = None) -> dict:
        return asyncio.run(
            collage_broll.generate_collage_broll(
                storyboard,
                self.task_dir,
                count=count if count is not None else len(storyboard["scenes"]),
                force_opening=False,
                frame=LANDSCAPE,
            )
        )


def test_scene_selection_forces_opening_and_spreads_the_rest():
    selected = collage_broll._scene_choices(_board(), 4, True)

    assert len(selected) == 4
    assert selected[0]["id"] == "scene-01"
    assert len({scene["id"] for scene in selected}) == 4
    assert selected[-1]["id"] == "scene-06"


def test_prompts_follow_orientation_keep_media_clean_and_leave_style_open():
    spec = {
        **collage_broll._fallback_spec(_board(1)["scenes"][0], 0),
        "art_direction": "surreal torn-photo assemblage with painted interruptions",
        "color_direction": "acid brights collide with dusty archival neutrals",
        "composition_direction": "an unstable edge-weighted diagonal",
        "motion_direction": "the barrier loses authority as the hidden system gains agency",
        "elements": [
            {
                "what": "a folded paper clock",
                "role": "the old constraint",
                "motion": "its hands race backward while its face visibly buckles",
                "placement": "pressed against the lower edge at the end",
            },
            {
                "what": "a flock of translucent moths",
                "role": "new agency",
                "motion": "they wake one by one and redirect the clock's shadow",
                "placement": "surrounding the clock without forming a border",
            },
        ],
    }

    still = collage_broll.image_prompt(spec, LANDSCAPE)
    motion = collage_broll.video_prompt(spec, PORTRAIT)

    assert "16:9" in still
    assert "9:16" in motion
    assert "Treat collage as an open medium, not a house style" in still
    assert spec["art_direction"] in still
    assert spec["composition_direction"] in still
    assert "Reinterpret, combine, crop, abstract, or subordinate" in still
    assert "Image 1 and Image 2 are endpoint constraints only" in motion
    assert spec["motion_direction"] in motion
    for element in spec["elements"]:
        assert all(str(value) in still for value in element.values())
        assert all(str(value) in motion for value in element.values())
    assert "video-generation model owns the entire intermediate visual grammar" in motion
    assert "No transition mechanism" in motion
    assert "named animation technique is prescribed" in motion
    assert "Suggested narrative progression" not in motion
    assert "one continuous" not in motion
    assert "Target running time: approximately 6.000 seconds" in motion
    assert "Do not restart or loop the generated sequence" in motion
    assert "Content exclusions only" in still
    assert "Content exclusions only" in motion


def test_normalize_spec_preserves_freeform_ai_art_direction_and_sparse_elements():
    scene = _board(1)["scenes"][0]
    raw = {
        "art_direction": "dense hand-painted maximalist scrapbook",
        "color_direction": "nearly monochrome oxblood with one electric yellow interruption",
        "composition_direction": "edge-to-edge layers with no central hero",
        "motion_direction": "one broad sheet tears open to reveal nested fragments",
        "accent_colors": [f"swatch-{index}" for index in range(9)],
        "elements": [
            {
                "what": f"dynamic ingredient {index}",
                "role": "the entire metaphor",
                "motion": "changes visibly",
                "placement": "edge to edge",
            }
            for index in range(11)
        ],
        "assembly_order": ["one monumental torn sheet"],
    }

    spec = collage_broll._normalize_spec(raw, scene, 0)

    assert spec["art_direction"] == raw["art_direction"]
    assert spec["color_direction"] == raw["color_direction"]
    assert spec["composition_direction"] == raw["composition_direction"]
    assert spec["motion_direction"] == raw["motion_direction"]
    assert spec["accent_colors"] == raw["accent_colors"]
    assert spec["elements"] == raw["elements"]
    assert "assembly_order" not in spec


def test_clip_duration_matches_script_and_respects_gemini_ceiling():
    short_scene = {**_board(1)["scenes"][0], "duration": 5.25}
    long_scene = {**_board(1)["scenes"][0], "duration": 12.0}

    with patch.object(collage_broll.config, "COLLAGE_GEMINI_MAX_SECONDS", 8):
        short = collage_broll._fallback_spec(short_scene, 0)
        long = collage_broll._fallback_spec(long_scene, 0)

    assert short["script_duration_seconds"] == 5.25
    assert short["target_duration_seconds"] == 5.25
    assert long["script_duration_seconds"] == 12.0
    assert long["target_duration_seconds"] == 8.0


def test_semantically_unchanged_collage_reuses_visual_spec_and_final(tmp_path: Path):
    storyboard = _board(1)

    with _CollageCacheHarness(tmp_path) as harness:
        first = harness.run(storyboard)
        final = tmp_path / first["items"][0]["video_path"]
        first_bytes = final.read_bytes()
        second = harness.run(storyboard)

    assert first["status"] == second["status"] == "ready"
    assert first["visual_spec_cache"] == "miss"
    assert second["visual_spec_cache"] == "hit"
    assert second["items"][0]["cache_reuse"] == {"still": False, "final": True}
    assert harness.calls == {
        "plan": 1,
        "still": 1,
        "frames": 1,
        "video": 1,
        "normalize": 1,
        "probe": 2,
        "sheet": 2,
    }
    assert final.read_bytes() == first_bytes
    envelope = json.loads(
        (tmp_path / "collage_broll" / "visual-spec.json").read_text()
    )
    assert envelope["specs_sha256"] == collage_broll._fingerprint(envelope["specs"])
    final_contract = json.loads(
        (
            tmp_path
            / "collage_broll"
            / "01-scene-01"
            / "video"
            / "cache-contract.json"
        ).read_text()
    )
    assert final_contract["artifact_sha256"] == collage_broll._file_sha256(final)
    assert final_contract["hold_frame_sha256"] == collage_broll._file_sha256(
        tmp_path / first["items"][0]["still_path"]
    )


def test_same_scene_id_with_changed_narration_invalidates_every_cache_layer(
    tmp_path: Path,
):
    storyboard = _board(1)

    with _CollageCacheHarness(tmp_path) as harness:
        first = harness.run(storyboard)
        item_dir = tmp_path / "collage_broll" / "01-scene-01"
        old_prompt = (item_dir / "image-prompt.txt").read_text()
        old_still = (item_dir / "stills" / "generated.png").read_bytes()
        old_final = (item_dir / "video" / "final-6s-noaudio.mp4").read_bytes()
        changed = json.loads(json.dumps(storyboard))
        changed["scenes"][0]["text"] = (
            "A different narration now describes a paper observatory opening."
        )
        observed: dict[str, str] = {}
        original_loader = collage_broll._load_final_cache

        def observe_prompt_before_validation(item_dir, **kwargs):
            observed["persisted"] = (item_dir / "image-prompt.txt").read_text()
            observed["expected"] = kwargs["still_prompt"] + "\n"
            return original_loader(item_dir, **kwargs)

        with patch.object(
            collage_broll, "_load_final_cache", side_effect=observe_prompt_before_validation
        ):
            second = harness.run(changed)

    assert first["planning_fingerprint"] != second["planning_fingerprint"]
    assert first["items"][0]["scene_id"] == second["items"][0]["scene_id"]
    assert first["items"][0]["scene_semantic_fingerprint"] != second["items"][0][
        "scene_semantic_fingerprint"
    ]
    assert second["visual_spec_cache"] == "miss"
    assert second["items"][0]["cache_reuse"] == {"still": False, "final": False}
    assert harness.calls["plan"] == 2
    assert harness.calls["still"] == 2
    assert harness.calls["normalize"] == 2
    assert observed["persisted"] == old_prompt
    assert observed["persisted"] != observed["expected"]
    assert (item_dir / "stills" / "generated.png").read_bytes() != old_still
    assert (item_dir / "video" / "final-6s-noaudio.mp4").read_bytes() != old_final


def test_partial_stale_final_preserves_exact_still_and_other_scene_artifacts(
    tmp_path: Path,
):
    storyboard = _board(2)

    with _CollageCacheHarness(tmp_path) as harness:
        harness.run(storyboard)
        root = tmp_path / "collage_broll"
        first_item = root / "01-scene-01"
        stale_item = root / "02-scene-02"
        first_sentinel = first_item / "keep-first.txt"
        stale_sentinel = stale_item / "keep-stale.txt"
        root_sentinel = root / "keep-root.txt"
        first_sentinel.write_text("first")
        stale_sentinel.write_text("stale")
        root_sentinel.write_text("root")
        still = stale_item / "stills" / "generated.png"
        still_bytes = still.read_bytes()
        (stale_item / "video" / "final-6s-noaudio.mp4").write_bytes(b"tampered")

        second = harness.run(storyboard)

    assert second["items"][0]["cache_reuse"] == {"still": False, "final": True}
    assert second["items"][1]["cache_reuse"] == {"still": True, "final": False}
    assert harness.calls["plan"] == 1
    assert harness.calls["still"] == 2
    assert harness.calls["normalize"] == 3
    assert still.read_bytes() == still_bytes
    assert first_sentinel.read_text() == "first"
    assert stale_sentinel.read_text() == "stale"
    assert root_sentinel.read_text() == "root"


def test_legacy_visual_spec_and_artifacts_fail_closed(tmp_path: Path):
    storyboard = _board(1)

    with _CollageCacheHarness(tmp_path) as harness:
        harness.run(storyboard)
        root = tmp_path / "collage_broll"
        item = root / "01-scene-01"
        envelope = json.loads((root / "visual-spec.json").read_text())
        (root / "visual-spec.json").write_text(json.dumps(envelope["specs"]))
        (item / "stills" / "cache-contract.json").unlink()
        (item / "video" / "cache-contract.json").unlink()

        second = harness.run(storyboard)

    assert second["visual_spec_cache"] == "miss"
    assert second["items"][0]["cache_reuse"] == {"still": False, "final": False}
    assert harness.calls["plan"] == 2
    assert harness.calls["still"] == 2
    assert harness.calls["normalize"] == 2
    rewritten = json.loads((root / "visual-spec.json").read_text())
    assert rewritten["cache_contract_version"] == collage_broll.CACHE_CONTRACT_VERSION
    assert rewritten["specs_sha256"] == collage_broll._fingerprint(rewritten["specs"])


def test_visual_spec_content_edit_with_stale_sha_replans_but_reuses_exact_media(
    tmp_path: Path,
):
    storyboard = _board(1)

    with _CollageCacheHarness(tmp_path) as harness:
        harness.run(storyboard)
        specs_path = tmp_path / "collage_broll" / "visual-spec.json"
        envelope = json.loads(specs_path.read_text())
        envelope["specs"][0]["visual_metaphor"] = "accidentally edited"
        specs_path.write_text(json.dumps(envelope))

        second = harness.run(storyboard)

    assert second["visual_spec_cache"] == "miss"
    assert second["items"][0]["cache_reuse"] == {"still": False, "final": True}
    assert harness.calls["plan"] == 2
    assert harness.calls["still"] == 1
    assert harness.calls["normalize"] == 1


def test_visual_spec_nan_is_a_cache_miss_instead_of_a_crash(tmp_path: Path):
    storyboard = _board(1)

    with _CollageCacheHarness(tmp_path) as harness:
        harness.run(storyboard)
        specs_path = tmp_path / "collage_broll" / "visual-spec.json"
        envelope = json.loads(specs_path.read_text())
        envelope["specs"][0]["elements"][0]["motion"] = float("nan")
        specs_path.write_text(json.dumps(envelope))

        recovered = harness.run(storyboard)

    assert recovered["status"] == "ready"
    assert recovered["visual_spec_cache"] == "miss"
    assert recovered["items"][0]["cache_reuse"] == {"still": False, "final": True}
    assert harness.calls["plan"] == 2
    assert harness.calls["still"] == 1
    assert "NaN" not in specs_path.read_text()


def test_fresh_nonfinite_planner_spec_falls_back_before_cache_write(tmp_path: Path):
    storyboard = _board(1)

    with _CollageCacheHarness(tmp_path) as harness:

        async def nonfinite_plan(*args, **kwargs):
            specs = await harness.plan_specs(*args, **kwargs)
            specs[0]["elements"][0]["motion"] = float("nan")
            return specs

        with patch.object(collage_broll, "plan_specs", nonfinite_plan):
            manifest = harness.run(storyboard)

    specs_path = tmp_path / "collage_broll" / "visual-spec.json"
    saved = specs_path.read_text()
    assert manifest["status"] == "ready"
    assert manifest["items"][0]["spec"]["planner"] == "deterministic_fallback"
    assert "NaN" not in saved
    assert json.loads(saved)["specs"][0]["planner"] == "deterministic_fallback"


def test_failed_still_decode_never_commits_reusable_contract(tmp_path: Path):
    storyboard = _board(1)

    with _CollageCacheHarness(tmp_path) as harness:
        with patch.object(
            collage_broll,
            "_prepare_frames",
            AsyncMock(side_effect=RuntimeError("decode failed")),
        ):
            failed = harness.run(storyboard)
        contract = (
            tmp_path
            / "collage_broll"
            / "01-scene-01"
            / "stills"
            / "cache-contract.json"
        )
        assert not contract.exists()

        recovered = harness.run(storyboard)

    assert failed["status"] == "failed"
    assert recovered["status"] == "ready"
    assert recovered["items"][0]["cache_reuse"] == {"still": False, "final": False}
    assert harness.calls["still"] == 2
    assert contract.is_file()


@pytest.mark.parametrize("symlink_level", ["task", "root", "item", "stage"])
def test_collage_cache_rejects_symlinked_owned_paths(
    tmp_path: Path,
    symlink_level: str,
):
    real_task = tmp_path / "real-task"
    real_task.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("untouched")
    task_dir = real_task
    root = real_task / "collage_broll"
    item = root / "01-scene-01"

    if symlink_level == "task":
        task_dir = tmp_path / "task-link"
        task_dir.symlink_to(real_task, target_is_directory=True)
    elif symlink_level == "root":
        root.symlink_to(outside, target_is_directory=True)
    elif symlink_level == "item":
        root.mkdir()
        item.symlink_to(outside, target_is_directory=True)
    else:
        item.mkdir(parents=True)
        (item / "video").symlink_to(outside, target_is_directory=True)

    with _CollageCacheHarness(task_dir) as harness:
        with pytest.raises(RuntimeError, match="symlink"):
            harness.run(_board(1))

    assert sentinel.read_text() == "untouched"
    assert list(outside.iterdir()) == [sentinel]


def test_collage_cache_rejects_scene_ids_that_can_escape_item_root(tmp_path: Path):
    storyboard = _board(1)
    storyboard["scenes"][0]["id"] = "../outside"

    with _CollageCacheHarness(tmp_path) as harness:
        with pytest.raises(ValueError, match="Unsafe collage scene id"):
            harness.run(storyboard)

    assert not (tmp_path / "collage_broll").exists()


def test_opencli_collage_operation_retries_until_success(monkeypatch):
    calls = 0

    async def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("browser busy")
        return "ready"

    monkeypatch.setattr(collage_broll, "run_opencli", fake_run)
    monkeypatch.setattr(collage_broll.config, "OPENCLI_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(collage_broll.asyncio, "sleep", AsyncMock())

    result = asyncio.run(
        collage_broll._run_opencli_retry(
            ["gemini", "video", "prompt"],
            timeout=10,
            label="test video",
            non_retryable=collage_broll._is_gemini_video_upload_capability_failure,
        )
    )

    assert result == "ready"
    assert calls == 3


def _blocked_gemini_video_upload_error() -> collage_broll.OpenCLIError:
    return collage_broll.OpenCLIError(
        'Gemini keyframe 1 upload failed at wait_for_attachment: '
        '{"method":"DataTransfer","nativeErrors":['
        '{"method":"page.setFileInput","error":"{\\"code\\":-32000,'
        '\\"message\\":\\"Not allowed\\"}"},'
        '{"method":"DOM.setFileInputFiles","error":'
        '"CDP method not permitted: Runtime.evaluate"}],'
        '"state":{"attachments":0,"busy":false,"inputFiles":[[]]}}'
    )


def _blocked_gemini_file_chooser_error() -> collage_broll.OpenCLIError:
    return collage_broll.OpenCLIError(
        'Gemini keyframe 1 upload failed at wait_for_attachment: '
        '{"method":"DataTransfer","nativeErrors":['
        '{"method":"page.setFileInput","error":"Page.fileChooserOpened '
        'not received within 5s"},'
        '{"method":"DOM.setFileInputFiles","error":'
        '"CDP method not permitted: Runtime.evaluate"}],'
        '"state":{"attachments":0,"busy":false,"inputFiles":[[]]}}'
    )


def _stuck_gemini_video_input_error(signature: str) -> collage_broll.OpenCLIError:
    return collage_broll.OpenCLIError(
        f"{collage_broll.GEMINI_VIDEO_INPUT_HYDRATION_STUCK_CODE}: "
        "Gemini keyframe 1 upload failed at discover_live_input: "
        f'{{"errorCode":"{collage_broll.GEMINI_VIDEO_INPUT_HYDRATION_STUCK_CODE}",'
        f'"stuckSignature":"{signature}"}}'
    )


def test_opencli_collage_operation_does_not_retry_stable_upload_capability_failure(
    monkeypatch,
):
    run = AsyncMock(side_effect=_blocked_gemini_video_upload_error())
    sleep = AsyncMock()
    monkeypatch.setattr(collage_broll, "run_opencli", run)
    monkeypatch.setattr(collage_broll.config, "OPENCLI_MAX_ATTEMPTS", 10)
    monkeypatch.setattr(collage_broll.asyncio, "sleep", sleep)

    with pytest.raises(
        collage_broll.GeminiVideoUploadCapabilityError,
        match=collage_broll.GEMINI_VIDEO_UPLOAD_CAPABILITY_CODE,
    ):
        asyncio.run(
            collage_broll._run_opencli_retry(
                ["gemini", "video", "prompt"],
                timeout=10,
                label="Gemini collage video",
                non_retryable=collage_broll._is_gemini_video_upload_capability_failure,
            )
        )

    assert run.await_count == 1
    sleep.assert_not_awaited()


def test_upload_classifier_recognizes_current_file_chooser_failure():
    assert collage_broll._is_gemini_video_upload_capability_failure(
        _blocked_gemini_file_chooser_error()
    )


def test_opencli_stops_after_two_identical_input_hydration_failures(monkeypatch):
    signature = "keyframe=2;attachments=1;busy=1;fresh=0;baseline=1;click=ok;button=ready;focus=1"
    run = AsyncMock(
        side_effect=[
            _stuck_gemini_video_input_error(signature),
            _stuck_gemini_video_input_error(signature),
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr(collage_broll, "run_opencli", run)
    monkeypatch.setattr(collage_broll.config, "OPENCLI_MAX_ATTEMPTS", 10)
    monkeypatch.setattr(collage_broll.asyncio, "sleep", sleep)

    with pytest.raises(collage_broll.GeminiVideoUploadCapabilityError) as raised:
        asyncio.run(
            collage_broll._run_opencli_retry(
                ["gemini", "video", "prompt"],
                timeout=10,
                label="Gemini collage video",
                non_retryable=collage_broll._is_gemini_video_upload_capability_failure,
                repeated_failure_signature=(
                    collage_broll._gemini_video_input_hydration_stuck_signature
                ),
                repeated_failure_threshold=2,
                repeated_failure_error_code=(
                    collage_broll.GEMINI_VIDEO_INPUT_HYDRATION_STUCK_CODE
                ),
            )
        )

    assert raised.value.error_code == (
        collage_broll.GEMINI_VIDEO_INPUT_HYDRATION_STUCK_CODE
    )
    assert run.await_count == 2
    assert sleep.await_count == 1


def _stuck_chatgpt_image_composer_error() -> collage_broll.OpenCLIError:
    return collage_broll.OpenCLIError(
        "OpenCLI chatgpt image failed with exit 1: ok: false\n"
        "code: COMMAND_EXEC\n"
        "message: Failed to send image prompt to ChatGPT\n"
        "help: Open https://chatgpt.com/new and verify the composer is ready."
    )


def test_chatgpt_still_stops_after_two_identical_composer_not_ready_failures(
    tmp_path: Path,
    monkeypatch,
):
    run = AsyncMock(
        side_effect=[
            _stuck_chatgpt_image_composer_error(),
            _stuck_chatgpt_image_composer_error(),
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr(collage_broll, "run_opencli", run)
    monkeypatch.setattr(collage_broll.config, "OPENCLI_MAX_ATTEMPTS", 10)
    monkeypatch.setattr(collage_broll.asyncio, "sleep", sleep)

    with pytest.raises(collage_broll.OpenCLIError) as raised:
        asyncio.run(collage_broll._generate_still("paper collage", tmp_path))

    assert not isinstance(
        raised.value, collage_broll.GeminiVideoUploadCapabilityError
    )
    assert collage_broll.CHATGPT_IMAGE_COMPOSER_NOT_READY_SIGNATURE in str(
        raised.value
    )
    assert run.await_count == 2
    assert sleep.await_count == 1


def test_chatgpt_composer_signature_streak_resets_after_a_different_failure(
    monkeypatch,
):
    run = AsyncMock(
        side_effect=[
            _stuck_chatgpt_image_composer_error(),
            collage_broll.OpenCLIError("browser temporarily busy"),
            _stuck_chatgpt_image_composer_error(),
            "ready",
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr(collage_broll, "run_opencli", run)
    monkeypatch.setattr(collage_broll.config, "OPENCLI_MAX_ATTEMPTS", 4)
    monkeypatch.setattr(collage_broll.asyncio, "sleep", sleep)

    result = asyncio.run(
        collage_broll._run_opencli_retry(
            ["chatgpt", "image", "prompt"],
            timeout=10,
            label="ChatGPT collage still",
            repeated_failure_signature=(
                collage_broll._chatgpt_image_composer_not_ready_signature
            ),
            repeated_failure_threshold=2,
        )
    )

    assert result == "ready"
    assert run.await_count == 4
    assert sleep.await_count == 3


@pytest.mark.parametrize(
    "message",
    [
        "OpenCLI chatgpt image failed with exit 1: code: COMMAND_EXEC",
        (
            "code: COMMAND_EXEC message: Failed to send image prompt to ChatGPT "
            "help: Open https://chatgpt.com/new and verify the composer is ready."
        ),
        (
            "OpenCLI chatgpt image failed with exit 1: code: COMMAND_EXEC "
            "message: Failed to send image prompt to ChatGPT"
        ),
        (
            "OpenCLI chatgpt image failed with exit 1: code: RATE_LIMIT "
            "message: Failed to send image prompt to ChatGPT "
            "help: Open https://chatgpt.com/new and verify the composer is ready."
        ),
    ],
)
def test_chatgpt_composer_classifier_requires_complete_stable_signature(message):
    assert (
        collage_broll._chatgpt_image_composer_not_ready_signature(
            collage_broll.OpenCLIError(message)
        )
        is None
    )


def test_opencli_does_not_combine_nonconsecutive_or_different_stuck_states(
    monkeypatch,
):
    first = "keyframe=1;attachments=0;busy=1;fresh=0;baseline=0;click=ok;button=ready;focus=1"
    different = "keyframe=2;attachments=1;busy=1;fresh=0;baseline=1;click=ok;button=ready;focus=1"
    run = AsyncMock(
        side_effect=[
            _stuck_gemini_video_input_error(first),
            _stuck_gemini_video_input_error(different),
            _stuck_gemini_video_input_error(first),
            "ready",
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr(collage_broll, "run_opencli", run)
    monkeypatch.setattr(collage_broll.config, "OPENCLI_MAX_ATTEMPTS", 4)
    monkeypatch.setattr(collage_broll.asyncio, "sleep", sleep)

    result = asyncio.run(
        collage_broll._run_opencli_retry(
            ["gemini", "video", "prompt"],
            timeout=10,
            label="Gemini collage video",
            non_retryable=collage_broll._is_gemini_video_upload_capability_failure,
            repeated_failure_signature=(
                collage_broll._gemini_video_input_hydration_stuck_signature
            ),
            repeated_failure_threshold=2,
        )
    )

    assert result == "ready"
    assert run.await_count == 4
    assert sleep.await_count == 3


@pytest.mark.parametrize(
    "message",
    [
        "browser busy",
        'page.setFileInput failed: {"code":-32000,"message":"Not allowed"}',
        "CDP method not permitted: Runtime.evaluate",
        '{"method":"DataTransfer","state":{"attachments":0,"inputFiles":[[]]}}',
    ],
)
def test_upload_capability_classifier_requires_the_complete_failure_signature(message):
    assert not collage_broll._is_gemini_video_upload_capability_failure(
        collage_broll.OpenCLIError(message)
    )


def test_agent_selects_beats_from_the_full_timeline():
    answer = json.dumps(
        [
            {"scene_id": "scene-03", "visual_metaphor": "a hinge reveals a hidden system"},
            {"scene_id": "scene-06", "visual_metaphor": "a bridge locks its final span"},
        ]
    )
    complete = AsyncMock(return_value=answer)
    with (
        patch(
            "backend.pipeline.digester._resolve_provider",
            AsyncMock(return_value=("https://example.test", "model", "key")),
        ),
        patch("backend.pipeline.agent.agent_complete", complete),
    ):
        specs = asyncio.run(
            collage_broll.plan_specs(
                _board(), count=2, force_opening=False, frame=LANDSCAPE
            )
        )

    assert [spec["scene_id"] for spec in specs] == ["scene-03", "scene-06"]
    assert all(spec["planner"] == "claude_agent_sdk" for spec in specs)
    planner_system = complete.await_args.args[0]
    assert "Paper collage is the medium, not a preset aesthetic" in planner_system
    assert "deliberately vary at least the composition strategy" in planner_system
    assert "video-generation model owns every intermediate visual" in planner_system
    assert "Do not prescribe a transition mechanism" in planner_system
    assert "every element's `what`, `role`, `motion`" in planner_system
    assert "assembly_order" not in planner_system
    assert '"art_direction"' in planner_system
    assert '"motion_direction"' in planner_system
    assert complete.await_args.kwargs["enable_skills"] is False
    assert complete.await_args.kwargs["disable_thinking"] is True
    assert complete.await_args.kwargs["max_tokens"] == 16384


@pytest.mark.parametrize("response", [
    '[{"scene_id":"scene-03","accent_colors":["#C9A227"],"final_frame\n[{"final_frame":"continued"}]',
    '[{"scene_id":"scene-03","elements":[{"what":"clock"}]',
    '["#C9A227", "#7B2D26"]',
    '[{"what":"clock"}]',
    '[]\n[{"scene_id":"scene-03"}]',
])
def test_invalid_outer_visual_spec_cannot_become_an_empty_plan(response):
    with pytest.raises(RuntimeError):
        collage_broll._json_array(response)


def test_visual_spec_parser_keeps_explicit_empty_and_complete_scene_arrays():
    assert collage_broll._json_array("[]") == []
    valid = [{"scene_id": "scene-03", "elements": [{"what": "clock"}]}]
    assert collage_broll._json_array("```json\n" + json.dumps(valid) + "\n```") == valid


@pytest.mark.parametrize("response", [
    '[{"scene_id":"scene-03","accent_colors":["#C9A227"],',
    '[{"scene_id":"not-a-candidate"}]',
])
def test_invalid_plan_preserves_response_and_uses_narration_fallback(tmp_path, response):
    log = []
    with (
        patch("backend.pipeline.digester._resolve_provider",
              AsyncMock(return_value=("https://example.test", "model", "key"))),
        patch("backend.pipeline.agent.agent_complete", AsyncMock(return_value=response)),
    ):
        specs = asyncio.run(collage_broll.plan_specs(
            _board(), count=None, force_opening=False, frame=LANDSCAPE,
            diagnostic_dir=tmp_path, log=log.append,
        ))
    assert specs
    assert (tmp_path / "planning-response.txt").read_text() == response
    assert any("using narration-derived specs" in item for item in log)


def test_agent_chooses_collage_quantity_when_no_count_is_configured():
    answer = json.dumps(
        [{"scene_id": "scene-03", "visual_metaphor": "a hinge reveals a hidden system"}]
    )
    complete = AsyncMock(return_value=answer)
    with (
        patch(
            "backend.pipeline.digester._resolve_provider",
            AsyncMock(return_value=("https://example.test", "model", "key")),
        ),
        patch("backend.pipeline.agent.agent_complete", complete),
    ):
        specs = asyncio.run(
            collage_broll.plan_specs(
                _board(), count=None, force_opening=False, frame=LANDSCAPE
            )
        )

    assert [spec["scene_id"] for spec in specs] == ["scene-03"]
    planner_system = complete.await_args.args[0]
    assert "there is no target count or quota" in planner_system
    assert "Select exactly" not in planner_system


def test_agent_choices_are_not_overridden_by_a_fixed_abstract_scene_rule():
    board = {
        "thesis": "Named systems produce an uncertain future",
        "scenes": [
            {
                "id": "scene-01",
                "start": 0.0,
                "duration": 6.0,
                "text": "Google announced a new system.",
                "keywords": [],
            },
            {
                "id": "scene-02",
                "start": 6.0,
                "duration": 6.0,
                "text": "Whether the future itself disappears remains the question.",
                "keywords": [],
            },
            {
                "id": "scene-03",
                "start": 12.0,
                "duration": 6.0,
                "text": "Microsoft released another product.",
                "keywords": [],
            },
        ],
    }
    answer = json.dumps(
        [
            {"scene_id": "scene-01", "visual_metaphor": "a paper search box"},
            {"scene_id": "scene-03", "visual_metaphor": "a folding software box"},
        ]
    )
    with (
        patch(
            "backend.pipeline.digester._resolve_provider",
            AsyncMock(return_value=("https://example.test", "model", "key")),
        ),
        patch("backend.pipeline.agent.agent_complete", AsyncMock(return_value=answer)),
    ):
        specs = asyncio.run(
            collage_broll.plan_specs(
                board,
                count=2,
                force_opening=False,
                frame=LANDSCAPE,
            )
        )

    assert [spec["scene_id"] for spec in specs] == ["scene-01", "scene-03"]
    assert specs[0]["planner"] == "claude_agent_sdk"


def test_attach_collage_only_promotes_ready_existing_clips(tmp_path: Path):
    clip = tmp_path / "collage_broll" / "01-scene-01" / "video" / "final-5s-noaudio.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"video")
    hold = tmp_path / "collage_broll" / "01-scene-01" / "frames" / "last-frame.jpg"
    hold.parent.mkdir(parents=True)
    hold.write_bytes(b"image")
    plans = [
        {"id": "scene-01", "archetype": "topic", "headline": "First"},
        {"id": "scene-02", "archetype": "topic", "headline": "Second"},
    ]
    manifest = {
        "items": [
            {
                "scene_id": "scene-01",
                "status": "ready",
                "video_path": str(clip.relative_to(tmp_path)),
                "still_path": str(hold.relative_to(tmp_path)),
                "script_duration_seconds": 12.0,
                "target_duration_seconds": 8.0,
                "spec": {"visual_metaphor": "a machine snaps together"},
                "qa": {"passed": True},
            },
            {
                "scene_id": "scene-02",
                "status": "failed",
                "video_path": "missing.mp4",
            },
        ]
    }

    assert collage_broll.attach_collage(plans, manifest, tmp_path) == 1
    assert plans[0]["archetype"] == "footage"
    assert plans[0]["collage_broll"] is True
    assert plans[0]["footage_src"] == f"../{clip.relative_to(tmp_path).as_posix()}"
    assert plans[0]["collage_hold_src"] == f"../{hold.relative_to(tmp_path).as_posix()}"
    assert plans[0]["collage_target_duration_seconds"] == 8.0
    assert plans[0]["footage_credit"] == ""
    assert plans[1]["archetype"] == "topic"

    # Refreshing an already-attached plan must update the same scene instead
    # of treating its own collage as occupied public footage and rehoming it.
    assert collage_broll.attach_collage(plans, manifest, tmp_path) == 1
    assert plans[0]["collage_placed_scene_id"] == "scene-01"
    assert plans[1]["archetype"] == "topic"


def test_attach_collage_keeps_fallback_when_a_required_hold_frame_is_missing(
    tmp_path: Path,
):
    clip = tmp_path / "collage_broll" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"video")
    plans = [{"id": "scene-01", "archetype": "topic", "headline": "Fallback"}]
    manifest = {
        "items": [
            {
                "scene_id": "scene-01",
                "status": "ready",
                "video_path": str(clip.relative_to(tmp_path)),
                "still_path": "collage_broll/missing-last-frame.jpg",
                "script_duration_seconds": 12.0,
                "target_duration_seconds": 8.0,
            }
        ]
    }

    assert collage_broll.attach_collage(plans, manifest, tmp_path) == 0
    assert plans[0] == {
        "id": "scene-01",
        "archetype": "topic",
        "headline": "Fallback",
    }


def test_attach_collage_shares_the_exact_story_without_overwriting_public_footage(
    tmp_path: Path,
):
    clip = tmp_path / "collage_broll" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"video")
    hold = tmp_path / "collage_broll" / "last-frame.jpg"
    hold.write_bytes(b"image")
    plans = [
        {
            "id": "scene-01",
            "archetype": "footage",
            "footage_src": "footage/public.mp4",
        },
        {"id": "scene-02", "archetype": "topic"},
    ]
    manifest = {
        "items": [
            {
                "scene_id": "scene-01",
                "status": "ready",
                "video_path": str(clip.relative_to(tmp_path)),
                "still_path": str(hold.relative_to(tmp_path)),
                "spec": {"visual_metaphor": "paper fleet"},
                "qa": {"passed": True},
            }
        ]
    }

    assert collage_broll.attach_collage(plans, manifest, tmp_path) == 1
    assert plans[0]["footage_src"] == "footage/public.mp4"
    assert plans[0]["collage_broll"] is True
    assert plans[0]["collage_src"] == "../collage_broll/clip.mp4"
    assert "collage_broll" not in plans[1]
    assert manifest["items"][0]["placed_scene_id"] == "scene-01"


def test_generate_rejects_procedural_imitation_when_web_video_fails(tmp_path: Path):
    specs = [
        collage_broll._fallback_spec(scene, index)
        for index, scene in enumerate(_board(2)["scenes"])
    ]
    still = tmp_path / "source.png"
    still.write_bytes(b"png")

    async def frames(_source, item_dir, _color, _frame):
        frame_dir = item_dir / "frames"
        frame_dir.mkdir(parents=True)
        first = frame_dir / "first-frame.png"
        last = frame_dir / "last-frame.png"
        first.write_bytes(b"first")
        last.write_bytes(b"last")
        return first, last

    calls = 0

    async def video(_prompt, _first, _last, item_dir, _frame):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("quota exhausted")
        raw = item_dir / "video" / "raw.mp4"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"raw")
        return raw, "https://gemini.google.com/videos/test"

    async def normalize(_raw, item_dir, _frame, target_duration):
        final = collage_broll._final_clip_path(item_dir, target_duration)
        final.write_bytes(b"final")
        return final

    async def sheet(_video, item_dir, _frame, _target_duration):
        output = item_dir / "video" / "contact-sheet.jpg"
        output.write_bytes(b"sheet")
        return output

    with (
        patch.object(collage_broll, "plan_specs", AsyncMock(return_value=specs)),
        patch.object(collage_broll, "_generate_still", AsyncMock(return_value=(still, "https://chatgpt.com/c/test"))),
        patch.object(collage_broll, "_prepare_frames", frames),
        patch.object(collage_broll, "_generate_video", video),
        patch.object(collage_broll, "_normalize_video", normalize),
        patch.object(collage_broll, "probe_video", AsyncMock(return_value={"passed": True})),
        patch.object(collage_broll, "_contact_sheet", sheet),
    ):
        manifest = asyncio.run(
            collage_broll.generate_collage_broll(
                _board(2), tmp_path, count=2, force_opening=False, frame=LANDSCAPE
            )
        )

    assert manifest["status"] == "partial"
    assert manifest["ready_count"] == 1
    assert manifest["gemini_api_key_used"] is False
    assert manifest["approval_gates"] == []
    assert manifest["motion_design_authority"] == "video_generation_model"
    assert manifest["procedural_motion_fallback_enabled"] is False
    assert [item["status"] for item in manifest["items"]] == ["ready", "failed"]
    assert [item["target_duration_seconds"] for item in manifest["items"]] == [6.0, 6.0]
    assert manifest["playback_policy"] == "play_once_then_hold_last_frame"
    assert manifest["items"][0]["video_provider"] == "gemini_web_create_video_via_opencli"
    assert "video_provider" not in manifest["items"][1]
    assert manifest["items"][1]["generation_warnings"] == []
    saved = json.loads((tmp_path / "collage_broll" / "manifest.json").read_text())
    assert len(saved["errors"]) == 1
    assert "quota exhausted" in saved["errors"][0]["message"]
    assert "procedural animation fallback is disabled" in saved["errors"][0]["message"]


def test_generate_reuses_upload_failure_without_creating_procedural_videos(
    tmp_path: Path,
):
    specs = [
        collage_broll._fallback_spec(scene, index)
        for index, scene in enumerate(_board(3)["scenes"])
    ]
    still = tmp_path / "source.png"
    still.write_bytes(b"png")

    async def frames(_source, item_dir, _color, _frame):
        frame_dir = item_dir / "frames"
        frame_dir.mkdir(parents=True)
        first = frame_dir / "first-frame.png"
        last = frame_dir / "last-frame.png"
        first.write_bytes(b"first")
        last.write_bytes(b"last")
        return first, last

    capability_error = collage_broll.GeminiVideoUploadCapabilityError(
        f"Gemini collage video cannot run "
        f"({collage_broll.GEMINI_VIDEO_INPUT_HYDRATION_STUCK_CODE})",
        error_code=collage_broll.GEMINI_VIDEO_INPUT_HYDRATION_STUCK_CODE,
    )
    generate_video = AsyncMock(side_effect=capability_error)
    normalize = AsyncMock()
    with (
        patch.object(collage_broll, "plan_specs", AsyncMock(return_value=specs)),
        patch.object(
            collage_broll,
            "_generate_still",
            AsyncMock(return_value=(still, "https://chatgpt.com/c/test")),
        ),
        patch.object(collage_broll, "_prepare_frames", frames),
        patch.object(collage_broll, "_generate_video", generate_video),
        patch.object(collage_broll, "_normalize_video", normalize),
    ):
        manifest = asyncio.run(
            collage_broll.generate_collage_broll(
                _board(3), tmp_path, count=3, force_opening=False, frame=LANDSCAPE
            )
        )

    assert manifest["status"] == "failed"
    assert manifest["ready_count"] == 0
    assert generate_video.await_count == 1
    assert normalize.await_count == 0
    assert [item["status"] for item in manifest["items"]] == ["failed"] * 3
    assert all("video_provider" not in item for item in manifest["items"])
    assert all(item["generation_warnings"] == [] for item in manifest["items"])
    assert manifest["gemini_video_upload_capability"] == {
        "status": "unavailable",
        "error_code": collage_broll.GEMINI_VIDEO_INPUT_HYDRATION_STUCK_CODE,
        "detected_at_scene_id": "scene-01",
    }
    assert len(manifest["errors"]) == 3
    assert all(
        "procedural animation fallback is disabled" in error["message"]
        for error in manifest["errors"]
    )


def test_generate_can_use_local_still_but_not_local_motion(tmp_path: Path):
    specs = [collage_broll._fallback_spec(_board(1)["scenes"][0], 0)]
    local_still = tmp_path / "local.png"
    local_still.write_bytes(b"png")

    async def frames(_source, item_dir, _color, _frame):
        frame_dir = item_dir / "frames"
        frame_dir.mkdir(parents=True)
        first = frame_dir / "first.png"
        last = frame_dir / "last.png"
        first.write_bytes(b"first")
        last.write_bytes(b"last")
        return first, last

    with (
        patch.object(collage_broll, "plan_specs", AsyncMock(return_value=specs)),
        patch.object(collage_broll, "_generate_still", AsyncMock(side_effect=RuntimeError("signed out"))),
        patch.object(collage_broll, "_render_local_still", AsyncMock(return_value=local_still)),
        patch.object(collage_broll, "_prepare_frames", frames),
        patch.object(collage_broll, "_generate_video", AsyncMock(side_effect=RuntimeError("quota"))),
    ):
        manifest = asyncio.run(
            collage_broll.generate_collage_broll(
                _board(1), tmp_path, count=1, force_opening=False, frame=LANDSCAPE
            )
        )

    assert manifest["status"] == "failed"
    assert manifest["ready_count"] == 0
    assert manifest["items"][0]["still_provider"] == "deterministic_local_paper_collage"
    assert "video_provider" not in manifest["items"][0]
    assert "procedural animation fallback is disabled" in manifest["items"][0]["error"]
    assert manifest["gemini_video_upload_capability"] == {
        "status": "unknown",
        "error_code": None,
        "detected_at_scene_id": None,
    }


def test_motion_probe_rejects_a_video_that_becomes_static(tmp_path: Path):
    frame = FrameSpec("landscape", "16:9", 320, 180, 320, 180, "landscape")
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    Image.new("RGB", (320, 180), "#315F4C").save(first)
    Image.new("RGB", (320, 180), "#D2A928").save(last)
    raw = tmp_path / "legacy.mp4"
    target_duration = 5.0
    asyncio.run(
        collage_broll._media_command(
            [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-framerate",
                str(collage_broll.CLIP_FPS),
                "-t",
                str(target_duration),
                "-i",
                str(first),
                "-loop",
                "1",
                "-framerate",
                str(collage_broll.CLIP_FPS),
                "-t",
                str(target_duration),
                "-i",
                str(last),
                "-filter_complex",
                "[0:v][1:v]xfade=transition=wiperight:duration=0.8:offset=0.35[out]",
                "-map",
                "[out]",
                "-t",
                str(target_duration),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(raw),
            ],
            timeout=120,
        )
    )

    qa = asyncio.run(collage_broll.probe_video(raw, frame, target_duration))

    assert qa["passed"] is False
    assert qa["checks"]["sustained_motion"] is False
    assert qa["motion"]["active_seconds"] < 2.5


def test_normalize_trims_without_replaying_the_source(tmp_path: Path):
    raw = tmp_path / "gemini.mp4"
    raw.write_bytes(b"video")
    target_duration = 6.25

    with patch.object(collage_broll, "_media_command", AsyncMock()) as media_command:
        final = asyncio.run(
            collage_broll._normalize_video(
                raw, tmp_path, LANDSCAPE, target_duration
            )
        )

    command = media_command.await_args.args[0]
    assert "-stream_loop" not in command
    assert command[command.index("-t") + 1] == str(target_duration)
    assert final.name == "final-6.25s-noaudio.mp4"


def test_candidate_scenes_exclude_spoken_program_bookends():
    storyboard = {
        "scenes": [
            {"id": "scene-01", "program_segment_kind": "opening"},
            {"id": "scene-02", "program_segment_kind": "news"},
            {"id": "scene-03", "program_segment_kind": "news"},
            {"id": "scene-04", "program_segment_kind": "closing"},
        ]
    }

    assert [
        scene["id"] for scene in collage_broll._candidate_scenes(storyboard)
    ] == ["scene-02", "scene-03"]
