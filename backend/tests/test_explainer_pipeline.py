import asyncio
import json
import math
import shutil
import struct
import sys
import wave
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from backend import explainer_pipeline as pipeline
from backend.pipeline import chatterbox, tts


pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg is required")


def _project():
    return {
        "id": "example", "title": "An ordinary decision", "language": "en",
        "tts_model": "vibevoice-0.5b", "voice": "Carter", "captions_enabled": True,
        "style": {"background_color": "#F4EBDD", "ink_color": "#302A25", "accent_color": "#C56545",
                  "character_description": "round-headed original line character",
                  "visual_direction": "simple clear ink drawings", "motion_mode": "none",
                  "motion_every_n_scenes": 0},
        "scenes": [
            {"id": "scene-001", "narration": "A decision begins here.", "visual_prompt": "A figure considers a door.", "on_screen_text": "A decision"},
            {"id": "scene-002", "narration": "Then we take one step.", "visual_prompt": "The same figure steps forward.", "motion": False},
        ],
    }


class Providers:
    def __init__(self):
        self.audio_calls = []
        self.image_calls = []
        self.motion_calls = []
        self.motion_failure = False
        self.review_calls = []
        self.review_failures = 0

    async def review(self, image, scene, style, directory, log):
        self.review_calls.append(pipeline._sha(image))
        passed = len(self.review_calls) > self.review_failures
        report = {"passed": passed, "image_sha256": pipeline._sha(image),
                  "checks": [{"criterion": "object_structure", "verdict": "pass" if passed else "fail",
                              "evidence": "The tool has a disconnected working part",
                              "repair": "Reconnect the working part" if not passed else ""}]}
        path = directory / "test-review.json"
        path.write_text(json.dumps(report))
        return report, path

    async def audio(self, script_path, output_dir, voices, tts_model, log):
        self.audio_calls.append((Path(script_path).read_text(), voices, tts_model))
        target = Path(output_dir) / "generated.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        seconds = 0.171 if "begins" in Path(script_path).read_text() else 0.217
        with wave.open(str(target), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(i * 2 * math.pi * 440 / 24000))) for i in range(round(seconds * 24000))))
        return str(target)

    async def image(self, prompt, stage):
        self.image_calls.append(prompt)
        target = stage / "provider.png"
        image = Image.new("RGB", (512, 288), "#F4EBDD")
        draw = ImageDraw.Draw(image)
        draw.ellipse((210, 40, 270, 100), outline="#302A25", width=5)
        draw.line([(240, 100), (240, 190), (190, 240)], fill="#302A25", width=5)
        image.save(target)
        return target, "https://chatgpt.com/c/test"

    async def motion(self, prompt, first, last, stage, frame):
        self.motion_calls.append(prompt)
        assert first.is_file() and last is None
        if self.motion_failure:
            raise RuntimeError("Gemini unavailable: sign in")
        target = stage / "provider.mp4"
        await pipeline._command([
            "ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(first),
            "-frames:v", "2", "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target),
        ])
        return target, "https://gemini.google.com/app/test"


@pytest.fixture
def providers(monkeypatch):
    fixture = Providers()
    monkeypatch.setattr(pipeline, "generate_tts", fixture.audio)
    monkeypatch.setattr(pipeline, "_generate_still", fixture.image)
    monkeypatch.setattr(pipeline, "_generate_video", fixture.motion)
    monkeypatch.setattr(pipeline, "review_image", fixture.review)
    return fixture


def test_image_logic_repair_retains_candidates_and_reuses_verified_result(tmp_path, providers):
    async def run():
        providers.review_failures = 1
        scene, style = _project()["scenes"][0], _project()["style"]
        final, record = await pipeline._image(scene, style, tmp_path)
        assert len(providers.image_calls) == len(providers.review_calls) == 2
        assert "Reconnect the working part" in providers.image_calls[1]
        assert len(list(final.parent.glob("attempt-*/image.png"))) == 2
        assert (tmp_path / record["visual_review_path"]).is_file()
        assert await pipeline._image(scene, style, tmp_path) == (final, record)
        assert len(providers.image_calls) == len(providers.review_calls) == 2
    asyncio.run(run())


def test_repair_prompt_excludes_review_upload_references():
    feedback = [{"criterion": "narrative_meaning", "verdict": "fail",
                 "evidence": "image_01d0c2.jpg has anime figures",
                 "repair": "Redraw image_01d0c2.jpg with dot eyes and charcoal limbs"}]
    prompt = pipeline._image_repair_feedback(feedback)
    assert "image_01d0c2.jpg" not in prompt
    assert "anime figures" not in prompt
    assert "dot eyes and charcoal limbs" in prompt
    assert "No earlier image or attachment is available" in prompt


@pytest.mark.parametrize("final_passes", [True, False])
def test_terminal_missing_input_consumes_candidate_without_replaying_request(
    tmp_path, providers, monkeypatch, final_passes,
):
    scene, style = _project()["scenes"][0], _project()["style"]
    prompt = pipeline._image_prompt(scene, style)
    key = pipeline._hash({"version": pipeline.CONTRACT_VERSION, "prompt": prompt,
                          "review_version": pipeline.REVIEW_VERSION, "review": pipeline.review_prompt(scene, style)})
    stage = pipeline._stage(tmp_path, scene["id"], "image", key)
    (stage / "prompt.txt").write_text(prompt)
    feedback = [{"criterion": "narrative_meaning", "verdict": "fail",
                 "evidence": "image_01d0c2.jpg depicts anime figures",
                 "repair": "Use dot eyes and charcoal limbs"}]
    pending = prompt + pipeline._IMAGE_REPAIR_PREFIX + json.dumps(feedback)
    originals = {}
    for attempt, request in [(1, prompt), (2, pending)]:
        candidate_key = pipeline._hash({"prompt": request, "attempt": attempt})
        directory = stage / f"attempt-{attempt:02d}-{candidate_key[:12]}"
        directory.mkdir()
        (directory / "prompt.txt").write_text(request)
        if attempt == 1:
            image, _ = asyncio.run(providers.image(request, directory))
            pipeline._complete(directory, candidate_key, [image], tmp_path)
        else:
            (directory / "chatgpt-image-request.json").write_text(json.dumps({
                "prompt": "Generate an image of: " + request,
                "conversationUrl": "https://chatgpt.com/c/original-owned-turn",
            }))
        originals.update({p: p.read_bytes() for p in directory.iterdir() if p.is_file()})
    calls = []

    async def generate(request, directory):
        calls.append(request)
        if request == pending:
            raise RuntimeError("CHATGPT_IMAGE_INPUT_MISSING: Please upload the earlier image so I can recompose it")
        assert "image_01d0c2.jpg" not in request
        assert "dot eyes and charcoal limbs" in request
        return await providers.image(request, directory)

    monkeypatch.setattr(pipeline, "_generate_still", generate)
    providers.review_failures = 0 if final_passes else 100
    for _ in range(2):
        if final_passes:
            assert asyncio.run(pipeline._image(scene, style, tmp_path))[0].is_file()
        else:
            with pytest.raises(RuntimeError, match="after 3 image candidates"):
                asyncio.run(pipeline._image(scene, style, tmp_path))
    assert len(calls) == 2
    assert len(list(stage.glob("attempt-*"))) == 3
    assert all(p.read_bytes() == content for p, content in originals.items())
    receipt = next(stage.glob("attempt-02-*/input-failure.json"))
    state = receipt.parent / "chatgpt-image-request.json"
    state.write_text('{}')
    with pytest.raises(RuntimeError, match="binding differs"):
        pipeline._existing_image_candidates(stage, prompt, tmp_path)


def test_rejected_illustration_never_reaches_scene_render_and_retries_are_bounded(tmp_path, providers):
    async def run():
        providers.review_failures = 100
        for _ in range(2):
            with pytest.raises(RuntimeError, match="visual logic failed"):
                await pipeline.render_project(_project(), tmp_path, lambda _: None, lambda _: None)
        assert len(providers.image_calls) == pipeline.MAX_IMAGE_ATTEMPTS
        assert len(providers.audio_calls) == 1
        assert not list(tmp_path.rglob("clip.mp4"))
        assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "failed"
    asyncio.run(run())


def test_review_outage_preserves_generated_candidate_for_retry(tmp_path, providers, monkeypatch):
    async def unavailable(*args):
        raise RuntimeError("Visual logic review unavailable")

    async def run():
        scene, style = _project()["scenes"][0], _project()["style"]
        monkeypatch.setattr(pipeline, "review_image", unavailable)
        with pytest.raises(RuntimeError, match="review unavailable"):
            await pipeline._image(scene, style, tmp_path)
        assert len(providers.image_calls) == 1
        monkeypatch.setattr(pipeline, "review_image", providers.review)
        final, _ = await pipeline._image(scene, style, tmp_path)
        assert final.is_file() and len(providers.image_calls) == 1
    asyncio.run(run())


@pytest.mark.parametrize("passes_new_review", [False, True])
def test_review_revision_reuses_last_candidate_without_resetting_budget(
    tmp_path, providers, monkeypatch, passes_new_review,
):
    async def run():
        scene, style = _project()["scenes"][0], _project()["style"]
        providers.review_failures = 100
        with pytest.raises(RuntimeError, match="visual logic failed"):
            await pipeline._image(scene, style, tmp_path)
        originals = {p: p.read_bytes() for p in tmp_path.rglob("*")
                     if p.is_file() and p.name in {"image.png", "prompt.txt", "complete.json"}}
        prior_prompt = pipeline.review_prompt
        monkeypatch.setattr(pipeline, "review_prompt", lambda *args: prior_prompt(*args) + " Revised geometry.")
        providers.review_failures = 0 if passes_new_review else 100
        if passes_new_review:
            _, record = await pipeline._image(scene, style, tmp_path)
            assert "attempt-03-" in record["provenance"]["candidate"]
        else:
            for _ in range(2):
                with pytest.raises(RuntimeError, match="visual logic failed after 3"):
                    await pipeline._image(scene, style, tmp_path)
        assert len(providers.image_calls) == 3
        assert all(p.read_bytes() == data for p, data in originals.items())
    asyncio.run(run())


def test_review_revision_recovers_pending_candidate_in_its_original_directory(tmp_path, providers, monkeypatch):
    scene, style = _project()["scenes"][0], _project()["style"]
    submitted = []

    async def interrupted(request, directory):
        (directory / "chatgpt-image-request.json").write_text('{"state":"submitted"}')
        submitted.append((request, directory))
        raise RuntimeError("Submission result unknown")

    async def recover(request, directory):
        assert (request, directory) == submitted[0]
        assert (directory / "chatgpt-image-request.json").is_file()
        return await providers.image(request, directory)

    async def run():
        monkeypatch.setattr(pipeline, "_generate_still", interrupted)
        with pytest.raises(RuntimeError, match="result unknown"):
            await pipeline._image(scene, style, tmp_path)
        prior_prompt = pipeline.review_prompt
        monkeypatch.setattr(pipeline, "review_prompt", lambda *args: prior_prompt(*args) + " Revised geometry.")
        monkeypatch.setattr(pipeline, "_generate_still", recover)
        final, _ = await pipeline._image(scene, style, tmp_path)
        assert final.is_file() and len(submitted) == 1 and len(providers.image_calls) == 1
    asyncio.run(run())


def test_corrupt_candidate_is_not_silently_regenerated_after_review_revision(tmp_path, providers, monkeypatch):
    async def run():
        scene, style = _project()["scenes"][0], _project()["style"]
        providers.review_failures = 100
        with pytest.raises(RuntimeError, match="visual logic failed"):
            await pipeline._image(scene, style, tmp_path)
        next(tmp_path.rglob("attempt-03-*/image.png")).write_bytes(b"corrupted")
        prior_prompt = pipeline.review_prompt
        monkeypatch.setattr(pipeline, "review_prompt", lambda *args: prior_prompt(*args) + " Revised geometry.")
        with pytest.raises(RuntimeError, match="hash verification"):
            await pipeline._image(scene, style, tmp_path)
        assert len(providers.image_calls) == 3
    asyncio.run(run())


@pytest.mark.parametrize("reject_old", [False, True])
def test_legacy_artwork_requires_review_before_reuse(tmp_path, providers, reject_old):
    async def run():
        scene, style = _project()["scenes"][0], _project()["style"]
        prompt = pipeline._legacy_image_prompt(scene, style)
        key = pipeline._hash({"version": pipeline.CONTRACT_VERSION, "prompt": prompt})
        stage = pipeline._stage(tmp_path, scene["id"], "image", key)
        old, _ = await providers.image(prompt, stage)
        old.rename(stage / "image.png")
        old = stage / "image.png"
        before = pipeline._sha(old)
        pipeline._write(stage / "prompt.txt", prompt, tmp_path)
        pipeline._complete(stage, key, [old, stage / "prompt.txt"], tmp_path, conversation_url="old-url")
        providers.image_calls.clear()
        providers.review_failures = int(reject_old)
        final, record = await pipeline._image(scene, style, tmp_path)
        assert pipeline._sha(old) == before
        assert len(providers.image_calls) == int(reject_old)
        assert record["provenance"]["kind"] == ("generated" if reject_old else "reviewed_legacy")
        assert final.is_file()
    asyncio.run(run())


def test_submitted_legacy_turn_is_recovered_before_new_policy_review(tmp_path, providers, monkeypatch):
    scene, style = _project()["scenes"][0], _project()["style"]
    prompt = pipeline._legacy_image_prompt(scene, style)
    key = pipeline._hash({"version": pipeline.CONTRACT_VERSION, "prompt": prompt})
    stage = pipeline._stage(tmp_path, scene["id"], "image", key)
    pipeline._write(stage / "prompt.txt", prompt, tmp_path)
    (stage / "chatgpt-image-request.json").write_text('{"state":"submitted"}')

    async def recover(request, target):
        assert request == prompt and target == stage
        assert (target / "chatgpt-image-request.json").is_file()
        return await providers.image(request, target)

    monkeypatch.setattr(pipeline, "_generate_still", recover)
    final, record = asyncio.run(pipeline._image(scene, style, tmp_path))
    assert final.is_file() and record["provenance"]["kind"] == "reviewed_legacy"
    assert len(providers.image_calls) == len(providers.review_calls) == 1


@pytest.mark.parametrize("mode", ["none", "subtle"])
def test_real_ffmpeg_scene_alignment_cache_and_voice_invalidation(tmp_path, providers, mode):
    async def run():
        project = _project()
        project["style"]["motion_mode"] = mode
        updates = []
        result = await pipeline.render_project(project, tmp_path, updates.append, lambda _: None)
        assert result["duration_seconds"] == 13 / 30
        timeline = json.loads((tmp_path / result["timeline_path"]).read_text())
        first, second = timeline["scenes"]
        assert first["start_frame"] == 0 and first["end_frame"] == 6
        assert second["start_frame"] == 6 and second["end_frame"] == 13
        assert second["start_seconds"] == first["duration_seconds"]
        assert first["narration_duration_seconds"] == pytest.approx(0.171)
        assert first["duration_seconds"] - first["narration_duration_seconds"] < 1 / 30
        assert providers.audio_calls[0][0] == project["scenes"][0]["narration"]
        assert providers.audio_calls[0][1] == ["Carter"]
        assert all(project["style"]["character_description"] in prompt for prompt in providers.image_calls)
        assert all(project["style"]["visual_direction"] in prompt for prompt in providers.image_calls)
        assert "00:00:00,000 --> 00:00:00,171" in (tmp_path / "captions.srt").read_text()
        assert updates[-1]["progress"]["completed"] == updates[-1]["progress"]["total"]
        await pipeline._verify(tmp_path / result["video_path"], duration=13 / 30, video=True, audio=True)
        with wave.open(str(tmp_path / result["audio_path"]), "rb") as audio:
            assert audio.getnframes() == 20800
        reused = await pipeline.render_project(project, tmp_path, lambda _: None, lambda _: None)
        assert reused == result
        assert len(providers.audio_calls) == len(providers.image_calls) == 2
        project["voice"] = "Alice"
        changed = await pipeline.render_project(project, tmp_path, lambda _: None, lambda _: None)
        assert len(providers.audio_calls) == 4
        assert len(providers.image_calls) == 2
        assert changed["scene_artifacts"][0]["audio_path"] != first["audio_path"]
        assert (tmp_path / first["audio_path"]).is_file()
        assert changed["scene_artifacts"][0]["image_path"] == first["image_path"]
    asyncio.run(run())


@pytest.mark.parametrize("setting,value", [
    ("CHATTERBOX_TTS_URL", "http://localhost:18092"),
    ("CHATTERBOX_TTS_SOURCE_REVISION", "new-source"),
    ("CHATTERBOX_TTS_MODEL_REVISION", "new-weights"),
    ("CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT", "new-references-and-config"),
    ("CHATTERBOX_TTS_CHUNK_WORDS", 20),
    ("CHATTERBOX_TTS_MAX_CHARS", 200),
    ("TTS_RANDOM_SEED", 1000001),
])
def test_chatterbox_audio_cache_tracks_deployment_and_chunk_contract(tmp_path, providers, monkeypatch, setting, value):
    async def run():
        project = _project()
        project.update(tts_model="chatterbox-multilingual", voice="jane")
        scene = project["scenes"][0]
        first, _ = await pipeline._audio(scene, project, tmp_path, lambda _: None)
        cached, _ = await pipeline._audio(scene, project, tmp_path, lambda _: None)
        assert first == cached
        assert len(providers.audio_calls) == 1
        monkeypatch.setattr(pipeline.config, setting, value)
        changed, _ = await pipeline._audio(scene, project, tmp_path, lambda _: None)
        assert changed != first
        assert first.is_file() and changed.is_file()
        assert len(providers.audio_calls) == 2
    asyncio.run(run())


@pytest.mark.parametrize("module,setting,value", [
    (chatterbox, "VERIFIER_VERSION", chatterbox.VERIFIER_VERSION + 1),
    (tts, "POCKET_TTS_INTEGRITY_VERIFIER_VERSION", tts.POCKET_TTS_INTEGRITY_VERIFIER_VERSION + 1),
    (chatterbox, "SYNTHESIS_PARAMETERS", {**chatterbox.SYNTHESIS_PARAMETERS, "temperature": 0.7}),
])
def test_chatterbox_outer_audio_cache_never_skips_changed_verification_or_parameters(
    tmp_path, providers, monkeypatch, module, setting, value,
):
    async def run():
        project = _project()
        project.update(tts_model="chatterbox-multilingual", voice="jane")
        scene = project["scenes"][0]
        first, _ = await pipeline._audio(scene, project, tmp_path, lambda _: None)
        cached, _ = await pipeline._audio(scene, project, tmp_path, lambda _: None)
        assert cached == first and len(providers.audio_calls) == 1
        monkeypatch.setattr(module, setting, value)
        changed, _ = await pipeline._audio(scene, project, tmp_path, lambda _: None)
        assert changed != first and len(providers.audio_calls) == 2
        assert first.is_file() and changed.is_file()
    asyncio.run(run())


def test_gemini_failure_is_visible_and_retry_preserves_audio_artwork(tmp_path, providers):
    async def run():
        project = _project()
        project["scenes"] = project["scenes"][:1]
        project["style"]["motion_mode"] = "gemini"
        project["scenes"][0]["motion"] = True
        providers.motion_failure = True
        with pytest.raises(RuntimeError, match="Gemini unavailable"):
            await pipeline.render_project(project, tmp_path, lambda _: None, lambda _: None)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["status"] == "failed"
        assert "Gemini unavailable" in manifest["error"]
        assert len(providers.audio_calls) == len(providers.image_calls) == 1
        assert not list(tmp_path.rglob("video.mp4"))
        providers.motion_failure = False
        result = await pipeline.render_project(project, tmp_path, lambda _: None, lambda _: None)
        assert len(providers.audio_calls) == len(providers.image_calls) == 1
        assert len(providers.motion_calls) == 2
        assert result["scene_artifacts"][0]["motion_path"]
        assert result["duration_seconds"] == 0.2
        timeline = json.loads((tmp_path / "timeline.json").read_text())
        assert timeline["playback_policy"] == "play_once_fit_longer_or_hold_shorter"
        assert timeline["scenes"][0]["motion_playback"]["policy"] == "play_once_then_hold"
        assert timeline["scenes"][0]["motion_playback"]["playback_rate"] == 1.0
    asyncio.run(run())


def test_changed_narration_invalidates_only_its_scene(tmp_path, providers):
    async def run():
        project = _project()
        original = await pipeline.render_project(project, tmp_path, lambda _: None, lambda _: None)
        project["scenes"][0]["narration"] = "Another decision begins here."
        revised = await pipeline.render_project(project, tmp_path, lambda _: None, lambda _: None)
        assert len(providers.audio_calls) == len(providers.image_calls) == 3
        assert revised["scene_artifacts"][0]["audio_hash"] != original["scene_artifacts"][0]["audio_hash"]
        assert revised["scene_artifacts"][1] == original["scene_artifacts"][1]
    asyncio.run(run())


def test_watermark_repair_pixels_retry_and_cache_preserve_provider_media(tmp_path, providers, monkeypatch):
    async def marked_motion(_prompt, _first, _last, stage, _frame):
        providers.motion_calls.append(_prompt)
        target = stage / "provider.mp4"
        await pipeline._command([
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
            "color=c=0x405060:s=640x360:r=30:d=0.4",
            "-vf", "drawbox=x=568:y=288:w=24:h=24:color=white:t=fill",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target),
        ])
        return target, "https://gemini.google.com/app/test"

    monkeypatch.setattr(pipeline, "_generate_video", marked_motion)
    real_command = pipeline._command

    async def run():
        project = _project()
        project["scenes"] = project["scenes"][:1]
        project["scenes"][0]["motion"] = True
        project["captions_enabled"] = False
        project["style"].update(motion_mode="gemini", gemini_watermark_removal=False)
        original = await pipeline.render_project(project, tmp_path, lambda _: None, lambda _: None)
        before = original["scene_artifacts"][0]
        source = tmp_path / before["motion_path"]
        source_sha = pipeline._sha(source)

        async def fail_repair(args, **kwargs):
            if any("delogo=" in arg for arg in args):
                raise RuntimeError("delogo unavailable")
            return await real_command(args, **kwargs)

        project["style"]["gemini_watermark_removal"] = True
        monkeypatch.setattr(pipeline, "_command", fail_repair)
        with pytest.raises(RuntimeError, match="delogo unavailable"):
            await pipeline.render_project(project, tmp_path, lambda _: None, lambda _: None)
        assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "failed"
        assert len(list(tmp_path.glob("artifacts/scene-001/render/*/complete.json"))) == 1
        monkeypatch.setattr(pipeline, "_command", real_command)

        repaired = await pipeline.render_project(project, tmp_path, lambda _: None, lambda _: None)
        after = repaired["scene_artifacts"][0]
        assert after["render_hash"] != before["render_hash"]
        assert after["motion_path"] == before["motion_path"]
        assert after["audio_sha256"] == before["audio_sha256"]
        assert after["watermark_repair"]["source_dimensions"] == [640, 360]
        assert pipeline._sha(source) == source_sha
        assert repaired["duration_seconds"] == original["duration_seconds"]

        async def pixel(artifact, x, y):
            return await real_command([
                "ffmpeg", "-v", "error", "-i", str(tmp_path / artifact["clip_path"]),
                "-vf", f"crop=4:4:{x}:{y},scale=1:1", "-frames:v", "1",
                "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
            ])

        assert min(await pixel(before, 1740, 900)) > 240
        assert max(await pixel(after, 1740, 900)) < 110
        # Pixels outside the repair retain the scene; the title overlay is unchanged.
        assert max(abs(a - b) for a, b in zip(await pixel(before, 200, 300), await pixel(after, 200, 300))) < 5
        assert pipeline._sha((tmp_path / before["clip_path"]).with_name("overlay.png")) == pipeline._sha((tmp_path / after["clip_path"]).with_name("overlay.png"))
        assert await pipeline.render_project(project, tmp_path, lambda _: None, lambda _: None) == repaired
        project["style"]["gemini_watermark_region"] = {"x": 87, "y": 77, "width": 7, "height": 12}
        adjusted = await pipeline.render_project(project, tmp_path, lambda _: None, lambda _: None)
        assert adjusted["scene_artifacts"][0]["render_hash"] != after["render_hash"]
        project["style"]["gemini_watermark_removal"] = False
        assert await pipeline.render_project(project, tmp_path, lambda _: None, lambda _: None) == original
        assert len(providers.audio_calls) == len(providers.image_calls) == len(providers.motion_calls) == 1

    asyncio.run(run())


def test_dialogue_tts_labels_are_provider_metadata_only(tmp_path, providers):
    async def run():
        project = _project()
        project["tts_model"] = "vibevoice-1.5b"
        project["scenes"] = project["scenes"][:1]
        project["scenes"][0]["narration"] = "A decision begins here.\n\nThen one step."
        result = await pipeline.render_project(project, tmp_path, lambda _: None, lambda _: None)
        assert providers.audio_calls[0] == (
            "Speaker 1: A decision begins here.\nSpeaker 1: Then one step.", ["Carter"], "vibevoice-1.5b",
        )
        artifact = result["scene_artifacts"][0]
        assert artifact["narration"] == project["scenes"][0]["narration"]
        assert (tmp_path / artifact["audio_path"]).with_name("script.txt").read_text() == artifact["narration"]
        assert "Speaker 1:" not in (tmp_path / "captions.srt").read_text()
        await pipeline.render_project(project, tmp_path, lambda _: None, lambda _: None)
        assert len(providers.audio_calls) == 1
    asyncio.run(run())


def test_long_motion_fits_complete_visual_to_unchanged_audio(tmp_path, providers, monkeypatch):
    async def long_motion(_prompt, _first, _last, stage, _frame):
        target = stage / "provider.mp4"
        # A distinct ending proves the full action reaches the final scene.
        await pipeline._command([
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=512x288:r=30:d=0.8",
            "-f", "lavfi", "-i", "color=c=white:s=512x288:r=30:d=0.2",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target),
        ])
        return target, "https://gemini.google.com/app/test"
    monkeypatch.setattr(pipeline, "_generate_video", long_motion)

    async def run():
        project = _project()
        project["scenes"] = project["scenes"][:1]
        project["scenes"][0].update(motion=True, on_screen_text="")
        project["style"]["motion_mode"] = "gemini"
        project["captions_enabled"] = False
        result = await pipeline.render_project(project, tmp_path, lambda _: None, lambda _: None)
        artifact = result["scene_artifacts"][0]
        playback = artifact["motion_playback"]
        assert playback["source_duration_seconds"] == 1.0
        assert playback["policy"] == "fit_full_clip"
        assert playback["playback_rate"] == 5.0
        assert playback["narration_playback_rate"] == 1.0
        assert result["duration_seconds"] == 0.2
        assert artifact["narration_duration_seconds"] == pytest.approx(0.171)
        last_pixel = await pipeline._command([
            "ffmpeg", "-v", "error", "-i", str(tmp_path / artifact["clip_path"]),
            "-vf", "select=eq(n\\,5),scale=1:1", "-frames:v", "1", "-pix_fmt", "rgb24",
            "-f", "rawvideo", "-",
        ])
        assert len(last_pixel) == 3 and min(last_pixel) > 230
    asyncio.run(run())


def test_paths_reject_duplicate_ids_parent_traversal_and_symlinks(tmp_path):
    project = _project()
    project["scenes"][1]["id"] = project["scenes"][0]["id"]
    with pytest.raises(ValueError, match="unique"):
        pipeline._validate(project, tmp_path)
    project["scenes"][1]["id"] = "../escape"
    with pytest.raises(ValueError, match="safe"):
        pipeline._validate(project, tmp_path)
    with pytest.raises(ValueError, match="parent"):
        pipeline._safe(tmp_path / ".." / "escape", tmp_path)
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        pipeline._validate(_project(), tmp_path)


def test_provider_cannot_return_image_from_another_project(tmp_path, providers, monkeypatch):
    async def image(_prompt, _stage):
        return tmp_path.parent / "foreign.png", "https://chatgpt.com/c/test"
    monkeypatch.setattr(pipeline, "_generate_still", image)
    async def run():
        with pytest.raises(ValueError, match="inside"):
            await pipeline._image(_project()["scenes"][0], {}, tmp_path)
    asyncio.run(run())


def test_command_cancellation_reaps_process(tmp_path):
    async def run():
        pid_file = tmp_path / "child.pid"
        command = asyncio.create_task(pipeline._command([
            sys.executable, "-c", "import os,time,pathlib; pathlib.Path('child.pid').write_text(str(os.getpid())); time.sleep(60)",
        ], cwd=tmp_path, timeout=120))
        for _ in range(100):
            if pid_file.exists():
                break
            await asyncio.sleep(0.02)
        assert pid_file.exists()
        pid = int(pid_file.read_text())
        command.cancel()
        with pytest.raises(asyncio.CancelledError):
            await command
        import os
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    asyncio.run(run())
