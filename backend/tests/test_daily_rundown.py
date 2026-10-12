import copy
import hashlib
import json
import wave
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend import config, publishing, social_copy
from backend.daily_news import freshness, research, review, rundown, scriptwriter
from backend.models import TaskConfig, TaskResponse
from backend.pipeline import composer, orchestrator, storyboard, title, visual_plan

EDITION = date(2026, 10, 12)
LEAD = "Reuters reports Atlas released a smaller chip. It cuts power use in the company's tests. Independent results remain unavailable."
SECOND = "BBC reports Boreal launched a satellite. It will test a new antenna."
THIRD = "Nature reports Cedar demonstrated a battery. The prototype lasted longer in laboratory tests."
CLOSING = "Thanks for watching ByteFront Espresso."


@pytest.fixture
def edition(tmp_path):
    stories = [LEAD, SECOND, THIRD]
    articles = [research.NewsArticle(
        id=f"story-{i}", source_id=source.lower(), source_name=source, language="en",
        title=text.split(".")[0], url=f"https://example.org/{i}",
        published_at="2026-10-12T00:00:00+00:00", summary=text, evidence_text=text,
    ) for i, (source, text) in enumerate(zip(["Reuters", "BBC", "Nature"], stories), 1)]
    dossier = research.ResearchDossier(EDITION.isoformat(), "2026-10-12T01:00:00+00:00",
                                       36, articles, articles, [], lead_story_id=articles[0].id)
    reviewed = "\n".join([scriptwriter.morning_opening(EDITION), *stories, CLOSING])
    script = rundown.prepare_delivery(reviewed, dossier, tmp_path, "en")
    (tmp_path / "script.txt").write_text(script)
    rundown.write_json(tmp_path / "research/dossier.json", dossier.as_dict())
    return SimpleNamespace(root=tmp_path, dossier=dossier, reviewed=reviewed, script=script,
                           data=rundown.read_rundown(tmp_path, script))


def test_delivery_opens_with_event_and_significance_without_repeating_or_reordering(edition):
    e = edition
    assert e.script.splitlines() == [
        "Reuters reports Atlas released a smaller chip. It cuts power use in the company's tests.",
        "Independent results remain unavailable.", SECOND, THIRD, CLOSING,
    ]
    assert "Good morning" not in e.script
    assert e.data["story_order"] == [a.id for a in e.dossier.selected]
    assert [s["narration"] for s in e.data["stories"]] == [LEAD, SECOND, THIRD]
    assert e.data["reviewed_script_sha256"] == rundown.script_hash(e.reviewed)


@pytest.mark.parametrize("text", [
    "Reuters reports the news.",
    "Good morning everyone. Reuters reports Atlas launched a chip.",
    "Reuters reports " + "detail " * 34 + ". The result matters.",
])
def test_rejects_incomplete_generic_or_overlong_cold_open(text):
    assert rundown.opening_failures(text, "en")


def test_sentence_split_preserves_initials_decimals_and_chinese():
    text = "Dr. Lee reports a U.S. trial at 2.5 watts. It may reduce power use. Details follow."
    assert rundown.split_lead(text) == (text.removesuffix(" Details follow."), "Details follow.")
    assert rundown.split_lead("研究团队发布新芯片。这能降低测试中的能耗。独立结果尚未公布。") == (
        "研究团队发布新芯片。这能降低测试中的能耗。", "独立结果尚未公布。",
    )


def test_changed_script_mapping_or_reviewed_evidence_cannot_reuse_rundown(edition):
    e = edition
    with pytest.raises(ValueError, match="stale"):
        rundown.read_rundown(e.root, e.script.replace("Atlas", "Changed"))
    data = copy.deepcopy(e.data)
    data["stories"][1]["narration"] = THIRD
    rundown.write_json(e.root / "rundown.json", data)
    with pytest.raises(ValueError, match="changed or duplicated"):
        rundown.read_rundown(e.root, e.script)
    rundown.write_json(e.root / "rundown.json", e.data)
    (e.root / "script.reviewed.txt").write_text(e.reviewed.replace("Atlas", "Changed"))
    with pytest.raises(ValueError, match="reviewed story"):
        rundown.read_rundown(e.root, e.script)


def test_missing_new_rundown_fails_but_legacy_scripts_are_unchanged(edition):
    e = edition
    (e.root / "rundown.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        rundown.read_rundown(e.root, e.script)
    e.dossier.lead_story_id = ""
    assert rundown.prepare_delivery(e.reviewed, e.dossier, e.root, "en") == e.reviewed


def test_review_checks_fixed_slots_instead_of_silently_matching_swapped_stories(edition):
    e = edition
    swapped = e.reviewed.replace(LEAD, "PLACEHOLDER").replace(SECOND, LEAD).replace("PLACEHOLDER", SECOND)
    assert review._matched_script_claims(swapped, e.dossier)[1] == SECOND
    prompt = review._batch_review_prompt(swapped, e.dossier, EDITION, [1, 2])
    assert "core event AND supported significance" in prompt
    assert "claim 1.0" in prompt


def test_edit_reconstructs_review_slots_and_keeps_the_users_changes(edition):
    e = edition
    assert rundown.review_input_for_edit(e.root, e.script) is None
    edited = e.script.replace("smaller chip", "more efficient chip")
    draft = rundown.review_input_for_edit(e.root, edited)
    assert draft == e.reviewed.replace("smaller chip", "more efficient chip")
    with pytest.raises(ValueError, match="one line"):
        rundown.review_input_for_edit(e.root, edited + "\nAnother item.")


@pytest.mark.asyncio
async def test_script_repair_shortens_only_the_lead_and_preserves_other_stories(edition):
    e = edition
    long_lead = "Reuters reports Atlas released " + "smaller " * 35 + "chips. Power use fell in tests."
    raw = "\n".join([long_lead, SECOND, THIRD])
    with (patch.object(scriptwriter, "_resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))),
          patch.object(scriptwriter, "_chat", AsyncMock(side_effect=[raw, json.dumps({"1": LEAD})])) as chat):
        script = await scriptwriter.generate_daily_script(
            e.dossier, EDITION, target_duration_minutes=None, language="en",
            closing_remarks=CLOSING, ai_endpoint=None, ai_model=None, provider_id=None,
        )
    assert script.splitlines()[1:-1] == [LEAD, SECOND, THIRD]
    assert "Repair ONLY story numbers [1]" in chat.await_args.args[0]


def make_task(e):
    return TaskResponse(id="test", source_type="news_daily", status="reviewing",
                        created_at="2026-10-12T00:00:00Z", updated_at="2026-10-12T00:00:00Z",
                        source_url=json.dumps({"edition_date": EDITION.isoformat()}),
                        config=TaskConfig(thumbnail_enabled=False, footage_enabled=False, auto_render=False,
                                          target_duration_minutes=None),
                        output_dir=str(e.root), script_path=str(e.root / "script.txt"))


@pytest.mark.asyncio
async def test_full_pipeline_freezes_rundown_before_title_and_tts(edition):
    e = edition
    task = make_task(e)
    async def generate_title(**kwargs):
        assert kwargs["daily_rundown"]["lead_story_id"] == "story-1"
        assert kwargs["script"].startswith(e.data["opening"])
        return SimpleNamespace(title="Atlas reports lower-power chip | Daily Tech Brief · Oct 12")
    async def generate_audio(script_path, *args, **kwargs):
        assert Path(script_path).read_text() == e.script
        return str(e.root / "audio.wav")
    with (patch.object(orchestrator, "update_task", AsyncMock()),
          patch.object(orchestrator, "run_research", AsyncMock(return_value=e.dossier)),
          patch.object(orchestrator, "generate_daily_script", AsyncMock(return_value=e.reviewed)),
          patch.object(orchestrator, "review_daily_script", AsyncMock(return_value=SimpleNamespace(script=e.reviewed))),
          patch.object(orchestrator, "generate_title", generate_title),
          patch.object(orchestrator, "generate_tts", generate_audio)):
        await orchestrator.run_pipeline(task)
    assert task.generated_title.startswith("Atlas")


@pytest.mark.asyncio
async def test_review_resume_preserves_an_edit_instead_of_recovering_old_candidate(edition):
    e = edition
    task = make_task(e)
    (e.root / "script.draft.txt").write_text(e.reviewed)
    (e.root / "review").mkdir()
    candidate = e.root / "review/candidate-duration-cycle-1.txt"
    candidate.write_text(e.reviewed)
    edited = e.reviewed.replace("smaller chip", "more efficient chip")
    with (patch.object(orchestrator, "update_task", AsyncMock()),
          patch.object(freshness, "refresh_review_history", AsyncMock()),
          patch.object(orchestrator, "review_daily_script", AsyncMock(return_value=SimpleNamespace(script=edited))) as check,
          patch.object(orchestrator, "run_regenerate", AsyncMock()) as regenerate):
        await orchestrator.run_daily_review_resume(task, script_override=edited)
    assert check.await_args.args[0] == edited
    assert check.await_args.kwargs["recover_saved_candidate"] is False
    assert not candidate.exists()
    assert list((e.root / "review").glob("candidate-before-edit-*.txt"))
    assert "more efficient chip" in Path(task.script_path).read_text()
    regenerate.assert_awaited_once()


def timeline(e, end=15.0):
    cursor = 0.0
    rows = []
    for i, segment in enumerate(e.data["segments"]):
        duration = end if i == 0 else 10.0
        rows.append({"text": segment["text"], "kind": "opening" if i == 0 else "news",
                     "program_start": cursor, "program_end": cursor + duration})
        cursor += duration
    return {"passed": True, "segments": rows, "paced_duration_seconds": cursor}


@pytest.mark.parametrize("end", [20.001, float("nan"), float("inf")])
def test_measured_opening_overrun_blocks_even_if_word_budget_passed(edition, end):
    with pytest.raises(ValueError, match="by 20s"):
        rundown.bind_timeline(edition.root, edition.data, timeline(edition, end), audio_sha256="a")


def test_chapters_and_visual_scenes_share_audio_order_and_only_publish_after_promotion(edition):
    e = edition
    audio = e.root / "audio.wav"
    audio.write_bytes(b"test audio")
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    pacing = timeline(e)
    rundown.bind_timeline(e.root, e.data, pacing, audio_sha256=digest)
    board = storyboard.build_program_storyboard(pacing_report=pacing, title="Atlas", alignment={"passed": True})
    board["daily_rundown"] = {"lead_story_id": "story-1", "story_order": e.data["story_order"]}
    assert [s.get("story_id") for s in board["scenes"]] == ["story-1", "story-1", "story-2", "story-3", None]
    assert board["scenes"][0]["program_segment_kind"] == "news"
    assert board["scenes"][0]["start"] == 0
    payload = json.loads(visual_plan._batch_prompt_payload(board, board["scenes"]))
    assert "No brand intro" in payload["opening_instruction"]
    assert rundown.read_chapters(e.root, e.data, str(audio)) == []
    video_hash = "f" * 64
    video_path = str(e.root / f"video-{video_hash[:16]}.mp4")
    Path(video_path).write_bytes(b"video")
    rundown.promote_chapters(e.root, video_hash)
    chapters = rundown.read_chapters(e.root, e.data, str(audio), video_path)
    assert [row["start_seconds"] for row in chapters] == [0, 25, 35]
    assert rundown.chapter_text(e.data, chapters, "Atlas releases chip | Daily Tech Brief · Oct 12").startswith("00:00 Atlas releases chip")
    previous = e.root / "video-previous.mp4"
    previous.write_bytes(b"prior video")
    with pytest.raises(ValueError, match="stale"):
        rundown.read_chapters(e.root, e.data, str(audio), str(previous))


@pytest.mark.asyncio
async def test_title_can_only_use_the_locked_lead_and_preserves_edition_date(edition):
    e = edition
    with (patch.object(title, "_resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))),
          patch.object(title, "_chat", AsyncMock(side_effect=["x" * 80, "Atlas reports lower-power chip"])) as chat):
        artifact = await title.generate_title(task_id="test", task_dir=e.root, source_title="ignored",
                                              summary=None, script=e.script, daily_rundown=e.data)
    assert chat.await_count == 2
    payload = json.loads(chat.await_args.args[1])
    assert payload["final_audio_script"] == LEAD
    assert "Boreal" not in payload["final_audio_script"]
    assert artifact.title == "Atlas reports lower-power chip | Daily Tech Brief · Oct 12"


def daily_copy_input(e):
    return {"final_narration": e.script, "source_references": e.data["stories"],
            "daily_rundown": e.data, "publication_title": "Atlas reports lower-power chip | Daily Tech Brief · Oct 12"}


def copy_draft(inputs):
    return {"youtube_title": inputs["publication_title"], "youtube_story_notes": [
        {"story_id": "story-1", "text": "Atlas reports lower chip power use in company tests."},
        {"story_id": "story-2", "text": "Boreal launched a satellite to test an antenna."},
        {"story_id": "story-3", "text": "Cedar demonstrated a laboratory battery prototype."},
    ], "x_paragraphs": ["Atlas reports lower chip power use in company tests. Independent results remain unavailable."]}


@pytest.mark.parametrize("change", ["title", "swap", "omit", "duplicate", "timestamp"])
def test_upload_copy_rejects_new_lead_missing_swapped_or_invented_chapters(edition, change):
    inputs = daily_copy_input(edition)
    draft = copy_draft(inputs)
    if change == "title":
        draft["youtube_title"] = "Boreal launches satellite"
    elif change == "swap":
        draft["youtube_story_notes"].reverse()
    elif change == "omit":
        draft["youtube_story_notes"].pop()
    elif change == "duplicate":
        draft["youtube_story_notes"][1] = draft["youtube_story_notes"][0]
    else:
        draft["youtube_story_notes"][0]["text"] = "00:00 Invented chapter"
    with pytest.raises(ValueError):
        social_copy._validate(json.dumps(draft), inputs)


def test_show_notes_and_auto_publication_both_lead_with_same_event(edition):
    e = edition
    inputs = daily_copy_input(e)
    result = social_copy._validate(json.dumps(copy_draft(inputs)), inputs)
    notes = result["youtube_show_notes"]
    assert notes.index("Atlas") < notes.index("Boreal") < notes.index("Cedar")
    assert notes.index("https://example.org/1") < notes.index("https://example.org/2")
    task = TaskResponse(id="test", source_type="news_daily", status="complete",
                        created_at="2026-10-12T00:00:00Z", updated_at="2026-10-12T00:00:00Z",
                        config=TaskConfig(), output_dir=str(e.root), script_path=str(e.root / "script.txt"),
                        generated_title=inputs["publication_title"])
    description = publishing._description(task)
    assert description.startswith(e.data["opening"])
    assert description.index("Atlas") < description.index("Boreal") < description.index("Cedar")
    assert "00:00" not in description  # no invented timing before a final video


@pytest.mark.asyncio
async def test_completion_accepts_only_the_exact_reviewed_split(edition, monkeypatch):
    e = edition
    raw = e.dossier.as_dict()
    raw["freshness_policy_version"] = 1
    rundown.write_json(e.root / "research/dossier.json", raw)
    rundown.write_json(e.root / "review/fact_check_report.json", {
        "passed": True, "freshness_review_key": freshness.review_binding(raw),
        "final_contract": {"script_sha256": rundown.script_hash(e.reviewed)},
    })
    task = SimpleNamespace(id="test", source_type="news_daily", output_dir=str(e.root), script_path=None)
    monkeypatch.setattr(config, "DB_PATH", e.root / "tasks.db")
    monkeypatch.setattr(freshness, "load_history", AsyncMock(return_value=[]))
    async with freshness.completion_guard(task):
        pass
    (e.root / "script.txt").write_text(e.script.replace("Atlas", "Not approved"))
    with pytest.raises(RuntimeError, match="freshness review receipt"):
        async with freshness.completion_guard(task):
            pass


@pytest.mark.asyncio
async def test_composer_enforces_cold_open_even_when_music_pacing_is_disabled(edition, monkeypatch):
    e = edition
    audio = e.root / "audio.wav"
    durations = [2, 1, 2, 2, 1]
    with wave.open(str(audio), "wb") as handle:
        handle.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        handle.writeframes(b"\x00\x00" * (sum(durations) * 8000))
    cursor = 0
    manifest_segments = []
    for segment, duration in zip(e.data["segments"], durations):
        manifest_segments.append({"text": segment["text"], "start": cursor, "duration": duration})
        cursor += duration
    monkeypatch.setattr(composer, "_narration_manifest_failures", lambda *args: [])
    monkeypatch.setattr(composer, "_manifest_program_segments", lambda *args: manifest_segments)
    monkeypatch.setattr(composer, "_resume_unavailable_visual_review", AsyncMock(return_value=None))
    monkeypatch.setattr(composer.av_sync, "ensure_word_transcript", AsyncMock(return_value=([], {})))
    monkeypatch.setattr(composer, "_detect_silence_boundaries", lambda *args: [])
    captured = {}
    class Planned(Exception):
        pass
    async def capture(_directory, board, **_kwargs):
        captured.update(board)
        raise Planned()
    monkeypatch.setattr(composer, "_load_or_plan_scene_visuals", capture)
    with pytest.raises(Planned):
        await composer.compose_video(script_path=str(e.root / "script.txt"), audio_path=str(audio),
                                     output_dir=str(e.root), title="Atlas", tts_model="orpheus-en",
                                     program_music_pacing_enabled=False)
    assert captured["daily_rundown"]["lead_story_id"] == "story-1"
    assert captured["scenes"][0]["text"] == e.data["opening"]
    assert captured["scenes"][0]["program_segment_kind"] == "news"
    assert captured["scenes"][0]["start"] == 0
    assert captured["scenes"][1]["start"] == 2  # no inserted greeting/music gap
