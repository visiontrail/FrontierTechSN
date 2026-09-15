from pathlib import Path

import pytest

from backend.pipeline import director, outros, scene_kit


def _install_fake_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library = tmp_path / "library"
    library.mkdir()
    monkeypatch.setattr(outros, "OUTRO_LIBRARY_DIR", library)
    monkeypatch.setattr(outros, "_verified_asset", lambda path, *_args: path)
    monkeypatch.setattr(outros, "_stage_background_hold", lambda _src, dest: dest.write_bytes(b"frame"))
    for preset in outros.OUTRO_PRESETS:
        (library / preset["filename"]).write_bytes(b"video")
    (library / outros.OUTRO_LOGO_FILENAME).write_bytes(b"logo")


def _storyboard() -> dict:
    return {
        "total_duration": 42.5,
        "outro_duration": 0.0,
        "outro_start": 42.5,
        "scene_count": 1,
        "scenes": [
            {
                "id": "scene-01",
                "index": 0,
                "start": 0.0,
                "duration": 42.5,
                "lines": [],
                "text": "Narrated content",
                "program_segment_kind": "closing",
            }
        ],
    }


def test_outro_catalog_has_three_presets_and_morning_default():
    presets = outros.list_outro_presets()

    assert [preset["id"] for preset in presets] == [
        "data-extraction",
        "morning-brief",
        "signal-shot",
    ]
    assert next(preset for preset in presets if preset["is_default"])["label"] == (
        "2. 晨间简报｜温暖、编辑感"
    )


def test_stage_outro_replaces_spoken_closing_without_extending_storyboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_library(tmp_path, monkeypatch)
    board = _storyboard()

    plan = outros.stage_outro(tmp_path / "task", board, "signal-shot")

    assert plan["archetype"] == "outro"
    assert plan["outro_style"] == "signal-shot"
    assert plan["id"] == "scene-01"
    assert plan["duration"] == 42.5
    assert board["outro_start"] == 0.0
    assert board["outro_duration"] == 42.5
    assert board["total_duration"] == 42.5
    assert len(board["scenes"]) == 1
    assert board["scenes"][-1]["scene_kind"] == "outro"
    assert (tmp_path / "task/assets/outro/background.mp4").read_bytes() == b"video"
    assert (tmp_path / "task/assets/outro/bytefront-logo.png").read_bytes() == b"logo"


def test_stage_outro_rejects_storyboard_without_timed_scenes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_library(tmp_path, monkeypatch)
    board = _storyboard()
    board["scenes"] = []

    with pytest.raises(ValueError, match="no timed scene"):
        outros.stage_outro(tmp_path / "task", board, "morning-brief")


def test_outro_scene_is_a_valid_editable_hyperframes_composition():
    plan = scene_kit.ScenePlan(
        id="scene-09",
        duration=6.0,
        archetype="outro",
        kicker="SEE YOU IN THE NEXT SHOT",
        headline="THANKS FOR WATCHING",
        body="Stay curious.",
        footage_src="assets/outro/background.mp4",
        footage_kind="video",
        outro_logo_src="assets/outro/bytefront-logo.png",
        outro_style="morning-brief",
    )

    html = scene_kit.render_scene(plan)

    assert director.validate_scene_html(
        html,
        "scene-09",
        plan={
            "archetype": "outro",
            "footage_src": "assets/outro/background.mp4",
            "outro_logo_src": "assets/outro/bytefront-logo.png",
        },
    ) == []
    assert "感谢观看" not in html
    assert 'data-outro-role="brand"' in html
    assert 'data-outro-brand-part="espresso"' in html
    assert "outro-badge" not in html
    assert "font:700 66px/1.2" in html
    assert 'data-outro-role="thanks"' in html
    assert html.count("data-outro-action=") == 3
    assert "muted playsinline" in html
    assert 'data-bookend-layer="background" style="z-index:0"' in html
    assert 'data-bookend-layer="overlay" style="z-index:2"' in html


def test_outro_preserves_spoken_subscription_and_next_morning(tmp_path, monkeypatch):
    _install_fake_library(tmp_path, monkeypatch)
    board = _storyboard()
    board["scenes"][-1]["text"] = "Subscribe, and we'll see you tomorrow morning."
    staged = outros.stage_outro(tmp_path / "task", board, "morning-brief")
    html = scene_kit.render_scene(scene_kit.ScenePlan.from_dict(
        staged, duration=staged["duration"], scene_id=staged["id"],
    ))
    assert 'data-outro-action="subscribe"' in html
    assert "See you tomorrow morning." in html
    assert html.count("data-outro-action=") == 4


def test_outro_agent_contract_rejects_removed_phrase_and_missing_actions():
    broken = '<div data-outro-role="brand" data-outro-role="thanks">感谢观看</div>'

    problems = outros.outro_overlay_problems(broken)

    assert "the removed Chinese closing phrase is still present" in problems
    assert "missing editable engagement-actions overlay" in problems
    assert "outro background stacking contract is missing" in problems


def _write_json(path, value):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_credits_include_selected_and_used_media_only(tmp_path):
    article = dict(title='A < B', source_name='News & Co', url='https://news.test/a',
                   published_at='2026-09-14T00:00:00Z', corroborating_sources=['News & Co', 'Second'])
    _write_json(tmp_path / 'research/dossier.json', dict(selected=[article, article],
                candidates=[dict(title='Unused', url='https://unused.test')]))
    _write_json(tmp_path / 'footage/manifest.json', dict(clips=[
        dict(local_path='footage/used.mp4', creator='Camera', title='Used', source_page_url='https://video.test/1'),
        dict(local_path='footage/unused.mp4', creator='Wrong', title='Unused'),
    ]))
    rows = outros.collect_credits(tmp_path, [dict(footage_sequence=[dict(src='footage/used.mp4')])])
    assert len(rows) == 2
    assert rows[0]['detail'] == '2026-09-14 · Second'
    markup = outros.credits_markup(rows)
    assert 'A &lt; B' in markup and 'News &amp; Co' in markup
    assert 'https://news.test/a' in markup and 'Unused' not in markup
    assert rows[1]['source'] == 'Camera'


def test_missing_credits_stays_empty_and_summary_urls_are_fallback(tmp_path):
    assert outros.collect_credits(tmp_path, []) == []
    _write_json(tmp_path / 'summary.json', dict(talking_points=[
        dict(headline='Report', source='Publisher', url='https://news.test/report'),
        dict(headline='Unsourced model summary'),
    ]))
    assert len(outros.collect_credits(tmp_path, [])) == 1


def test_corrupt_source_file_fails_instead_of_silently_dropping_credits(tmp_path):
    import json
    (tmp_path / 'summary.json').write_text('{broken')
    with pytest.raises(json.JSONDecodeError):
        outros.collect_credits(tmp_path, [])


def test_credit_tail_preserves_speech_and_is_idempotent(tmp_path, monkeypatch):
    _install_fake_library(tmp_path, monkeypatch)
    task = tmp_path / 'task'
    _write_json(task / 'research/dossier.json', dict(selected=[
        dict(title=f'A complete headline for news story {i}', source_name='Publisher', url=f'https://news.test/{i}')
        for i in range(20)
    ]))
    board = _storyboard()
    board['scenes'][0].update(start=100, duration=6, lines=[dict(start=100, duration=6, text='Goodbye')])
    board['audio_duration'] = 106
    staged = outros.stage_outro(task, board, 'morning-brief')
    assert 6 < staged['duration'] <= 29.9
    assert board['total_duration'] == pytest.approx(100 + staged['duration'])
    assert board['audio_duration'] == 106
    assert board['scenes'][0]['lines'][0]['duration'] == 6
    tail = board['credits_tail_duration']
    outros.stage_outro(task, board, 'morning-brief')
    assert board['credits_tail_duration'] == tail
    assert len(staged['outro_credits']) == 20
    assert (task / staged['outro_hold_src']).read_bytes() == b'frame'


@pytest.mark.parametrize('orientation', ['landscape', 'portrait'])
def test_source_roll_caps_duration_and_replaces_old_long_tail(tmp_path, monkeypatch, orientation):
    _install_fake_library(tmp_path, monkeypatch)
    task = tmp_path / 'task'
    _write_json(task / 'research/dossier.json', dict(selected=[
        dict(title=f'完整新闻标题 {i} ' * 8, source_name='Publisher', url=f'https://news.test/{i}')
        for i in range(40)
    ]))
    board = _storyboard()
    board['scenes'][0].update(start=100, duration=80, spoken_closing_duration=6)
    staged = outros.stage_outro(task, board, 'morning-brief', video_orientation=orientation)
    assert staged['duration'] == 29.9
    assert len(staged['outro_credits']) == 40
    assert board['total_duration'] == 129.9
    assert board['credits_tail_duration'] == 23.9


def test_overlong_closing_is_rejected_without_cutting_speech(tmp_path, monkeypatch):
    _install_fake_library(tmp_path, monkeypatch)
    task = tmp_path / 'task'
    _write_json(task / 'summary.json', dict(talking_points=[dict(headline='News', url='https://news.test')]))
    board = _storyboard()
    with pytest.raises(ValueError, match='shorten the closing narration'):
        outros.stage_outro(task, board, 'morning-brief')
    assert board['scenes'][0]['duration'] == 42.5


def test_source_roll_is_locked_against_director_removal():
    from backend.pipeline import composer
    rows = [dict(kind='NEWS', title='Report', source='Publisher', url='https://news.test/a', detail='')]
    data = dict(id='scene-09', archetype='outro', outro_credits=rows,
                footage_src='assets/outro/background.mp4', outro_logo_src='assets/outro/logo.png')
    plan = scene_kit.ScenePlan.from_dict(data, duration=10, scene_id='scene-09')
    rendered = scene_kit.render_scene(plan)
    assert director.validate_scene_html(rendered, plan.id, plan=data) == []
    assert director.validate_scene_html(rendered.replace('Report', 'Changed'), plan.id, plan=data)
    assert composer._director_scene_plans([data], quality_retry=False) == []
    assert composer._director_scene_plans([{**data, 'visual_review_feedback': {'issues': ['x']}}], quality_retry=True) == []


def test_spine_plays_padded_audio_but_keeps_original_caption_end():
    from backend.pipeline import assembler
    board = _storyboard()
    board.update(content_start=0, audio_duration=6, program_audio_duration=20, total_duration=20)
    board['scenes'][0]['lines'] = [dict(start=0, duration=6, text='Goodbye')]
    rendered = assembler.build_spine(board, audio_src='padded.wav', mounts=[])
    assert 'data-duration="20.0"' in rendered
    assert 'data-duration="6.0" data-track-index="4"' in rendered
