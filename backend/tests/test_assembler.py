import re

from backend.pipeline import assembler, scene_kit


def board(scenes: list[dict], *, audio_duration: float = 60.0) -> dict:
    return {
        "title": "T",
        "audio_duration": audio_duration,
        "content_start": 5.0,
        "outro_start": 5.0 + audio_duration,
        "outro_duration": 5.0,
        "total_duration": 10.0 + audio_duration,
        "scenes": scenes,
    }


def scene(id_: str, start: float, duration: float, lines: list[dict]) -> dict:
    return {"id": id_, "start": start, "duration": duration, "lines": lines, "text": " "}


def test_spine_declares_the_runtime_contract_the_old_template_was_missing():
    data = board([scene("scene-01", 5.0, 55.0, [{"start": 5.0, "duration": 5.0, "text": "hi", "speaker": 1}])])
    html = assembler.build_spine(
        data,
        audio_src="audio/narration.wav",
        mounts=[{"id": "scene-01", "start": 5.0, "duration": 55.0}],
    )
    root = re.search(r'<div id="root"[^>]*>', html).group(0)
    assert 'data-start="0"' in root
    assert 'data-composition-id="root"' in root
    assert re.search(r'window\.__timelines\[\s*"root"\s*\]\s*=', html)
    assert "vendor/gsap.min.js" in html
    # Every timed element must be a .clip or the runtime never hides it.
    for tag in re.findall(r"<div [^>]*data-start=[^>]*>", html):
        if 'id="root"' in tag:
            continue
        assert 'class="clip' in tag, tag


def test_spine_has_no_remote_asset_references():
    data = board([scene("scene-01", 5.0, 55.0, [])])
    html = assembler.build_spine(
        data, audio_src="audio/a.wav", mounts=[{"id": "scene-01", "start": 5.0, "duration": 55.0}]
    )
    assert "https://" not in html and "http://" not in html


def test_spine_omits_brand_overlay_by_default():
    html = assembler.build_spine(board([]), audio_src="audio/a.wav", mounts=[])

    assert '<div class="brand">' not in html


def test_adjacent_scenes_alternate_tracks_so_boundaries_cannot_overlap():
    mounts = [
        {"id": "scene-01", "start": 0.0, "duration": 10.0},
        {"id": "scene-02", "start": 10.0, "duration": 10.0},
        {"id": "scene-03", "start": 20.0, "duration": 10.0},
    ]
    html = assembler.build_spine(board([]), audio_src="a.wav", mounts=mounts)
    tracks = re.findall(r'data-track-index="(\d)"></div>', html)
    assert tracks[:3] == [
        str(assembler.TRACK_SCENE_A),
        str(assembler.TRACK_SCENE_B),
        str(assembler.TRACK_SCENE_A),
    ]


def test_captions_are_clamped_so_they_never_overlap_on_their_track():
    # Silence-map rounding routinely produces a caption that ends a hundredth of
    # a second after the next one starts, which lint reports as an overlap.
    lines = [
        {"start": 0.0, "duration": 5.02, "text": "first", "speaker": 1},
        {"start": 5.0, "duration": 4.0, "text": "second", "speaker": 1},
    ]
    html = assembler.build_spine(
        board([scene("scene-01", 0.0, 9.0, lines)]),
        audio_src="a.wav",
        mounts=[{"id": "scene-01", "start": 0.0, "duration": 9.0}],
    )
    caps = re.findall(r'id="cap-\d+"[^>]*data-start="([\d.]+)" data-duration="([\d.]+)"', html)
    assert len(caps) == 2
    first_end = float(caps[0][0]) + float(caps[0][1])
    assert first_end <= float(caps[1][0])


def test_captions_are_escaped():
    lines = [{"start": 0.0, "duration": 2.0, "text": '<b>x</b> & "y"', "speaker": 1}]
    html = assembler.build_spine(
        board([scene("scene-01", 0.0, 2.0, lines)]),
        audio_src="a.wav",
        mounts=[{"id": "scene-01", "start": 0.0, "duration": 2.0}],
    )
    assert "<b>x</b>" not in html
    assert "&lt;b&gt;" in html


def test_captions_can_be_disabled():
    lines = [{"start": 0.0, "duration": 2.0, "text": "hidden caption", "speaker": 1}]
    html = assembler.build_spine(
        board([scene("scene-01", 0.0, 2.0, lines)]),
        audio_src="a.wav",
        mounts=[{"id": "scene-01", "start": 0.0, "duration": 2.0}],
        captions_enabled=False,
    )
    assert 'class="clip caption"' not in html
    assert "hidden caption" not in html


def test_long_captions_are_split_into_short_single_line_groups():
    text = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    html = assembler.build_spine(
        board([scene("scene-01", 0.0, 8.0, [{"start": 0.0, "duration": 8.0, "text": text}])]),
        audio_src="a.wav",
        mounts=[{"id": "scene-01", "start": 0.0, "duration": 8.0}],
    )
    captions = re.findall(r'<span class="caption-inner">([^<]+)</span>', html)
    assert len(captions) == 3
    assert all(len(caption.split()) <= assembler.CAPTION_MAX_WORDS for caption in captions)
    assert "white-space:nowrap" in html
    assert "text-overflow:ellipsis" in html


def test_unspaced_cjk_captions_are_split_to_the_character_limit():
    text = "这是一段很长的中文字幕用于验证字幕始终只占一行并且会按照可读长度自动拆分成多个连续的字幕时间片"
    chunks = assembler._caption_chunks(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= assembler.CAPTION_MAX_CJK_CHARS for chunk in chunks)


def test_audio_starts_immediately_with_the_first_content_scene():
    data = board([], audio_duration=100.0)
    data["content_start"] = 0.0
    html = assembler.build_spine(data, audio_src="audio/n.wav", mounts=[])
    audio = re.search(r"<audio[^>]*>", html).group(0)
    assert 'data-start="0.0"' in audio
    assert 'data-duration="100.0"' in audio
    assert 'src="audio/n.wav"' in audio


def test_write_scene_files_creates_one_file_per_plan(tmp_path):
    plans = [
        scene_kit.ScenePlan(id="scene-01", duration=5.0, headline="One"),
        scene_kit.ScenePlan(id="scene-02", duration=5.0, headline="Two"),
    ]
    written = assembler.write_scene_files(tmp_path, plans)
    assert [p.name for p in written] == ["scene-01.html", "scene-02.html"]
    assert 'data-composition-id="scene-02"' in written[1].read_text()


def test_theme_reaches_the_spine():
    html = assembler.build_spine(
        board([]), audio_src="a.wav", mounts=[], theme=scene_kit.THEMES["swiss"]
    )
    assert scene_kit.THEMES["swiss"].bg in html


def test_shanshui_spine_uses_paper_caption_and_muted_progress_colours():
    theme = scene_kit.THEMES["shanshui"]
    html = assembler.build_spine(board([]), audio_src="a.wav", mounts=[], theme=theme)

    assert theme.caption_bg in html
    assert "border:2px solid rgba(78,105,90,.18)" in html
    assert scene_kit.accent_hex("amber", theme) in html
    assert scene_kit.accent_hex("teal", theme) in html
    assert ".scene-mount { position:absolute; inset:0; z-index:1" in html
    assert "pointer-events:none; z-index:20" in html
