from pathlib import Path

import pytest

from backend.pipeline import storyboard as sb


def write_script(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "script.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def timed_line(start: float, duration: float, text: str) -> dict:
    return {
        "start": start,
        "duration": duration,
        "speaker": 1,
        "text": text,
        "word_count": max(1, len(text.split())),
    }


def test_parse_script_lines_reads_bare_monologue_lines(tmp_path):
    path = write_script(tmp_path, ["First line here.", "", "Second line here."])
    lines = sb.parse_script_lines(path, is_monologue=True)
    assert [line["text"] for line in lines] == ["First line here.", "Second line here."]
    assert {line["speaker"] for line in lines} == {1}


def test_parse_script_lines_reads_speaker_labels(tmp_path):
    path = write_script(tmp_path, ["Speaker 1: Hello.", "Speaker 2: Hi back."])
    lines = sb.parse_script_lines(path)
    assert [line["speaker"] for line in lines] == [1, 2]
    assert [line["text"] for line in lines] == ["Hello.", "Hi back."]


def test_parse_script_lines_splits_multi_sentence_paragraphs(tmp_path):
    path = write_script(tmp_path, ["First idea. Second idea! Third idea?"])
    lines = sb.parse_script_lines(path, is_monologue=True)
    assert [line["text"] for line in lines] == ["First idea.", "Second idea!", "Third idea?"]
    assert {line["speaker"] for line in lines} == {1}


def test_line_timing_falls_back_to_word_count_and_fills_the_audio(tmp_path):
    lines = sb.parse_script_lines(write_script(tmp_path, ["one two", "three four five six"]))
    timed = sb.assign_line_timing(lines, audio_duration=60.0, silence_boundaries=[])
    assert timed[0]["start"] == 0.0
    # Six words total, two in the first line: a third of the runtime.
    assert timed[0]["duration"] == pytest.approx(20.0, abs=0.01)
    assert timed[1]["start"] == pytest.approx(20.0, abs=0.01)
    assert timed[-1]["start"] + timed[-1]["duration"] == pytest.approx(60.0, abs=0.01)


def test_line_timing_prefers_the_silence_map_when_it_is_dense_enough(tmp_path):
    lines = sb.parse_script_lines(write_script(tmp_path, ["a b", "c d", "e f"]))
    timed = sb.assign_line_timing(lines, audio_duration=30.0, silence_boundaries=[8.0, 21.0])
    assert [line["start"] for line in timed] == [0.0, 8.0, 21.0]
    assert timed[-1]["duration"] == pytest.approx(9.0)


def test_silence_map_selects_marks_near_expected_positions_instead_of_first_marks(tmp_path):
    lines = sb.parse_script_lines(write_script(tmp_path, ["one two", "three four", "five six"]))
    timed = sb.assign_line_timing(
        lines,
        audio_duration=30.0,
        silence_boundaries=[1.0, 2.0, 9.5, 20.5, 28.0],
    )
    assert [line["start"] for line in timed] == pytest.approx([0.0, 9.5, 20.5])
    assert timed[-1]["duration"] == pytest.approx(9.5)


def test_transcript_forced_alignment_tracks_actual_delivery_not_word_proportions(tmp_path):
    lines = sb.parse_script_lines(
        write_script(tmp_path, ["alpha beta gamma.", "delta epsilon.", "zeta eta theta."]),
        is_monologue=True,
    )
    transcript = [
        {"text": "alpha", "start": 0.2, "end": 0.6},
        {"text": "beta", "start": 0.7, "end": 1.1},
        {"text": "gamma", "start": 1.2, "end": 1.7},
        # A long rhetorical pause makes word-proportional timing wrong.
        {"text": "delta", "start": 5.0, "end": 5.4},
        {"text": "epsilon", "start": 5.5, "end": 6.0},
        {"text": "zeta", "start": 7.0, "end": 7.4},
        {"text": "eta", "start": 7.5, "end": 7.8},
        {"text": "theta", "start": 7.9, "end": 8.4},
    ]

    timed, report = sb.align_lines_to_transcript(lines, transcript, audio_duration=9.0)

    assert report["passed"] is True
    assert report["word_coverage"] == 1.0
    assert timed[1]["start"] == pytest.approx(3.325, abs=0.01)
    assert timed[2]["start"] == pytest.approx(6.475, abs=0.01)
    assert timed[-1]["start"] + timed[-1]["duration"] == pytest.approx(9.0)


def test_transcript_alignment_fails_closed_when_words_do_not_match(tmp_path):
    lines = sb.parse_script_lines(write_script(tmp_path, ["alpha beta.", "gamma delta."]))
    transcript = [
        {"text": f"unrelated{i}", "start": i * 0.5, "end": i * 0.5 + 0.4}
        for i in range(12)
    ]

    _, report = sb.align_lines_to_transcript(lines, transcript, audio_duration=6.0)

    assert report["passed"] is False
    assert report["word_coverage"] == 0.0
    assert report["failure_reasons"]


def test_transcript_alignment_blocks_a_missing_final_sentence(tmp_path):
    lines = sb.parse_script_lines(
        write_script(
            tmp_path,
            [
                "A long complete opening with several clearly spoken words.",
                "Thanks for watching and see you tomorrow.",
            ],
        ),
        is_monologue=True,
    )
    transcript = [
        {"text": word, "start": index * 0.4, "end": index * 0.4 + 0.3}
        for index, word in enumerate(
            "A long complete opening with several clearly spoken words".split()
        )
    ]

    _, report = sb.align_lines_to_transcript(lines, transcript, audio_duration=8.0)

    assert report["passed"] is False
    assert report["line_coverage"] == 0.5
    assert any("100.0%" in reason for reason in report["failure_reasons"])


def test_scenes_and_audio_start_immediately_with_no_title_card_gap(tmp_path):
    path = write_script(tmp_path, [f"Sentence number {i} with several words in it." for i in range(30)])
    board = sb.build_storyboard(
        script_path=path, audio_duration=300.0, title="T", is_monologue=True
    )
    scenes = board["scenes"]
    assert scenes[0]["start"] == 0.0
    assert board["content_start"] == 0.0
    assert board["title_duration"] == 0.0
    for earlier, later in zip(scenes, scenes[1:]):
        assert earlier["start"] + earlier["duration"] == pytest.approx(later["start"], abs=0.05)
    last = scenes[-1]
    assert last["start"] + last["duration"] == pytest.approx(board["outro_start"], abs=0.05)
    assert board["total_duration"] == pytest.approx(
        300.0 + sb.OUTRO_DURATION, abs=0.05
    )


def test_program_storyboard_preserves_every_physical_segment_and_music_gap():
    report = {
        "passed": True,
        "paced_duration_seconds": 14.0,
        "segments": [
            {"kind": "opening", "text": "Good morning.", "program_start": 2.0, "program_end": 5.0},
            {"kind": "news", "text": "Reuters says story one.", "program_start": 8.0, "program_end": 10.0},
            {"kind": "closing", "text": "Thanks.", "program_start": 11.5, "program_end": 14.0},
        ],
    }

    board = sb.build_program_storyboard(
        pacing_report=report,
        title="Daily",
        alignment={"passed": True},
    )

    assert board["program_timeline"] is True
    assert board["scene_count"] == 3
    assert [scene["start"] for scene in board["scenes"]] == [0.0, 8.0, 11.5]
    assert [scene["duration"] for scene in board["scenes"]] == [8.0, 3.5, 2.5]
    assert board["scenes"][0]["lines"][0]["start"] == 2.0
    assert board["total_duration"] == 19.0


def test_short_trailing_scene_is_folded_into_its_predecessor(tmp_path):
    # A long line followed by a very short one would otherwise leave a stub scene.
    path = write_script(tmp_path, ["word " * 200, "ok"])
    board = sb.build_storyboard(
        script_path=path, audio_duration=120.0, title="T", is_monologue=True
    )
    assert all(scene["duration"] >= sb.SCENE_MIN_SECONDS for scene in board["scenes"][1:])


def test_different_named_report_sources_force_the_real_scene_two_boundary():
    lines = [
        timed_line(
            19.19,
            7.47,
            "DeepTech China reports that researchers at MIT are exploring whether living "
            "bacteria can be used to build computing systems analogous to circuit boards.",
        ),
        timed_line(
            26.66,
            13.11,
            "QbitAI reports that Perfect World's 2026 semiannual report shows first-half "
            "revenue of 2.751 billion yuan and a net loss of 118 million yuan.",
        ),
    ]

    scenes = sb.group_lines_into_scenes(lines)

    assert [(scene["start"], scene["duration"]) for scene in scenes] == [
        (19.19, 7.47),
        (26.66, 13.11),
    ]
    assert scenes[1]["semantic_boundary_before"] == {
        "kind": "citation_source_change",
        "from": "DeepTech China",
        "to": "QbitAI",
    }
    assert all(scene["duration"] >= sb.SCENE_MIN_SECONDS for scene in scenes)


def test_repeated_reports_from_the_same_named_source_do_not_force_a_boundary():
    lines = [
        timed_line(0.0, 5.0, "The Financial Times reports that robot revenue increased."),
        timed_line(5.0, 6.0, "Financial Times reports that training data sales also grew."),
    ]

    scenes = sb.group_lines_into_scenes(lines)

    assert len(scenes) == 1
    assert scenes[0]["duration"] == 11.0
    assert "semantic_boundary_before" not in scenes[0]


def test_generic_lowercase_report_subjects_are_not_named_source_boundaries():
    lines = [
        timed_line(0.0, 5.0, "The company reports that robot revenue increased."),
        timed_line(5.0, 6.0, "The division reports that training data sales also grew."),
    ]

    scenes = sb.group_lines_into_scenes(lines)

    assert len(scenes) == 1
    assert "semantic_boundary_before" not in scenes[0]


@pytest.mark.parametrize(
    ("durations", "short_index"),
    [((5.0, 9.0), 0), ((9.0, 5.0), 1)],
)
def test_short_named_source_scene_is_marked_instead_of_cross_boundary_merge(
    durations, short_index
):
    first_duration, second_duration = durations
    lines = [
        timed_line(0.0, first_duration, "Reuters reports that the first topic changed."),
        timed_line(
            first_duration,
            second_duration,
            "Bloomberg reports that a separate second topic changed.",
        ),
    ]

    scenes = sb.group_lines_into_scenes(lines)

    assert len(scenes) == 2
    assert scenes[0]["text"].startswith("Reuters reports")
    assert scenes[1]["text"].startswith("Bloomberg reports")
    assert scenes[short_index]["short_scene_reason"] == (
        "preserved_citation_source_boundary"
    )


def test_short_source_lead_absorbs_its_own_continuation_before_source_change():
    lines = [
        timed_line(0.0, 4.0, "Reuters reports that the first topic changed."),
        timed_line(4.0, 4.0, "The same report adds supporting context."),
        timed_line(8.0, 9.0, "Bloomberg reports that a separate topic changed."),
    ]

    scenes = sb.group_lines_into_scenes(lines)

    assert [scene["duration"] for scene in scenes] == [8.0, 9.0]
    assert len(scenes[0]["lines"]) == 2
    assert "short_scene_reason" not in scenes[0]
    assert scenes[1]["semantic_boundary_before"]["to"] == "Bloomberg"


def test_target_closeness_splits_long_scene_before_third_sentence():
    lines = [
        timed_line(0.0, 7.5, "IEEE Spectrum notes that organizations allocate idea time."),
        timed_line(7.5, 4.0, "The research indicates that the practice pays off."),
        timed_line(
            11.5,
            9.91,
            "The article discusses bootlegging and skunkworks projects and promoter networks.",
        ),
    ]

    scenes = sb.group_lines_into_scenes(lines)

    assert [scene["duration"] for scene in scenes] == [11.5, 9.91]
    assert len(scenes[0]["lines"]) == 2
    assert len(scenes[1]["lines"]) == 1


def test_target_closeness_keeps_next_sentence_when_combined_span_is_closer():
    lines = [
        timed_line(0.0, 8.0, "The first sentence establishes one visual topic."),
        timed_line(8.0, 8.0, "The second sentence continues that same visual topic."),
    ]

    scenes = sb.group_lines_into_scenes(lines)

    assert len(scenes) == 1
    assert scenes[0]["duration"] == 16.0


def test_scene_ids_are_sequential_after_merging(tmp_path):
    path = write_script(tmp_path, [f"Line {i} here with words." for i in range(40)])
    board = sb.build_storyboard(
        script_path=path, audio_duration=400.0, title="T", is_monologue=True
    )
    ids = [scene["id"] for scene in board["scenes"]]
    assert ids == [f"scene-{i + 1:02d}" for i in range(len(ids))]
    assert [scene["index"] for scene in board["scenes"]] == list(range(len(ids)))


def test_keywords_skip_stopwords(tmp_path):
    path = write_script(tmp_path, ["The sunflowers in the fields were there because of the war."])
    board = sb.build_storyboard(
        script_path=path, audio_duration=20.0, title="T", is_monologue=True
    )
    keywords = board["scenes"][0]["keywords"]
    assert "sunflowers" in keywords
    assert "the" not in keywords and "were" not in keywords
