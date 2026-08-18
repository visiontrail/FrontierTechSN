"""Turn a narration script plus its rendered audio into a timed storyboard.

This is the stage that used to be missing. The old composer went straight from
"script lines" to "one caption div per line", which is why a 12-minute episode
rendered as 158 stacked subtitle clips and nothing else. A storyboard sits in
between: it groups the spoken lines into *scenes* (a handful of seconds each,
cut on natural pauses) and hands every downstream consumer — the AI visual
planner, the scene-kit renderer, and the Claude Agent SDK director — the same
timing truth.

Nothing here talks to a model. Timing must be reproducible and must line up with
the audio to the frame, so segmentation is pure arithmetic over the script and
the ffmpeg silence map. The creative layer (what each scene *looks* like) is
:mod:`backend.pipeline.visual_plan`.
"""

from __future__ import annotations

import json
import re
import wave
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from difflib import SequenceMatcher
from pathlib import Path

LogCallback = Callable[[str], None]

# Scene pacing. A scene is the unit of visual change: too short and the video
# strobes, too long and it reads as a static slide. 14s is roughly two to three
# spoken sentences, which is also about how long a single visual idea holds.
SCENE_TARGET_SECONDS = 14.0
SCENE_MIN_SECONDS = 7.0
SCENE_MAX_SECONDS = 24.0

# The narration and its first content-led scene start immediately. Older
# versions reserved five seconds for a metadata-style title card, which delayed
# the actual subject and made every video open the same way.
CONTENT_START = 0.0
OUTRO_DURATION = 5.0

SPEAKER_LABEL_RE = re.compile(r"^Speaker\s*(\d+)\s*[:：\-—–]\s*(.+)$", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?|[\u3400-\u9fff]")

# Words too generic to describe a scene visually. Used only for the keyword hint
# that helps the visual planner and the footage matcher; not user-visible.
_STOPWORDS = frozenset(
    """
    a about after all also am an and any are as at be because been before being but by came can come
    could did do does doing done down each even every for from get got had has have he her here him his
    how i if in into is it its just like make many me might more most much must my no not now of off on
    once one only or other our out over own said same say see she should so some still such take than
    that the their them then there these they thing think this those though through to too under until
    up us very was way we well were what when where which while who why will with would you your
    """.split()
)


def get_audio_duration(wav_path: str | Path) -> float:
    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes() / w.getframerate()


def parse_script_lines(script_path: str | Path, is_monologue: bool = False) -> list[dict]:
    """Read the script into one record per spoken line.

    Accepts both the ``Speaker 1: ...`` dialogue form and the bare-line
    monologue form the scriptwriter prompt emits.
    """
    lines: list[dict] = []
    for raw in Path(script_path).read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        match = SPEAKER_LABEL_RE.match(text)
        if match:
            speaker = int(match.group(1))
            text = match.group(2)
        else:
            speaker = 1 if is_monologue else (len(lines) % 2) + 1
        # A model occasionally emits an entire paragraph on one physical line.
        # Treating that as one indivisible timing unit can leave the same visual
        # on screen for a minute or more. Sentence-sized units remain faithful
        # to the TTS input while giving the forced aligner useful cut points.
        sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
        for sentence in sentences or [text]:
            token_count = len(_tokens(sentence))
            lines.append(
                {
                    "speaker": speaker,
                    "text": sentence,
                    "word_count": token_count or 1,
                }
            )
    return lines


def _tokens(value: object) -> list[str]:
    """Comparable speech tokens for scripts and Whisper output."""
    text = str(value or "").replace("’", "'").casefold()
    return TOKEN_RE.findall(text)


def _apply_boundaries(lines: list[dict], boundaries: list[float], audio_duration: float) -> list[dict]:
    """Land monotonically increasing boundaries on line timing records."""
    if not lines:
        return lines

    minimum = min(0.2, audio_duration / max(2, len(lines) * 2))
    marks = [0.0]
    for raw in boundaries[: len(lines) - 1]:
        marks.append(max(marks[-1] + minimum, min(float(raw), audio_duration)))
    marks.append(audio_duration)

    # A pathological recognizer can put several boundaries at the very end.
    # Repair from right to left so every line retains a positive interval.
    for i in range(len(marks) - 2, 0, -1):
        marks[i] = min(marks[i], marks[i + 1] - minimum)
    marks[0], marks[-1] = 0.0, audio_duration

    for i, line in enumerate(lines):
        start, end = marks[i], marks[i + 1]
        line["start"] = round(start, 3)
        line["duration"] = round(max(minimum, end - start), 3)
    return lines


def _select_silence_boundaries(
    lines: list[dict], audio_duration: float, candidates: list[float]
) -> list[float]:
    """Choose silence marks near expected line positions, not the first N.

    VibeVoice can insert sentence-internal pauses longer than a second. The old
    implementation consumed the first ``line_count - 1`` silence marks, which
    pushed every later line early and left the final sentence holding the
    remainder of the audio. This dynamic program finds the monotonic subset
    closest to word-proportional boundary estimates.
    """
    needed = len(lines) - 1
    usable = sorted({float(value) for value in candidates if 0 < value < audio_duration})
    if needed <= 0 or len(usable) < needed:
        return []

    total = sum(max(1, int(line.get("word_count") or 1)) for line in lines)
    running = 0
    expected: list[float] = []
    for line in lines[:-1]:
        running += max(1, int(line.get("word_count") or 1))
        expected.append(audio_duration * running / total)

    infinity = float("inf")
    previous = [(mark - expected[0]) ** 2 for mark in usable]
    parents: list[list[int]] = []
    for boundary_index in range(1, needed):
        prefix_cost = infinity
        prefix_index = -1
        best_before: list[tuple[float, int]] = []
        for candidate_index, cost in enumerate(previous):
            best_before.append((prefix_cost, prefix_index))
            if cost < prefix_cost:
                prefix_cost, prefix_index = cost, candidate_index

        current = [infinity] * len(usable)
        parent = [-1] * len(usable)
        for candidate_index, mark in enumerate(usable):
            cost, prior = best_before[candidate_index]
            if prior >= 0:
                current[candidate_index] = cost + (mark - expected[boundary_index]) ** 2
                parent[candidate_index] = prior
        parents.append(parent)
        previous = current

    end_index = min(range(len(usable)), key=previous.__getitem__)
    if previous[end_index] == infinity:
        return []
    chosen = [end_index]
    for parent in reversed(parents):
        end_index = parent[end_index]
        chosen.append(end_index)
    chosen.reverse()
    return [usable[index] for index in chosen]


def assign_line_timing(
    lines: list[dict],
    audio_duration: float,
    silence_boundaries: list[float] | None = None,
) -> list[dict]:
    """Give every line a start/duration inside ``audio_duration``.

    Prefers the ffmpeg silence map when it has at least one boundary per line
    gap — that tracks the actual synthesis. Otherwise falls back to splitting
    the timeline in proportion to word count, which is stable if imprecise.
    """
    if not lines:
        return lines

    total_words = sum(line["word_count"] for line in lines) or len(lines)
    boundaries = silence_boundaries or []

    selected = _select_silence_boundaries(lines, audio_duration, boundaries)
    if selected:
        return _apply_boundaries(lines, selected, audio_duration)

    elapsed = 0.0
    for line in lines:
        share = (line["word_count"] or 1) / total_words
        duration = share * audio_duration
        line["start"] = round(elapsed, 3)
        line["duration"] = round(duration, 3)
        elapsed += duration
    return lines


def align_lines_to_transcript(
    lines: list[dict],
    word_transcript: list[dict],
    audio_duration: float,
    *,
    minimum_word_coverage: float = 0.65,
    maximum_boundary_uncertainty: float = 3.0,
) -> tuple[list[dict], dict]:
    """Force-align the canonical script to Whisper word timestamps.

    Whisper supplies the acoustic clock; the script supplies the exact words.
    ``SequenceMatcher`` tolerates recognizer substitutions while preserving
    order. Line boundaries are interpolated between the nearest matched word
    anchors, so an ASR error cannot create the catastrophic cumulative drift
    caused by consuming silence marks by ordinal position.
    """
    expected: list[str] = []
    line_starts: list[int] = []
    for line in lines:
        line_starts.append(len(expected))
        expected.extend(_tokens(line.get("text", "")))

    observed: list[str] = []
    observed_words: list[dict] = []
    for word in word_transcript or []:
        tokens = _tokens(word.get("text", ""))
        if not tokens:
            continue
        try:
            start = max(0.0, float(word["start"]))
            end = min(audio_duration, max(start, float(word["end"])))
        except (KeyError, TypeError, ValueError):
            continue
        # Whisper normally emits one token per record. If punctuation/imported
        # text expands into several tokens, share the acoustic span between them.
        span = max(0.001, end - start)
        for index, token in enumerate(tokens):
            token_start = start + span * index / len(tokens)
            token_end = start + span * (index + 1) / len(tokens)
            observed.append(token)
            observed_words.append({"start": token_start, "end": token_end})

    matcher = SequenceMatcher(a=expected, b=observed, autojunk=False)
    pairs: list[tuple[int, int]] = []
    for block in matcher.get_matching_blocks():
        pairs.extend((block.a + offset, block.b + offset) for offset in range(block.size))

    anchors: list[tuple[float, float]] = [(0.0, 0.0)]
    for expected_index, observed_index in pairs:
        word = observed_words[observed_index]
        anchors.append(
            (expected_index + 0.5, (float(word["start"]) + float(word["end"])) / 2)
        )
    anchors.append((float(len(expected)), audio_duration))
    anchors.sort()

    positions = [position for position, _ in anchors]
    internal_boundaries: list[float] = []
    uncertainties: list[float] = []
    for token_position in line_starts[1:]:
        right_index = min(len(anchors) - 1, bisect_left(positions, token_position))
        left_index = max(0, right_index - 1)
        left_position, left_time = anchors[left_index]
        right_position, right_time = anchors[right_index]
        if right_position <= left_position:
            boundary = left_time
        else:
            fraction = (token_position - left_position) / (right_position - left_position)
            boundary = left_time + fraction * (right_time - left_time)
        internal_boundaries.append(boundary)
        uncertainties.append(min(abs(boundary - left_time), abs(right_time - boundary)))

    _apply_boundaries(lines, internal_boundaries, audio_duration)

    matched_lines = {
        max(0, bisect_right(line_starts, expected_index) - 1)
        for expected_index, _ in pairs
    }
    word_coverage = len(pairs) / max(1, len(expected))
    line_coverage = len(matched_lines) / max(1, len(lines))
    speech_end = max((float(word["end"]) for word in observed_words), default=0.0)
    audio_coverage = speech_end / max(0.001, audio_duration)
    max_uncertainty = max(uncertainties, default=0.0)

    failures: list[str] = []
    if word_coverage < minimum_word_coverage:
        failures.append(
            f"matched-word coverage {word_coverage:.1%} is below {minimum_word_coverage:.1%}"
        )
    # A high aggregate word score can hide a completely missing short sentence
    # at the end of an otherwise long episode. Every canonical sentence must
    # therefore retain at least one acoustic anchor before rendering. This is a
    # completeness gate, not a demand that Whisper spell every proper noun.
    if line_coverage < 1.0:
        failures.append(
            f"matched-line coverage {line_coverage:.1%} is below 100.0%; "
            "one or more script sentences have no acoustic anchor"
        )
    if audio_coverage < 0.80:
        failures.append(f"transcript covers only {audio_coverage:.1%} of the audio")
    if max_uncertainty > maximum_boundary_uncertainty:
        failures.append(
            f"maximum boundary uncertainty {max_uncertainty:.2f}s exceeds "
            f"{maximum_boundary_uncertainty:.2f}s"
        )

    report = {
        "method": "whisper_script_forced_alignment",
        "script_words": len(expected),
        "transcript_words": len(observed),
        "matched_words": len(pairs),
        "word_coverage": round(word_coverage, 4),
        "line_coverage": round(line_coverage, 4),
        "audio_coverage": round(audio_coverage, 4),
        "max_boundary_uncertainty_seconds": round(max_uncertainty, 3),
        "passed": not failures,
        "failure_reasons": failures,
    }
    return lines, report


def _keywords(text: str, limit: int = 8) -> list[str]:
    """Content words of a scene, most frequent first, for visual/footage hints."""
    counts: dict[str, int] = {}
    for word in re.findall(r"[A-Za-z][A-Za-z'\-]+", text.lower()):
        if len(word) < 4 or word in _STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [word for word, _ in ranked[:limit]]


def group_lines_into_scenes(lines: list[dict], offset: float = 0.0) -> list[dict]:
    """Batch timed lines into scenes of roughly ``SCENE_TARGET_SECONDS``.

    A scene closes once it has reached the target length; a line that would push
    it past ``SCENE_MAX_SECONDS`` starts a new scene instead. ``offset`` shifts
    every time into absolute composition time when a caller needs a lead-in.
    """
    scenes: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        if not current:
            return
        start = current[0]["start"]
        end = current[-1]["start"] + current[-1]["duration"]
        text = " ".join(line["text"] for line in current)
        scenes.append(
            {
                "id": f"scene-{len(scenes) + 1:02d}",
                "index": len(scenes),
                "start": round(start + offset, 2),
                "duration": round(max(0.5, end - start), 2),
                "lines": [
                    {
                        "start": round(line["start"] + offset, 2),
                        "duration": round(line["duration"], 2),
                        "speaker": line["speaker"],
                        "text": line["text"],
                    }
                    for line in current
                ],
                "text": text,
                "word_count": sum(line["word_count"] for line in current),
                "keywords": _keywords(text),
            }
        )
        current.clear()

    for line in lines:
        if current:
            span = (line["start"] + line["duration"]) - current[0]["start"]
            reached_target = (line["start"] - current[0]["start"]) >= SCENE_TARGET_SECONDS
            if reached_target or span > SCENE_MAX_SECONDS:
                flush()
        current.append(line)
    flush()

    # A trailing stub (one short line orphaned by the max-length cut) reads as a
    # flash frame. Fold anything under the minimum back into its predecessor.
    merged: list[dict] = []
    for scene in scenes:
        if merged and scene["duration"] < SCENE_MIN_SECONDS:
            prev = merged[-1]
            prev["lines"].extend(scene["lines"])
            prev["duration"] = round(
                (scene["start"] + scene["duration"]) - prev["start"], 2
            )
            prev["text"] = f"{prev['text']} {scene['text']}"
            prev["word_count"] += scene["word_count"]
            prev["keywords"] = _keywords(prev["text"])
            continue
        merged.append(scene)

    for i, scene in enumerate(merged):
        scene["id"] = f"scene-{i + 1:02d}"
        scene["index"] = i
    return merged


def build_storyboard(
    *,
    script_path: str | Path,
    audio_duration: float,
    title: str,
    silence_boundaries: list[float] | None = None,
    word_transcript: list[dict] | None = None,
    minimum_word_coverage: float = 0.65,
    maximum_boundary_uncertainty: float = 3.0,
    is_monologue: bool = True,
    summary: dict | None = None,
    log: LogCallback | None = None,
) -> dict:
    """Assemble the full storyboard document for one task."""
    lines = parse_script_lines(script_path, is_monologue=is_monologue)
    if word_transcript:
        lines, alignment = align_lines_to_transcript(
            lines,
            word_transcript,
            audio_duration,
            minimum_word_coverage=minimum_word_coverage,
            maximum_boundary_uncertainty=maximum_boundary_uncertainty,
        )
    else:
        lines = assign_line_timing(lines, audio_duration, silence_boundaries)
        alignment = {
            "method": "silence_guided_estimate" if silence_boundaries else "word_count_estimate",
            "passed": False,
            "failure_reasons": ["word-level transcript unavailable"],
        }
    scenes = group_lines_into_scenes(lines, offset=CONTENT_START)

    if log:
        log(
            f"Storyboard: {len(lines)} spoken lines -> {len(scenes)} scenes "
            f"(avg {audio_duration / max(1, len(scenes)):.1f}s each)"
        )

    return {
        "title": title,
        "thesis": (summary or {}).get("thesis", ""),
        "audio_duration": round(audio_duration, 2),
        "title_duration": 0.0,
        "outro_duration": OUTRO_DURATION,
        "content_start": CONTENT_START,
        "outro_start": round(CONTENT_START + audio_duration, 2),
        "total_duration": round(CONTENT_START + audio_duration + OUTRO_DURATION, 2),
        "scene_count": len(scenes),
        "scenes": scenes,
        "alignment": alignment,
    }


def write_storyboard(task_dir: str | Path, storyboard: dict) -> Path:
    path = Path(task_dir) / "storyboard.json"
    path.write_text(json.dumps(storyboard, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_storyboard(task_dir: str | Path) -> dict | None:
    path = Path(task_dir) / "storyboard.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
