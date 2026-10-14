"""One ordered, script-bound contract for a Morning Desk edition's delivery.

Fact review keeps its opening/story/closing paragraph indices. Delivery retains
the approved greeting and splits the lead's first two sentences into a measured
news introduction. No approved narration is removed, rewritten or repeated.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

OPENING_DEADLINE_SECONDS = 20.0
OPENING_MAX_WORDS = 34
OPENING_MAX_CHARACTERS = 76

LEAD_RULES = """The first selected story is the locked publication-title lead.
Begin that story with exactly two short sentences: who did what (the new event),
then its concrete, evidence-supported significance or practical consequence.
Keep necessary attribution and uncertainty inside those sentences. Together
they must fit within 34 English words or 76 Chinese spoken characters, including
embedded Latin terms. They will introduce the first news story; do not refer to a
previous greeting, tease another story, list headlines or withhold the event.
Use the rest of that same paragraph for supporting detail without repeating the
opening. If significance is uncertain, explain the supported scope or limitation;
never invent impact. Preserve this opening through all corrections.
"""


def script_hash(script: str) -> str:
    return hashlib.sha256(script.strip().encode("utf-8")).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sentence_ends(paragraph: str):
    for match in re.finditer(r'[.!?。！？][”’"\']*(?=\s|$)|[。！？][”’"\']*', paragraph):
        prefix = paragraph[:match.end()]
        if re.search(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|vs|e\.g|i\.e|[A-Z](?:\.[A-Z])*)\.$", prefix):
            continue
        yield match.end()


def split_lead(paragraph: str) -> tuple[str, str]:
    """Split after two complete sentences without cutting names or decimals."""
    for index, end in enumerate(_sentence_ends(paragraph), 1):
        if index == 2:
            return paragraph[:end].strip(), paragraph[end:].strip()
    raise ValueError("story 1 cold open requires two complete sentences: event and significance")


def opening_failures(paragraph: str, language: str) -> list[str]:
    try:
        opening, _ = split_lead(paragraph)
    except ValueError as exc:
        return [str(exc)]
    latin = len(re.findall(r"[A-Za-z0-9]+(?:['’.-][A-Za-z0-9]+)*", opening))
    units = latin + (len(re.findall(r"[\u3400-\u9fff]", opening)) if language == "zh" else 0)
    maximum = OPENING_MAX_CHARACTERS if language == "zh" else OPENING_MAX_WORDS
    failures = []
    if units > maximum:
        failures.append(f"story 1 cold open has {units} spoken units; maximum {maximum} for the first 20 seconds")
    if re.match(r"(?:good morning|welcome|hello|today['’]s headlines|早上好|欢迎)", opening, re.I):
        failures.append("story 1 cold open must start with the core event, not a greeting")
    return failures


def review_rule(dossier) -> str:
    if not dossier.lead_story_id:
        return ""
    return (
        LEAD_RULES + "\nFor story 1, also check that its first two sentences convey the "
        "core event AND supported significance, with no unrelated news or generic opener. "
        "Missing either is missing lead coverage: flag A with claim 1.0. "
        "Unsupported significance still fails D on the affected claim."
    )


def prepare_delivery(script: str, dossier, task_dir: Path, language: str) -> str:
    """Preserve approved story order and wording; legacy tasks retain their layout."""
    if not dossier.lead_story_id:
        return script
    paragraphs = [line.strip() for line in script.splitlines() if line.strip()]
    ids = [article.id for article in dossier.selected]
    if not ids or ids[0] != dossier.lead_story_id or len(ids) != len(set(ids)):
        raise ValueError("Daily rundown lead/order is invalid")
    if len(paragraphs) != len(ids) + 2:
        raise ValueError("Daily rundown requires the approved opening/story/closing paragraphs")
    failures = opening_failures(paragraphs[1], language)
    if failures:
        raise ValueError("; ".join(failures))
    opening, continuation = split_lead(paragraphs[1])
    stories = []
    segments = [{"story_id": None, "kind": "opening", "text": paragraphs[0]}]
    for index, (article, narration) in enumerate(zip(dossier.selected, paragraphs[1:-1])):
        stories.append({
            "story_id": article.id, "order": index + 1, "narration": narration,
            "source_name": article.source_name, "url": article.url,
            "source_title": article.title,
        })
        texts = [opening, continuation] if index == 0 else [narration]
        for text in filter(None, texts):
            segments.append({"story_id": article.id, "kind": "news", "text": text})
    segments.append({"story_id": None, "kind": "closing", "text": paragraphs[-1]})
    delivered = "\n".join(row["text"] for row in segments)
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "script.reviewed.txt").write_text(script, encoding="utf-8")
    write_json(task_dir / "rundown.json", {
        "version": 2, "edition_date": dossier.edition_date, "language": language,
        "lead_story_id": ids[0], "story_order": ids, "stories": stories,
        "segments": segments, "opening": paragraphs[0], "lead_opening": opening,
        "opening_deadline_seconds": OPENING_DEADLINE_SECONDS,
        "reviewed_script_sha256": script_hash(script), "script_sha256": script_hash(delivered),
    })
    summary_path = task_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    summary.update(edition_date=dossier.edition_date, lead_story_id=ids[0], story_order=ids,
                   talking_points=[{"story_id": s["story_id"], "headline": s["source_title"],
                                    "source": s["source_name"], "url": s["url"],
                                    "evidence": s["narration"]} for s in stories])
    write_json(summary_path, summary)
    return delivered


def read_rundown(task_dir: Path, script: str) -> dict | None:
    path = task_dir / "rundown.json"
    if not path.exists():
        dossier_path = task_dir / "research" / "dossier.json"
        if dossier_path.is_file() and json.loads(dossier_path.read_text()).get("lead_story_id"):
            raise ValueError("Daily rundown is missing; finish script review before generating media or copy")
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") not in {1, 2} or data.get("script_sha256") != script_hash(script):
        raise ValueError("Daily rundown is stale; review the edited script before regenerating media or copy")
    segments = data.get("segments") or []
    if "\n".join(row["text"] for row in segments) != script.strip():
        raise ValueError("Daily rundown segments do not match the saved narration")
    order = data.get("story_order") or []
    if (not order or len(order) != len(set(order)) or data.get("lead_story_id") != order[0]
            or [s["story_id"] for s in data["stories"]] != order
            or segments[0]["text"] != data["opening"]):
        raise ValueError("Daily rundown story identity/order is invalid")
    observed = []
    for row in segments:
        if row["story_id"] and (not observed or observed[-1] != row["story_id"]):
            observed.append(row["story_id"])
    if observed != order:
        raise ValueError("Daily rundown segment story order is invalid")
    for story in data["stories"]:
        text = " ".join(row["text"] for row in segments if row["story_id"] == story["story_id"])
        if text.split() != story["narration"].split():
            raise ValueError("Daily rundown changed or duplicated approved story narration")
    reviewed = (task_dir / "script.reviewed.txt").read_text(encoding="utf-8")
    paragraphs = [line.strip() for line in reviewed.splitlines() if line.strip()]
    if (script_hash(reviewed) != data["reviewed_script_sha256"]
            or paragraphs[1:-1] != [s["narration"] for s in data["stories"]]
            or paragraphs[-1] != segments[-1]["text"]):
        raise ValueError("Daily rundown is not an exact split of the reviewed story paragraphs")
    if data["version"] == 2:
        if (segments[0] != {"story_id": None, "kind": "opening", "text": paragraphs[0]}
                or segments[-1] != {"story_id": None, "kind": "closing", "text": paragraphs[-1]}
                or any(row["kind"] != "news" or not row["story_id"] for row in segments[1:-1])
                or segments[1]["text"] != data.get("lead_opening")
                or split_lead(paragraphs[1])[0] != data.get("lead_opening")):
            raise ValueError("Daily rundown must retain the exact reviewed opening, lead and closing")
    return data


def review_input_for_edit(task_dir: Path, edited: str) -> str | None:
    """Reassemble edited delivery lines into their original review slots."""
    path = task_dir / "rundown.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    original = "\n".join(row["text"] for row in data["segments"])
    read_rundown(task_dir, original)
    if script_hash(edited) == data["script_sha256"]:
        return None
    lines = [line.strip() for line in edited.splitlines() if line.strip()]
    if len(lines) != len(data["segments"]):
        raise ValueError("Keep one line per existing delivery segment when editing Morning Desk narration")
    opening = (lines[0] if data["version"] == 2 else
               (task_dir / "script.reviewed.txt").read_text(encoding="utf-8").splitlines()[0])
    stories = [" ".join(text for text, row in zip(lines, data["segments"])
                        if row["story_id"] == story_id) for story_id in data["story_order"]]
    return "\n".join([opening, *stories, lines[-1]])


def bind_timeline(task_dir: Path, rundown: dict, pacing: dict, *, audio_sha256: str) -> dict:
    """Check measured lead timing after the greeting and bind scenes to stories."""
    rows = pacing.get("segments") or []
    expected = rundown["segments"]
    if pacing.get("passed") is not True or len(rows) != len(expected):
        raise ValueError("Daily rundown requires a verified complete audio timeline")
    for row, segment in zip(rows, expected):
        if row.get("text") != segment["text"]:
            raise ValueError("Daily rundown audio segment order differs from narration")
        row.update(story_id=segment["story_id"], kind=segment["kind"])
    lead_index = 1 if rundown["version"] == 2 else 0
    end = float(rows[lead_index]["program_end"])
    start = float(rows[lead_index]["program_start"])
    if (not math.isfinite(start) or start < 0 or not math.isfinite(end)
            or not 0 < end - start <= OPENING_DEADLINE_SECONDS
            or (lead_index == 0 and start != 0)):
        raise ValueError(
            f"Daily lead must finish event/significance by 20s of news narration; measured {start:.3f}-{end:.3f}s. "
            "Shorten and re-review the lead before regenerating narration."
        )
    if lead_index:
        opening_start = float(rows[0]["program_start"])
        opening_end = float(rows[0]["program_end"])
        if not (math.isfinite(opening_start) and 0 <= opening_start < opening_end <= start):
            raise ValueError("Daily opening requires a complete timed greeting before the lead")
    chapters = []
    for story in rundown["stories"]:
        row = next(row for row in rows if row["story_id"] == story["story_id"])
        chapters.append({"story_id": story["story_id"], "start_seconds": row["program_start"]})
    report = {
        "passed": True, "script_sha256": rundown["script_sha256"],
        "source_audio_sha256": audio_sha256,
        "opening_end_seconds": float(rows[0]["program_end"]),
        "lead_start_seconds": start, "lead_end_seconds": end,
        "story_order": rundown["story_order"], "chapters": chapters,
    }
    write_json(task_dir / "chapters.next.json", report)
    return report


def read_chapters(task_dir: Path, rundown: dict, audio_path: str | None,
                  video_path: str | None = None) -> list[dict]:
    path = task_dir / "chapters.json"
    if (not path.exists() or not audio_path or not Path(audio_path).is_file()
            or not video_path or not Path(video_path).is_file()):
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    chapters = data.get("chapters") or []
    starts = [float(row["start_seconds"]) for row in chapters]
    if (data.get("passed") is not True or data.get("script_sha256") != rundown["script_sha256"]
            or data.get("source_audio_sha256") != hashlib.sha256(Path(audio_path).read_bytes()).hexdigest()
            or data.get("story_order") != rundown["story_order"]
            or [row["story_id"] for row in chapters] != rundown["story_order"]
            or not starts or starts[0] != (data.get("lead_start_seconds") if rundown["version"] == 2 else 0)
            or any(not math.isfinite(t) or t < 0 for t in starts)
            or any(b <= a for a, b in zip(starts, starts[1:]))
            or Path(video_path).name != f"video-{data.get('rendered_video_sha256', '')[:16]}.mp4"):
        raise ValueError("Daily chapters are stale; compose again from the current narration")
    return chapters


def promote_chapters(task_dir: Path, video_sha256: str) -> None:
    path = task_dir / "chapters.next.json"
    if path.is_file():
        report = json.loads(path.read_text(encoding="utf-8"))
        report["rendered_video_sha256"] = video_sha256
        write_json(task_dir / "chapters.json", report)
        path.unlink()


def chapter_text(rundown: dict, chapters: list[dict], title: str) -> str:
    if not chapters:
        return ""
    labels = {s["story_id"]: s["narration"].splitlines()[0] for s in rundown["stories"]}
    labels[rundown["lead_story_id"]] = title.split(" | Daily Tech Brief", 1)[0]
    lines = ["00:00 ByteFront Espresso"] if float(chapters[0]["start_seconds"]) > 0 else []
    for row in chapters:
        seconds = int(float(row["start_seconds"]))
        hours, rest = divmod(seconds, 3600)
        minutes, seconds = divmod(rest, 60)
        timestamp = f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
        label = labels[row["story_id"]]
        # A whole first sentence from approved narration, never a truncated claim.
        if row["story_id"] != rundown["lead_story_id"]:
            label = label[:next(_sentence_ends(label), len(label))]
        lines.append(f"{timestamp} {label}")
    return "\n".join(lines)
