"""ByteFront Espresso branded intro staging and runtime variables.

The Gemini plate supplies the moving paper-collage background. HyperFrames owns
the logo and edition date, so each render can inject its actual date without
regenerating or baking text into the video asset.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from backend.pipeline import outros

INTRO_DURATION_SECONDS = 6.0
INTRO_SCENE_ID = "scene-intro"
INTRO_VARIABLES_FILENAME = "composition.variables.json"
DEFAULT_EDITION_TIMEZONE = "Asia/Singapore"

_MONTHS = (
    "JANUARY",
    "FEBRUARY",
    "MARCH",
    "APRIL",
    "MAY",
    "JUNE",
    "JULY",
    "AUGUST",
    "SEPTEMBER",
    "OCTOBER",
    "NOVEMBER",
    "DECEMBER",
)
_WEEKDAYS = (
    "MONDAY",
    "TUESDAY",
    "WEDNESDAY",
    "THURSDAY",
    "FRIDAY",
    "SATURDAY",
    "SUNDAY",
)


def _edition_date(value: str | date | datetime | None, storyboard: dict) -> date:
    """Resolve a stable edition date, preferring persisted task metadata."""
    persisted = storyboard.get("edition_date")
    candidate: str | date | datetime | None = value if value is not None else persisted
    if isinstance(candidate, datetime):
        return candidate.date()
    if isinstance(candidate, date):
        return candidate
    if isinstance(candidate, str) and candidate.strip():
        try:
            return date.fromisoformat(candidate.strip())
        except ValueError as exc:
            raise ValueError("edition_date must use ISO YYYY-MM-DD") from exc

    # Manual tasks do not carry a scheduler edition date. Capture the render
    # date once here and persist it into the storyboard so retries do not drift.
    return datetime.now(ZoneInfo(DEFAULT_EDITION_TIMEZONE)).date()


def edition_variables(
    storyboard: dict,
    edition_date: str | date | datetime | None = None,
) -> dict[str, str]:
    """Return the exact string values injected into the HyperFrames intro."""
    resolved_date = _edition_date(edition_date, storyboard)
    return {
        "edition_date": (
            f"{_MONTHS[resolved_date.month - 1]} {resolved_date.day:02d}, "
            f"{resolved_date.year}"
        ),
        "edition_weekday": _WEEKDAYS[resolved_date.weekday()],
    }


def composition_variable_specs(values: dict[str, str]) -> list[dict[str, str]]:
    """HyperFrames variable declarations used by Studio and strict rendering."""
    labels = {
        "edition_date": "Edition date",
        "edition_weekday": "Edition weekday",
    }
    return [
        {
            "id": variable_id,
            "type": "string",
            "label": labels[variable_id],
            "default": value,
        }
        for variable_id, value in values.items()
    ]


def _shift_narrated_timeline(storyboard: dict, offset: float) -> None:
    for scene in storyboard.get("scenes") or []:
        scene["start"] = round(float(scene.get("start") or 0) + offset, 2)
        for line in scene.get("lines") or []:
            line["start"] = round(float(line.get("start") or 0) + offset, 3)

    storyboard["content_start"] = round(
        float(storyboard.get("content_start") or 0) + offset,
        2,
    )
    storyboard["outro_start"] = round(
        float(storyboard.get("outro_start") or storyboard.get("total_duration") or 0)
        + offset,
        2,
    )
    storyboard["total_duration"] = round(
        float(storyboard.get("total_duration") or 0) + offset,
        2,
    )


def stage_intro(
    task_dir: str | Path,
    storyboard: dict,
    style: str,
    *,
    edition_date: str | date | datetime | None = None,
) -> dict[str, Any]:
    """Stage media, prepend the intro scene, and shift narration by six seconds."""
    if any(scene.get("scene_kind") == "intro" for scene in storyboard.get("scenes") or []):
        raise ValueError("storyboard already contains a branded intro")

    preset = outros.resolve_outro_preset(style)
    task_root = Path(task_dir)
    asset_dir = task_root / "assets" / "intro"
    asset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(preset["background_path"], asset_dir / "background.mp4")
    shutil.copyfile(preset["logo_path"], asset_dir / "bytefront-logo.png")

    resolved_date = _edition_date(edition_date, storyboard)
    values = edition_variables(storyboard, resolved_date)
    resolved_iso_date = resolved_date.isoformat()
    _shift_narrated_timeline(storyboard, INTRO_DURATION_SECONDS)

    scene = {
        "id": INTRO_SCENE_ID,
        "index": -1,
        "start": 0.0,
        "duration": INTRO_DURATION_SECONDS,
        "lines": [],
        "text": (
            "ByteFront Espresso branded opening with the current edition date, "
            "weekday, and unified wordmark over the selected Gemini motion plate."
        ),
        "word_count": 0,
        "keywords": ["ByteFront Espresso", values["edition_date"]],
        "scene_kind": "intro",
        "intro_style": style,
    }
    storyboard.setdefault("scenes", []).insert(0, scene)
    storyboard["intro_duration"] = INTRO_DURATION_SECONDS
    storyboard["intro_style"] = style
    storyboard["edition_date"] = resolved_iso_date
    storyboard["intro_variables"] = values
    storyboard["composition_variables"] = composition_variable_specs(values)
    storyboard["scene_count"] = len(storyboard["scenes"])

    variables_path = task_root / INTRO_VARIABLES_FILENAME
    variables_path.write_text(
        json.dumps(values, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "id": INTRO_SCENE_ID,
        "archetype": "intro",
        "kicker": "YOUR DAILY SHOT OF FRONTIER TECH",
        "headline": "ByteFront Espresso",
        "body": "Fresh signals, served daily.",
        "items": [],
        "accent": "coral",
        "motif": "none",
        "footage_src": "assets/intro/background.mp4",
        "footage_kind": "video",
        "intro_logo_src": "assets/intro/bytefront-logo.png",
        "intro_style": style,
        "intro_label": preset["label"],
        **values,
        "grounding_source": "configured_intro",
    }


def intro_overlay_problems(text: str) -> list[str]:
    """Check the editable and runtime-variable contract after agent rewrites."""
    problems: list[str] = []
    required = {
        'data-intro-role="brand"': "missing editable intro brand overlay",
        'data-intro-role="edition"': "missing editable intro edition overlay",
        'data-intro-field="date"': "missing dynamic edition date field",
        'data-intro-field="weekday"': "missing dynamic weekday field",
        "window.__hyperframes.getVariables()": "intro does not read HyperFrames variables",
    }
    for marker, message in required.items():
        if marker not in text:
            problems.append(message)
    forbidden = {
        'data-intro-field="label"': "removed briefing-label field is still present",
        'data-intro-field="story-count"': "removed story-count field is still present",
    }
    for marker, message in forbidden.items():
        if marker in text:
            problems.append(message)
    if "daily tech briefing" in text.lower():
        problems.append("removed Daily Tech Briefing label is still present")
    if re.search(r"\b(?:\d{1,3}\s+stor(?:y|ies)|special edition)\b", text, re.IGNORECASE):
        problems.append("removed story-count label is still present")
    return problems
