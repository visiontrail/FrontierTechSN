"""Pipeline prompt registry — the source of truth for the admin Prompts console.

The digestion/scriptwriting stages load their system prompts from plain ``.txt``
files under :data:`backend.config.PROMPTS_DIR` at call time (see
``digester.summarize`` / ``digester.generate_script``), so editing a file takes
effect on the next task with no restart. This module gives those files a small
amount of structure the UI needs:

* a stable ``key`` and human ``label``
* the pipeline ``stage`` the prompt belongs to
* a ``description`` of what it does
* the ``{placeholder}`` variables the pipeline substitutes at runtime

It also snapshots the shipped defaults once (into ``PROMPTS_DIR/.defaults``) so
the console can offer a non-destructive "reset to default".
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field

from backend import config

DEFAULTS_DIR = config.PROMPTS_DIR / ".defaults"


@dataclass(frozen=True)
class PromptSpec:
    key: str
    file: str
    label: str
    stage: str
    description: str
    # Runtime {placeholders} the pipeline replaces before sending the prompt.
    variables: list[str] = field(default_factory=list)


# Registry of every prompt the pipeline reads from disk. Adding a prompt here
# (plus its .txt file) is all it takes to expose it in the admin console.
PROMPT_REGISTRY: list[PromptSpec] = [
    PromptSpec(
        key="summarize",
        file="summarize.txt",
        label="Content Summarization",
        stage="Digest",
        description=(
            "Distills extracted source material (YouTube / EPUB / PDF) into a "
            "JSON brief of talking points and key quotes that drives scriptwriting."
        ),
        variables=[],
    ),
    PromptSpec(
        key="scriptwrite_monologue",
        file="scriptwrite_monologue.txt",
        label="Script — Monologue",
        stage="Script",
        description=(
            "Turns the summary brief into a solo talk-show host script "
            "(the default, primary style)."
        ),
        variables=["word_count", "duration_minutes"],
    ),
    PromptSpec(
        key="scriptwrite_dialogue",
        file="scriptwrite_dialogue.txt",
        label="Script — Dialogue",
        stage="Script",
        description="Turns the summary brief into a two-host back-and-forth conversation script.",
        variables=["word_count", "duration_minutes"],
    ),
    PromptSpec(
        key="title",
        file="title.txt",
        label="Video Title Agent",
        stage="Title",
        description=(
            "Guides the independent Agent SDK task that turns the final narration "
            "and content brief into one publication-ready video title."
        ),
        variables=[],
    ),
    PromptSpec(
        key="footage_plan",
        file="footage_plan.txt",
        label="Public Footage Plan",
        stage="Footage",
        description=(
            "Turns the title and narration into concrete Wikimedia Commons "
            "B-roll search queries for the media-scouting agent."
        ),
        variables=[],
    ),
    PromptSpec(
        key="thumbnail",
        file="thumbnail.txt",
        label="Viral Thumbnail Formula",
        stage="Thumbnail",
        description=(
            "Turns the final audio narration script into one high-impact 16:9 "
            "ChatGPT image prompt. Edit this formula to tune every future cover."
        ),
        variables=[],
    ),
    PromptSpec(
        key="visual_plan",
        file="visual_plan.txt",
        label="Scene Art Direction",
        stage="Compose",
        description=(
            "Decides what the viewer sees during each storyboard scene — layout "
            "archetype, on-screen copy, accent colour and motif. Drives both the "
            "deterministic scene kit and the brief handed to the authoring agents."
        ),
        variables=[],
    ),
]

_BY_KEY = {spec.key: spec for spec in PROMPT_REGISTRY}


def snapshot_defaults() -> None:
    """Copy the current prompt files into ``.defaults`` once, so the console can
    offer reset-to-default. Existing snapshots are never overwritten."""
    DEFAULTS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in PROMPT_REGISTRY:
        src = config.PROMPTS_DIR / spec.file
        dst = DEFAULTS_DIR / spec.file
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def get_spec(key: str) -> PromptSpec | None:
    return _BY_KEY.get(key)


def read_content(spec: PromptSpec) -> str:
    path = config.PROMPTS_DIR / spec.file
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_default(spec: PromptSpec) -> str | None:
    path = DEFAULTS_DIR / spec.file
    return path.read_text(encoding="utf-8") if path.exists() else None


def write_content(spec: PromptSpec, content: str) -> None:
    path = config.PROMPTS_DIR / spec.file
    path.write_text(content, encoding="utf-8")


def reset_content(spec: PromptSpec) -> str:
    """Restore a prompt file from its default snapshot. Returns the restored
    content. Raises ``FileNotFoundError`` when no snapshot exists."""
    default = read_default(spec)
    if default is None:
        raise FileNotFoundError(f"No default snapshot for prompt '{spec.key}'")
    write_content(spec, default)
    return default


def _missing_variables(spec: PromptSpec, content: str) -> list[str]:
    """Required runtime placeholders that the edited content no longer contains.

    Dropping e.g. ``{word_count}`` from a script prompt is almost always a
    mistake — the pipeline would emit a script of the wrong length — so the
    console surfaces it as a warning (not a hard block)."""
    present = set(re.findall(r"\{(\w+)\}", content))
    return [v for v in spec.variables if v not in present]


def to_dict(spec: PromptSpec, include_content: bool = True) -> dict:
    default = read_default(spec)
    content = read_content(spec)
    data = {
        "key": spec.key,
        "file": spec.file,
        "label": spec.label,
        "stage": spec.stage,
        "description": spec.description,
        "variables": spec.variables,
        "has_default": default is not None,
    }
    if include_content:
        data["content"] = content
        data["is_modified"] = default is not None and content != default
        data["missing_variables"] = _missing_variables(spec, content)
    return data
