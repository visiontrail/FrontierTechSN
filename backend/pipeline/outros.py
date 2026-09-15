"""ByteFront Espresso system bookend library and render-task staging.

The selected preset owns the moving picture. HyperFrames owns the editable
brand, closing message, and engagement controls, which lets the video director
agent revise copy and layout without regenerating the background video.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
import subprocess
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from backend import config

DEFAULT_OUTRO_STYLE = "morning-brief"
OUTRO_DURATION_SECONDS = 6.0
# Leave room for the renderer's inclusive final frame and AAC packet rounding.
MAX_CREDITS_OUTRO_SECONDS = 29.9
OUTRO_LIBRARY_DIR = config.PROJECT_ROOT / "data" / "outros"
OUTRO_LOGO_FILENAME = "bytefront-logo-transparent.png"
OUTRO_LOGO_SHA256 = "d4e2fd3d0b0c8b8a0cacc0c4b710acde141961652363af414a9e262bdef40c57"

OUTRO_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "data-extraction",
        "label": "1. 数据萃取｜深色、科技感",
        "name": "数据萃取",
        "tone": "深色、科技感",
        "description": "钴蓝数据流、铜色汇聚脉冲与纵深粒子。",
        "filename": "01-data-extraction-gemini.mp4",
        "sha256": "dfd9431e4834407a3ea6f33b40ef4e756ce1b40b1165bc2ee9e68ca88fc375ad",
    },
    {
        "id": "morning-brief",
        "label": "2. 晨间简报｜温暖、编辑感",
        "name": "晨间简报",
        "tone": "温暖、编辑感",
        "description": "纸张层次、日出几何、电路微光与轻盈纸尘。",
        "filename": "02-morning-brief-gemini.mp4",
        "sha256": "41ec0265271b2117c7f647155b43190bc4599a504fa5af264b9b8d34c781893e",
    },
    {
        "id": "signal-shot",
        "label": "3. 信号快讯｜克制、播报感",
        "name": "信号快讯",
        "tone": "克制、播报感",
        "description": "海军蓝信号条、铜色脉冲、心跳轨迹与网格标记。",
        "filename": "03-signal-shot-gemini.mp4",
        "sha256": "8959237ca86124aac20d24e13276dcde143e2161a2806c02766c3b5b5b9adbf0",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preset(style: str) -> dict[str, Any]:
    for preset in OUTRO_PRESETS:
        if preset["id"] == style:
            return preset
    available = ", ".join(str(preset["id"]) for preset in OUTRO_PRESETS)
    raise ValueError(f"Unknown outro style {style!r}; available: {available}")


def _verified_asset(path: Path, expected_sha256: str, label: str) -> Path:
    if not path.is_file() or path.stat().st_size < 4096:
        raise RuntimeError(f"{label} is missing or empty: {path}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} checksum mismatch: expected {expected_sha256}, got {actual}"
        )
    return path


def resolve_outro_preset(style: str, *, verify_assets: bool = True) -> dict[str, Any]:
    """Resolve one configured outro and optionally verify its local media."""
    resolved = dict(_preset(style))
    resolved["duration_seconds"] = OUTRO_DURATION_SECONDS
    resolved["is_default"] = style == DEFAULT_OUTRO_STYLE
    resolved["background_path"] = OUTRO_LIBRARY_DIR / str(resolved["filename"])
    resolved["logo_path"] = OUTRO_LIBRARY_DIR / OUTRO_LOGO_FILENAME
    if verify_assets:
        _verified_asset(
            resolved["background_path"],
            str(resolved["sha256"]),
            f"Outro background {style}",
        )
        _verified_asset(resolved["logo_path"], OUTRO_LOGO_SHA256, "ByteFront logo")
    return resolved


def list_outro_presets() -> list[dict[str, Any]]:
    """Return the stable operator-facing catalog without exposing local paths."""
    rows: list[dict[str, Any]] = []
    for preset in OUTRO_PRESETS:
        resolved = resolve_outro_preset(str(preset["id"]), verify_assets=False)
        resolved.pop("background_path")
        resolved.pop("logo_path")
        resolved.pop("filename", None)
        resolved.pop("sha256", None)
        try:
            resolve_outro_preset(str(preset["id"]), verify_assets=True)
            resolved["available"] = True
        except RuntimeError:
            resolved["available"] = False
        rows.append(resolved)
    return rows


def collect_credits(task_root: Path, media_plans: list[dict]) -> list[dict[str, str]]:
    """Credit selected reporting and media actually placed, never unused candidates."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    def read(relative: str) -> dict:
        path = task_root / relative
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def add(kind: str, title: str, source: str, url: str = "", detail: str = "") -> None:
        key = url or f"{source}:{title}"
        if key in seen or not (title or source):
            return
        seen.add(key)
        rows.append(dict(kind=kind, title=title, source=source, url=url, detail=detail))

    dossier = read("research/dossier.json")
    for article in dossier.get("selected", []):
        source = str(article.get("source_name") or "")
        others = [s for s in article.get("corroborating_sources", []) if s != source]
        detail = str(article.get("published_at") or "")[:10]
        if others:
            detail += " · " + ", ".join(dict.fromkeys(others))
        add("NEWS", str(article.get("title") or ""), source,
            str(article.get("url") or article.get("evidence_url") or ""), detail.strip(" ·"))
    if not rows:
        for point in read("summary.json").get("talking_points", []):
            if isinstance(point, dict) and point.get("url"):
                add("NEWS", str(point.get("headline") or ""), str(point.get("source") or ""), str(point["url"]))

    used: set[str] = set()
    for plan in media_plans:
        shots = plan.get("media_shots") or plan.get("footage_sequence") or []
        used.update(str(shot.get("src") or "").removeprefix("../") for shot in shots)
        if not plan.get("media_shots"):
            used.add(str(plan.get("footage_src") or "").removeprefix("../"))
    for clip in read("footage/manifest.json").get("clips", []):
        if clip.get("local_path") in used:
            add("FOOTAGE", str(clip.get("title") or ""), str(clip.get("creator") or ""),
                str(clip.get("source_page_url") or ""))
    for plan in media_plans:
        if plan.get("news_webpage_src"):
            add("NEWS", str(plan.get("news_webpage_headline") or ""),
                str(plan.get("news_webpage_source") or ""), str(plan.get("news_webpage_url") or ""))
        if plan.get("news_image_src") or plan.get("news_image_srcs"):
            for credit in plan.get("news_image_credits") or [plan.get("news_image_credit")]:
                if credit:
                    add("IMAGE", str(credit), "")
    return rows


def credit_lines(row: dict[str, str]) -> list[str]:
    """Display concise attribution; the markup and manifest retain full permalinks."""
    return [s for s in [row.get("source", ""), row.get("title", ""),
                       row.get("detail", ""), urlsplit(row.get("url", "")).netloc] if s]


def credits_duration(rows: list[dict[str, str]], *, portrait: bool = False) -> float:
    # Conservative wrapping at 28px type, with reading holds at both ends.
    units, viewport = (24, 1000) if portrait else (40, 600)
    height = sum(68 + sum(36 * max(1, math.ceil(sum(
        2 if unicodedata.east_asian_width(c) in "WF" else 1.1 for c in line
    ) / units)) for line in credit_lines(row)) for row in rows)
    return min(MAX_CREDITS_OUTRO_SECONDS, round(6 + max(36, height - viewport) / 60, 2)) if rows else 0


def credits_markup(rows: list[dict[str, str]]) -> str:
    return "".join(
        f'<div class="outro-credit" data-credit-index="{i}" data-source-url="{html.escape(row.get("url", ""), quote=True)}">'
        + f'<div class="outro-credit-kind">{html.escape(row["kind"])} / {i + 1:02d}</div>'
        + "".join(f'<div class="outro-credit-line">{html.escape(line)}</div>' for line in credit_lines(row))
        + '</div>' for i, row in enumerate(rows)
    )


def _stage_background_hold(background: Path, destination: Path) -> None:
    """Keep the original plate's final picture after its finite video ends."""
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-sseof", "-0.08", "-i", str(background),
        "-frames:v", "1", str(destination),
    ], check=True, capture_output=True, timeout=60)
    if not destination.is_file():
        raise RuntimeError("Outro background did not produce a hold frame")


def stage_outro(
    task_dir: str | Path,
    storyboard: dict,
    style: str,
    media_plans: list[dict] | None = None,
    video_orientation: str = "landscape",
) -> dict[str, Any]:
    """Stage selected media and bind it to the spoken closing scene."""
    preset = resolve_outro_preset(style)
    task_root = Path(task_dir)
    asset_dir = task_root / "assets" / "outro"
    asset_dir.mkdir(parents=True, exist_ok=True)

    background_dest = asset_dir / "background.mp4"
    logo_dest = asset_dir / "bytefront-logo.png"
    shutil.copyfile(preset["background_path"], background_dest)
    shutil.copyfile(preset["logo_path"], logo_dest)

    scenes = list(storyboard.get("scenes") or [])
    closing = next(
        (scene for scene in reversed(scenes) if scene.get("program_segment_kind") == "closing"),
        scenes[-1] if scenes else None,
    )
    if closing is None:
        raise ValueError("storyboard has no timed scene for the branded outro")
    start = round(float(closing.get("start") or 0), 2)
    duration = round(float(closing.get("duration") or 0), 2)
    if duration <= 0:
        raise ValueError("spoken closing scene has no positive duration")
    credits = collect_credits(task_root, media_plans or [])
    spoken_duration = float(closing.get("spoken_closing_duration", duration))
    if credits and spoken_duration > MAX_CREDITS_OUTRO_SECONDS:
        raise ValueError("Spoken closing exceeds the 30-second outro budget; shorten the closing narration")
    if credits:
        _stage_background_hold(background_dest, asset_dir / "hold.png")
    closing["spoken_closing_duration"] = spoken_duration
    # Recompute from speech on retry, discarding a previously extended credit tail.
    duration = max(spoken_duration, credits_duration(credits, portrait=video_orientation == "portrait"))
    closing["duration"] = duration
    storyboard["total_duration"] = round(start + duration, 2)
    storyboard["credits_tail_duration"] = round(duration - spoken_duration, 2)
    (task_root / "outro_credits.json").write_text(json.dumps(
        {"start": start, "duration": duration, "spoken_duration": spoken_duration, "entries": credits},
        ensure_ascii=False, indent=2), encoding="utf-8")
    closing["scene_kind"] = "outro"
    closing["outro_style"] = style
    storyboard["outro_start"] = start
    storyboard["outro_duration"] = duration
    storyboard["outro_style"] = style
    closing_text = str(closing.get("text") or "").casefold()
    actions = ["LIKE", "COMMENT", "SHARE"]
    if "subscribe" in closing_text:
        actions.insert(0, "SUBSCRIBE")
    body = (
        "Stay curious. See you tomorrow morning."
        if "tomorrow morning" in closing_text
        else "Stay curious. Tomorrow's frontier arrives in one sharp shot."
    )

    return {
        "id": str(closing["id"]),
        "duration": duration,
        "archetype": "outro",
        "kicker": "SEE YOU IN THE NEXT SHOT",
        "headline": "THANKS FOR WATCHING",
        "body": body,
        "items": actions,
        "accent": "coral",
        "motif": "none",
        "footage_src": "assets/outro/background.mp4",
        "footage_kind": "video",
        "outro_logo_src": "assets/outro/bytefront-logo.png",
        "outro_style": style,
        "outro_label": preset["label"],
        "outro_credits": credits,
        "outro_hold_src": "assets/outro/hold.png" if credits else "",
        "grounding_source": "configured_outro",
    }


def outro_overlay_problems(text: str) -> list[str]:
    """Check the editable overlay contract after a director-agent rewrite."""
    problems: list[str] = []
    required = {
        'data-outro-role="brand"': "missing editable brand overlay",
        'data-outro-brand-part="espresso"': "missing prominent Espresso wordmark",
        'data-outro-role="thanks"': "missing editable closing-message overlay",
        'data-outro-role="actions"': "missing editable engagement-actions overlay",
        'data-outro-action="like"': "missing Like action",
        'data-outro-action="comment"': "missing Comment action",
        'data-outro-action="share"': "missing Share action",
        'data-bookend-layer="background" style="z-index:0"': "outro background stacking contract is missing",
        'data-bookend-layer="overlay" style="z-index:2"': "outro overlay stacking contract is missing",
    }
    for marker, message in required.items():
        if marker not in text:
            problems.append(message)
    if "感谢观看" in text:
        problems.append("the removed Chinese closing phrase is still present")
    return problems
