"""ByteFront Espresso system bookend library and render-task staging.

The selected preset owns the moving picture. HyperFrames owns the editable
brand, closing message, and engagement controls, which lets the video director
agent revise copy and layout without regenerating the background video.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from backend import config

DEFAULT_OUTRO_STYLE = "morning-brief"
OUTRO_DURATION_SECONDS = 6.0
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


def stage_outro(
    task_dir: str | Path,
    storyboard: dict,
    style: str,
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
