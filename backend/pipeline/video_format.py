"""One source of truth for task-level video orientation and media sizes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSpec:
    orientation: str
    aspect_ratio: str
    width: int
    height: int
    media_width: int
    media_height: int
    render_resolution: str

    @property
    def is_portrait(self) -> bool:
        return self.orientation == "portrait"


LANDSCAPE = FrameSpec(
    orientation="landscape",
    aspect_ratio="16:9",
    width=1920,
    height=1080,
    media_width=1280,
    media_height=720,
    render_resolution="landscape",
)

PORTRAIT = FrameSpec(
    orientation="portrait",
    aspect_ratio="9:16",
    width=1080,
    height=1920,
    media_width=720,
    media_height=1280,
    render_resolution="portrait",
)

_SPECS = {
    LANDSCAPE.orientation: LANDSCAPE,
    PORTRAIT.orientation: PORTRAIT,
}


def resolve_frame_spec(orientation: str | None) -> FrameSpec:
    """Resolve a task value, deliberately falling back to landscape."""
    return _SPECS.get(str(orientation or "").strip().lower(), LANDSCAPE)
