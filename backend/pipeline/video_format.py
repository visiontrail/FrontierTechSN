"""One source of truth for task-level video orientation and media sizes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from backend import config


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


@dataclass(frozen=True)
class RenderSpec:
    """Delivery pixels are independent of the composition's CSS coordinates."""

    resolution: str
    width: int
    height: int
    fps: int
    quality: str
    workers: str
    protocol_timeout_ms: int

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_render_spec(frame: FrameSpec = LANDSCAPE) -> RenderSpec:
    value = str(config.RENDER_RESOLUTION).strip().lower()
    if value in {"4k", "uhd", "landscape-4k", "portrait-4k", "square-4k"}:
        scale = 2
    elif value in {"1080p", "hd", "landscape", "portrait", "square"}:
        # Older installations saved orientation here. Task orientation remains
        # authoritative; these legacy values select 1080p output density only.
        scale = 1
    else:
        raise ValueError(f"Unsupported render resolution: {value}")
    return RenderSpec(
        resolution=frame.orientation + ("-4k" if scale == 2 else ""),
        width=frame.width * scale,
        height=frame.height * scale,
        fps=int(config.RENDER_FPS),
        quality=str(config.RENDER_QUALITY),
        workers=str(config.RENDER_WORKERS),
        protocol_timeout_ms=int(config.RENDER_PROTOCOL_TIMEOUT_MS),
    )


def resolve_frame_spec(orientation: str | None) -> FrameSpec:
    """Keep layout coordinates stable while preparing media for delivery."""
    frame = _SPECS.get(str(orientation or "").strip().lower(), LANDSCAPE)
    render = resolve_render_spec(frame)
    return replace(frame, media_width=render.width, media_height=render.height,
                   render_resolution=render.resolution)
