"""Draw a storyboard scene as a self-contained HyperFrames sub-composition.

Two consumers share this module:

* the deterministic fallback path, when the director agent is off or fails —
  every scene still gets real, on-topic artwork rather than a caption on black;
* the Claude Agent SDK director, which is handed these files as worked examples
  and is free to rewrite any of them.

Everything here is pure: the same plan produces byte-identical HTML. Motif
artwork is generated from a seeded PRNG *at build time* and baked into the SVG,
which keeps the composition free of the `Math.random()` the HyperFrames capture
engine forbids.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR

from backend.pipeline.video_format import FrameSpec, LANDSCAPE

# --- Design system ---------------------------------------------------------
# Dark editorial palette. Accents carry the emotional arc of an episode: the
# planner assigns one per scene, so consecutive scenes read as deliberate colour
# choreography rather than noise.

BG = "#0B0D17"
INK = "#F5F2EA"
MUTED = "#98A1BA"

ACCENTS: dict[str, str] = {
    "amber": "#F0A63C",
    "coral": "#E4572E",
    "teal": "#2EC4B6",
    "violet": "#8A6BFF",
    "rose": "#FF6B8A",
    "lime": "#A3D45C",
    "sky": "#4CC9F0",
}
DEFAULT_ACCENT = "amber"


@dataclass(frozen=True)
class Theme:
    """Surface colours for one video template.

    The task's ``video_template`` selects a theme; the art director's per-scene
    accent still rides on top, so a template changes the room without discarding
    the colour choreography.
    """

    name: str
    bg: str
    ink: str
    muted: str
    wash_alpha: float = 0.16
    caption_bg: str = "rgba(9,11,19,.62)"
    caption_ink: str = "rgba(245,242,234,.94)"
    accents: tuple[tuple[str, str], ...] = ()


THEMES: dict[str, Theme] = {
    # Deep navy-black, warm ink. The default documentary look.
    "podcast": Theme("podcast", "#0B0D17", "#F5F2EA", "#98A1BA"),
    # Higher-contrast, more saturated washes for statement-led cuts.
    "kinetic": Theme("kinetic", "#08070C", "#FFFFFF", "#A79FB8", wash_alpha=0.24),
    # Paper-white editorial grid.
    "swiss": Theme(
        "swiss", "#F4F1EA", "#14161F", "#5C6273",
        wash_alpha=0.12,
        caption_bg="rgba(20,22,31,.86)",
        caption_ink="rgba(248,246,240,.96)",
    ),
    # Near-black with restrained washes; type does the work.
    "minimal": Theme("minimal", "#0E0E10", "#EDEDED", "#8B8B92", wash_alpha=0.09),
    # Warm rice paper, layered ink-green terrain and ochre route marks. This
    # mirrors the visual language used by the project's X banner and avatar.
    "shanshui": Theme(
        "shanshui", "#F7F0E4", "#263A30", "#5E6D61",
        wash_alpha=0.13,
        caption_bg="rgba(247,240,228,.94)",
        caption_ink="rgba(38,58,48,.98)",
        accents=(
            ("amber", "#B46F35"),
            ("coral", "#A85F43"),
            ("teal", "#4E695A"),
            ("violet", "#70655C"),
            ("rose", "#9E6B61"),
            ("lime", "#747D59"),
            ("sky", "#708079"),
        ),
    ),
}
DEFAULT_THEME = THEMES["podcast"]


def resolve_theme(name: str | None) -> Theme:
    return THEMES.get((name or "").lower(), DEFAULT_THEME)

SANS = "Inter, 'Helvetica Neue', Arial, sans-serif"
SERIF = "Georgia, 'Times New Roman', serif"

MOTIFS = (
    "sunburst",
    "bloom",
    "skyline",
    "waves",
    "orbit",
    "grid",
    "arcs",
    "prism",
    "none",
)

ARCHETYPES = (
    "title",
    "statement",
    "topic",
    "contrast",
    "list",
    "stat",
    "quote",
    "footage",
    "news_image",
    "intro",
    "outro",
)

MAX_NARRATIVE_ITEMS = 5


def accent_hex(name: str | None, theme: Theme | None = None) -> str:
    key = (name or "").lower()
    themed = dict(theme.accents) if theme and theme.accents else {}
    return themed.get(key, ACCENTS.get(key, themed.get(DEFAULT_ACCENT, ACCENTS[DEFAULT_ACCENT])))


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:g})"


def _esc(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


class _Rand:
    """mulberry32, the seeded PRNG the HyperFrames docs recommend.

    Run at build time so motif geometry varies per scene but never per render.
    """

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + ((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF) & 0xFFFFFFFF
        return (((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0)

    def between(self, low: float, high: float) -> float:
        return low + (high - low) * self.next()


def _seed_from(text: str) -> int:
    seed = 2166136261
    for ch in text:
        seed = ((seed ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return seed


# --- Motif artwork ---------------------------------------------------------
# Each builder returns SVG markup sized to a 600x600 viewBox. The scene layouts
# place that box; the art never has to know where it lives.


def _motif_sunburst(rnd: _Rand, color: str) -> str:
    rays = []
    count = 28
    for i in range(count):
        angle = (i / count) * math.tau
        inner = 128 + rnd.between(0, 14)
        outer = inner + rnd.between(60, 168)
        width = 0.012 + rnd.next() * 0.016
        x1, y1 = 300 + inner * math.cos(angle), 300 + inner * math.sin(angle)
        x2, y2 = 300 + outer * math.cos(angle), 300 + outer * math.sin(angle)
        x3 = 300 + outer * math.cos(angle + width)
        y3 = 300 + outer * math.sin(angle + width)
        rays.append(
            f'<polygon class="ray" points="{x1:.1f},{y1:.1f} {x2:.1f},{y2:.1f} {x3:.1f},{y3:.1f}" '
            f'fill="{_rgba(color, 0.55)}"/>'
        )
    return (
        "".join(rays)
        + f'<circle class="core" cx="300" cy="300" r="112" fill="{color}" opacity="0.92"/>'
        + f'<circle class="core-ring" cx="300" cy="300" r="150" fill="none" '
        f'stroke="{_rgba(color, 0.4)}" stroke-width="2"/>'
    )


def _motif_bloom(rnd: _Rand, color: str) -> str:
    petals = []
    for ring in range(3):
        count = 8 + ring * 4
        radius = 92 + ring * 74
        size = 46 - ring * 8
        for i in range(count):
            angle = (i / count) * math.tau + ring * 0.24
            cx, cy = 300 + radius * math.cos(angle), 300 + radius * math.sin(angle)
            petals.append(
                f'<ellipse class="petal" cx="{cx:.1f}" cy="{cy:.1f}" rx="{size:.0f}" ry="{size * 0.42:.0f}" '
                f'transform="rotate({math.degrees(angle):.1f} {cx:.1f} {cy:.1f})" '
                f'fill="{_rgba(color, 0.22 + ring * 0.12)}"/>'
            )
    return "".join(petals) + f'<circle cx="300" cy="300" r="52" fill="{color}"/>'


def _motif_skyline(rnd: _Rand, color: str) -> str:
    bars = []
    x = 40
    while x < 560:
        width = rnd.between(26, 58)
        height = rnd.between(90, 380)
        bars.append(
            f'<rect class="bar" x="{x:.0f}" y="{520 - height:.0f}" width="{width:.0f}" height="{height:.0f}" '
            f'rx="4" fill="{_rgba(color, 0.28 + rnd.next() * 0.5)}"/>'
        )
        x += width + rnd.between(10, 24)
    return "".join(bars) + f'<rect x="30" y="520" width="540" height="3" fill="{_rgba(color, 0.7)}"/>'


def _motif_waves(rnd: _Rand, color: str) -> str:
    paths = []
    for line in range(6):
        amp = rnd.between(26, 62)
        phase = rnd.between(0, math.tau)
        y0 = 120 + line * 62
        points = []
        for step in range(0, 61):
            x = step * 10
            y = y0 + amp * math.sin(phase + step / 7.5)
            points.append(f"{x},{y:.1f}")
        paths.append(
            f'<polyline class="wave" points="{" ".join(points)}" fill="none" '
            f'stroke="{_rgba(color, 0.25 + line * 0.11)}" stroke-width="{2 + line * 0.7:.1f}" '
            f'stroke-linecap="round"/>'
        )
    return "".join(paths)


def _motif_orbit(rnd: _Rand, color: str) -> str:
    parts = []
    for i in range(4):
        r = 84 + i * 62
        parts.append(
            f'<circle class="orbit-ring" cx="300" cy="300" r="{r}" fill="none" '
            f'stroke="{_rgba(color, 0.42 - i * 0.07)}" stroke-width="2"/>'
        )
        angle = rnd.between(0, math.tau)
        cx, cy = 300 + r * math.cos(angle), 300 + r * math.sin(angle)
        parts.append(
            f'<circle class="orbit-dot" cx="{cx:.1f}" cy="{cy:.1f}" r="{13 - i * 2}" fill="{color}"/>'
        )
    return "".join(parts) + f'<circle cx="300" cy="300" r="30" fill="{_rgba(color, 0.9)}"/>'


def _motif_grid(rnd: _Rand, color: str) -> str:
    dots = []
    for row in range(11):
        for col in range(11):
            x, y = 60 + col * 48, 60 + row * 48
            weight = rnd.next()
            radius = 3 + weight * 9
            dots.append(
                f'<circle class="dot" cx="{x}" cy="{y}" r="{radius:.1f}" '
                f'fill="{_rgba(color, 0.12 + weight * 0.62)}"/>'
            )
    return "".join(dots)


def _motif_arcs(rnd: _Rand, color: str) -> str:
    arcs = []
    for i in range(5):
        r = 90 + i * 58
        start = rnd.between(0, math.tau)
        sweep = rnd.between(1.4, 3.6)
        x1, y1 = 300 + r * math.cos(start), 300 + r * math.sin(start)
        x2, y2 = 300 + r * math.cos(start + sweep), 300 + r * math.sin(start + sweep)
        large = 1 if sweep > math.pi else 0
        arcs.append(
            f'<path class="arc" d="M {x1:.1f} {y1:.1f} A {r} {r} 0 {large} 1 {x2:.1f} {y2:.1f}" '
            f'fill="none" stroke="{_rgba(color, 0.3 + i * 0.13)}" stroke-width="{4 + i * 2}" '
            f'stroke-linecap="round"/>'
        )
    return "".join(arcs)


def _motif_prism(rnd: _Rand, color: str) -> str:
    shapes = []
    for i in range(4):
        cx, cy = rnd.between(200, 400), rnd.between(200, 400)
        size = rnd.between(120, 230)
        rot = rnd.between(0, 90)
        shapes.append(
            f'<rect class="prism" x="{cx - size / 2:.0f}" y="{cy - size / 2:.0f}" '
            f'width="{size:.0f}" height="{size:.0f}" rx="18" '
            f'transform="rotate({rot:.0f} {cx:.0f} {cy:.0f})" '
            f'fill="{_rgba(color, 0.16)}" stroke="{_rgba(color, 0.5)}" stroke-width="2"/>'
        )
    return "".join(shapes)


_MOTIF_BUILDERS = {
    "sunburst": _motif_sunburst,
    "bloom": _motif_bloom,
    "skyline": _motif_skyline,
    "waves": _motif_waves,
    "orbit": _motif_orbit,
    "grid": _motif_grid,
    "arcs": _motif_arcs,
    "prism": _motif_prism,
}


def render_motif(
    motif: str, accent: str, seed_text: str, theme: Theme | None = None
) -> str:
    """Inline SVG for ``motif``, or an empty string for ``none``/unknown."""
    builder = _MOTIF_BUILDERS.get((motif or "").lower())
    if builder is None:
        return ""
    color = accent_hex(accent, theme)
    svg = builder(_Rand(_seed_from(seed_text)), color)
    return (
        '<svg class="motif-svg" viewBox="0 0 600 600" width="100%" height="100%" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">{svg}</svg>'
    )


# --- Type scale ------------------------------------------------------------


def headline_size(text: str, *, base: int = 118, floor: int = 56) -> int:
    """Shrink a headline as it lengthens so it never overruns the safe area.

    Rendered video has no scrollbar to bail us out, and `hyperframes inspect`
    reports overflow as a hard finding, so the size is derived from length
    rather than left to chance.
    """
    n = len(text or "")
    if n <= 22:
        return base
    if n <= 40:
        return int(base * 0.82)
    if n <= 64:
        return int(base * 0.66)
    if n <= 96:
        return int(base * 0.54)
    return floor


def body_size(text: str, *, base: int = 42, floor: int = 28) -> int:
    n = len(text or "")
    if n <= 90:
        return base
    if n <= 180:
        return int(base * 0.86)
    if n <= 300:
        return int(base * 0.74)
    return floor


@dataclass
class ScenePlan:
    """Normalised creative direction for one storyboard scene."""

    id: str
    duration: float
    archetype: str = "topic"
    kicker: str = ""
    headline: str = ""
    body: str = ""
    items: tuple[str, ...] = ()
    quote: str = ""
    attribution: str = ""
    stat: str = ""
    stat_label: str = ""
    left_label: str = ""
    left_text: str = ""
    right_label: str = ""
    right_text: str = ""
    motif: str = "orbit"
    accent: str = DEFAULT_ACCENT
    footage_src: str = ""
    footage_kind: str = ""
    footage_credit: str = ""
    footage_sequence: tuple[dict, ...] = ()
    media_shots: tuple[dict, ...] = ()
    collage_broll: bool = False
    collage_hold_src: str = ""
    collage_target_duration_seconds: float = 0.0
    news_image_src: str = ""
    news_image_srcs: tuple[str, ...] = ()
    news_image_mode: str = ""
    news_image_kind: str = ""
    news_image_fit: str = "cover"
    news_image_fits: tuple[str, ...] = ()
    news_image_credit: str = ""
    news_image_credits: tuple[str, ...] = ()
    news_image_caption: str = ""
    news_webpage_src: str = ""
    news_webpage_url: str = ""
    news_webpage_source: str = ""
    news_webpage_headline: str = ""
    intro_logo_src: str = ""
    intro_style: str = ""
    edition_date: str = ""
    edition_weekday: str = ""
    outro_logo_src: str = ""
    outro_style: str = ""
    outro_credits: tuple[dict[str, str], ...] = ()
    outro_hold_src: str = ""
    theme: Theme = DEFAULT_THEME
    frame: FrameSpec = LANDSCAPE

    @classmethod
    def from_dict(
        cls,
        data: dict,
        *,
        duration: float,
        scene_id: str,
        theme: Theme = DEFAULT_THEME,
        frame: FrameSpec = LANDSCAPE,
    ) -> "ScenePlan":
        left = data.get("left") or {}
        right = data.get("right") or {}
        archetype = (data.get("archetype") or "topic").lower()
        if archetype not in ARCHETYPES:
            archetype = "topic"
        motif = (data.get("motif") or "orbit").lower()
        if motif not in MOTIFS:
            motif = "orbit"
        items = tuple(
            str(item)
            for item in (data.get("items") or [])[:MAX_NARRATIVE_ITEMS]
            if str(item).strip()
        )
        return cls(
            id=scene_id,
            duration=duration,
            archetype=archetype,
            kicker=str(data.get("kicker") or ""),
            headline=str(data.get("headline") or ""),
            body=str(data.get("body") or ""),
            items=items,
            quote=str(data.get("quote") or ""),
            attribution=str(data.get("attribution") or ""),
            stat=str(data.get("stat") or ""),
            stat_label=str(data.get("stat_label") or ""),
            left_label=str(left.get("label") or ""),
            left_text=str(left.get("text") or ""),
            right_label=str(right.get("label") or ""),
            right_text=str(right.get("text") or ""),
            motif=motif,
            accent=(data.get("accent") or DEFAULT_ACCENT).lower(),
            footage_src=str(data.get("footage_src") or ""),
            footage_kind=str(data.get("footage_kind") or ""),
            footage_credit=str(data.get("footage_credit") or ""),
            footage_sequence=tuple(
                dict(item)
                for item in data.get("footage_sequence") or []
                if isinstance(item, dict) and str(item.get("src") or "").strip()
            ),
            media_shots=tuple(dict(item) for item in data.get("media_shots") or []),
            collage_broll=bool(data.get("collage_broll")),
            collage_hold_src=str(data.get("collage_hold_src") or ""),
            collage_target_duration_seconds=float(
                data.get("collage_target_duration_seconds") or 0.0
            ),
            news_image_src=str(data.get("news_image_src") or ""),
            news_image_srcs=tuple(
                str(item)
                for item in data.get("news_image_srcs") or []
                if str(item).strip()
            ),
            news_image_mode=str(data.get("news_image_mode") or ""),
            news_image_kind=str(data.get("news_image_kind") or ""),
            news_image_fit=(
                str(data.get("news_image_fit") or "cover")
                if str(data.get("news_image_fit") or "cover") in {"cover", "contain"}
                else "cover"
            ),
            news_image_credit=str(data.get("news_image_credit") or ""),
            news_image_fits=tuple(
                "contain" if value == "contain" else "cover"
                for value in data.get("news_image_fits") or []
            ),
            news_image_credits=tuple(
                str(item)
                for item in data.get("news_image_credits") or []
                if str(item).strip()
            ),
            news_image_caption=str(data.get("news_image_caption") or ""),
            news_webpage_src=str(data.get("news_webpage_src") or ""),
            news_webpage_url=str(data.get("news_webpage_url") or ""),
            news_webpage_source=str(data.get("news_webpage_source") or ""),
            news_webpage_headline=str(data.get("news_webpage_headline") or ""),
            intro_logo_src=str(data.get("intro_logo_src") or ""),
            intro_style=str(data.get("intro_style") or ""),
            edition_date=str(data.get("edition_date") or ""),
            edition_weekday=str(data.get("edition_weekday") or ""),
            outro_logo_src=str(data.get("outro_logo_src") or ""),
            outro_style=str(data.get("outro_style") or ""),
            outro_credits=tuple(data.get("outro_credits") or []),
            outro_hold_src=str(data.get("outro_hold_src") or ""),
            theme=theme,
            frame=frame,
        )


# --- Scene rendering -------------------------------------------------------

_BASE_CSS = """
  #{sid} {{ position:relative; width:{width}px; height:{height}px; overflow:hidden; background:{bg}; }}
  #{sid} .plate {{ position:absolute; inset:0; }}
  #{sid} .wash {{ position:absolute; inset:0;
      background: radial-gradient(1500px 1000px at {wx}% {wy}%, {glow} 0%, {bg} 70%); }}
  #{sid} .vignette {{ position:absolute; inset:0;
      background: radial-gradient(circle at 50% 50%, rgba(0,0,0,0) 45%, rgba(0,0,0,.55) 100%); }}
  #{sid} .stage {{ position:relative; width:100%; height:100%;
      padding:130px 150px 210px; box-sizing:border-box;
      display:flex; flex-direction:column; justify-content:center; gap:26px; }}
  #{sid} .kicker {{ font:600 28px {sans}; letter-spacing:.22em; text-transform:uppercase;
      color:{accent}; display:flex; align-items:center; gap:18px; }}
  #{sid} .kicker::before {{ content:""; width:56px; height:3px; background:{accent}; border-radius:2px; }}
  #{sid} .headline {{ font-family:{sans}; font-weight:700; line-height:1.06;
      color:{ink}; letter-spacing:-0.015em; }}
  #{sid} .body {{ font-family:{sans}; font-weight:400; line-height:1.5; color:{muted}; }}
  #{sid} .motif {{ position:absolute; pointer-events:none; }}
"""


def _portrait_css(plan: ScenePlan) -> str:
    """Keep deterministic layouts inside a 9:16 safe area.

    Full-bleed generated footage needs no special casing; the rules below only
    reflow the typography-heavy fallback archetypes when a portrait task is
    requested.
    """
    if not plan.frame.is_portrait:
        return ""
    sid = plan.id
    return f"""
  #{sid} .stage {{ padding:190px 92px 330px; gap:34px; }}
  #{sid} .headline {{ max-width:896px; font-size:min(92px, 9.2vw); line-height:1.08; }}
  #{sid} .body {{ max-width:850px; font-size:min(42px, 4.2vw); }}
  #{sid} .kicker {{ font-size:25px; }}
  #{sid} .motif {{ opacity:.24 !important; }}
  #{sid} .cols {{ flex-direction:column; gap:26px; }}
  #{sid} .col {{ padding:34px 36px; }}
  #{sid} .col-text {{ font-size:34px; }}
  #{sid} .vs {{ align-self:flex-start; }}
  #{sid} .rows {{ gap:28px; }}
  #{sid} .row-text {{ max-width:760px; font-size:39px; }}
  #{sid} .figure {{ font-size:min(210px, 21vw); }}
  #{sid} .stat-label {{ max-width:820px; font-size:38px; }}
  #{sid} .quote {{ max-width:850px; font-size:min(68px, 6.8vw); }}
  #{sid} .frame, #{sid} .media {{ width:100%; height:100%; }}
  #{sid} .credit {{ right:34px; top:42px; max-width:820px; }}
  #{sid} .news-inline-stage {{ flex-direction:column; align-items:stretch; padding:170px 92px 330px; }}
  #{sid} .news-inline-copy {{ width:100%; max-width:880px; }}
  #{sid} .news-inline-visual {{ width:100%; min-height:720px; }}
  #{sid} .news-inline-window {{ width:100%; height:720px; }}
  #{sid} .news-full-stage {{ padding:190px 92px 330px; }}
  #{sid} .news-full-headline {{ max-width:850px; }}
  #{sid} .collage-caption {{ left:72px; right:72px; bottom:330px; width:auto; max-width:none;
      padding:26px 30px 28px; }}
  #{sid} .collage-headline {{ font-size:min(52px, 5.2vw); }}
"""


def _shanshui_backdrop(plan: ScenePlan) -> str:
    """Full-frame paper-and-terrain artwork for the Shan Shui template.

    The geometry is baked at build time. Small seeded offsets keep consecutive
    scenes related without making every frame a duplicate of the banner.
    """
    rnd = _Rand(_seed_from(plan.id + plan.headline))
    dx = int(rnd.between(-70, 55))
    dy = int(rnd.between(-22, 28))
    noise_seed = _seed_from(plan.id) % 89 + 1
    sid = plan.id
    return f"""    <svg class="shanshui-backdrop" data-layout-ignore viewBox="0 0 1920 1080"
         xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <defs>
        <filter id="{sid}-paper-grain" x="0" y="0" width="100%" height="100%">
          <feTurbulence type="fractalNoise" baseFrequency=".72" numOctaves="3" seed="{noise_seed}"/>
          <feColorMatrix type="saturate" values="0"/>
          <feComponentTransfer><feFuncA type="table" tableValues="0 .22"/></feComponentTransfer>
        </filter>
      </defs>
      <g class="shanshui-terrain" transform="translate({dx} {dy})">
        <path d="M120 1080 C330 958 486 1002 646 878 C808 752 864 790 1010 664 C1172 524 1308 548 1428 384 C1554 212 1742 254 1990 100 L1990 1080Z" fill="#E9D9BB" opacity=".56"/>
        <path d="M470 1080 C650 956 748 970 852 832 C974 670 1092 720 1204 582 C1338 416 1490 484 1602 326 C1722 158 1836 174 1990 72 L1990 1080Z" fill="#D4B17B" opacity=".66"/>
        <path d="M700 1080 C842 926 956 976 1060 810 C1178 622 1304 690 1412 524 C1532 340 1652 420 1742 250 C1802 138 1884 100 1990 66 L1990 1080Z" fill="#7A8A75" opacity=".74"/>
        <path d="M900 1080 C1040 930 1148 984 1242 828 C1350 648 1446 710 1544 558 C1660 378 1750 432 1826 276 C1874 178 1932 136 1990 118 L1990 1080Z" fill="#4E6656" opacity=".9"/>
        <path d="M1120 1080 C1248 938 1350 982 1430 840 C1526 670 1628 724 1708 586 C1796 432 1876 454 1990 314 L1990 1080Z" fill="#263A30" opacity=".96"/>
        <g fill="none" stroke="#D9C9A8" stroke-width="1.6" opacity=".34">
          <path d="M1090 1030 C1260 910 1362 948 1470 814 C1586 670 1720 692 1960 470"/>
          <path d="M1060 988 C1240 866 1354 916 1460 774 C1580 616 1724 656 1970 422"/>
          <path d="M1040 946 C1228 826 1340 864 1444 728 C1566 568 1714 612 1978 380"/>
          <path d="M1000 900 C1196 786 1320 814 1428 680 C1550 526 1706 564 1982 336"/>
          <path d="M970 852 C1170 744 1300 764 1408 634 C1530 486 1688 516 1988 292"/>
          <path d="M938 806 C1140 700 1270 716 1388 588 C1510 444 1676 476 1992 248"/>
        </g>
      </g>
      <g class="shanshui-routes" fill="none" stroke="#B46F35" stroke-width="2.1" opacity=".62">
        <path d="M-40 968 C300 598 632 830 934 620 C1192 440 1308 158 1662 198 C1812 216 1902 296 1970 378"/>
        <path d="M1032 1100 C1258 770 1322 522 1578 480 C1710 458 1832 478 1990 438"/>
      </g>
      <g class="shanshui-nodes" fill="#B46F35" opacity=".92">
        <circle cx="934" cy="620" r="10"/><circle cx="1662" cy="198" r="10"/>
        <circle cx="1578" cy="480" r="10"/><circle cx="1868" cy="462" r="9"/>
      </g>
      <rect width="1920" height="1080" fill="#8C6A42" opacity=".12" filter="url(#{sid}-paper-grain)"/>
    </svg>
"""


def _shanshui_css(plan: ScenePlan) -> str:
    if plan.theme.name != "shanshui":
        return ""
    sid = plan.id
    return f"""
  #{sid} .plate {{ z-index:0; background-image:radial-gradient(circle at 18% 18%, rgba(255,255,255,.5), transparent 38%), repeating-linear-gradient(8deg, rgba(117,87,48,.025) 0 1px, transparent 1px 7px); }}
  #{sid} .wash {{ background:radial-gradient(1200px 820px at 24% 34%, rgba(255,250,241,.82) 0%, rgba(247,240,228,.28) 58%, rgba(247,240,228,0) 100%); }}
  #{sid} .shanshui-backdrop {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; z-index:1; }}
  #{sid} .motif {{ z-index:2; mix-blend-mode:multiply; filter:saturate(.62); }}
  #{sid} .motif-svg .arc {{ stroke-width:2px; opacity:.3; }}
  #{sid} .stage {{ z-index:3; }}
  #{sid} .vignette {{ z-index:4; background:radial-gradient(circle at 46% 44%, rgba(255,255,255,0) 52%, rgba(111,78,42,.11) 100%); }}
  #{sid} .headline {{ font-family:Georgia, 'Times New Roman', serif; font-weight:700; letter-spacing:-.025em; text-shadow:0 2px 0 rgba(255,255,255,.24); }}
  #{sid} .figure {{ font-family:Georgia, 'Times New Roman', serif; font-weight:700; }}
  #{sid} .quote {{ font-family:Georgia, 'Times New Roman', serif; }}
  #{sid} .kicker {{ color:#9B5D2F; }}
  #{sid} .kicker::before {{ background:#B46F35; height:2px; }}
  #{sid} .col {{ background:rgba(247,240,228,.72); border:2px solid rgba(78,105,90,.22); border-radius:5px; box-shadow:0 18px 60px rgba(75,55,33,.08); }}
  #{sid} .vs {{ color:#526B5B; }}
  #{sid} .rule {{ height:3px; }}
"""


def _shell(plan: ScenePlan, *, css: str, markup: str, timeline: str, wash: tuple[int, int]) -> str:
    """Wrap scene-specific CSS/markup/timeline in the sub-composition envelope."""
    accent = accent_hex(plan.accent, plan.theme)
    theme = plan.theme
    base = _BASE_CSS.format(
        sid=plan.id,
        width=plan.frame.width,
        height=plan.frame.height,
        bg=theme.bg,
        ink=theme.ink,
        muted=theme.muted,
        accent=accent,
        sans=SANS,
        glow=_rgba(accent, theme.wash_alpha),
        wx=wash[0],
        wy=wash[1],
    )
    themed_css = _shanshui_css(plan)
    portrait_css = _portrait_css(plan)
    themed_backdrop = _shanshui_backdrop(plan) if theme.name == "shanshui" else ""
    themed_timeline = ""
    if theme.name == "shanshui":
        drift_duration = max(2.0, plan.duration - 0.3)
        themed_timeline = f'''        inAt("#{plan.id} .shanshui-backdrop", {{ opacity: 0 }}, {{ opacity: 1, duration: 1.25, ease: "sine.out" }}, 0.1);
        inAt("#{plan.id} .shanshui-terrain", {{ x: 18 }}, {{ x: 0, duration: {drift_duration:.2f}, ease: "none" }}, 0.15);
        inAt("#{plan.id} .shanshui-routes path", {{ strokeDasharray: 1200, strokeDashoffset: 1200 }}, {{ strokeDashoffset: 0, duration: 2.15, ease: "power2.out", stagger: 0.16 }}, 0.24);
        inAt("#{plan.id} .shanshui-nodes circle", {{ scale: .25, opacity: 0 }}, {{ scale: 1, opacity: 1, duration: .58, ease: "back.out(1.8)", stagger: 0.17, transformOrigin: "50% 50%" }}, 0.72);
'''
    return f"""<template id="{plan.id}-template">
  <div id="{plan.id}" data-composition-id="{plan.id}" data-width="{plan.frame.width}" data-height="{plan.frame.height}">
    <style>
{base}{css}{themed_css}{portrait_css}
    </style>
{themed_backdrop}
{markup}
    <script src="../vendor/gsap.min.js"></script>
    <script>
      window.__timelines = window.__timelines || {{}};
      (function () {{
        const tl = gsap.timeline({{ paused: true }});
        const q = gsap.utils.selector("#{plan.id}");
        // Optional elements are omitted from the markup when unused; skip their
        // tweens rather than handing GSAP an empty target.
        const inAt = (sel, from, to, at) => {{ const el = q(sel); if (el.length) tl.fromTo(el, from, to, at); }};
        const outAt = (sel, to, at) => {{ const el = q(sel); if (el.length) tl.to(el, to, at); }};
{themed_timeline}{timeline}
        window.__timelines["{plan.id}"] = tl;
      }})();
    </script>
  </div>
</template>
"""


def _motif_block(plan: ScenePlan, *, style: str) -> str:
    art = render_motif(plan.motif, plan.accent, plan.id + plan.headline, plan.theme)
    if not art:
        return ""
    return (
        f'    <div class="motif" id="{plan.id}-motif" data-layout-ignore '
        f'style="{style}">{art}</div>\n'
    )


def _render_topic(plan: ScenePlan) -> str:
    hsize = headline_size(plan.headline, base=104)
    bsize = body_size(plan.body)
    css = f"""
  #{plan.id} .headline {{ font-size:{hsize}px; max-width:1080px; }}
  #{plan.id} .body {{ font-size:{bsize}px; max-width:960px; }}
"""
    markup = (
        '    <div class="plate"><div class="wash"></div></div>\n'
        + _motif_block(plan, style="right:90px; top:50%; width:560px; height:560px; margin-top:-280px; opacity:.85;")
        + '    <div class="stage">\n'
        + (f'      <div class="kicker" id="{plan.id}-kicker">{_esc(plan.kicker)}</div>\n' if plan.kicker else "")
        + f'      <div class="headline" id="{plan.id}-head">{_esc(plan.headline)}</div>\n'
        + (f'      <div class="body" id="{plan.id}-body">{_esc(plan.body)}</div>\n' if plan.body else "")
        + '    </div>\n'
        + '    <div class="vignette"></div>\n'
    )
    timeline = f"""        inAt("#{plan.id}-kicker", {{ x: -40, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .55, ease: "power3.out" }}, 0.15);
        inAt("#{plan.id}-head", {{ y: 64, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .85, ease: "expo.out" }}, 0.3);
        inAt("#{plan.id}-body", {{ y: 30, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .7, ease: "power2.out" }}, 0.55);
        inAt("#{plan.id}-motif", {{ scale: .78, opacity: 0, rotate: -8 }}, {{ scale: 1, opacity: .85, rotate: 0, duration: 1.5, ease: "power2.out", transformOrigin: "50% 50%" }}, 0.2);
        outAt("#{plan.id}-motif", {{ rotate: 6, duration: Math.max(2, {plan.duration:.2f} - 1.6), ease: "none" }}, 1.5);
"""
    return _shell(plan, css=css, markup=markup, timeline=timeline, wash=(74, 46))


def _render_statement(plan: ScenePlan) -> str:
    words = [w for w in (plan.headline or "").split() if w]
    hsize = headline_size(plan.headline, base=132, floor=62)
    spans = "".join(
        f'<span class="w" style="display:inline-block;">{_esc(w)}</span>{" " if i < len(words) - 1 else ""}'
        for i, w in enumerate(words)
    )
    css = f"""
  #{plan.id} .stage {{ align-items:flex-start; }}
  #{plan.id} .headline {{ font-size:{hsize}px; max-width:1560px; }}
  #{plan.id} .rule {{ width:180px; height:6px; background:{accent_hex(plan.accent, plan.theme)}; border-radius:3px; }}
"""
    markup = (
        '    <div class="plate"><div class="wash"></div></div>\n'
        + _motif_block(plan, style="left:50%; top:50%; width:1100px; height:1100px; margin:-550px 0 0 -550px; opacity:.16;")
        + '    <div class="stage">\n'
        + f'      <div class="rule" id="{plan.id}-rule"></div>\n'
        + f'      <div class="headline" id="{plan.id}-head">{spans}</div>\n'
        + '    </div>\n'
        + '    <div class="vignette"></div>\n'
    )
    timeline = f"""        inAt("#{plan.id}-rule", {{ scaleX: 0, transformOrigin: "0 50%" }}, {{ scaleX: 1, duration: .6, ease: "power3.out" }}, 0.15);
        inAt("#{plan.id}-head .w", {{ y: 70, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .7, ease: "expo.out", stagger: 0.055 }}, 0.3);
        inAt("#{plan.id}-motif", {{ scale: 1.15, opacity: 0 }}, {{ scale: 1, opacity: .16, duration: 1.8, ease: "power2.out", transformOrigin: "50% 50%" }}, 0.1);
"""
    return _shell(plan, css=css, markup=markup, timeline=timeline, wash=(30, 62))


def _render_contrast(plan: ScenePlan) -> str:
    accent = accent_hex(plan.accent, plan.theme)
    css = f"""
  #{plan.id} .stage {{ padding:120px 130px 200px; gap:34px; }}
  #{plan.id} .headline {{ font-size:{headline_size(plan.headline, base=76)}px; max-width:1500px; }}
  #{plan.id} .cols {{ display:flex; gap:44px; width:100%; }}
  #{plan.id} .col {{ flex:1; padding:44px 46px; border-radius:22px;
      background:rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.09);
      display:flex; flex-direction:column; gap:18px; }}
  #{plan.id} .col.b {{ background:{_rgba(accent, 0.12)}; border-color:{_rgba(accent, 0.42)}; }}
  #{plan.id} .col-label {{ font:600 26px {SANS}; letter-spacing:.16em; text-transform:uppercase; color:{plan.theme.muted}; }}
  #{plan.id} .col.b .col-label {{ color:{accent}; }}
  #{plan.id} .col-text {{ font:500 40px {SANS}; line-height:1.34; color:{plan.theme.ink}; }}
  #{plan.id} .vs {{ align-self:center; font:700 30px {SANS}; color:{plan.theme.muted}; letter-spacing:.2em; }}
"""
    markup = (
        '    <div class="plate"><div class="wash"></div></div>\n'
        + '    <div class="stage">\n'
        + (f'      <div class="kicker" id="{plan.id}-kicker">{_esc(plan.kicker)}</div>\n' if plan.kicker else "")
        + (f'      <div class="headline" id="{plan.id}-head">{_esc(plan.headline)}</div>\n' if plan.headline else "")
        + '      <div class="cols">\n'
        + f'        <div class="col a" id="{plan.id}-left">\n'
        + f'          <div class="col-label">{_esc(plan.left_label or "Before")}</div>\n'
        + f'          <div class="col-text">{_esc(plan.left_text)}</div>\n'
        + '        </div>\n'
        + f'        <div class="vs" id="{plan.id}-vs">VS</div>\n'
        + f'        <div class="col b" id="{plan.id}-right">\n'
        + f'          <div class="col-label">{_esc(plan.right_label or "Now")}</div>\n'
        + f'          <div class="col-text">{_esc(plan.right_text)}</div>\n'
        + '        </div>\n'
        + '      </div>\n'
        + '    </div>\n'
        + '    <div class="vignette"></div>\n'
    )
    timeline = f"""        inAt("#{plan.id}-kicker", {{ x: -36, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .5, ease: "power3.out" }}, 0.15);
        inAt("#{plan.id}-head", {{ y: 44, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .7, ease: "expo.out" }}, 0.3);
        inAt("#{plan.id}-left", {{ x: -70, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .75, ease: "power3.out" }}, 0.5);
        inAt("#{plan.id}-right", {{ x: 70, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .75, ease: "power3.out" }}, 0.62);
        inAt("#{plan.id}-vs", {{ scale: .4, opacity: 0 }}, {{ scale: 1, opacity: 1, duration: .6, ease: "back.out(2)", transformOrigin: "50% 50%" }}, 0.85);
"""
    return _shell(plan, css=css, markup=markup, timeline=timeline, wash=(50, 30))


def _render_list(plan: ScenePlan) -> str:
    accent = accent_hex(plan.accent, plan.theme)
    rows = "".join(
        f'        <div class="row" id="{plan.id}-row-{i}">'
        f'<div class="num">{i + 1:02d}</div>'
        f'<div class="row-text">{_esc(item)}</div></div>\n'
        for i, item in enumerate(plan.items)
    )
    css = f"""
  #{plan.id} .headline {{ font-size:{headline_size(plan.headline, base=82)}px; max-width:1440px; }}
  #{plan.id} .rows {{ display:flex; flex-direction:column; gap:26px; margin-top:18px; }}
  #{plan.id} .row {{ display:flex; align-items:baseline; gap:30px; }}
  #{plan.id} .num {{ font:700 40px {SANS}; color:{accent}; font-variant-numeric:tabular-nums;
      min-width:82px; }}
  #{plan.id} .row-text {{ font:500 46px {SANS}; line-height:1.32; color:{plan.theme.ink}; max-width:1200px; }}
"""
    markup = (
        '    <div class="plate"><div class="wash"></div></div>\n'
        + _motif_block(plan, style="right:70px; bottom:180px; width:440px; height:440px; opacity:.32;")
        + '    <div class="stage">\n'
        + (f'      <div class="kicker" id="{plan.id}-kicker">{_esc(plan.kicker)}</div>\n' if plan.kicker else "")
        + (f'      <div class="headline" id="{plan.id}-head">{_esc(plan.headline)}</div>\n' if plan.headline else "")
        + '      <div class="rows">\n'
        + rows
        + '      </div>\n'
        + '    </div>\n'
        + '    <div class="vignette"></div>\n'
    )
    timeline = f"""        inAt("#{plan.id}-kicker", {{ x: -36, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .5, ease: "power3.out" }}, 0.15);
        inAt("#{plan.id}-head", {{ y: 48, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .75, ease: "expo.out" }}, 0.28);
        inAt("#{plan.id} .row", {{ x: 54, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .6, ease: "power3.out", stagger: 0.22 }}, 0.55);
        inAt("#{plan.id}-motif", {{ scale: .8, opacity: 0 }}, {{ scale: 1, opacity: .32, duration: 1.4, ease: "power2.out", transformOrigin: "50% 50%" }}, 0.3);
"""
    return _shell(plan, css=css, markup=markup, timeline=timeline, wash=(24, 70))


def _render_stat(plan: ScenePlan) -> str:
    figure = plan.stat or plan.headline
    size = 300 if len(figure) <= 4 else (200 if len(figure) <= 8 else 130)
    css = f"""
  #{plan.id} .stage {{ align-items:center; text-align:center; }}
  #{plan.id} .figure {{ font:800 {size}px {SANS}; color:{plan.theme.ink}; line-height:1;
      letter-spacing:-0.03em; font-variant-numeric:tabular-nums; }}
  #{plan.id} .stat-label {{ font:500 44px {SANS}; color:{plan.theme.muted}; max-width:1200px; line-height:1.4; }}
  #{plan.id} .kicker {{ justify-content:center; }}
"""
    markup = (
        '    <div class="plate"><div class="wash"></div></div>\n'
        + _motif_block(plan, style="left:50%; top:50%; width:1000px; height:1000px; margin:-500px 0 0 -500px; opacity:.2;")
        + '    <div class="stage">\n'
        + (f'      <div class="kicker" id="{plan.id}-kicker">{_esc(plan.kicker)}</div>\n' if plan.kicker else "")
        + f'      <div class="figure" id="{plan.id}-figure">{_esc(figure)}</div>\n'
        + (f'      <div class="stat-label" id="{plan.id}-label">{_esc(plan.stat_label or plan.body)}</div>\n'
           if (plan.stat_label or plan.body) else "")
        + '    </div>\n'
        + '    <div class="vignette"></div>\n'
    )
    timeline = f"""        inAt("#{plan.id}-kicker", {{ y: -22, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .5, ease: "power2.out" }}, 0.15);
        inAt("#{plan.id}-figure", {{ scale: .72, opacity: 0 }}, {{ scale: 1, opacity: 1, duration: .95, ease: "back.out(1.5)", transformOrigin: "50% 50%" }}, 0.28);
        inAt("#{plan.id}-label", {{ y: 34, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .65, ease: "power2.out" }}, 0.62);
        inAt("#{plan.id}-motif", {{ scale: 1.2, opacity: 0, rotate: 10 }}, {{ scale: 1, opacity: .2, rotate: 0, duration: 1.9, ease: "power2.out", transformOrigin: "50% 50%" }}, 0.1);
"""
    return _shell(plan, css=css, markup=markup, timeline=timeline, wash=(50, 44))


def _render_quote(plan: ScenePlan) -> str:
    accent = accent_hex(plan.accent, plan.theme)
    text = plan.quote or plan.headline
    size = 84 if len(text) <= 90 else (66 if len(text) <= 160 else 50)
    css = f"""
  #{plan.id} .stage {{ padding:150px 190px 220px; }}
  #{plan.id} .mark {{ font:700 220px {SERIF}; color:{_rgba(accent, 0.35)}; line-height:.7; height:120px; }}
  #{plan.id} .quote {{ font-family:{SERIF}; font-size:{size}px; font-style:italic;
      line-height:1.32; color:{plan.theme.ink}; max-width:1440px; }}
  #{plan.id} .attrib {{ font:600 32px {SANS}; letter-spacing:.14em; text-transform:uppercase;
      color:{accent}; display:flex; align-items:center; gap:18px; margin-top:12px; }}
  #{plan.id} .attrib::before {{ content:""; width:48px; height:2px; background:{accent}; }}
"""
    markup = (
        '    <div class="plate"><div class="wash"></div></div>\n'
        + _motif_block(plan, style="right:-120px; top:-120px; width:840px; height:840px; opacity:.18;")
        + '    <div class="stage">\n'
        + f'      <div class="mark" id="{plan.id}-mark">&ldquo;</div>\n'
        + f'      <div class="quote" id="{plan.id}-quote">{_esc(text)}</div>\n'
        + (f'      <div class="attrib" id="{plan.id}-attrib">{_esc(plan.attribution)}</div>\n'
           if plan.attribution else "")
        + '    </div>\n'
        + '    <div class="vignette"></div>\n'
    )
    timeline = f"""        inAt("#{plan.id}-mark", {{ y: 40, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .7, ease: "power3.out" }}, 0.12);
        inAt("#{plan.id}-quote", {{ y: 50, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .9, ease: "expo.out" }}, 0.3);
        inAt("#{plan.id}-attrib", {{ x: -30, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .6, ease: "power2.out" }}, 0.7);
        inAt("#{plan.id}-motif", {{ rotate: -12, opacity: 0 }}, {{ rotate: 0, opacity: .18, duration: 1.8, ease: "power2.out", transformOrigin: "50% 50%" }}, 0.15);
"""
    return _shell(plan, css=css, markup=markup, timeline=timeline, wash=(78, 22))


def _footage_intervals(sequence, scene_duration: float):
    """Serialize one shared centisecond timeline without extending a source."""
    limit = int((Decimal(str(scene_duration)) * 100).to_integral_value(rounding=ROUND_FLOOR))
    cursor = 0
    for index, item in enumerate(sequence):
        source_duration = max(0.0, float(item.get("duration_seconds") or 0)) or 5.0
        ticks = int((Decimal(str(source_duration)) * 100).to_integral_value(rounding=ROUND_FLOOR))
        duration = min(ticks, limit - cursor)
        if duration <= 0:
            continue
        yield index, item, cursor / 100, duration / 100
        cursor += duration
        if cursor >= limit:
            break


def _editorial_details(plan: ScenePlan) -> str:
    """Keep body and parallel facts readable when media changes the layout."""
    items = tuple(item for item in plan.items if item.strip())
    body = plan.body.strip()
    # Footage attachment historically flattened a list into the body. Render
    # that list once, while retaining independent body/stat/attribution copy.
    if body == " · ".join(items):
        body = ""
    markup = (
        f'<div class="body" id="{plan.id}-body">{_esc(body)}</div>\n'
        if body else ""
    )
    if items:
        markup += (
            f'<ul class="editorial-facts" id="{plan.id}-facts">'
            + "".join(f"<li>{_esc(item)}</li>" for item in items)
            + "</ul>\n"
        )
    return markup


def _render_footage(plan: ScenePlan) -> str:
    """Render a full-bleed media plate that remains populated for the full scene."""
    accent = accent_hex(plan.accent, plan.theme)
    credits = ""
    if plan.footage_kind == "video":
        # Every public clip is one-pass.  AI-selected clips may be sequenced;
        # when their combined duration is shorter than the narration, the
        # animated HyperFrame plate underneath carries the remainder.  Never
        # hide a short source with loop/boomerang playback.
        video_duration = plan.duration
        hold_media = ""
        if plan.collage_broll and plan.collage_hold_src:
            target_duration = min(
                plan.duration,
                max(0.0, plan.collage_target_duration_seconds),
            )
            if 0 < target_duration < plan.duration:
                video_duration = target_duration
                hold_duration = plan.duration - target_duration
                hold_media = (
                    f'      <img id="{plan.id}-hold" class="clip media collage-hold" '
                    f'src="{_esc(plan.collage_hold_src)}" data-start="{target_duration:.2f}" '
                    f'data-duration="{hold_duration:.2f}" data-track-index="0" alt="" '
                    f'crossorigin="anonymous">\n'
                )
        if plan.collage_broll:
            media = (
                f'      <video id="{plan.id}-media" class="clip media" src="{_esc(plan.footage_src)}" '
                f'data-start="0" data-duration="{video_duration:.2f}" data-track-index="0" '
                'muted playsinline crossorigin="anonymous"></video>\n'
                + hold_media
            )
        else:
            sequence = plan.footage_sequence or (
                {
                    "src": plan.footage_src,
                    "kind": plan.footage_kind,
                    "duration_seconds": min(5.0, plan.duration),
                    "credit": plan.footage_credit,
                },
            )
            sequence_markup: list[str] = []
            sequence_credits: list[str] = []
            for index, item, cursor, duration in _footage_intervals(sequence, plan.duration):
                sequence_markup.append(
                    f'      <video id="{plan.id}-media-{index + 1}" class="clip media public-footage-once" '
                    f'src="{_esc(str(item.get("src") or ""))}" data-start="{cursor:.2f}" '
                    f'data-duration="{duration:.2f}" data-track-index="0" muted playsinline '
                    'crossorigin="anonymous"></video>\n'
                )
                if item.get("credit"):
                    sequence_credits.append(
                        f'    <div id="{plan.id}-credit-{index + 1}" class="clip credit" '
                        f'data-start="{cursor:.2f}" data-duration="{duration:.2f}" '
                        f'data-track-index="1">{_esc(str(item["credit"]))}</div>\n'
                    )
            media = "".join(sequence_markup)
            credits = "".join(sequence_credits)
    else:
        media = f'      <img id="{plan.id}-media" class="media" src="{_esc(plan.footage_src)}" alt="">\n'
        if plan.footage_credit:
            credits = f'    <div class="credit" id="{plan.id}-credit">{_esc(plan.footage_credit)}</div>\n'
    css = f"""
  #{plan.id} .frame {{ position:absolute; inset:0; overflow:hidden; }}
  #{plan.id} .media {{ display:block; width:100%; height:100%; object-fit:cover; }}
  #{plan.id} .frame > .clip {{ position:absolute; inset:0; }}
  #{plan.id} .scrim {{ position:absolute; inset:0;
      background:linear-gradient(180deg, rgba(17,19,24,.4) 0%, rgba(17,19,24,.6) 38%, rgba(17,19,24,.94) 100%); }}
  #{plan.id} .stage {{ justify-content:flex-end; padding:130px 150px 230px; }}
  #{plan.id} .headline {{ font-size:{headline_size(plan.headline, base=92)}px; max-width:1340px;
      color:#F5F2EA; text-shadow:0 8px 40px rgba(0,0,0,.6); }}
  #{plan.id} .body {{ font-size:{body_size(plan.body, base=38)}px; max-width:1100px; color:rgba(245,242,234,.9); }}
  #{plan.id} .editorial-facts {{ margin:0; padding-left:30px; display:grid; gap:12px;
      font:500 34px/1.28 {SANS}; color:#F5F2EA; max-width:1450px; }}
  #{plan.id} .editorial-facts li::marker {{ color:{accent}; }}
  #{plan.id} .credit {{ position:absolute; right:44px; top:40px; font:500 20px {SANS};
      color:#F5F2EA; letter-spacing:.06em;
      background:rgba(17,19,24,.8); padding:9px 16px; border-radius:999px;
      border:1px solid {_rgba(accent, 0.3)}; }}
"""
    comparison_markup = ""
    if plan.left_text and plan.right_text:
        comparison_markup = (
            f'<div class="footage-comparison" id="{plan.id}-comparison">'
            + "".join(
                '<div class="comparison-side">'
                f'<div class="comparison-label">{_esc(label)}</div>'
                f'<div class="comparison-text">{_esc(value)}</div></div>'
                for label, value in (
                    (plan.left_label or "Before", plan.left_text),
                    (plan.right_label or "After", plan.right_text),
                )
            )
            + '</div>\n'
        )
        css += f"""
  #{plan.id} .footage-comparison {{ display:grid; grid-template-columns:1fr 1fr; gap:32px; width:100%; max-width:1500px; }}
  #{plan.id} .comparison-side {{ padding:24px 28px; border-left:5px solid {accent}; background:rgba(17,19,24,.88); }}
  #{plan.id} .comparison-label {{ font:700 24px/1.2 {SANS}; letter-spacing:.08em; color:#F5F2EA; margin-bottom:14px; }}
  #{plan.id} .comparison-text {{ font:800 46px/1.12 {SANS}; font-variant-numeric:tabular-nums; color:#F5F2EA; }}
"""
    if plan.items or comparison_markup:
        css += f"""
  #{plan.id} .stage {{ justify-content:center; padding:140px 150px 210px; gap:20px; }}
  #{plan.id} .headline {{ font-size:60px; max-width:1500px; }}
  #{plan.id} .body {{ font-size:34px; max-width:1480px; line-height:1.28; }}
"""
    if plan.collage_broll:
        caption_headline = plan.headline.strip() or plan.body.strip()
        # The generated plate carries the metaphor, not the factual copy.
        # Keep the planned supporting facts when attachment changes the
        # archetype to footage, including reviewer-requested corrections.
        details = _editorial_details(plan)
        if plan.quote.strip():
            details += f'<blockquote class="collage-quote">{_esc(plan.quote)}</blockquote>\n'
        if plan.attribution.strip():
            details += f'<div class="collage-attribution">{_esc(plan.attribution)}</div>\n'
        if plan.stat.strip() or plan.stat_label.strip():
            details += (
                f'<div class="collage-stat">{_esc(plan.stat)}</div>'
                f'<div class="collage-stat-label">{_esc(plan.stat_label)}</div>\n'
            )
        details += comparison_markup
        details_markup = (
            f'<aside class="collage-details" id="{plan.id}-collage-details">{details}</aside>\n'
            if details else ""
        )
        caption_size = max(
            42,
            headline_size(caption_headline, base=64, floor=42),
        )
        collage_css = f"""
  #{plan.id} .collage-caption {{ position:absolute; left:96px; bottom:220px; z-index:3;
      width:min(860px, calc(100% - 192px)); padding:24px 28px 27px; box-sizing:border-box;
      display:flex; flex-direction:column; gap:11px; background:rgba(11,13,23,.90);
      border:2px solid {_rgba(accent, .72)}; border-left:8px solid {accent}; border-radius:0 18px 18px 0;
      box-shadow:0 22px 70px rgba(0,0,0,.42); }}
  #{plan.id} .collage-kicker {{ font:700 22px {SANS}; letter-spacing:.18em; text-transform:uppercase;
      color:{accent}; line-height:1.2; }}
  #{plan.id} .collage-headline {{ max-width:790px; font:800 {caption_size}px/1.08 {SANS};
      letter-spacing:-.025em; color:#F5F2EA; text-shadow:0 3px 18px rgba(0,0,0,.72); }}
"""
        if details:
            collage_css += f"""
  #{plan.id} .collage-caption {{ width:760px; }}
  #{plan.id} .collage-details {{ position:absolute; right:96px; bottom:220px; z-index:3;
      width:820px; padding:32px 36px; box-sizing:border-box; display:flex; flex-direction:column;
      gap:20px; color:#F5F2EA; background:rgba(11,13,23,1);
      border-top:4px solid {accent}; box-shadow:0 22px 70px rgba(0,0,0,.42); }}
  #{plan.id} .collage-details .body, #{plan.id} .collage-details .editorial-facts {{
      font:400 30px/1.30 {SANS}; color:#FFFFFF; max-width:100%; }}
  #{plan.id} .collage-details .editorial-facts {{ gap:14px; padding-left:25px; }}
  #{plan.id} .collage-quote {{ margin:0; font:500 30px/1.3 {SANS}; }}
  #{plan.id} .collage-attribution {{ font:500 24px/1.3 {SANS}; color:#F5F2EA; }}
  #{plan.id} .collage-stat {{ font:800 48px/1.1 {SANS}; font-variant-numeric:tabular-nums; }}
  #{plan.id} .collage-stat-label {{ font:400 28px/1.3 {SANS}; }}
  #{plan.id} .collage-details .footage-comparison {{ gap:16px; }}
  #{plan.id} .collage-details .comparison-side {{ padding:16px; }}
  #{plan.id} .collage-details .comparison-text {{ font-size:30px; }}
"""
            if plan.frame.is_portrait:
                collage_css += f"""
  #{plan.id} .collage-caption {{ top:170px; bottom:auto !important; }}
  #{plan.id} .collage-details {{ left:72px; right:72px; width:auto; bottom:330px; }}
"""
        caption_markup = ""
        if plan.kicker or caption_headline:
            caption_markup = (
                f'    <div class="collage-caption" id="{plan.id}-collage-caption">\n'
                + (
                    f'      <div class="collage-kicker" id="{plan.id}-collage-kicker">'
                    f'{_esc(plan.kicker)}</div>\n'
                    if plan.kicker
                    else ""
                )
                + (
                    f'      <div class="collage-headline" id="{plan.id}-collage-headline">'
                    f'{_esc(caption_headline)}</div>\n'
                    if caption_headline
                    else ""
                )
                + "    </div>\n"
            )
        markup = (
            f'    <div class="frame collage-frame" id="{plan.id}-frame">\n'
            + media
            + "    </div>\n"
            + caption_markup
            + details_markup
        )
        timeline = f"""        inAt("#{plan.id}-media", {{ opacity: 0 }}, {{ opacity: 1, duration: .12, ease: "power1.out" }}, 0.1);
        inAt("#{plan.id}-collage-caption", {{ x: -72, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .66, ease: "power4.out" }}, 0.24);
        inAt("#{plan.id}-collage-kicker", {{ y: 14, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .46, ease: "power2.out" }}, 0.42);
        inAt("#{plan.id}-collage-headline", {{ y: 26, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .72, ease: "expo.out" }}, 0.5);
"""
        if details:
            details_at = min(
                max(.8, plan.collage_target_duration_seconds),
                max(.8, plan.duration * .3),
            )
            timeline += f'''        inAt("#{plan.id}-collage-details", {{ x: 56, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .7, ease: "power3.out" }}, {details_at:.2f});
'''
        return _shell(
            plan,
            css=css + collage_css,
            markup=markup,
            timeline=timeline,
            wash=(50, 50),
        )

    fallback_plate = (
        '    <div class="plate public-footage-fallback"><div class="wash"></div></div>\n'
        + _motif_block(
            plan,
            style="left:50%; top:50%; width:1180px; height:1180px; margin:-590px 0 0 -590px; opacity:.2;",
        )
    )
    markup = (
        fallback_plate
        + f'    <div class="frame" id="{plan.id}-frame" data-layout-allow-overflow>\n'
        + media
        + '    </div>\n'
        + '    <div class="scrim"></div>\n'
        + credits
        + '    <div class="stage">\n'
        + (f'      <div class="kicker" id="{plan.id}-kicker">{_esc(plan.kicker)}</div>\n' if plan.kicker else "")
        + (f'      <div class="headline" id="{plan.id}-head">{_esc(plan.headline)}</div>\n' if plan.headline else "")
        + comparison_markup
        + _editorial_details(plan)
        + '    </div>\n'
    )
    primary_media_id = f"{plan.id}-media-1" if plan.footage_kind == "video" else f"{plan.id}-media"
    timeline = f"""        inAt("#{primary_media_id}", {{ scale: 1.04, opacity: 0 }}, {{ scale: 1.12, opacity: 1, duration: 0.35, ease: "power1.out" }}, 0.05);
        inAt("#{plan.id}-motif", {{ scale: .84, opacity: 0, rotate: -10 }}, {{ scale: 1, opacity: .2, rotate: 0, duration: {max(2.0, plan.duration):.2f}, ease: "sine.out", transformOrigin: "50% 50%" }}, 0.05);
        inAt("#{plan.id}-kicker", {{ x: -36, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .55, ease: "power3.out" }}, 0.25);
        inAt("#{plan.id}-head", {{ y: 56, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .85, ease: "expo.out" }}, 0.4);
        inAt("#{plan.id}-body", {{ y: 26, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .65, ease: "power2.out" }}, 0.62);
        inAt("#{plan.id}-facts", {{ y: 26, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .65, ease: "power2.out" }}, 0.72);
        inAt("#{plan.id}-credit", {{ opacity: 0 }}, {{ opacity: 1, duration: .6, ease: "power1.out" }}, 0.8);
"""
    return _shell(plan, css=css, markup=markup, timeline=timeline, wash=(50, 50))


def _render_news_webpage_overlay(plan: ScenePlan) -> str:
    """Layer a captured English article page over image or moving footage."""
    accent = accent_hex(plan.accent, plan.theme)
    if plan.footage_src and plan.footage_kind == "video":
        sequence = plan.footage_sequence or (
            {"src": plan.footage_src, "duration_seconds": min(5.0, plan.duration)},
        )
        background_parts: list[str] = []
        for index, item, cursor, duration in _footage_intervals(sequence, plan.duration):
            element_id = (
                f"{plan.id}-background"
                if index == 0
                else f"{plan.id}-background-{index + 1}"
            )
            background_parts.append(
                f'      <video id="{element_id}" class="clip news-web-background" '
                f'src="{_esc(str(item.get("src") or ""))}" data-start="{cursor:.2f}" '
                f'data-duration="{duration:.2f}" data-track-index="0" muted playsinline '
                'crossorigin="anonymous"></video>\n'
            )
        background = "".join(background_parts)
    else:
        background_src = plan.footage_src or plan.news_image_src
        background = (
            f'      <img id="{plan.id}-background" class="news-web-background" '
            f'src="{_esc(background_src)}" alt="" crossorigin="anonymous">\n'
        )
    source_label = plan.news_webpage_source or "English news source"
    has_editorial_copy = bool(plan.headline or plan.body or plan.items)
    css = f"""
  #{plan.id} .news-web-bg-frame {{ position:absolute; inset:0; overflow:hidden; background:{plan.theme.bg}; }}
  #{plan.id} .news-web-background {{ position:absolute; inset:0; display:block; width:100%; height:100%;
      object-fit:cover; filter:saturate(.78) contrast(1.06); }}
  #{plan.id} .news-web-scrim {{ position:absolute; inset:0; background:rgba(8,10,18,.43); }}
  #{plan.id} .news-web-card-wrap {{ position:absolute; left:50%; top:50%; width:1260px; height:760px;
      margin:-380px 0 0 -630px; z-index:4; perspective:1800px; }}
  #{plan.id} .news-web-card {{ position:absolute; inset:0; overflow:hidden; border:3px solid rgba(255,255,255,.92);
      border-radius:12px; background:#FFFFFF; box-shadow:0 46px 130px rgba(0,0,0,.58);
      transform-style:preserve-3d; }}
  #{plan.id} .news-web-shot-crop {{ position:absolute; inset:0; overflow:hidden; border-radius:9px; }}
  #{plan.id} .news-web-shot {{ display:block; width:100%; height:100%; object-fit:cover; object-position:50% 0%; }}
  #{plan.id} .news-web-source {{ position:absolute; left:28px; top:26px; z-index:6; max-width:720px;
      padding:12px 20px; border-radius:999px; background:rgba(9,12,20,.88); border:2px solid {accent};
      color:#F5F2EA; font:800 19px {SANS}; line-height:1.2; letter-spacing:.12em;
      text-transform:uppercase; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  #{plan.id} .news-web-corner {{ position:absolute; right:-22px; bottom:-22px; width:126px; height:126px;
      z-index:5; border-right:7px solid {accent}; border-bottom:7px solid {accent}; }}
"""
    if has_editorial_copy:
        css += f"""
  #{plan.id} .news-web-card-wrap {{ left:120px; top:280px; width:470px; height:340px; margin:0; }}
  #{plan.id} .news-web-shot {{ object-fit:contain; object-position:50% 100%; }}
  #{plan.id} .news-web-source {{ left:14px; right:14px; top:16px; font-size:17px;
      white-space:normal; padding:12px; letter-spacing:.06em; }}
  #{plan.id} .news-web-editorial {{ position:relative; box-sizing:border-box; width:100%; height:100%;
      padding:130px 120px 190px 660px; display:flex; flex-direction:column; justify-content:center;
      gap:20px; z-index:3; background:linear-gradient(90deg, transparent 28%, rgba(8,10,18,.93) 36%); }}
  #{plan.id} .news-web-editorial .headline {{ font-size:50px; line-height:1.08; color:#F5F2EA; }}
  #{plan.id} .news-web-editorial .body {{ font:500 32px/1.28 {SANS}; color:#F5F2EA; }}
  #{plan.id} .editorial-facts {{ margin:0; padding-left:27px; display:grid; gap:13px;
      font:500 32px/1.28 {SANS}; color:#F5F2EA; }}
  #{plan.id} .editorial-facts li::marker {{ color:{accent}; }}
"""
    markup = (
        f'    <div class="news-web-bg-frame" id="{plan.id}-background-frame">\n'
        + background
        + "    </div>\n"
        + f'    <div class="news-web-scrim" id="{plan.id}-web-scrim"></div>\n'
        + (
            f'<div class="news-web-editorial" id="{plan.id}-editorial">'
            + (f'<div class="kicker">{_esc(plan.kicker)}</div>' if plan.kicker else "")
            + (f'<div class="headline">{_esc(plan.headline)}</div>' if plan.headline else "")
            + _editorial_details(plan)
            + "</div>\n"
            if has_editorial_copy else ""
        )
        + f'    <div class="news-web-card-wrap" id="{plan.id}-web-wrap" data-layout-allow-overflow>\n'
        + f'      <div class="news-web-card" id="{plan.id}-web-card">\n'
        + '        <div class="news-web-shot-crop">\n'
        + f'          <img class="news-web-shot" id="{plan.id}-web-shot" '
        + f'src="{_esc(plan.news_webpage_src)}" alt="{_esc(plan.news_webpage_headline)}" '
        + 'crossorigin="anonymous">\n'
        + "        </div>\n"
        + f'        <div class="news-web-source" id="{plan.id}-web-source">English news · {_esc(source_label)}</div>\n'
        + "      </div>\n"
        + f'      <div class="news-web-corner" id="{plan.id}-web-corner"></div>\n'
        + "    </div>\n"
    )
    direction = -1 if _seed_from(plan.id) % 2 else 1
    drift = max(2.4, plan.duration - 0.25)
    timeline = f"""        inAt("#{plan.id}-background", {{ scale: 1.08, x: {direction * -18} }}, {{ scale: 1.16, x: {direction * 18}, duration: {drift:.2f}, ease: "none", transformOrigin: "50% 50%" }}, 0.05);
        inAt("#{plan.id}-web-scrim", {{ opacity: 0 }}, {{ opacity: 1, duration: .48, ease: "sine.out" }}, 0.12);
        inAt("#{plan.id}-editorial", {{ x: 36, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .65, ease: "power2.out" }}, 0.4);
        inAt("#{plan.id}-web-card", {{ x: {direction * 210}, y: 70, opacity: 0, rotationY: {direction * -10}, rotationZ: {direction * 1.2}, scale: .9 }}, {{ x: 0, y: 0, opacity: 1, rotationY: 0, rotationZ: 0, scale: 1, duration: 1.02, ease: "power4.out", transformPerspective: 1800, transformOrigin: "50% 50%" }}, 0.22);
        inAt("#{plan.id}-web-shot", {{ scale: 1.035, y: -8 }}, {{ scale: 1, y: 8, duration: {drift:.2f}, ease: "none", transformOrigin: "50% 0%" }}, 0.28);
        inAt("#{plan.id}-web-source", {{ x: -34, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .58, ease: "expo.out" }}, 0.64);
        inAt("#{plan.id}-web-corner", {{ scale: .45, opacity: 0, transformOrigin: "100% 100%" }}, {{ scale: 1, opacity: 1, duration: .7, ease: "circ.out" }}, 0.72);
"""
    return _shell(plan, css=css, markup=markup, timeline=timeline, wash=(50, 50))


def _render_intro(plan: ScenePlan) -> str:
    """Edition-aware ByteFront opener over the selected Gemini motion plate."""
    if not plan.footage_src or not plan.intro_logo_src:
        plan.archetype = "statement"
        return _render_statement(plan)

    light = plan.intro_style in {"morning-brief", "signal-shot"}
    foreground = "#10243F" if light else "#F8FAFC"
    quiet = "rgba(16,36,63,.68)" if light else "rgba(248,250,252,.72)"
    panel = "rgba(250,248,243,.76)" if light else "rgba(5,18,37,.70)"
    border = "rgba(16,36,63,.16)" if light else "rgba(248,250,252,.22)"
    veil = (
        "linear-gradient(90deg,rgba(244,240,231,.18),rgba(244,240,231,.04) 58%,transparent 84%)"
        if light
        else "linear-gradient(90deg,rgba(3,13,28,.30),rgba(3,13,28,.05) 58%,transparent 84%)"
    )
    portrait = plan.frame.is_portrait
    shell_width = "calc(100% - 144px)" if portrait else "1040px"
    shell_margin = "0 auto" if portrait else "0 auto 0 360px"
    shell_padding = "54px 46px 48px" if portrait else "46px 58px 42px"
    logo_width = "560px"
    espresso_size = 66
    date_size = 62 if portrait else 66
    css = f"""
  #{plan.id} .intro-background {{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }}
  #{plan.id} .intro-veil {{ position:absolute; inset:0; background:{veil}; pointer-events:none; }}
  #{plan.id} .intro-overlay {{ position:absolute; inset:0; display:grid; grid-template-rows:auto 1fr auto;
      padding:{'86px 72px 104px' if portrait else '62px 82px 56px'}; color:{foreground}; font-family:{SANS}; }}
  #{plan.id} .intro-topline, #{plan.id} .intro-footer {{ display:flex; align-items:center;
      justify-content:space-between; color:{quiet}; font:800 17px/1.2 'Courier New',monospace;
      letter-spacing:.18em; text-transform:uppercase; }}
  #{plan.id} .intro-topline span:first-child {{ display:flex; align-items:center; gap:16px; }}
  #{plan.id} .intro-topline span:first-child::before {{ content:''; width:42px; height:3px; background:#C98758; }}
  #{plan.id} .intro-shell {{ align-self:center; width:{shell_width}; margin:{shell_margin}; padding:{shell_padding};
      border:1px solid {border}; background:{panel}; box-shadow:0 28px 86px rgba(7,20,39,.17);
      backdrop-filter:blur(12px) saturate(112%); }}
  #{plan.id} .intro-brand {{ display:flex; align-items:center; gap:26px; }}
  #{plan.id} .intro-logo {{ display:block; width:{logo_width}; max-width:100%; height:auto; object-fit:contain; }}
  #{plan.id} .intro-espresso {{ display:flex; align-items:center; padding-left:26px;
      border-left:3px solid #C98758; color:{foreground}; white-space:nowrap;
      font:700 {espresso_size}px/1.2 {SANS}; letter-spacing:-.045em; }}
  #{plan.id} .intro-copper-rule {{ width:100%; height:2px; margin:34px 0 30px;
      background:linear-gradient(90deg,#C98758 0 34%,rgba(201,135,88,.08) 78%,transparent); transform-origin:left center; }}
  #{plan.id} .intro-edition {{ display:block; }}
  #{plan.id} .intro-weekday {{ color:#C98758; font:800 18px/1.25 'Courier New',monospace;
      letter-spacing:.22em; text-transform:uppercase; }}
  #{plan.id} .intro-date {{ margin:12px 0 0; color:{foreground};
      font:700 {date_size}px/.98 {SERIF}; letter-spacing:-.035em; white-space:nowrap; }}
  #{plan.id} .intro-footer-rule {{ width:{'270px' if portrait else '430px'}; height:2px;
      background:linear-gradient(90deg,#2F64AE,transparent); transform-origin:left center; }}
  #{plan.id} .intro-fade {{ position:absolute; inset:0; background:#F4F0E7; opacity:0; pointer-events:none; }}
  #{plan.id} .intro-overlay * {{ box-sizing:border-box; }}
"""
    if portrait:
        css += f"""
  #{plan.id} .intro-brand {{ flex-wrap:wrap; }}
  #{plan.id} .intro-date {{ white-space:normal; }}
  #{plan.id} .intro-footer {{ gap:26px; }}
"""

    markup = f"""    <video id="{plan.id}-background" class="clip intro-background" data-bookend-layer="background" style="z-index:0" src="{_esc(plan.footage_src)}"
      data-start="0" data-duration="{plan.duration:.2f}" data-track-index="0" muted playsinline preload="auto" crossorigin="anonymous"></video>
    <div class="intro-veil" data-bookend-layer="veil" style="z-index:1" data-layout-ignore></div>
    <section class="intro-overlay" data-bookend-layer="overlay" style="z-index:2">
      <header class="intro-topline"><span>{_esc(plan.kicker)}</span><span>ByteFront / Intro 06.00</span></header>
      <article class="intro-shell">
        <div class="intro-brand" data-intro-role="brand">
          <img class="intro-logo" src="{_esc(plan.intro_logo_src)}" alt="ByteFront">
          <div class="intro-espresso">Espresso</div>
        </div>
        <div class="intro-copper-rule"></div>
        <div class="intro-edition" data-intro-role="edition">
          <div id="{plan.id}-weekday" class="intro-weekday" data-intro-field="weekday">{_esc(plan.edition_weekday)}</div>
          <div id="{plan.id}-date" class="intro-date" data-intro-field="date">{_esc(plan.edition_date)}</div>
        </div>
      </article>
      <footer class="intro-footer"><div class="intro-footer-rule"></div><span>{_esc(plan.body)}</span></footer>
    </section>
    <div id="{plan.id}-fade" class="intro-fade" data-bookend-layer="fade" style="z-index:3" data-layout-ignore></div>
"""
    fallbacks = {
        "edition_date": plan.edition_date,
        "edition_weekday": plan.edition_weekday,
    }
    fallback_json = json.dumps(fallbacks, ensure_ascii=False)
    timeline = f"""        const variables = window.__hyperframes.getVariables();
        const fallbackValues = {fallback_json};
        const value = (key) => String(variables[key] || fallbackValues[key] || "");
        q('[data-intro-field="date"]')[0].textContent = value("edition_date");
        q('[data-intro-field="weekday"]')[0].textContent = value("edition_weekday");
        inAt("#{plan.id} .intro-topline", {{ y: -24, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .55, ease: "power3.out" }}, .12);
        inAt("#{plan.id} .intro-shell", {{ y: 34, opacity: 0, scale: .985 }}, {{ y: 0, opacity: 1, scale: 1, duration: .82, ease: "expo.out" }}, .24);
        inAt("#{plan.id} .intro-logo", {{ x: -42, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .68, ease: "power4.out" }}, .44);
        inAt("#{plan.id} .intro-espresso", {{ x: 28, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .48, ease: "back.out(1.5)" }}, .72);
        inAt("#{plan.id} .intro-copper-rule", {{ scaleX: 0 }}, {{ scaleX: 1, duration: .78, ease: "power4.inOut" }}, .74);
        inAt("#{plan.id} .intro-weekday", {{ y: 18, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .48, ease: "power2.out" }}, .96);
        inAt("#{plan.id} .intro-date", {{ x: -44, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .76, ease: "expo.out" }}, 1.06);
        inAt("#{plan.id} .intro-footer-rule", {{ scaleX: 0 }}, {{ scaleX: 1, duration: .72, ease: "power3.out" }}, 1.56);
        inAt("#{plan.id} .intro-footer span", {{ y: 14, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .52, ease: "power2.out" }}, 1.72);
        outAt("#{plan.id}-fade", {{ opacity: 1, duration: .42, ease: "power2.inOut" }}, {max(0.1, plan.duration - 0.42):.2f});
"""
    return _shell(plan, css=css, markup=markup, timeline=timeline, wash=(50, 50))


def _render_outro(plan: ScenePlan) -> str:
    """Known-good editable overlay over the selected Gemini motion plate."""
    if not plan.footage_src or not plan.outro_logo_src:
        plan.archetype = "statement"
        return _render_statement(plan)
    from backend.pipeline import outros

    light = plan.outro_style in {"morning-brief", "signal-shot"}
    foreground = "#10243F" if light else "#F8FAFC"
    quiet = "rgba(16,36,63,.68)" if light else "rgba(248,250,252,.74)"
    panel = "rgba(246,248,251,.82)" if light else "rgba(7,20,39,.74)"
    border = "rgba(16,36,63,.14)" if light else "rgba(248,250,252,.20)"
    action_bg = "rgba(255,255,255,.58)" if light else "rgba(248,250,252,.10)"
    veil = (
        "linear-gradient(90deg,rgba(246,248,251,.64),rgba(246,248,251,.18) 54%,transparent 82%)"
        if light
        else "linear-gradient(90deg,rgba(3,13,28,.36),rgba(3,13,28,.08) 48%,transparent 80%)"
    )
    portrait = plan.frame.is_portrait
    panel_width = "calc(100% - 144px)" if portrait else "1180px"
    overlay_padding = "92px 72px 110px" if portrait else "70px 86px 62px"
    panel_padding = "52px 44px" if portrait else "46px 54px 42px"
    logo_width = "560px"
    headline_size_px = 72 if portrait else 91
    subscribe_action = (
        '<div class="outro-action" data-outro-action="subscribe"><span>Subscribe</span></div>'
        if any(item.casefold() == "subscribe" for item in plan.items) else ""
    )
    css = f"""
  #{plan.id} .outro-background {{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }}
  #{plan.id} .outro-veil {{ position:absolute; inset:0; background:{veil}; pointer-events:none; }}
  #{plan.id} .outro-overlay {{ position:absolute; inset:0; display:grid; grid-template-rows:auto 1fr auto;
      padding:{overlay_padding}; color:{foreground}; font-family:{SANS}; }}
  #{plan.id} .outro-topline, #{plan.id} .outro-footer {{ display:flex; align-items:center;
      justify-content:space-between; color:{quiet}; font:700 17px/1.2 'Courier New',monospace;
      letter-spacing:.18em; text-transform:uppercase; }}
  #{plan.id} .outro-topline span:first-child {{ display:flex; align-items:center; gap:16px; }}
  #{plan.id} .outro-topline span:first-child::before {{ content:''; width:42px; height:3px; background:#C98758; }}
  #{plan.id} .outro-panel {{ align-self:center; width:{panel_width}; min-height:{'840px' if portrait else '585px'};
      padding:{panel_padding}; border:1px solid {border}; background:{panel};
      box-shadow:0 28px 80px rgba(7,20,39,.22); backdrop-filter:blur(16px) saturate(120%); }}
  #{plan.id} .outro-brand-row {{ display:flex; align-items:center; gap:26px; flex-wrap:wrap; }}
  #{plan.id} .outro-logo-shell {{ display:flex; align-items:center; width:{logo_width}; }}
  #{plan.id} .outro-logo {{ display:block; width:100%; height:auto; object-fit:contain; }}
  #{plan.id} .outro-espresso {{ display:flex; align-items:center; padding-left:26px;
      border-left:3px solid #C98758; color:{foreground}; white-space:nowrap;
      font:700 66px/1.2 {SANS}; letter-spacing:-.045em; }}
  #{plan.id} .outro-message {{ margin-top:42px; }}
  #{plan.id} .outro-kicker {{ color:#C98758; font:800 19px/1.45 'Courier New',monospace;
      letter-spacing:.23em; text-transform:uppercase; }}
  #{plan.id} .outro-headline {{ max-width:1020px; margin:24px 0 0; color:{foreground};
      font:760 {headline_size_px}px/.98 {SANS}; letter-spacing:-.058em; }}
  #{plan.id} .outro-subline {{ margin:22px 0 0; color:{quiet}; font:600 25px/1.35 {SANS}; }}
  #{plan.id} .outro-actions {{ display:flex; gap:14px; margin-top:38px; flex-wrap:wrap; }}
  #{plan.id} .outro-action {{ display:flex; align-items:center; gap:13px; min-width:220px; height:64px;
      padding:0 20px; border:1px solid {border}; border-radius:14px; background:{action_bg};
      color:{foreground}; font:800 17px/1 'Courier New',monospace; letter-spacing:.12em; text-transform:uppercase; }}
  #{plan.id} .outro-action svg {{ width:27px; height:27px; flex:0 0 27px; color:#C98758; }}
  #{plan.id} .outro-footer-rule {{ width:{'260px' if portrait else '390px'}; height:2px;
      background:linear-gradient(90deg,#C98758,transparent); transform-origin:left center; }}
  #{plan.id} .outro-fade {{ position:absolute; inset:0; background:#071427; opacity:0; pointer-events:none; }}
"""
    credits_html = ""
    if plan.outro_credits:
        css += f"""
  #{plan.id} .outro-body {{ display:grid; grid-template-columns:minmax(0,1.45fr) minmax(0,1fr); gap:36px; align-items:center; padding-bottom:90px; min-height:0; }}
  #{plan.id} .outro-panel {{ width:100%; min-height:0; padding:38px; }}
  #{plan.id} .outro-logo-shell {{ width:440px; max-width:100%; }}
  #{plan.id} .outro-espresso {{ font-size:54px; }}
  #{plan.id} .outro-headline {{ font-size:76px; }}
  #{plan.id} .outro-action {{ min-width:0; font-size:16px; padding:0 14px; }}
  #{plan.id} .outro-sources {{ min-width:0; padding:28px; background:{panel}; border:1px solid {border}; color:{foreground}; }}
  #{plan.id} .outro-sources h2 {{ margin:0 0 20px; font:700 26px/1.3 {SANS}; letter-spacing:.06em; }}
  #{plan.id} .outro-credits-window {{ height:600px; overflow:hidden; }}
  #{plan.id} .outro-credits-roll {{ min-height:100%; will-change:transform; }}
  #{plan.id} .outro-credit {{ padding:0 0 28px; margin:0 0 16px; border-bottom:1px solid {border}; }}
  #{plan.id} .outro-credit-kind {{ color:{foreground}; font:700 18px/24px 'Courier New',monospace; }}
  #{plan.id} .outro-credit-line {{ font:500 28px/36px {SANS}; overflow-wrap:anywhere; }}
  #{plan.id} .outro-credit-line:nth-child(2) {{ font-weight:800; }}
"""
        if portrait:
            css += f"""
  #{plan.id} .outro-overlay {{ padding:92px 44px 70px; }}
  #{plan.id} .outro-body {{ grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:24px; }}
  #{plan.id} .outro-panel {{ padding:26px; }}
  #{plan.id} .outro-headline {{ font-size:54px; }}
  #{plan.id} .outro-espresso {{ font-size:42px; padding-left:14px; }}
  #{plan.id} .outro-sources {{ padding:20px; }}
  #{plan.id} .outro-credits-window {{ height:1000px; }}
  #{plan.id} .outro-topline, #{plan.id} .outro-footer {{ font-size:14px; letter-spacing:.06em; }}
"""
        credits_html = (
            '<aside class="outro-sources" data-outro-role="sources">'
            '<h2>SOURCES &amp; CREDITS</h2>'
            '<div class="outro-credits-window" data-layout-allow-overflow data-layout-allow-overlap>'
            '<div class="outro-credits-roll">'
            + outros.credits_markup(list(plan.outro_credits)) + '</div></div></aside>'
        )
    hold = (
        f'<img id="{plan.id}-hold" class="clip outro-background" src="{_esc(plan.outro_hold_src)}" '
        f'data-start="0" data-duration="{plan.duration:.2f}" data-track-index="1" style="z-index:0" alt="">'
        if plan.outro_hold_src else ""
    )
    markup = f"""    {hold}
    <video id="{plan.id}-background" class="clip outro-background" data-bookend-layer="background" style="z-index:0" src="{_esc(plan.footage_src)}"
      data-start="0" data-duration="{plan.duration:.2f}" data-track-index="0" muted playsinline preload="auto" crossorigin="anonymous"></video>
    <div class="outro-veil" data-bookend-layer="veil" style="z-index:1" data-layout-ignore></div>
    <section class="outro-overlay" data-bookend-layer="overlay" style="z-index:2">
      <header class="outro-topline"><span>Your daily shot of frontier tech</span><span>ByteFront / Outro 06.00</span></header>
      {'<div class="outro-body">' if plan.outro_credits else ''}
      <article class="outro-panel">
        <div class="outro-brand-row" data-outro-role="brand">
          <div class="outro-logo-shell"><img class="outro-logo" src="{_esc(plan.outro_logo_src)}" alt="ByteFront"></div>
          <div class="outro-espresso" data-outro-brand-part="espresso">Espresso</div>
        </div>
        <div class="outro-message" data-outro-role="thanks">
          <div class="outro-kicker">{_esc(plan.kicker or 'SEE YOU IN THE NEXT SHOT')}</div>
          <h1 class="outro-headline">{_esc(plan.headline or 'THANKS FOR WATCHING')}</h1>
          <p class="outro-subline">{_esc(plan.body)}</p>
        </div>
        <div class="outro-actions" data-outro-role="actions" aria-label="Engagement actions">
          {subscribe_action}
          <div class="outro-action" data-outro-action="like"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 10v11H3V10h4Z"/><path d="M7 19h10.2a2 2 0 0 0 1.96-1.61l1.2-6A2 2 0 0 0 18.4 9H14l.72-3.08A2.45 2.45 0 0 0 12.34 3L7 10"/></svg><span>Like</span></div>
          <div class="outro-action" data-outro-action="comment"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12a8 8 0 0 1-8 8H6l-4 3V12a9 9 0 1 1 19 0Z"/><path d="M8 12h.01M12 12h.01M16 12h.01"/></svg><span>Comment</span></div>
          <div class="outro-action" data-outro-action="share"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 5l5-3v16l-5-3"/><path d="M19 10H9a6 6 0 0 0-6 6v3"/></svg><span>Share</span></div>
        </div>
      </article>
      {credits_html}{'</div>' if plan.outro_credits else ''}
      <footer class="outro-footer"><div class="outro-footer-rule"></div><span>ByteFront Espresso · Fresh signals, served daily</span></footer>
    </section>
    <div id="{plan.id}-fade" class="outro-fade" data-bookend-layer="fade" style="z-index:3" data-layout-ignore></div>
"""
    timeline = f"""        inAt("#{plan.id} .outro-topline", {{ y: -26, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .55, ease: "power3.out" }}, .14);
        inAt("#{plan.id} .outro-panel", {{ x: -72, opacity: 0, scale: .985 }}, {{ x: 0, opacity: 1, scale: 1, duration: .82, ease: "expo.out" }}, .24);
        inAt("#{plan.id} .outro-logo-shell", {{ y: 20, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .56, ease: "power3.out" }}, .50);
        inAt("#{plan.id} .outro-espresso", {{ x: 28, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .48, ease: "back.out(1.5)" }}, .67);
        inAt("#{plan.id} .outro-kicker", {{ y: 18, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .48, ease: "power2.out" }}, .76);
        inAt("#{plan.id} .outro-headline", {{ x: -48, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .72, ease: "expo.out" }}, .84);
        inAt("#{plan.id} .outro-subline", {{ y: 24, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .52, ease: "power3.out" }}, 1.06);
        inAt("#{plan.id} .outro-action", {{ y: 30, opacity: 0, scale: .96 }}, {{ y: 0, opacity: 1, scale: 1, duration: .48, stagger: .13, ease: "back.out(1.4)" }}, 1.25);
        inAt("#{plan.id} .outro-footer-rule", {{ scaleX: 0 }}, {{ scaleX: 1, duration: .72, ease: "power4.out" }}, 1.58);
        inAt("#{plan.id} .outro-footer span", {{ y: 14, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .55, ease: "power2.out" }}, 1.72);
        outAt("#{plan.id}-fade", {{ opacity: 1, duration: .58, ease: "power2.inOut" }}, {max(0.1, plan.duration - 0.58):.2f});
"""
    if plan.outro_credits:
        # Percentage translation follows the final font layout. Pixel heights
        # sampled at timeline creation can clip the last source after fonts load.
        timeline += f'''        inAt("#{plan.id} .outro-credits-roll", {{ y: 36, yPercent: 0 }}, {{
          y: {1000 if portrait else 600}, yPercent: -100,
          duration: {max(.1, plan.duration - 6):.2f}, ease: "none"
        }}, 2.5);
'''
    return _shell(plan, css=css, markup=markup, timeline=timeline, wash=(50, 50))


def _render_news_image_inline(plan: ScenePlan) -> str:
    """Place a sourced image inside the HyperFrames editorial text flow."""
    accent = accent_hex(plan.accent, plan.theme)
    five_item_layout = len(plan.items) == MAX_NARRATIVE_ITEMS
    hsize = headline_size(
        plan.headline,
        base=76 if five_item_layout else 86,
        floor=48 if five_item_layout else 52,
    )
    bsize = body_size(
        plan.body,
        base=30 if five_item_layout else 34,
        floor=25 if five_item_layout else 27,
    )
    item_size = max(
        22 if five_item_layout else 24,
        body_size(
            " ".join(plan.items),
            base=28 if five_item_layout else 30,
            floor=22 if five_item_layout else 24,
        ),
    )
    stat_size = headline_size(plan.stat, base=60, floor=36)
    copy_class = " news-inline-copy-five" if five_item_layout else ""
    items_class = " news-inline-items-five" if five_item_layout else ""
    contain = plan.news_image_fit == "contain"
    media_background = _rgba(plan.theme.ink, 0.94) if contain else plan.theme.bg
    image_padding = "70px" if contain else "0"
    object_position = "50% 50%"
    css = f"""
  #{plan.id} .news-inline-stage {{ flex-direction:row; align-items:center; justify-content:space-between;
      gap:72px; padding:112px 124px 218px; }}
  #{plan.id} .news-inline-copy {{ width:47%; display:flex; flex-direction:column; gap:25px; position:relative; z-index:3; }}
  #{plan.id} .news-inline-copy-five {{ gap:16px; }}
  #{plan.id} .news-inline-headline {{ font-size:{hsize}px; max-width:790px; }}
  #{plan.id} .news-inline-body {{ font-size:{bsize}px; max-width:760px; }}
  #{plan.id} .news-inline-items {{ display:flex; flex-direction:column; gap:13px; max-width:760px;
      margin:0; padding:0; list-style:none; }}
  #{plan.id} .news-inline-item {{ display:grid; grid-template-columns:42px minmax(0, 1fr); gap:15px;
      align-items:start; padding:14px 16px; border-left:4px solid {accent};
      background:{_rgba(plan.theme.ink, .07)}; }}
  #{plan.id} .news-inline-item-index {{ font:700 18px {SANS}; line-height:1.45;
      letter-spacing:.08em; color:{accent}; font-variant-numeric:tabular-nums; }}
  #{plan.id} .news-inline-item-text {{ font:600 {item_size}px/1.3 {SANS}; color:{plan.theme.ink}; }}
  #{plan.id} .news-inline-items-five {{ gap:8px; }}
  #{plan.id} .news-inline-items-five .news-inline-item {{ grid-template-columns:34px minmax(0, 1fr);
      gap:10px; padding:9px 12px; }}
  #{plan.id} .news-inline-items-five .news-inline-item-text {{ line-height:1.2; }}
  #{plan.id} .news-inline-stat {{ display:flex; flex-direction:column; gap:7px; max-width:760px;
      padding:17px 20px; border-left:5px solid {accent}; background:{_rgba(plan.theme.ink, .1)}; }}
  #{plan.id} .news-inline-stat-value {{ font:800 {stat_size}px/1.05 {SANS}; color:{plan.theme.ink};
      letter-spacing:-.025em; font-variant-numeric:tabular-nums; }}
  #{plan.id} .news-inline-stat-label {{ font:500 22px/1.3 {SANS}; color:{plan.theme.muted}; }}
  #{plan.id} .news-inline-rule {{ width:190px; height:4px; border-radius:3px; background:{accent}; }}
  #{plan.id} .news-inline-visual {{ width:48%; height:690px; display:flex; align-items:center; position:relative;
      perspective:1400px; z-index:2; }}
  #{plan.id} .news-inline-window {{ position:relative; width:100%; height:620px; overflow:visible;
      border:3px solid {_rgba(accent, .72)}; border-radius:24px; background:{media_background};
      box-shadow:0 34px 100px {_rgba(plan.theme.bg, .45)}; transform-style:preserve-3d; }}
  #{plan.id} .news-inline-media-crop {{ position:absolute; inset:0; overflow:hidden;
      border-radius:20px; background:{media_background}; }}
  #{plan.id} .news-inline-media {{ display:block; width:100%; height:100%; object-fit:{plan.news_image_fit};
      object-position:{object_position}; padding:{image_padding}; box-sizing:border-box; }}
  #{plan.id} .news-inline-credit {{ position:absolute; left:22px; right:22px; bottom:20px; z-index:4;
      font:500 18px {SANS}; line-height:1.3; letter-spacing:.035em; color:#F5F2EA;
      background:rgba(11,13,23,.78); border:1px solid {_rgba(accent, .48)};
      border-radius:999px; padding:11px 18px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  #{plan.id} .news-inline-corner {{ position:absolute; left:-18px; bottom:8px; width:94px; height:94px;
      border-left:3px solid {accent}; border-bottom:3px solid {accent}; opacity:.78; }}
"""
    items_markup = ""
    if plan.items:
        rows = "".join(
            f'          <li class="news-inline-item" id="{plan.id}-item-{index}">'
            f'<span class="news-inline-item-index">{index:02d}</span>'
            f'<span class="news-inline-item-text">{_esc(item)}</span></li>\n'
            for index, item in enumerate(plan.items, start=1)
        )
        items_markup = (
            f'        <ul class="news-inline-items{items_class}" id="{plan.id}-items">\n'
            + rows
            + "        </ul>\n"
        )
    stat_markup = ""
    if plan.stat:
        stat_markup = (
            f'        <div class="news-inline-stat" id="{plan.id}-stat">\n'
            f'          <div class="news-inline-stat-value" id="{plan.id}-stat-value">'
            f'{_esc(plan.stat)}</div>\n'
            + (
                f'          <div class="news-inline-stat-label" id="{plan.id}-stat-label">'
                f'{_esc(plan.stat_label)}</div>\n'
                if plan.stat_label
                else ""
            )
            + "        </div>\n"
        )
    markup = (
        '    <div class="plate"><div class="wash"></div></div>\n'
        + _motif_block(
            plan,
            style="left:-120px; top:-140px; width:620px; height:620px; opacity:.16;",
        )
        + f'    <div class="stage news-inline-stage" id="{plan.id}-inline-stage">\n'
        + f'      <div class="news-inline-copy{copy_class}">\n'
        + (f'        <div class="kicker" id="{plan.id}-kicker">{_esc(plan.kicker)}</div>\n' if plan.kicker else "")
        + f'        <div class="headline news-inline-headline" id="{plan.id}-head">{_esc(plan.headline)}</div>\n'
        + f'        <div class="news-inline-rule" id="{plan.id}-rule"></div>\n'
        + (f'        <div class="body news-inline-body" id="{plan.id}-body">{_esc(plan.body)}</div>\n' if plan.body else "")
        + stat_markup
        + items_markup
        + '      </div>\n'
        + f'      <div class="news-inline-visual" id="{plan.id}-visual" data-layout-allow-overflow>\n'
        + f'        <div class="news-inline-window" id="{plan.id}-image-frame">\n'
        + '          <div class="news-inline-media-crop" data-layout-allow-overflow>\n'
        + f'            <img class="news-inline-media" id="{plan.id}-image" src="{_esc(plan.news_image_src)}" '
        + f'alt="{_esc(plan.news_image_caption)}" crossorigin="anonymous">\n'
        + '          </div>\n'
        + (f'          <div class="news-inline-credit" id="{plan.id}-credit">{_esc(plan.news_image_credit)}</div>\n' if plan.news_image_credit else "")
        + '        </div>\n'
        + f'        <div class="news-inline-corner" id="{plan.id}-corner"></div>\n'
        + '      </div>\n'
        + '    </div>\n'
        + '    <div class="vignette"></div>\n'
    )
    drift = max(2.2, plan.duration - 0.35)
    direction = -1 if _seed_from(plan.id) % 2 else 1
    timeline = f"""        inAt("#{plan.id}-kicker", {{ x: -42, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .52, ease: "power3.out" }}, 0.16);
        inAt("#{plan.id}-head", {{ y: 58, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .82, ease: "expo.out" }}, 0.28);
        inAt("#{plan.id}-rule", {{ scaleX: 0, transformOrigin: "0 50%" }}, {{ scaleX: 1, duration: .64, ease: "power2.inOut" }}, 0.58);
        inAt("#{plan.id}-body", {{ y: 26, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .66, ease: "sine.out" }}, 0.68);
        inAt("#{plan.id}-stat", {{ y: 22, opacity: 0, scale: .97 }}, {{ y: 0, opacity: 1, scale: 1, duration: .62, ease: "circ.out", transformOrigin: "0 50%" }}, 0.72);
        inAt("#{plan.id}-items .news-inline-item", {{ x: -24, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .52, ease: "power2.out", stagger: .1 }}, 0.76);
        inAt("#{plan.id}-image-frame", {{ x: {direction * 220}, opacity: 0, rotationY: {direction * -11}, scale: .94 }}, {{ x: 0, opacity: 1, rotationY: 0, scale: 1, duration: .92, ease: "power4.out", transformPerspective: 1400, transformOrigin: "50% 50%" }}, 0.24);
        inAt("#{plan.id}-image", {{ scale: {1 if contain else 1.08}, x: {0 if contain else direction * -12} }}, {{ scale: {1 if contain else 1.02}, x: {0 if contain else direction * 12}, duration: {drift:.2f}, ease: "none", transformOrigin: "50% 50%" }}, 0.18);
        inAt("#{plan.id}-corner", {{ scale: .5, opacity: 0, transformOrigin: "0 100%" }}, {{ scale: 1, opacity: .78, duration: .7, ease: "circ.out" }}, 0.66);
        inAt("#{plan.id}-credit", {{ y: 14, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .5, ease: "power1.out" }}, 0.92);
"""
    return _shell(plan, css=css, markup=markup, timeline=timeline, wash=(22, 40))


def _render_news_image_collage(plan: ScenePlan) -> str:
    """Fill the frame with three overlapping sourced images and no story copy."""
    sources = tuple(dict.fromkeys(plan.news_image_srcs))[:3]
    if len(sources) < 3:
        return _render_news_image_fullscreen(plan)
    credits = tuple(plan.news_image_credits)
    accent = accent_hex(plan.accent, plan.theme)
    placements = (
        ("top", "left:560px; top:54px; width:800px; height:400px; z-index:6;", -1.2),
        ("left", "left:76px; top:338px; width:1080px; height:632px; z-index:4;", 0.8),
        ("right", "right:72px; top:304px; width:820px; height:610px; z-index:5;", -0.7),
    )
    css = f"""
  #{plan.id} .news-collage-canvas {{ position:absolute; inset:0; overflow:hidden; background:#0B090B; }}
  #{plan.id} .news-collage-glow {{ position:absolute; inset:-18%; background:
      radial-gradient(circle at 72% 38%, {_rgba(accent, .24)} 0%, rgba(11,9,11,0) 42%),
      radial-gradient(circle at 22% 72%, rgba(255,255,255,.12) 0%, rgba(11,9,11,0) 38%); }}
  #{plan.id} .news-collage-panel {{ position:absolute; overflow:hidden; background:#161218;
      border:3px solid rgba(255,255,255,.84); box-shadow:0 34px 100px rgba(0,0,0,.62);
      transform-style:preserve-3d; }}
  #{plan.id} .news-collage-crop {{ position:absolute; inset:0; overflow:hidden; }}
  #{plan.id} .news-collage-image {{ display:block; width:100%; height:100%; object-fit:cover; }}
  #{plan.id} .news-collage-credit {{ position:absolute; left:16px; right:16px; bottom:14px; z-index:2;
      padding:8px 12px; color:rgba(245,242,234,.88); background:rgba(8,9,15,.76);
      font:500 15px {SANS}; line-height:1.2; letter-spacing:.025em; white-space:nowrap;
      overflow:hidden; text-overflow:ellipsis; }}
  #{plan.id} .news-collage-rail {{ position:absolute; left:0; right:0; bottom:0; height:12px;
      z-index:7; background:{accent}; }}
"""
    panel_markup = []
    timeline_rows = []
    entrances = ((-170, -72), (-190, 94), (190, 52))
    eases = ("power4.out", "expo.out", "circ.out")
    starts = (0.18, 0.38, 0.58)
    for index, ((name, placement, rotation), source) in enumerate(
        zip(placements, sources, strict=True), start=1
    ):
        credit = credits[index - 1] if index - 1 < len(credits) else ""
        fit = plan.news_image_fits[index - 1] if index <= len(plan.news_image_fits) else plan.news_image_fit
        contain = fit == "contain"
        panel_markup.append(
            f'    <div class="news-collage-panel news-collage-{name}" '
            f'id="{plan.id}-collage-panel-{index}" style="{placement} transform:rotate({rotation}deg)" '
            'data-layout-allow-overflow>\n'
            f'      <div class="news-collage-crop"><img class="news-collage-image" '
            f'id="{plan.id}-collage-image-{index}" src="{_esc(source)}" alt="" '
            f'style="object-fit:{fit}" crossorigin="anonymous"></div>\n'
            + (
                f'      <div class="news-collage-credit" id="{plan.id}-collage-credit-{index}">'
                f'{_esc(credit)}</div>\n'
                if credit
                else ""
            )
            + "    </div>\n"
        )
        x, y = entrances[index - 1]
        direction = -1 if index % 2 else 1
        timeline_rows.append(
            f'        inAt("#{plan.id}-collage-panel-{index}", '
            f'{{ x: {x}, y: {y}, opacity: 0, scale: .9, rotationZ: {rotation + direction * 3:.1f} }}, '
            f'{{ x: 0, y: 0, opacity: 1, scale: 1, rotationZ: {rotation:.1f}, duration: '
            f'{0.82 + index * 0.09:.2f}, ease: "{eases[index - 1]}", transformPerspective: 1500, '
            f'transformOrigin: "50% 50%" }}, {starts[index - 1]:.2f});'
        )
        timeline_rows.append(
            f'        inAt("#{plan.id}-collage-image-{index}", '
            f'{{ scale: {1 if contain else 1.08}, x: {0 if contain else direction * -12} }}, '
            f'{{ scale: {1 if contain else 1.02}, x: {0 if contain else direction * 12}, duration: {max(2.4, plan.duration - starts[index - 1]):.2f}, '
            'ease: "none", transformOrigin: "50% 50%" }, '
            f'{starts[index - 1]:.2f});'
        )
        if credit:
            timeline_rows.append(
                f'        inAt("#{plan.id}-collage-credit-{index}", {{ y: 18, opacity: 0 }}, '
                f'{{ y: 0, opacity: 1, duration: .42, ease: "sine.out" }}, '
                f'{starts[index - 1] + 0.5:.2f});'
            )
    markup = (
        f'    <div class="news-collage-canvas" id="{plan.id}-collage-canvas">\n'
        f'      <div class="news-collage-glow" id="{plan.id}-collage-glow" '
        'data-layout-allow-overflow></div>\n'
        + "".join(panel_markup)
        + f'      <div class="news-collage-rail" id="{plan.id}-collage-rail"></div>\n'
        + "    </div>\n"
    )
    timeline = (
        f'        inAt("#{plan.id}-collage-glow", {{ opacity: 0, scale: .9 }}, '
        f'{{ opacity: 1, scale: 1.04, duration: {max(2.8, plan.duration):.2f}, ease: "sine.inOut", '
        'transformOrigin: "50% 50%" }, 0.1);\n'
        + "\n".join(timeline_rows)
        + f'\n        inAt("#{plan.id}-collage-rail", {{ scaleX: 0, transformOrigin: "0 50%" }}, '
        f'{{ scaleX: 1, duration: .72, ease: "power3.inOut" }}, 0.72);\n'
    )
    return _shell(plan, css=css, markup=markup, timeline=timeline, wash=(50, 50))


def _render_news_image_fullscreen(plan: ScenePlan) -> str:
    """Give one factual still the frame with a smooth, seekable reveal."""
    if len(tuple(dict.fromkeys(plan.news_image_srcs))) >= 3:
        return _render_news_image_collage(plan)
    accent = accent_hex(plan.accent, plan.theme)
    body_copy = plan.body.strip()
    support_copy = body_copy or plan.quote.strip()
    support_class = " news-full-quote-support" if not body_copy and support_copy else ""
    contain = plan.news_image_fit == "contain"
    media_background = plan.theme.ink if contain else plan.theme.bg
    image_padding = "150px 220px 260px" if contain else "0"
    hsize = headline_size(plan.headline, base=88, floor=50)
    css = f"""
  #{plan.id} .news-full-frame {{ position:absolute; inset:0; overflow:hidden; background:{media_background};
      transform-style:preserve-3d; }}
  #{plan.id} .news-full-media {{ display:block; width:100%; height:100%; object-fit:{plan.news_image_fit};
      padding:{image_padding}; box-sizing:border-box; }}
  #{plan.id} .news-full-scrim {{ position:absolute; inset:0; background:
      linear-gradient(90deg, {_rgba(plan.theme.bg, .92)} 0%, {_rgba(plan.theme.bg, .5)} 48%, {_rgba(plan.theme.bg, .08)} 76%),
      linear-gradient(0deg, {_rgba(plan.theme.bg, .94)} 0%, {_rgba(plan.theme.bg, 0)} 62%); }}
  #{plan.id} .news-full-stage {{ justify-content:flex-end; padding:130px 150px 224px; gap:22px; z-index:3; }}
  #{plan.id} .news-full-headline {{ font-size:{hsize}px; max-width:1240px; color:{plan.theme.ink};
      text-shadow:0 2px 16px {_rgba(plan.theme.bg, .9)}; }}
  #{plan.id} .news-full-body {{ font-size:{body_size(support_copy, base=34)}px; max-width:990px;
      color:{plan.theme.ink}; text-shadow:0 2px 12px {_rgba(plan.theme.bg, .9)}; }}
  #{plan.id} .news-full-rule {{ width:220px; height:5px; border-radius:3px; background:{accent}; }}
  #{plan.id} .news-full-credit {{ position:absolute; top:38px; right:44px; max-width:820px; z-index:4;
      font:500 18px {SANS}; line-height:1.3; letter-spacing:.035em; color:#F5F2EA;
      background:rgba(11,13,23,.76); border:1px solid {_rgba(accent, .46)};
      border-radius:999px; padding:11px 18px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  #{plan.id} .news-full-marker {{ position:absolute; left:44px; top:40px; z-index:4;
      font:700 20px {SANS}; letter-spacing:.2em; color:{accent}; text-transform:uppercase; }}
  #{plan.id} .news-full-rail {{ position:absolute; left:0; top:0; bottom:0; width:10px; background:{accent}; z-index:5; }}
"""
    markup = (
        f'    <div class="news-full-frame" id="{plan.id}-image-frame" data-layout-allow-overflow>\n'
        + f'      <img class="news-full-media" id="{plan.id}-image" src="{_esc(plan.news_image_src)}" '
        + f'alt="{_esc(plan.news_image_caption)}" crossorigin="anonymous">\n'
        + '    </div>\n'
        + '    <div class="news-full-scrim"></div>\n'
        + f'    <div class="news-full-marker" id="{plan.id}-marker">News image</div>\n'
        + (f'    <div class="news-full-credit" id="{plan.id}-credit">{_esc(plan.news_image_credit)}</div>\n' if plan.news_image_credit else "")
        + f'    <div class="news-full-rail" id="{plan.id}-rail"></div>\n'
        + f'    <div class="stage news-full-stage" id="{plan.id}-full-stage">\n'
        + (f'      <div class="kicker" id="{plan.id}-kicker">{_esc(plan.kicker)}</div>\n' if plan.kicker else "")
        + f'      <div class="headline news-full-headline" id="{plan.id}-head">{_esc(plan.headline)}</div>\n'
        + f'      <div class="news-full-rule" id="{plan.id}-rule"></div>\n'
        + (
            f'      <div class="body news-full-body{support_class}" id="{plan.id}-body">'
            f'{_esc(support_copy)}</div>\n'
            if support_copy
            else ""
        )
        + '    </div>\n'
    )
    drift = max(2.4, plan.duration - 0.3)
    direction = -1 if _seed_from(plan.id) % 2 else 1
    timeline = f"""        inAt("#{plan.id}-image-frame", {{ clipPath: "inset(0 {100 if direction > 0 else 0}% 0 {0 if direction > 0 else 100}%)", x: {direction * 120}, rotationY: {direction * -6} }}, {{ clipPath: "inset(0 0% 0 0%)", x: 0, rotationY: 0, duration: .96, ease: "power3.inOut", transformPerspective: 1600, transformOrigin: "50% 50%" }}, 0.12);
        inAt("#{plan.id}-image", {{ scale: {1 if contain else 1.08}, x: {0 if contain else direction * -16} }}, {{ scale: {1 if contain else 1.025}, x: {0 if contain else direction * 16}, duration: {drift:.2f}, ease: "none", transformOrigin: "50% 50%" }}, 0.15);
        inAt("#{plan.id}-rail", {{ scaleY: 0, transformOrigin: "50% 0" }}, {{ scaleY: 1, duration: .62, ease: "expo.out" }}, 0.18);
        inAt("#{plan.id}-marker", {{ x: -32, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .5, ease: "power2.out" }}, 0.48);
        inAt("#{plan.id}-kicker", {{ x: -38, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .54, ease: "power3.out" }}, 0.42);
        inAt("#{plan.id}-head", {{ y: 62, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .86, ease: "expo.out" }}, 0.58);
        inAt("#{plan.id}-rule", {{ scaleX: 0, transformOrigin: "0 50%" }}, {{ scaleX: 1, duration: .62, ease: "circ.out" }}, 0.88);
        inAt("#{plan.id}-body", {{ y: 24, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .66, ease: "sine.out" }}, 0.96);
        inAt("#{plan.id}-credit", {{ y: -14, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .5, ease: "power1.out" }}, 0.74);
"""
    return _shell(plan, css=css, markup=markup, timeline=timeline, wash=(50, 50))


def _render_title(plan: ScenePlan) -> str:
    accent = accent_hex(plan.accent, plan.theme)
    hsize = headline_size(plan.headline, base=124, floor=58)
    css = f"""
  #{plan.id} .stage {{ align-items:center; text-align:center; padding:150px 180px; gap:34px; }}
  #{plan.id} .kicker {{ justify-content:center; }}
  #{plan.id} .headline {{ font-size:{hsize}px; max-width:1520px; }}
  #{plan.id} .body {{ font-size:38px; max-width:1100px; }}
  #{plan.id} .rule {{ width:0; height:4px; background:{accent}; border-radius:2px; }}
"""
    markup = (
        '    <div class="plate"><div class="wash"></div></div>\n'
        + _motif_block(plan, style="left:50%; top:50%; width:1250px; height:1250px; margin:-625px 0 0 -625px; opacity:.22;")
        + '    <div class="stage">\n'
        + (f'      <div class="kicker" id="{plan.id}-kicker">{_esc(plan.kicker)}</div>\n' if plan.kicker else "")
        + f'      <div class="headline" id="{plan.id}-head">{_esc(plan.headline)}</div>\n'
        + f'      <div class="rule" id="{plan.id}-rule"></div>\n'
        + (f'      <div class="body" id="{plan.id}-body">{_esc(plan.body)}</div>\n' if plan.body else "")
        + '    </div>\n'
        + '    <div class="vignette"></div>\n'
    )
    timeline = f"""        inAt("#{plan.id}-motif", {{ scale: .82, opacity: 0, rotate: -14 }}, {{ scale: 1, opacity: .22, rotate: 0, duration: 2.4, ease: "power2.out", transformOrigin: "50% 50%" }}, 0);
        inAt("#{plan.id}-kicker", {{ y: -26, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .6, ease: "power2.out" }}, 0.25);
        inAt("#{plan.id}-head", {{ y: 66, opacity: 0 }}, {{ y: 0, opacity: 1, duration: 1.0, ease: "expo.out" }}, 0.4);
        inAt("#{plan.id}-rule", {{ width: 0 }}, {{ width: 260, duration: .8, ease: "power3.inOut" }}, 0.9);
        inAt("#{plan.id}-body", {{ y: 24, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .7, ease: "power2.out" }}, 1.05);
"""
    return _shell(plan, css=css, markup=markup, timeline=timeline, wash=(50, 38))


_RENDERERS = {
    "title": _render_title,
    "statement": _render_statement,
    "topic": _render_topic,
    "contrast": _render_contrast,
    "list": _render_list,
    "stat": _render_stat,
    "quote": _render_quote,
    "footage": _render_footage,
    "news_image": _render_news_image_fullscreen,
    "intro": _render_intro,
    "outro": _render_outro,
}


def _render_media_shots(plan: ScenePlan) -> str:
    """Render AI-ordered shots, never a video overlay left on an empty plate."""
    accent = accent_hex(plan.accent, plan.theme)
    ink, bg = plan.theme.ink, plan.theme.bg
    light = plan.theme.name in {"swiss", "shanshui"}
    label_ink = ink if light else accent
    card = _rgba(ink, .065 if light else .09)
    sid = plan.id
    css = f"""
  #{sid} .shot-media {{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }}
  #{sid} .shot-media.article {{ object-fit:contain; background:{bg}; padding:48px 100px 160px; box-sizing:border-box; }}
  #{sid} .shot-media.contained {{ object-fit:contain; background:{bg}; padding:56px 100px 110px; box-sizing:border-box; }}
  #{sid} .shot-media.contained-photo {{ width:46%; padding:56px 32px 100px 64px; }}
  #{sid} .shot-panel {{ position:absolute; inset:0; padding:76px 110px 180px; box-sizing:border-box; font-family:{SANS}; }}
  #{sid} .shot-panel.editorial {{ background:{bg}; color:{ink}; display:flex; flex-direction:column; justify-content:center; }}
  #{sid} .shot-panel.visual {{ color:#F5F2EA; background:linear-gradient(0deg,rgba(9,11,19,.92),rgba(9,11,19,0) 76%); display:flex; flex-direction:column; justify-content:flex-end; }}
  #{sid} .shot-panel.contained-photo {{ left:46%; padding:100px 80px 160px 40px; justify-content:center; background:{bg}; color:{ink}; }}
  #{sid} .contained-photo .shot-credit {{ top:34px; right:32px; max-width:calc(100% - 64px); box-sizing:border-box; }}
  #{sid} .shot-kicker {{ font-size:25px; text-transform:uppercase; letter-spacing:.18em; font-weight:700; margin-bottom:22px; color:{label_ink}; }}
  #{sid} .visual .shot-kicker {{ color:#F5F2EA; }}
  #{sid} .shot-title {{ font-size:{headline_size(plan.headline, base=86, floor=54)}px; line-height:1.08; font-weight:800; letter-spacing:-.025em; max-width:1530px; margin:0 0 35px; }}
  #{sid} .visual .shot-title {{ max-width:1400px; margin-bottom:24px; font-size:66px; }}
  #{sid} .shot-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:22px; }}
  #{sid} .shot-grid.cards .shot-card:last-child:nth-child(odd) {{ grid-column:1 / -1; }}
  #{sid} .shot-card {{ position:relative; background:{card}; border-left:5px solid {accent}; padding:30px 34px; min-height:126px; display:flex; align-items:center; gap:24px; }}
  #{sid} .shot-number {{ color:{label_ink}; font-family:Georgia,serif; font-size:54px; font-variant-numeric:tabular-nums; flex-shrink:0; }}
  #{sid} .shot-copy {{ font-size:34px; line-height:1.3; overflow-wrap:anywhere; }}
  #{sid} .shot-card.dense .shot-copy {{ font-size:28px; }}
  #{sid} .shot-grid.focus {{ grid-template-columns:1fr; }}
  #{sid} .focus .shot-card {{ padding:42px; min-height:180px; }}
  #{sid} .focus .shot-copy {{ font-size:46px; max-width:1400px; }}
  #{sid} .focus .dense .shot-copy {{ font-size:34px; }}
  #{sid} .shot-grid.split {{ grid-template-columns:1.25fr 1fr; }}
  #{sid} .visual .shot-grid {{ display:block; max-width:1370px; }}
  #{sid} .visual .shot-card {{ padding:0; margin-top:16px; border:0; min-height:0; background:none; }}
  #{sid} .visual .shot-number {{ display:none; }}
  #{sid} .contained-photo .shot-kicker {{ color:{label_ink}; }}
  #{sid} .contained-photo .shot-title {{ font-size:54px; }}
  #{sid} .contained-photo .shot-copy {{ font-size:30px; }}
  #{sid} .shot-credit {{ position:absolute; top:34px; right:48px; max-width:1100px; border-radius:16px; padding:9px 18px; background:rgba(9,11,19,.82); color:#F5F2EA; font-size:20px; }}
  #{sid} .shot-rule {{ height:5px; width:140px; background:{accent}; margin-bottom:24px; transform-origin:left; }}
  #{sid} .shot-orbit {{ position:absolute; width:820px; height:820px; right:-280px; top:-340px; border:95px solid {_rgba(accent,.17)}; border-radius:50%; pointer-events:none; }}
  #{sid} .shot-wipe {{ position:absolute; inset:0; background:{accent}; pointer-events:none; }}
"""
    if plan.frame.orientation == "portrait":
        css += f"""
  #{sid} .shot-panel {{ padding:130px 64px 300px; }}
  #{sid} .shot-title, #{sid} .visual .shot-title {{ font-size:66px; }}
  #{sid} .shot-grid, #{sid} .shot-grid.split {{ grid-template-columns:1fr; }}
  #{sid} .shot-credit {{ max-width:860px; right:35px; top:65px; }}
  #{sid} .shot-media.contained-photo {{ width:100%; height:58%; padding:70px 64px 28px; }}
  #{sid} .shot-panel.contained-photo {{ left:0; top:58%; padding:35px 64px 160px; justify-content:flex-start; }}
  #{sid} .contained-photo .shot-title {{ font-size:46px; }}
  #{sid} .contained-photo .shot-credit {{ position:static; order:10; margin-top:18px; max-width:100%; }}
"""
    markup = []
    timeline = []
    for index, shot in enumerate(plan.media_shots):
        prefix = f"{sid}-shot-{index + 1}"
        start, duration = float(shot["start"]), float(shot["duration"])
        kind = shot["kind"]
        editorial = kind == "editorial"
        contain = kind == "image" and shot.get("fit") == "contain"
        portrait_photo = contain and float(shot.get("height", 0)) > float(shot.get("width", 0))
        media_class = " contained" if contain else ""
        if portrait_photo:
            media_class += " contained-photo"
        # Adjacent shots use alternating lanes. This keeps exact centisecond
        # cuts intact even when the CLI adds decimal times as binary floats.
        media_track = 2 * (index % 2)
        panel_track = media_track + 1
        timing = f'data-start="{start:.2f}" data-duration="{duration:.2f}"'
        if not editorial:
            video = kind in {"public_footage", "paper_collage"}
            tag = "video" if video else "img"
            extra = (f'muted playsinline data-media-start="{float(shot.get("source_start", 0)):.2f}"'
                     if video else 'alt=""')
            if contain:
                extra += ' style="object-fit:contain"'
            markup.append(f'<{tag} id="{prefix}-media" class="clip shot-media {kind}{media_class}" '
                          f'src="{_esc(shot["src"])}" {timing} data-track-index="{media_track}" '
                          f'{extra} crossorigin="anonymous">' + ("</video>" if video else ""))
            if not video and not contain:
                timeline.append(f'inAt("#{prefix}-media", {{scale:1}}, {{scale:1.035, duration:{duration:.2f}, ease:"none"}}, {start:.2f});')
            elif contain:
                timeline.append(f'inAt("#{prefix}-media", {{opacity:0}}, {{opacity:1, duration:{min(.4, duration):.2f}, ease:"sine.out"}}, {start:.2f});')
        texts = [text for text in shot["copy"] if text != plan.headline]
        if not texts and editorial:
            texts = list(shot["copy"])
        title = plan.headline if editorial else shot["copy"][0]
        if not editorial:
            texts = shot["copy"][1:]
        cards = "".join(
            f'<div class="shot-card {"dense" if len(text) > 135 else ""}" id="{prefix}-card-{i}">'
            f'<span class="shot-number">{i + 1:02d}</span><div class="shot-copy">{_esc(text)}</div></div>'
            for i, text in enumerate(texts)
        )
        layout = shot.get("layout", "cards") if len(texts) > 1 else "focus"
        markup.append(
            f'<div id="{prefix}-panel" class="clip shot-panel {"editorial" if editorial else "visual"}{" contained-photo" if portrait_photo else ""}" '
            f'{timing} data-track-index="{panel_track}">'
            + (f'<div class="shot-orbit" id="{prefix}-orbit" data-layout-ignore></div>' if editorial else "")
            + f'<div class="shot-kicker" id="{prefix}-kicker">{_esc(plan.kicker)}</div>'
            + f'<div class="shot-rule" id="{prefix}-rule"></div>'
            + f'<div class="shot-title" id="{prefix}-title">{_esc(title)}</div>'
            + f'<div class="shot-grid {layout}">{cards}</div>'
            + (f'<div class="shot-credit">{_esc(shot["credit"])}</div>' if shot.get("credit") else "")
            + '</div>'
        )
        # Clip visibility is owned by HyperFrames. Animate inner content only;
        # no timed parent wraps a video and no video seeks beyond its source.
        entrance = min(.55, duration * .3)
        timeline += [
            f'inAt("#{prefix}-rule", {{scaleX:.15}}, {{scaleX:1,duration:{entrance:.2f},ease:"power3.out"}}, {start + .08:.2f});',
            f'inAt("#{prefix}-kicker", {{x:-20}}, {{x:0,duration:{entrance:.2f},ease:"sine.out"}}, {start + .1:.2f});',
            f'inAt("#{prefix}-title", {{y:24}}, {{y:0,duration:{entrance:.2f},ease:"expo.out"}}, {start + .1:.2f});',
        ]
        for i in range(len(texts)):
            timeline.append(f'inAt("#{prefix}-card-{i}", {{x:35,opacity:.35}}, {{x:0,opacity:1,duration:{entrance:.2f},ease:"power2.out"}}, {start + .12 + i * .06:.2f});')
        if editorial:
            timeline.append(f'inAt("#{prefix}-orbit", {{scale:.85}}, {{scale:1.08,duration:{duration:.2f},ease:"none"}}, {start:.2f});')
        if index:
            # A cover transition spans the cut; outgoing content remains intact
            # until the source ends, so even the boundary frame has content.
            half = min(.18, float(plan.media_shots[index - 1]["duration"]) / 3, duration / 3)
            at = max(0, start - half)
            markup.append(f'<div id="{prefix}-wipe" class="clip shot-wipe" data-start="{at:.3f}" data-duration="{half * 2:.3f}" data-track-index="4" data-layout-ignore></div>')
            if shot.get("transition") == "iris":
                timeline.append(f'inAt("#{prefix}-wipe", {{clipPath:"circle(0% at 50% 50%)"}}, {{clipPath:"circle(80% at 50% 50%)",duration:{half:.3f},ease:"power2.in"}}, {at:.3f});')
                timeline.append(f'outAt("#{prefix}-wipe", {{clipPath:"circle(0% at 50% 50%)",duration:{half:.3f},ease:"power2.out"}}, {start:.2f});')
            else:
                timeline.append(f'inAt("#{prefix}-wipe", {{xPercent:-100}}, {{xPercent:0,duration:{half:.3f},ease:"power2.in"}}, {at:.3f});')
                timeline.append(f'outAt("#{prefix}-wipe", {{xPercent:100,duration:{half:.3f},ease:"power2.out"}}, {start:.2f});')
    return _shell(plan, css=css, markup="\n".join(markup), timeline="\n".join(timeline), wash=(60, 40))


def render_scene(plan: ScenePlan) -> str:
    """Full sub-composition HTML for one scene."""
    if plan.media_shots:
        return _render_media_shots(plan)
    if plan.news_webpage_src and (plan.footage_src or plan.news_image_src):
        return _render_news_webpage_overlay(plan)
    if plan.news_image_src and plan.news_image_mode == "inline":
        return _render_news_image_inline(plan)
    if plan.news_image_src and plan.news_image_mode == "fullscreen":
        return _render_news_image_fullscreen(plan)
    if plan.archetype == "contrast" and not (plan.left_text and plan.right_text):
        plan.archetype = "topic"
    if plan.archetype == "list" and len(plan.items) < 2:
        plan.archetype = "topic"
    if plan.archetype == "footage" and not plan.footage_src:
        plan.archetype = "topic"
    if plan.archetype == "news_image" and not plan.news_image_src:
        plan.archetype = "topic"
    if plan.archetype == "intro" and not (plan.footage_src and plan.intro_logo_src):
        plan.archetype = "statement"
    if plan.archetype == "outro" and not (plan.footage_src and plan.outro_logo_src):
        plan.archetype = "statement"
    if plan.archetype == "stat" and not plan.stat:
        # Without a figure the renderer would blow a full sentence up to 130px.
        plan.archetype = "topic"
    if plan.archetype == "quote" and not (plan.quote or plan.headline):
        plan.archetype = "statement"
    renderer = _RENDERERS.get(plan.archetype, _render_topic)
    return renderer(plan)
