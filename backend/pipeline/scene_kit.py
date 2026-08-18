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
import math
from dataclasses import dataclass

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
    "outro",
)


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
    collage_broll: bool = False
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
        items = tuple(str(i) for i in (data.get("items") or [])[:4] if str(i).strip())
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
            collage_broll=bool(data.get("collage_broll")),
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


def _render_footage(plan: ScenePlan) -> str:
    """Render a full-bleed media plate that remains populated for the full scene."""
    accent = accent_hex(plan.accent, plan.theme)
    if plan.footage_kind == "video":
        media = (
            f'      <video id="{plan.id}-media" class="clip media" src="{_esc(plan.footage_src)}" '
            f'data-start="0" data-duration="{plan.duration:.2f}" data-track-index="0" '
            f'muted playsinline loop crossorigin="anonymous"></video>\n'
        )
    else:
        media = f'      <img id="{plan.id}-media" class="media" src="{_esc(plan.footage_src)}" alt="">\n'
    css = f"""
  #{plan.id} .frame {{ position:absolute; inset:0; overflow:hidden; }}
  #{plan.id} .media {{ width:100%; height:100%; object-fit:cover; }}
  #{plan.id} .scrim {{ position:absolute; inset:0;
      background:linear-gradient(180deg, {_rgba(plan.theme.bg, .55)} 0%, {_rgba(plan.theme.bg, .15)} 38%, {_rgba(plan.theme.bg, .92)} 100%); }}
  #{plan.id} .stage {{ justify-content:flex-end; padding:130px 150px 230px; }}
  #{plan.id} .headline {{ font-size:{headline_size(plan.headline, base=92)}px; max-width:1340px;
      color:#F5F2EA; text-shadow:0 8px 40px rgba(0,0,0,.6); }}
  #{plan.id} .body {{ font-size:{body_size(plan.body, base=38)}px; max-width:1100px; color:{plan.theme.muted}; }}
  #{plan.id} .credit {{ position:absolute; right:44px; top:40px; font:500 20px {SANS};
      color:{_rgba(plan.theme.ink, .62)}; letter-spacing:.06em;
      background:{_rgba(plan.theme.bg, .5)}; padding:9px 16px; border-radius:999px;
      border:1px solid {_rgba(accent, 0.3)}; }}
"""
    if plan.collage_broll:
        markup = f'    <div class="frame collage-frame" id="{plan.id}-frame">\n' + media + '    </div>\n'
        timeline = f"""        inAt("#{plan.id}-media", {{ opacity: 0 }}, {{ opacity: 1, duration: .12, ease: "power1.out" }}, 0.1);
"""
        return _shell(plan, css=css, markup=markup, timeline=timeline, wash=(50, 50))

    markup = (
        f'    <div class="frame" id="{plan.id}-frame">\n'
        + media
        + '    </div>\n'
        + '    <div class="scrim"></div>\n'
        + (f'    <div class="credit" id="{plan.id}-credit">{_esc(plan.footage_credit)}</div>\n'
           if plan.footage_credit else "")
        + '    <div class="stage">\n'
        + (f'      <div class="kicker" id="{plan.id}-kicker">{_esc(plan.kicker)}</div>\n' if plan.kicker else "")
        + (f'      <div class="headline" id="{plan.id}-head">{_esc(plan.headline)}</div>\n' if plan.headline else "")
        + (f'      <div class="body" id="{plan.id}-body">{_esc(plan.body)}</div>\n' if plan.body else "")
        + '    </div>\n'
    )
    timeline = f"""        inAt("#{plan.id}-media", {{ scale: 1.06, x: -14 }}, {{ scale: 1.16, x: 14, duration: {max(2.0, plan.duration):.2f}, ease: "none", transformOrigin: "50% 50%" }}, 0);
        inAt("#{plan.id}-kicker", {{ x: -36, opacity: 0 }}, {{ x: 0, opacity: 1, duration: .55, ease: "power3.out" }}, 0.25);
        inAt("#{plan.id}-head", {{ y: 56, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .85, ease: "expo.out" }}, 0.4);
        inAt("#{plan.id}-body", {{ y: 26, opacity: 0 }}, {{ y: 0, opacity: 1, duration: .65, ease: "power2.out" }}, 0.62);
        inAt("#{plan.id}-credit", {{ opacity: 0 }}, {{ opacity: 1, duration: .6, ease: "power1.out" }}, 0.8);
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


def _render_outro(plan: ScenePlan) -> str:
    out = _render_title(plan)
    # The final scene is the one place an exit animation is allowed, so fade the
    # whole stage after the hold instead of cutting to black on the last frame.
    # Mark its deliberate 26px upward travel so layout inspection does not
    # report the exit as content accidentally escaping the scene container.
    out = out.replace(
        '<div class="stage">',
        '<div class="stage" data-layout-allow-overflow>',
        1,
    )
    fade = (
        f'        outAt("#{plan.id} .stage", {{ opacity: 0, y: -26, duration: .8, ease: "power2.in" }}, '
        f'{max(1.2, plan.duration - 1.0):.2f});\n'
    )
    return out.replace('        window.__timelines[', fade + '        window.__timelines[')


_RENDERERS = {
    "title": _render_title,
    "statement": _render_statement,
    "topic": _render_topic,
    "contrast": _render_contrast,
    "list": _render_list,
    "stat": _render_stat,
    "quote": _render_quote,
    "footage": _render_footage,
    "outro": _render_outro,
}


def render_scene(plan: ScenePlan) -> str:
    """Full sub-composition HTML for one scene."""
    if plan.archetype == "contrast" and not (plan.left_text and plan.right_text):
        plan.archetype = "topic"
    if plan.archetype == "list" and len(plan.items) < 2:
        plan.archetype = "topic"
    if plan.archetype == "footage" and not plan.footage_src:
        plan.archetype = "topic"
    if plan.archetype == "stat" and not plan.stat:
        # Without a figure the renderer would blow a full sentence up to 130px.
        plan.archetype = "topic"
    if plan.archetype == "quote" and not (plan.quote or plan.headline):
        plan.archetype = "statement"
    renderer = _RENDERERS.get(plan.archetype, _render_topic)
    return renderer(plan)
