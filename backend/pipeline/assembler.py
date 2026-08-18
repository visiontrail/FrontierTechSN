"""Assemble the render project: spine, scene files, vendored assets, lint.

The spine (``index.html``) is generated, never authored by a model. It owns the
three things a wrong answer would silently break — the audio offset, the scene
mount times, and the HyperFrames runtime contract — so a creative failure in a
scene file costs one bad-looking scene instead of a blank 12-minute video.

Everything the earlier renders got wrong is enforced here:

* the root composition carries ``data-start="0"`` and a registered timeline;
* every timed element has ``class="clip"``, an ``id``, and a track index;
* clips on one track never overlap (each layer gets its own track);
* fonts and GSAP are local files, so nothing depends on the network at capture.
"""

from __future__ import annotations

import html
import logging
import re
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

from backend import config
from backend.pipeline import scene_kit
from backend.pipeline.process_logging import run_capture_logged
from backend.pipeline.video_format import FrameSpec, LANDSCAPE

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

# Track layout. Same-track clips may not overlap, so each concern gets its own
# track and adjacent scenes alternate between two tracks — that way a scene may
# hold a frame or two past its neighbour's start without tripping the linter.
TRACK_SCENE_A = 0
TRACK_SCENE_B = 1
TRACK_CAPTION = 4
TRACK_CHROME = 5
TRACK_CHARACTER = 6

LINT_TIMEOUT = 180
# `inspect` drives the composition in headless Chrome and samples the timeline,
# so it costs real time on a long episode — but it is the only check that sees
# text overflowing a card or two blocks landing on top of each other.
INSPECT_TIMEOUT = 900

CAPTION_MAX_WORDS = 6
CAPTION_MAX_CJK_CHARS = 28
CAPTION_MAX_DISPLAY_UNITS = 34.0


def _esc(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def vendor_assets(task_dir: Path, *, include_lottie: bool = False) -> None:
    """Copy the runtime libraries into the project so capture never hits a CDN.

    HyperFrames sandboxes asset access to the project directory, and the render
    host may be offline; a `<script src="https://...">` that fails leaves every
    timeline unregistered, which is exactly the failure mode that produced blank
    frames before.
    """
    vendor = task_dir / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)

    gsap_src = config.HYPERFRAME_DIR / "node_modules" / "gsap" / "dist" / "gsap.min.js"
    gsap_dst = vendor / "gsap.min.js"
    if gsap_src.exists():
        shutil.copyfile(gsap_src, gsap_dst)
    elif not gsap_dst.exists():
        raise FileNotFoundError(
            f"GSAP not found at {gsap_src}. Run `npm install gsap` in {config.HYPERFRAME_DIR}."
        )

    if include_lottie:
        lottie_src = (
            config.HYPERFRAME_DIR / "node_modules" / "lottie-web" / "build" / "player" / "lottie.min.js"
        )
        if lottie_src.exists():
            shutil.copyfile(lottie_src, vendor / "lottie.min.js")


def stage_character(task_dir: Path) -> str | None:
    """Copy the host-character Lottie into the project; return its relative path.

    Returns None when the asset or the player is missing, so the caller can drop
    the overlay rather than emit a broken reference.
    """
    source = config.PROJECT_ROOT / "assets" / "lottie" / "podcast_host.json"
    if not source.exists():
        return None
    if not (task_dir / "vendor" / "lottie.min.js").exists():
        return None
    dest = task_dir / "assets" / "podcast_host.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    return "assets/podcast_host.json"


def write_scene_files(task_dir: Path, plans: Sequence[scene_kit.ScenePlan]) -> list[Path]:
    """Render every scene plan to ``compositions/<scene-id>.html``."""
    comps = task_dir / "compositions"
    comps.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for plan in plans:
        path = comps / f"{plan.id}.html"
        path.write_text(scene_kit.render_scene(plan), encoding="utf-8")
        written.append(path)
    return written


def _display_units(text: str) -> float:
    """Approximate rendered width in ems for safe, deterministic grouping."""
    units = 0.0
    for char in text:
        if "\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff":
            units += 1.0
        elif char.isspace():
            units += 0.3
        else:
            units += 0.55
    return units


def _split_dense_caption(text: str, limit: int) -> list[str]:
    """Split unspaced CJK (and other dense text) at nearby punctuation."""
    chunks: list[str] = []
    remaining = text.strip()
    punctuation = "。！？；，、.!?;,"
    while len(remaining) > limit:
        window = remaining[: limit + 1]
        cut = max(window.rfind(mark) for mark in punctuation) + 1
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _caption_chunks(value: object) -> list[str]:
    """Create readable, single-line groups from a storyboard spoken line."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return []

    has_cjk = bool(re.search(r"[\u3400-\u9fff\uf900-\ufaff]", text))
    if " " not in text and has_cjk:
        return _split_dense_caption(text, CAPTION_MAX_CJK_CHARS)

    words = text.split(" ")
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and (
            len(current) >= CAPTION_MAX_WORDS
            or _display_units(candidate) > CAPTION_MAX_DISPLAY_UNITS
        ):
            chunks.append(" ".join(current))
            current = []

        # A single unbroken token can still exceed the safe width (for example,
        # CJK text mixed with a Latin speaker label). Split it rather than
        # relying on clipping as the primary behaviour.
        if not current and _display_units(word) > CAPTION_MAX_DISPLAY_UNITS:
            limit = CAPTION_MAX_CJK_CHARS if has_cjk else int(CAPTION_MAX_DISPLAY_UNITS / 0.55)
            parts = _split_dense_caption(word, limit)
            chunks.extend(parts[:-1])
            current = parts[-1:]
        else:
            current.append(word)

        if len(current) >= 3 and re.search(r"[.!?。！？；;][\"']?$", current[-1]):
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))
    return chunks


def _caption_clips(storyboard: dict) -> list[str]:
    """Timed single-line caption groups, clamped so neighbours never overlap."""
    lines: list[dict] = []
    for scene in storyboard.get("scenes", []):
        lines.extend(scene.get("lines", []))
    lines.sort(key=lambda line: line["start"])

    segments: list[dict] = []
    for i, line in enumerate(lines):
        line_start = float(line["start"])
        line_end = line_start + max(0.02, float(line["duration"]))
        if i + 1 < len(lines):
            # Silence-map rounding can leave adjacent spoken lines overlapping.
            line_end = min(line_end, max(line_start + 0.02, float(lines[i + 1]["start"]) - 0.02))

        chunks = _caption_chunks(line.get("text", ""))
        if not chunks:
            continue
        weights = [max(1.0, _display_units(chunk)) for chunk in chunks]
        total_weight = sum(weights)
        elapsed_weight = 0.0
        for chunk, weight in zip(chunks, weights):
            start = line_start + (line_end - line_start) * elapsed_weight / total_weight
            elapsed_weight += weight
            end = line_start + (line_end - line_start) * elapsed_weight / total_weight
            segments.append({"start": start, "end": end, "text": chunk})

    out: list[str] = []
    for i, segment in enumerate(segments):
        start = round(float(segment["start"]), 2)
        end = round(float(segment["end"]), 2)
        next_start = round(float(segments[i + 1]["start"]), 2) if i + 1 < len(segments) else None
        if next_start is not None:
            end = min(end, next_start - 0.02)
        duration = round(max(0.02, end - start), 2)
        text = _esc(segment["text"])
        out.append(
            f'      <div id="cap-{i:03d}" class="clip caption" data-start="{start}" '
            f'data-duration="{duration}" data-track-index="{TRACK_CAPTION}">'
            f'<span class="caption-inner">{text}</span></div>'
        )
    return out


def _scene_mounts(mounts: Sequence[dict]) -> list[str]:
    """Mount tags for every scene, alternating tracks so neighbours can abut.

    Adjacent scenes share a boundary time. Putting them on the same track makes
    a one-hundredth-of-a-second rounding overlap a lint error, so odd and even
    scenes alternate between two tracks — visual layering is unaffected because
    HyperFrames uses CSS z-index for that, not the track index.
    """
    tags: list[str] = []
    for i, mount in enumerate(mounts):
        track = TRACK_SCENE_A if i % 2 == 0 else TRACK_SCENE_B
        tags.append(
            f'      <div id="mount-{mount["id"]}" class="clip scene-mount"\n'
            f'           data-composition-id="{mount["id"]}"\n'
            f'           data-composition-src="compositions/{mount["id"]}.html"\n'
            f'           data-start="{round(float(mount["start"]), 2)}" '
            f'data-duration="{round(float(mount["duration"]), 2)}"\n'
            f'           data-track-index="{track}"></div>'
        )
    return tags


def _spine_css(theme: scene_kit.Theme, frame: FrameSpec = LANDSCAPE) -> str:
    paper_style = theme.name == "shanshui"
    caption_radius = "5px" if paper_style else "14px"
    caption_shadow = (
        "0 14px 42px rgba(78,55,31,.16)" if paper_style
        else "0 12px 44px rgba(0,0,0,.45)"
    )
    caption_border = "2px solid rgba(78,105,90,.18)" if paper_style else "0 solid transparent"
    progress_from = scene_kit.accent_hex("amber", theme)
    progress_to = scene_kit.accent_hex("teal" if paper_style else "coral", theme)
    brand_alpha = 0.78 if paper_style else 0.45
    portrait = ""
    if frame.is_portrait:
        portrait = """
      .caption { left:70px; right:70px; bottom:122px; }
      .caption-inner { font-size:38px; max-width:920px; padding:16px 24px; }
      .character { right:42px; bottom:188px; width:230px; height:230px; }
      .brand { left:38px; bottom:44px; }
"""
    return f"""
      * {{ margin:0; padding:0; box-sizing:border-box; }}
      html, body {{ width:{frame.width}px; height:{frame.height}px; overflow:hidden; background:{theme.bg}; }}
      .scene-mount {{ position:absolute; inset:0; z-index:1; isolation:isolate; }}

      .caption {{
        position:absolute; left:210px; right:210px; bottom:96px;
        display:flex; justify-content:center; align-items:flex-end;
        text-align:center; pointer-events:none; z-index:20;
      }}
      .caption-inner {{
        font:500 34px {scene_kit.SANS}; line-height:1.36;
        color:{theme.caption_ink};
        background:{theme.caption_bg};
        padding:14px 30px; border-radius:{caption_radius};
        border:{caption_border}; box-shadow:{caption_shadow};
        max-width:1360px; white-space:nowrap;
        overflow:hidden; text-overflow:ellipsis;
      }}

      .chrome {{ position:absolute; inset:0; pointer-events:none; z-index:40; }}
      .progress-track {{
        position:absolute; left:0; right:0; bottom:0; height:6px;
        background:rgba(128,128,128,.18);
      }}
      .progress-fill {{
        height:100%; width:100%; transform-origin:0 50%;
        background:linear-gradient(90deg, {progress_from}, {progress_to});
      }}
      .character {{ position:absolute; right:64px; bottom:132px; width:210px; height:210px; z-index:30;
        pointer-events:none; }}
      .character #character-stage {{ width:100%; height:100%; }}
      .brand {{
        position:absolute; left:56px; bottom:36px;
        font:600 22px {scene_kit.SANS}; letter-spacing:.18em; text-transform:uppercase;
        color:{scene_kit._rgba(theme.ink, brand_alpha)};
      }}
{portrait}
"""


def build_spine(
    storyboard: dict,
    *,
    audio_src: str,
    mounts: Sequence[dict],
    brand: str = "",
    theme: scene_kit.Theme = scene_kit.DEFAULT_THEME,
    frame: FrameSpec = LANDSCAPE,
    character_src: str | None = None,
    captions_enabled: bool = True,
) -> str:
    """The root composition: mounts, captions, audio, progress chrome.

    ``mounts`` is the full cover of the timeline — every storyboard scene from
    zero, then the outro — as ``{"id", "start", "duration"}`` records. Any gap
    left here renders as a black frame, so the composer builds the list to span
    0 to ``total_duration`` with no holes.
    """
    total = float(storyboard["total_duration"])
    content_start = float(storyboard["content_start"])
    audio_duration = float(storyboard["audio_duration"])

    mount_tags = _scene_mounts(mounts)
    captions = _caption_clips(storyboard) if captions_enabled else []

    brand_html = (
        f'        <div class="brand">{_esc(brand)}</div>\n' if brand else ""
    )

    # Optional animated host, driven by the HyperFrames lottie adapter: the
    # player is registered on window.__hfLottie and seeked by the runtime, so it
    # must be created with autoplay off.
    character_html = ""
    character_script = ""
    if character_src:
        character_html = (
            f'      <div id="character" class="clip character" data-start="{content_start}" '
            f'data-duration="{audio_duration}" data-track-index="{TRACK_CHARACTER}">'
            f'<div id="character-stage"></div></div>\n'
        )
        character_script = f"""
    <script src="vendor/lottie.min.js"></script>
    <script>
      window.__hfLottie = window.__hfLottie || [];
      window.__hfLottie.push(lottie.loadAnimation({{
        container: document.getElementById("character-stage"),
        renderer: "svg", loop: true, autoplay: false,
        path: "{_esc(character_src)}",
      }}));
    </script>"""

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width={frame.width}, height={frame.height}" />
    <style>{_spine_css(theme, frame)}    </style>
  </head>
  <body>
    <div id="root" data-composition-id="root" data-start="0" data-duration="{total}"
         data-width="{frame.width}" data-height="{frame.height}"
         style="position:relative;width:{frame.width}px;height:{frame.height}px;overflow:hidden;">

{chr(10).join(mount_tags)}

{chr(10).join(captions)}

{character_html}      <div id="chrome" class="clip chrome" data-start="0" data-duration="{total}"
           data-track-index="{TRACK_CHROME}">
{brand_html}        <div class="progress-track"><div id="progress-fill" class="progress-fill"></div></div>
      </div>

      <audio id="narration" src="{_esc(audio_src)}" data-start="{content_start}"
             data-duration="{audio_duration}" data-track-index="9" data-volume="1"></audio>
    </div>

    <script src="vendor/gsap.min.js"></script>
    <script>
      window.__timelines = window.__timelines || {{}};
      (function () {{
        const tl = gsap.timeline({{ paused: true }});
        tl.fromTo("#progress-fill", {{ scaleX: 0 }}, {{ scaleX: 1, duration: {total}, ease: "none" }}, 0);
        window.__timelines["root"] = tl;
      }})();
    </script>{character_script}
  </body>
</html>
"""


def write_spine(task_dir: Path, html_text: str) -> Path:
    path = task_dir / "index.html"
    path.write_text(html_text, encoding="utf-8")
    # A stale entry file from an older run trips the multiple_root_compositions
    # check and can duplicate the audio track.
    stale = task_dir / "composition.html"
    if stale.exists():
        stale.unlink()
    return path


def _hyperframes_bin() -> list[str]:
    local = config.HYPERFRAME_DIR / "node_modules" / ".bin" / "hyperframes"
    if local.exists():
        return [str(local)]
    return ["npx", "--yes", f"hyperframes@{config.HYPERFRAMES_VERSION}"]


def lint_project(task_dir: Path, log: LogCallback | None = None) -> tuple[bool, str]:
    """Run ``hyperframes lint``; return (clean, output).

    Warnings are tolerated — errors are not. A composition that lints with an
    error is exactly the composition that renders blank, so the caller treats a
    False here as a repair trigger rather than a cosmetic note.
    """
    result = run_capture_logged(
        name="HyperFrames lint",
        command=_hyperframes_bin() + ["lint", str(task_dir)],
        logger=logger,
        log=None,
        timeout=LINT_TIMEOUT,
    )
    output = f"{result.stdout}\n{result.stderr}".strip()
    errors = [line.strip() for line in output.splitlines() if "✗" in line]
    clean = not errors
    if log:
        if clean:
            log(f"Lint: clean ({output.count('⚠')} warning(s))")
        else:
            log(f"Lint: {len(errors)} error(s)")
            for line in errors[:12]:
                log(f"  {line}")
    return clean, output


def inspect_project(task_dir: Path, log: LogCallback | None = None) -> tuple[bool, list[str]]:
    """Run ``hyperframes inspect``; return (clean, error findings).

    Unlike lint, these are *layout* failures — overlapping text, content spilling
    out of a card. They do not stop the render, but they are what makes an
    otherwise correct scene look broken, so the director gets a chance to fix
    them before capture starts.
    """
    result = run_capture_logged(
        name="HyperFrames inspect",
        command=_hyperframes_bin() + ["inspect", str(task_dir)],
        logger=logger,
        log=None,
        timeout=INSPECT_TIMEOUT,
    )
    output = f"{result.stdout}\n{result.stderr}".strip()
    findings = [line.strip() for line in output.splitlines() if "✗" in line]
    if log:
        if findings:
            log(f"Inspect: {len(findings)} layout issue(s)")
            for line in findings[:8]:
                log(f"  {line}")
        else:
            log("Inspect: layout clean")
    return (not findings), findings
