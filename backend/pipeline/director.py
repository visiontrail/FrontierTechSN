"""Claude Agent SDK video-production agents that author HyperFrames scenes.

The composer hands each agent a slice of the storyboard — the narration, the
art director's brief, and the seconds available — and the agent writes the
actual HyperFrames sub-composition for those scenes. Several agents run
concurrently over disjoint slices, so a 40-scene episode is directed by a small
crew rather than one long serial session.

Two invariants make it safe to put a model in the render path:

1. **The agent never touches the spine.** ``index.html`` mounts scenes at times
   derived from the audio; a model cannot desynchronise picture from sound.
2. **Every authored file is gated before it ships.** :func:`validate_scene_html`
   enforces the HyperFrames runtime contract that, when violated, is precisely
   what rendered blank frames. A scene that fails the gate is reverted to its
   deterministic scene-kit version, so a bad agent costs one plain scene — never
   a black video.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from backend import config, skills_admin
from backend.pipeline import intros, outros, scene_kit
from backend.pipeline.video_format import FrameSpec, LANDSCAPE

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

# Scenes handed to one agent. Small enough that the agent can hold every scene's
# narration in working memory and give each one real attention; large enough
# that it can shape continuity across a run of scenes.
SCENES_PER_AGENT = 6

# Concurrent agents. Each spawns a `claude` CLI process, so this is a real
# resource knob — the box also has to run headless Chrome for the render.
MAX_CONCURRENT_AGENTS = 3

# Turns per agent: read the brief, write N scenes, self-check. Generous, since
# the agent writes one file per tool call.
MAX_TURNS = 40

AGENT_TIMEOUT = 900

# Structural gate. Each pattern must appear in an authored scene file.
_FORBIDDEN = (
    (re.compile(r"Math\.random\s*\("), "Math.random() breaks deterministic capture"),
    (re.compile(r"Date\.now\s*\("), "Date.now() breaks deterministic capture"),
    (re.compile(r"performance\.now\s*\("), "performance.now() breaks deterministic capture"),
    (re.compile(r"repeat\s*:\s*-1"), "repeat: -1 wedges the capture engine"),
    (re.compile(r"""<(?:script|link)[^>]+(?:src|href)\s*=\s*["']https?://""", re.I),
     "external script/stylesheet will not load during capture"),
    (re.compile(r"@import\s+url\(\s*['\"]?https?://", re.I),
     "remote @import will not load during capture"),
    (re.compile(r"\.(?:play|pause|seek)\s*\(\s*\)"), "the framework owns media playback"),
)


@dataclass
class DirectorOutcome:
    authored: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    agents_run: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.authored)


SYSTEM_PROMPT = """\
You are a video-production agent. You author HyperFrames scene compositions —
the HTML files that become the picture track of a narrated documentary video.

The narration audio already exists and is fixed. You are given the exact number
of seconds each scene occupies. You write what the viewer SEES.

# Output contract — one file per scene

Write each scene to `compositions/<scene-id>.html`, using the Write tool. The
file must have exactly this shape:

```html
<template id="scene-07-template">
  <div id="scene-07" data-composition-id="scene-07" data-width="1920" data-height="1080">
    <style>
      #scene-07 { position:relative; width:1920px; height:1080px; overflow:hidden; background:#0B0D17; }
      /* every selector prefixed with #scene-07 */
    </style>

    <!-- markup -->

    <script src="../vendor/gsap.min.js"></script>
    <script>
      window.__timelines = window.__timelines || {};
      (function () {
        const tl = gsap.timeline({ paused: true });
        const q = gsap.utils.selector("#scene-07");
        tl.fromTo(q("#scene-07-head"), { y: 60, opacity: 0 },
                  { y: 0, opacity: 1, duration: .8, ease: "expo.out" }, 0.3);
        window.__timelines["scene-07"] = tl;
      })();
    </script>
  </div>
</template>
```

# Non-negotiable rules

These are not style preferences. Violating any one of them produces a blank
video, which is the exact bug this system was built to fix.

1. `<template>` wrapper, with the composition div directly inside it.
2. The composition div carries `id`, `data-composition-id` (both the scene id),
   `data-width="1920"`, `data-height="1080"`. It must NOT carry `data-start`,
   `data-duration`, or `data-track-index` — the spine owns scene timing.
3. Register the timeline: `window.__timelines["<scene-id>"] = tl`, built
   synchronously, `{ paused: true }`. Never inside async/setTimeout/Promise.
4. Load GSAP with exactly `<script src="../vendor/gsap.min.js"></script>`.
   Never a CDN URL. Never a remote font `@import` — write `font-family: Inter,
   sans-serif` and the compiler embeds it.
5. Scope EVERY CSS selector under `#<scene-id>` so scenes cannot leak styles
   into each other.
6. Deterministic only: no `Math.random()`, no `Date.now()`, no `fetch()`, no
   network requests. Hard-code any "random" geometry you want.
7. Never use `repeat: -1`. Compute a finite repeat count from the duration.
8. Use `gsap.fromTo()` — not `gsap.from()` — because scene elements are mounted
   dynamically. Animate only visual properties (opacity, x, y, scale, rotation,
   colour, transforms).
9. Every timeline position must land inside the scene's duration.

# Craft

- Build the end state in CSS first, then animate INTO it with `fromTo`.
- Entrances only. Do not animate elements out; the cut to the next scene is the
  exit. Motion should settle well before the scene ends.
- Stagger entrances by 0.1–0.3s. Vary easing — use at least three different
  eases per scene. Never start a tween at exactly t=0.
- Type for video: headlines 70–130px, body 32–46px, labels 20px+. Long headlines
  need a smaller size — text must never overflow the frame.
- Layout: a full-size container with `padding: 130px 150px 210px` and flexbox.
  Never `position:absolute; top:Npx` for content. The bottom padding is reserved
  — a caption bar sits there and must not be covered.
- **Safe area.** The frame is exactly 1920x1080 and nothing scrolls. Keep every
  element at least 60px from each edge, and keep the bottom 200px empty. A
  decorative element pinned near `top: 0` or `bottom: 0` will be cut off.
- Two text blocks must never occupy the same space. Lay the scene out as rows in
  one flex column and let the gap do the spacing; if you position a decorative
  element absolutely, keep it clear of the text column.
- On-screen copy is not the transcript. Compress. A viewer reads roughly three
  words per second.
- `scene-01` is the opening: enter through a concrete image, claim, question,
  quote, or tension drawn from its narration, and make it foreshadow the later
  argument. Never turn it into a title slate. Never show a video/episode number,
  filename, task identifier, production label, or other metadata.
- Palette: background #0B0D17, ink #F5F2EA, muted #98A1BA. Use the accent colour
  given in each scene's brief for emphasis, rules, and artwork.
- Draw with SVG, CSS gradients, and shapes. Abstract artwork that echoes the
  subject beats a decorative blob. You have no image assets unless a scene's
  brief names one.

# Dedicated ByteFront Espresso outro

When a brief says `dedicated outro scene`, the Gemini video is the moving
background and you own only the editable HyperFrames overlay. Preserve the
exact background and logo `src` values. Preserve the inline bookend stacking
contract: background `z-index:0`, veil `z-index:1`, overlay `z-index:2`, and
fade `z-index:3`. Preserve these semantic hooks while
you may freely improve their text, size, and position:
`data-outro-role="brand"`, `data-outro-role="thanks"`,
`data-outro-role="actions"`, and one each of `data-outro-action="like"`,
`"comment"`, and `"share"`. Preserve `data-outro-brand-part="espresso"` and
keep `Espresso` at the same apparent type size and weight as `Bytefront`; do not
render it as a small badge. Keep an English closing message, the ByteFront
Espresso identity, and recognizable Like / Comment / Share icons. Do not place
the removed Chinese closing phrase anywhere in the file. The scene is timed to
the spoken closing narration; do not add audio or captions inside the scene.
The root program owns both and may reserve its normal caption safe area.
Do not replace the Gemini motion with a CSS pan, zoom, gradient, or generated
animation. The `<video>` must remain `muted playsinline`; the root program owns
all audio.

# Dedicated ByteFront Espresso intro

When a brief says `dedicated intro scene`, the Gemini video is the moving
background and you own only the editable, edition-aware HyperFrames overlay.
Preserve the exact background and logo `src` values and the inline bookend
stacking contract: background `z-index:0`, veil `z-index:1`, overlay `z-index:2`,
and fade `z-index:3`. Preserve these semantic
hooks: `data-intro-role="brand"`, `data-intro-role="edition"`, and one each of
`data-intro-field="date"` and `"weekday"`.
The visible field values must come from one synchronous
`window.__hyperframes.getVariables()` call using the declared
`edition_date` and `edition_weekday` variables. Keep `Espresso` at the same
apparent type size and weight as `Bytefront`; do not render it as a small badge.
Never add a briefing-label or story-count chip. Never compute the date in
JavaScript. The scene is timed to the spoken opening narration; do not add audio
or captions inside the scene. The root program owns both and may reserve its
normal caption safe area.
Do not replace the Gemini motion with a CSS pan, zoom, gradient, or generated
animation. The `<video>` must remain `muted playsinline`; the root program owns
all audio.

# Working method

1. Read `compositions/<scene-id>.html` — a deterministic draft already exists.
   It is a correct, working baseline. Your job is to make it better: sharper
   layout, artwork that actually reflects what is being said, motion with intent.
2. Write your version to the same path.
3. Do not create, delete, or edit any file other than the scene files you were
   assigned. Never touch `index.html`.

Report at the end with one line per scene: the scene id and the visual idea.
"""


def _system_prompt(
    theme: scene_kit.Theme,
    frame: FrameSpec = LANDSCAPE,
) -> str:
    """Give authoring crews the selected template instead of a dark default."""
    prompt = SYSTEM_PROMPT.replace("background:#0B0D17;", f"background:{theme.bg};")
    prompt = prompt.replace(
        "- Palette: background #0B0D17, ink #F5F2EA, muted #98A1BA. Use the accent colour\n"
        "  given in each scene's brief for emphasis, rules, and artwork.",
        f"- Palette: background {theme.bg}, ink {theme.ink}, muted {theme.muted}. Use the accent colour\n"
        "  given in each scene's brief for emphasis, rules, and artwork.",
    )
    if theme.name == "shanshui":
        prompt = prompt.replace(
            "- Draw with SVG, CSS gradients, and shapes.",
            "- Shan Shui template: evoke warm rice paper, layered organic terrain in sand, sage, and ink green, "
            "fine topographic veins, plus thin ochre route arcs with solid circular nodes. Motion is contemplative: "
            "slow terrain drift, route-line drawing, and soft reveals. Avoid neon, glossy UI, rounded cards, and "
            "generic geometric blobs.\n- Draw with SVG, CSS gradients, and shapes.",
        )
    if frame != LANDSCAPE:
        prompt = prompt.replace("1920x1080", f"{frame.width}x{frame.height}")
        prompt = prompt.replace('data-width="1920"', f'data-width="{frame.width}"')
        prompt = prompt.replace('data-height="1080"', f'data-height="{frame.height}"')
        prompt = prompt.replace("width:1920px", f"width:{frame.width}px")
        prompt = prompt.replace("height:1080px", f"height:{frame.height}px")
        prompt += (
            f"\n- Portrait layout: the frame is {frame.width}x{frame.height}. Use a vertical flex stack, "
            "keep text within 92px side margins, and reserve the bottom 300px for captions/chrome.\n"
        )
    return prompt


def _scene_brief(scene: dict, plan: dict, theme: scene_kit.Theme) -> str:
    accent = scene_kit.accent_hex(plan.get("accent"), theme)
    lines = [
        f"## {scene['id']}  —  {scene['duration']:.1f} seconds",
        f"accent: {plan.get('accent', 'amber')} ({accent})",
        f"suggested archetype: {plan.get('archetype', 'topic')}",
    ]
    if plan.get("kicker"):
        lines.append(f"section label: {plan['kicker']}")
    if plan.get("headline"):
        lines.append(f"draft headline: {plan['headline']}")
    if plan.get("body"):
        lines.append(f"draft support: {plan['body']}")
    if plan.get("items"):
        lines.append("draft items: " + " | ".join(plan["items"]))
    if plan.get("footage_src"):
        if plan.get("archetype") in {"intro", "outro"}:
            lines.append(
                f"Gemini background asset: {plan['footage_src']} "
                "(preserve this exact src value in the existing video element)"
            )
        else:
            lines.append(
                f"footage asset: {plan['footage_src']} (relative to the project root; "
                f"reference it as ../{plan['footage_src']} from compositions/)"
            )
    if plan.get("archetype") == "intro":
        lines.extend(
            [
                "dedicated intro scene timed to this spoken opening narration",
                f"narration spoken over this scene:\n\"{scene['text']}\"",
                f"exact logo asset: {plan.get('intro_logo_src', '')} "
                "(reference it unchanged from compositions/)",
                "dynamic HyperFrames variables: edition_date, edition_weekday",
                "overlay method: edit the existing ByteFront brand and edition metadata "
                "layers while preserving every data-intro-role/data-intro-field hook; "
                "keep Espresso the same apparent size as Bytefront and do not add "
                "briefing-label or story-count chips",
            ]
        )
    elif plan.get("archetype") == "outro":
        lines.extend(
            [
                "dedicated outro scene timed to this spoken closing narration",
                f"narration spoken over this scene:\n\"{scene['text']}\"",
                f"exact logo asset: {plan.get('outro_logo_src', '')} "
                "(reference it unchanged from compositions/)",
                "overlay method: edit the existing HyperFrames brand, English closing "
                "message, and Like / Comment / Share layers; copy and positions remain editable",
            ]
        )
    else:
        lines.append(f"narration spoken over this scene:\n\"{scene['text']}\"")
    return "\n".join(lines)


def _batch_prompt(
    storyboard: dict,
    batch: Sequence[tuple[dict, dict]],
    *,
    batch_no: int,
    batch_total: int,
    theme: scene_kit.Theme,
) -> str:
    briefs = "\n\n".join(_scene_brief(scene, plan, theme) for scene, plan in batch)
    ids = ", ".join(scene["id"] for scene, _ in batch)
    return f"""Overall thesis: {storyboard.get('thesis', '')}

You are crew {batch_no} of {batch_total}. Author these scenes and only these: {ids}

{briefs}

Write each scene to compositions/<scene-id>.html now."""


def validate_scene_html(
    text: str,
    scene_id: str,
    frame: FrameSpec = LANDSCAPE,
    plan: dict | None = None,
) -> list[str]:
    """Structural problems that would break the render. Empty list means ship it."""
    problems: list[str] = []
    if len(text) < 400:
        problems.append("file is too short to be a real scene")
    if "<template" not in text:
        problems.append("missing <template> wrapper")
    if f'data-composition-id="{scene_id}"' not in text:
        problems.append(f'missing data-composition-id="{scene_id}"')
    if f'data-width="{frame.width}"' not in text or f'data-height="{frame.height}"' not in text:
        problems.append(
            f"composition dimensions must be {frame.width}x{frame.height}"
        )
    if not re.search(rf"""window\.__timelines\[\s*["']{re.escape(scene_id)}["']\s*\]\s*=""", text):
        problems.append(f'missing window.__timelines["{scene_id}"] registration')
    if "paused" not in text:
        problems.append("timeline is not created with { paused: true }")
    if "../vendor/gsap.min.js" not in text:
        problems.append("does not load ../vendor/gsap.min.js")
    # The spine owns scene timing; a scene that declares its own would double-schedule.
    if re.search(r"data-composition-id=\"" + re.escape(scene_id) + r"\"[^>]*data-(?:start|track-index)=", text):
        problems.append("composition root must not declare data-start/data-track-index")
    for pattern, why in _FORBIDDEN:
        if pattern.search(text):
            problems.append(why)
    if plan and plan.get("archetype") == "outro":
        problems.extend(outros.outro_overlay_problems(text))
        for key in ("footage_src", "outro_logo_src"):
            source = str(plan.get(key) or "")
            if source and f'src="{source}"' not in text:
                problems.append(f"outro dropped locked {key}")
    if plan and plan.get("archetype") == "intro":
        problems.extend(intros.intro_overlay_problems(text))
        for key in ("footage_src", "intro_logo_src"):
            source = str(plan.get(key) or "")
            if source and f'src="{source}"' not in text:
                problems.append(f"intro dropped locked {key}")
    return problems


def _agent_options(
    task_dir: Path,
    model: str | None,
    env: dict[str, str],
    system_prompt: str = SYSTEM_PROMPT,
):
    from claude_agent_sdk import ClaudeAgentOptions

    # File tools only. The HyperFrames CLI is run by the pipeline, not the agent,
    # so there is no shell in the loop and the lint result stays authoritative.
    tools = ["Read", "Write", "Edit", "Glob", "Grep"]
    allowed = list(tools)

    enabled_skills, _ = skills_admin.runtime_skill_names()
    framework_skills = [
        name for name in enabled_skills if name in {"hyperframes", "gsap", "css-animations"}
    ]
    if framework_skills:
        tools.append("Skill")
        allowed.extend(f"Skill({name})" for name in framework_skills)

    options: dict = {
        "system_prompt": system_prompt,
        "model": model,
        "cwd": str(task_dir),
        "tools": tools,
        "allowed_tools": allowed,
        "disallowed_tools": ["Bash", "WebFetch", "WebSearch", "Task", "NotebookEdit"],
        # Edits are pre-approved: the agent runs headless inside the task's own
        # output directory, so there is no operator to answer a prompt and
        # nothing outside that directory it can reach.
        "permission_mode": "acceptEdits",
        "setting_sources": ["project"],
        "max_turns": MAX_TURNS,
        "env": env,
    }
    if framework_skills and "skills" in getattr(ClaudeAgentOptions, "__dataclass_fields__", {}):
        options["skills"] = framework_skills
    if config.CLAUDE_CLI_PATH:
        options["cli_path"] = config.CLAUDE_CLI_PATH
    return ClaudeAgentOptions(**options)


async def _run_agent_attempt(
    task_dir: Path,
    prompt: str,
    *,
    batch_no: int,
    model: str | None,
    env: dict[str, str],
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[int, str | None]:
    """Run one bounded SDK session and return its assistant-turn count/error."""
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, query

    options = _agent_options(task_dir, model, env, system_prompt)
    stderr_lines: list[str] = []
    options.stderr = stderr_lines.append
    turns = 0
    try:
        async with asyncio.timeout(AGENT_TIMEOUT):
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    turns += 1
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text.strip():
                            logger.debug("director[%s] %s", batch_no, block.text[:400])
                elif isinstance(message, ResultMessage) and message.is_error:
                    detail = "; ".join(str(e) for e in (message.errors or [message.result or "?"]))
                    return turns, f"crew {batch_no} reported an error: {detail}"
    except TimeoutError:
        return turns, f"crew {batch_no} timed out after {AGENT_TIMEOUT}s"
    except Exception as exc:  # noqa: BLE001 - a crew failure must not sink the render
        tail = f" | stderr: {' '.join(stderr_lines)[-400:]}" if stderr_lines else ""
        return turns, f"crew {batch_no} failed: {exc.__class__.__name__}: {exc}{tail}"

    return turns, None


async def _run_agent(
    task_dir: Path,
    prompt: str,
    ids: Sequence[str],
    *,
    batch_no: int,
    batch_total: int,
    model: str | None,
    env: dict[str, str],
    log: LogCallback | None,
    label: str = "authoring",
    system_prompt: str = SYSTEM_PROMPT,
    max_attempts: int | None = None,
) -> tuple[list[str], str | None]:
    """Run one crew with the same configured-provider retry policy as text stages."""
    ids = list(ids)
    maximum_attempts = max(
        1,
        int(max_attempts) if max_attempts is not None else int(config.AI_MAX_RETRIES) + 1,
    )
    if log:
        log(f"Director crew {batch_no}/{batch_total}: {label} {', '.join(ids)}")

    last_error: str | None = None
    total_turns = 0
    for attempt in range(1, maximum_attempts + 1):
        if log and maximum_attempts > 1:
            log(
                f"Director crew {batch_no}/{batch_total}: provider attempt "
                f"{attempt}/{maximum_attempts}"
            )
        turns, error = await _run_agent_attempt(
            task_dir,
            prompt,
            batch_no=batch_no,
            model=model,
            env=env,
            system_prompt=system_prompt,
        )
        total_turns += turns
        if error is None:
            if log:
                log(
                    f"Director crew {batch_no}/{batch_total}: finished after "
                    f"{total_turns} turn(s) on attempt {attempt}/{maximum_attempts}"
                )
            return ids, None

        last_error = error
        if log:
            log(
                f"Director crew {batch_no}/{batch_total}: attempt "
                f"{attempt}/{maximum_attempts} failed ({error})"
            )
        if attempt < maximum_attempts:
            delay = min(
                config.AI_RETRY_MAX_SECONDS,
                config.AI_RETRY_BASE_SECONDS * 2 ** (attempt - 1),
            )
            if log:
                log(
                    f"Director crew {batch_no}/{batch_total}: retrying same provider in "
                    f"{delay:.0f}s ({attempt + 1}/{maximum_attempts})"
                )
            await asyncio.sleep(delay)

    return ids, last_error or f"crew {batch_no} exhausted {maximum_attempts} attempts"


async def direct_scenes(
    task_dir: Path,
    storyboard: dict,
    plans: Sequence[dict],
    kit_plans: Sequence[scene_kit.ScenePlan],
    *,
    model: str | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    log: LogCallback | None = None,
    frame: FrameSpec = LANDSCAPE,
) -> DirectorOutcome:
    """Run the director crews and gate everything they wrote.

    ``kit_plans`` are the deterministic drafts already written to disk; they are
    both the agent's starting point and the revert target for anything that
    fails the gate.
    """
    from backend.pipeline.agent import build_agent_env

    outcome = DirectorOutcome()
    scenes = {scene["id"]: scene for scene in storyboard.get("scenes", [])}
    kit_by_id = {plan.id: plan for plan in kit_plans}
    plans_by_id = {str(plan.get("id")): plan for plan in plans}
    theme = next(iter(kit_by_id.values())).theme if kit_by_id else scene_kit.DEFAULT_THEME

    pairs = [(scenes[plan["id"]], plan) for plan in plans if plan["id"] in scenes]
    if not pairs:
        return outcome

    # Snapshot the drafts so a scene the crew never touched is not mistaken for
    # one it authored — the drafts are valid HTML and would sail through the gate.
    comps_dir = task_dir / "compositions"
    drafts: dict[str, str] = {}
    for plan in plans:
        draft = comps_dir / f"{plan['id']}.html"
        if draft.exists():
            drafts[plan["id"]] = draft.read_text(encoding="utf-8")

    batches = [pairs[i : i + SCENES_PER_AGENT] for i in range(0, len(pairs), SCENES_PER_AGENT)]
    env = build_agent_env(model, endpoint, api_key)
    resolved_model = (config.ANTHROPIC_MODEL or model or "").strip() or None
    outcome.agents_run = len(batches)

    if log:
        log(
            f"Director: {len(pairs)} scenes across {len(batches)} agent crew(s), "
            f"{MAX_CONCURRENT_AGENTS} at a time (model={resolved_model or 'default'})"
        )

    gate = asyncio.Semaphore(MAX_CONCURRENT_AGENTS)

    async def run(batch_no: int, batch: Sequence[tuple[dict, dict]]):
        async with gate:
            return await _run_agent(
                task_dir,
                _batch_prompt(
                    storyboard,
                    batch,
                    batch_no=batch_no,
                    batch_total=len(batches),
                    theme=theme,
                ),
                [scene["id"] for scene, _ in batch],
                batch_no=batch_no,
                batch_total=len(batches),
                model=resolved_model,
                env=env,
                log=log,
                system_prompt=_system_prompt(theme, frame),
            )

    results = await asyncio.gather(
        *(run(i, batch) for i, batch in enumerate(batches, start=1)),
        return_exceptions=True,
    )

    claimed: list[str] = []
    for result in results:
        if isinstance(result, BaseException):
            outcome.failures.append(f"crew crashed: {result}")
            continue
        ids, error = result
        claimed.extend(ids)
        if error:
            outcome.failures.append(error)
            if log:
                log(f"Director: {error}")

    # Gate every claimed scene, reverting anything that would break the render.
    comps = task_dir / "compositions"
    for scene_id in claimed:
        path = comps / f"{scene_id}.html"
        if not path.exists():
            outcome.rejected.append(scene_id)
            continue
        text = path.read_text(encoding="utf-8")
        if text == drafts.get(scene_id):
            # The deterministic draft is still on disk untouched: the crew never
            # got as far as writing this scene. Counting it as authored would
            # report a failed run as a success.
            outcome.rejected.append(scene_id)
            continue
        problems = validate_scene_html(
            text,
            scene_id,
            frame,
            plan=plans_by_id.get(scene_id),
        )
        if problems:
            outcome.rejected.append(scene_id)
            if log:
                log(f"Director: rejected {scene_id} ({problems[0]}) — using the deterministic scene")
            kit = kit_by_id.get(scene_id)
            if kit is not None:
                path.write_text(scene_kit.render_scene(kit), encoding="utf-8")
            continue
        outcome.authored.append(scene_id)

    if log:
        log(
            f"Director: {len(outcome.authored)} scene(s) authored by agents, "
            f"{len(outcome.rejected)} left as the deterministic draft"
        )
    return outcome


async def repair_scenes(
    task_dir: Path,
    scene_ids: Sequence[str],
    findings: Sequence[str],
    *,
    model: str | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    log: LogCallback | None = None,
    frame: FrameSpec = LANDSCAPE,
    theme: scene_kit.Theme = scene_kit.DEFAULT_THEME,
) -> list[str]:
    """Hand HyperFrames' own layout findings back to an agent to fix.

    ``hyperframes inspect`` reports overlapping text and content spilling out of
    its container with timestamps and selectors — exactly the failures a model
    cannot see from the source alone. Returns the scene ids that still parse
    correctly after the repair pass.
    """
    from backend.pipeline.agent import build_agent_env

    if not scene_ids:
        return []

    env = build_agent_env(model, endpoint, api_key)
    resolved_model = (config.ANTHROPIC_MODEL or model or "").strip() or None

    report = "\n".join(f"- {line}" for line in findings[:24])
    prompt = f"""`hyperframes inspect` ran your scenes in a real browser and found layout failures.

{report}

Fix these scenes: {', '.join(scene_ids)}

Rules for the fix:
- `content_overlap` means two text blocks land on top of each other. Give each
  its own row in the flex column, or reduce sizes so they fit. Do not paper over
  it with `data-layout-allow-overlap`.
- `container_overflow ... overflowed bottom Npx` means the content is taller than
  the frame. Cut copy, reduce font sizes, or reduce gaps until it fits.
- Everything must sit inside the {frame.width}x{frame.height} frame with a 60px margin from every
  edge, and nothing may enter the bottom {300 if frame.is_portrait else 200}px — the caption bar lives there.
- Keep every rule from your original instructions. Do not break the timeline
  registration or the template wrapper.

Read each file, fix it, and write it back."""

    ids, error = await _run_agent(
        task_dir,
        prompt,
        scene_ids,
        batch_no=1,
        batch_total=1,
        model=resolved_model,
        env=env,
        log=log,
        label="repairing",
        system_prompt=_system_prompt(theme, frame),
        # Layout repair is advisory and already has a deterministic fallback.
        # Retrying a rate-limited provider here only delays the render.
        max_attempts=1,
    )
    if error and log:
        log(f"Director: repair pass {error}")

    comps = task_dir / "compositions"
    healthy: list[str] = []
    for scene_id in ids:
        path = comps / f"{scene_id}.html"
        if path.exists() and not validate_scene_html(
            path.read_text(encoding="utf-8"), scene_id, frame
        ):
            healthy.append(scene_id)
    return healthy


def revert_scenes(
    task_dir: Path, scene_ids: Sequence[str], kit_plans: Sequence[scene_kit.ScenePlan]
) -> list[str]:
    """Restore the deterministic version of the named scenes."""
    kit_by_id = {plan.id: plan for plan in kit_plans}
    comps = task_dir / "compositions"
    reverted: list[str] = []
    for scene_id in scene_ids:
        plan = kit_by_id.get(scene_id)
        if plan is None:
            continue
        (comps / f"{scene_id}.html").write_text(scene_kit.render_scene(plan), encoding="utf-8")
        reverted.append(scene_id)
    return reverted


def scenes_named_in(lint_output: str, candidates: Sequence[str]) -> list[str]:
    """Scene ids a lint run complained about, for targeted reverts."""
    error_lines = [line for line in lint_output.splitlines() if "✗" in line]
    blob = "\n".join(error_lines)
    return [scene_id for scene_id in candidates if scene_id in blob]


_FINDING_TIME = re.compile(r"t=([\d.]+)(?:-([\d.]+))?s")


def scenes_named_in_findings(
    findings: Sequence[str],
    candidates: Sequence[str],
    mounts: Sequence[dict] | None = None,
) -> list[str]:
    """Scene ids implicated by ``hyperframes inspect`` findings.

    Some findings name the mount (``#mount-scene-05``); others only describe the
    offending selectors, so the timestamp is the reliable link. Both are used,
    and the result preserves ``candidates`` order.
    """
    blamed: set[str] = set()
    spans = {
        mount["id"]: (float(mount["start"]), float(mount["start"]) + float(mount["duration"]))
        for mount in (mounts or [])
    }
    for line in findings:
        for scene_id in candidates:
            if scene_id in line:
                blamed.add(scene_id)
        match = _FINDING_TIME.search(line)
        if not match:
            continue
        start = float(match.group(1))
        end = float(match.group(2)) if match.group(2) else start
        for scene_id in candidates:
            span = spans.get(scene_id)
            if span and start < span[1] and end > span[0]:
                blamed.add(scene_id)
    return [scene_id for scene_id in candidates if scene_id in blamed]
