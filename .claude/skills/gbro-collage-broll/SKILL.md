---
name: gbro-collage-broll
description: "Automatically turn selected narration beats into visually diverse editorial collage B-roll. Use whenever a FrontierTechSN task enables collage B-roll, chooses the paper-collage opening, or asks for collage/纸拼贴/混合媒介拼贴 B-roll. This project variant is non-interactive: it replaces the original approval gates, Codex image_gen, and GEMINI_API_KEY SDK path with the project Agent SDK plus OpenCLI-driven signed-in ChatGPT Web and Gemini Web."
metadata:
  compatibility: "FrontierTechSN project; Claude Agent SDK; project-local OpenCLI and Browser Bridge; signed-in ChatGPT Web and Gemini Web; ffmpeg and ffprobe. No GEMINI_API_KEY or google-genai package."
---

# gbro Collage B-roll — FrontierTechSN edition

Turn a narration beat into one sharp visual metaphor, a finished original
editorial collage still, and a model-directed B-roll clip whose duration matches
the narration scene up to Gemini's configured generation cap.

This is the project-adapted version of
`https://github.com/pyang5166/gbro-collage-broll` at commit
`a1a4ee2e2abf7d44e460026b706d0c72c2cf8a91`.

## Execution contract

Run the whole workflow without pausing for metaphor, still, or video approval.
The task form is the user's authorization. Persist every intermediate artifact
and QA result so a person can inspect or rerun a failed item later.

Use these boundaries:

1. The project's Claude Agent SDK selects beats and writes visual specs.
2. `scripts/opencli.sh chatgpt image` generates the finished still through the
   signed-in ChatGPT Web session.
3. `scripts/opencli.sh gemini video` opens Gemini Web Create Video, selects the
   task aspect ratio, uploads the ordered empty and completed frames, waits,
   downloads the result, and never reads `GEMINI_API_KEY`.
4. FFmpeg trims every clip without looping to the shorter of its narration
   scene and Gemini's configured single-generation limit, at 24fps, H.264, and
   with no audio.
5. HyperFrames mounts successful clips as full-bleed scene plates
   that play once and then hold their completed final frame.

If one generated item fails, record the failure and continue with the remaining
items. Never promote a procedural imitation, missing path, or unverified media
path as a successful Paper-Collage clip; an incomplete requested set must keep
the final composition blocked.

## Agent visual-spec contract

Choose visually rich narration beats across the whole timeline. When the task
opening style is `paper_collage`, include `scene-01`. Avoid adjacent beats unless
the script is too short to distribute them.

Return a JSON array only. Each item must contain:

```json
{
  "scene_id": "scene-01",
  "script_meaning": "one concrete audience takeaway",
  "emotion": "the precise emotional register of this beat",
  "visual_metaphor": "one sentence showing a physical relationship",
  "background_hex": "#D96B35",
  "accent_colors": ["optional planner swatches; may be empty"],
  "art_direction": "a distinctive, story-specific visual world",
  "color_direction": "how color should serve this beat",
  "composition_direction": "a distinctive spatial strategy",
  "motion_direction": "the story relationship, transformation, or feeling that must become visible; describe meaning rather than an animation technique or phase structure",
  "elements": [
    {"what": "film clock", "role": "structure", "motion": "the visible behavior this story ingredient should communicate", "placement": "its relationship in the completed frame"}
  ],
  "final_frame": "a concise description of the completed composition"
}
```

Use as many or as few visual ingredients as the idea needs, while retaining the
semantic behavior of every important dynamic ingredient. Do not simplify,
panelize, or pre-segment the visual concept for an assumed animation method. A
single beat expresses one metaphor; do not illustrate the transcript word by
word.

## Visual language

Paper collage is the medium, not a preset aesthetic. Invent a fresh art
direction for every selected beat. The planner may use torn or precisely cut
paper, archival photography, contemporary color, photocopy, fabric, tape,
paint, ink, vellum, diagrams, found ephemera, geometric abstraction,
surreal photomontage, dimensional layers, or another collage-compatible
language that serves the narration. These are examples, not a menu or a limit.

Do not repeat a default combination of centered subject, flat color field,
black-and-white halftone, cream keyline, cyan accent, and generous negative
space. Across a batch, deliberately vary at least the composition strategy,
material/mark language, palette behavior, density, edge treatment, and motion
character. Repetition is allowed only when it creates a clear narrative motif.

`background_hex` is a technical color anchor for the first keyframe, not
a command to make the finished composition a flat monochrome field. Integrate
it naturally into the AI's chosen palette. `elements` are narrative ingredients
with semantic behaviors and relationships; they are not an animation recipe or
a rigid object checklist.

Content exclusions only: no readable typography, letters, numerals,
subtitles, logos, watermarks, or UI. These exclusions do not constrain the
overall collage aesthetic.

## Motion language

Use Image 1 and Image 2 only as first- and final-frame boundary conditions. The
video-generation model owns every intermediate visual and temporal decision.
Do not prescribe a transition mechanism, shot structure, camera behavior,
layer count, phase count, rhythm, or named animation technique. In particular,
do not convert the completed still into screen pieces merely because two
endpoint images were supplied.

Pass the planner's complete `script_meaning`, `visual_metaphor`, `emotion`,
`motion_direction`, `final_frame`, and every element's `what`, `role`, `motion`,
and `placement` to the video model. These fields define the meaning and visible
relationships that must survive, not how the model should implement them. Every
non-static element behavior must be visibly legible; the model may reinterpret
the wording and invent any intermediate imagery needed to express it.

The generated sequence must not restart or loop, and it must end by settling
into and holding the supplied final composition. Content exclusions are limited
to unrelated subject matter, readable text, logos, UI, and sound. The final
render must match the task orientation directly:

- landscape task: 16:9 media, normalized to 1280x720;
- portrait task: 9:16 media, normalized to 720x1280.

Never generate portrait media and crop it into landscape, or the reverse.

If Gemini Web video generation is unavailable, record the item as failed and
leave the ordinary scene unmodified. A flattened completed still cannot expose
its semantic objects or material layers, so procedural subdivision is not an
acceptable substitute for model-generated Paper-Collage motion.

## Automated QA

Accept an item only when ffprobe confirms the required width, height, its
scene-derived target duration, 24fps, and no audio stream. Generate a one-second
contact sheet. The clip should begin from the supplied first-frame condition,
keep the planned dynamic relationships visibly legible, avoid restart/text/UI,
and finish near the planned final composition. Record the script duration,
target duration, machine checks, and provenance in
`collage_broll/manifest.json`.

The manifest must state that approval gates are automated, the planner is the
Claude Agent SDK, the still provider is ChatGPT Web through OpenCLI, the video
provider is Gemini Web Create Video through OpenCLI, motion-design authority
belongs to the video-generation model, procedural motion fallback is disabled,
and no API key was used.
