---
name: gbro-collage-broll
description: "Automatically turn selected narration beats into premium editorial halftone paper-collage B-roll. Use whenever a FrontierTechSN task enables collage B-roll, chooses the paper-collage opening, or asks for collage/纸拼贴/半调拼贴 B-roll. This project variant is non-interactive: it replaces the original approval gates, Codex image_gen, and GEMINI_API_KEY SDK path with the project Agent SDK plus OpenCLI-driven signed-in ChatGPT Web and Gemini Web."
metadata:
  compatibility: "FrontierTechSN project; Claude Agent SDK; project-local OpenCLI and Browser Bridge; signed-in ChatGPT Web and Gemini Web; ffmpeg and ffprobe. No GEMINI_API_KEY or google-genai package."
---

# gbro Collage B-roll — FrontierTechSN edition

Turn a narration beat into one sharp visual metaphor, a finished editorial
paper-collage still, and a single-play assemble-from-empty B-roll clip whose
duration matches the narration scene up to Gemini's configured generation cap.

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
5. HyperFrames mounts successful clips as locked-off, full-bleed scene plates
   that play once and then hold their completed final frame.

If one generated item fails, record the failure and continue with the remaining
items. The ordinary deterministic scene remains the fallback; never replace it
with a missing or unverified media path.

## Agent visual-spec contract

Choose visually rich narration beats across the whole timeline. When the task
opening style is `paper_collage`, include `scene-01`. Avoid adjacent beats unless
the script is too short to distribute them.

Return a JSON array only. Each item must contain:

```json
{
  "scene_id": "scene-01",
  "script_meaning": "one concrete audience takeaway",
  "emotion": "calm | surprise | urgency | clarity | irony",
  "visual_metaphor": "one sentence showing a physical relationship",
  "background_hex": "#D96B35",
  "accent_colors": ["cream", "cyan"],
  "elements": [
    {"what": "film clock", "role": "structure", "motion": "slides in", "placement": "center"}
  ],
  "assembly_order": ["film clock", "editor", "scissors", "short output strip"],
  "final_frame": "a concise description of the completed composition"
}
```

Use three to six large separable objects. A single beat expresses one metaphor;
do not illustrate the transcript word by word.

## Visual language

- Flat, forceful paper color field selected for meaning, not a universal blue.
- Black-and-white halftone photographic cut-outs as the structural skeleton.
- Selective colored cardstock accents for hierarchy.
- Crisp cut edges, thin warm-cream keylines, soft physical shadows, and fine
  uncoated-paper grain.
- Generous negative space and a concentrated subject.
- No typography, letters, numerals, subtitles, logos, watermarks, UI, glossy
  3D, photoreal environment, or clutter.

Color tendencies: burnt orange/red for labor and urgency; mustard for tools and
warnings; ink green for cognition; deep purple for rules and memory; teal for
judgment and collaboration. Vary fields across a batch while preserving the
same print and paper treatment.

## Motion language

Use Image 1 as the exact empty paper field and Image 2 as the exact completed
composition. Build in a locked shot: foundation first, subject/cards second,
connectors third, action/result last. Pieces slide, drop, stamp, and snap into
place with tactile stop-motion timing. Hold the supplied final composition.

No cuts, camera motion, zoom, morphing, new objects, text, or sound. The final
render must match the task orientation directly:

- landscape task: 16:9 media, normalized to 1280x720;
- portrait task: 9:16 media, normalized to 720x1280.

Never generate portrait media and crop it into landscape, or the reverse.

## Automated QA

Accept an item only when ffprobe confirms the required width, height, its
scene-derived target duration, 24fps, and no audio stream. Generate a one-second
contact sheet. The clip should begin near the empty color field, visibly
assemble in stages exactly once, avoid cuts/zoom/text/UI, and finish near the
planned final composition. Record the script duration, target duration, machine
checks, and provenance in `collage_broll/manifest.json`.

The manifest must state that approval gates are automated, the planner is the
Claude Agent SDK, the still provider is ChatGPT Web through OpenCLI, the video
provider is Gemini Web Create Video through OpenCLI, and no API key was used.
