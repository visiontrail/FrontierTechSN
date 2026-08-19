# News imagery production breakdown

## Style block

- Preserve the existing FrontierTechSN editorial identity from `backend/pipeline/scene_kit.py`: the selected template owns its background, ink, muted text, accent, typography, and safe area.
- News photography and logos are factual evidence, not decoration. Every image must be tied to one narrated scene, retain source and license metadata, and remain readable after video compression.
- Primary image treatment: editorial slide reveal with a soft perspective settle. Accent treatment: masked full-frame reveal with a restrained Ken Burns move.

## Rhythm declaration

`text-build - INLINE-EVIDENCE - breathe - FULLSCREEN-EVENT - text-resolve`

## Global rules

- Use OpenCLI web search to identify the named brand, person, place, product, or event actually present in the narration.
- Acquire only local, license-ledgered image assets before HyperFrames authoring. Compositions never fetch network media.
- Mix two modes when at least two eligible narrated scenes exist: an inline editorial image beside the text stream and an independent full-screen image scene.
- Parent wrappers own entrances; image children own slow scale or pan motion so GSAP transforms never conflict.
- Keep all motion synchronous, deterministic, seekable, and attached to the paused scene timeline.
- Retain captions and source credits inside the existing video safe area.

## Beat 1: Inline brand or object evidence

- **Concept:** The typography establishes the claim, then a real logo, product, person, or object slides into the adjacent editorial column as proof. The frame should feel like a well-directed magazine spread coming alive.
- **Mood direction:** Precise technology journalism, restrained confidence, strong information hierarchy.
- **Depth layers:** BG uses the current template texture and accent wash; MG contains kicker, headline, supporting copy, and a bordered image viewport; FG contains source credit, corner registration marks, and a short accent rule.
- **Animation choreography:** Rule DRAWS, kicker SLIDES, headline RISES, image viewport ROLLS in from the outer edge with a slight perspective settle, image PUSHES slowly, source credit FADES.
- **Transition out:** The normal scene boundary owns the handoff; no element exits early.

## Beat 2: Full-screen event image

- **Concept:** A concrete news moment takes over the entire frame. The image arrives as a deliberate editorial reveal, not a flat slideshow cut, while the narrated headline remains anchored over a legible scrim.
- **Mood direction:** Cinematic documentary still, factual rather than sensational.
- **Depth layers:** BG is the full-bleed image or contained logo field; MG is a template-tinted scrim and subtle accent wash; FG is kicker, headline, optional body, and source credit.
- **Animation choreography:** Image frame SWEEPS in with a clipped horizontal reveal, frame SETTLES from a shallow perspective angle, image DRIFTS with a slow Ken Burns move, headline RISES, credit FADES.
- **Transition out:** Scene boundary continues the editorial rhythm without an early fade.

## Recurring motifs

- Source-credit pills use the scene accent and current template neutrals.
- Inline image corners and full-screen registration marks repeat across the episode.
- Motion alternates direction by scene index while remaining deterministic.

## Negative prompt

- No unrelated stock photography, generic AI imagery, invented logos, screenshots of search results, watermarked assets, low-resolution thumbnails, remote image URLs, flat unanimated image drops, or uncredited third-party media.
- No image may replace a requested Paper-Collage or already assigned public-footage scene.
- No image may be accepted solely because it downloaded successfully; its expected subject must overlap the exact narrated scene.
