## Context

The news worker includes evidence research, public-footage procurement and editorial reviews. The requested feature instead starts from user-provided research and uses mostly generated stills. Existing OpenCLI media generators already own browser sessions, request pacing and task-local downloads; configured writing and speech providers are reusable.

## Goals / Non-Goals

**Goals:** Persistent independent workspace, editable long-form scripts and narration-bound scenes, adjustable style presets, exact audio-led timing, generated stills and optional Gemini motion, retryable media jobs, observable cancellation/errors, playable downloads.

**Non-Goals:** Topic discovery, source web research, news review gates, footage procurement, publishing, channel analytics, guaranteeing character identity from independent generations.

## Decisions

1. Keep projects in atomically written task-local JSON under `outputs/explainers`, separate from news task rows. Revision checks and single-process operation locks prevent editor/job races. Restart marks interrupted jobs explicitly and preserves artifacts. This avoids overloading news worker states.
2. Generate a chapter outline, then write chapters into short scenes with exact narration and a concrete visual prompt. A shared style bible is repeated for every illustration. User research is the sole substantive evidence input; instructions in supplied research are data. Shape/language/length checks are generation validation, not editorial approval.
3. Reuse the configured model dispatcher, TTS and OpenCLI media primitives. Do not call news composition or review modules. Render 1080p with FFmpeg because this workflow needs stills, small camera moves, optional short motion and captions rather than a bespoke HTML animation composition.
4. Measure each scene's synthesized speech, round coverage up to frames, and pad audio only to that boundary. Record a gapless timeline and per-scene provenance. Gemini plays once and holds its final frame; no looping or speech retiming.
5. Cache media with content/style/voice hashes. Editing a scene invalidates dependent media; retries retain completed media. Save channel defaults separately from per-project snapshots.

## Risks / Trade-offs

- Browser login, quotas, request pacing and eventual provider completion can delay jobs → retain checkpoints and conversation provenance, expose actual failures and retry controls.
- Independent text-to-image calls can vary character appearance → persistent detailed character bible; no claim of identity lock.
- Per-scene TTS increases call count and can sound segmented → use coherent sentence-sized scenes and native speech pace; test actual clip boundaries.
- This app owns one ASGI process → same restriction as the existing worker; do not launch multiple workers against one output directory.

## Migration Plan

No database migration. Register router and lifecycle cleanup, build the UI, run isolated integration checks, then activate only when existing production work can be preserved. Reverting code leaves generated artifacts untouched.
