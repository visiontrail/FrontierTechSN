---
name: video-web-footage
description: Discover Bilibili and YouTube B-roll with the project-local OpenCLI/yt-dlp runtime, ask Gemini Web to select script-relevant timestamps, trim with FFmpeg, and maintain a source and rights-review ledger for FrontierTechSN.
---

# Video Web Footage Loop

Use this skill when a FrontierTechSN task needs footage beyond Wikimedia
Commons, when the user supplies a Bilibili/YouTube link, or when a candidate
must be matched to a narration excerpt and trimmed.

## Project isolation

- Run OpenCLI through `./scripts/opencli.sh`, or rely on the project-injected
  `PATH`. Never run `npm install -g`, `npx skills add` without a project target,
  or copy skills into `~/.claude`.
- The pinned runtime is `tools/opencli/package.json`.
- All OpenCLI skills live under this repository's `.claude/skills`.
- Verify Browser Bridge with `./scripts/opencli.sh doctor` before browser-backed
  operations.

## End-to-end loop

1. Read the task's `script.txt` and derive concrete visual queries from each
   paragraph/scene. Prefer nouns, places, actions, eras, and camera language.
2. Discover candidates:

   ```bash
   ./scripts/opencli.sh bilibili search "<query>" --limit 5 -f json
   yt-dlp --flat-playlist --dump-json "ytsearch5:<query>"
   ```

3. For a YouTube candidate, send the public link plus the matching narration
   excerpt to Gemini Web through OpenCLI. Require JSON with `start_seconds`,
   `end_seconds`, `confidence`, and `reason`. Set an explicit timeout and keep a
   deterministic trim fallback. Gemini Web cannot reliably open Bilibili links;
   use the fallback there and retain sampled evidence frames for review.
4. Download through the platform adapter:

   ```bash
   ./scripts/opencli.sh bilibili download <BV-id> --quality 480p --output <raw-dir> -f json
   yt-dlp --no-playlist --merge-output-format mp4 -o '<raw-dir>/%(id)s.%(ext)s' <youtube-url>
   ```

5. Probe and trim with `ffprobe`/`ffmpeg`. Remove source audio, normalize the
   plate to the target aspect ratio, encode H.264/yuv420p, and use `+faststart`.
6. Add the result to `footage/manifest.json`, including source URL, creator,
   provider, source duration, selected timestamps, analyzer result, matching
   script excerpt, local SHA-256, and evidence-frame paths.
7. Let `visual_plan.attach_footage` match the clip to the storyboard scene and
   let HyperFrames lint/inspect/render the final composition.

## Rights and publishing gate

Downloadability is not a license. Wikimedia files may pass the explicit
open-license allowlist. Bilibili/YouTube files must always carry:

- `rights_status: review_required`
- `review_required: true`
- `license: Rights not verified — human review required`
- the canonical source page and creator/channel

Do not claim that a title containing “free stock” proves commercial reuse
permission. Surface the publication blocker in the UI and ask the operator to
review the source terms before publishing outside a private draft.

## Failure policy

- A Bilibili or Gemini failure must not destroy a valid narration/video job.
- Continue with the other source, then fall back to Wikimedia or deterministic
  scenes.
- Never ship a missing media reference: verify the local file, probe it, and
  hash it before adding it to the manifest.
- Delete only the raw file created for the current candidate after the trimmed
  derivative and evidence frames have been verified.
