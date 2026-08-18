# OpenCLI integration

- Upstream: `jackwener/opencli`
- Project-local package: `@jackwener/opencli@1.8.6`
- Runtime install: `npm install --prefix tools/opencli`
- Wrapper: `scripts/opencli.sh`
- Upstream skills copied project-locally to `.claude/skills/opencli-*`
- Reproducible postinstall patch: `patch-opencli.mjs` adds Gemini local-image
  upload support for rendered-frame A/V review plus a `gemini video` adapter
  for the signed-in Create Video UI. The adapter accepts ordered empty/final
  frames, selects 16:9 or 9:16, waits for generation, and captures the browser
  download. It detects Gemini's `generated-video` completion component (rather
  than assuming a native `<video>` element) and can resume an existing Gemini
  conversation when a late result only needs downloading. The patch is pinned
  to 1.8.6 and fails installation if its upstream anchors change.

No global OpenCLI npm package or global Claude Code skill is required.
