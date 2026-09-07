# OpenCLI integration

- Upstream: `jackwener/opencli`
- Project-local package: `@jackwener/opencli@1.8.7` (npm `latest` as of 2026-09-01)
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
  to 1.8.7 and fails installation if its upstream anchors change.

No global OpenCLI npm package or global Claude Code skill is required.

## Browser imports and model-check recovery

Chrome profile imports into Ego Lite can copy OpenCLI's non-secret
`opencli_context_id_v1` routing identifier. Two extensions with that identifier
replace each other's daemon connection, causing stale page identities,
navigation timeouts, and `SESSION_BUSY` after retries. A daemon status showing
one profile does not prove that only one browser is connected. Compare the
profile IDs in both extensions' popups.

Assign the imported Ego Lite extension a distinct routing ID in its own
`chrome.storage.local`, reload only that extension, and verify that the daemon
lists two distinct profiles. Save the Ego Lite ID in Admin's `OPENCLI_PROFILE`;
do not change the global default or clear login/cookie data. The setting is
machine-specific and belongs in ignored `data/settings.json`.

The project patch navigates to ChatGPT before model inspection and gives
`chatgpt model` a 45-second browser deadline, inside a 75-second subprocess
ceiling. Failed preflight attempts caused by stale pages, timeouts, or busy
leases use fresh owned sessions, all cleaned up afterward. This recovery is
limited to model preflight, before any review prompt has been submitted;
the successful session is reused for the prompt and its response recovery.
The observed model must still pass the configured `medium..xhigh` gate.

## Partial ChatGPT answers and recovery

ChatGPT now also renders conversation turns as `section` elements. Generation
checks must locate turns by `data-testid`, including status text outside the
inner message, instead of requiring an `article` tag. Otherwise a paused stream
can look like a stable final answer, such as `W` or `W1P;2BF@2.3,2`.

`chatgpt detail` checks the current conversation ID before navigating. Reading
the already-owned target preserves its live stream; a fresh tab or a different
conversation still navigates to the requested ID. The Python review parser
continues to reject partial verdicts and unknown claim IDs. Both changes live
in the idempotent postinstall patch and have adapter regression coverage.

Live verification should call `backend.daily_news.review.review_daily_script`
with the saved task dossier, draft, and task configuration. This exercises the
configured model gate, project OpenCLI wrapper, pacing, owned-turn recovery,
claim validation, correction loop, and session cleanup together. Do not replace
that check with direct OpenCLI commands or a different model.
