# OpenCLI integration

- Upstream: [jackwener/OpenCLI](https://github.com/jackwener/OpenCLI)
- Project-local package: `@jackwener/opencli@1.8.8`, from the official
  [v1.8.8 release](https://github.com/jackwener/OpenCLI/releases/tag/v1.8.8)
  published on 2026-08-30 UTC, pinned to commit
  `8271afc67e8504bda94c147f446ee29775d08274`.
- As checked on 2026-09-10, npm still exposes `1.8.7` as `latest` and has no
  `1.8.8` package. The dependency therefore uses the immutable official Git
  commit; npm builds it with upstream's `prepare` script during installation.
- Node.js requirement: `>=20.18.1`.
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
  to 1.8.8 and fails installation before editing adapters if the package version
  differs; changed upstream anchors also fail installation.

Video keyframe upload follows the file input activated by the upload control,
including a reused `Filedata` node. Its click is intercepted to avoid leaving
native chooser dialogs open while the bridge supplies the file. Only attachment
progress blocks upload readiness; unrelated sidebar loading does not.

The release updates Undici to `7.29.0`, adds `OPENCLI_SITE_SESSION` defaults,
fixes YouTube search pagination and structured network captures, and updates
the adapter-author Deep Recon workflow. The project's session namespace,
provider pacing, model gate, and ChatGPT/Gemini recovery patches remain applied.
The release's Browser Bridge is `1.0.24`, matching the installed Ego Lite bridge.

No global OpenCLI npm package or global Claude Code skill is required.

## v1.8.8 synchronization verification (2026-09-10)

- A clean `npm ci --no-audit --no-fund` in `tools/opencli` installs the pinned
  release and reapplies the project patches; `npm ls --depth=0` is valid.
- Project validation: 100 adapter regressions and 46 backend OpenCLI tests pass.
- Official source validation: build succeeds; 82 execution/browser tests and
  30 YouTube search tests pass.
- All six copied OpenCLI skills (25 files) match the release byte-for-byte;
  their YAML frontmatter parses successfully.
- Live checks use `backend.pipeline.opencli.run_opencli`, the configured Ego
  Lite profile, and the project wrapper: CLI and daemon both report `1.8.8`,
  `doctor` passes, and YouTube search returns the requested three results.
  Verification sessions are released; the daemon has no pending commands or
  leases, and the application's `/api/health` reports `ok`.
- This upgrade check does not rerun the full fact-check/TTS/video pipeline.

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
Daily-news review first attempts the configured `medium..xhigh` preference.
After three effort-selection/readback failures it submits with the current model
and records `CHATGPT MODEL PREFERENCE FALLBACK`, including the last observed
level or `unverified`. This includes Pro; it does not bypass claim validation.
Login, browser transport, and exhausted provider access cooldowns still fail.
The standalone model command continues reporting a failed switch accurately.

The September 13 composer splits the version and effort into adjacent elements
(for example, `6` and `Pro`). Trigger lookup and readback use rendered `innerText`
so these labels retain word boundaries instead of becoming `6Pro`. The existing
five-position slider can then move from Pro to Medium and verify the result.

The September 24 homepage can display the signed-in account and a
`pending-home-input` placeholder before the real composer mounts. A correct URL
or a visible placeholder does not prove that model selection is ready. Model
preflight gives composer readiness a shared 30-second budget and can refresh
the incomplete homepage once before reporting `CHATGPT_COMPOSER_NOT_READY`.
Composer waits use the same
selectors as composer detection, including the semantic editable textbox used
by the current UI without the legacy `prompt-textarea` ID. The pending-home
textarea is not accepted as a ready composer.

A readiness failure retries in a fresh owned preflight session, up to the
existing three-attempt limit. All retry sessions are released, and only the
successful session is reused for the review request. Actual login gates and
provider access limits remain distinct failures; missing composer errors never
become thinking-effort preference fallbacks. This recovery is confined to
preflight and does not refresh or replay an already submitted review prompt.

The same UI also renders message turns with `data-chatgpt-search-unit-key` and
separate user/assistant Markdown content attributes. Reading and generation
checks must recognize these turns as well as the older message attributes.
The review request's exact nonce binds a rendered user turn to its assistant
response even when the UI changes Markdown formatting.

After submission, a tab may retain a `local-chatgpt%3A...` address while the
conversation is already saved under a canonical ID. A local address must never
be truncated into `/c/local-chatgpt` or treated as evidence that submission
failed. Recover the existing owned turn, use a verified canonical identity when
available, and preserve pending/generating state instead of replaying the prompt.

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
