# FrontierTechSN

FrontierTechSN is a fully automated daily frontier-technology video-podcast desk. It researches a bilingual source roster, selects a balanced edition, writes and independently reviews a broadcast script, synthesizes speech, generates matching Paper-Collage B-roll and music, renders a verified video, and distributes it through the accounts signed into the local browser.

The project is a new repository derived from the proven media pipeline in Video-Promotional. Runtime data, credentials, browser identities, generated episodes, and publication receipts are local and ignored by Git.

## Daily edition contract

Every daily run is fail-closed around these checks:

1. Fetch all 12 configured sources and require at least three working sources.
2. Deduplicate and balance six stories across source, language, and technology category.
3. Inject an exact, date-stamped morning-news opening and an exact spoken closing.
4. Generate the script from the evidence dossier only through yhroot AI.
5. Submit every material claim to Gemini Web through project-local OpenCLI. Each story group gets up to two bounded Gemini attempts; a failed, late, or malformed primary review falls back to a fresh ChatGPT conversation at the configured non-Pro reasoning level. Blocking findings trigger a corrected-script review cycle.
6. Synthesize TTS using the same engines and complete spoken-text integrity gate as Video-Promotional.
7. Generate four Paper-Collage B-roll clips by default and hard-fail if requested and placed clip counts differ.
8. Generate an instrumental Gemini Create Music bed (with a deterministic local musical fallback), duck it to `-25 dB` under narration, side-chain it against speech, and duration-lock the program mix.
9. Render with HyperFrames and run final A/V validation.
10. When enabled, publish with the current signed-in browser accounts. Account identities and exact URLs are discovered at run time and recorded; nothing is hard-coded.

All yhroot stages get ten total attempts (one initial call plus nine retries). Most browser-backed OpenCLI stages also get ten attempts. The daily-news claim audit instead makes at most two Gemini primary attempts and one ChatGPT fallback, so a slow web turn cannot consume the old ten-attempt review window. Gemini briefly uses a foreground window because its current composer requires trusted keyboard input; ChatGPT stays in a background window. Every review request carries an ownership marker, and recovery accepts only the assistant turn paired with that marker. FrontierTechSN also namespaces its persistent OpenCLI site sessions, preventing another local OpenCLI project from navigating these Gemini or ChatGPT tabs. The default provider wait is 90 seconds. Gemini and ChatGPT prompt/image submissions retain the cross-process start limiter within this checkout: adjacent generation requests begin at least three minutes apart. Read-only recovery, conversation detail, status checks, and model selection do not consume a generation slot. Operators can choose 3–10 minutes under **Admin → System → Footage Sources** with `OPENCLI_WEB_REQUEST_INTERVAL_SECONDS`; the Gemini primary model, ChatGPT fallback level, and per-provider review timeout are configured in the same panel.

Each Paper-Collage clip targets the duration of its selected narration scene, capped by Gemini's configured single-generation limit (eight seconds by default). The generated assembly plays once and then holds its completed final frame for the remainder of a longer scene; neither FFmpeg nor HyperFrames replays it. Operators can change the provider ceiling under **Admin → System → Paper-collage B-roll** with `COLLAGE_GEMINI_MAX_SECONDS`.

## Source roster

The catalog lives in [`config/news_sources.json`](config/news_sources.json). Primary desks are Techmeme, TLDR AI, 机器之心 AI Daily, 量子位 QbitAI, IEEE Spectrum, and DeepTech 深科技. Secondary desks are AIBase AI 日报, IT之家 AI / 智能时代, 极客公园, The Rundown AI, Ars Technica, and VentureBeat AI.

One blocked or changed website cannot abort an edition. Its failure is preserved in `research/dossier.json`; the source quorum and selected-story gates decide whether the edition may continue.

## Setup and start

Requirements: macOS, Python 3.11, Node.js/npm, Chrome with the OpenCLI Browser Bridge, FFmpeg/FFprobe, and the original local TTS model paths configured in `.env`.

For YouTube/X media upload, open `chrome://extensions`, select **Details** for the ChatGPT browser extension, and enable **Allow access to file URLs**. The publication adapter detects this missing permission before it can create a remote post and returns an actionable error instead of silently waiting through retry delays.

```bash
cp .env.example .env
./scripts/setup.sh
./scripts/opencli.sh doctor
./scripts/start.sh
```

### Isolated headless OpenCLI browser

The existing signed-in Chrome Browser Bridge remains the default. To run web
automation without creating or focusing any window in the operator's Chrome,
configure the opt-in `isolated-headless` runtime under **Admin → System →
Footage Sources**. It starts a separate Chrome for Testing/Chromium process,
uses an ignored project-local user-data directory, loads a separate OpenCLI
extension instance, and pins every command to its exact Browser Bridge profile.
It fails closed when any isolation boundary is unavailable and never falls back
to the human profile.

One visible bootstrap is required to discover the dedicated profile and sign in:

```bash
# First configure the isolated Chrome binary, user-data directory, and unpacked
# OpenCLI extension path in Admin. Chrome for Testing is recommended because
# current normal Chrome releases ignore command-line unpacked extensions.
./scripts/opencli-browser-runtime.sh bootstrap
./scripts/opencli.sh profile list
./scripts/opencli.sh profile rename <context-id> frontiertechsn-headless

# Sign into Gemini, ChatGPT, YouTube, and X in the dedicated bootstrap window.
# Save frontiertechsn-headless as OPENCLI_ISOLATED_PROFILE, then stop it.
./scripts/opencli-browser-runtime.sh stop
```

After saving `OPENCLI_BROWSER_RUNTIME=isolated-headless` and restarting the app,
the first OpenCLI command starts the same dedicated profile in headless mode.
Use `./scripts/opencli-browser-runtime.sh status` to inspect it and `stop` for a
controlled shutdown. The profile directory is retained across restarts; the
manager never deletes it. Keep `bridge` selected until a complete production
run, including uploads and publishing, passes against the isolated accounts.

Open [http://localhost:8101](http://localhost:8101). The production launcher builds the frontend and serves the UI/API/outputs from one port. Override with `PORT` only when needed.

### Model routing

Configure model endpoints and credentials under **Admin → Models**. Routing
roles are persisted in the application database; `.env` does not select the
primary or backup provider.

- **Primary** carries normal task traffic and accepts up to 64 API keys, one
  per line. Calls reserve keys through a persisted round-robin cursor, and an
  HTTP 429 before any model output rotates to an untried primary key first.
- **Backup** accepts one API key and activates only after the primary route's
  bounded attempts are exhausted.
- **Standalone** providers are used only when a task explicitly selects them.

The route test in the same panel exercises the real task dispatcher and reports
the provider role and non-reversible key ID actually used. Once model or tool
output has begun, the dispatcher refuses to replay that turn through another
key or provider.

With no saved desk configuration, the next edition runs automatically at 05:30 Asia/Singapore, publishes to YouTube and X through the accounts signed into Chrome, and updates the Apple Podcasts RSS feed. YouTube production visibility defaults to `public`. A first launch later than the two-hour morning window waits for the next scheduled edition instead of publishing stale news.

The **Morning Desk** page configures:

- daily execution time and IANA timezone;
- language, duration, story count, and research window;
- TTS voice/model, Paper-Collage clip count, and the separate public-footage clip budget;
- Gemini or deterministic local music;
- automatic distribution targets and YouTube visibility.

Use **Save & run 1-min test** for an end-to-end test edition. The current desk recipe is saved before the task is queued, the test never auto-publishes, and it does not consume that day's scheduled edition. External test publishing is a separate explicit action on the completed task.

Use **Start full run now** to save the current recipe and immediately queue a full-length edition without waiting for the schedule. It follows the same video-generation and configured automatic-distribution path as a scheduled edition, and counts as that day's production run.

The operator UI is deliberately limited to **Morning Desk**, **Tasks**, and **Admin**. The manual **New Task** workbench and **Content Plan** calendar inherited from Video-Promotional are not part of this autonomous-desk product; old browser bookmarks redirect to Morning Desk. Manual task creation and content-planning APIs are not exposed. The database keeps non-destructive compatibility with historical planned tasks so their output remains readable and removable from Tasks.

## Distribution

### YouTube

The adapter opens YouTube Studio and verifies both the visible channel name and stable Channel ID against **Admin → Publishing** before it uploads anything. It then uploads the final MP4, sets the title/description/audience, and publishes with the configured visibility. Production defaults to `public`; test runs use `private` and retain an exact deletion receipt.

### X

The adapter asks OpenCLI for the current handle and compares it with the expected `@handle` in **Admin → Publishing** before the composer opens. It attaches the final MP4, adds a unique task marker, and verifies the exact post on that handle before recording success.

### Apple Podcasts

The pipeline prepares RSS 2.0 at `/outputs/podcast/feed.xml` and adds the finished narration as an episode. External Podcasts Connect submission is deliberately deferred until an Apple account and a stable public HTTPS feed URL are available.

Unattended publishing is fail-closed: the global kill switch, target-platform switch, task-level opt-in, and exact configured account identity must all agree. Missing or mismatched identities leave the completed task awaiting review and perform no social write.

### Test cleanup

The task page can delete only publications whose manifest says `test_mode: true`. X deletion requires the same active handle; YouTube deletion requires the same active channel. Both operations require an exact recorded URL/video ID. These deletes are permanent.

Publication receipts live under:

```text
outputs/<task-id>/publications/manifest.json
```

## Episode evidence

Each task directory contains the evidence needed to audit an edition:

```text
research/dossier.json                 source fetches, candidates, selected stories
research/dossier.md                   human-readable claim ledger
review/story-review-prompt-*.txt      independent fact-check requests
review/story-review-response-*.txt    raw primary and fallback responses
review/fact_check_report.json         approval and correction cycles
tts/                                  engine manifest and speech-integrity evidence
collage_broll/manifest.json           requested/generated/placed Paper-Collage clips
music/manifest.json                   prompt, provider, media hash and fallback reason
audio/music_mix_report.json           levels, durations, loudness and mixed-audio hash
final_audio_integrity_report.json     narration/program/final waveform integrity
final_multimodal_review.json          per-scene Gemini A/V relevance scores
av_sync_report.json                   final delivery quality gate
publications/manifest.json            discovered platform identity and lifecycle receipts
video.mp4                              final rendered program
```

## Development

```bash
# Backend tests
.venv/bin/python -m pytest backend/tests -q

# Frontend checks
cd frontend
npx eslint src/components/MorningDesk.tsx src/components/PublicationPanel.tsx src/App.tsx src/api.ts
npm run build
```

Runtime outputs are ignored. Every implementation change is committed on local `main`; no remote is configured or pushed by setup.
