# FrontierTechSN

FrontierTechSN automatically produces a daily video podcast about frontier technology. It gathers stories from sources in two languages and balances the selection for each edition. The pipeline writes a broadcast script, sends it for independent review, synthesizes narration, and assembles news imagery, optional public footage, and music. It then renders and validates the video and, when publishing is enabled, distributes it through accounts signed into the local browser.

The project uses a media pipeline derived from Video-Promotional. Runtime data, credentials, browser identities, generated episodes, and publication receipts stay local and are ignored by Git.

## Daily production

Explainer productions retain a persisted production slot across failures, repairs,
and service interruptions. The shared worker waits for that same project to be
resumed or explicitly cancelled before starting another video, including Morning
Desk. A Morning Desk run that is already executing finishes first. Ordinary new
plans keep the existing queue policy. Recovery does not automatically retry a
failed provider request; resume the original project to reuse its checkpoints.
The Explainer queue API exposes the retained slot as `reservation`, separately
from the task that is actually `active`. Use **Cancel and release slot** (or remove
a queued recovery) to intentionally allow other productions to proceed. Service
shutdown records `interrupted`; only an explicit cancellation records
`cancellation_reason=user`. Historical cancelled projects are not auto-resumed.
For maintenance, call `POST /api/explainers/{id}/pause` instead of `/cancel`;
after repairing, resume through the original `/script`, `/render`, or `/queue`
operation. Pausing retains the slot and provider checkpoints without retrying.

Each daily run follows these steps and stops if a required check fails:

1. Fetch all enabled sources in the 23-source catalog and require at least three working sources.
2. Deduplicate and balance six stories across source, language, and technology category.
3. Include the required morning-news opening with the edition date and the required spoken closing, both verbatim.
4. Use yhroot AI to generate the script from the evidence dossier alone.
5. Submit every material claim to Gemini Web through project-local OpenCLI. Each story group gets up to two Gemini attempts. If the review fails, times out, or returns a malformed response, start a fresh ChatGPT conversation at the configured non-Pro reasoning level. Correct and re-review the script when findings block approval.
6. Synthesize speech with WAV integrity checks and advisory ASR transcript comparisons.
7. Acquire news imagery and, when enabled, public footage. New Morning Desk editions disable Paper-Collage generation.
8. Generate instrumental music with Gemini Create Music, using deterministic local music as a fallback. Duck it to `-25 dB` under narration, side-chain it against speech, and match the program mix to the required duration.
9. Render with HyperFrames and run final A/V validation.
10. When publishing is enabled, use the accounts currently signed into the browser. Discover and record account identities and exact URLs at runtime rather than hard-coding them.

### Reviews and browser requests

All yhroot stages allow ten attempts: one initial call and nine retries. Most OpenCLI stages that use a browser have the same limit. The daily-news claim audit allows at most two Gemini attempts and one ChatGPT fallback. The default provider wait is 90 seconds.

Before each Web review, script constraints are checked again after duration edits and factual corrections. A failed check enters the shared bounded script editor instead of stopping immediately: it repairs failing story paragraphs using the dossier and preserves passing paragraphs. Each repair has up to three model responses; a review run allows at most three contract-repair cycles to prevent duration/edit oscillation. Changed paragraphs require fresh matching Web evidence. Repair inputs, outputs, and diagnostics are saved as `review/contract-repair-*`; exhausted repairs are recorded in `review/fact_check_report.json`.

Gemini briefly uses a foreground window because its composer requires trusted keyboard input. ChatGPT stays in the background. Each review request carries an ownership marker; recovery accepts only the assistant turn paired with that marker. Persistent OpenCLI site sessions use a FrontierTechSN namespace so another local OpenCLI project cannot navigate these Gemini or ChatGPT tabs.

Gemini generation requests (prompts, images, and new videos) use a random start-to-start interval sampled independently from 60 to 120 seconds for each request. The limiter persists across processes and restarts. ChatGPT uses a separate limiter, so its longer waits do not block Gemini. Read-only recovery, resumed video retrieval, conversation details, and status checks do not consume a generation slot. ChatGPT model checks do not reserve a slot either, but must wait until its previous generation's configured quiet period ends. Explicit provider access-limit cooldowns still apply separately.

Choose the ChatGPT interval from 10 to 30 minutes under **Admin → System → Footage Sources** with `OPENCLI_WEB_REQUEST_INTERVAL_SECONDS`; this setting does not override Gemini's 60–120 second random interval. The same panel configures the Gemini primary model, ChatGPT fallback level, and review timeout for each provider.

### Paper-Collage acquisition

Paper-Collage is disabled for all new Morning Desk editions, including scheduled, hourly, manual, and test runs. Existing tasks retain their saved configuration and assets for ongoing production, retries, and recovery. The acquisition behavior below applies to those existing tasks. Source changes take effect when the service next loads the updated code; do not restart active production to apply this change.

Each Paper-Collage clip targets the length of its narration scene, up to Gemini's configured limit for one generation (eight seconds by default). Public Footage and Paper-Collage can share a news story. A collage stays with its assigned story even when footage already occupies the scene. Change the duration limit under **Admin → System → Paper-collage B-roll** with `COLLAGE_GEMINI_MAX_SECONDS`.

Each Paper-Collage browser session belongs to a task, scene, and provider. Tabs stay open during retries. When the operation succeeds, fails, or is cancelled, it closes its session and checks that all owned tabs are gone. A persistent ownership journal supports cleanup after a worker restart or before cached acquisition results are reused. Cleanup leaves other sessions and user tabs alone and records failures for later recovery.

Gemini videos are fetched through the signed-in page and written directly to `outputs/<task_id>/collage_broll/<item>/video/gemini-web-original.mp4`. Transfers use a temporary file in that same directory and publish it only after all bytes arrive and the MP4 header is checked. Failed transfers remove their partial file and preserve any existing complete video. The browser's Downloads folder is not used.

Gemini video submissions use its 60–120 second randomized generation limiter. After a transport failure, a retry resumes the conversation associated with the same input if its URL is known, keeping the original deadline. If submission is uncertain and the URL cannot be recovered, the operation stops to avoid a duplicate submission.

A generation response that stays unchanged reaches the stall limit at `COLLAGE_GEMINI_STALL_TIMEOUT` (600 seconds by default, configurable in **Admin → System → Paper-collage B-roll**). Spinner animation does not count as progress. A stall, expired deadline, or explicit generation failure ends the acquisition without starting a fresh video. If the collage is optional, the pipeline can use available media instead. Explicit clip counts must still be met.

### Shot editing and footage recovery

Public-footage preview review allows ten attempts per candidate by default: up to five in a batch, then five individual attempts using the saved preview. Configure `WEB_FOOTAGE_PREVIEW_MAX_ATTEMPTS` (2–12) under **Admin → System → Footage Sources**. Explicit suitability rejections are not retried. Gemini image reviews reserve an additional 180 seconds for model selection, upload-menu hydration and verified image attachment before the response deadline; timed-out responses get bounded longer retries. The adapter uses the composer's image paste handler, verifies attachment readiness, and falls back to an image-capable file picker only when paste is unsupported. Native uploads are not dispatched twice. Downloaded previews, validated clips and every review attempt remain available when acquisition resumes.

After acquisition, an AI shot editor uses the available media to choose each story's shot order, durations, copy, and layout. There is no fixed media sequence: collage can come before or after footage, and editorial graphics, acquired photos, and article captures appear where they explain the narration.

The pipeline measures video durations with ffprobe and plays each video once. Any remaining narration gets a complete information layout, avoiding empty gradients and long frozen frames. An incomplete or invalid schedule gets validation feedback and one repair attempt. If the provider remains unavailable, the pipeline records a recovery plan that keeps verified assets and fills the remaining time with editorial graphics based on the story's evidence. Automatic collage selection can replan after a generation failure; an explicitly requested clip count must still be delivered.

`media_shots.json` records the AI's reasoning, exact shot intervals, source hashes, measured durations, recovery diagnostics, and coverage checks. A story's checkpoints become invalid when its narration, assets, or editor prompt changes. Before capture, the renderer checks the generated media elements against the schedule. The `av_sync_report.json` report includes `visual_coverage` along with audio and semantic quality checks.

Public-footage retries resume the saved shot plan if the script, orientation, provider, and requested count still match. Hybrid and YouTube clips must pass checksum and narration-binding checks before reuse. A changed binding requires a fresh review. When a search runs out of candidates, the pipeline uses the rejection evidence to choose new search directions.

### 4K delivery

Final videos default to **4K at 30 fps**, using HyperFrames' `high` quality H.264
SDR preset. Landscape delivery is 3840×2160; portrait delivery is 2160×3840.
The existing 1920×1080 / 1080×1920 CSS layouts stay unchanged: Chrome captures
them at twice the pixel density, including titles, captions and credits.

**Admin → System → Render** controls output density (`4k` or `1080p`), frame rate
and encoder quality. Existing explicit `.env` or Admin overrides are preserved;
upgrading an installation with overrides requires selecting `4k`, `30` and
`high`. Legacy resolution values `landscape`, `portrait` and `square` select
1080p density; the task's orientation determines the actual aspect ratio.
The selected render profile is saved in `av_sync_report.json`. Changing it
invalidates a pending review checkpoint. Validation checks both output pixels
and frame rate, followed by a complete audio/video decode before promotion.

YouTube footage acquisition prefers the highest available resolution within the
3840×2160 / 2160×3840 source limits, regardless of container. All download
fallbacks retain this policy. The default download ceiling is 250 MiB per file.
Normalization preserves source detail with H.264 CRF 18 and never enlarges a
low-resolution video to claim native 4K. Footage acquired under the old 720p
policy is reacquired when a task is rendered again; its old files and audit
history remain available. Article captures use a 1440×900 CSS viewport at 2×
density, producing 2880×1800 PNGs with crop coordinates converted to physical
pixels. Wikimedia still requests use a 3840-pixel target width. Generated
collage clips retain their available native resolution, up to the delivery
dimensions, and are normalized to 30 fps; provider output can still be below 4K.

The encoder uses constant-quality encoding, so final bitrate and file size
depend on content. A 4K output file does not imply every embedded source is
native 4K. Preview at 1080p when needed; final UHD rendering has a larger
pixel-aware deadline and may consume substantially more memory and time.
The quiet-period limit also scales with duration, pixels and frame rate because
software encoding may produce no progress output. Nested FFmpeg deadlines use
these same limits, avoiding the renderer's fixed ten-minute encoding cutoff.

If URL inspection is unavailable, one Gemini request can include up to four labelled contact sheets, with a separate suitability verdict for each candidate. Missing shots block rendering. The task's `footage/manifest.json` records missing searches, rejections, and invalidated clips; `footage/history/` keeps earlier ledgers.

## Source roster

The catalog lives in [`config/news_sources.json`](config/news_sources.json). The 23-source roster combines bilingual specialist sources with Bloomberg, Financial Times, The Wall Street Journal, CNBC Technology, BBC Technology, TechCrunch, The Verge, WIRED, MIT Technology Review, Nature, a16z, and Sequoia Capital. The Signal board uses the enabled catalog count and shows reporting, aggregators, institutional viewpoints, and access notes.

Every source, including institutional analysis, uses the edition's configured news window (36 hours by default), with no extra stale/future allowance. Legacy per-source `lookback_hours` values are ignored. Undated discovery cards must acquire a valid publication timestamp before selection. Analysis needs an article excerpt and competes in the ordinary ranking; it has no reserved slot.

Eligible analysis explains the attributed thesis and one supporting example in 3 to 4 sentences. It identifies the investor perspective and publication date and keeps source-supported limitations. Generation and final script checks limit each interpretation to 120 English words or 220 Chinese characters. The dossier records `content_kind` and `evidence_status` so public feed summaries remain distinguishable from article access.

Before ranking, every candidate receives an evidence-bound freshness assessment through the configured AI provider. It distinguishes the publication date, original event date, and latest substantive development, with a verbatim source quotation and date basis. A later article date or a new URL does not re-date an old announcement. New reporting or analysis can qualify on its publication date only when its substance is new. Date-only event evidence on the partially covered cutoff day is rejected when it cannot establish in-window timing. Invalid assessment responses get at most three attempts per batch. Repairs receive the previous response and candidate-specific validation errors, and only rejected candidates are requested again. Quotes must match a continuous span of captured evidence: whitespace, case and straight/curly quotation-mark differences are resolved back to the exact source text; title/summary splicing, paraphrases and changed facts remain invalid. Unavailable or unsupported evidence never falls back to eligibility. Research saves each attempt's input, response and validation diagnostics, plus the completed batch's validated decisions, under `research/freshness/`.

Relative date evidence also accepts unqualified weekdays such as "on Monday" or "周一", resolved to the most recent matching day at the captured publication date in the edition timezone. Qualified weeks such as "last Monday" or "next Monday" require stronger date evidence. Missing or duplicate candidate decisions receive feedback naming the affected IDs; valid decisions in a partial response are retained. During candidate discovery, a candidate that still fails individual validation after three attempts is excluded with a `validation_exhausted` receipt; accepted candidates are retained and research continues through the remaining candidates. Provider errors and malformed response envelopes still fail the assessment. Research still requires at least three eligible stories (or the configured maximum if lower), and reassessment of an already selected script remains strict.

Cross-edition comparison uses the previous 14 calendar days of **completed** daily editions in the configured task database, together with their saved selected stories and actual scripts. Failed, running and test tasks, the current task, and the same or later edition dates do not count as prior coverage. Repeated artifacts of one edition/article/script collapse into one historical item. URL normalization catches tracking/transport variants; semantic comparison checks all history across publishers, languages and rewritten titles, using the specific subject, action and milestone rather than broad topic similarity. A previously covered event requires a quoted, dated substantive change. Missing evidence for a relevant completed edition fails closed. This history horizon is the named `HISTORY_DAYS` policy constant; it is separate from the configurable news window.

The script must lead with the qualifying new development; an old product introduction cannot replace it. The existing independent live-web review receives the freshness evidence and completed-history snapshot, checks novelty as well as accuracy, and uses the existing bounded claim-correction loop for stale framing or omitted updates. Its cache keys include the new context. Review resume refreshes changed history, preserves the old dossier as an audit copy, and rejects selections that need fresh research. Final publication and completion are serialized across local workers; a prior edition completed after review blocks publication until review is resumed. Same-edition retries are deliberately excluded, so they cannot disqualify themselves. Legacy completed videos remain available for media-only rework; legacy **review** resumes must pass the new eligibility checks.

When a Web audit disputes freshness, the correction loop obtains a sourced explanation once per affected story and reassesses the captured evidence. If the story is stale, it checks up to three already researched candidates for a replacement, preserving passing story slots and source balance. The replacement paragraph requires a new independent Web verdict; a diagnostic never grants approval or expands the writable evidence. Diagnosis requests, raw responses, dossier backups, and replacement records are saved under the task's `review/` and `research/` directories. Resume can recover a saved diagnosis only when its entire request matches and its structured findings pass validation. Missing fresh replacements or repeated blocking verdicts still stop the bounded loop.

`config/news_sources.json` gives explicit preferences to a16z, Bloomberg, Financial Times, IEEE Spectrum and Nature. A qualified candidate in that source's `preferred_categories` receives `selection_bonus=6` with an article excerpt, or half that bonus with a public feed summary; headlines alone receive none. The allowed bonus range is 0–8, and all other sources default to zero. Bloomberg/FT favor business, technology policy, chips and AI; IEEE favors engineering fields; Nature favors science and AI research; a16z favors its technology, business and policy coverage. The bonus is applied after date and novelty eligibility, alongside recency, corroboration and topic/source diversity; it is neither a credibility verdict nor an admission guarantee. Each selected story records the awarded `preference_bonus`. Adding a new source does not exempt it from the common checks.

a16z uses article-list HTML because its RSS endpoint returns 404. Access to a public feed does not guarantee access to its articles. Each run's fetch audit records subscription requirements, authorization failures, and connection failures. No credentials or paywall workarounds are configured.

TLDR extracts all editorial blocks from its dated newsletter archive, skipping the first recruiting card. AIBase reads dated article records from its Chinese daily page and excludes navigation cards. ITHome publication times come from the article's publisher timestamp. GeekPark uses its official RSS, and Techmeme uses the feed summary after discarding archive notices.

Machine Heart uses its official article-library JSON endpoints to discover and read articles. The adapter converts local publication timestamps from Asia/Shanghai to UTC.

When the operating system resolver cannot connect to FT, the adapter uses an HTTPS DNS fallback scoped to FT. It preserves the original HTTP Host, TLS SNI, and certificate verification. Public addresses are resolved at runtime and cached for the DNS TTL, up to five minutes. They are never pinned in configuration, and system DNS stays unchanged.

Reuters and VentureBeat were removed from the active catalog on 2026-09-07: Reuters' public sitemap was accessible but article requests returned 401 without usable article evidence; VentureBeat's RSS and pages returned a Vercel Security Checkpoint (429), requiring a browser verification challenge. Neither is counted as monitored coverage. They can be reconsidered when an authorized, unattended access route is available.

Run `curl -X POST http://localhost:8101/api/daily-news/sources/check` to verify the deployed service's own production adapters. The check fetches every enabled source, reads up to three articles per source, requires at least one dated article excerpt or feed summary per source, and saves the exact evidence in `outputs/source-checks/`. `fresh_sample_count` is separate: a working periodic source need not have published within today's news window. HTTP success or navigation-only pages do not pass the check.

A blocked or changed website is recorded in `research/dossier.json`. That failure alone does not stop an edition: the source quorum and selected-story checks determine whether it can continue.

## Setup and start

Requirements: macOS, Python 3.11, Node.js/npm, Chrome with the OpenCLI Browser Bridge, FFmpeg/FFprobe, and the original local TTS model paths configured in `.env`.

Pocket TTS and Orpheus record ASR text comparisons as reference warnings.
Recognition differences, missing transcripts, and model adjudication do not
block generation, cause retries, or trim speech. Reports retain actual ASR
coverage and differences without claiming that warning-bearing audio was
verified word for word. Rendering accepts every generated part covered by the
`asr_advisory` manifest while preserving script/audio hashes, natural speed,
WAV decoding, duration, and physical continuity checks. Genuine audio failures
still retry with cached valid parts and complete sentence boundaries.
Pocket pause checks allow up to 0.20 seconds of word-timestamp drift when
identifying a sentence boundary; the 0.80-second internal and 1.20-second
sentence-boundary pause ceilings remain unchanged. Sentence recovery includes
closing quotes and brackets, and revalidates cached audio only when its source,
voice, deployment revision, and waveform hashes still match.

For YouTube/X media upload, open `chrome://extensions`, select **Details** for the ChatGPT browser extension, and enable **Allow access to file URLs**. Before creating a remote post, the publication adapter checks this permission and reports how to fix it if it is missing.

```bash
cp .env.example .env
./scripts/setup.sh
./scripts/opencli.sh doctor
./scripts/start.sh
```

### Isolated headless OpenCLI browser

The signed-in Chrome Browser Bridge is the default. To run web
automation without creating or focusing any window in the operator's Chrome,
configure the opt-in `isolated-headless` runtime under **Admin → System →
Footage Sources**. It starts a separate Chrome for Testing/Chromium process,
uses an ignored project-local user-data directory, loads a separate OpenCLI
extension instance, and pins every command to its exact Browser Bridge profile.
If any isolation boundary is unavailable, automation stops without falling back
to the operator's profile.

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
controlled shutdown. The profile directory survives restarts; the
manager never deletes it. Keep `bridge` selected until a complete production
run, including uploads and publishing, passes against the isolated accounts.

Open [http://localhost:8101](http://localhost:8101). The production launcher builds the frontend and serves the UI/API/outputs from one port. Override with `PORT` only when needed.

### Model routing

Configure model endpoints and credentials under **Admin → Models**. Routing
roles are saved in the application database; `.env` does not select the
primary or backup provider.

- Primary handles normal task traffic and accepts up to 64 API keys, one
  per line. Calls reserve keys through a persisted round-robin cursor, and an
  HTTP 429 before any model output rotates to an untried primary key first.
- Backup accepts one API key and activates only after the primary route
  exhausts its allowed attempts.
- Standalone providers are used only when a task explicitly selects them.

The route test in the same panel calls the task dispatcher and reports the
provider role and non-reversible key ID used. Agent turns are not replayed after
model or tool output begins. For requests with Skills and tools disabled, a 429
may discard partial text that has not been returned to the caller and retry
through the bounded cooldown/key/provider ladder. Observed tool activity still
blocks replay, and partial responses are never accepted as completed output.

### Scheduling and manual runs

With no saved desk configuration, the next edition runs automatically at 05:30 Asia/Singapore, publishes to YouTube and X through the accounts signed into Chrome, and updates the Apple Podcasts RSS feed. YouTube production visibility defaults to `public`. A first launch later than the two-hour morning window waits for the next scheduled edition instead of publishing stale news.

The **Morning Desk** page configures:

- daily execution time and IANA timezone;
- language, story count, and research window;
- automatic report length based on each story's evidence and complexity, with no fixed minute or word quota (including test runs); saved legacy duration settings are retired automatically;
- TTS voice/model and the public-footage switch; news imagery remains enabled, and AI chooses the media quantity;
- Gemini or deterministic local music;
- automatic distribution targets and YouTube visibility.

Use **Save & run test** for an end-to-end test edition. This saves the current desk recipe before queuing the task. A test never publishes automatically or consumes that day's scheduled edition. To publish a test, use the separate publishing action on the completed task.

Use **Start full run now** to save the current recipe and queue a full-length edition immediately. It generates video and uses the configured automatic distribution settings just like a scheduled edition, and counts as that day's production run.

The operator UI has three pages: **Morning Desk**, **Tasks**, and **Admin**. Bookmarks for the unavailable **New Task** workbench and **Content Plan** calendar redirect to Morning Desk. Manual task creation and content-planning APIs are not exposed. The database preserves historical planned tasks, whose output can still be read and removed from Tasks.

## Distribution

### YouTube

The adapter opens YouTube Studio and verifies both the visible channel name and stable Channel ID against **Admin → Publishing** before it uploads anything. It then uploads the final MP4, sets the title/description/audience, and publishes with the configured visibility. Production defaults to `public`; test runs use `private` and retain an exact deletion receipt.

### X

The adapter asks OpenCLI for the current handle and compares it with the expected `@handle` in **Admin → Publishing** before the composer opens. It attaches the final MP4, adds a unique task marker, and verifies the exact post on that handle before recording success.

### Apple Podcasts

The pipeline prepares RSS 2.0 at `/outputs/podcast/feed.xml` and adds the finished narration as an episode. Submission to Podcasts Connect is deferred until an Apple account and a stable public HTTPS feed URL are available.

Unattended publishing requires the global kill switch and target-platform switch to permit it, the task to opt in, and the active account identity to match the configuration exactly. If an identity is missing or mismatched, the completed task waits for review without posting.

### Test cleanup

The task page can delete only publications whose manifest says `test_mode: true`. X deletion requires the same active handle; YouTube deletion requires the same active channel. Both operations require an exact recorded URL/video ID. These deletes are permanent.

Publication receipts live under:

```text
outputs/<task-id>/publications/manifest.json
```

## Episode evidence

Task results include a **发布文案** panel with a YouTube upload title, Show Notes,
and one X post. Completed pipeline runs generate this copy in the background;
opening an older completed task generates its missing copy once. Each field can
be copied separately, and **重新生成** requests a fresh draft. Pipeline Logs are
available under the collapsed Execution flow details.

The copy is based on the saved narration and selected source references.
Every generation and repair request embeds the full pinned [blader/humanizer](https://github.com/blader/humanizer)
skill; its MIT license and upstream revision are retained in
`backend/prompts/vendor/humanizer/`. The writing prompt is editable in Admin.
Each task's `social_copy.json` stores the results and skill/source fingerprints.
Changing the script marks existing copy as stale. Interrupted or failed
generations can be retried without losing the previous good copy. This feature
drafts text for manual upload and does not submit posts or change publication receipts.

YouTube titles are limited to 100 characters and descriptions to 5,000. X copy
has 2 to 4 short sentences, a blank line after every two, at most two hashtags, and
at most 280 weighted characters including whitespace. The `twitter-text-parser`
dependency handles CJK, emoji, NFC normalization and URL weights. Its bundled
emoji data loader requires the `setuptools<81` compatibility bound in
`requirements.txt`. Length/format failures trigger up to two repair attempts;
invented URLs and chapter timestamps are rejected.

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

## Explainer animation watermark repair

In the explainer studio, open **Pacing & motion**, select **Gemini animation**,
and enable **Remove visible Gemini watermark locally** (on by default, including
older projects). Save the project and render again to repair an existing video;
completed narration, illustrations and Gemini downloads are reused. Saving as
defaults also preserves this choice for new projects.

The local FFmpeg `delogo` filter interpolates the selected area from surrounding
pixels before scaling, padding, titles and captions. It runs in the existing
render pass, leaves the original download intact, and preserves narration timing
and delivery dimensions. FFmpeg must include the `delogo` filter; a processing
error fails the render visibly, retaining completed source media for retry.

**Watermark area** accepts left/top/width/height percentages of the original
animation. The default covers the lower-right star in the measured 1280×720
Gemini explainer sample (roughly x=1120, y=560, 80×81 pixels). Adjust it for a
different placement; this is a fixed region repair, not automatic detection.
Fine detail inside that region may soften. Disable it for unmarked clips.
Changing the toggle or area invalidates only the affected scene renders and final
delivery. Each repaired scene records its method and source-pixel region in the
manifest and timeline. Still scenes and public-footage news videos are unaffected.

## Development

For incident diagnosis, shared contracts, generated invariant tests and runtime
acceptance, see [Pipeline reliability method](docs/PIPELINE_RELIABILITY_METHOD.md).

```bash
# Backend tests
.venv/bin/python -m pytest backend/tests -q

# Frontend checks
cd frontend
npx eslint src/components/MorningDesk.tsx src/components/PublicationPanel.tsx src/App.tsx src/api.ts
npm run build
```

Git ignores runtime outputs. Commit each implementation change on local `main`. Setup does not configure a remote or push commits.
