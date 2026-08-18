# FrontierTechSN

FrontierTechSN is a fully automated daily frontier-technology video-podcast desk. It researches a bilingual source roster, selects a balanced edition, writes and independently reviews a broadcast script, synthesizes speech, generates matching Paper-Collage B-roll and music, renders a verified video, and distributes it through the accounts signed into the local browser.

The project is a new repository derived from the proven media pipeline in Video-Promotional. Runtime data, credentials, browser identities, generated episodes, and publication receipts are local and ignored by Git.

## Daily edition contract

Every daily run is fail-closed around these checks:

1. Fetch all 12 configured sources and require at least three working sources.
2. Deduplicate and balance six stories across source, language, and technology category.
3. Inject an exact, date-stamped morning-news opening and an exact spoken closing.
4. Generate the script from the evidence dossier only through yhroot AI.
5. Submit every material claim to ChatGPT Web through project-local OpenCLI; blocking findings trigger a corrected-script review cycle.
6. Synthesize TTS using the same engines and complete spoken-text integrity gate as Video-Promotional.
7. Generate four Paper-Collage B-roll clips by default and hard-fail if requested and placed clip counts differ.
8. Generate an instrumental Gemini Create Music bed (with a deterministic local musical fallback), duck it to `-25 dB` under narration, side-chain it against speech, and duration-lock the program mix.
9. Render with HyperFrames and run final A/V validation.
10. When enabled, publish with the current signed-in browser accounts. Account identities and exact URLs are discovered at run time and recorded; nothing is hard-coded.

All yhroot stages get ten total attempts (one initial call plus nine retries). Browser-backed OpenCLI stages also get ten attempts. A provider turn that has already exceeded its whole-turn hard timeout is stopped immediately because the underlying CLI has already spent that turn retrying.

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

Open [http://localhost:8101](http://localhost:8101). The production launcher builds the frontend and serves the UI/API/outputs from one port. Override with `PORT` only when needed.

The **Morning Desk** page configures:

- daily execution time and IANA timezone;
- language, duration, story count, and research window;
- TTS voice/model and Paper-Collage clip count;
- Gemini or deterministic local music;
- automatic distribution targets and YouTube visibility.

Use **Run 1-min test** for an end-to-end test edition. It never auto-publishes; external test publishing is a separate explicit action on the completed task.

## Distribution

### YouTube

The adapter opens YouTube Studio, discovers the current channel from the signed-in session, uploads the final MP4, sets the title/description/audience, and publishes with the configured visibility. The safe default is `private`.

### X

The adapter asks OpenCLI for the current handle, attaches the final MP4 in the X composer, adds a unique task marker, and verifies the exact post on that handle before recording success.

### Apple Podcasts

The pipeline prepares RSS 2.0 at `/outputs/podcast/feed.xml` and adds the finished narration as an episode. External Podcasts Connect submission is deliberately deferred until an Apple account and a stable public HTTPS feed URL are available.

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
review/chatgpt-prompt-*.txt           independent fact-check requests
review/chatgpt-response-*.txt         raw review responses
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
