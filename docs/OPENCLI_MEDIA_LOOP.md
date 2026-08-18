# OpenCLI media loop: web footage + ChatGPT thumbnails

## Product contract

The footage loop turns a narration script into attributable, editable B-roll without expanding
Claude Code's global state. OpenCLI is pinned in `tools/opencli/package.json`; its upstream skills
and the project workflow are stored in `.claude/skills`. The backend executes only
`scripts/opencli.sh` and the project virtualenv's `yt-dlp`.

```text
narration script
  -> viral thumbnail formula -> ChatGPT Web image -> thumbnail image + prompt + manifest
  -> query plan
  -> Wikimedia search -----------------------> explicit-open-license candidate
  -> project yt-dlp ------------> YouTube ----> rights-review candidate
  -> Gemini Web link analysis (YouTube) ------> trim JSON or recorded fallback
  -> FFprobe + FFmpeg ------------------------> silent normalized clip + evidence frames
  -> footage/manifest.json -------------------> visual plan -> HyperFrames -> final MP4
```

OpenCLI is used where a real signed-in browser session matters. It connects to ChatGPT Web for
script-driven cover generation and to Gemini Web for visual interval analysis. After HyperFrames
renders the MP4, the pipeline also extracts a midpoint frame for every narration scene, combines
them into labeled contact sheets, and uploads each sheet with the acoustic scene excerpts to
Gemini. Missing uploads, omitted scene ids, and malformed responses are retried. Low semantic
scores are quality findings rather than transport failures, so they are retained as warnings. In
both cases the video is still delivered and the final `av_sync_report.json` records the detail.
YouTube discovery/download stays on `yt-dlp`.

Microsoft VibeVoice remains the only narration generator. MLX Whisper is a read-only, post-TTS
analyzer used to place word timestamps on the finished VibeVoice WAV. HyperFrames is not used for
speech generation or transcription. If MLX transcription retries are exhausted, scene timing falls
back to the existing silence/word estimate and delivery continues with an explicit warning.

## Setup and isolation

1. Run `./scripts/setup.sh` or `npm install --prefix tools/opencli --no-audit --no-fund`.
2. Install the official OpenCLI Browser Bridge Chrome extension and keep Chrome running.
3. Sign in to Gemini and ChatGPT as needed.
4. Run `./scripts/opencli.sh doctor`.
5. Verify `./scripts/opencli.sh chatgpt status -f json` reports `Login: Yes`.
6. In the New Task form leave Viral thumbnail enabled; select Hybrid or Web Platforms for web footage.

The pinned OpenCLI install receives a project-local postinstall patch that adds `gemini ask --file`.
It uses the same `File`/`DataTransfer` compatibility path as OpenCLI's other upload-capable adapters
when Chrome rejects CDP `setFileInput`. The patch is stored under `tools/opencli/patches/`; it does
not create a user adapter under `~/.opencli`.

Project isolation can be audited with:

```bash
./scripts/opencli.sh --version
find .claude/skills -maxdepth 2 -name SKILL.md -print
```

No workflow step writes to `~/.claude`, installs `opencli` globally, or changes global Claude
Code settings. Browser login/extension state is intentionally user-owned Chrome state.

## Thumbnail contract

The final audio narration script—not the summary or source transcript—is the sole content input
to thumbnail art direction. `backend/prompts/thumbnail.txt` contains the independently editable
master formula and is loaded on every task. The configured AI provider resolves one visual story,
then the backend executes only this project wrapper:

```bash
./scripts/opencli.sh chatgpt image "<final prompt>" \
  --op outputs/<task-id>/thumbnail/images \
  --window background --site-session persistent --keep-tab false -f json
```

The backend accepts only valid image-signature files inside that task's thumbnail image directory.
`thumbnail/manifest.json` records generation status, source script, final prompt path, all exported
image paths, the primary image, ChatGPT conversation link, timestamps, and any error. A provider,
bridge, or web failure is logged and recorded but does not discard a valid script or block TTS/video.

## Manifest and failure semantics

Every retained clip has a canonical source link, creator, provider, source duration, exact trim,
script excerpt, analyzer/status/reason, normalized media facts, SHA-256, and three evidence frame
paths. Manifest writes are atomic. Raw downloads are deleted only after the normalized derivative and
evidence frames succeed; failed raw files are retained for diagnosis.

Gemini's web product can usually inspect public YouTube links, but OpenCLI may time out before a
slow response becomes visible. The scout polls the same conversation for late assistant JSON
before using a low-confidence `deterministic-safe-offset` result.

Downloading is not a rights grant. Wikimedia clips pass an explicit license allowlist. Web clips
are always `rights_status=review_required` and add a publication blocker. The editor must review
the original page and reuse terms before publishing.

## TTS-free validation

For thumbnail-only validation, call `pipeline.thumbnail.generate_thumbnail` with a checked text
script and inspect `thumbnail/prompt.txt`, the image, and `thumbnail/manifest.json`. This path does
not invoke a TTS subprocess. For the footage/video path, use a text script and a short silent
timing WAV rather than generating local TTS, then compose at draft quality. Verify the final file
with:

```bash
ffprobe -v error -show_entries format=duration:stream=codec_name,width,height,pix_fmt \
  -of json outputs/<task-id>/video.mp4
```

The checked development fixtures `outputs/opencli-e2e/` and `outputs/thumbnail-e2e/` are
intentionally ignored by Git.

## Next product bets

1. **Shot quality ranker.** Score motion continuity, sharpness, occlusion, subtitles, faces, logos,
   and watermarks from evidence frames; let Gemini explain only the top candidates.
2. **Visual continuity pass.** Match color temperature, camera direction, subject scale, and motion
   between neighboring shots; generate per-clip LUT/crop instructions for FFmpeg.
3. **Script-to-shot memory.** Store successful query/shot/trim pairs as a local searchable ledger.
   Similar future narration can reuse proven sources without repeating browser work.
4. **Beat-aware editing.** Derive energy and cut points from the final audio waveform, then align
   B-roll entrances and transitions without regenerating narration.
5. **Motion-aware multimodal review.** The shipped evidence-first pass judges one rendered midpoint
   frame per scene. Add a second early/late frame only for scenes whose visual plan contains motion
   or B-roll, so continuity can be checked without tripling every browser upload.
6. **Rights inbox.** Present all web candidates in one approval queue with source snapshot, creator
   terms, intended use, and accept/replace controls; only approved clips can clear publication.
7. **Self-healing web adapters.** Run the project copies of `opencli-browser-sitemap` and
   `opencli-autofix` against changed sites, then require a recorded smoke test before adopting a
   repaired action.
8. **Automatic reframing.** Detect subject/saliency across sampled frames and animate a safe crop
   for portrait exports rather than using a fixed center crop.

The first two bets most directly improve perceived video quality; the rights inbox is the most
important release-safety investment. Frame upload is the highest-value OpenCLI extension because
it keeps link-analysis failures auditable.
