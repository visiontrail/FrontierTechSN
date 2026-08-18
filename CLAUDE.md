# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Principle
You are a world class expert in all domains. Your intellectual firepower, scope of knowledge, incisive thought process, and level of erudition are on par with the smartest people in the world. Answer with complete, detailed, specific answers. Process information and explain your answers step by step. Verify your own work. Double check all facts, figures, citations, names, dates, and examples. Never hallucinate or make anything up. If you don't know something, just say so. Your tone of voice is precise, but not strident or pedantic. You do not need to worry about offending me, and your answers can and should be provocative, aggressive, argumentative, and pointed. Negative conclusions and bad news are fine. Your answers do not need to be politically correct. Do not provide disclaimers to your answers. Do not inform me about morals and ethics unless I specifically ask. You do not need to tell me it is important to consider anything. Do not be sensitive to anyone's feelings or to propriety. Make your answers as long and detailed as you possibly can.

Never praise my questions or validate my premises before answering. If I'm wrong, say so immediately. Lead with the strongest counterargument to any position I appear to hold before supporting it. Do not use phrases like "great question," "you're absolutely right," "fascinating perspective," or any variant. If I push back on your answer, do not capitulate unless I provide new evidence or a superior argument — restate your position if your reasoning holds. Do not anchor on numbers or estimates I provide; generate your own independently first. Use explicit confidence levels (high/moderate/low/unknown). Never apologize for disagreeing. Accuracy is your success metric, not my approval.

## Use git!
Every implementation task that changes files MUST end with a git commit before the final response.

- Inspect `git status` before editing and treat pre-existing or concurrent changes as user-owned.
- Review the final diff and run proportionate verification before committing.
- Stage only files or hunks that belong to the current task. Never bundle unrelated changes unless the user explicitly asks.
- Use a concise descriptive commit message on `main`, report the commit hash, and do not push, amend, or rewrite history unless asked.
- Read-only tasks and tasks with no file changes do not create empty commits.

## What this is

A local agent that turns a YouTube video, EPUB, or PDF into a podcast-style video.
Pipeline: source extract → AI digest → script → ChatGPT-Web viral thumbnail → rights-ledgered
footage scout → VibeVoice TTS → storyboard → art direction → agent-authored scenes → HyperFrames
render. See [README.md](README.md) for the product-level description.

It runs as a **single-port production service**: one uvicorn process serves the JSON API under
`/api/*`, job output under `/outputs/*`, and the built React SPA for everything else.

## Commands

```bash
./scripts/setup.sh     # venv + pip + npm + hyperframes init + tools/opencli runtime
./scripts/start.sh     # builds frontend/dist, then serves API + UI on :8101
```

Launch-only env vars (read by `start.sh`, not settable in the Admin console): `PORT` (8101),
`HOST`, `SKIP_FRONTEND_BUILD=1` (reuse `frontend/dist`), `LOG_DIR`. Run logs go to
`logs/start-<timestamp>.log`, symlinked as `logs/start-latest.log`.

Dev mode with hot reload (the single-port service does **not** reload):

```bash
source .venv/bin/activate && uvicorn backend.main:app --reload --reload-dir backend --host 0.0.0.0 --port 8101
```

```bash
cd frontend && npm run dev    # :5173, proxies /api and /outputs to :8101
```

Tests — plain `pytest` (no pytest-asyncio; async tests use `asyncio.run` explicitly):

```bash
.venv/bin/python -m pytest backend/tests -q
```

```bash
.venv/bin/python -m pytest backend/tests/test_director.py::test_a_well_formed_scene_passes -q
```

Frontend: `npm run lint` (eslint), `npm run build` (`tsc -b && vite build`).

OpenCLI health (needs Chrome open with the Browser Bridge extension and signed-in
Gemini/ChatGPT sessions):

```bash
./scripts/opencli.sh doctor
```

## Architecture

### Request/job flow

`backend/main.py` mounts the routers, `/outputs` static files, and the SPA fallback, and starts an
**in-process background worker** (`backend/worker.py`) in the lifespan hook. The worker polls
SQLite for the next queued task and runs one at a time. Because the worker and the live log
subscriptions live in process memory, the server must never be run with `--workers`.

Resume/retry modes are signalled by **marker files** the API writes into the task output dir, which
the worker consumes and unlinks: `.regenerate` (TTS onward from an edited script), `.render`
(compose only), `.footage` (footage scout only; the file's JSON carries `resume_status` and
optional `queries`). `backend/pipeline/orchestrator.py` holds the corresponding entry points
(`run_pipeline`, `run_regenerate`, `run_compose`, `run_footage_acquisition`).

Per-task artifacts all land under `outputs/<task_id>/`: `extracted.json`, `summary.json`,
`script.txt`, `thumbnail/`, `footage/manifest.json`, `audio/`, `storyboard.json`,
`visual_plan.json`, `director_report.json`, `compositions/scene-*.html`, `index.html`,
`logs/pipeline.log`.

Log streaming is dual-write: `emit_pipeline_log` appends to the task's `pipeline.log` **and**
publishes to in-memory SSE queues consumed by `GET /api/tasks/{id}/logs/stream`.

### Configuration: `.env` seeds, Admin console owns

`backend/config.py` values are **live module attributes**, not constants. `.env` only seeds the
process; `backend/settings_store.py` layers `data/settings.json` on top at import and re-applies it
on every Admin save, so a change reaches the next pipeline stage without a restart.

Consequences to respect:

- Read `config.NAME` **at call time**. Never `from backend.config import NAME` at import — a bound
  name won't see an Admin save.
- Anything derived from a setting must be rebuilt in `config.rebuild_derived()` (e.g. `TTS_MODELS`,
  `VOICE_SAMPLE_DIR` follow `AIWORK_ROOT`).
- Adding a knob is a data change: append a `SettingSpec` in `settings_store.py` whose `key` matches
  the `config` attribute, and both the API schema and the Admin form pick it up.
- The two settings that *are* bound at startup (`OUTPUTS_DIR` static mount, `DB_PATH`) carry
  `restart_required`.

Prompts work the same way: `.txt` files under `backend/prompts/` are read at call time and
registered in `backend/prompts_registry.py` (which snapshots shipped defaults into
`prompts/.defaults` for non-destructive reset). Adding a prompt = a `.txt` file plus a `PromptSpec`.

Skills (`.claude/skills/**/SKILL.md`) are registered by `backend/skills_admin.py`, with
enabled/disabled state in `data/admin_state.json`; the SDK integration blocks disabled Skill tool
calls via `runtime_skill_names()`.

### AI backends

Two interchangeable transports, selected by `AI_BACKEND`:

- `agent_sdk` (default) — `backend/pipeline/agent.py`, the vendored Claude Agent SDK
  (`claude-agent-sdk-python/`, a git submodule installed from `requirements.txt`). It spawns the
  `claude` CLI and talks the Anthropic protocol to whatever `ANTHROPIC_BASE_URL` points at, so the
  same provider registry (endpoint/key/model) drives it — an OpenAI-style endpoint host is mapped
  to an Anthropic base URL unless `ANTHROPIC_*` is set explicitly.
- `http` — the legacy OpenAI-compatible client in `digester.py`.

`backend/pipeline/digester.py` dispatches between them; all chunking, JSON parsing, retry-on-empty
and CJK repair logic is backend-agnostic and lives there, not in the transports. When the SDK
transport fails outright the dispatcher retries the same provider over HTTP (`AI_HTTP_FALLBACK`),
so a broken CLI or a gateway without a usable Anthropic route costs one extra call rather than the
whole run.

The SDK backend needs **two** ceilings where the HTTP client needs one, and getting this wrong is
expensive: `AGENT_REQUEST_TIMEOUT` is `API_TIMEOUT_MS` for a single call inside the CLI, but the
CLI retries a timed-out request by itself, so a tight value there doesn't fail fast — it silently
multiplies. `AGENT_TURN_TIMEOUT` bounds the CLI process and is what actually stops a stage from
burning half an hour. Digestion of a full transcript by a reasoning model legitimately takes two to
three minutes per call, so both default well above that. On failure the CLI's stderr is captured
with `--debug-to-stderr` and filtered to non-DEBUG lines — without it a failed turn reports only
"exit code 1" with an empty stderr.

### Compose stage: generated spine, model-authored scenes

The compose stage (`backend/pipeline/composer.py`) is five steps, and the split exists for a
specific safety reason — **a model is never allowed to own the timeline**:

| Module | Role |
| --- | --- |
| `storyboard.py` | Cuts the script into timed scenes against the audio (ffmpeg silence map, else word count). Pure arithmetic, no model. |
| `visual_plan.py` | Art director. JSON in / JSON out, so any provider works. Falls back to direction derived from the narration. |
| `scene_kit.py` | Deterministic renderer for nine archetypes. Produces the drafts *and* the fallback. |
| `director.py` | Claude Agent SDK crews rewrite scene files. `SCENES_PER_AGENT=6`, `MAX_CONCURRENT_AGENTS=3`. |
| `assembler.py` | Generates `index.html` (the spine that mounts scenes on the audio timeline) and runs `hyperframes lint`. |

Invariants worth preserving when editing any of these:

- The spine is generated, never authored by a model — it owns the audio offset, the mount times,
  and the HyperFrames runtime contract.
- Every agent-authored file passes `director.validate_scene_html` before it ships; failures revert
  to the deterministic scene-kit draft. A dead crew costs plain scenes, never a blank video.
- The gate bans nondeterminism in captured frames (`Math.random`, `Date.now`, `performance.now`,
  `repeat: -1`), network assets, and direct media playback calls.
- Same-track clips may not overlap, so `assembler.py` gives each concern its own track and
  alternates adjacent scenes between two scene tracks.
- After assembly, `hyperframes lint` (blocking contract errors) and `hyperframes inspect` (layout
  errors) run; inspect findings are mapped back to the scenes that caused them and handed to a
  repair agent, and anything still failing is reverted.

Cost dials live in Admin → System → Video Direction: `DIRECTOR_ENABLED`, `DIRECTOR_MAX_SCENES`,
`INSPECT_ENABLED`. `video_template` (`podcast` / `kinetic` / `swiss` / `minimal` / `shanshui`)
selects the theme.

### Subprocess discipline

TTS, HyperFrames render, ffmpeg, yt-dlp and OpenCLI all run through
`backend/pipeline/process_logging.py`. Two rules encoded there, both learned from real hangs:

- **Progress lines are sampled, not logged** (`PROGRESS_LOG_INTERVAL`). Logging every tqdm redraw
  outran the log reader, filled the 64 KiB stdout pipe, and wedged the asyncio event loop.
- **Stall timeouts, not just elapsed timeouts, are the real watchdog.** These stages stream progress
  continuously, so silence means wedged while a long elapsed time may be perfectly healthy
  (`TTS_STALL_TIMEOUT`, `RENDER_STALL_TIMEOUT`; `TTS_TIMEOUT` is only a coarse backstop).

Relatedly, `backend/logging_setup.py` deliberately avoids `logging.basicConfig` — log writes must
never block the event loop that is streaming subprocess output.

### Web automation (OpenCLI)

`backend/pipeline/opencli.py` executes **only** the repository wrapper `scripts/opencli.sh`, with
`tools/opencli/node_modules/.bin` and the project venv prepended to `PATH`. It never invokes a
global `opencli` and never writes Claude skills outside `.claude/skills`. Used by
`thumbnail.py` (ChatGPT image generation) and `web_footage.py` (Gemini clip-interval selection,
yt-dlp download). Both require Chrome running with signed-in sessions and the Browser Bridge
extension, so both stages are written to fail soft — a web outage is recorded in the stage manifest
and the pipeline continues rather than discarding a valid script or forcing a TTS re-run.

Footage rights are tracked, not assumed: the Wikimedia branch accepts only explicit PD/CC0/CC BY/
CC BY-SA metadata, and every YouTube entry is marked `review_required` with a publication blocker
in `footage/manifest.json` and the UI until a human verifies reuse terms. See
[docs/OPENCLI_MEDIA_LOOP.md](docs/OPENCLI_MEDIA_LOOP.md).

### Frontend

React 19 + Vite + react-router + TanStack Query in `frontend/src/`. `api.ts` is the single API
client; `components/SettingsPage.tsx` composes the Admin panels (System / Providers / Prompts /
Skills), which are generated from the backend registries rather than hand-written forms — so a new
setting or prompt needs no frontend change. Theming is CSS variables in `index.css` with a
light/dark toggle in `App.tsx` (see [DESIGN-Light.md](DESIGN-Light.md), [DESIGN-Dark.md](DESIGN-Dark.md)).

## Notes

- Python 3.11 (`~/.pyenv/versions/3.11.9`), Node via `nvm`.
- After a fresh clone: `git submodule update --init --recursive` (vendored Agent SDK + demos).
- Adding a TTS model is a data change in `config._build_tts_models`, not code in `tts.py`; the
  entry carries the whole invocation contract (env script, project dir, inference script, speaker
  flag, voice aliases).
- `openspec/` holds spec-driven change proposals; the `opsx:*` / `openspec-*` skills drive it.
- Not this project's source, don't edit as if it were: `claude-agent-sdk-python/` and
  `claude-agent-sdk-demos/` (git submodules), `tools/opencli/` (pinned npm runtime), and
  `hyperframe/` (the HyperFrames project scaffolded by `setup.sh`, whose CLI the render invokes —
  it has its own `CLAUDE.md`/`AGENTS.md`).
- Gitignored working state: `outputs/`, `uploads/`, `logs/`, `data/` (settings + admin state),
  `*.db`, `.env`, `backend/prompts/.defaults/`.
