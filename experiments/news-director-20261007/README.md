# ByteFront director experiment / 2026-10-07

A standalone six-stage video-director experiment using the latest task's existing footage and TTS. This does not change the production pipeline or original task. See `brief.md` for editorial intent; actual assets and media remain under ignored `outputs/director-lab-20261007-v1/`.

## Reproduce from the frozen snapshot

Run from the repository root, on macOS with FFmpeg/ffprobe, Python 3.11+ and Pillow 12.3.0. This experiment uses the installed Arial and Georgia font files under `/System/Library/Fonts/Supplemental`; font files are not redistributed. On another platform, set `FONT_FILES` in `render.py` to available equivalents and review layout again.

```sh
.venv/bin/python experiments/news-director-20261007/build_plan.py --out outputs/director-lab-20261007-v1
.venv/bin/python experiments/news-director-20261007/render.py --out outputs/director-lab-20261007-v1
.venv/bin/python experiments/news-director-20261007/render.py --out outputs/director-lab-20261007-v1 --render --workers 2
```

First render without `--render` to inspect every shot. Use `--only shot-08,shot-18` for a local revision. Cached shots are invalidated by the edit, generated graphics, input manifest, encoder and renderer source. The full export concatenates the same shot files and muxes the unchanged original program mix, re-encoding audio to AAC only. After rendering an individual revised shot, use `--mux-only` to rebuild the delivery without re-rendering unrelated shots. Explicit per-shot durations and a 90 kHz video time base prevent container rounding from accumulating across cuts. It pads the silent tail to the final frame boundary, with no narration speed change.

The V1 plan is intentionally specific to this edition. `freeze_inputs.py` can create a new immutable asset snapshot, but another edition requires a newly authored `EDITS` list; do not apply this fixed news copy to an unrelated task.

```sh
.venv/bin/python experiments/news-director-20261007/freeze_inputs.py --source outputs/20261007-070828-b0349f --out outputs/a-new-experiment
```

The script refuses to overwrite an accepted snapshot, checks every hash before and after copying, and compares storyboard and audio-mix durations. Do not call the source task's regenerate APIs.

## Review

Copy `review.html` into the experiment output directory. Keep this loopback server running:

```sh
.venv/bin/python experiments/news-director-20261007/serve.py --directory outputs/director-lab-20261007-v1 --port 8127
```

Then open `http://127.0.0.1:8127/review.html`. Playback, chapter selection, frame stepping and timestamped review notes all use `timeline.json`. Notes stay in local browser storage until the user exports them. The server supports media byte ranges, defaults to loopback, and does not write feedback to the repository.

For access from another device on the same LAN, start it with an explicit bind address:

```sh
.venv/bin/python experiments/news-director-20261007/serve.py --directory outputs/director-lab-20261007-v1 --host 0.0.0.0 --port 8127
```

Open `http://<this-computer-LAN-IP>:8127/review.html` on that device. All video, timeline and image URLs are relative, so they use the same LAN address. Feedback remains local to each browser and address; export notes to transfer them between devices.

## Delivery checks

```sh
.venv/bin/python /Users/leoguo/.codex/skills/video-director/scripts/check_delivery.py outputs/director-lab-20261007-v1/bytefront-director-v1.mp4 --timeline outputs/director-lab-20261007-v1/timeline.json --out outputs/director-lab-20261007-v1/evidence/delivery
```

The helper is supplied by the installed `video-director` skill. Its report proves decoding and timeline consistency. The companion review report separately records visual sampling, source-media checks, audio reuse, browser playback and limitations. Neither existing task fact-check flags nor a successful encode prove that the production quality improved; that requires review of the actual experiment.
