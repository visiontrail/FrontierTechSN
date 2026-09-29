# Chatterbox production video acceptance

`scripts/validate_chatterbox.py` creates a separate 52-word English sample with
three physical lines: opening, technology explanation, and closing. The sample
makes no time-sensitive news claims. It uses the real `generate_tts`, program
music pacing, `compose_video`, HyperFrames renderer, multimodal review, and final
hash-versioned video promotion. It does not create a database task, change saved
defaults, purchase services, or publish anything. The existing configured AI
providers still incur their ordinary inference usage.

The fixed test uses the current Swiss template, landscape 4K at 30 fps,
the selected model's default voice (`space-en-f1` for Chatterbox),
the existing `morning-briefing` music track, and the hash-verified
`morning-brief` moving bookend preset and logo. Burned captions are enabled by
default; `--no-captions` matches the latest Daily News setting while retaining
the normal exported SRT. Public-footage acquisition, generated collage clips,
and generated news images are disabled for this short integration test. Normal
scene direction and animation remain active. No previous task's storyboard,
composition, transcript, review, or AI conversation checkpoint is copied.

## Prepare without generation

Run from the repository root with the project virtual environment:

```sh
.venv/bin/python scripts/validate_chatterbox.py prepare
```

The command verifies local music/bookend availability and the configured 4K30
and English-review settings, then writes `script.txt` and
`validation_request.json` under a new child of `outputs/chatterbox-validation/`.
It does not call TTS, inference providers, or the renderer. Save its returned
`output_dir`; using an existing task directory or symlink is rejected.

## Generate and verify

Once the remote service passes deployment checks and its API key and capability
fingerprint are configured, use the exact directory printed by preparation:

```sh
.venv/bin/python scripts/validate_chatterbox.py run --output-dir /absolute/path/from/prepare
```

For a candidate deployment, keep the application's saved fingerprint blank and
pass `--deployment-fingerprint` with the complete reviewed capability SHA-256.
This overrides only the validator process; it does not enable an unaccepted
candidate for application jobs. Save the fingerprint in Admin only after the
audio and final video checks pass.

The runner reads the live application task list and shared Explainer queue
before TTS and before every compose attempt. It refuses to start a stage while
production work is active or queued. An advisory file lock prevents concurrent
instances of this validator. The normal application worker is not paused or
reconfigured; the operator should also check that no new production run will
start during validation.

The normal quality gate may request another compose cycle. The runner permits
up to three cycles by default, retaining evidence and production checkpoints in
the same owned directory. `--compose-attempts` sets a bounded one-to-five-cycle
budget. It never disables review or marks a failed candidate as a final video.
Re-running the same command preserves validated TTS parts and idempotent remote
jobs. Provider outages remain visible failures.

To generate an optional Pocket/Jane counterpart with the identical source,
prepare a **different directory** and pass `--model pocket-tts-en --voice jane`
to both `prepare` and `run`. Use the same caption and music options on both invocations.
Comparison establishes delivery-contract parity; two synthesis models are not
expected to produce identical voices, samples, duration, or pronunciation.

Chatterbox's `space-en-f1` uses the exact official Space English FLAC, verified
by size and SHA256 in the deployed service. The legacy `jane` ID deliberately
clones a Pocket/Jane reference and is not the official Space voice. Always use
a new directory when changing voice or deployment fingerprint; a historical
Jane sample cannot establish acceptance of the official reference. The CLI
still accepts `--voice jane` for an explicit legacy comparison.

## Evidence and audio preservation

After normal final promotion, the runner writes `validation_delivery.json` and
`validation_report.json`. The latter contains the final MP4 and SHA-256, full
FFprobe result, 24 kHz mono PCM16 source properties, natural 1.0x declaration,
current ASR-advisory results, production quality report paths, and SRT path.
The physical WAV/container, source-hash, complete accepted-part, pause, program
pacing, music-mix, multimodal-review, and full-video-decode checks stay active.

Two separate waveform checks distinguish narration preservation from mixing:

1. Compare each raw narration interval with its corresponding interval in
   `paced_narration.wav`, before background music. Report sample count ratio,
   measured offset, correlation, best-fit amplitude gain, and residual error.
   Pass requires correlation at least 0.995 and matching segment frame counts
   within two samples. Resampling boundaries exclude a documented 20 ms margin.
2. Compare the exact `program_mix.wav` supplied to the renderer with decoded
   final AAC. Permit a constant offset of at most 50 ms for codec delay, require
   correlation at least 0.97, and allow at most 90 ms of duration difference for
   AAC packet/final-frame rounding. Alignment never stretches time or warps
   speech. Music is already in both signals, so it cannot conceal a different
   narration waveform through an inappropriate speech-only comparison.

These additional thresholds were checked read-only against the retained
Pocket/Jane final `20260928-235341-4aa80a`: sampled source-to-paced correlations
were 0.99985–0.99998, and a ten-second rendered-AAC comparison was 0.99987 with
42.67 ms codec delay. A synthetic three-percent time change failed the waveform
comparison. These historical measurements calibrate the audit; they do not
certify a new Chatterbox output.

Re-audit a completed sample without generating media:

```sh
.venv/bin/python scripts/validate_chatterbox.py audit --output-dir /absolute/path/from/prepare
```

The report explicitly leaves browser playback pending. After the runner passes,
verify HTTP Range serving of the exact final MP4 under `/outputs/`, then use Ego
Lite to check loaded dimensions/duration, playback time progression, audio, and
absence of browser errors. The sample is an isolated acceptance artifact, not a
published episode or an application task whose database status is `complete`.
