# Pocket TTS on nr-test

This deployment keeps the upstream checkout and deployment state separate under
`/home/guoliang`:

- `/home/guoliang/pocket-tts`: clean upstream checkout pinned to
  `adde0654090b1d54f6ee416ed10fd1b108069f13`.
- `/home/guoliang/pocket-tts-deploy`: this Compose file and operational state.
- TCP `8090`: Pocket TTS HTTP service; Orpheus remains on `8088`.

The application submits one physical script line (opening, story, or closing)
per request. Pocket TTS then applies its native sentence/token splitting inside
that request. This removes Orpheus-style arbitrary 12-word joins while retaining
per-story acoustic verification and retry recovery.

Deploy from `/home/guoliang/pocket-tts-deploy`:

```sh
export POCKET_TTS_SOURCE_DIR=/home/guoliang/pocket-tts
export POCKET_TTS_DOCKERFILE=/home/guoliang/pocket-tts-deploy/Dockerfile
export POCKET_TTS_BIND_HOST=10.60.11.3
export POCKET_TTS_PORT=8090
export POCKET_TTS_CPU_THREADS=16
docker compose -f compose.nr-test.yaml build
docker compose -f compose.nr-test.yaml up -d
docker compose -f compose.nr-test.yaml ps
curl --fail http://10.60.11.3:8090/health
```

The deployment Dockerfile uses a digest-pinned official Python 3.10 slim base
and the exact non-development versions exported from the upstream lock. It does
not retain a compiler toolchain, uv, or CUDA packages in the runtime image. The
official ARM64 CPU-only PyTorch 2.5.1 wheel is pinned by SHA256; the model code
remains the unmodified upstream checkout. The build verifies that Pocket imports,
Torch is 2.5.1, and CUDA is absent before the image can complete. Compose disables
Hugging Face Xet because it can collapse to a single very slow transfer on this
network; the normal HTTPS cache is still persistent and resumable.

`POCKET_TTS_CPU_THREADS=16` is the measured nr-test default. Repeated same-prompt
tests found normalized audio throughput of approximately 0.35x realtime at 16
threads versus 0.34 at 4, 0.32 at 8/32, and 0.28 at the 64-thread host default.
The service leaves Torch inter-op scheduling unchanged: forcing it from 64 to 1
reduced throughput to approximately 0.30x. Re-benchmark these values if nr-test's
CPU topology or co-tenancy changes.

Do not add `--quantize` until the unquantized and quantized builds have been
compared on the same nr-test script for lexical coverage, boundary continuity,
voice quality, real-time factor, and memory use. The selected production mode
must be recorded in the end-to-end evidence.
