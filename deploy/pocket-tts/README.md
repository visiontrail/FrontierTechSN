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
docker compose -f compose.nr-test.yaml build
docker compose -f compose.nr-test.yaml up -d
docker compose -f compose.nr-test.yaml ps
curl --fail http://127.0.0.1:8090/health
```

The deployment Dockerfile uses a digest-pinned multi-architecture uv base and
the upstream locked runtime dependency graph, but omits its development
dependency group and unused CUDA packages. It pins the official ARM64 CPU-only
PyTorch 2.5.1 wheel by SHA256; the model code remains the unmodified upstream
checkout. The build verifies that CUDA is absent before the image can complete.
Compose disables Hugging Face Xet because it can collapse to a single very slow
transfer on this network; the normal HTTPS cache is still persistent and
resumable.

Do not add `--quantize` until the unquantized and quantized builds have been
compared on the same nr-test script for lexical coverage, boundary continuity,
voice quality, real-time factor, and memory use. The selected production mode
must be recorded in the end-to-end evidence.
