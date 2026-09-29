# Chatterbox Multilingual V2 on nr-test

This separate service uses the multilingual V2 model from the requested
[ResembleAI Space](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS).
It binds `10.60.11.3:8092` by default. Its container, cache, data directory, and
CPU thread setting are independent of Pocket TTS (`8090`) and Orpheus (`8088`).

Pinned inputs:

- Chatterbox source: `5de7a54aa4e5e2baadb0182dde554908b48b85c2`.
- Space alignment/EOS algorithm: `c612a94210c0af895d6b649850d15b7ccca36594`.
- Model repository snapshot: `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`.
- Model variant: `t3_mtl23ls_v2.safetensors`, selected explicitly with `t3_model="v2"`.
- Perth watermark source: `ff1c8ac55a976971245cdd53c18d6131ca00d993`.
- Default conditioning `conds.pt` SHA256:
  `6552d70568833628ba019c6b03459e77fe71ca197d5c560cef9411bee9d87f4e`.
- Official Space English reference `en_f1.flac`: GCS generation
  `1755807478904629`, 83,323 bytes, SHA256
  `e81950378b53827aed08d164ebd332a7b8c5805b8a97b4f79ab057071e359c72`.
- PKUSEG `spacy_ontonotes.zip`: release `v0.0.26`, 34,567,143 bytes, SHA256
  `b216e7f92de7ae285aeab8feba2faa8ea8216e5995ff6fb3d391cc8356db1bfe`.

The source revision contains CPU loading fixes missing from older copies of the
Space. The wrapper loads a revision-pinned snapshot using `from_local`, including
the Cangjie mapping. A narrow tokenizer download override points at that local
mapping because upstream otherwise fetches it from mutable `main`.

The newer GitHub source removed the requested Space's `AlignmentStreamAnalyzer`.
`apply_space_alignment.py` checks the original T3 file SHA256 and restores that
fixed Space algorithm immediately after CFG combination and before sampling
processors. `space_alignment.py` retains its EOS suppression, long-tail,
return-to-earlier-text, and repeated-token thresholds. Transformers 5.2 requires
explicit eager attention to provide those weights; the complete Llama backbone
uses eager so its causal masks remain consistent. Missing or stale attention
weights fail generation. Hooks are removed in `finally` after each request.
For texts with at most five tokens, the empty earlier-text region is treated as
having no repetition instead of raising an empty-reduction error.

The build runs a tiny real Llama CPU prefill/KV-cache/hook smoke check without
downloading weights. The patch manifest is verified before model loading, and
the Space revision, eager setting, compatibility-source hashes, and manifest
hash participate in the service fingerprint. Existing accepted audio must be
revalidated under the new fingerprint; unchanged model weights do not establish
unchanged synthesis behavior. See the fixed [Space analyzer source](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS/blob/c612a94210c0af895d6b649850d15b7ccca36594/src/chatterbox/models/t3/inference/alignment_stream_analyzer.py).

## Deploy

Use an isolated directory, for example `/home/guoliang/chatterbox-tts-deploy`.
Copy this directory there, inspect existing ports/containers first, and create a
private `.env` with `CHATTERBOX_API_KEY` set to a strong generated secret. Keep
the secret outside Git and avoid printing it in commands or logs. The API refuses
to start when it is missing.

```sh
mkdir -p state cache voices
chmod 700 state cache voices
python3 prefetch_reference.py --output ./voices/en_f1.flac
docker compose -f compose.nr-test.yaml build
docker compose -f compose.nr-test.yaml up -d
docker compose -f compose.nr-test.yaml ps
curl --fail http://10.60.11.3:8092/health
```

The Python 3.11 Debian Bookworm container supplies the glibc needed by ARM64
Torch wheels even on nr-test's older host OS. The image build verifies Torch
2.6.0 and the absence of CUDA. Direct framework dependencies are pinned in
`requirements.txt`; the complete resolved environment is captured at
`/app/runtime-requirements.txt` and its hash participates in the API fingerprint.
The base image tag and remaining transitive dependencies are not a complete
reproducible-build lock. Preserve the built image digest and the freeze file with
deployment acceptance evidence.

The default is eight Torch/BLAS threads. Measure real-time factor and host load
before changing it. Model loading happens on the first job, so `/health`
only proves that the HTTP worker is ready. A successful authenticated synthesis
and downloaded audio are required to verify model readiness.

Optional build arguments select faster package mirrors without modifying the
default official endpoints:

```sh
docker compose -f compose.nr-test.yaml build --build-arg DEBIAN_MIRROR=mirrors.ustc.edu.cn
# PIP_INDEX_URL is also a build argument; its default is https://pypi.org/simple.
```

The Debian argument is a hostname, and only the base image's existing Debian
source files are rewritten. HTTPS, APT signature verification, bounded connection
timeouts, and retry limits remain enabled. The `.dockerignore` excludes host
cache/state/reference files and secrets from the build context.

For a slow target-host connection, `PIP_FIND_LINKS` can point at a reviewed
wheel index prepared for Linux aarch64 and CPython 3.11. Set `PIP_NO_INDEX=1`
only after verifying the complete runtime and isolated-build dependency set.
`PIP_TRUSTED_HOST` is optional and should name only the specific internal host
serving those artifacts. Never use macOS binary wheels on nr-test.
`CHATTERBOX_SOURCE_ARCHIVE` and `PERTH_SOURCE_ARCHIVE` optionally accept mirrored
archives of the exact commits above, with a required `#sha256=...` URL fragment
in the deployment command. Record archive hashes and verify their commit-named
top-level directories before use. Blank arguments retain the pinned Git source
install. These build arguments occur after APT so changing the download path
can reuse completed system-package layers; they do not alter synthesis settings.

Weights must be prefetched before the first request. Prefetch can run during
image build using host Python 3.6 or newer and curl, without installing Torch on
the host:

```sh
python3 -u prefetch_models.py --cache-home ./cache/huggingface --workers 2
python3 prefetch_models.py --cache-home ./cache/huggingface --verify-only
python3 -u prefetch_pkuseg.py --cache-home ./cache/pkuseg
python3 prefetch_pkuseg.py --cache-home ./cache/pkuseg --verify-only
```

The pinned multilingual tokenizer initializes its Chinese segmenter even for
English requests. Its `spacy-pkuseg` wheel does not include that model. The
separate PKUSEG prefetch uses the official release URL and SHA256 recorded in
`spacy_pkuseg/config.py` of version 1.0.1. It validates the ZIP size/hash, rejects
path traversal and symlinks, extracts into a staging directory, verifies every
extracted file against the pinned archive, and then promotes the directory.
Existing valid cache files are reused; unexpected modifications fail without
overwriting them.

The container sets `PKUSEG_HOME=/cache/pkuseg`, covered by the same persistent
`./cache` mount. Before loading Chatterbox, the worker validates the archive and
extracted model and replaces PKUSEG's implicit downloader with a cache-only
validator. Missing, corrupt, or unexpected segmenter models therefore fail
locally instead of making an external request. The archive and loader hashes
participate in the deployment fingerprint. Finish both model prefetch commands
before the first synthesis request.

If an individual connection is throttled, optional multipart transfers retain
the current contiguous `.incomplete` prefix and reuse completed byte fragments:

```sh
python3 -u prefetch_models.py --cache-home ./cache/huggingface --workers 2 --connections-per-file 8 --range-mib 64
```

The default remains one connection per file. Do not run two prefetch processes
against the same cache; the script enforces an exclusive lock. When changing
download mode, stop the old prefetch process and its curl children first, keeping
all partial files. Each ranged response must be HTTP 206 with the exact requested
`Content-Range`, total size, and byte length. Ignored or changed ranges fail
without falling back to a full-file response. Files smaller than a fragment use
the ordinary download path. The assembled full SHA256 must match before its
existing prefix can be replaced or a cache blob promoted. `.ranges` directories
retain completed fragments for reconciliation; remove them only after the final
cache verification succeeds. Redirect headers and signed URLs stay out of logs.

Interrupted ranges retain `.downloading` candidate prefixes. A retry requests
only the missing suffix into a separate tail file. A failed transfer's received
tail is joined only when the final response headers prove HTTP 206, the exact
requested interval and total size, and identity encoding. Unknown, ignored, or
shifted responses remain separate for inspection. Legacy prefixes without saved
headers are explicitly marked as unverified candidates in a JSON sidecar and
are not redownloaded. No candidate fragment is accepted as model content until
the assembled whole-model SHA256 passes; that remains the acceptance gate for
all resumed bytes. Prefixes longer than their requested range fail unchanged.
Range transfers force HTTP/1.1 because the nr-test and local download attempts
both encountered curl exit 92 from HTTP/2 stream framing failures.

The script fetches six files totaling 3,208,951,748 bytes from the official pinned
Hugging Face revision. Downloads resume from `.incomplete` files, and exact size
and SHA256 checks precede promotion into the standard Hugging Face cache. It
preserves unexpected existing files for investigation. The report is
`cache/huggingface/hub/models--ResembleAI--chatterbox/prefetch-report.json`.
After successful verification, set `HF_HUB_OFFLINE=1` in the deployment `.env`.
The worker also explicitly passes `local_files_only=True` to `snapshot_download`,
because the installed Hugging Face Hub 1.33.0 attempted a repository tree query
with only the offline environment setting. Model initialization/import/cache
failures return durable `failed/model_load_failed`, without exception details in
HTTP responses. This deployment failure must not consume acoustic retry seeds.
Private stderr diagnostics retain the failure stage, exception class, and stack
file/function/line locations; exception text, source lines, and locals are omitted.

The default for new requests is `space-en-f1`, the original English reference
selected by the fixed [Space app](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS/blob/c612a94210c0af895d6b649850d15b7ccca36594/app.py).
Prefetch is mandatory: `prefetch_reference.py` downloads the original FLAC from
the fixed GCS object generation, verifies its size and SHA256 before promotion,
and reuses valid existing files. Unexpected existing content is preserved and
rejected. `--verify-only` checks an installed file without network access.
Compose sets `CHATTERBOX_SPACE_EN_REFERENCE=/voices/en_f1.flac`; direct launches
must set the same environment variable to their local verified file. The
service fails startup if this default reference is missing or incorrect, and
checks it again before each job. Generation never downloads a voice reference.

The explicit `default` ID retains the model's bundled `conds.pt` voice. The
explicit `jane` ID retains the Pocket-derived project reference: put its WAV at
`voices/jane.wav` and set `CHATTERBOX_JANE_REFERENCE=/voices/jane.wav` to enable
it. Neither legacy ID is remapped to the official voice. The worker restores
original conditioning before each call, including switches between all three.

Capabilities expose `default_voice`, reference hashes, and the official source
revision, object generation and URL. These identities participate in deployment
and voice fingerprints. An omitted request voice is resolved to its explicit ID
before hashing and persisting the job. Historical jobs/audio retain their
original voice and fingerprint; a reused idempotency key with changed deployment
identity is rejected. After upgrading this isolated service, verify synthesis
before updating the application's expected fingerprint. Use a new acceptance
output directory so historical Jane results remain available for comparison.

## HTTP contract

All `/v1/*` routes require `X-API-Key`. `/health` is public and reports basic
service state and pinned revisions. OpenAPI and interactive docs are disabled.

- `GET /v1/capabilities`: source/model/Perth revisions, V2 variant, deployment
  fingerprint, default voice, defaults, languages, limits, and per-voice reference hashes and
  fingerprints.
- `GET /v1/voices`: `{ "voices": [{ "id", "reference_sha256", "fingerprint" }] }`.
- `POST /v1/jobs`: requires `Idempotency-Key` and returns HTTP 202 with durable
  job metadata. Repeating an identical request with the same key returns the
  existing job; changed text/config/fingerprint returns HTTP 409.
- `GET /v1/jobs/{id}`: status `queued`, `running`, `completed`, or `failed`.
- `GET /v1/jobs/{id}/audio`: complete audio only, with `X-Audio-SHA256` and
  `X-Model-Fingerprint`. Missing/changed audio is HTTP 410.

Example request body, with all inference settings explicitly recorded:

```json
{
  "text": "The full sentence is synthesized without truncation.",
  "language_id": "en",
  "voice": "space-en-f1",
  "seed": 0,
  "exaggeration": 0.5,
  "cfg_weight": 0.5,
  "temperature": 0.8,
  "repetition_penalty": 2.0,
  "min_p": 0.05,
  "top_p": 1.0,
  "speed": 1.0
}
```

The application submits sentence-aware chunks of at most 250 characters. The
server rejects oversized text, unsupported languages, unavailable voices,
unknown fields, and non-natural speed. It preserves the original request text
and saves both its SHA256 and the full generation configuration. Chatterbox's
own internal text normalization still runs in the model.

Output is RIFF WAV, mono, PCM16, 24 kHz with the official Perth watermark. The
wrapper does not change speed, loudness, silence, or edit spoken content. The
project's downstream acoustic verification, retry recovery, pause assembly,
subtitle alignment, and final video checks remain necessary.

The response's `fingerprint` is the deployment fingerprint;
`voice_fingerprint` equals the selected voice's capability fingerprint.
`text_sha256` covers exact source UTF-8 text and `request_sha256` covers the
canonical full request. Completed jobs add `audio_sha256`, `duration_seconds`,
`sample_rate`, `channels`, `sample_width`, and `encoding`.

## Recovery and limits

SQLite WAL records job acceptance before the response. One owner lock prevents
multiple workers using the same data directory, and one model subprocess
serializes conditioning and global RNG. A maximum of 16 active jobs is admitted;
overflow returns HTTP 429 with `Retry-After`. Jobs exceeding 900 seconds terminate
the model process. Its next job lazily reloads the model. The general resource
limit rejects audio longer than 90 seconds, and the pinned V2 model additionally
rejects duration at or above 38.8 seconds as `decoder_duration_limit`, preventing
likely decoder token-limit truncation from being promoted. This bound is exposed
as `max_decoder_seconds` in capabilities. The wrapper also checks that physical
PCM bytes match the declared WAV frame count. It never truncates output to pass
validation. A decoder-limit failure can consume an explicit acoustic recovery
attempt with a new seed and idempotency key; replaying its existing key returns
the same durable failure.

On service restart, interrupted queued/running jobs become durable
`failed/service_restarted` records. Reusing their keys returns that failure.
This deliberately avoids silently rerunning a request with an uncertain result.
The application must explicitly choose a new attempt key for a known failed job;
transport errors should retry the same key. Keep the `state` directory and its
idempotency records together with accepted audio; deleting records loses the
ability to reconcile old submissions.

Fixed seeds and fingerprints make the requested inference configuration
auditable. They do not assert identical floating-point output across different
hardware or rebuilt environments. Acceptance requires generated audio and a
final video from the deployed image, beyond the mocked HTTP tests.

Run wrapper contract tests locally without downloading Torch or the model:

```sh
.venv/bin/python -m pytest backend/tests/test_chatterbox_service.py -q
```
