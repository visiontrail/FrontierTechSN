# Diagnosing public-footage visual review failures

There are three distinct response contracts:

| Stage | Required review envelope |
| --- | --- |
| Single downloaded footage preview | `image_received`, `suitable` |
| Batch of downloaded footage previews | `results`, with one verdict per candidate ID |
| Final rendered video | `image_received`, `reviews`, with one score per scene ID |

OpenCLI normally wraps model text as `[{"response":"<JSON text>"}]`. The shared
extractor in `backend/pipeline/review_response.py` handles that wrapper, direct
objects, and fenced JSON using the stage's own contract. It rejects multiple
complete envelopes rather than selecting a favorable verdict. Stage-specific
validators still enforce image receipt, IDs, interval selection, confidence,
visible evidence, and the final video score rubric.

The September 10 failure of task `20260909-134336-68335a` exposed a regression:
footage previews reused the final-video parser, which required `reviews` only
when the reply was a string. Direct-object test fixtures bypassed that check.
Both single and batch production wrappers therefore failed even with otherwise
valid preview answers. The outer error incorrectly recommended restoring the
browser connection for every failure. The original preview response was not
retained in the task logs. After unlocking the Mac, its authenticated Gemini
conversation was recovered: the image was received, `suitable` was `false`,
confidence was `0.68`, and `selected_window` was `null`. Gemini rejected the
conference interview as talking-head filler. This was a valid rejection that
should have advanced discovery to another candidate, rather than failing the
task as a browser outage.

Preview requests now make at most three attempts for transport failures or
unusable response envelopes, retaining the same sampled media and prompt and
respecting normal OpenCLI pacing. Explicit suitability rejections and scores
below the 0.65 confidence floor are not retried for a better verdict. In a
partially usable batch, valid candidates remain independent: missing or
duplicate candidate rows stay unavailable, and valid negative verdicts remain
rejections. Missing/unreadable images are unavailable reviews, not evidence
that the footage itself is unsuitable.

For each invocation, inspect
`outputs/<task>/footage/evidence/{previews,batches}/<hash>/review-attempts/<id>/`:

- `request.json` records the prompt, expected contract and contact-sheet SHA-256.
- `attempt-XX.json` records stdout, stderr, exit code when available, and the
  error category (`transport` or `response_contract`).
- A new invocation creates a new directory; retries do not erase older proof.

The manifest retains verified clips, downloaded previews, pending source URLs,
and the failure category. Reconnect/unlock the automation browser when the
transport error calls for it. A response-contract failure calls for examining
the retained response; reconnecting alone does not repair a schema mismatch.

After loading repaired code, retry the existing task's footage acquisition via
`POST /api/tasks/<task_id>/footage/acquire` with `{}` to reuse its saved plan and
eligible artifacts. This operation only retries the scout and restores the
previous stable task status; it does not complete video generation. A task that
failed before audio generation can then resume through its saved-script TTS
endpoint. Verify the final quality report, promoted artifact, full video decode,
HTTP Range response, browser playback, and terminal task status before reporting
the video as recovered.

If the saved plan contains unsuitable discovery terms, the same endpoint accepts
`{"queries": ["story-specific product demonstration", "..."]}`. The retry request
has no legacy six-query ceiling: it can carry a complete eight- or sixteen-shot
plan just like automatic planning. Supplied terms still pass through normal
script grounding, source discovery, and visual suitability checks.

Regression checks:

```sh
.venv/bin/python -m pytest backend/tests/test_review_response.py backend/tests/test_multimodal_review.py backend/tests/test_web_footage.py backend/tests/test_footage_recovery.py -q
```
