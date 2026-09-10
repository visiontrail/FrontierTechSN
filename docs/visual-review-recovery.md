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

Direct publisher media prepared through the existing local-source preview path
can also survive recovery. Its source identity, narration binding, file checksum,
actual-frame Gemini approval, and pending rights review must all remain intact.
A link-only verdict does not qualify a publisher clip for reuse.

The end-to-end recovery also exposed a separate OpenCLI extraction failure in
the final video's second batch. Gemini returned one answer, but the adapter's
transcript fallback included the entire English page: navigation, the user
prompt's example JSON, and the assistant's JSON. Rejecting those two objects was
correct. Current Gemini turns are now read from `user-query` and `model-response`
boundaries, with full `.query-text-line` prompt text and the assistant's Markdown
body. Collapsed prompt summaries and nested speaker labels no longer destabilize
turn ownership. The transcript fallback rejects page chrome even when escaping
or whitespace prevents an exact prompt match. No parser rule selects one of
multiple complete answers.

Apply the checked-in OpenCLI patches with `node tools/opencli/patch-opencli.mjs`
after installation, then run `npm --prefix tools/opencli run test:adapter`.
The next CLI subprocess loads the patched adapter; a running backend does not
need a restart for this JavaScript-only repair.

Do not attach Computer Use's browser debugger or DevTools to an OpenCLI-owned
provider tab while the pipeline is running. A read-only DOM inspection can still
retain debugger ownership after the inspection finishes. OpenCLI may reuse that
tab for its next ephemeral session, causing navigation timeouts and
`Another debugger is already attached to the tab`. Inspect backend and daemon
logs during generation, and use a separate application tab for playback only
after provider work finishes. If a diagnostic debugger is left attached, wait
for the affected command to stop, then disconnect that diagnostic debugger
(the diagnostic extension's browser debugging infobar has a Cancel control).
Confirm the intended browser remains connected and daemon pending commands and
session leases are clear before retrying. Do not cancel another active worker's
debugger or restart a shared browser to clear an unrelated session.

ChatGPT's vision fallback can briefly expose a stable-looking partial answer
before the complete answer reaches the page. If an ask receipt identifies one
conversation but contains zero complete review objects, the reviewer now reads
that same conversation with `detail --wait true --stable 12` before submitting
another prompt. Recovery requires exactly one user/assistant pair, an identical
review prompt, and an idle, stable answer. Its actual verdict is retained even
when negative. Multiple complete objects remain ambiguous and never enter this
recovery path. The detail response is saved beside the original ask receipt.

Regression checks:

```sh
.venv/bin/python -m pytest backend/tests/test_review_response.py backend/tests/test_multimodal_review.py backend/tests/test_web_footage.py backend/tests/test_footage_recovery.py -q
```

### macOS foreground browser availability

Explicit `--window foreground` calls on the macOS bridge runtime check the active
console session before reserving a provider request and again before launching
the browser command. A locked screen produces `BROWSER_SCREEN_LOCKED` with an
unlock instruction instead of spending repeated upload attempts on an unusable
desktop. During provider pacing and browser execution, a call-owned `caffeinate`
assertion prevents idle display/system sleep; completion, failure or cancellation
releases it. It is also tied to the backend PID. This does not change system
preferences, unlock the Mac, or override a manual lock. Headless and non-macOS
runtimes do not use this desktop guard.

If the backend stops during final review, retry can resume the completed render
checkpoint even when the review report is still pending. The render input
fingerprint, candidate SHA-256, full decode and duration must still match. A
pending report is never treated as approval: the real multimodal review and all
final quality gates run before promotion. Missing or changed evidence requires
the ordinary render path.
