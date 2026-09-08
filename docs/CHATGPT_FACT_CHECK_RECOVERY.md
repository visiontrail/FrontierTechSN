# ChatGPT fact-check response recovery

The review parser must reject partial verdicts, even when their prefix appears to
approve some stories. It requires the complete ordered story set, valid error
codes, known claim IDs, and the web-search acknowledgement. A negative verdict
continues through correction/manual review; transport recovery never approves it.

A completed ChatGPT answer can remain partially rendered in the original tab.
On 2026-09-08 the latest task saved `W5B@5.` repeatedly, while opening the exact
conversation independently returned `W5B@5.1;6P`. The submitted prompt matched the
saved group-3 prompt. Waiting longer on that frozen DOM did not help.

Recovery proceeds in this order:

0. Check visible ChatGPT dialogs/alerts for access limiting (including "Too many
   requests" and "temporarily limited access"). Never read the answer behind that
   modal. Surface `CHATGPT_RATE_LIMITED` with the target URL and persist a shared
   provider cooldown: 5 minutes initially, doubling on recurring limits within
   an hour, capped at 30 minutes. During cooldown, generation, model selection,
   reads, and refreshes all wait before starting their command timeout. The wait
   is cancellable and the breaker survives service restarts. Direct wrapper
   invocations also honor this saved cooldown. Recovery keeps the original URL
   and request marker; it does not immediately submit another audit. After the
   cooldown, `detail --cooldown true` reloads the same target only if the old
   blocking dialog is still visible. Without this step, a stale dialog could
   trigger endless cooldowns even after server access has recovered. A fresh
   limit after reloading opens the longer cooldown; an active stream without a
   limit dialog is preserved.
1. Read the exact conversation returned by `ask`, without navigating away from
   an active stream. Match the current `REVIEW_REQUEST_ID` to its assistant turn.
2. Reject responses still marked as generating. Parse only complete verdicts.
3. After two identical invalid owned responses explicitly marked non-generating,
   request one `chatgpt detail --refresh true`. The adapter checks generation
   again and uses `window.location.reload()` to refresh the selected conversation.
   Navigating to the same URL is insufficient: the bridge can reuse the page.
4. Wait for the reloaded conversation to settle, verify ownership again, and
   apply the same strict parser. Continue bounded recovery if it is still invalid;
   never fill in a missing verdict or claim ID. Each submission gets at most one
   refresh. Missing generation metadata or an unowned turn cannot trigger it.
5. Retain conversation URLs, raw invalid responses, and refresh events in provider
   diagnostics. If recovery is exhausted, retain the failed review artifacts.

The implementation is in `backend/daily_news/review.py` and the reproducible
OpenCLI 1.8.7 patch in `tools/opencli/patch-opencli.mjs`. Test with:

```sh
source .venv/bin/activate
PYTHONPATH=. pytest backend/tests/test_daily_news.py backend/tests/test_opencli.py
PYTHONPATH=. pytest backend/tests/test_opencli_access_cooldown.py
npm run test:adapter --prefix tools/opencli
```

Live verification must use `review_daily_script` with the saved task dossier,
script, and configured providers/model range, or the task's `resume-review` route.
A passed parser test alone does not establish that browser recovery works.

The screenshot supplied during verification showed a real access-limit modal
over the exact `[5, 6]` conversation. An otherwise complete token is insufficient
while that modal is present. Generation spacing alone (normally three minutes)
does not handle this provider-wide conversation-access limit; the access circuit
breaker is separate and applies to recovery commands too. Avoid repeated manual
or automated probes during its cooldown; resume the same saved conversation once
the wait ends.
