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
npm run test:adapter --prefix tools/opencli
```

Live verification must use `review_daily_script` with the saved task dossier,
script, and configured providers/model range, or the task's `resume-review` route.
A passed parser test alone does not establish that browser recovery works.
