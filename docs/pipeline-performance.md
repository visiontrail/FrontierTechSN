# Pipeline latency and acceptance

The optimization preserves the live-web claim protocol, narration integrity,
licensed-asset checks, every-scene rendered-frame review, and final promotion gate.
No provider pacing, scoring threshold, word coverage requirement, or story quota
is reduced.

## Implemented changes

- After factual approval and title generation, thumbnail/public-footage preparation
  runs alongside TTS. Browser work stays serial. Both branches are drained before
  composition; verified audio is persisted even when media scouting fails.
- Approved story evidence survives correction cycles. The cache key contains the
  complete single-story review prompt, dated source evidence, numbered claims,
  attribution/live-web rules and reviewer effort policy. The original ChatGPT
  group response is retained and reparsed. Rejected, malformed, non-web, changed
  or policy-incompatible evidence cannot approve a story. The final full-coverage
  cycle still runs the deterministic script contract and verifies every story.
- Automatic picture-editor plans are cached before their AI-chosen image count is
  used to check the existing asset cache. Exact prompt and policy versions govern
  reuse. Provider fallback plans are not frozen. Original image grounding, license,
  file-integrity and completeness checks remain in place.
- `logs/timing.jsonl` records actual start/end events for stages, browser pacing,
  provider cooldowns, responses, audio integrity, ASR and retry waits.
  `timing_report.json` distinguishes elapsed wall time, active interval unions,
  cumulative stage work and pause/recovery gaps. Overlaps are counted once in the
  wall-clock partition. Processing is elapsed work, not a CPU-time claim.
- `/api/runtime` exposes the commit and backend source digest captured at process
  import; task timing evidence records this identity for each root attempt.
- Live acceptance exposed an existing ASR judge incorrectly waiving a source
  word missing from both normal and slow transcripts (`US National Security
  Agency` became `U .S. Security Agency`). A deterministic shared-omission guard
  now rejects this before model adjudication. Verifier versions invalidate old
  audio approvals; existing WAVs must pass acoustic verification again before
  reuse. Number formatting and corroborated spelling differences retain their
  existing adjudication path.
- When both decodes repeatedly omit one complete word, resynthesis can add a
  local pause before that word. This requires the saved rejection and a fresh
  matching omission check, a unique text boundary, and unchanged canonical
  tokens. The new waveform must pass the full acoustic checks; the pause itself
  grants no approval. This addresses an observed repeated loss of `National`.
- Title generation receives only the final narration, preventing rejected or
  unconverted figures in the original brief from re-entering the publication
  title and cover. Source metadata remains in the artifact's audit manifest.
- After a primary SDK route exhausts its configured attempts with a transient
  outage and the configured backup succeeds, calls for the same endpoint, model
  and credential pool try that backup first for five minutes. Backup failure
  still tries the primary with its full retry allowance. Backup successes do
  not extend the window; expiry or primary success restores normal order.
  Explicit single-route probes bypass this ephemeral health observation, and
  content failures or partially committed output cannot create it. Provider
  pacing and downstream quality checks are unchanged.
- Public-footage batches prepare up to two independent source previews at once.
  Duplicate source URLs remain serial to protect their shared cache files.
  Failed downloads retain their own candidate index; successful candidates still
  require the same explicit pixel-based verdict. Cancellation drains every
  preparation job before returning, and browser requests remain serial and paced.

## Measurement protocol

Use a newly created full daily-news task, eight requested news stories, the same
Pocket TTS voice and public-footage/collage/news-image options, with auto-publication
turned off. Start from research, with a new output directory. Do not copy artifacts
from a successful task. Compare duration, actual story/scene/media counts and final
quality reports; fewer stories or missing media do not demonstrate equivalent work.

The stored baseline is `20260913-032431-9ceb10`: 2026-09-13 11:24:31 through
2026-09-14 01:42:26 Singapore time, 14:17:55 wall time. Its five explicitly identified
failure/recovery gaps total 4:03:38, leaving about 10:14:17 including automatic waits
and retries. This is not pure processing time. The baseline delivered 317.35 seconds,
10/10 reviewed scenes and average visual score 83. Some restart gaps were not
fully recorded. Six renders account for only about 25 minutes, so rendering alone
cannot explain or eliminate the long runtime.

Five hours is an engineering target, not a measured claim. Record the new task ID,
loaded runtime identity, final artifact paths, quality status and measured stage
intervals after end-to-end acceptance. Do not equate regression/build success with
video delivery.
