# Chatterbox English initialisms

The pinned multilingual tokenizer lowercases text before encoding, so uppercase
`CNBC` alone does not preserve a spelling cue. See the [pinned tokenizer](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/models/tokenizers/tokenizer.py#L248-L264).

## Provider input policy

`backend/pipeline/tts_pronunciation.py` asks the application's configured AI
model to review the full English script before synthesis. The prompt is
`backend/prompts/tts_pronunciation.txt`; there is no named-initialism allowlist.
Typography identifies candidate occurrences, including uppercase fragments in
mixed-case brands, plural forms and numbered models. The model decides per
occurrence whether to preserve a word reading or supply spoken letter names.
The same spelling may receive different decisions in different contexts.
URLs, email addresses and domains are excluded.

The prompt explains the installed voice's observed articulation behavior:
explicit letter-name words generally work better than separated literal
letters. Policy 3 uses `eigh` for the letter A: the installed voice can read
`ay` as `eye`, dropping A from AI or OpenAI. Validated spell decisions also
normalize the old `ay` alias to `eigh`, including separated letter-name forms.
An `N` can turn phonetic `en` into `and`, so candidates containing N
use hyphens throughout. Conventional repeated-letter readings are allowed.
These are general prompt rules derived from the samples below, not a complete
acoustic guarantee for untested names. The model remains responsible for
deciding which occurrences need spelling.

Every model decision is validated before any TTS job is submitted. A generic
English alphabet checks that a spoken spelling retains exactly the original
letters in order. Missing decisions, invented expansions, omitted letters,
changed candidate text and invalid output fail validation. One bounded repair
attempt includes the validation error; persistent failures stop before
synthesis. This alphabet is a validation vocabulary, not a name-to-reading
dictionary. Text outside approved candidate spans cannot be rewritten.

The transformation applies only to Chatterbox's English synthesis request:

- Original script, canonical chunk files, source hashes, ASR reference,
  manifest source counts and subtitle/timing inputs retain the original text.
- Actual provider input is stored in `audio/synthesis/tts_input_part_NNN.txt`
  and the audio sidecar's `identity.synthesis_text` / `synthesis_request.text`.
  `audio/tts_pronunciation.json` stores the source-bound AI plan, reasons,
  requested/actual model route and hashes; `audio/tts_synthesis.txt` stores the
  complete revised speech text. Raw AI attempts remain available for audit.
- Source chunks are bounded using the expanded character and spoken-word
  lengths. An initialism stays within one request; oversized indivisible input
  is rejected before any synthesis request. Audio duration sanity also uses
  the expanded spoken-word count.
- Pronunciation policy version, prompt hash and alphabet hash participate in
  part, narration and preview cache identities. The accepted plan hash also
  binds each part. Request hashing validates the actual provider text.
- Save the validated plan before submitting audio jobs, then reuse it without
  another AI call when resuming. Changing the configured AI model does not
  rewrite an already accepted plan. A missing/stale plan beside submitted job
  checkpoints fails closed instead of rebinding their idempotency keys.
- Natural 1.0x synthesis, voice reference, service build and physical audio
  checks are retained. General ASR spelling differences remain advisory.
- Chatterbox verifier 3 additionally requires each initialism occurrence selected
  for spelling to match the transcript, including its plural/possessive suffix.
  Only aligned splits with exactly the same letters can be joined (`Open AI`,
  `C N B C`); wrong letters, omissions and repetitions are rejected. Adjacent
  source initialisms are checked separately. Distant repeated initialisms are
  rejected both when inserted and when they replace ordinary source words.
  Source-aligned English letter names in recognition output also retain their
  exact meaning (`you ex` -> UX, `A are` -> AR). Names are consumed only when
  every observed atom reproduces the complete source spelling; `exited` cannot
  become X. Normalized observations preserve the raw transcript and still
  reject missing letters, repetitions and extra speech after the utterance.
  Brand prefixes and model-selected
  word acronyms keep their advisory spelling policy. URLs and email addresses
  remain excluded. This recognition gate is evidence of the decoded pronunciation,
  not a universal guarantee against recognition errors.
- The separate trailing-speech gate rejects a single extra phrase of at least
  one word after an exact complete source utterance. Its version participates
  in the outer cache contract. Existing hash-bound ASR observations are rechecked
  without changing submitted job identities; good audio is reused and rejected
  speech advances through the same bounded seed attempts.
- Single capital letters joined by `&`, such as `R&D`, are spoken as letter,
  "and", letter in provider input, outside protected URLs and addresses. The
  expanded input counts toward chunk limits, and both letters are verified.
  This policy participates in the outer cache contract; source text stays intact.
- A pronunciation rejection retains the WAV, transcript and reports before
  retrying with the next seed within the existing three-attempt limit. Resume
  never replenishes that limit. An unavailable recognizer stops verification
  while preserving the completed service job for resume, without resynthesis.
  Accepted cache entries must contain a passing pronunciation report. Pocket
  and Orpheus keep their existing advisory behavior.
  Earlier pronunciation-only rejections can be rechecked once after the
  observation repair, following their original completed job and idempotency
  key. Synthesis inputs, seeds and attempt budgets do not change; original
  rejection files remain archived. Physical failures are not cleared by this
  observation repair.

## Real AI validation, 2026-10-02

Evidence is under `outputs/chatterbox-ai-pronunciation-20261002/`. Calls used
the configured `yinhe-thinking` model through the existing provider route.
The context fixture produced:

| Source/context | Model-selected speech text |
| --- | --- |
| CNBC | `C-N-B-C` |
| AI | `ay eye` |
| Unfamiliar QZVX protocol | `cue zee vee ex` |
| NASA | `NASA` (keep word reading) |
| The word IT is emphasized | `IT` (keep) |
| The IT department | `eye tee` |
| OpenAI | `Open ay eye` |
| IEEE | `I triple E` |

A copy of completed video `20261001-062032-2ae717`'s script also passed a real
model call: 26 candidate occurrences, 26 accepted edits, 54 bounded synthesis
chunks, and exact canonical-token preservation. See
`script-check/validation.json` and the saved plan. These checks establish AI
planning, preservation and segmentation, not fresh acoustic acceptance of
every new pronunciation. An active production video was left running with its
existing inputs during these checks.

## Earlier acoustic comparison, 2026-10-02

Evidence is under `outputs/chatterbox-initialisms-20261002/` (ignored runtime
artifacts). Tests used the deployed pinned V2 service, `space-en-f1`, seed 42,
standard production generation parameters, 24 kHz mono PCM16 and local Whisper.
The canonical sentence was **CNBC reports on AI chips.**

| Provider input strategy | Observed ASR text |
| --- | --- |
| Original uppercase | `CNB reports on A -chips.` |
| Hyphens | `CNBC reports on iChips.` |
| Spaces | `CNBC reports on iChips. A pair age.` |
| Phonetic names throughout | `C and B, see reports on AI chips.` |
| Periods | `CNB, see reports on iCHIPS.` |

The space sample's seed-42 WAV failed the unchanged internal-pause check
(0.96 seconds versus the 0.80-second limit); its displayed result is the normal
bounded retry at seed 43. The other rows used seed 42. These are acoustic
transcription observations, not a claim of subjective listening review.

These checks exercised the preceding deterministic adapter implementation and
informed the current AI prompt. Canonical-input checks preserved source text.
In `acceptance/`, `CNBC reports on AI chips.` matched the independent transcript
exactly with `C-N-B-C reports on ay eye chips.` as provider input. The second
chunk confirmed BBC/TSMC, but failed API/GPU/GPT with hyphens alone; that failure
is retained. `acceptance-v2/` tests the latter sentence with explicit letter
names. Existing finished videos and remote service files are untouched.

Final result: both canonical sentences matched exactly (5/5 and 10/10 source
words, no token differences). The first sample is 4.16 seconds; the final
BBC/TSMC/API/GPU/GPT sample is 4.76 seconds. Both pass the current physical
validator and retain their recorded audio hashes. The first sentence's current
AI-produced synthesis text matches the previously accepted input exactly.
`summary.json` binds these two passing samples without overwriting the failed
alternatives.

Regression coverage includes contextual decisions, unknown names, exact
letter validation, unchanged factual text, URLs, expanded chunk/duration
budgets, plan tampering, prompt invalidation and resume without another model
call. This provider-input change requires the local backend to load the new
code; it requires no remote TTS redeployment or settings migration.

### Explicit punctuation repair for an exhausted part

When all three durable jobs for one part fail initialism pronunciation because
of a fused spoken boundary, a reviewed provider-only comma can be requested in
`audio/tts_input_part_NNN.punctuation-repair.json`. This is an opt-in repair, not
an automatic extension of the unchanged input's retry limit. Its version-1
object contains `source_sha256`, `synthesis_sha256`, `insert_after` (a character
index at an existing inter-word space), a nonempty `reason`, and
`original_request_sha256` (the three original checkpoint file hashes in attempt
order). No words, canonical script, voice, speed or verification gates change.

The adapter verifies that each original request belongs to the unchanged input
and is a confirmed pronunciation rejection. It refuses to replace accepted
audio. A single immutable activation receipt binds the repair; changing the
manifest after activation fails closed. Corrected requests use the separate
`tts_input_part_NNN.punctuation-repair.request-N.json` namespace with the same
three-attempt bound. Verification evidence is likewise separate. Transport or
verification uncertainty resumes the same job; repeated runs cannot reset the
repair budget. Existing audio for other parts keeps its original identity.

Version 3 supports a reviewed inter-word comma after all three original jobs
were rejected specifically for nonempty unrequested trailing speech. It rejects
mixed failure classes, malformed rejection evidence, pending jobs and accepted
audio. This changes only provider prosody; every new sample must still pass the
original speech and independent waveform checks. It is not an automatic retry
or permission to trim extra words from a rejected sample.

On 2026-10-13, Explainer scene 156 exhausted its original seeds on added tail
words. The last sample repeated “hidden” after the complete sentence; both
Whisper large-v3-turbo and an independent Whisper-small observation detected it
at a separate acoustic interval. A version-3 repair inserts a comma after
“method” in the provider input, retaining the complete canonical sentence and
all original rejected requests. This is a bounded recovery of that input, not
a demonstrated general cure for Chatterbox repetition.

Acoustic-reference upgrades also recheck persisted initialism pronunciation
rejections, as these can originate in the complete-source phoneme fallback.
Rechecking does not submit a new synthesis request or erase the old rejection.

### Literal repeated-letter hints

The planner can express BBC as `double bee see`, but Chatterbox speaks the word
`double` literally. Its exact-initialism verifier correctly rejects that audio.
The provider adapter now expands repetition hints within validated initialism
edits (`bee bee see`, `eye ee ee ee`), while keeping W's `double you`, ordinary
words, the canonical script, and the frozen pronunciation plan intact.

Fresh requests use the expanded input. Existing accepted audio and pending jobs
retain their original identity. After exactly three confirmed pronunciation
rejections, the same part may use one deterministic `letter-repetition` request
namespace with the usual three-attempt limit and unchanged acceptance gates.
The applied marker binds both identities and all original request file hashes;
a changed marker or history fails closed. This does not reset or delete old
requests and does not accept `doublebc` as BBC.

### HG-DAgger recovery (2026-10-12)

Morning Desk `20261012-042251-af6108` exhausted part 16's three original
requests. Its provider text ended in `aitch gee-DAgger`; observed transcripts
included `itchgdagger`, `hgdagger`, and `each` in place of HG. The second
transcript alone may reflect ASR token fusion rather than wrong speech; none
passed the existing pronunciation contract, so the original audio stays rejected.

The existing version-2 punctuation repair is applied only to this part, changing
provider input to `aitch gee, DAgger`. The source remains `HG-DAgger`. The marker
binds source/synthesis hashes and all three original request hashes; accepted
parts 1–15, the pronunciation plan, voice, seeds, and normal integrity checks
remain unchanged. The corrected input has the existing three-candidate limit.
This is a targeted input repair, not evidence that general pronunciation or
ASR reliability is fixed. Recovery requires a newly passing part and continued
progress in the same task; final delivery requires the normal final checks.
