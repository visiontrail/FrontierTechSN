# Chatterbox common-uppercase vocabulary verification

The corpus in `backend/tests/fixtures/chatterbox_common_vocabulary.json` is
verification data. It does not select production pronunciations or replace
the contextual model with a fixed pronunciation dictionary.

The baseline includes computing, companies, media, institutions, business,
health and everyday abbreviations. Additional cases exercise word acronyms,
ordinary words written in capitals, mixed-case names, plural and possessive
endings, dotted spellings, adjacent initialisms, ampersands and product numbers.
An observed inventory can add every distinct uppercase-containing form from
the project's actual news scripts. The frozen manifest reports the exact cases,
distinct forms, source hashes and alphabet coverage; these counts are inventory
coverage, not evidence that audio has passed.

Prepare or resume the same isolated run:

```sh
.venv/bin/python scripts/validate_chatterbox_vocabulary.py prepare \
  --observed outputs/chatterbox-uppercase-20261006/observed-vocabulary.json
.venv/bin/python scripts/validate_chatterbox_vocabulary.py run \
  --observed outputs/chatterbox-uppercase-20261006/observed-vocabulary.json
```

The CLI uses the installed `space-en-f1` reference, natural 1.0x speech and the
existing pinned Chatterbox provider contract. It checks the production queue
before each batch, freezes the source and model plan, and uses the production
adapter's persisted jobs and bounded retries. A changed corpus or contract
cannot rebind an existing run's submitted jobs. Observation/transport errors
stop the run while retaining its checkpoints. Confirmed bounded integrity
rejections are recorded before other batches continue.

Accepted WAVs retain the production physical and spelling checks. A second
recognizer audits the same hash-bound waveform. The additional corpus guard
checks kept words and whole mixed-case names, preserves compound word parts,
and checks numeric suffix values and multiplicity. It records raw transcripts
alongside every normalization. The documented word readings are source-bound:
SQL can use letter names or [sequel](https://learn.microsoft.com/en-us/sql/odbc/reference/structured-query-language-sql),
and SaaS can use [sass](https://info.microsoft.com/rs/157-GQE-382/images/Enterprise_Cloud_Strategy_2nd_Edition_ebook_mobile.pdf).
Roman `III` in the Phase III case can be represented as three. These
representations do not authorize omitted letters, extra repetitions or changed
product values.

The corpus attaches its independent per-part auditor to the production adapter.
It runs before each new WAV is accepted and when a cached WAV is reused. A
positive rejection of cached speech archives its original WAV and metadata and
marks the matching durable job rejected. Only the remaining original bounded
attempts may synthesize again; resuming cannot replenish that budget. An
unavailable recognizer leaves the waveform and checkpoint intact and stops the
run. Both raw and source-normalized extra-tail checks are required. The final
audit compares metadata with the frozen source and provider input, rather than
accepting metadata as its own expected identity.

Complete corpus auditing records every source part, including rejected parts,
in `audio/tts_audit_coverage.json`. A positive integrity rejection after the
original bounded attempts does not skip the batch's later vocabulary cases.
Observation or transport uncertainty still stops immediately and preserves the
current job. The coverage report is updated per part, cannot claim completion
after a partial run, and binds the complete source and plan hashes. Any rejected
part prevents narration concatenation and passing batch acceptance. Normal
production generation retains its immediate failure behavior.

By default validation requires an idle application queue. `--allow-composing`
also permits a private audio run while identified production tasks are composing
video, after their narration is complete. Every active task must be composing;
its final WAV hash, required source coverage and manifest part count must match.
Unidentified queue owners, queued work and any speech/preparation stage block
new submissions. This check repeats immediately before each new durable job;
following an existing job does not resubmit it. `execution-window.json` and
`synthesis-windows.json` retain the production audio anchors and application
runtime separately from the isolated validation process's code identity. This
option does not permit an application restart during composition.

Each verification attempt writes `observation_binding.json` with the downloaded
waveform hash, durable job ID, source hash and request identity. Reports from a
different waveform move into immutable `observation_history` before verification.
Decision reports are refreshed even when rechecking the same waveform; its raw
transcription remains reusable. A physical failure that precedes pronunciation
verification therefore cannot archive a previous job's passing verdict. Every
rejected WAV retains a hash-bound `rejection.json` alongside the reports actually
available for that attempt.

Spelling observation version 3 accepts source-bound conventional `double` and
`triple` letter names, including ASR fusions such as `double SH` for `double S H`.
It counts the named repeat and retains every remaining glyph. Extra repetitions
and missing glyphs still reject. `double you` and `double U` remain the letter W
and cannot silently satisfy two U glyphs. This observation change does not alter
source text, provider hints, synthesis parameters or durable request identities.

`scripts/chatterbox_phoneme_evidence.py` supplies an additional complete-source
comparison for unconditioned phone observations from the pinned acoustic model.
The WAV hash, model revision and absence of reference-conditioned recognition
must match. It compares all source words and every validated spelling atom with
CMU word references and English letter-name phones. Compound rhotic token
continuations do not count twice; vowel repetitions are never collapsed.
Missing letters, extra named syllables, wrong vowels and an incomplete closing
remain differences. Unsupported phones, words, multi-digit numbers or symbols remain
unverified. Passing this additional comparison does not override production
gates or claim overall audio acceptance.

Observation version 4 can corroborate an ASR spelling mismatch with a complete
source phone sequence after the physical checks have passed. The raw failed
ASR report remains visible, and the accepted report explicitly names the
acoustic confirmation method. Positive ASR extra-tail detection remains a
rejection. Source/word/letter omissions, extra phones and wrong vowels cannot
be cleared by the source comparison. Previously rejected spelling checkpoints
are followed under their original completed job IDs, without new POSTs or a
replenished seed budget.

The optional local runtime defaults to `.run/tts-phoneme-runtime/lib`, or
`CHATTERBOX_PHONEME_RUNTIME_LIB` can select another isolated library directory.
The tested isolated packages are transformers 4.57.6, huggingface-hub 0.36.2,
tokenizers 0.22.2 and safetensors 0.6.2; the invoking Python also needs Torch and
NumPy. The pinned model snapshot must already exist in the local Hugging Face
cache. The observer receives only a WAV path, never source text, loads offline
on two CPU threads, and does not edit the original waveform. A missing runtime
retains the strict ASR behavior. An attempted but unavailable local observation
stops with the completed job preserved. TTS service/model/reference parameters
remain unchanged.

Pronunciation policy 4 includes isolated uppercase fragments adjacent to digits:
`A2A`, `B2B`, `C3I`, `2D` and `4G` are reviewed in their complete script context.
Only the alphabetic source spans can receive letter-name hints; digits, ordinary
source text and protected URLs/domains/email addresses remain untouched. The
candidate-pattern hash joins the frozen policy identity. Existing submitted v3
jobs retain their original candidate set, plan, provider input and request keys.
Frozen vocabulary runs also retain their full original case order and counts.
A new repair input uses a separate directory and the current policy; it cannot
overwrite an already submitted v3 request or replenish that request's attempts.
Single digits have English number references in the acoustic comparison, and an
isolated uppercase model letter adjacent to a digit uses its letter-name phones.
The A in `A2A` cannot be accepted as the weak indefinite article.

Policy 5 uses `backend/prompts/tts_pronunciation_v5.txt`. The original
`tts_pronunciation.txt` remains the immutable policy 3/4 resource, so submitted
jobs and ongoing frozen corpus runs can retain their exact prompt hashes. In
new spelling plans, only N stays a standalone glyph; all other letters retain
explicit names, including A as `eigh`. Consecutive repeated letters prefer the
accurate conventional double/triple reading when natural.

`--require-phones` additionally requires a supported complete-source phone
sequence on every accepted part. If that observation is unavailable, the
completed job is preserved; confirmed differences use only its remaining
bounded attempts. Acoustic evidence version 2 recognizes a word-internal
nasal/fricative stop transition (N SH to N CH) within dictionary words, as in
the observed pronunciation of `mentions`. Nasal/fricative stop epenthesis is
documented in [MIT's pronunciation teaching material](https://ocw.mit.edu/courses/21g-223-listening-speaking-and-pronunciation-fall-2004/d54f3bba263684cab866eb7b25fe1533_MIT21G_223F04_epenthesis.pdf).
This word transition rule never applies to independently named letters and
does not discard vowels, additional letter names, wrong H affricates or tails.
Evidence version 3 also retains the syllabic-L variant of `double` in ordinary
words and the letter name W. This is an unstressed word-final schwa plus L,
not the stressed vowel in the named letter L. The variant is exemplified by
[Berkeley's English pronunciation workbook](https://linguistics.berkeley.edu/~kjohnson/English_Phonetics/cons_syllabic.html).
Missing A, changed N/M, added named syllables and word omissions remain failures.
Older phone-sequence rejections can be rechecked against the same completed
waveform without new synthesis submissions.

Phone evidence version 4 also permits the voiced homorganic stop in a dictionary
word's final `/nz/` coda (`mentions` can contain `n-d-z`). This is a restricted
application of the nasal/fricative stop-epenthesis rule described in
[MIT's pronunciation material](https://ocw.mit.edu/courses/21g-223-listening-speaking-and-pronunciation-fall-2004/d54f3bba263684cab866eb7b25fe1533_MIT21G_223F04_epenthesis.pdf).
It does not apply across named letters or at an internal syllable boundary.
Missing vowels, changed nasals, extra spoken letter names and trailing speech
remain failures. This reference alternative does not change the waveform or
authorize a word-reading hint in a production pronunciation plan.

Policy 6 uses `tts_pronunciation_v6.txt` and retains the policy 5 file unchanged.
After actual source-bound trials, M and R inside longer spelled candidates can
use standalone glyphs to avoid weak/clitic readings of `em` and `are`. Distinct
single-letter hints can use `M.` or `R.` without changing following digits.
All candidate decisions remain contextual and letter/order/count validated;
there is no production list of entity names to replace.

Each batch writes its mode audit, production request checkpoints, WAV metadata,
primary and independent transcripts, detailed target audits and outcome.
`summary.json` becomes passing only after every frozen batch has completed and
every required gate has passed. Disagreements remain failed or unverified until
investigated; a green production manifest alone cannot clear them. This CLI
does not create task records, change defaults, restart services or publish.
