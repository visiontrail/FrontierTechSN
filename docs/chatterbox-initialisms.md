# Chatterbox English initialisms

The pinned multilingual tokenizer lowercases text before encoding, so uppercase
`CNBC` alone does not preserve a spelling cue. See the [pinned tokenizer](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/models/tokenizers/tokenizer.py#L248-L264).

## Provider input policy

`backend/pipeline/chatterbox.py` maintains an explicit English initialism map.
Most listed names use letter separators (`CNBC` → `C-N-B-C`). Observed
exceptions use explicit letter names: `AI` → `ay eye`, `API` → `ay pee eye`,
`GPU` → `gee pee you`, and `GPT` → `gee pee tee`. The official English voice
collapsed `A-I`, `A I` and `A. I.` into a sound transcribed as `iChips` in the
test sentence. A subsequent list of API, GPU and GPT also lost letters with
hyphens alone, so separator-only handling was not accepted for those names.
The [pinned punctuation normalization](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/mtl_tts.py#L65-L102)
preserves hyphens between letters; it rewrites a space-hyphen-space sequence.

Only explicitly listed uppercase names are changed. This avoids changing word
acronyms such as NASA, NATO, ARM, DRAM and RAM or ordinary uppercase emphasis.
Possessives and hyphenated compounds are supported. Mixed-case names, plural
suffixes such as GPUs, attached model numbers such as GPT4, URLs and email
addresses stay unchanged. Add new exceptions only with a representative live
voice sample; no spelling hint guarantees every stochastic utterance.

The transformation applies only to Chatterbox's English synthesis request:

- Original script, canonical chunk files, source hashes, ASR reference,
  manifest source counts and subtitle/timing inputs retain the original text.
- Actual provider input is stored in `audio/synthesis/tts_input_part_NNN.txt`
  and the audio sidecar's `identity.synthesis_text` / `synthesis_request.text`.
- Source chunks are bounded using the expanded character and spoken-word
  lengths. An initialism stays within one request; oversized indivisible input
  is rejected before any synthesis request. Audio duration sanity also uses
  the expanded spoken-word count.
- Pronunciation version and the complete map participate in part, narration
  and preview cache identities. An old pronunciation cannot reuse an old WAV
  under the new contract. Request hashing validates the actual provider text;
  uncertain requests still retain their idempotency key when resumed.
- Natural 1.0x synthesis, voice reference, service build, physical audio checks
  and the existing ASR-advisory policy are retained.

## Live comparison, 2026-10-02

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

Canonical-input checks exercise the actual adapter and preserve source text.
In `acceptance/`, `CNBC reports on AI chips.` matched the independent transcript
exactly with `C-N-B-C reports on ay eye chips.` as provider input. The second
chunk confirmed BBC/TSMC, but failed API/GPU/GPT with hyphens alone; that failure
is retained. `acceptance-v2/` tests the latter sentence with explicit letter-name
exceptions. Existing finished videos and remote service files are untouched.

Final result: both canonical sentences matched exactly (5/5 and 10/10 source
words, no token differences). The first sample is 4.16 seconds; the final
BBC/TSMC/API/GPU/GPT sample is 4.76 seconds. Both pass the current physical
validator, retain their recorded audio hashes and have the same synthesis text
under the final pronunciation map. `summary.json` binds these two passing
samples without overwriting the failed alternatives.

Regression checks passed: 379 tests and 53 subtests across TTS, resume,
Chatterbox service, previews and validation; after the last map adjustments,
106 tests and 22 subtests passed in the affected Chatterbox/voice suites.
No remote TTS redeployment, settings migration or full-video regeneration was
needed for this provider-input repair.
