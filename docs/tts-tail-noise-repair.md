# Chatterbox post-speech tail cleanup

## Observed defect (2026-10-07)

Task `20261006-065449-764041` contained residual sound after the final spoken
word in Chatterbox parts 001 and 010. The service's downloaded WAV hashes matched
the retained source WAVs exactly. This establishes that the artifact already
existed in provider output; it does not identify a particular neural decoder
operation as the cause.

| Part | Final word ends in source | Safe fade in source | Program cleanup interval |
| --- | --- | --- | --- |
| 001 | 9.08 s | 9.20–9.22 s | 11.20–12.28 s |
| 010 | 4.62 s | 4.74–4.76 s | 78.46–79.84 s |

The existing trailing-speech check finds extra recognized words. These tails
contained no recognized extra words, decoded correctly, and passed duration and
internal-silence checks. Concatenation preserved them; mixing retained them in
the final video.

## Delivery behavior

`tts_tail_cleanup.clean_verified_tail` requires a hash-bound accepted ASR
observation, at least 95% exact source coverage, equal source/transcript word
counts, exact opening/closing anchors, and no omissions, extra words or repeats.
After a 120 ms guard beyond the final word, it requires a separate 120 ms quiet
interval (10 ms windows, RMS at most -42 dBFS, peak at most -30 dBFS). Only then
may a 20 ms fade within that quiet interval silence subsequent residual sound.
This is a conservative tail treatment, not a general noise classifier.

Weak evidence leaves the source unchanged. Invalid or concurrently modified
waveforms cannot be substituted into the delivery join. Every sample before
the fade and the total sample count remain exact; speech is never retimed.

`chatterbox_tail.prepare_delivery_parts` creates separate delivery WAVs and
sidecars. Provider WAVs, request checkpoints, job IDs, and original acoustic
observations remain intact. The delivery sidecar records both hashes and the
prefix-preservation proof. Derived continuity measurements are refreshed.
Cache identity includes the cleanup version only at the outer narration level;
the accepted provider request identity is not changed or resubmitted.

## Repaired final artifact

The original final video `video-733a5ccbe5d761f6.mp4` is retained. The repaired
final is `video-c87ed2c1ac21492a.mp4`. Its full SHA-256 is
`c87ed2c1ac21492a16939867fe8fbf564677feb5d331b601f0507dbdc27f3349`.

Evidence lives in the ignored task directory
`outputs/20261006-065449-764041/recovery/audio-noise-20261007/`:

- `repair-evidence.json`: original/derived hashes, exact retained PCM, unchanged
  music, and video packet/timing equality for all 9,453 packets.
- `corrected-av-sync-report.json`: fresh full decode and audio checks, with
  explicitly inherited original visual review and timing evidence. The visual
  model was not rerun on the new MP4.
- `promotion-receipt.json`: conditional switch of the completed task's final
  video pointer; original synthesis artifacts and automatic publishing settings
  remain intact.
- `endpoint-verification.json` and `playback-verification.json`: matching HTTP
  206 bytes and native Ego Lite task-page playback.
- `opening-before.wav`, `opening-after.wav`, `part10-before.wav`,
  `part10-after.wav`: short comparison extracts from the actual MP4s.

The corrected master mix is byte-identical outside the two cleanup windows
(allowing 20 ms around mixer edges); the music WAV is byte-identical. Video
packets, packet timestamps, subtitles and total container duration
(315.121029 s) are unchanged. The original final audio's 2,048-sample encoder
delay was measured and retained; four untouched windows have zero relative
lag between original and repaired MP4 audio.

The running service was not restarted during the active successor's TTS work.
The completed task serves the repaired final immediately; automatic cleanup
requires the service to load this code at its next safe restart. The original
task's Audio link intentionally remains its immutable provider-source narration.

## Verification

```sh
.venv/bin/python -m pytest backend/tests/test_tts_tail_cleanup.py backend/tests/test_chatterbox_tail.py backend/tests/test_chatterbox.py -q
```

Coverage includes source immutability, full sample-count retention, exact speech
prefix preservation, nontrivial spelling evidence, weak/missing evidence,
concurrent modification, output conflicts, and derivative-cache provenance.
