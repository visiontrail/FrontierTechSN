# Continuous background music

`mix_narration_and_music` builds `audio/continuous_music.wav` before either the
program pacing envelope or sidechain compressor. Both modes measure the rendered
bed's LUFS and keep the existing music targets and narration protection.

The old implementation repeated the entire decoded file with `aloop`. A musical
outro followed by a fresh introduction remains obvious even when their samples
meet at zero. The replacement compares interior passages using log spectra and
energy envelopes, excludes very quiet file edges, and matches up to two seconds
of recurring material. This aligns observed transient patterns without assuming
that a generated prompt's BPM, meter, or “seamless” claim is accurate. It does not
infer or certify musical phrase structure.

The selected tail overlaps the matching head with a correlation-adjusted
constant-power crossfade. A single global headroom adjustment avoids clipping
without pumping at every repeat. The resulting cycle joins contiguous source
samples at its boundary. Rendering writes whole cycles plus an exact final
partial cycle; only the program opening/ending receives an edge fade. Short
programs use a single pass. NumPy is a required dependency. Invalid/silent inputs
fail instead of silently reverting to hard repeats.

`continuous_music.json` and `music_mix_report.json.music_loop` record the source
length, selected interval, overlap, period, every transition time, and headroom.
The recurrence score is a ranking metric, not a perceptual acceptance threshold.
Review new material by listening around multiple transitions, particularly
harmonic changes and percussion. Automatic point selection cannot guarantee
that an arbitrary song has a musically seamless interior loop.

## Verification

Run `python -m pytest backend/tests/test_music.py backend/tests/test_music_loop.py
backend/tests/test_daily_news.py`. Coverage includes both ducking modes, silent
file edges, multiple cycles, exact partial-cycle duration, sample continuity,
short sources/programs, and invalid audio.

For a real task, copy its `program_pacing_report.json` to a separate output's
`audio/` folder and call `mix_narration_and_music` with the original paced
narration/music and the original loudness settings. Remux the resulting WAV with
`-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac`; preserve the approved original video.
Compare video stream hashes, fully decode the final AAC/MP4, inspect peak and A/V
duration, and listen to isolated music and narration together at early/middle/late
transitions. Keep source music and original video available for comparison.

## Sources

- [Ableton: clip-edge fades](https://help.ableton.com/hc/en-us/articles/209069969-Create-Fades-on-Clip-Edges-to-avoid-clicks)
  explains why edge fades prevent sample discontinuity clicks.
- [Ableton: audio clips, tempo and warping](https://www.ableton.com/en/manual/audio-clips-tempo-and-warping/)
  distinguishes transient timing, loop crossings, and fade behavior.
- [FFmpeg audio crossfade](https://ffmpeg.org/ffmpeg-filters.html#acrossfade)
  documents overlap and fade curves. This implementation constructs reusable
  cycles in PCM rather than growing an FFmpeg filter graph with program length.
