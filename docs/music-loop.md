# Continuous background music

`mix_narration_and_music` builds `audio/continuous_music.wav` before either the
program pacing envelope or sidechain compressor. Both modes measure the rendered
bed's LUFS and keep the existing music targets and narration protection.

## Musical timing before crossfading

Repeating the complete file replays its outro and intro. A long crossfade is also
unsafe for rhythmic music: two percussion patterns or incompatible parts of a
phrase can overlap even when spectra and RMS match.

For newsroom beds in common time, the renderer estimates a 75--150 BPM pulse by
checking onset agreement at beat, bar, and phrase scales. If the grid is strong,
it chooses whole multiples of 16 beats (four bars), finds a compatible quiet
boundary, and phase-aligns within five milliseconds. A ten-millisecond linear
microfade removes an edit click without sustaining two rhythmic passages at
once. A bounded tempo-analysis window does not cap the retained source length:
a three-minute source keeps a long phrase rather than becoming a short loop.

For sources without a strong grid, interior spectral/envelope recurrence remains
a fallback with correlation-adjusted equal-power blending. This fallback is not
certified for arbitrary rhythmic songs or other meters. Weak tempo detection,
incorrect meter assumptions, changing harmonies, and approximate phase matches
still require listening; neither selection scores nor RMS certify naturalness.

A global headroom adjustment avoids clipping without pumping at each repeat.
The cycle joins contiguous source samples at its boundary. Rendering writes
whole cycles plus an exact final partial cycle; only the program opening/ending
receives an edge fade. Short programs use a single pass. NumPy is required.
Invalid/silent inputs fail instead of reverting to hard file repeats.

`continuous_music.json` and `music_mix_report.json.music_loop` record the source
length, selected interval, overlap, period, every transition time, headroom, and
selection mode. `beat_grid_score` is onset agreement; `recurrence_score` applies
only to the non-metered fallback. Neither is a perceptual acceptance threshold.

## Listening and regression requirements

The first implementation was rejected in actual user listening: a two-second
crossfade over a roughly 22-beat selection produced severe repetition. Its
three-excerpt preview also inserted 250 ms of silence at 8.00 seconds, creating a
misleading pause. That preview is failed evidence and must not be reused as proof
of continuity.

Deliver continuous audio of at least 90 seconds spanning multiple real repeats,
with no artificial pauses or concatenated excerpts. For longer loop periods,
provide enough continuous audio to cross at least two boundaries. Listen for
melodic restarts, double percussion attacks, dropped beats, gaps, and abrupt
harmony changes. User rejection overrides automated or model-based evaluation.

Run `python -m pytest backend/tests/test_music.py backend/tests/test_music_loop.py
backend/tests/test_daily_news.py`. Coverage includes both ducking modes, silent
file edges, many cycles, exact duration, sample continuity, short sources/programs,
invalid audio, metered percussion, and retention of long source phrases.

For a real task, copy its `program_pacing_report.json` to a separate output's
`audio/` folder and call `mix_narration_and_music` with the original paced
narration/music and loudness settings. Remux the WAV with `-map 0:v:0 -map 1:a:0
-c:v copy -c:a aac`. Compare video stream hashes, fully decode the AAC/MP4, inspect
peak and A/V duration, and listen with narration at early/middle/late boundaries.
Retain the original video and music for comparison.

## Sources

- [Ableton: clip-edge fades](https://help.ableton.com/hc/en-us/articles/209069969-Create-Fades-on-Clip-Edges-to-avoid-clicks)
- [Ableton: audio clips, tempo and warping](https://www.ableton.com/en/manual/audio-clips-tempo-and-warping/)
- [FFmpeg audio crossfade](https://ffmpeg.org/ffmpeg-filters.html#acrossfade)

These sources explain clicks, transients, and fades; they do not establish that
an automatically selected musical loop sounds natural.
