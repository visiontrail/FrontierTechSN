# ByteFront Espresso outro assets

The video pipeline exposes three selectable 6-second outro treatments. Their
moving backgrounds were generated in Google Gemini; the ByteFront brand,
English closing message, and Like / Comment / Share controls are editable
HyperFrames layers authored by the downstream video-director agent.

Runtime media is deliberately kept out of Git. Install the accepted source
files in `data/outros/` with these exact names:

- `01-data-extraction-gemini.mp4`
- `02-morning-brief-gemini.mp4`
- `03-signal-shot-gemini.mp4`
- `bytefront-logo-transparent.png`

The catalog and SHA-256 integrity values live in
`backend/pipeline/outros.py`. The default selection is
`2. 晨间简报｜温暖、编辑感` (`morning-brief`). The generated composition must
not contain the removed Chinese closing phrase; closing copy remains editable
in the HyperFrames scene source.
