## Why

The existing product turns current news into multi-story briefings. The user needs a separate long-form illustrated explainer workflow, beginning with supplied research and using narration-led line-character imagery inspired by PsychToonsHQ.

## What Changes

- Add a dedicated sidebar workspace for research, editable long scripts and scene prompts, configurable persistent visual styles, generation progress, and video playback/download.
- Reuse configured writing models, TTS, ChatGPT image generation via OpenCLI, and optional Gemini image-to-video generation.
- Render each illustration against the measured narration for that scene; retain successful media on interruption or retry.
- Keep topic research, public footage acquisition, news editorial reviews, automatic publishing, and channel analytics outside this feature.
- Document representative video observations and the existing capability inventory.

## Capabilities

### New Capabilities
- `illustrated-explainers`: Research-to-script-to-video studio with persistent styles, scene editing, generated artwork, narration alignment, and resumable renders.

### Modified Capabilities
None.

## Impact

New isolated backend store, API, writing and rendering modules; new React workspace and navigation entry; application lifecycle registration. Existing news pipeline behavior and provider configuration remain compatible. Uses installed FFmpeg, Pillow, TTS and OpenCLI dependencies.
