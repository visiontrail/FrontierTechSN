## ADDED Requirements

### Requirement: Independent research-led workspace
The system SHALL provide an independent sidebar destination storing user research, language, duration, voice, visual style, scenes and job history without creating a news edition.

#### Scenario: Reopen a saved project
- **WHEN** a user saves and reopens an explainer
- **THEN** its research, style and scene edits persist and are independent of later default changes.

### Requirement: Editable long-form writing
The system SHALL turn supplied research into chapter-structured narration and concrete corresponding scene prompts, with bounded structural recovery and user editing before a direct render action.

#### Scenario: Draft a script
- **WHEN** a user requests script generation
- **THEN** the configured writing provider creates narration grounded in the research, without news research, footage acquisition or editorial approval stages.

### Requirement: Configurable illustrated media
The system SHALL support saved style defaults and project overrides, generate stills through ChatGPT OpenCLI, and allow optional Gemini image-to-video for selected scenes.

#### Scenario: Animate selected scenes
- **WHEN** Gemini motion is selected
- **THEN** selected scene keyframes are passed to Gemini and any failed generation is visible and retryable.

### Requirement: Narration-bound composition
The system SHALL use each scene's measured narration duration to establish continuous frame-aligned video coverage and export a traceable timeline and playable MP4.

#### Scenario: Speech lasts longer than motion
- **WHEN** a generated motion clip ends before its narration
- **THEN** its final frame holds until the scene boundary without looping or speeding speech.

### Requirement: Safe resumption and edit concurrency
The system SHALL checkpoint successful assets, invalidate changed inputs, reject stale or in-flight edits, and expose cancellation, interrupted state and failure details.

#### Scenario: Retry a failed render
- **WHEN** a user retries after a later scene fails
- **THEN** valid earlier media is reused and only missing or changed work is generated.

#### Scenario: Stale editor
- **WHEN** an editor saves an obsolete project revision
- **THEN** the API rejects the update without overwriting newer data.
