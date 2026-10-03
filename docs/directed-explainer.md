# Directed explainer production

The `directed-explainer-v1` compositor adapts the six-stage workflow documented in
XChineseTrans's `output/satellite-beam-steering/article-method-v3.md` and the
`video-director` skill: director brief, purposeful shots, shared timing, motion
and sound, inspect/revise, then verify the delivered file. HyperFrames remains
the renderer. The satellite example's subject, six shots, 3D implementation and
36-second duration are not requirements for another video.

## Runtime and scope

Morning Desk and regular tasks that call `compose_video` use this contract.
The separate Explainer Studio FFmpeg pipeline is unchanged. Generation calls
continue through the configured project provider and its existing routing;
neither Codex nor the three-hour maintenance automation generates the film's
content in place of that provider. The existing TTS, media, source review,
publication policy and final semantic/A/V gates remain in force.

The task's narration-derived timing is authoritative. This project already uses
seconds at centisecond precision, so the upgrade retains that convention and
exports a frame projection rather than introducing a second editable clock.

## What changes on screen

Every content scene, including scenes without acquired media, receives a shot
plan. Each shot records its purpose, subject, action, continuity anchor and
evidence type. The model selects existing grounded copy and measured assets;
the application validates durations, complete asset use, readability and exact
narration excerpts. Provider failure uses an explicitly marked recovery plan.

In addition to focus, cards and split layouts, editorial shots can use traced
flow relationships and paired comparisons. These require an exact narration
excerpt with a supported relationship. This lexical check is conservative; the
final rendered-frame reviewer still determines semantic correctness. Captured
footage and generated illustration cannot masquerade as a new causal diagram.

The renderer consumes the same shot-local reveal, emphasis and settle cues as
preview/review. It keeps copied claims intact, removes duplicated headlines,
and holds after the explanatory action. No perpetual decorative orbit is used
as a substitute for explanation. Sound remains the existing narration-led
paced mix; no separate synthesized effect track is introduced.

## Production artifacts

| File | Meaning |
| --- | --- |
| `director_brief.json` | Audience, takeaway, visual constraints, sound and delivery intent |
| `media_shots.json` | Contract v3, model/recovery provenance, validated shot sequence and cue timing |
| `production_timeline.json` | Ordered half-open frame ownership projected from actual scene/shot times |
| `production_preflight.json` | Fail-closed HyperFrames layout inspection at representative shot/action times |
| `production_preview.json` | Hash-bound real PNG snapshots, dimensions and capture checks |
| `snapshots/` | Current early review images; prior images retained in `snapshots-history/` |
| `delivery_timeline.next.json` | Actual decoded frame count and every presentation timestamp, bound to candidate hash |
| `av_sync_report.json` | Final promoted report including production method, preflight and delivery timing evidence |

Snapshots are captured with HyperFrames' automatic description disabled, so the
CLI cannot silently invoke an unrelated Gemini credential. Producing an image
or passing DOM layout checks does not establish semantic visual quality. The
existing final review retains every planned shot midpoint and adds bounded
editorial action/settled samples. Its judgment still blocks promotion when it
finds a mismatch. Sample review does not constitute manual review of every
frame, and audio measurements do not constitute listening.

Delivery checks decode the encoded file, verify resolution/audio/rate/duration,
then compare all video PTS values against the expected frame grid. An average
FPS value alone is insufficient. Previous good exports and failed candidates
retain their existing protection and recovery behavior.

## Rollout and recurring checks

After tests and local render checks, restart the service and recover the same
task through the appropriate existing API. Preserve validated research, speech
and media. The updated shot contract invalidates old shot checkpoints without
invalidating their approved source inputs. Automatic publishing stays off.

The existing `frontiertechsn` automation remains every three hours and uses
`gpt-6-astra` with `ultra` reasoning. It checks the new artifacts and final
candidate hash as part of the original conjunctive acceptance gate. It should
record real layout/motion/semantic defects, repairs and measured timings in the
ignored watch state. A passing fixture or a queued recovery is not acceptance
of the full production episode; end-to-end savings require comparable completed
runs. Healthy progress is not a reason to repeatedly restart the task.
