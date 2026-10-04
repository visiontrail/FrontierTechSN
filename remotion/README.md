# Real 3D explainer clips

This package renders silent H.264 clips with **Remotion → React Three Fiber → Three.js WebGL**. It follows XChineseTrans's `satellite-beam-steering` stack, updated to Remotion **4.0.532**, React Three Fiber **9.8.1**, and Three.js **0.186.1**, with exact versions pinned. HyperFrames remains the final assembler and owns source-bound text, captions, narration, bookends, and real media. Nothing in this package invokes a language model, writes generated JavaScript, or fetches scene assets.

The model supplies a bounded scene description. The renderer supplies actual meshes: chip substrate/die/heatsink/pins, server rack/trays/fans, satellite bus/solar cells/dish, jointed robot, particle network, or a generic primitive. Perspective camera, lights, shadows, layered exploded views, directional tubes, and moving particles are real Three.js geometry. Actors are illustrative models, not accurate representations of a specific vendor's hardware.

## Install and render

Requirements: Node >=22.18, FFmpeg and FFprobe on PATH. From this directory, run `npm ci`, `npm run typecheck`, and `npm test`.

From the repository root:

```sh
node remotion/scripts/render.mjs \
  --spec remotion/examples/compute-flow.json \
  --output outputs/three-d-example/clip.mp4 \
  --stills 0,45,89 \
  --verify-determinism
```

Remotion manages its local Chrome Headless Shell installation in this package's ignored `node_modules/.remotion` directory. The first render may download it; `npm run browser` prepares it beforehand. `--browser /absolute/path/to/chrome-headless-shell` or `REMOTION_BROWSER_EXECUTABLE` can select an existing headless executable. This is render infrastructure; it never opens or drives the user's browser profile. ANGLE is enabled explicitly. Concurrency defaults to 2 and is bounded to 1–4 with `--concurrency`.

`npm run benchmark` renders the 3-second 720p example to ignored `outputs/three-d-benchmark/` and verifies out-of-order PNG determinism. The command accepts an alternative output file as its first positional argument.

## Input contract

The authoritative validator and TypeScript types are in `src/spec.ts`. Unknown fields fail validation. The root fields are:

| Field | Contract |
| --- | --- |
| `contract` | `three-d-explainer-v1` |
| `id` | Letter followed by up to 47 letters, digits, `_` or `-` |
| `width`, `height` | Even integer 320–3840 |
| `fps` | Integer 24–60 |
| `duration_frames` | Integer 2–3600, total duration at most 60 seconds |
| `background`, `accent` | Six-digit hex colors |
| `representation` | `illustration` |
| `camera` | `position`, `target`; optional `end_position`, `end_target`, `fov` |
| `actors` | 1–12 semantic actors |
| `links` | Optional 0–24 directed actor connections |
| `copy`, `narration_excerpt`, `semantic_intent` | Optional provenance metadata, never painted into this clip |

Camera position coordinates are in [-30,30], target coordinates in [-12,12], and FOV is 20–75 degrees (default 40). Its entire interpolated path must stay at least 2 units from the target. Start/end camera coordinates should frame all objects; the renderer does not silently replace the model's camera or crop its geometry. Use a close framing when the final assembler puts the clip in a smaller visual pane.

Each actor has `id`, `kind` (`chip`, `rack`, `satellite`, `robot`, `data`, `object`), and `position`. Optional fields are:

| Field | Meaning |
| --- | --- |
| `end_position` | Ending position; all coordinates in [-12,12] |
| `rotation`, `end_rotation` | XYZ Euler angles in radians, each [-4π,4π] |
| `scale` | XYZ size multipliers in [0.1,4], default [1,1,1] |
| `color` | Six-digit hex color, defaults to scene accent |
| `primitive` | Only for `object`: box (default), sphere, cylinder, torus |
| `explode`, `end_explode` | Layer separation [0,1], only chip/rack/satellite |
| `motion_start`, `motion_end` | Normalized frame positions [0,1], start < end; defaults 0.12 and 0.72 |
| `action` | reveal, translate, rotate, explode, hold; omitted keeps gentle entrance |

Transform fields determine the actual motion. `hold` disables actor transform interpolation; other actions allow supplied transforms. Explicit `reveal` hides an actor through `motion_start`, then scales it from zero to full size by `motion_end`; links stay hidden until both endpoints are visible. In a directly rendered spec, an omitted action retains the gentle 86%-to-full-size entrance with recognizable first-frame geometry. The production planner normalizes omitted actions to explicit `reveal`, so its primary subject must use explicit `hold` or another non-reveal action to remain visible at onset; reserve explicit reveals for later subjects. The existing nonblank first-frame and semantic review gates still apply. Robot articulation follows the same actor progress. Camera and data/link particles stop at the largest actor `motion_end`, including hold actors. Frame normalization uses `frame / (duration_frames - 1)`. The pipeline stores separate 3D review cues instead of truncating the 3D motion to the title's short reveal animation.

Links contain `from` and `to` actor IDs, optional `color`, and optional integer `particles` in [0,12] (default 5). Tubes and arrowheads encode direction; particle motion is illustrative, not throughput data. Metadata limits: up to 12 copy strings of 500 characters; narration excerpt up to 10,000 characters; semantic intent up to 2,000 characters. The Python planner is responsible for source grounding and shot selection.

## Delivery and verification

The CLI creates the MP4 only after validating the candidate's dimensions, frame count, cadence, silence, full FFmpeg decode, and requested nonblank WebGL stills. It exits nonzero on failure. The sidecar is `${output}.render.json`, with contract `three-d-render-v1` and `passed: true` only after all checks pass. It includes input/output SHA-256, pinned engine versions, render duration, exact frame count, and still metadata.

Stills have stable names: `<output basename>.stills/frame-000045.png`. The default frames are first, middle, last; `--stills` supplies exact indices. `--verify-determinism` re-renders a previously captured frame after the later frame and requires identical PNG hashes. This validates deterministic rendering for the supplied scene on the current machine; it is not a guarantee of cross-GPU byte identity or semantic correctness. Actual scene meaning and final composite legibility still require review.

Animation uses `useCurrentFrame()` and pure functions in `src/state.ts`. No animation uses wall-clock time, random numbers, accumulated frame deltas, external textures, physics simulation, or `useFrame`. This permits parallel and out-of-order rendering. The underlying R3F dependency constructs a deprecated Three.js Clock internally and can log that library warning; this package never uses it for scene state.

Implementation references: [Remotion parameterized rendering](https://www.remotion.dev/docs/parameterized-rendering), [Remotion ThreeCanvas](https://www.remotion.dev/docs/three-canvas), [Remotion rendering API](https://www.remotion.dev/docs/renderer/render-media), and [official 3D animation guidance](https://github.com/remotion-dev/skills/blob/main/skills/remotion-best-practices/remotion-markup/3d.md).
