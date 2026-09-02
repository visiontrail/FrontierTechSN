# ByteFront Espresso Motion Design

The branded opening and closing use the **Morning Brief** visual language: a
warm ivory paper field, layered editorial cut-paper forms, ByteFront cobalt
circuits, and a restrained copper sunrise/coffee-ring accent. The design sits
between HyperFrames' Soft Signal and Swiss Pulse families: tactile and calm,
but aligned to a precise information grid.

## Palette

- Paper: `#F4F0E7`
- ByteFront navy: `#10243F`
- Signal blue: `#2F64AE`
- Copper accent: `#C98758`
- Quiet ink: `rgba(16, 36, 63, 0.68)`
- Translucent paper panel: `rgba(246, 248, 251, 0.82)`

## Typography

- Brand and headlines: Inter/Helvetica Neue, strong but not oversized.
- Edition date: Georgia, high-contrast editorial serif.
- Metadata: Courier New, uppercase, tabular, wide tracking.
- Video-safe minimums: 18 px labels, 28 px support, 72 px hero type at
  1920x1080.

## Motion

- The Gemini plate owns environmental motion; HyperFrames never replaces it
  with a synthetic pan or gradient.
- Overlay motion is deterministic and seekable: short directional reveals,
  staggered metadata, a copper rule draw, then a settled hold.
- Entrances complete by roughly 2.2 seconds. The six-second opening holds long
  enough to read and dissolves cleanly into the first narrated scene.

## Dynamic opening contract

The workflow declares `edition_date`, `edition_weekday`, `edition_label`, and
`edition_story_count` as HyperFrames composition variables. The opening scene
reads them through `window.__hyperframes.getVariables()` so the edition data is
set per render rather than burned into the reusable video plate.
