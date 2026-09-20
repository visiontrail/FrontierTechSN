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

- Brand and headlines: Inter/Helvetica Neue, strong but not oversized. In both
  bookends, `Espresso` matches the apparent size and weight of `Bytefront` as
  one unified wordmark.
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

The workflow declares `edition_date` and `edition_weekday` as HyperFrames
composition variables. The opening scene reads them through
`window.__hyperframes.getVariables()` so the edition data is set per render
rather than burned into the reusable video plate. Briefing-label and story-count
chips are intentionally absent.

## Closing source roll

During the spoken closing, keep the brand and thanks on the left and place
`SOURCES & CREDITS` in a separate column on the right, in both orientations.
Use 28px source text with 36px line height, navy on the light presets and white
on the dark preset. The list scrolls upward within a clipped window; it holds
briefly at the beginning and end and completes before the closing fade.

Credit every selected news article and the media used in the finished scenes.
Show publisher, complete headline, publication date when available, and website;
retain full permalinks in `outro_credits.json` and the editable source markup.
Extend the picture and music when necessary for reading, preserving the spoken
closing and caption timestamps. Choose a 10–15 second credit outro based on
the source-list height and spoken closing, capped at 14.9 seconds to leave room
for final-frame/audio rounding. Scroll through the complete list during the
middle 80% of the outro, reserving 10% at each end for reading holds (the final
hold includes the closing fade). Speed follows the actual duration and rendered
list height in both orientations. Reject an overlong spoken closing instead of
cutting off narration. The source roll stays deterministic during director edits
and retries.
