# ByteFront Espresso YouTube banner

## Deliverables

- Upload artwork: `outputs/youtube-branding/bytefront-espresso-banner-v1-2560x1440.png`
- Central crop preview: `outputs/youtube-branding/bytefront-espresso-banner-v1-mobile-preview.png`

Generated media stays in ignored `outputs/`. The full banner is the upload file; the crop is for inspection only.

## Design and copy

The banner uses an ink navy background, cobalt circuitry and the existing B monogram, with a warm copper arc on the right. The reference is `data/outros/bytefront-logo-transparent.png`. The wordmark and supporting copy share the center of the image so they survive narrow device crops.

Exact copy:

```text
ByteFront Espresso
Your morning brief on tech.
AI, chips, robotics and the business behind them.
```

## Export and verification

- Final PNG: 2560 × 1440 pixels, 4,109,998 bytes.
- Preview: centered 1544 × 423 pixel crop.
- Visually checked the complete wordmark, monogram, spelling, punctuation and supporting copy in the central crop.
- Built-in image generation produced 1672 × 941 pixel artwork, including after the requested resolution pass. The final file was resampled with macOS `sips` to the upload dimensions; it is not a native 2560 pixel generation.
- The banner has not been uploaded to YouTube.

YouTube's [channel branding guidance](https://support.google.com/youtube/answer/10456525?co=GENIE.Platform%3DAndroid&hl=en), checked September 20, 2026, recommends 2560 × 1440 pixels, requires at least 2048 × 1152 pixels, and limits the image to 6 MB. Its 1235 × 338 safe area at the minimum canvas scales to approximately 1544 × 423 at the recommended canvas.

## Generation method

Created with the built-in `image_gen` tool, using the existing logo as a brand reference. No API or CLI generation fallback was used. Export resizing and the inspection crop used `sips`.

Initial prompt:

```text
Use case: ads-marketing.
Create ONE finished YouTube channel banner for ByteFront Espresso, an English technology and business morning briefing. This must be the actual flat banner artwork, not a presentation, template, browser screenshot, wall mockup or multiple variants.
Output canvas: exactly 2560 by 1440 pixels, 16:9 landscape. Crucial YouTube crop constraint: ALL lettering and the B brand symbol must fit entirely inside the centered rectangle x=570 to 1990, y=565 to 875. Leave generous background above and below; do not enlarge the text to fill the full canvas. Desktop and phone both crop to the narrow central strip.
Input image 1 is a BRAND REFERENCE ONLY: use the recognizable angular blue B monogram with its small circuit traces as a clean supporting brand mark. Remove the noisy/ragged edge appearance from the reference and render the mark crisply. Do not copy the reference white background, speckles, or its separate wordmark.
Art direction: sophisticated, restrained editorial technology branding with the warmth of morning espresso. Deep ink navy background (#101C2B), subtle fine paper grain, rich cobalt details, a restrained amber/copper sunrise glow toward the far right. Sparse fine circuit lines and softly layered geometric paper shapes near the far outer edges, beautiful negative space through the center. A soft large warm arc at the right margin and a few precise cobalt circuit traces at the left margin; they should extend naturally across the background, not form a frame. Extremely subtle detail. Refined visual depth, premium print-like material quality, not a generic sci-fi wallpaper.
Layout: a compact, beautifully balanced horizontal brand lockup centered in the safe zone. The blue B monogram is to the left of the main title, small enough to respect the safety margins. Main title on ONE line in bold, highly legible warm ivory geometric sans-serif. ByteFront and Espresso must be the same apparent font size and weight. Underneath the title, centered, a smaller humanist sans-serif tagline. A third even smaller, still legible line below, muted warm ivory. No other text.
Text verbatim (case and punctuation exact):
"ByteFront Espresso"
"Your morning brief on tech."
"AI, chips, robotics and the business behind them."
Hierarchy: brand title strongest, tagline clear, third line understated. Text must be perfect, clean, crisp, flat and readable. Keep the whole typography block inside x=780 to 1940 and y=600 to 845, with the monogram inside x=600 to 740 and y=615 to 765. Use roughly 88 px title, 44 px tagline, 30 px topic line at 2560 px canvas width, with tasteful spacing. These coordinates refer to the full 2560x1440 output and may be proportionally followed.
Avoid: coffee cup clipart, photographic mugs, robot heads, random code, stock market charts, extra badges, dates, posting schedules, subscribe buttons, YouTube logos, watermarks, decorative borders, crop guides, safe-area outlines, visual clutter, illegible text, glowing neon lettering.
Deliver a polished upload-ready PNG no larger than 6MB if possible.
```

Second-pass prompt:

```text
Edit target: the supplied finished ByteFront Espresso banner. Make a high-resolution final export suitable for direct upload to YouTube.
REQUIRED PIXEL SIZE: output 2560x1440 pixels, 16:9. The input is only 1672x941, which YouTube rejects. Output MUST be at least 2048 pixels wide and 1152 pixels tall. Prefer exactly 2560x1440 pixels. Produce a 4K 3840x2160 version if that is necessary to satisfy resolution. This is a resolution / export pass, not a redesign.
Preserve the entire composition, palette, circuit B logo, background, and all text exactly:
ByteFront Espresso
Your morning brief on tech.
AI, chips, robotics and the business behind them.
Keep the same centered text block and generous upper/lower negative space, with all text and B mark inside the central 58% width and central 25% height.
Use full detail, crisp antialiased typography, navy fine-paper grain, copper glow on right, cobalt circuitry on left. No new objects, no added text, no crop guides, no borders. Deliver a single high-resolution banner PNG.
```
