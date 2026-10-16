# Bytefront green identity previews

Prepared October 16, 2026 for visual review. These assets extend the green banner preview into an avatar and two reusable cover directions. They are not published, do not replace the existing channel artwork, and do not change the production generation settings.

## Deliverables

All generated assets remain in ignored `outputs/channel-branding-20261016/green-v1/`:

| File | Actual dimensions | Role |
| --- | --- | --- |
| `avatar.png` | 1254 × 1254 | Forest-green B monogram on warm ivory; generous circular-crop clearance |
| `byte-by-byte-cover-sample.png` | 1672 × 941 | Illustrated explainer cover, sample hook `WHY WE OVERTHINK` |
| `espresso-series-cover-v2.png` | 1672 × 941 | Current Espresso series cover: `Bytefront Espresso / Tech News Briefings`, spanning AI, chips, robotics and business |
| `espresso-series-cover-v2-prompt.txt` | Text | Exact built-in image-generation prompt for the corrected series cover |
| `espresso-cover-sample.png` | 1672 × 941 | Superseded chip-topic example; rejected as the permanent Espresso series identity |
| `generation-prompts.json` | Text | Exact initial prompts, reference path and Espresso refinement prompt |

The covers are flattened raster samples plus reusable generation instructions, not layered design files. The Byte by Byte subject and headline demonstrate an episode layout; they are not assertions that a corresponding episode has been published. Espresso's corrected v2 is a permanent series cover. The old processor illustration is retained only as a superseded iteration.

The user corrected the initial Espresso sample: `THE CHIP RACE` narrowed a general technology news briefing into a chip-specific topic. The series definition is concise briefings on AI, chips, robotics, and the business behind them. A permanent series cover must express this scope. A particular news headline or subject belongs to an episode cover and must come from that episode's actual rundown.

## Shared identity

- Warm ivory `#FDFBF7`, forest green `#1A4A3A`, muted green `#426454`, sparse dark gold `#80601B`.
- Extremely subtle natural paper texture and modern Geist/Inter-like sans-serif lettering. Generated fonts and raster colors are approximate.
- Generous whitespace; no large brand panel, blue/navy base, terracotta accent, glow, ornamental foliage or distressed paper.
- On episode covers, keep the small Bytefront/series lockup at the upper left and give the episode headline and subject visual priority. On permanent series covers, make the series name and purpose dominant. Leave the lower-right timestamp area quiet.
- Avatar: recognizable angular B, diagonal lower cut and three simplified circuit terminals. No small wordmark or tagline.

Style references were the green banner preview and XChineseTrans's `assets/brand/leoguolabs-light-v1/reference-infographic-paper-v2.png` / `AGENTS.md`. Preserve Bytefront's own identity and names; do not import the LeoGuoLabs signature.

## Reusable Byte by Byte cover prompt

Attach the finished Byte by Byte sample as a style/layout reference. Replace the three braced fields with the reviewed episode's content before generation.

```text
Create one finished landscape 16:9 YouTube cover for Bytefront's Byte by Byte series.
Use the attached sample only for visual identity and layout, not as a source of episode facts.
Warm ivory #FDFBF7, barely visible paper fibers, forest green #1A4A3A,
muted green #426454 and sparse dark gold #80601B. Clean modern sans-serif lettering.
Keep a discreet top-left Bytefront / Byte by Byte identity lockup within about 20% width
and 10% height. The main headline and illustration dominate, not the branding.
Render this exact short hook, at most four English words: {HOOK}.
Set it in large forest-green lettering on the left, with generous margins and no more
than two lines. Use one original expressive line-drawn scene on the right: {SCENE}.
Retain the sample's friendly round-headed slender character and muted-green shirt when
a person helps explain the topic. Use one clear action and only story-essential props.
The reviewed episode context is {EPISODE_CONTEXT}; do not exaggerate its claims or
turn uncertain conclusions into factual headlines.
No photorealistic people, crowded collage, glossy 3D, decorative circuits, glow,
extra captions, date or episode number. Keep the bottom-right timestamp zone quiet.
Return one final flat cover image, not a mockup, layout guide or comparison sheet.
```

## Reusable Espresso permanent series cover prompt

Use `espresso-series-cover-v2.png` as the current style/layout reference. Do not use the superseded chip-topic sample as the default subject or headline.

```text
Create one finished landscape 16:9 permanent series/playlist cover for Bytefront Espresso,
a general technology news briefing spanning AI, chips, robotics and the business behind them.
Warm ivory #FDFBF7 with extremely subtle paper texture; forest-green #1A4A3A typography,
muted-green #426454 supporting elements and one short dark-gold #80601B accent rule.
Use modern sans-serif lettering and generous outer margins. Keep the B monogram small.
Make the permanent title 'Bytefront Espresso' dominant, followed by 'Tech News Briefings'.
Include the smaller scope line 'AI · Chips · Robotics · Business'.
Use a restrained editorial line illustration of news sheets as a secondary visual.
All domains are peers; do not use a single technology topic as the defining hero image.
No specific news headline, company, product announcement, date or episode number.
No decorative circuit backgrounds, gradients, glow or extra captions.
Keep the lower-right timestamp zone quiet. Return one final flat cover image.
```

For future episode covers, derive the headline and subject imagery from the reviewed news rundown. No fixed chip, chip-race headline or semiconductor-only topic restriction applies. Preserve necessary uncertainty in the headline and authentic colors/marks in supplied source imagery. Keep the shared color, type and branding system while adapting the composition to the actual news.

## Generation and checks

Generated with the built-in `image_gen` tool. The avatar used the existing B logo as a shape reference; the initial covers used the green banner as a style reference. The superseded Espresso iteration had its decorative background circuits removed. Following the user's scope correction, v2 replaced the chip and topical headline with the permanent series title, briefing descriptor, full coverage line and general news-sheet illustration.

The images were visually inspected for spelling, recognizable identity, spacing, consistent palette and subject clarity. Espresso v2 was also checked against the series definition: no chip-race headline or dominant single-sector subject, and all four coverage areas appear in the scope line. PNG dimensions were read from their headers after copying them into the workspace. No automated test suite was needed for these image previews. Approval of these visual directions is separate from publishing or integrating them into the production prompts.
