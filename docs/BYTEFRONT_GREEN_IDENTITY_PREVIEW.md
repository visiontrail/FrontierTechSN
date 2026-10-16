# Bytefront green identity previews

Prepared October 16, 2026 for visual review. These assets extend the green banner preview into an avatar and two reusable cover directions. They are not published, do not replace the existing channel artwork, and do not change the production generation settings.

## Deliverables

All generated assets remain in ignored `outputs/channel-branding-20261016/green-v1/`:

| File | Actual dimensions | Role |
| --- | --- | --- |
| `avatar.png` | 1254 × 1254 | Forest-green B monogram on warm ivory; generous circular-crop clearance |
| `byte-by-byte-cover-sample.png` | 1672 × 941 | Illustrated explainer cover, sample hook `WHY WE OVERTHINK` |
| `espresso-cover-sample.png` | 1672 × 941 | Editorial subject cover, sample hook `THE CHIP RACE` |
| `generation-prompts.json` | Text | Exact initial prompts, reference path and Espresso refinement prompt |

The two covers are flattened raster samples plus reusable generation instructions, not layered design files. Their subjects and headlines demonstrate the layout; they are not assertions that corresponding episodes have been published. The processor is an unbranded illustration, not evidence of a real product announcement.

## Shared identity

- Warm ivory `#FDFBF7`, forest green `#1A4A3A`, muted green `#426454`, sparse dark gold `#80601B`.
- Extremely subtle natural paper texture and modern Geist/Inter-like sans-serif lettering. Generated fonts and raster colors are approximate.
- Generous whitespace; no large brand panel, blue/navy base, terracotta accent, glow, ornamental foliage or distressed paper.
- On covers, keep the small Bytefront/series lockup at the upper left. Give the headline and subject visual priority. Leave the lower-right timestamp area quiet.
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

## Reusable Espresso cover prompt

Attach the finished Espresso sample as a style/layout reference. For real news, additionally provide an appropriate source image when the subject's real appearance or product identity matters. Preserve authentic colors and marks in source imagery.

```text
Create one finished landscape 16:9 YouTube cover for Bytefront Espresso.
Use the attached sample only for visual identity and layout. Never retain its chip
or headline unless the current story actually calls for them.
Warm ivory #FDFBF7 with extremely subtle paper texture; forest-green #1A4A3A typography,
muted-green #426454 supporting elements and one short dark-gold #80601B accent rule.
Keep the small Bytefront / Espresso lockup at top left and generous outer margins.
Render this exact reviewed hook, at most four English words: {HOOK}.
Place large clean modern sans-serif text on the left and one dominant news subject
on the right: {SUBJECT}. Use the supplied authentic subject image when available;
otherwise make a clearly illustrative, unbranded object image without invented labels.
The verified story context is {STORY_CONTEXT}. Preserve its uncertainty and factual scope.
Avoid invented announcements, logos, numbers, busy collages, decorative circuits,
news tickers, gradients, glow, dates, episode numbers and additional small captions.
Keep the lower-right timestamp zone quiet. Return one final flat cover image.
```

## Generation and checks

Generated with the built-in `image_gen` tool. The avatar used the existing B logo as a shape reference; both covers used the green banner as a style reference. A focused image edit removed the first Espresso sample's decorative background circuits while preserving the chip, headline and branding.

The final three images were visually inspected for spelling, recognizable identity, spacing, consistent palette and subject clarity. PNG dimensions were read from their headers after copying them into the workspace. No automated test suite was needed for these image previews. Approval of these visual directions is separate from publishing or integrating them into the production prompts.
