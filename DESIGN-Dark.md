---
version: "alpha"
name: "Nexus | Intelligent Data Routing"
description: "Nexus Intelligent Background Effect is designed for delivering a visual treatment or immersive background effect. Key features include atmospheric visuals, motion depth, and flexible presentation layering. It is suitable for visual-first pages, motion studies, and atmospheric hero treatments."
colors:
  primary: "#25254D"
  secondary: "#1C1C38"
  tertiary: "#9CA3AF"
  neutral: "#111115"
  background: "#111115"
  surface: "#FFFFFF"
  text-primary: "#FFFFFF"
  text-secondary: "#9CA3AF"
  accent: "#25254D"
typography:
  display-lg:
    fontFamily: "Geist"
    fontSize: "88px"
    fontWeight: 300
    lineHeight: "88px"
    letterSpacing: "-0.05em"
  body-md:
    fontFamily: "Geist"
    fontSize: "14px"
    fontWeight: 500
    lineHeight: "20px"
rounded:
  md: "0px"
spacing:
  base: "4px"
  sm: "4px"
  md: "10px"
  lg: "16px"
  xl: "20px"
  gap: "6px"
  card-padding: "13px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface}"
    typography: "{typography.body-md}"
    rounded: "{rounded.md}"
    padding: "10px"
  button-link:
    textColor: "{colors.tertiary}"
    rounded: "{rounded.md}"
    padding: "10px"
---

## Overview

- **Composition cues:**
  - Layout: Flex
  - Content Width: Bounded
  - Framing: Glassy
  - Grid: Minimal

## Colors

The color system uses dark mode with #25254D as the main accent and #111115 as the neutral foundation.

- **Primary (#25254D):** Main accent and emphasis color.
- **Secondary (#1C1C38):** Supporting accent for secondary emphasis.
- **Tertiary (#9CA3AF):** Reserved accent for supporting contrast moments.
- **Neutral (#111115):** Neutral foundation for backgrounds, surfaces, and supporting chrome.

- **Usage:** Background: #111115; Surface: #FFFFFF; Text Primary: #FFFFFF; Text Secondary: #9CA3AF; Accent: #25254D

## Typography

Typography relies on Geist across display, body, and utility text.

- **Display (`display-lg`):** Geist, 88px, weight 300, line-height 88px, letter-spacing -0.05em.
- **Body (`body-md`):** Geist, 14px, weight 500, line-height 20px.

## Layout

Layout follows a flex composition with reusable spacing tokens. Preserve the flex, bounded structural frame before changing ornament or component styling. Use 4px as the base rhythm and let larger gaps step up from that cadence instead of introducing unrelated spacing values.

Treat the page as a flex / bounded composition, and keep that framing stable when adding or remixing sections.

- **Layout type:** Flex
- **Content width:** Bounded
- **Base unit:** 4px
- **Scale:** 4px, 10px, 16px, 20px, 24px, 32px, 48px, 128px
- **Card padding:** 13px
- **Gaps:** 6px, 8px

## Elevation & Depth

Depth is communicated through glass, border contrast, and reusable shadow or blur treatments. Keep those recipes consistent across hero panels, cards, and controls so the page reads as one material system.

Surfaces should read as glass first, with borders, shadows, and blur only reinforcing that material choice.

- **Surface style:** Glass
- **Borders:** 1px #FFFFFF
- **Shadows:** rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(0, 0, 0, 0.1) 0px 10px 15px -3px, rgba(0, 0, 0, 0.1) 0px 4px 6px -4px; rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(50, 50, 150, 0.2) 0px 0px 20px 0px
- **Blur:** 12px

## Shapes

Shapes rely on a tight radius system anchored by 4px and scaled across cards, buttons, and supporting surfaces. Icon geometry should stay compatible with that soft-to-controlled silhouette.

Use the radius family intentionally: larger surfaces can open up, but controls and badges should stay within the same rounded DNA instead of inventing sharper or pill-only exceptions.

- **Corner radii:** 4px
- **Icon treatment:** Linear
- **Icon sets:** Solar

## Components

Anchor interactions to the detected button styles.

### Buttons
- **Primary:** background #25254D, text #FFFFFF, radius 0px, padding 10px, border 0px solid rgb(229, 231, 235).
- **Links:** text #9CA3AF, radius 0px, padding 10px, border 0px solid rgb(229, 231, 235).

### Iconography
- **Treatment:** Linear.
- **Sets:** Solar.

## Do's and Don'ts

Use these constraints to keep future generations aligned with the current system instead of drifting into adjacent styles.

### Do
- Do use the primary palette as the main accent for emphasis and action states.
- Do keep spacing aligned to the detected 4px rhythm.
- Do reuse the Glass surface treatment consistently across cards and controls.
- Do keep corner radii within the detected 4px family.

### Don't
- Don't introduce extra accent colors outside the core palette roles unless the page needs a new semantic state.
- Don't mix unrelated shadow or blur recipes that break the current depth system.
- Don't exceed the detected moderate motion intensity without a deliberate reason.

## Motion

Motion feels controlled and interface-led across text, layout, and section transitions. Timing clusters around 150ms and 1000ms. Easing favors ease and 0. Hover behavior focuses on text and color changes.

**Motion Level:** moderate

**Durations:** 150ms, 1000ms, 300ms

**Easings:** ease, 0, 0.2, 1), cubic-bezier(0.4, cubic-bezier(0

**Hover Patterns:** text, color, shadow

## WebGL

Reconstruct the graphics as a full-bleed background field using canvas-backed effect. The effect should read as technical, meditative, and atmospheric: dot-matrix particle field with black and sparse spacing. Build it from dot particles + soft depth fade so the effect reads clearly. Animate it as slow breathing pulse. Interaction can react to the pointer, but only as a subtle drift. Preserve dom fallback.

**Id:** webgl

**Label:** WebGL

**Stack:** WebGL

**Insights:**
  - **Scene:**
    - **Value:** Full-bleed background field
  - **Effect:**
    - **Value:** Dot-matrix particle field
  - **Primitives:**
    - **Value:** Dot particles + soft depth fade
  - **Motion:**
    - **Value:** Slow breathing pulse
  - **Interaction:**
    - **Value:** Pointer-reactive drift
  - **Render:**
    - **Value:** Canvas-backed effect

**Techniques:** Dot matrix, Breathing pulse, Pointer parallax, Noise fields, DOM fallback

**Code Evidence:**
  - **HTML reference:**
    - **Language:** html
    - **Snippet:**
      ```html
      <!-- Canvas for WebGL/Pixel animated background -->
      <canvas id="pixelGrid" class="absolute inset-0 z-0 pointer-events-none w-full h-full"></canvas>

      <!-- Navigation -->
      ```
  - **JS reference:**
    - **Language:** js
    - **Snippet:**
      ```
      // Trigger reveal animations on load
      window.addEventListener('load', () => {
          setTimeout(() => {
              document.querySelectorAll('.reveal').forEach(el => {
                  el.classList.remove('opacity-0', 'translate-y-4', 'translate-y-6');
              });
          }, 50);
      });
      …
      ```
