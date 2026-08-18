---
version: "alpha"
name: "Aura — A Mindful Companion for Your Spiritual Journey"
description: "Aura Mindful CTA Section is designed for building reusable UI components in modern web projects. Key features include reusable structure, responsive behavior, and production-ready presentation. It is suitable for component libraries and responsive product interfaces."
colors:
  primary: "#1A4A3A"
  secondary: "#E5C158"
  tertiary: "#D4AF37"
  neutral: "#FDFBF7"
  background: "#1A4A3A"
  surface: "#E5C158"
  text-primary: "#1A4A3A"
  text-secondary: "#FDFBF7"
  border: "#1A4A3A"
  accent: "#1A4A3A"
typography:
  display-lg:
    fontFamily: "Cormorant Garamond"
    fontSize: "72px"
    fontWeight: 300
    lineHeight: "72px"
    letterSpacing: "-0.025em"
  body-md:
    fontFamily: "Plus Jakarta Sans"
    fontSize: "14px"
    fontWeight: 300
    lineHeight: "22.75px"
  label-md:
    fontFamily: "Plus Jakarta Sans"
    fontSize: "14px"
    fontWeight: 500
    lineHeight: "20px"
    letterSpacing: "0.35px"
rounded:
  md: "0px"
  full: "9999px"
spacing:
  base: "4px"
  sm: "4px"
  md: "6.5px"
  lg: "8px"
  xl: "12px"
  gap: "8px"
  card-padding: "12px"
  section-padding: "24px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.neutral}"
    typography: "{typography.label-md}"
    rounded: "{rounded.full}"
    padding: "16px"
  button-secondary:
    textColor: "{colors.primary}"
    typography: "{typography.label-md}"
    rounded: "{rounded.full}"
    padding: "16px"
  button-link:
    textColor: "{colors.neutral}"
    rounded: "{rounded.md}"
    padding: "0px"
  card:
    rounded: "24px"
    padding: "32px"
---

## Overview

- **Composition cues:**
  - Layout: Grid
  - Content Width: Bounded
  - Framing: Glassy
  - Grid: Strong

## Colors

The color system uses light mode with #1A4A3A as the main accent and #FDFBF7 as the neutral foundation.

- **Primary (#1A4A3A):** Main accent and emphasis color.
- **Secondary (#E5C158):** Supporting accent for secondary emphasis.
- **Tertiary (#D4AF37):** Reserved accent for supporting contrast moments.
- **Neutral (#FDFBF7):** Neutral foundation for backgrounds, surfaces, and supporting chrome.

- **Usage:** Background: #1A4A3A; Surface: #E5C158; Text Primary: #1A4A3A; Text Secondary: #FDFBF7; Border: #1A4A3A; Accent: #1A4A3A

## Typography

Typography pairs Cormorant Garamond for display hierarchy with Plus Jakarta Sans for supporting content and interface copy.

- **Display (`display-lg`):** Cormorant Garamond, 72px, weight 300, line-height 72px, letter-spacing -0.025em.
- **Body (`body-md`):** Plus Jakarta Sans, 14px, weight 300, line-height 22.75px.
- **Labels (`label-md`):** Plus Jakarta Sans, 14px, weight 500, line-height 20px, letter-spacing 0.35px.

## Layout

Layout follows a grid composition with reusable spacing tokens. Preserve the grid, bounded structural frame before changing ornament or component styling. Use 4px as the base rhythm and let larger gaps step up from that cadence instead of introducing unrelated spacing values.

Treat the page as a grid / bounded composition, and keep that framing stable when adding or remixing sections.

- **Layout type:** Grid
- **Content width:** Bounded
- **Base unit:** 4px
- **Scale:** 4px, 6.5px, 8px, 12px, 16px, 24px, 32px, 40px
- **Section padding:** 24px, 32px, 48px, 64px
- **Card padding:** 12px, 16px, 24px, 32px
- **Gaps:** 8px, 12px, 16px, 24px

## Elevation & Depth

Depth is communicated through glass, border contrast, and reusable shadow or blur treatments. Keep those recipes consistent across hero panels, cards, and controls so the page reads as one material system.

Surfaces should read as glass first, with borders, shadows, and blur only reinforcing that material choice.

- **Surface style:** Glass
- **Borders:** 1px #1A4A3A; 1px #FFFFFF; 1px #E5C158; 1px #FDFBF7
- **Shadows:** rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(0, 0, 0, 0.1) 0px 20px 25px -5px, rgba(0, 0, 0, 0.1) 0px 8px 10px -6px; rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(0, 0, 0, 0.05) 0px 1px 2px 0px; rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(0, 0, 0, 0.1) 0px 10px 15px -3px, rgba(0, 0, 0, 0.1) 0px 4px 6px -4px
- **Blur:** 4px

### Techniques
- **Gradient border shell:** Use a thin gradient border shell around the main card. Wrap the surface in an outer shell with 12px padding and a 48px radius. Drive the shell with none so the edge reads like premium depth instead of a flat stroke. Keep the actual stroke understated so the gradient shell remains the hero edge treatment. Inset the real content surface inside the wrapper with a slightly smaller radius so the gradient only appears as a hairline frame.

## Shapes

Shapes rely on a tight radius system anchored by 16px and scaled across cards, buttons, and supporting surfaces. Icon geometry should stay compatible with that soft-to-controlled silhouette.

Use the radius family intentionally: larger surfaces can open up, but controls and badges should stay within the same rounded DNA instead of inventing sharper or pill-only exceptions.

- **Corner radii:** 16px, 24px, 38px, 48px, 9999px
- **Icon treatment:** Linear
- **Icon sets:** Solar

## Components

Anchor interactions to the detected button styles. Reuse the existing card surface recipe for content blocks.

### Buttons
- **Primary:** background #1A4A3A, text #FDFBF7, radius 9999px, padding 16px, border 0px solid rgb(229, 231, 235).
- **Secondary:** text #1A4A3A, radius 9999px, padding 16px, border 1px solid rgba(26, 74, 58, 0.2).
- **Links:** text #FDFBF7, radius 0px, padding 0px, border 0px solid rgb(229, 231, 235).

### Cards and Surfaces
- **Card surface:** background rgba(255, 255, 255, 0.05), border 1px solid rgba(255, 255, 255, 0.1), radius 24px, padding 32px, shadow none.
- **Card surface:** background #FFFFFF, border 1px solid rgba(26, 74, 58, 0.05), radius 16px, padding 16px, shadow rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(0, 0, 0, 0.1) 0px 20px 25px -5px, rgba(0, 0, 0, 0.1) 0px 8px 10px -6px.
- **Card surface:** background rgba(26, 74, 58, 0.05), border 1px solid rgba(26, 74, 58, 0.1), radius 48px, padding 64px, shadow none.

### Iconography
- **Treatment:** Linear.
- **Sets:** Solar.

## Do's and Don'ts

Use these constraints to keep future generations aligned with the current system instead of drifting into adjacent styles.

### Do
- Do use the primary palette as the main accent for emphasis and action states.
- Do keep spacing aligned to the detected 4px rhythm.
- Do reuse the Glass surface treatment consistently across cards and controls.
- Do keep corner radii within the detected 16px, 24px, 38px, 48px, 9999px family.

### Don't
- Don't introduce extra accent colors outside the core palette roles unless the page needs a new semantic state.
- Don't mix unrelated shadow or blur recipes that break the current depth system.
- Don't exceed the detected expressive motion intensity without a deliberate reason.

## Motion

Motion feels expressive but remains focused on interface, text, and layout transitions. Timing clusters around 150ms and 2000ms. Easing favors ease and cubic-bezier(0.4. Hover behavior focuses on text and color changes.

**Motion Level:** expressive

**Durations:** 150ms, 2000ms, 40000ms, 500ms

**Easings:** ease, cubic-bezier(0.4, 0, 1), 0.2, 0.6

**Hover Patterns:** text, color, stroke, shadow
