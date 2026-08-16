---
title: Non-Additive Linear Context HPM
---

# Non-Additive Linear Context HPM

Last updated: 2026-08-15

This model isolates lineup construction from player-level box-score credit. It
uses standardized linear Ridge context with only five non-additive unit-shape
features: bottom-two shooting, credible-shooter count, usage concentration,
shooting-by-usage, and shooter-by-passing.

It deliberately excludes every raw summed player rate, summed ORB% claims,
and additive profile-uncertainty total. Those inputs can be expressed as a
linear player-level box-score adjustment and therefore do not identify five-man
context by themselves.

## Run

```bash
uv run nba-train-hpm-linear-nonadditive --through-season 2025-26
```
