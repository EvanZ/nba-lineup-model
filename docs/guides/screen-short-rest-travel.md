---
last_updated: "2026-08-30"
---

# Screen Short-Rest Travel

Run the low-cost travel-control diagnostic before any recursive NAIL refit:

```bash
uv run nba-screen-short-rest-travel
```

The screen uses the full frozen production NAIL prediction, including its
source-season home-court intercept and back-to-back adjustment. It regresses
the remaining target-season stint residual on the signed, home-minus-away
travel distance within 48 scheduled-tipoff hours.

The screen excludes a game only when either team has no preceding competitive
game in that season. It writes stints, decile summaries, a season summary, and
an SVG diagnostic under `artifacts/models/analysis/short_rest_travel_screen/`.
