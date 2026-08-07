---
last_updated: "2026-08-05"
---

# Fetch Draft History

Fetch one official NBA draft class directly from the `drafthistory` Stats
endpoint and normalize it without converting NBA player IDs to floats:

```bash
uv run nba-fetch-draft-history 2026-27
```

The raw API response is cached at:

```text
data/raw/drafthistory/<season>.json
```

The normalized class is written to:

```text
data/curated/draft_history/<season>/part-00000.parquet
```

For an upcoming class, score the cached response with the pinned pre-season
draft-rate, low-exposure gate, and replacement-token artifacts:

```bash
uv run nba-fetch-team-rosters 2026-27
uv run nba-build-forward-draft-history-rankings --season 2026-27 --render-docs-page
```

This produces a versioned artifact under:

```text
artifacts/models/forward_draft_history_cold_start/<season>/<run_id>/
```

`drafthistory` supplies draft slot, team, affiliation, and NBA player ID.
`commonteamroster` supplies the active roster's player IDs, position, height,
weight, birth date, age, experience, school, and acquisition status. The
ranking joins the sources by player ID and only imputes a field from the
historical drafted-player reference profile when a listed roster leaves it
missing. Its draft rate, exposure gate, and replacement token are fit from the
completed forward exposure-gated RAPM state through 2025-26. The resulting
ranking is still drafted-player-only; the roster table also provides the
candidate pool needed for a separate undrafted-player prior.
