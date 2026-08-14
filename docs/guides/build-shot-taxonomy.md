---
last_updated: "2026-08-13"
---

# Build Shot Taxonomy

Build the validated player-season panel and curated event partitions first,
then run:

```bash
uv run nba-build-shot-taxonomy
```

The command atomically refreshes `data/analytical/shot_taxonomy/`. It reads
only local curated regular-season events and makes no NBA API requests.

Validate the resulting contract with:

```bash
uv run python -c "from nba_lineup_model.modeling.shot_taxonomy import validate_shot_taxonomy; validate_shot_taxonomy('data/analytical/shot_taxonomy')"
```

The output contains player-season raw attempts/makes, possession-based
stabilized rates, subtype coverage, and league references. Future HPM or
profile-token models must use a prior-season row only.
