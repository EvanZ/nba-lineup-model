# Build The Box-Score Prior Panel

Build the validated player-season panel first, then run:

```bash
uv run nba-build-box-score-prior-panel
```

The command atomically refreshes `data/analytical/box_score_prior_panel/`.
It derives all dynamic features from the immediately preceding season and
excludes the initial 1996-97 archive season from model-ready targets because
it has no in-archive prior season.

The panel is a data-preparation artifact. It does not fit a box-score model or
select predictive hyperparameters. See [Box-Score Prior Panel](../data/box-score-prior-panel.md)
for the contract and [Box-Score RAPM Prior](../models/box-score-prior.md) for
the planned model program.
