# Train the RAPM Baselines

The first training command runs three regular-season models:

1. a possession-weighted mean;
2. signed schedule-adjusted team strengths;
3. signed one-number player RAPM.

```bash
uv run nba-train-rapm 2025-26
```

The command first rebuilds and validates the RAPM stint mart. It then uses
three expanding chronological validation folds, holds out the final 15% of
games for one test evaluation, and refits the selected models on the complete
regular season for ratings and rankings.

## Sparse encoding

The player design matrix has one column per player appearing in the modeling
sample. A row contains `+1` for each home player and `-1` for each away player.
The team matrix uses the same signs with one column per team. Both are SciPy
CSR matrices consumed directly by Scikit-learn's sparse `lsqr` ridge solver.

The fitted intercept estimates home-court advantage. The unpenalized lambda
endpoint is available for the team baseline; the player model selects from the
same normalized grid.

## Chronological validation

Games, rather than stints, are the split unit. Every stint from one game stays
in one fold. By default:

- three expanding validation folds each hold out the next 10% of games;
- the final 15% of games remain untouched during lambda selection;
- lambda minimizes possession-weighted validation MSE.

The normalized objective is weighted mean squared error plus lambda times the
squared coefficient norm. This keeps lambda comparable as training folds grow.

Customize the experiment with:

```bash
uv run nba-train-rapm 2025-26 \
  --cv-folds 3 \
  --validation-fraction 0.10 \
  --test-fraction 0.15 \
  --minimum-ranking-possessions 500
```

## Outputs

The analytical input is written under `data/analytical/rapm_stints/`. Each
immutable run is written under:

```text
artifacts/models/rapm/{season}/{run_id}/
```

It contains CV results, final-test metrics and predictions, player rankings,
team ratings, game split assignments, sparse column mappings, model
parameters, and a hash-validated manifest. `latest.json` identifies the newest
successful run.

These outputs are ignored by Git. A ranking is published in the documentation
only after its run has been reviewed and explicitly promoted.
