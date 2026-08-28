---
last_updated: "2026-08-28"
---

# Tune NAIL Context Block Penalties

Run the pre-frozen development grid and selection:

```bash
uv run python -m nba_lineup_model.modeling.nail_v1213_block_penalty_development \
  --log-path artifacts/logs/nail-v1213-block-penalty-development.log
```

The selected r=16 completed fit can resume from its immutable 2022-23
checkpoint. Only 2023-24 through 2025-26 are refit:

```bash
uv run python -m nba_lineup_model.modeling.forward_nail_v1213_block_penalty \
  --nonadditive-ratio 16 \
  --through-season 2025-26 \
  --resume-from artifacts/models/forward_nail_rapm_v1213_block_penalty_r16/2022-23/forward-nail-rapm-v1213-block-penalty-r16-2022-23-20260828T143610Z-99190259 \
  --log-path artifacts/logs/forward-nail-v1213-block-penalty-r16-resume-2025-26.log
```

The resumed artifact stores only new model-state dictionaries plus an
immutable base-artifact reference. Parquet player/prior tables remain complete,
while evaluators recursively overlay the three-season model delta.

Run the selected candidate and production through the corrected frozen
evaluator:

```bash
uv run python -m nba_lineup_model.modeling.nail_v1213_block_penalty_development \
  --locked-backtest \
  --log-path artifacts/logs/nail-v1213-block-penalty-r16-locked-backtest.log

uv run python -m nba_lineup_model.modeling.nail_v1213_block_penalty_development \
  --locked-incumbent \
  --log-path artifacts/logs/nail-v1213-block-penalty-incumbent-corrected-backtest.log
```

Finally, regenerate all ten coefficient histories from the selected completed
fit with `nba_lineup_model.modeling.nail_v1213_promotion_review`, passing the
r=16 source run, output root, chart paths, expected model name, and chart title
shown in the [study page](../models/nail-context-block-penalty.md).
