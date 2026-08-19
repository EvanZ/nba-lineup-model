---
title: Train NAIL Token Residual Models
last_updated: "2026-08-17"
---

# Train NAIL Token Residual Models

Last updated: 2026-08-17

Run the fixed five-epoch token-MLP and within-unit Set Attention comparison on
the three frozen seasons:

```bash
uv run nba-evaluate-nail-token-residual --epochs 5
```

Follow the durable project run log when the command is launched with `tee`:

```bash
tail -f artifacts/logs/nail-token-residual-frozen.log
```

Historical residual stints are cached under
`artifacts/models/analysis/nail_token_residual_inputs/`. The cache key includes
the source additive-only NAIL run and current player-season panel hash. Model
checkpoints, frozen predictions, and aggregate metrics are written under
`artifacts/models/analysis/nail_token_residual_frozen/`.

For a single architecture:

```bash
uv run nba-evaluate-nail-token-residual \
  --architectures set_attention \
  --epochs 5
```
