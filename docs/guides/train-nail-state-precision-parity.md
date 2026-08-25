# Train State-Precision NAIL Parity Replay

*Last updated: 2026-08-24*

This replay runs the exact NAIL-RAPM v1.2.1.2 feature, prior, context, and B2B
contracts with the State-Precision ridge solver. Every player has relative
precision one, so its results must match the production artifact before any
non-uniform state uncertainty is introduced.

```bash
uv run nba-train-nail-state-precision-parity \
  --through-season 2025-26 \
  --log-path artifacts/logs/nail-state-precision-parity.log
```

After training, compare the replay with the production run:

```bash
uv run nba-audit-nail-state-precision-parity \
  artifacts/models/forward_nail_rapm_v1212_back_to_back/2025-26/<production-run> \
  artifacts/models/forward_nail_state_precision_parity/2025-26/<parity-run>
```

The audit checks player coefficients, prior vectors, possession predictions,
and game predictions. The maximum absolute difference must be at most
`1e-10`; otherwise the state-precision integration is invalid and no
non-uniform refit should proceed.
