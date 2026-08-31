# Train NAIL Lead-Secondary Usage Gap

Last updated: 2026-08-30

This candidate preserves every component of production NAIL-RAPM v1.2.1.3 and
adds one non-additive, source-season context coordinate:

\[
g(U) = \max_{i \in U}\operatorname{USG\%}_i
- \operatorname{second\_max}_{i \in U}\operatorname{USG\%}_i.
\]

The five player profiles are shrinkage-adjusted from the immediately preceding
completed regular season. The game-level context term receives the home-minus-
away contrast, \(g(H)-g(A)\). It is a lead-handler allocation feature, not a
claim that lineups should lack a secondary ball handler.

The candidate passed the frozen residual screen and its association remained
positive after conditioning on the home-minus-away maximum frozen player prior.
That conditional audit is documented in [Screen a Frozen
Feature](screen-frozen-feature.md#candidate-result-lead-secondary-usage-gap).

## Train

```bash
uv run python -m nba_lineup_model.modeling.forward_nail_lead_secondary_usage_gap \
  --through-season 2025-26 \
  --log-path artifacts/logs/nail-lead-secondary-usage-gap.log
```

The run selects the player lambda season by season using the same
residualized-target chronological CV policy as production. It keeps the
value-conditioned aging prior, exposure-gated cold starts, additive profile
features, two retained non-additive terms, home-court advantage, and the
back-to-back schedule control unchanged.

## Frozen Replay

After training completes, compare it with production on the shared 2023-24 to
2025-26 frozen evaluation set:

```bash
uv run python -m nba_lineup_model.modeling.nail_lead_secondary_usage_gap_frozen_backtest \
  --log-path artifacts/logs/nail-lead-secondary-usage-gap-frozen.log
```

The candidate is not eligible for promotion until this three-season replay and
its bootstrap comparison are complete.

## Bootstrap Gate

```bash
uv run python -m nba_lineup_model.modeling.nail_lead_secondary_usage_gap_bootstrap
```

The paired bootstrap resamples games within each frozen season. Its primary
gate is full-game margin RMSE: the candidate's paired 95% upper confidence
bound may not exceed `0.5%` of the incumbent's RMSE in the pooled sample or in
any individual season.
