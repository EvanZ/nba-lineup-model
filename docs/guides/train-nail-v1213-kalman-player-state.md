# Train NAIL-RAPM v1.2.1.3 Kalman Player State

*Last updated: 2026-08-24*

Train the forward-only player-state candidate, then replay the same three
frozen regular-season and playoff cohorts used by the production model:

```bash
uv run nba-train-nail-v1213-kalman-player-state \
  --log-path artifacts/logs/nail-v1213-kalman-player-state.log

uv run nba-backtest-nail-v1213-kalman-player-state \
  --log-path artifacts/logs/nail-v1213-kalman-player-state-frozen.log
```

Follow training with:

```bash
tail -f artifacts/logs/nail-v1213-kalman-player-state.log
```

The candidate retains the production B2B schedule control, standard-USG%
profile contract, and two retained non-additive lineup features. It changes
only the established-player RAPM prior.

For completed season \(t\), the fitted player coefficient \(y_{i,t}\) is a
noisy observation of the pre-season player state \(m^-_{i,t}\):

\[
K_{i,t}=\frac{P^-_{i,t}}{P^-_{i,t}+R_{i,t}},
\qquad
m^+_{i,t}=m^-_{i,t}+K_{i,t}(y_{i,t}-m^-_{i,t}),
\]

where \(R_{i,t}=4000 / \text{on-court possessions}_{i,t}\). The posterior is
advanced with the existing value-conditioned aging model; its variance receives
one random-walk process increment of \(1.0\) per off-season. These constants
are fixed for this first frozen experiment and persisted in the model artifact.

Cold starts retain the existing exposure-gated replacement/draft path. Players
returning after an absence are advanced through each missing age transition from
their last filtered posterior state.
