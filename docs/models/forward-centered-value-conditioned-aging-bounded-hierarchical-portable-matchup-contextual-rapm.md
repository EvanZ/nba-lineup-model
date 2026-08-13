---
last_updated: "2026-08-10"
---

# Value-Conditioned Aging HPM

**HIPSTER PM with Value-Conditioned Aging** extends the centered aging
portable contextual model with a regularized interaction between age and the
completed prior-season player rating:

\[
\mu_{i,t}=f(\operatorname{age}_{i,t}, \ldots) +
\beta_{\mathrm{value-age}}
(\operatorname{age}_{i,t}-27)\operatorname{RAPM}_{i,t-1}.
\]

The previous aging branch modeled the effect of age and prior RAPM additively.
This specification can learn a different aging trajectory for a player with a
high prior rating, but retains ridge shrinkage and strictly forward expanding
validation. It remains a population estimate, not a hand-tuned exception for
individual stars.

The full prior vector is then centered with prior-season possession weights and
combined with the same exposure-gated cold-start branch and bounded
hierarchical portable-matchup contextual term:

\[
C(A,B)=h(x(A))-h(x(B))+q(x(A),x(B)).
\]

## Rebuilt Frozen 2025-26 Evaluation

The historical raw-to-stint recovery rebuilt the one-season RAPM layer and the
player-season panel before this strictly forward rerun:
`forward-centered-value-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm-2025-26-20260813T050955Z-c9b2e37d`.
The final-season aging branch again selected regularization `0.01` and recorded
the value-conditioned feature in its season metadata.

| Metric | Rebuilt Value-Conditioned Aging HPM |
| --- | ---: |
| Regular possession RMSE | **1.198736** |
| Regular eligible game-margin RMSE | **14.3214** |
| Full-game margin RMSE | **14.5898** |
| Full-game winner accuracy | 68.46% |
| Playoff possession RMSE | 1.192465 |
| Playoff eligible game-margin RMSE | 16.4069 |
| Team NetRtg RMSE | 3.5858 |
| Pythagorean wins RMSE | 7.6300 |

This is the current primary lineup-prediction reference: it has the best
regular possession and eligible game-margin performance among the models
rerun on the recovered history. The frozen leaderboard will be regenerated
only after its full candidate set is rebuilt on the same data contract.

See [the training guide](../guides/train-forward-centered-value-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm.md)
for the reproducible command and artifact contract.

<!-- value-conditioned-aging-reproducibility:start -->
## Deterministic Reproducibility

Value-Conditioned Aging HPM has no random initialization, sampling, or split
selection. A second seed is therefore not a meaningful perturbation. This audit
compares a full rerun against the previous immutable artifact with absolute
tolerance `1e-10`.

| Output | Reference rows | Rerun rows | Shared rows | Max absolute difference | All close |
| --- | ---: | ---: | ---: | ---: | --- |
| `cohort_metrics.parquet` | 2 | 2 | 2 | 0.000e+00 | yes |
| `game_predictions.parquet` | 1,315 | 1,315 | 1,315 | 0.000e+00 | yes |
| `historical_player_coefficients.parquet` | 14,560 | 14,560 | 14,560 | 0.000e+00 | yes |
| `team_net_rating_predictions.parquet` | 30 | 30 | 30 | 0.000e+00 | yes |
| `team_win_predictions.parquet` | 30 | 30 | 30 | 0.000e+00 | yes |

Audit artifact: `artifacts/analysis/value_conditioned_aging_reproducibility/2025-26/value-conditioned-aging-reproducibility-2025-26-20260810T174447Z-c3454967`. It retains per-column numerical differences.
<!-- value-conditioned-aging-reproducibility:end -->
