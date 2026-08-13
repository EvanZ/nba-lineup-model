---
last_updated: "2026-08-12"
---

# Context-Reattributed RAPM (CR-RAPM)

HPM deliberately reserves some predictive signal for a frozen lineup-context
term. That improves forecasting when a lineup's profile and matchup matter,
but a conventional player table can understate players who repeatedly create
favorable context. CR-RAPM is an accounting audit that asks how much of that
frozen context signal can be represented as a player term on the usual RAPM
design, and how much remains genuinely lineup-specific.

It is **not a replacement predictive model** and is not eligible for the
frozen leaderboard. It observes the realized target-season lineup exposures
only after the HPM forecast state was frozen.

## Ledger

For a target-season stint, the completed-season HPM ledger has the form

\[
\hat y = X\beta + b + C(A,B),
\]

where \(X\) is the signed home-minus-away player matrix, \(\beta\) is the
completed-season HPM player rating vector, \(b\) is its home-court intercept,
and \(C(A,B)\) is the portable composition-plus-matchup context prediction
from the prior-season frozen state.

CR-RAPM makes a separate possession-weighted ridge projection on those same
stints:

\[
C(A,B) \approx X\gamma + a + r(A,B).
\]

\(\gamma\) is the **context-reattribution** vector, \(a\) is its intercept,
and \(r\) is residual synergy. The projection uses HPM's selected target-year
player regularization, so \(\gamma\) is constrained on the same sparse-design
scale as the player fit. It is not tuned against game outcomes.

Substitution gives an exact accounting identity:

\[
\hat y = X(\beta + \gamma) + (b+a) + r(A,B).
\]

The player-facing CR-RAPM value is therefore

\[
\operatorname{CRRAPM}_i = \beta_i + \gamma_i.
\]

A positive \(\gamma_i\) says that, across the realized target-season units in
this audit, player \(i\)'s presence helps reconstruct HPM's context term after
the other players in a lineup are considered. It is not a causal estimate and
does not say that the player owns every nonlinear lineup effect.

## Reading Residual Synergy

The residual \(r(A,B)\) is the portion of HPM context that the additive player
projection cannot express. A residual near zero means that the frozen context
signal for that stint is effectively portable to its players under this design.
A large positive or negative residual means the model still needs a
configuration-specific interaction after all reattributed player terms are
included.

This distinction addresses the central attribution question directly:

- a player can gain CR-RAPM credit for recurring profile-based context;
- a unit can retain residual synergy without forcing that interaction onto any
  one teammate;
- the original HPM forecast is unchanged in every case.

## Initial 2025-26 Audit

The initial audit evaluates the frozen 2024-25 HPM context state on all 39,918
realized 2025-26 regular-season stints (122,886 estimated possessions). It
uses the HPM target-season regularization, \(\lambda=0.03\), for the player
projection.

| Diagnostic | Result |
| --- | ---: |
| Context variance represented by player reattribution (weighted \(R^2\)) | 68.76% |
| Residual-synergy RMSE | 2.352 |
| Residual-synergy MAE | 1.864 |
| 5th / 50th / 95th residual percentile | -3.99 / -0.00 / +3.94 |
| Correlation: frozen context vs. player projection | 0.859 |

This is strong agreement, but it is not an argument that residual synergy is
zero. Roughly one-third of the context variance remains outside the additive
player projection, and individual stints can still retain several net-rating
points per 100 of configuration-specific signal. The fixed 2024-25 context
state is forward-looking; the 2025-26 player reattribution is retrospective
and therefore not a new forecast evaluation.

The reattribution moves several high-context players upward. The examples below
are descriptive, not a second predictive leaderboard.

| Player | HPM player rating | Context reattribution | CR-RAPM |
| --- | ---: | ---: | ---: |
| Shai Gilgeous-Alexander | +6.78 | +1.94 | +8.72 |
| Nikola Jokić | +5.24 | +2.02 | +7.26 |
| Stephen Curry | +1.01 | +1.25 | +2.26 |
| Draymond Green | -2.87 | +1.00 | -1.88 |
| James Harden | -1.34 | +1.69 | +0.35 |

Curry is the useful motivating example: CR-RAPM finds a meaningful recurring
player-associated share of the frozen HPM context term, but does not make the
entire context effect his. That is the intended separation between portable
player reattribution and residual lineup-specific interaction.

## Audit Outputs

Run the [build guide](../guides/build-context-reattributed-rapm-audit.md) to
create an immutable audit artifact. It contains:

- `player_context_reattribution.parquet`: HPM rating, context reattribution,
  CR-RAPM, and target-season on-court possessions for every identified player.
- `stint_context_ledger.parquet`: frozen HPM context, player-projected context,
  and residual synergy for each realized stint.
- `lineup_residual_synergy.parquet`: possession-weighted exact lineup-pair
  residuals above the configured exposure threshold.
- `audit_metrics.parquet`: projection \(R^2\), RMSE, residual quantiles, and
  coverage.

Published artifact:

```text
artifacts/models/context_reattributed_rapm_audit/2025-26/
context-reattributed-rapm-audit-2025-26-20260812T053955Z-a045caca
```
