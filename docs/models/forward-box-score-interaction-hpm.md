---
last_updated: "2026-08-12"
---

# Forward Box-Score Interaction HPM

This is the first controlled extension of [Forward Box-Score Residual HPM](forward-box-score-residual-hpm.md).
It retains the additive lagged box-score residual and adds six predeclared,
interpretable player-profile interactions. The base HPM player prior,
exposure-gated cold-start behavior, recursive portable context, and frozen
evaluation boundary are unchanged.

For returning player \(i\) in target season \(t\), the model predicts the
completed HPM residual using prior-season box features \(z_{i,t-1}\) and their
products:

\[
\mu^{\mathrm{box-int}}_{i,t}=
\mu^{\mathrm{HPM}}_{i,t}+\widehat r(z_{i,t-1}, z_{i,t-1}\otimes z_{i,t-1}).
\]

Every numeric feature, including each product, is median-imputed and
standardized inside the possession-weighted annual ridge pipeline. Ridge
regularization therefore decides whether an interaction retains incremental
weight beyond its two constituent additive terms.

## Interaction Set

| Interaction | Construction | Intended player-profile signal |
| --- | --- | --- |
| Usage x assists | \((FGA + 0.44 FTA + TOV) \times AST\) | High-volume creation |
| Usage x turnovers | \((FGA + 0.44 FTA + TOV) \times TOV\) | Burden and ball-security cost |
| 3PA x eFG% | \(3PA \times eFG\%\) | High-volume efficient shooting |
| FGA x FTA | \(FGA \times FTA\) | Scoring and free-throw pressure |
| OREB x DREB | \(OREB \times DREB\) | Rebounding profile |
| STL x BLK | \(STL \times BLK\) | Defensive-event profile |

All counts are on-court events per 100 on-court possessions. Shooting uses the
same prior-season empirically stabilized rate as the additive residual model.
The set is intentionally small: it tests concrete archetype hypotheses without
creating a large, unstable all-pairs feature search.

## Boundary And Cold Starts

The interaction branch learns each season only from earlier completed HPM
transitions, with regularization selected by expanding completed-season folds.
It cannot observe target-season box scores, RAPM, lineup outcomes, or playoff
outcomes when constructing a target-season prior.

Returning players with a complete prior-season box profile receive the
adjustment. Rookies and other cold starts remain on HPM's established
exposure-gated replacement/draft branch.

## Frozen 2025-26 Result

The complete recursive run is:

```text
artifacts/models/forward_box_score_interaction_hpm/2025-26/
forward-box-score-interaction-hpm-2025-26-20260812T203047Z-e2680654
```

The interactions did not improve the primary frozen contract. Relative to the
additive box-score residual HPM, regular possession RMSE is essentially tied
but marginally worse (**1.198771** versus **1.198770**), eligible game-margin
RMSE rises (**14.3577** versus **14.3548**), and full-game margin RMSE rises
(**14.6309** versus **14.6268**). The playoff game-margin result also rises,
from **16.5027** to **16.5138**.

There is one narrow aggregate improvement: Pythagorean-win RMSE is **7.9928**,
slightly better than the additive residual's **7.9943**. It is not sufficient
to promote this branch over HPM or the additive residual, particularly because
the full-game winner accuracy remains **68.05%** and team NetRtg RMSE rises
from **3.7536** to **3.7595**.

For the frozen 2025-26 prior, the selected interaction ridge penalty is
**1.0**, based on 27 expanding completed-season folds. The final standardized
interaction coefficients are modest: FGA x FTA is **+0.0196**, steals x blocks
is **+0.0163**, OREB x DREB is **+0.0075**, 3PA x eFG% is **-0.0051**,
usage x assists is **+0.0046**, and usage x turnovers is **+0.0002**. The
products are therefore available to the model, but do not carry enough stable
incremental signal to beat the simpler prior on the primary metrics.
