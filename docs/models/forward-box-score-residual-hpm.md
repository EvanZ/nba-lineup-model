---
last_updated: "2026-08-12"
---

# Forward Box-Score Residual HPM

This branch adds a true individual box-score prior inside the existing HPM state,
instead of replacing HPM with a standalone box-score estimate.

For target season \(t\), HPM supplies the strictly forward prior
\(\mu^{\mathrm{HPM}}_{i,t}\). A possession-weighted ridge model learns the
completed historical residual

\[
r_{i,t}=\operatorname{HPM}_{i,t}-\mu^{\mathrm{HPM}}_{i,t}
\]

from the player's immediately prior-season possession-normalized box profile.
The next prior becomes

\[
\mu^{\mathrm{box}}_{i,t}=\mu^{\mathrm{HPM}}_{i,t}+\widehat r_{i,t}.
\]

The residual excludes prior RAPM, age, experience, draft, physical dimensions,
and position because HPM already contains those signals. It uses lagged on-court
possession exposure, scoring volume, assists, turnovers, offensive and defensive
rebounds, steals, blocks, fouls, and stabilized shooting rates.

## Boundary And Cold Starts

Each annual ridge fit uses only earlier completed HPM seasons. Its penalty is
chosen by expanding completed-season folds. No target-season box score, RAPM,
or lineup outcome enters the target prior.

Only returning players with a complete prior NBA box profile receive the
residual adjustment. Cold starts remain on HPM's exposure-gated
replacement/draft branch. The adjusted prior is then possession-centered using
the preceding season's exposure.

## Rebuilt Frozen 2025-26 Result

The recovered-history rerun is:

```text
artifacts/models/forward_box_score_residual_hpm/2025-26/
forward-box-score-residual-hpm-2025-26-20260813T043212Z-7f53370b
```

Compared with the matched rebuilt Value-Conditioned Aging HPM run, the
box-score residual is effectively tied but slightly worse on the primary
lineup-prediction contract: regular possession RMSE is **1.198739** versus
**1.198736**, eligible game-margin RMSE is **14.3256** versus **14.3214**, and
playoff game-margin RMSE is **16.4632** versus **16.4069**.

It modestly improves aggregate team estimates: regular-season team NetRtg RMSE
is **3.5825** versus **3.5858**, and Pythagorean-win RMSE is **7.6174** versus
**7.6300**. The differences are too small to promote it over the parent HPM,
but they retain evidence that a strictly lagged box profile supplies some
team-level signal.

For 2025-26, the residual moves Stephen Curry's frozen prior from **+0.97** to
**+1.32**, and his retrospective rating from **+0.93** (111th) to **+1.18**
(98th). This is a learned box-score residual, not a direct reattribution of
contextual value.
