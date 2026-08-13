---
last_updated: "2026-08-12"
---

# Forward Context-Reattributed HPM

This is the predictive continuation of the
[Context-Reattributed RAPM audit](context-reattributed-rapm.md). A completed
season's frozen context term is projected onto players as \(X\gamma_t\), and
a constrained fraction is moved into the next-year player prior:

\[
\mu_{i,t+1}=\mu^{\mathrm{aging/cold}}_{i,t+1}+\rho\gamma_{i,t}.
\]

To avoid double counting, the same transferred component is removed from the
next season's context correction:

\[
C^{\mathrm{residual}}_t(A,B)=C_t(A,B)-\rho X\gamma_t.
\]

Thus the model retains a player-facing CR adjustment and residual portable
composition/matchup context. It is genuinely forward-looking: \(\gamma_t\)
uses only a context function fit on completed season \(t\), and affects only
season \(t+1\).

## Frozen 2025-26 Results

The first complete recursive candidate uses \(\rho=0.5\):

```text
artifacts/models/forward_context_reattributed_hpm/2025-26/
forward-context-reattributed-hpm-2025-26-20260812T072008Z-f68f69b1
```

It is a useful negative result. Compared with current Value-Conditioned Aging
HPM, the transfer worsens regular eligible-game margin RMSE from **14.3469** to
**14.3909**, team NetRtg RMSE from **3.7568** to **3.8791**, and Pythagorean
win RMSE from **8.0346** to **8.3649**. It also does not improve the separate
playoff evaluation. The model is therefore documented as an experimental
ablation, not promoted to the website or gold-standard rating.

The run still produces `season_context_reattributions.parquet` with annual
\(\gamma_t\) coefficients and `season_context_reattribution_metadata.parquet`
with annual projection fit diagnostics. This makes the next question precise:
whether a smaller, forward-selected \(\rho\) can retain any player-persistence
benefit without diluting the contextual state.

The smaller \(\rho=0.25\) candidate improves upon the half-transfer run:

```text
artifacts/models/forward_context_reattributed_hpm/2025-26/
forward-context-reattributed-hpm-2025-26-20260812T143351Z-991dad66
```

Its frozen regular-season possession RMSE is **1.198777** and eligible-game
margin RMSE is **14.3677**, versus **1.198794** and **14.3909** for
\(\rho=0.5\). It also improves team NetRtg RMSE from **3.8791** to **3.8211**,
Pythagorean-win RMSE from **8.3649** to **8.1949**, full-game margin RMSE from
**14.6861** to **14.6531**, and winner accuracy from **68.21%** to **68.37%**.
It remains behind current HPM (\(\rho=0\)) on these primary regular-season
measures, and its playoff eligible-game margin RMSE is slightly worse
(**16.4769** versus **16.4742**). The leaderboard therefore represents the
more competitive \(\rho=0.25\) candidate while retaining HPM as the leader.

The still smaller \(\rho=0.1\) run is the best transfer weight tested so far:

```text
artifacts/models/forward_context_reattributed_hpm/2025-26/
forward-context-reattributed-hpm-2025-26-20260812T163434Z-c15576de
```

It reduces regular eligible-game margin RMSE to **14.3543**, team NetRtg RMSE
to **3.7827**, Pythagorean-win RMSE to **8.0940**, and full-game margin RMSE to
**14.6344**. It also improves regular and playoff possession RMSE relative to
every larger transfer weight tested. It remains just behind HPM (\(\rho=0\))
on those error measures, so this remains an experimental branch, but the
frozen leaderboard and model tree now represent \(\rho=0.1\).

## Full-Transfer Endpoint

The other informative endpoint, \(\rho=1.0\), transfers all player-projectable
context into the next-year player prior and forwards only the residual context
state. Its completed run is:

```text
artifacts/models/forward_context_reattributed_hpm/2025-26/
forward-context-reattributed-hpm-2025-26-20260812T132121Z-917b17eb
```

It is decisively worse than both HPM (\(\rho=0\)) and the half-transfer
candidate. Frozen 2025-26 regular-season possession RMSE is **1.198826**,
eligible-game margin RMSE is **14.4337**, team NetRtg RMSE is **3.9742**, and
Pythagorean-win RMSE is **8.6687**. This rejects complete transfer: some
context is player-projectable, but treating all of it as portable player value
degrades the next-season forecast.
