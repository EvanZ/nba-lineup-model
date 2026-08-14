---
last_updated: "2026-08-13"
---

# Forward RAPM Memory Baselines

These are the rigorous RAPM-only controls for the frozen evaluation contract.
They are rolling, player-prior-centered RAPM models rather than pooled
zero-centered windows. No age, draft, physical, box-score, exposure-gate, or
lineup-context input is included.

For a completed season \(t\), let \(r_{i,t}\) be player \(i\)'s annual RAPM
estimate and \(n_{i,t}\) their on-court possessions. Before season \(t+1\):

\[
\mu^{(1)}_{i,t+1} = r_{i,t}
\]

for the one-season model, while the three-season model uses the player-specific
possession-weighted memory

\[
\mu^{(3)}_{i,t+1} =
\frac{\sum_{k=0}^{2} n_{i,t-k}r_{i,t-k}}
     {\sum_{k=0}^{2} n_{i,t-k}}.
\]

The sums include only seasons in which the player appeared. A player absent
from the remembered state is scored with a cold-start prior of zero. Each
completed season selects its own ridge penalty with chronological folds, fits
on all of that season's regular-season stints, and becomes input only to later
seasons.

## Evaluation Contract

The initial rigorous pool freezes three target seasons: 2023-24, 2024-25, and
2025-26. For each target, all player priors, source home-court effect, and
source scoring mean are fixed before opening night. Target-season realized
lineup allocation is used as an oracle solely to make the lineup forecast
comparable across models. Target outcomes are not used until after that frozen
forecast is written, when they may update the next season's prior.

All three regular seasons and their matching postseason possession partitions
are available. Playoff results are scored from the frozen regular-season
boundary and never enter training.

<!-- forward-rapm-memory-results:start -->
## Results

Artifact: `artifacts/models/forward_rapm_memory_prior_baselines/2023-24_to_2025-26/forward-rapm-memory-baselines-2023-24-to-2025-26-20260813T210525Z-190bfd3b`.

The regular-season pool combines 2023-24, 2024-25, and 2025-26. Lower is better except winner accuracy.

| Model | Possession RMSE | Eligible game RMSE | Full-game RMSE | Winner accuracy | Team NetRtg RMSE | Pythagorean-win RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward 1-year RAPM-prior baseline | **1.198196** | **14.5265** | **14.8154** | **65.39%** | **4.2374** | **9.1066** |
| Forward 3-year RAPM-prior baseline | 1.198246 | 14.6559 | 14.9714 | 64.68% | 4.5326 | 9.7789 |

### Per-Season Regular Results

| Season | Model | Possession RMSE | Eligible game RMSE |
| --- | --- | ---: | ---: |
| 2023-24 | Forward 1-year RAPM-prior baseline | 1.193190 | 14.0586 |
| 2023-24 | Forward 3-year RAPM-prior baseline | 1.193288 | 14.3123 |
| 2024-25 | Forward 1-year RAPM-prior baseline | 1.202214 | 14.5310 |
| 2024-25 | Forward 3-year RAPM-prior baseline | 1.202242 | 14.6348 |
| 2025-26 | Forward 1-year RAPM-prior baseline | 1.198994 | 14.9019 |
| 2025-26 | Forward 3-year RAPM-prior baseline | 1.199021 | 14.9537 |

### Frozen Playoff Check

Each playoff cohort is scored from the matching frozen pre-season state; playoff outcomes never enter the fit.

| Season | Model | Possession RMSE | Eligible game RMSE |
| --- | --- | ---: | ---: |
| 2023-24 | Forward 1-year RAPM-prior baseline | 1.191977 | 17.5914 |
| 2023-24 | Forward 3-year RAPM-prior baseline | 1.192143 | 17.7534 |
| 2024-25 | Forward 1-year RAPM-prior baseline | 1.194540 | 16.8274 |
| 2024-25 | Forward 3-year RAPM-prior baseline | 1.194630 | 17.1473 |
| 2025-26 | Forward 1-year RAPM-prior baseline | 1.192825 | 17.4317 |
| 2025-26 | Forward 3-year RAPM-prior baseline | 1.192365 | 16.6833 |
<!-- forward-rapm-memory-results:end -->

