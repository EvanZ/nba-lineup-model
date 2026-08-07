# Models

The modeling program starts with transparent predictive baselines before adding
player priors, separate offensive and defensive effects, or nonlinear lineup
interactions.

## Baseline ladder

| Model | Information available |
| --- | --- |
| Mean | Training-window average home net rating |
| Team | Signed home and away team identities |
| RAPM | Signed identities of all ten players |
| Bayesian RAPM | RAPM posterior uncertainty under an explicit Gaussian model |
| RAPM aging model | Forward player-season priors from prior RAPM and age |
| Age-informed prior RAPM | Lineup RAPM centered on the frozen aging forecast |
| Forward RAPM calibration | Frozen lagged RAPM mapped to team wins using realized lineup seconds |
| Frozen lagged RAPM | Preseason 2024-25 player values scored on all 2025-26 oracle lineups |
| Frozen aging prior | Preseason age, experience, prior RAPM, draft, and physical-profile values scored on the same oracle lineups |
| Frozen O/D RAPM | Forward offense and defense player values scored on the same oracle lineups |
| Box-score RAPM prior | Returning-player forecast from lagged RAPM and possession-native box profile |
| Draft-informed cold starts | First-NBA-season RAPM diagnostic using draft profile fields only |
| Replacement-level exposure study | Rejected coefficient-average diagnostic for the low-exposure player pool |
| Pooled replacement-token RAPM | Retrospective shared-token estimate for the low-exposure player pool |
| Cold-start exposure gate | First-NBA-season preseason probability of entering the low-exposure pool |
| Exposure-gated cold-start prior | Continuous draft-rate/replacement blend for first-year players |
| Exposure-gated O/D cold starts | Separate O/D draft-rate/replacement blend for first-year players |
| Forward exposure-gated RAPM | Recursive regular-only preseason RAPM state with cold starts |
| Student-t forward RAPM | Robust stint-error likelihood for the same recursive forward state |
| Student-t talent-prior RAPM | Heavy-tailed player adjustments with Gaussian stint errors |
| Forward contextual RAPM | Recursive RAPM with the prior season's contextual offset |
| Additive neural RAPM | Possession-level signed scalar player embeddings |
| Deep Sets | Nonlinear permutation-invariant lineup aggregation |
| CatBoost | Categorical player states and boosted-tree interactions |
| RAPM + Transformer | Frozen ridge prediction plus contextual player-player attention residual |

The first four use the same regular-season stint target, chronological game
splits, possession weights, and test window. The first three compare predictive
information sets. Bayesian RAPM deliberately retains the ridge point estimate
and adds coefficient, rank, and predictive uncertainty. Neural models move to
single-lineup possession rows while retaining chronological game boundaries.
RAPM + Transformer reconnects the two samples by converting a leakage-safe,
stint-weighted ridge prediction into the possession target and learning only
its nonlinear residual.

See [Baseline methodology](baselines.md) for the model contract, the
[2025-26 RAPM case study](2025-26-rapm-case-study.md) for a worked diagnostic
review, the [Bayesian RAPM methodology](bayesian-rapm.md) and
[2025-26 Bayesian case study](2025-26-bayesian-rapm-case-study.md) for the
probabilistic baseline, and [Promoted rankings](rankings.md) for reviewed
public releases. [Neural Networks](neural-networks.md) defines the staged
additive, Deep Sets, and RAPM + Transformer program. [Tree Models](tree-models.md)
defines the orthogonal CatBoost baseline. [Leaderboard](leaderboard.md) defines
the shared regular-holdout and playoff metrics and maintains the cross-model
scoreboard. [Forward RAPM Calibration](forward-calibration.md) maps frozen
player priors to regular-season team wins. The [Frozen Preseason
Leaderboard](preseason-leaderboard.md) holds out the complete target regular
season and playoffs without a player refit. [RAPM Aging Model](aging-model.md) defines the first temporal player
prior. [Draft-Informed Cold Starts](draft-prior.md) is an interpretable
first-NBA-season diagnostic; it is not yet eligible for the Frozen Preseason
Leaderboard. The [Replacement-Level Exposure Study](replacement-level.md)
records why separately ridged low-exposure coefficients fail as a replacement
estimate. [Pooled Replacement-Token RAPM](replacement-token.md) estimates that
group directly before a cold-start gate is introduced. The [Cold-Start Exposure
Gate](cold-start-exposure.md) provides a calibrated first-year-only probability
for a replacement-token blend. The [Exposure-Gated Cold-Start Prior](exposure-gated-cold-start.md)
uses that blend in the frozen preseason evaluation. The [Modeling Roadmap](roadmap.md)
records the approved joint dynamic RAPM extension.

[Exposure-Gated O/D Cold Starts](exposure-gated-offense-defense.md) applies
the same cold-start logic without collapsing offense and defense into a single
replacement number.

[Forward Exposure-Gated RAPM](forward-exposure-gated-rapm.md) applies that
one-number cold-start prior recursively across completed seasons.

[Student-t Forward RAPM](student-t-forward-rapm.md) holds that state and its
lambda schedule fixed while replacing the Gaussian stint-error likelihood with
a robust Student-t alternative.

[Student-t Talent-Prior RAPM](student-t-talent-forward-rapm.md) reverses that
ablation: it keeps Gaussian stint errors and uses a heavy-tailed Student-t
prior for player departures from the forward state.

[Forward Contextual RAPM](forward-contextual-rapm.md) carries the completed
lineup-composition state into the next season's RAPM target before updating its
player coefficients.
