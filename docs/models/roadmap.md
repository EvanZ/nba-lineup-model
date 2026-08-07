---
last_updated: "2026-08-05"
---

# Modeling Roadmap

This page records approved model designs that should remain visible but are not
part of the current baseline implementation.

## TODO: Joint Dynamic RAPM

Stack all seasons into one sparse observation matrix with a separate
`(player_id, season)` coefficient:

\[
y_{s,t}=X_{s,t}\beta_t+\epsilon_{s,t}.
\]

Connect consecutive player-season coefficients with an aging transition:

\[
\beta_{i,t}
=
\beta_{i,t-1}
+
g(a_{i,t};\gamma)
+
\eta_{i,t}.
\]

The corresponding penalized objective is:

\[
\sum_{s,t} w_{s,t}(y_{s,t}-X_{s,t}\beta_t)^2
+
\lambda_0\sum_{i,t}\beta_{i,t}^2
+
\lambda_T\sum_{i,t}
\left[
  \beta_{i,t}-\beta_{i,t-1}-g(a_{i,t};\gamma)
\right]^2.
\]

Temporal differences can be encoded as sparse pseudo-observation rows, making
the complete model one augmented sparse regression. A Bayesian state-space
version would express the same structure as a latent player trajectory.

### Required outputs

- **Filtered estimates** use only seasons available through each target date
  and are eligible for prediction, Leaderboard evaluation, and Transformer
  features.
- **Smoothed estimates** may borrow information from later seasons and must be
  published separately as retrospective evaluations.

### Acceptance criteria

- reproduce the standalone aging model's forward folds;
- compare against persistence, two-stage aging priors, and zero-centered RAPM;
- retain cold starts and retired players without inventing observed seasons;
- report temporal penalty sensitivity and trajectory uncertainty;
- prevent future-season outcomes from entering filtered estimates;
- expose the same player-season side-information keys used by neural tokens.

The two-stage forward aging model is implemented first because it makes the
aging curve, target labels, uncertainty proxy, and leakage boundary directly
inspectable.

## Exposure-Gated Cold Starts

The [pooled replacement-token study](replacement-token.md) establishes a
retrospective low-exposure group effect. The [cold-start exposure gate](cold-start-exposure.md)
estimates a preseason-only rotation probability for first-NBA-season players,
and the [exposure-gated cold-start prior](exposure-gated-cold-start.md) now
blends that token with the profile rate prior:

\[
\widehat{R}^{cold}_i =
(1-p^{rotation}_i)R^{replacement}
+ p^{rotation}_i\widehat{R}^{profile}_i.
\]

The remaining limitation is candidate population: the gate conditions on a
player appearing in the NBA data at least once. Add drafted players with zero
NBA possessions before interpreting it as a full draft-to-rotation model.
Then repeat the frozen comparison across future-season holdouts rather than
promoting a single-season result.

## O/D Exposure-Gated Cold Starts

The initial [exposure-gated O/D cold-start model](exposure-gated-offense-defense.md)
uses separate draft-rate and replacement-token estimates for offense and
defense, while sharing the preseason probability of low exposure. Its 2025-26
run is temporally clean: each component ends with 2024-25. The next extension
is a rolling historical evaluator that re-estimates the gate, the two draft
rate models, and the two replacement-token aggregates for every target season
using only earlier seasons.
