---
last_updated: "2026-08-15"
---

# Train Additive Profile-Prior RAPM

Run the controlled no-context experiment through the frozen target season:

```bash
uv run nba-train-forward-additive-profile-prior-rapm --through-season 2025-26
```

The command builds the same lagged player profiles used by the HPM x3 additive
context terms, moves them into a residual player-prior ridge model, and fits no
lineup context state. It writes the usual forward RAPM artifacts plus annual
box-score residual models and regularization selections under:

```text
artifacts/models/forward_additive_profile_prior_rapm/2025-26/
```

Evaluate the immutable 2023-24 through 2025-26 states without any replay-season
refit:

```bash
uv run nba-evaluate-additive-profile-prior-rapm
```

The focused report is published under
`artifacts/models/analysis/additive_profile_prior_frozen/` and does not replace
the shared leaderboard's latest-report pointer.

Run 10,000 paired, season-stratified game-block bootstrap draws against both
the complete no-context player-prior control and full linear HPM x3 context:

```bash
uv run nba-bootstrap-additive-profile-prior-rapm
```
