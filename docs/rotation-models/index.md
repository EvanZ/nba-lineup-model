---
last_updated: "2026-09-12"
---

# Rotation Models

Rotation models forecast playing-time allocation rather than player value.
Their first target is a player's share of a team's total minutes in its next
game. This includes whether a player appears and how much the player is used.

The initial family member is deliberately parameter-free:

| Model | Forecast | Inputs |
| --- | --- | --- |
| [L0-MSP v0.0](l0-opening-minute-share-persistence.md) | Opening team-game minute share | Prior final-game shares plus transaction-derived opening candidates |
| [L20-MSP v0.0](l20-preseason-minute-share-persistence.md) | Cumulative share across first 20 team games | Prior-season total minutes plus transaction-derived opening candidates |
| [L20-MSP v0.1](l20-gap-returner-minute-share.md) | Cumulative share across first 20 team games | L20-MSP plus a strictly preseason gap-returner prior |
| L20-NAIL-MSP v0.1 | Cumulative share across first 20 team games | L20-MSP plus completed prior-season NAIL-RAPM |
| [L20-NAIL-MSP v0.2](l20-preseason-nail-forecast-minute-share.md) | Cumulative share across first 20 team games | L20-MSP plus the exact target-season preseason NAIL forecast |
| [L20-NAIL-MSP v0.3](l20-source-aware-preseason-minute-share.md) | Cumulative share across first 20 team games | Source-aware role evidence for continuous players, gap returners, and cold starts |
| [Forward Availability](forward-availability.md) | Season medical-availability share | Promoted v0.2 component; age baseline plus exposure-shrunk player state |
| [Forward Conditional Minutes](forward-conditional-minutes.md) | Preseason minutes per available game | Promoted v0.3 component; returner state plus draft-informed and team-strength-aware rookie prior |
| [Forward Player Plackett-Luce Rotation v0.2](forward-plackett-luce-rotation.md) | Game-level conditional minute allocation | Global player residual over the fixed FCM v0.2 prior; follows in-season trades but has no selected cross-season carry |
| [Win Projections](win-projections.md) | Schedule-aware expected team wins | W0 minute-weighted NAIL, logistic game calibration, and Win-Loss Envelope |
| [AC-MSP v0.1](ac-minute-share-projection-v01.md) | Game-level conditional minute allocation | Observed availability mask; all regulation regular-season games |
| [L1-MSP v0.0](l1-minute-share-persistence.md) | Next team-game minute share | That player's prior team-game minute share only |
| [L5-MMP v0.1](l5-median-minutes-persistence.md) | Next team-game minute share | Five prior team-game minute totals |

NAIL-RAPM remains a rating model. Rotation models will eventually provide the
player and five-man exposure forecasts needed to turn ratings into game and
season predictions.
