---
last_updated: "2026-09-08"
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
| [Forward Availability v0.1](forward-availability-v01.md) | Season medical-availability share | Initial benchmark; superseded by v0.2 |
| [Forward Availability v0.2](forward-availability-v02.md) | Season medical-availability share | Promoted availability component; paired with Forward Conditional Minutes v0.1 in the preseason projection stack |
| [Forward Conditional Minutes v0.1](forward-conditional-minutes-v01.md) | Preseason minutes per available game | Promoted conditional-minutes component; paired with Forward Availability v0.2 and top-15 roster normalization |
| [AC-MSP v0.1](ac-minute-share-projection-v01.md) | Game-level conditional minute allocation | Observed availability mask; all regulation regular-season games |
| [L1-MSP v0.0](l1-minute-share-persistence.md) | Next team-game minute share | That player's prior team-game minute share only |
| [L5-MMP v0.1](l5-median-minutes-persistence.md) | Next team-game minute share | Five prior team-game minute totals |

NAIL-RAPM remains a rating model. Rotation models will eventually provide the
player and five-man exposure forecasts needed to turn ratings into game and
season predictions.
