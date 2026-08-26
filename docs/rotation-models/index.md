---
last_updated: "2026-08-25"
---

# Rotation Models

Rotation models forecast playing-time allocation rather than player value.
Their first target is a player's share of a team's total minutes in its next
game. This includes whether a player appears and how much the player is used.

The initial family member is deliberately parameter-free:

| Model | Forecast | Inputs |
| --- | --- | --- |
| [L1-MSP v0.0](l1-minute-share-persistence.md) | Next team-game minute share | That player's prior team-game minute share only |
| [L5-MMP v0.1](l5-median-minutes-persistence.md) | Next team-game minute share | Five prior team-game minute totals |

NAIL-RAPM remains a rating model. Rotation models will eventually provide the
player and five-man exposure forecasts needed to turn ratings into game and
season predictions.
