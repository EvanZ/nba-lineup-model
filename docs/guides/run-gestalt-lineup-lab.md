---
last_updated: "2026-08-07"
---

# Run NBA GESTALT Locally

NBA GESTALT is a local interactive Lineup Lab built around the completed
2025-26 forward contextual RAPM state. The browser app and API are separate
processes during development.

## Install dependencies

From the repository root, install the Python environment:

```bash
uv sync --group dev
```

Install the frontend dependencies once:

```bash
cd apps/lineup-explorer
npm install
```

## Start the API

In a terminal at the repository root:

```bash
uv run nba-gestalt-api --port 8001
```

The API listens at `http://127.0.0.1:8001`. It loads the latest published
`forward_contextual_rapm/2025-26` artifact on its first request, then retains
the completed player coefficients, player profiles, and contextual spline
model in memory for the lifetime of the process. It does not read raw
possession data or retrain a model for each browser session.

## Start the Lineup Lab

In a second terminal:

```bash
cd apps/lineup-explorer
npm run dev
```

Open [http://127.0.0.1:5174](http://127.0.0.1:5174). Vite proxies `/api`
requests to the local API on port 8001, while the Zensical documentation site
continues to use port 8000.

## Current model contract

The initial Lineup Lab is intentionally defined as a retrospective 2025-26
explorer. A neutral-court matchup estimate is:

\[
\widehat{NR}_{A,B} =
\sum_{i \in A}\hat{r}_i -
\sum_{j \in B}\hat{r}_j +
g_{2025\text{-}26}(A,B).
\]

- `\hat{r}_i` is the completed 2025-26 additive forward contextual RAPM
  coefficient for player `i`.
- `g_{2025-26}(A,B)` is the completed contextual spline residual, evaluated
  from the profile inputs used when that model was fitted.
- No home-court term is applied in the Lineup Lab.

This is not a 2026-27 forecast and is not an observed lineup net rating. It is
a portable, retrospective model estimate for a user-selected matchup. The
context panel reports exact spline-Ridge contributions grouped by the original
context features; the contextual intercept is included in the displayed total
but is not assigned to an individual feature.

## API contract

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Loaded artifact identity and player count. |
| `GET /api/players?q=<query>` | Accent-insensitive player search over the current model universe. |
| `GET /api/players/{player_id}` | One display profile. |
| `GET /api/default-opponent` | Random five-player sample from the upper possession quartile. |
| `POST /api/matchups` | Score two distinct five-player units. |

`POST /api/matchups` accepts:

```json
{
  "unit_player_ids": [203999, 201939, 202691, 203110, 202710],
  "opponent_player_ids": [2544, 1629029, 1630162, 203507, 203954]
}
```

The response includes the additive margin, contextual adjustment, final net
rating estimate, player profiles, and feature-level contextual contributions.

## Verify changes

Run the backend contract tests and build the frontend before committing:

```bash
uv run ruff check src/nba_lineup_model/web_api tests/test_web_api.py
uv run pytest tests/test_web_api.py
cd apps/lineup-explorer && npm run build
```
