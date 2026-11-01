# NBA GESTALT operating contract

These instructions apply to every task in this repository.

## Execution environment

- Treat `gestalt-web-server:~/nba-lineup-model` as the development checkout.
- Implement, test, materialize data, and build releases on EC2. Do not use the
  laptop checkout as the working implementation environment.
- Keep large datasets and model artifacts on EC2/S3. Never copy them to the
  laptop without the user's explicit approval.

## Release workflow

- Deploy every site change to `gestalt-dev.toplines.app` first.
- Inspect the rendered dev page and validate the relevant API/data coverage
  before saying a change is ready.
- Do not promote or deploy to production without the user's explicit approval.
- Dev and production must use the same canonical artifact schemas and coverage
  checks; environment-specific pointers may differ.

## Correctness gates

- Prefer the existing canonical artifacts over reconstructing, approximating,
  or inventing substitute inputs.
- Validate complete game/date coverage, not a sample. A missing or quarantined
  game must be reported and repaired without silently removing other games.
- Do not claim success from unit tests alone. Verify the materialized EC2
  artifacts, the API response, and the rendered dev UI.
- Preserve immutable pregame projections and source provenance. Postgame
  retrospective artifacts must be separately labeled.
- NBA GESTALT basketball data must come from the official NBA sources selected
  by the repository architecture. Do not add ESPN data or identifiers.

## Collaboration

- Surface actual findings and failures to the user; do not bury them in files.
- Ask for input before making a new modeling assumption, changing a defined
  metric, or introducing a fallback that alters product behavior.
- When the user stops or redirects work, stop immediately.
