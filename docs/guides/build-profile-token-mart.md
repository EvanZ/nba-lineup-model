---
last_updated: "2026-08-13"
---

# Build Profile Token Mart

Build the player-season panel, shot taxonomy mart, and no-context forward RAPM
state first. Then materialize the profile-token contract:

~~~bash
uv run nba-build-profile-token-mart
~~~

The default source is the latest completed run under:

~~~text
artifacts/models/forward_centered_value_conditioned_aging_no_context_rapm/2025-26/
~~~

To use an explicit immutable run instead:

~~~bash
uv run nba-build-profile-token-mart \
  --prior-run-root artifacts/models/forward_centered_value_conditioned_aging_no_context_rapm/2025-26/<run-id>
~~~

Validate the completed output:

~~~bash
uv run python -c "from nba_lineup_model.modeling.profile_token_mart import validate_profile_token_mart; validate_profile_token_mart('data/analytical/profile_tokens')"
~~~

The builder is local-only. It makes no NBA API requests and atomically replaces
the output only after hashes, rows, feature completeness, and source-season
boundaries validate.
