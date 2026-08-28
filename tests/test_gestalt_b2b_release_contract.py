from __future__ import annotations

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
)
from nba_lineup_model.web_api.inference import (
    MODEL_ARTIFACT,
    MODEL_NAME,
    _linear_x3_additive_feature_map,
)


def test_promoted_release_uses_the_standard_usage_artifact_contract() -> None:
    mapping = _linear_x3_additive_feature_map(CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE)

    assert MODEL_ARTIFACT == "forward_nail_rapm_v1212_residualized_lambda"
    assert MODEL_NAME == MODEL_ARTIFACT
    assert mapping["usage_pct"] == "usage_pct"
    assert "usage_per_100" not in mapping
