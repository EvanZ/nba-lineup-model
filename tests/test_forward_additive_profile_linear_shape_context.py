from __future__ import annotations

import pandas as pd

import nba_lineup_model.modeling.forward_additive_profile_linear_shape_context as additive_shape
from nba_lineup_model.modeling.forward_box_score_hpm import ForwardBoxScoreResidualPriorBuilder


def test_wrapper_combines_additive_player_prior_with_shape_only_context(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame({"season": ["2020-21"], "player_id": [1]}).to_parquet(panel_path, index=False)

    monkeypatch.setattr(
        additive_shape,
        "build_additive_profile_prior_features",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        additive_shape,
        "train_forward_portable_matchup_contextual_rapm",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    additive_shape.train_forward_additive_profile_linear_shape_context(
        player_season_panel_path=panel_path
    )

    builder = captured["player_prior_builder"]
    assert captured["model_name"] == additive_shape.MODEL_NAME
    assert captured["context_feature_set"] == "x3_nonadditive_shape_context"
    assert isinstance(builder, ForwardBoxScoreResidualPriorBuilder)
    assert builder.feature_columns == additive_shape.ADDITIVE_PROFILE_PRIOR_COLUMNS
