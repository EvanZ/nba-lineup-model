from __future__ import annotations

import pandas as pd

import nba_lineup_model.modeling.forward_additive_profile_prior_rapm as additive_prior
from nba_lineup_model.modeling.forward_box_score_hpm import ForwardBoxScoreResidualPriorBuilder


def test_wrapper_moves_only_additive_hpm_profile_terms_into_prior(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame({"season": ["2020-21"], "player_id": [1]}).to_parquet(panel_path, index=False)

    monkeypatch.setattr(
        additive_prior,
        "build_additive_profile_prior_features",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        additive_prior,
        "train_forward_portable_matchup_contextual_rapm",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    additive_prior.train_forward_additive_profile_prior_rapm(player_season_panel_path=panel_path)

    builder = captured["player_prior_builder"]
    assert captured["use_context"] is False
    assert captured["model_name"] == additive_prior.MODEL_NAME
    assert isinstance(builder, ForwardBoxScoreResidualPriorBuilder)
    assert builder.feature_columns == additive_prior.ADDITIVE_PROFILE_PRIOR_COLUMNS


def test_additive_profile_contract_excludes_nonadditive_hpm_terms() -> None:
    assert additive_prior.ADDITIVE_PROFILE_COLUMNS == (
        "three_pa_per_100",
        "three_pm_per_100",
        "assists_per_100",
        "turnovers_per_100",
        "usage_per_100",
        "offensive_rebound_pct",
        "steals_per_100",
        "blocks_per_100",
    )
