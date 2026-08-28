import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
    LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
    side_context_feature_columns,
)
from nba_lineup_model.modeling.matchup_contextual import (
    fit_block_penalized_linear_ridge_matchup_contextual_model,
    fit_linear_ridge_matchup_contextual_model,
)
from nba_lineup_model.modeling import nail_v1213_block_penalty_development
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    _serialize_nested_metadata,
)


def test_nested_metadata_serializes_mixed_numpy_arrays_and_lists(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "season": ["2022-23", "2023-24"],
            "aging_training_target_seasons": [
                np.array(["2020-21", "2021-22"]),
                ["2021-22", "2022-23"],
            ],
        }
    )

    serialized = _serialize_nested_metadata(frame)
    serialized.to_parquet(tmp_path / "metadata.parquet", index=False)

    assert serialized["aging_training_target_seasons"].map(type).eq(str).all()


def test_equal_block_penalties_match_shared_alpha_predictions() -> None:
    rng = np.random.default_rng(7)
    columns = side_context_feature_columns(CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE)
    home = pd.DataFrame(rng.normal(size=(12, len(columns))), columns=columns)
    away = pd.DataFrame(rng.normal(size=(12, len(columns))), columns=columns)
    target = rng.normal(size=12)
    weights = rng.uniform(1, 4, size=12)
    shared = fit_linear_ridge_matchup_contextual_model(
        home, away, target, weights, alpha=10_000.0,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
    )
    blocked = fit_block_penalized_linear_ridge_matchup_contextual_model(
        home, away, target, weights, alpha=10_000.0,
        additive_alpha=10_000.0, nonadditive_alpha=10_000.0,
        additive_features=LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
        nonadditive_features=("top_two_assists", "usage_concentration"),
        feature_set=CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
    )
    np.testing.assert_allclose(
        shared.predict_side_pairs(home, away),
        blocked.predict_side_pairs(home, away),
        atol=1e-12,
    )


def test_additive_only_control_structurally_excludes_nonadditive_terms() -> None:
    rng = np.random.default_rng(17)
    columns = side_context_feature_columns(CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE)
    home = pd.DataFrame(rng.normal(size=(12, len(columns))), columns=columns)
    away = pd.DataFrame(rng.normal(size=(12, len(columns))), columns=columns)
    target = rng.normal(size=12)
    weights = rng.uniform(1, 4, size=12)

    model = fit_block_penalized_linear_ridge_matchup_contextual_model(
        home,
        away,
        target,
        weights,
        alpha=10_000.0,
        additive_alpha=10_000.0,
        nonadditive_alpha=None,
        additive_features=LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
        nonadditive_features=("top_two_assists", "usage_concentration"),
        feature_set=CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
    )

    assert model.block_penalties == {"additive": 10_000.0, "nonadditive": None}
    assert model.pipeline.named_steps["ridge"].coef_.shape == (
        len(LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES),
    )
    assert np.isfinite(model.predict_side_pairs(home, away)).all()


def test_block_penalty_development_replay_scores_possessions(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_backtest(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        nail_v1213_block_penalty_development,
        "run_frozen_multiseason_backtest",
        fake_backtest,
    )
    nail_v1213_block_penalty_development.run_development_backtest()

    assert nail_v1213_block_penalty_development.DEVELOPMENT_SEASONS == (
        "2019-20",
        "2020-21",
        "2021-22",
        "2022-23",
    )
    assert "score_possessions" not in captured
