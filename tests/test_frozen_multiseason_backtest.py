from __future__ import annotations

import pandas as pd

import nba_lineup_model.modeling.frozen_multiseason_backtest as backtest


class _Context:
    def __init__(self, season: str) -> None:
        self.season = season

    def predict_lineups(self, home: object, away: object, profiles: object) -> list[float]:
        return [0.0] * len(home)  # type: ignore[arg-type]


def test_replay_uses_target_prior_and_immediately_prior_context(monkeypatch, tmp_path) -> None:
    target = "2024-25"
    source = "2023-24"
    state = backtest._RecursiveState(
        candidate=backtest.BacktestModel("test", "Test"),
        run_dir=tmp_path,
        run_id="test-run",
        priors=pd.DataFrame(
            {
                "season": [source, target],
                "player_id": [1, 1],
                "prior_rapm": [1.0, 2.0],
            }
        ),
        coefficients=pd.DataFrame(
            {"season": [source], "player_id": [1], "rapm": [3.0]}
        ),
        context_models={source: _Context(source)},  # type: ignore[arg-type]
    )
    observed: dict[str, object] = {}
    stints = pd.DataFrame({"home_player_ids": [[1]], "away_player_ids": [[2]]})

    monkeypatch.setattr(backtest, "read_rapm_stints", lambda *args, **kwargs: stints)
    monkeypatch.setattr(
        backtest,
        "build_contextual_player_profiles",
        lambda *args, **kwargs: pd.DataFrame({"player_id": [1, 2]}),
    )

    monkeypatch.setattr(backtest, "_recover_home_intercept", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        backtest,
        "_read_regular_possessions",
        lambda *args, **kwargs: (pd.DataFrame({"target_offense_margin": [0.0]}), tmp_path),
    )
    monkeypatch.setattr(backtest, "read_neural_possessions", lambda *args, **kwargs: pd.DataFrame())

    def score_possessions(*args: object, **kwargs: object) -> pd.DataFrame:
        observed.update(kwargs)
        return pd.DataFrame({"cohort": ["regular_season"]})

    monkeypatch.setattr(backtest, "_score_possessions", score_possessions)
    monkeypatch.setattr(
        backtest,
        "_contextual_stint_predictions",
        lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        backtest,
        "_historical_team_seasons",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        backtest,
        "fit_pythagorean_win_model",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        backtest,
        "_team_win_evaluation",
        lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        backtest,
        "score_possession_cohort",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        backtest,
        "_game_prediction_frame",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        backtest,
        "_team_net_rating_metrics",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    backtest._replay_regular_target_season(
        target,
        state=state,
        panel=pd.DataFrame(),
        exposure_cohort=pd.DataFrame(),
        analytical_dir=tmp_path,
        curated_dir=tmp_path,
    )

    assert getattr(observed["context_predictor"], "__self__", None) is state.context_models[source]
    assert observed["priors"].equals(
        state.priors.loc[state.priors["season"].eq(target)].rename(
            columns={"prior_rapm": "prior_rapm_mean"}
        )[["player_id", "prior_rapm_mean"]]
    )

    def fail_neural_read(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("Possession mart should not be read")

    monkeypatch.setattr(backtest, "read_neural_possessions", fail_neural_read)
    game_only = backtest._replay_regular_target_season(
        target,
        state=state,
        panel=pd.DataFrame(),
        exposure_cohort=pd.DataFrame(),
        analytical_dir=tmp_path,
        curated_dir=tmp_path,
        score_possessions=False,
    )
    assert game_only["possession_predictions"].empty


def test_validated_seasons_require_chronological_distinct_targets() -> None:
    assert backtest._validated_seasons(["2023-24", "2024-25"]) == ("2023-24", "2024-25")
    for invalid in (("2024-25", "2023-24"), ("2024-25", "2024-25")):
        try:
            backtest._validated_seasons(invalid)
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid targets to fail: {invalid}")
