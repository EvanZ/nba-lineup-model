from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm as portable
from nba_lineup_model.modeling.prior_rapm import PRIOR_MEAN_COLUMN, ForwardLaggedRapmSeason


@dataclass(frozen=True)
class _Context:
    name: str
    reference_features: pd.DataFrame
    reference_weights: np.ndarray

    def predict_lineups(self, home: object, away: object, profiles: object) -> np.ndarray:
        return np.zeros(len(home), dtype=float)  # type: ignore[arg-type]


def test_frozen_evaluation_uses_previous_season_portable_matchup_state(
    monkeypatch, tmp_path: Path
) -> None:
    """The target-season C state cannot score its own frozen holdout."""

    target = "2025-26"
    source = "2024-25"
    initial = "2023-24"
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    pd.DataFrame(
        {
            "season": [initial, source, target],
            "selected_lambda": [1.0, 1.0, 1.0],
        }
    ).to_parquet(reference_root / "historical_player_coefficients.parquet", index=False)
    panel_path = tmp_path / "player_seasons.parquet"
    pd.DataFrame({"season": [initial, source, target], "player_id": [1, 1, 1]}).to_parquet(
        panel_path, index=False
    )
    stints = pd.DataFrame(
        {
            "home_player_ids": [[1]],
            "away_player_ids": [[2]],
            "target_home_net_rating": [2.0],
            "possessions": [10.0],
            "game_time_utc": ["2025-01-01T00:00:00Z"],
            "game_id": ["001"],
            "stint_index": [0],
        }
    )
    contexts = {
        source: _Context(source, pd.DataFrame({"feature": [1.0]}), np.array([1.0])),
        target: _Context(target, pd.DataFrame({"feature": [1.0]}), np.array([1.0])),
    }
    observed: dict[str, object] = {}

    monkeypatch.setattr(portable, "HISTORICAL_SEASONS", (initial, source))
    monkeypatch.setattr(portable, "_latest_run", lambda _: reference_root)
    monkeypatch.setattr(
        portable, "prepare_player_exposure_cohort", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(portable, "read_rapm_stints", lambda *args, **kwargs: stints.copy())
    playoff_seasons: list[str] = []

    def available_playoffs(season: str) -> tuple[str, ...]:
        playoff_seasons.append(season)
        return ("004",)

    monkeypatch.setattr(portable, "_available_processed_playoff_game_ids", available_playoffs)
    monkeypatch.setattr(
        portable,
        "build_rapm_stints_from_legacy_processed_games",
        lambda game_ids: (stints.assign(game_id="004"), ()),
    )
    monkeypatch.setattr(
        portable,
        "build_contextual_player_profiles",
        lambda *args, **kwargs: pd.DataFrame({"player_id": [1, 2]}),
    )
    monkeypatch.setattr(
        portable,
        "_cold_start_priors",
        lambda **kwargs: (pd.DataFrame({"player_id": [1, 2], PRIOR_MEAN_COLUMN: [0.0, 0.0]}), {}),
    )
    monkeypatch.setattr(portable, "_returning_priors", lambda _: pd.DataFrame())
    monkeypatch.setattr(
        portable,
        "_combine_priors",
        lambda returning, cold: cold.loc[:, ["player_id", PRIOR_MEAN_COLUMN]].copy(),
    )
    monkeypatch.setattr(
        portable,
        "_context_offset",
        lambda stints, model, profiles: np.zeros(len(stints), dtype=float),
    )
    monkeypatch.setattr(
        portable,
        "fit_forward_lagged_rapm_season",
        lambda season, stints, priors, lambda_grid: ForwardLaggedRapmSeason(
            season=season,
            selected_lambda=lambda_grid[0],
            cv_results=pd.DataFrame(),
            player_estimates=pd.DataFrame(
                {"season": [season, season], "player_id": [1, 2], "rapm": [0.0, 0.0]}
            ),
            player_priors=priors,
        ),
    )
    monkeypatch.setattr(
        portable,
        "player_exposure_shares",
        lambda _: pd.DataFrame({"player_id": [1, 2], "on_court_possessions": [10.0, 10.0]}),
    )
    monkeypatch.setattr(portable, "_fit_replacement_token", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        portable,
        "_fit_matchup_contextual_season",
        lambda stints, fitted, profiles, **kwargs: (
            contexts[fitted.season],
            {"season": fitted.season},
        ),
    )

    def evaluate_target(
        target_season: str, *, model: object, **kwargs: object
    ) -> dict[str, object]:
        observed["target"] = target_season
        observed["model"] = model
        return {
            "source_state": {},
            "cohort_metrics": pd.DataFrame(),
            "possession_predictions": pd.DataFrame(),
            "game_predictions": pd.DataFrame(),
            "regular_game_predictions": pd.DataFrame(),
            "team_net_rating_predictions": pd.DataFrame(),
            "team_net_rating_metrics": pd.DataFrame(),
            "team_win_predictions": pd.DataFrame(),
            "team_win_metrics": pd.DataFrame(),
        }

    monkeypatch.setattr(portable, "_evaluate_target", evaluate_target)
    monkeypatch.setattr(portable, "_write_run", lambda **kwargs: object())

    portable.train_forward_portable_matchup_contextual_rapm(
        through_season=target,
        player_season_panel_path=panel_path,
        analytical_dir=tmp_path / "analytical",
        curated_dir=tmp_path / "curated",
        artifacts_dir=tmp_path / "artifacts",
        include_historical_playoffs=True,
    )

    assert observed == {"target": target, "model": contexts[source]}
    assert playoff_seasons == [initial, source]
