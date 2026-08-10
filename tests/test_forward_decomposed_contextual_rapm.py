from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import nba_lineup_model.modeling.forward_decomposed_contextual_rapm as decomposed
from nba_lineup_model.modeling.prior_rapm import PRIOR_MEAN_COLUMN, ForwardLaggedRapmSeason


@dataclass(frozen=True)
class _Context:
    name: str

    def predict_lineups(self, home: object, away: object, profiles: object) -> np.ndarray:
        return np.zeros(len(home), dtype=float)  # type: ignore[arg-type]


def test_frozen_evaluation_uses_previous_season_decomposed_context_state(
    monkeypatch, tmp_path: Path
) -> None:
    """The completed target side function cannot score its own frozen holdout."""

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
        }
    )
    contexts = {source: _Context(source), target: _Context(target)}
    observed: dict[str, object] = {}

    monkeypatch.setattr(decomposed, "HISTORICAL_SEASONS", (initial, source))
    monkeypatch.setattr(decomposed, "_latest_run", lambda _: reference_root)
    monkeypatch.setattr(
        decomposed, "prepare_player_exposure_cohort", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(decomposed, "read_rapm_stints", lambda *args, **kwargs: stints.copy())
    monkeypatch.setattr(
        decomposed,
        "build_contextual_player_profiles",
        lambda *args, **kwargs: pd.DataFrame({"player_id": [1, 2]}),
    )
    monkeypatch.setattr(
        decomposed,
        "_cold_start_priors",
        lambda **kwargs: (pd.DataFrame({"player_id": [1, 2], PRIOR_MEAN_COLUMN: [0.0, 0.0]}), {}),
    )
    monkeypatch.setattr(decomposed, "_returning_priors", lambda _: pd.DataFrame())
    monkeypatch.setattr(
        decomposed,
        "_combine_priors",
        lambda returning, cold: cold.loc[:, ["player_id", PRIOR_MEAN_COLUMN]].copy(),
    )
    monkeypatch.setattr(
        decomposed, "_context_offset", lambda stints, model, profiles: [0.0] * len(stints)
    )
    monkeypatch.setattr(
        decomposed,
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
        decomposed,
        "player_exposure_shares",
        lambda _: pd.DataFrame({"player_id": [1, 2], "on_court_possessions": [10.0, 10.0]}),
    )
    monkeypatch.setattr(decomposed, "_fit_replacement_token", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        decomposed,
        "_fit_decomposed_contextual_season",
        lambda stints, fitted, profiles, alpha: (
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

    monkeypatch.setattr(decomposed, "_evaluate_target", evaluate_target)
    monkeypatch.setattr(decomposed, "_write_run", lambda **kwargs: object())

    decomposed.train_forward_decomposed_contextual_rapm(
        through_season=target,
        player_season_panel_path=panel_path,
        analytical_dir=tmp_path / "analytical",
        curated_dir=tmp_path / "curated",
        artifacts_dir=tmp_path / "artifacts",
    )

    assert observed == {"target": target, "model": contexts[source]}
