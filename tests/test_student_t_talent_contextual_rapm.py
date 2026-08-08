from __future__ import annotations

from pathlib import Path

import pandas as pd

import nba_lineup_model.modeling.student_t_talent_contextual_rapm as combined
from nba_lineup_model.modeling.prior_rapm import PRIOR_MEAN_COLUMN, ForwardLaggedRapmSeason


def test_combined_frozen_evaluation_uses_previous_season_context_state(
    monkeypatch, tmp_path: Path
) -> None:
    target = "2025-26"
    source = "2024-25"
    initial = "2023-24"
    panel_path = tmp_path / "player_seasons.parquet"
    pd.DataFrame({"season": [initial, source, target], "player_id": [1, 1, 1]}).to_parquet(
        panel_path,
        index=False,
    )
    stints = pd.DataFrame(
        {
            "home_player_ids": [[1]],
            "away_player_ids": [[2]],
            "target_home_net_rating": [2.0],
            "possessions": [10.0],
        }
    )
    contexts = {source: object(), target: object()}
    observed: dict[str, object] = {}

    monkeypatch.setattr(combined, "HISTORICAL_SEASONS", (initial, source))
    monkeypatch.setattr(
        combined,
        "_baseline_lambda_schedule",
        lambda *_: {initial: 1.0, source: 1.0, target: 1.0},
    )
    monkeypatch.setattr(
        combined,
        "prepare_player_exposure_cohort",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(combined, "read_rapm_stints", lambda *args, **kwargs: stints.copy())
    monkeypatch.setattr(
        combined,
        "build_contextual_player_profiles",
        lambda *args, **kwargs: pd.DataFrame({"player_id": [1, 2]}),
    )
    monkeypatch.setattr(
        combined,
        "_cold_start_priors",
        lambda **kwargs: (pd.DataFrame({"player_id": [1, 2], PRIOR_MEAN_COLUMN: [0.0, 0.0]}), {}),
    )
    monkeypatch.setattr(combined, "_returning_priors", lambda _: pd.DataFrame())
    monkeypatch.setattr(combined, "_combine_priors", lambda returning, cold: cold.copy())
    monkeypatch.setattr(combined, "_context_offset", lambda stints, model, profiles: [0.0])
    monkeypatch.setattr(
        combined,
        "_fit_student_t_talent_season",
        lambda season, stints, priors, regularization, *args: (
            ForwardLaggedRapmSeason(
                season=season,
                selected_lambda=regularization,
                cv_results=pd.DataFrame(),
                player_estimates=pd.DataFrame(
                    {
                        "season": [season, season],
                        "player_id": [1, 2],
                        "rapm": [0.0, 0.0],
                        "prior_rapm": [0.0, 0.0],
                        "rapm_adjustment_from_prior": [0.0, 0.0],
                    }
                ),
                player_priors=priors,
            ),
            {},
        ),
    )
    monkeypatch.setattr(
        combined,
        "player_exposure_shares",
        lambda _: pd.DataFrame({"player_id": [1, 2], "on_court_possessions": [10.0, 10.0]}),
    )
    monkeypatch.setattr(combined, "_fit_replacement_token", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        combined,
        "_fit_contextual_season",
        lambda stints, fitted, profiles, alpha: (
            contexts[fitted.season],
            {"season": fitted.season},
        ),
    )
    monkeypatch.setattr(combined, "_write_checkpoint", lambda *args, **kwargs: None)

    def evaluate_target(
        target_season: str,
        *,
        model: object,
        **kwargs: object,
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

    monkeypatch.setattr(combined, "_evaluate_target", evaluate_target)
    monkeypatch.setattr(
        combined,
        "_write_run",
        lambda **kwargs: combined.StudentTTalentContextualRapmRun(
            tmp_path / "run", "run", target
        ),
    )

    combined.train_student_t_talent_contextual_rapm(
        through_season=target,
        player_season_panel_path=panel_path,
        analytical_dir=tmp_path / "analytical",
        curated_dir=tmp_path / "curated",
        artifacts_dir=tmp_path / "artifacts",
        checkpoint_path=tmp_path / "checkpoint.joblib",
    )

    assert observed == {"target": target, "model": contexts[source]}
