from __future__ import annotations

from pathlib import Path

import pandas as pd

import nba_lineup_model.modeling.forward_contextual_rapm as contextual
from nba_lineup_model.modeling.prior_rapm import PRIOR_MEAN_COLUMN, ForwardLaggedRapmSeason


def test_frozen_evaluation_uses_previous_season_context_state(
    monkeypatch, tmp_path: Path
) -> None:
    """The target-season context model may be stored, but cannot score its own holdout."""

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
    contexts = {source: object(), target: object()}
    observed: dict[str, object] = {}

    monkeypatch.setattr(contextual, "HISTORICAL_SEASONS", (initial, source))
    monkeypatch.setattr(contextual, "_latest_run", lambda _: reference_root)
    monkeypatch.setattr(
        contextual, "prepare_player_exposure_cohort", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(contextual, "read_rapm_stints", lambda *args, **kwargs: stints.copy())
    monkeypatch.setattr(
        contextual,
        "build_contextual_player_profiles",
        lambda *args, **kwargs: pd.DataFrame({"player_id": [1, 2]}),
    )
    monkeypatch.setattr(
        contextual,
        "_cold_start_priors",
        lambda **kwargs: (pd.DataFrame({"player_id": [1, 2], PRIOR_MEAN_COLUMN: [0.0, 0.0]}), {}),
    )
    monkeypatch.setattr(contextual, "_returning_priors", lambda _: pd.DataFrame())
    monkeypatch.setattr(
        contextual,
        "_combine_priors",
        lambda returning, cold: cold.loc[:, ["player_id", PRIOR_MEAN_COLUMN]].copy(),
    )
    monkeypatch.setattr(
        contextual, "_context_offset", lambda stints, model, profiles: [0.0] * len(stints)
    )
    monkeypatch.setattr(
        contextual,
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
        contextual,
        "player_exposure_shares",
        lambda _: pd.DataFrame({"player_id": [1, 2], "on_court_possessions": [10.0, 10.0]}),
    )
    monkeypatch.setattr(contextual, "_fit_replacement_token", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        contextual,
        "_fit_contextual_season",
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

    monkeypatch.setattr(contextual, "_evaluate_target", evaluate_target)

    contextual.train_forward_contextual_rapm(
        through_season=target,
        player_season_panel_path=panel_path,
        analytical_dir=tmp_path / "analytical",
        curated_dir=tmp_path / "curated",
        artifacts_dir=tmp_path / "artifacts",
    )

    assert observed == {"target": target, "model": contexts[source]}


def test_forward_contextual_rankings_publish_completed_player_state(tmp_path: Path) -> None:
    source = tmp_path / "contextual-run"
    source.mkdir()
    (source / "metadata.json").write_text(
        '{"model": "forward_contextual_offset_rapm", "run_id": "source-run", '
        '"target_season": "2025-26"}\n'
    )
    pd.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "player_id": [1, 2],
            "rapm": [4.25, 2.0],
            "prior_rapm": [3.0, 2.5],
            "rapm_adjustment_from_prior": [1.25, -0.5],
        }
    ).to_parquet(source / "historical_player_coefficients.parquet", index=False)
    panel = tmp_path / "player_seasons.parquet"
    pd.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "player_id": [1, 2],
            "player_name": ["Example One", "Example Two"],
            "listed_position": ["G", "F"],
            "rapm_possessions": [1000.0, 2000.0],
        }
    ).to_parquet(panel, index=False)

    run = contextual.build_forward_contextual_rankings(
        source_run_dir=source,
        player_season_panel_path=panel,
        artifacts_dir=tmp_path / "artifacts",
    )
    rankings = pd.read_parquet(run.run_dir / "next_season_player_rankings.parquet")
    assert rankings.loc[0, "player_name"] == "Example One"
    assert rankings.loc[0, "rank"] == 1
    assert run.next_season == "2026-27"

    page = tmp_path / "forward-contextual-rapm.md"
    page.write_text(
        "# Forward Contextual RAPM\n\n"
        "<!-- forward-contextual-rankings:start -->\n"
        "<!-- forward-contextual-rankings:end -->\n"
    )
    contextual.render_forward_contextual_rankings_page(run.run_dir, page_path=page)

    text = page.read_text()
    assert "## 2026-27 Player Rankings" in text
    assert "| 1 | Example One | G | +4.25 | +3.00 | +1.25 | 1,000 |" in text
