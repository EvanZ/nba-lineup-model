from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.modeling import replacement_level
from nba_lineup_model.modeling.replacement_level import (
    candidate_replacement_prior,
    exposure_band,
    player_exposure_shares,
    prepare_player_exposure_cohort,
    summarize_exposure_bands,
    summarize_low_exposure_seasons,
)


def test_player_exposure_share_uses_full_team_possession_opportunities() -> None:
    stints = pd.DataFrame(
        {
            "possessions": [4.0, 6.0],
            "home_team_id": [1, 1],
            "away_team_id": [2, 2],
            "home_player_ids": [[10, 11, 12, 13, 14], [10, 11, 12, 13, 15]],
            "away_player_ids": [[20, 21, 22, 23, 24], [20, 21, 22, 23, 25]],
        }
    )

    output = player_exposure_shares(stints).set_index("player_id")

    assert output.loc[10, "on_court_possessions"] == pytest.approx(10.0)
    assert output.loc[10, "team_opportunity_possessions"] == pytest.approx(10.0)
    assert output.loc[10, "exposure_share"] == pytest.approx(1.0)
    assert output.loc[14, "exposure_share"] == pytest.approx(0.4)
    assert output.loc[15, "exposure_share"] == pytest.approx(0.6)


def test_exposure_band_summary_and_prior_are_season_balanced() -> None:
    cohort = pd.DataFrame(
        {
            "season": ["2020-21", "2020-21", "2021-22", "2021-22"],
            "season_start_year": [2020, 2020, 2021, 2021],
            "player_id": [1, 2, 3, 4],
            "rapm": [-1.0, -3.0, 1.0, -1.0],
            "rapm_possessions": [25.0, 25.0, 25.0, 25.0],
            "exposure_share": [0.01, 0.10, 0.01, 0.40],
        }
    )
    cohort["exposure_band"] = exposure_band(cohort["exposure_share"])

    bands = summarize_exposure_bands(cohort, bootstrap_samples=20, bootstrap_seed=1)
    low = summarize_low_exposure_seasons(cohort, replacement_share_cutoff=0.05)
    prior = candidate_replacement_prior(low, bootstrap_samples=20, bootstrap_seed=1)

    assert bands.loc[bands["exposure_band"].eq("0-5%"), "player_count"].item() == 2
    assert bands.loc[
        bands["exposure_band"].eq("0-5%"), "season_balanced_mean_rapm"
    ].item() == pytest.approx(0.0)
    assert low["equal_player_mean_rapm"].tolist() == [-1.0, 1.0]
    assert prior["candidate_replacement_prior"] == pytest.approx(0.0)
    assert np.isfinite(prior["season_block_bootstrap_lower"])


def test_player_exposure_cohort_includes_rookies_and_returning_players(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stints = _stints()
    monkeypatch.setattr(replacement_level, "read_rapm_stints", lambda *_: stints)
    panel = pd.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "season_start_year": [2025, 2025],
            "player_id": [10, 11],
            "player_name": ["Rookie", "Veteran"],
            "listed_position": ["G", "F"],
            "is_rookie": [True, False],
            "nba_experience_years": [0, 7],
            "rapm": [-0.2, -0.4],
            "rapm_possessions": [10.0, 10.0],
        }
    )

    cohort = prepare_player_exposure_cohort(
        panel,
        through_season="2025-26",
        analytical_dir="unused",
    )

    assert cohort["player_name"].tolist() == ["Rookie", "Veteran"]
    assert cohort["experience_band"].astype(str).tolist() == ["First year", "Years 4+"]


def _stints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "possessions": [4.0, 6.0],
            "home_team_id": [1, 1],
            "away_team_id": [2, 2],
            "home_player_ids": [[10, 11, 12, 13, 14], [10, 11, 12, 13, 15]],
            "away_player_ids": [[20, 21, 22, 23, 24], [20, 21, 22, 23, 25]],
        }
    )
