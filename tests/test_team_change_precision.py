"""Tests for clean-offseason team-change prior-precision candidates."""

from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.modeling.team_change_precision import (
    build_team_change_precision_candidates,
)


def test_team_change_candidates_relax_only_clean_offseason_movers() -> None:
    panel = pd.DataFrame(
        {
            "season": ["2024-25"] * 4 + ["2025-26"] * 4,
            "player_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "primary_team_tricode": ["AAA", "BBB", "CCC", "DDD", "EEE", "BBB", "FFF", "GGG"],
            "team_count": [1, 1, 2, 1, 1, 1, 1, 1],
        }
    )
    priors = pd.DataFrame(
        {"player_id": [1, 2, 3, 4], "lagged_rapm_prior": [1.0, 1.0, 1.0, 1.0]}
    )
    exposure = pd.DataFrame(
        {"player_id": [1, 2, 3], "on_court_possessions": [3_000.0, 500.0, 2_000.0]}
    )

    candidates, metadata = build_team_change_precision_candidates(
        season="2025-26",
        panel=panel,
        player_ids=(1, 2, 3, 4),
        player_priors=priors,
        exposure_history=[exposure],
    )

    baseline = candidates["move_variance_ratio=0"]
    assert baseline == {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    strongest = candidates["move_variance_ratio=0.4"]
    assert strongest[1] < 1.0
    assert strongest[2] == 1.0
    assert strongest[3] == 1.0  # Prior season was multi-team, so exclude it.
    assert strongest[4] == 1.0  # No completed-season exposure.
    assert metadata["team_change_precision_clean_mover_count"] == 1
    assert metadata["team_change_precision_eligible_returner_count"] == 3


def test_team_change_candidates_require_panel_team_columns() -> None:
    with pytest.raises(ValueError, match="team-change columns"):
        build_team_change_precision_candidates(
            season="2025-26",
            panel=pd.DataFrame({"season": ["2025-26"], "player_id": [1]}),
            player_ids=(1,),
            player_priors=pd.DataFrame({"player_id": [1], "lagged_rapm_prior": [0.0]}),
            exposure_history=[],
        )
