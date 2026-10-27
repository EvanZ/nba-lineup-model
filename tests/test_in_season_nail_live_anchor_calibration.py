from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from nba_lineup_model.modeling.in_season_nail import GameContribution
from nba_lineup_model.modeling.in_season_nail_calibration import (
    CORE_MODEL_VERSION,
    HistoricalReplayGame,
    HistoricalReplaySeason,
)
from nba_lineup_model.modeling.in_season_nail_live_anchor_calibration import (
    HistoricalLiveAnchorReplaySeason,
    _recommended_weight,
    build_live_anchor_replays,
    replay_live_anchor_candidate,
)


def _game(season: str, index: int, game_date: date) -> HistoricalReplayGame:
    game_time = datetime.combine(game_date, datetime.min.time(), tzinfo=UTC)
    contribution = GameContribution(
        season=season,
        game_id=f"00{int(season[:4]) % 100:02d}{index:06d}",
        game_time_utc=game_time,
        finalized_at=game_time + timedelta(hours=4),
        source_sha256=f"{index + 1:064x}",
        source_uri=f"fixture://{season}/{index}",
        core_model_version=CORE_MODEL_VERSION,
        context_model_version="fixture-context",
        player_ids=(1, 2),
        xtwx=sparse.csr_matrix([[100.0, -100.0], [-100.0, 100.0]]),
        xtwy=np.array([200.0, -200.0]),
        on_court_possessions=np.array([100.0, 100.0]),
        game_possessions=100.0,
        stint_count=1,
        context_offset_sha256="f" * 64,
    )
    return HistoricalReplayGame(
        game_id=contribution.game_id,
        game_date=game_date,
        game_time_utc=game_time,
        actual_home_margin=4.0,
        frozen_home_margin=0.0,
        margin_design=sparse.csr_matrix([[1.0, -1.0]]),
        contribution=contribution,
    )


def _season(season: str) -> HistoricalReplaySeason:
    start = date(int(season[:4]), 10, 20)
    return HistoricalReplaySeason(
        season=season,
        player_ids=(1, 2),
        frozen_preseason_nail=np.zeros(2),
        games=(
            _game(season, 0, start),
            _game(season, 1, start + timedelta(days=1)),
        ),
        source_run_id="fixture-run",
    )


def test_previous_live_anchor_is_a_forward_safe_source_endpoint(tmp_path) -> None:
    ratings_path = tmp_path / "ratings.parquet"
    pd.DataFrame(
        {
            "season": ["2021-22", "2021-22", "2022-23", "2022-23"],
            "player_id": [1, 2, 1, 2],
            "rapm": [0.0, 0.0, 0.0, 0.0],
        }
    ).to_parquet(ratings_path, index=False)

    replays, players, endpoints = build_live_anchor_replays(
        [_season("2022-23"), _season("2023-24")],
        target_seasons=("2023-24",),
        completed_ratings_path=ratings_path,
        player_panel_path=tmp_path / "missing-panel.parquet",
        prior_equivalent_possessions=10.0,
    )

    source = endpoints.loc[endpoints["season"].eq("2022-23")].set_index("player_id")
    target = players.set_index("player_id")
    assert target.loc[1, "previous_live_endpoint_nail"] == pytest.approx(
        source.loc[1, "source_live_endpoint_nail"]
    )
    assert target["previous_live_endpoint_available"].all()
    assert replays[0].previous_live_endpoint_nail[0] > 0.0


def test_live_endpoint_weight_changes_the_first_game_without_current_signal() -> None:
    scalar = _season("2023-24")
    replay = HistoricalLiveAnchorReplaySeason(
        scalar_replay=scalar,
        forward_preseason_nail=np.zeros(2),
        previous_live_endpoint_nail=np.array([1.0, 0.0]),
        previous_live_endpoint_available=np.array([True, True]),
    )

    forward = replay_live_anchor_candidate(
        replay, live_endpoint_weight=0.0, prior_equivalent_possessions=10.0
    )
    carried = replay_live_anchor_candidate(
        replay, live_endpoint_weight=1.0, prior_equivalent_possessions=10.0
    )

    assert forward.iloc[0]["predicted_home_margin"] == pytest.approx(0.0)
    assert carried.iloc[0]["predicted_home_margin"] == pytest.approx(1.0)
    assert carried.iloc[1]["predicted_home_margin"] > 1.0


def test_live_endpoint_holdout_gate_requires_all_primary_metrics() -> None:
    metrics = pd.DataFrame(
        [
            {
                "split": "holdout",
                "scope": "pooled",
                "live_endpoint_weight": 0.0,
                "margin_mae": 10.0,
                "margin_rmse": 13.0,
                "brier_score": 0.20,
                "log_loss": 0.59,
            },
            {
                "split": "holdout",
                "scope": "pooled",
                "live_endpoint_weight": 0.5,
                "margin_mae": 9.9,
                "margin_rmse": 12.9,
                "brier_score": 0.199,
                "log_loss": 0.591,
            },
        ]
    )

    assert _recommended_weight(metrics, 0.5) == 0.0


def test_live_anchor_rejects_mismatched_endpoint_vector() -> None:
    with pytest.raises(ValueError, match="match the player vocabulary"):
        HistoricalLiveAnchorReplaySeason(
            scalar_replay=_season("2023-24"),
            forward_preseason_nail=np.zeros(2),
            previous_live_endpoint_nail=np.zeros(1),
            previous_live_endpoint_available=np.ones(1, dtype=bool),
        )
