from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from nba_lineup_model.modeling.in_season_nail import GameContribution
from nba_lineup_model.modeling.in_season_nail_anchor_calibration import (
    AnchorCalibration,
    HistoricalAnchorReplaySeason,
    _recommended_completed_weight,
    build_anchor_replays,
    reanchored_score,
    replay_anchor_candidate,
)
from nba_lineup_model.modeling.in_season_nail_calibration import (
    CORE_MODEL_VERSION,
    HistoricalReplayGame,
    HistoricalReplaySeason,
)


def _game(
    season: str,
    index: int,
    *,
    game_date: date,
    score: float = 200.0,
) -> HistoricalReplayGame:
    game_time = datetime.combine(game_date, datetime.min.time(), tzinfo=UTC) + timedelta(
        hours=index
    )
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
        xtwy=np.array([score, -score]),
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


def _scalar_season(season: str = "2023-24") -> HistoricalReplaySeason:
    start = date(int(season[:4]), 10, 20)
    return HistoricalReplaySeason(
        season=season,
        player_ids=(1, 2),
        frozen_preseason_nail=np.zeros(2),
        games=(
            _game(season, 0, game_date=start),
            _game(season, 1, game_date=start),
            _game(season, 2, game_date=start + timedelta(days=1)),
        ),
        source_run_id="fixture-run",
    )


def test_reanchored_score_exactly_removes_new_anchor_from_residual() -> None:
    game = _scalar_season().games[0]
    delta = np.array([1.0, 0.0])

    transformed = reanchored_score(game, delta)

    assert transformed == pytest.approx(np.array([100.0, -100.0]))


def test_same_day_games_cannot_update_each_other() -> None:
    replay = HistoricalAnchorReplaySeason(
        scalar_replay=_scalar_season(),
        forward_preseason_nail=np.zeros(2),
        previous_completed_nail=np.array([1.0, 0.0]),
        previous_completed_available=np.array([True, True]),
    )

    predictions = replay_anchor_candidate(
        replay,
        completed_weight=1.0,
        prior_equivalent_possessions=10.0,
    )

    first_date = predictions.loc[predictions["game_date"].eq(date(2023, 10, 20))]
    next_date = predictions.loc[predictions["game_date"].eq(date(2023, 10, 21))]
    assert first_date["predicted_home_margin"].tolist() == pytest.approx([1.0, 1.0])
    assert next_date.iloc[0]["predicted_home_margin"] > 1.0


def test_zero_weight_reproduces_original_forward_anchor() -> None:
    scalar = _scalar_season()
    replay = HistoricalAnchorReplaySeason(
        scalar_replay=scalar,
        forward_preseason_nail=np.zeros(2),
        previous_completed_nail=np.array([8.0, -4.0]),
        previous_completed_available=np.array([True, True]),
    )

    predictions = replay_anchor_candidate(
        replay,
        completed_weight=0.0,
        prior_equivalent_possessions=6000.0,
    )

    assert predictions.iloc[0]["predicted_home_margin"] == pytest.approx(
        scalar.games[0].frozen_home_margin
    )


def test_missing_previous_completed_rating_falls_back_to_forward(tmp_path) -> None:
    scalar = _scalar_season()
    ratings_path = tmp_path / "ratings.parquet"
    pd.DataFrame(
        {
            "season": ["2022-23"],
            "player_id": [1],
            "rapm": [3.0],
        }
    ).to_parquet(ratings_path, index=False)

    replays, anchors = build_anchor_replays(
        [scalar],
        completed_ratings_path=ratings_path,
        player_panel_path=tmp_path / "missing-panel.parquet",
    )

    assert replays[0].previous_completed_nail == pytest.approx(np.array([3.0, 0.0]))
    assert replays[0].previous_completed_available.tolist() == [True, False]
    missing = anchors.loc[anchors["player_id"].eq(2)].iloc[0]
    assert missing["anchor_gap"] == pytest.approx(0.0)
    assert missing["experience_cohort"] == "cold start / no prior rating"


def test_anchor_replay_rejects_a_mismatched_player_vector() -> None:
    with pytest.raises(ValueError, match="match the replay vocabulary"):
        HistoricalAnchorReplaySeason(
            scalar_replay=_scalar_season(),
            forward_preseason_nail=np.zeros(2),
            previous_completed_nail=np.array([1.0]),
            previous_completed_available=np.array([True]),
        )


def test_development_blend_must_improve_every_primary_holdout_metric() -> None:
    metrics = pd.DataFrame(
        [
            {
                "split": "holdout",
                "scope": "pooled",
                "completed_weight": 0.0,
                "margin_mae": 10.0,
                "margin_rmse": 13.0,
                "brier_score": 0.20,
                "log_loss": 0.59,
            },
            {
                "split": "holdout",
                "scope": "pooled",
                "completed_weight": 0.3,
                "margin_mae": 9.9,
                "margin_rmse": 12.9,
                "brier_score": 0.199,
                "log_loss": 0.591,
            },
        ]
    )
    calibration = AnchorCalibration(
        selected_completed_weight=0.3,
        predictions=pd.DataFrame(),
        metrics=metrics,
        player_anchors=pd.DataFrame(),
        player_states=pd.DataFrame(),
        diagnostics=pd.DataFrame(),
    )

    assert _recommended_completed_weight(calibration) == 0.0
