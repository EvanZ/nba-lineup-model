from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
from scipy import sparse

from nba_lineup_model.modeling.in_season_nail import GameContribution
from nba_lineup_model.modeling.in_season_nail_calibration import (
    CORE_MODEL_VERSION,
    HistoricalReplayGame,
    HistoricalReplaySeason,
    calibrate_prior_strength,
    read_historical_replay_mart,
    write_historical_replay_mart,
)


def _game(
    season: str,
    index: int,
    *,
    score: float,
    actual_margin: float,
    margin_sign: float = 1.0,
) -> HistoricalReplayGame:
    start_year = int(season[:4])
    game_time = datetime(start_year, 10, 20, tzinfo=UTC) + timedelta(days=index)
    player_ids = (1, 2)
    contribution = GameContribution(
        season=season,
        game_id=f"00{start_year % 100:02d}{index:06d}",
        game_time_utc=game_time,
        finalized_at=game_time + timedelta(hours=4),
        source_sha256=f"{index + 1:064x}",
        source_uri=f"fixture://{season}/{index}",
        core_model_version=CORE_MODEL_VERSION,
        context_model_version="fixture-context",
        player_ids=player_ids,
        xtwx=sparse.csr_matrix([[100.0, -100.0], [-100.0, 100.0]]),
        xtwy=np.array([score, -score]),
        on_court_possessions=np.array([100.0, 100.0]),
        game_possessions=100.0,
        stint_count=1,
        context_offset_sha256="f" * 64,
    )
    return HistoricalReplayGame(
        game_id=contribution.game_id,
        game_date=date(start_year, 10, 20) + timedelta(days=index),
        game_time_utc=game_time,
        actual_home_margin=actual_margin,
        frozen_home_margin=0.0,
        margin_design=sparse.csr_matrix([[margin_sign, -margin_sign]]),
        contribution=contribution,
    )


def _season(season: str) -> HistoricalReplaySeason:
    return HistoricalReplaySeason(
        season=season,
        player_ids=(1, 2),
        frozen_preseason_nail=np.zeros(2),
        games=(
            _game(season, 0, score=200.0, actual_margin=4.0),
            _game(season, 1, score=200.0, actual_margin=4.0),
            _game(season, 2, score=200.0, actual_margin=4.0),
        ),
        source_run_id="fixture-run",
    )


def test_calibration_is_chronological_and_first_game_has_no_current_season_signal() -> None:
    result = calibrate_prior_strength(
        [_season("2023-24"), _season("2024-25")],
        prior_grid=(10.0, 1000.0),
        selection_seasons=("2023-24",),
    )
    first_games = result.predictions.loc[result.predictions["game_date"].eq(date(2023, 10, 20))]

    assert first_games["predicted_home_margin"].eq(0.0).all()
    assert result.selected_prior_equivalent_possessions == 10.0
    assert result.holdout_seasons == ("2024-25",)


def test_smaller_prior_moves_farther_from_frozen_control_after_evidence() -> None:
    result = calibrate_prior_strength(
        [_season("2023-24")],
        prior_grid=(10.0, 1000.0),
        selection_seasons=("2023-24",),
    )
    second = result.predictions.loc[
        result.predictions["game_date"].eq(date(2023, 10, 21))
        & result.predictions["prior_equivalent_possessions"].notna()
    ].set_index("prior_equivalent_possessions")

    assert second.loc[10.0, "predicted_home_margin"] > second.loc[1000.0, "predicted_home_margin"]
    assert second.loc[10.0, "predicted_home_margin"] > 0.0


def test_calibration_rejects_selection_season_outside_replay() -> None:
    with pytest.raises(ValueError, match="subset"):
        calibrate_prior_strength(
            [_season("2023-24")],
            prior_grid=(100.0,),
            selection_seasons=("2022-23",),
        )


def test_sparse_historical_replay_mart_round_trip(tmp_path) -> None:
    replays = (_season("2023-24"), _season("2024-25"))
    path = write_historical_replay_mart(replays, tmp_path / "replay.joblib")

    loaded = read_historical_replay_mart(
        path,
        expected_seasons=("2023-24", "2024-25"),
        expected_source_run_id="fixture-run",
    )

    assert tuple(replay.season for replay in loaded) == ("2023-24", "2024-25")
    assert loaded[0].games[0].contribution.contribution_sha256 == (
        replays[0].games[0].contribution.contribution_sha256
    )
    assert loaded[0].games[0].contribution.xtwx.toarray() == pytest.approx(
        replays[0].games[0].contribution.xtwx.toarray()
    )


def test_calibration_rejects_empty_replay() -> None:
    with pytest.raises(ValueError, match="At least one"):
        calibrate_prior_strength(
            [],
            prior_grid=(100.0,),
            selection_seasons=("2023-24",),
        )
