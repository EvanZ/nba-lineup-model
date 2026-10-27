from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
from scipy import sparse

from nba_lineup_model.modeling.in_season_additive_profile import (
    EVIDENCE_COLUMNS,
    FEATURE_TO_PROFILE_COLUMN,
    PROFILE_FEATURES,
    AdditiveProfileUpdaterConfig,
    FrozenAdditiveProfileContract,
    ProfileGameContribution,
)
from nba_lineup_model.modeling.in_season_additive_profile_calibration import (
    HistoricalProfileReplaySeason,
    _replay_candidate,
)
from nba_lineup_model.modeling.in_season_nail import GameContribution
from nba_lineup_model.modeling.in_season_nail_calibration import (
    CORE_MODEL_VERSION,
    HistoricalReplayGame,
    HistoricalReplaySeason,
)


def _contract(season: str) -> FrozenAdditiveProfileContract:
    profiles = pd.DataFrame(
        {
            "player_id": [1, 2],
            "three_pa_per_100": [10.0, 5.0],
            "three_pm_per_100": [4.0, 2.0],
            "assists_per_100": [8.0, 4.0],
            "turnovers_per_100": [3.0, 2.0],
            "usage_pct": [30.0, 20.0],
            "steals_per_100": [2.0, 1.0],
            "blocks_per_100": [0.5, 1.0],
            "offensive_rebound_pct": [2.0, 5.0],
        }
    )
    coefficients = {feature: 0.1 for feature in PROFILE_FEATURES}
    references = {
        feature: float(profiles[column].mean())
        for feature, column in FEATURE_TO_PROFILE_COLUMN.items()
    }
    published = {
        int(row.player_id): sum(
            coefficients[feature]
            * (float(getattr(row, column)) - references[feature])
            for feature, column in FEATURE_TO_PROFILE_COLUMN.items()
        )
        for row in profiles.itertuples(index=False)
    }
    return FrozenAdditiveProfileContract(
        season=season,
        core_model_version=CORE_MODEL_VERSION,
        source_run_id="fixture",
        profiles=profiles,
        coefficients=coefficients,
        reference_values=references,
        published_adjustments=published,
    )


def _profile_contribution(
    season: str, game_id: str, game_time: datetime, index: int
) -> ProfileGameContribution:
    evidence = pd.DataFrame(
        [
            {
                "player_id": 1,
                "on_court_possessions": 100.0,
                "minutes": 36.0,
                "three_pointers_attempted": 12.0 + index,
                "three_pointers_made": 6.0,
                "assists": 8.0,
                "turnovers": 2.0,
                "steals": 2.0,
                "blocks": 1.0,
                "usage_events": 30.0,
                "usage_opportunities": 100.0,
                "offensive_rebound_pct_minutes": 72.0,
            },
            {
                "player_id": 2,
                "on_court_possessions": 100.0,
                "minutes": 36.0,
                "three_pointers_attempted": 2.0,
                "three_pointers_made": 0.0,
                "assists": 2.0,
                "turnovers": 4.0,
                "steals": 0.0,
                "blocks": 0.0,
                "usage_events": 15.0,
                "usage_opportunities": 100.0,
                "offensive_rebound_pct_minutes": 180.0,
            },
        ],
        columns=EVIDENCE_COLUMNS,
    )
    return ProfileGameContribution(
        season=season,
        game_id=game_id,
        game_time_utc=game_time,
        finalized_at=game_time + timedelta(hours=4),
        source_sha256=f"{index + 1:064x}",
        source_uri=f"fixture://{season}/{game_id}",
        evidence=evidence,
    )


def _replay(season: str) -> HistoricalProfileReplaySeason:
    start = datetime(int(season[:4]), 10, 20, tzinfo=UTC)
    scalar_games = []
    profile_games = []
    for index in range(3):
        game_time = start + timedelta(days=index)
        game_id = f"{season}-{index}"
        scalar_contribution = GameContribution(
            season=season,
            game_id=game_id,
            game_time_utc=game_time,
            finalized_at=game_time + timedelta(hours=4),
            source_sha256=f"{index + 11:064x}",
            source_uri=f"fixture://scalar/{season}/{game_id}",
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
        scalar_games.append(
            HistoricalReplayGame(
                game_id=game_id,
                game_date=date(int(season[:4]), 10, 20) + timedelta(days=index),
                game_time_utc=game_time,
                actual_home_margin=4.0,
                frozen_home_margin=0.0,
                margin_design=sparse.csr_matrix([[1.0, -1.0]]),
                contribution=scalar_contribution,
            )
        )
        profile_games.append(_profile_contribution(season, game_id, game_time, index))
    return HistoricalProfileReplaySeason(
        scalar_replay=HistoricalReplaySeason(
            season=season,
            player_ids=(1, 2),
            frozen_preseason_nail=np.zeros(2),
            games=tuple(scalar_games),
            source_run_id="fixture",
        ),
        contract=_contract(season),
        profile_contributions=tuple(profile_games),
    )


def test_profile_replay_is_pregame_and_residualization_changes_naive_sum() -> None:
    predictions, states = _replay_candidate(
        _replay("2025-26"),
        config=AdditiveProfileUpdaterConfig(),
        capture_states=True,
    )
    first = predictions.loc[predictions["game_date"].eq(date(2025, 10, 20))]
    third = predictions.loc[
        predictions["game_date"].eq(date(2025, 10, 22))
    ].set_index("model")

    assert first["predicted_home_margin"].eq(0.0).all()
    assert third.loc[
        "profile_plus_residualized_scalar", "predicted_home_margin"
    ] != third.loc["naive_profile_plus_scalar", "predicted_home_margin"]
    # The materialized endpoint includes the final game's evidence, while all
    # scored games remain strictly pregame.
    assert states["prediction_date"].max() == date(2025, 10, 23)
    assert states.loc[states["prediction_date"].eq(date(2025, 10, 23)), "observed_games"].eq(
        3
    ).all()
