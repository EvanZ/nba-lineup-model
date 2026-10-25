from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.modeling.in_season_nail import (
    ContributionLedger,
    InSeasonNailConfig,
    build_game_contribution,
    publish_game_contribution,
    publish_player_state,
    read_game_contribution,
    solve_in_season_state,
)

CORE_VERSION = "nail-v1.2.1.3"
CONTEXT_VERSION = "nail-v1.2.1.3-context"
UPDATER_VERSION = "in-season-v0.1.0"
PLAYERS = (1, 2, 3, 4)
PRIOR = {1: 1.454, 2: 0.5, 3: -0.25, 4: -1.0}
GAME_TIME = datetime(2026, 10, 21, 0, 0, tzinfo=UTC)
FINALIZED_AT = datetime(2026, 10, 21, 3, 0, tzinfo=UTC)


def _stints(game_id: str, *, target_shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [game_id, game_id],
            "home_player_ids": [(1, 2), (1, 3)],
            "away_player_ids": [(3, 4), (2, 4)],
            "target_home_net_rating": [8.0 + target_shift, -2.0 + target_shift],
            "possessions": [20.0, 30.0],
        }
    )


def _contribution(
    game_id: str = "0022600001",
    *,
    target_shift: float = 0.0,
    finalized_at: datetime = FINALIZED_AT,
    source_character: str = "a",
):
    return build_game_contribution(
        _stints(game_id, target_shift=target_shift),
        season="2026-27",
        game_id=game_id,
        game_time_utc=GAME_TIME,
        finalized_at=finalized_at,
        source_sha256=source_character * 64,
        source_uri=f"s3://nba-gestalt-data/lake/raw/cdn/{game_id}.json",
        core_model_version=CORE_VERSION,
        context_model_version=CONTEXT_VERSION,
        player_ids=PLAYERS,
        preseason_nail=PRIOR,
        fixed_context_offset=np.array([2.0, 2.0]),
    )


def _config() -> InSeasonNailConfig:
    return InSeasonNailConfig(
        core_model_version=CORE_VERSION,
        updater_version=UPDATER_VERSION,
        prior_equivalent_possessions=1000.0,
        observation_variance=25.0,
    )


def _solve(contributions=(), *, cutoff_time=FINALIZED_AT + timedelta(seconds=1)):
    return solve_in_season_state(
        season="2026-27",
        cutoff_time=cutoff_time,
        player_ids=PLAYERS,
        preseason_nail=PRIOR,
        contributions=contributions,
        config=_config(),
    ).set_index("player_id")


def test_zero_possessions_exactly_preserve_complete_preseason_forecast() -> None:
    state = _solve()

    assert state.loc[1, "frozen_preseason_nail"] == pytest.approx(1.454)
    assert state.loc[1, "current_season_residual"] == 0.0
    assert state.loc[1, "updated_nail"] == pytest.approx(1.454)
    assert state["raw_on_court_possessions"].eq(0.0).all()
    assert state["evidence_weight"].eq(0.0).all()
    assert state["prior_weight"].eq(1.0).all()


def test_game_contribution_residualizes_complete_prior_and_context() -> None:
    contribution = _contribution()

    assert contribution.game_possessions == pytest.approx(50.0)
    assert contribution.on_court_possessions.tolist() == pytest.approx([50.0, 50.0, 50.0, 50.0])
    assert contribution.xtwx.diagonal().tolist() == pytest.approx(contribution.on_court_possessions)


def test_accumulating_possessions_increases_information_without_normalizing() -> None:
    first = _contribution()
    second = _contribution(
        "0022600002",
        finalized_at=FINALIZED_AT + timedelta(days=1),
        source_character="b",
    )
    one_game = _solve([first], cutoff_time=second.finalized_at + timedelta(seconds=1))
    two_games = _solve([first, second], cutoff_time=second.finalized_at + timedelta(seconds=1))

    np.testing.assert_allclose(
        two_games["effective_information"],
        2.0 * one_game["effective_information"],
    )
    assert (two_games["evidence_weight"] > one_game["evidence_weight"]).all()


def test_strict_finalization_cutoff_prevents_own_game_leakage() -> None:
    contribution = _contribution()

    at_final = _solve([contribution], cutoff_time=contribution.finalized_at)
    after_final = _solve(
        [contribution], cutoff_time=contribution.finalized_at + timedelta(microseconds=1)
    )

    assert at_final["current_season_residual"].eq(0.0).all()
    assert not after_final["current_season_residual"].eq(0.0).all()


def test_incremental_ledger_matches_from_scratch_and_replaces_correction() -> None:
    first = _contribution()
    second = _contribution(
        "0022600002",
        finalized_at=FINALIZED_AT + timedelta(days=1),
        source_character="b",
    )
    corrected = _contribution(target_shift=4.0, source_character="c")
    ledger = ContributionLedger()

    assert ledger.upsert(first) == "added"
    assert ledger.upsert(first) == "unchanged"
    assert ledger.upsert(second) == "added"
    incremental = _solve(ledger.values(), cutoff_time=second.finalized_at + timedelta(seconds=1))
    rebuilt = _solve([first, second], cutoff_time=second.finalized_at + timedelta(seconds=1))
    pd.testing.assert_frame_equal(incremental, rebuilt)

    assert ledger.upsert(corrected) == "revised"
    revised = _solve(ledger.values(), cutoff_time=second.finalized_at + timedelta(seconds=1))
    expected = _solve([corrected, second], cutoff_time=second.finalized_at + timedelta(seconds=1))
    pd.testing.assert_frame_equal(revised, expected)
    assert not revised["updated_nail"].equals(incremental["updated_nail"])


def test_multiple_unresolved_game_revisions_are_rejected() -> None:
    with pytest.raises(ValueError, match="Multiple revisions"):
        _solve([_contribution(), _contribution(target_shift=2.0, source_character="d")])


def test_sparse_contribution_round_trip_is_immutable_and_idempotent(
    tmp_path: Path,
) -> None:
    contribution = _contribution()

    path, created = publish_game_contribution(contribution, tmp_path)
    repeated_path, repeated_created = publish_game_contribution(contribution, tmp_path)
    loaded = read_game_contribution(path)

    assert created is True
    assert repeated_created is False
    assert repeated_path == path
    assert loaded.contribution_sha256 == contribution.contribution_sha256
    assert loaded.xtwx.toarray() == pytest.approx(contribution.xtwx.toarray())
    assert loaded.xtwy == pytest.approx(contribution.xtwy)
    assert loaded.on_court_possessions == pytest.approx(contribution.on_court_possessions)


def test_player_state_publication_is_content_addressed_and_idempotent(
    tmp_path: Path,
) -> None:
    state = _solve([_contribution()]).reset_index()

    path, created = publish_player_state(state, tmp_path)
    repeated_path, repeated_created = publish_player_state(state, tmp_path)

    assert created is True
    assert repeated_created is False
    assert repeated_path == path
    assert (path / "manifest.json").is_file()
    persisted = pd.read_parquet(path / "player_state.parquet").set_index("player_id")
    pd.testing.assert_frame_equal(persisted, state.set_index("player_id"))
