from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.modeling.in_season_additive_profile import (
    FEATURE_TO_PROFILE_COLUMN,
    PROFILE_FEATURES,
    AdditiveProfileUpdaterConfig,
    FrozenAdditiveProfileContract,
    ProfileContributionLedger,
    build_profile_game_contribution,
    load_published_profile_contract,
    profile_delta_map,
    publish_additive_profile_state,
    publish_profile_game_contribution,
    read_profile_game_contribution,
    solve_additive_profile_state,
)

GAME_TIME = datetime(2026, 10, 21, 2, 0, tzinfo=UTC)
FINALIZED_AT = GAME_TIME + timedelta(hours=3)


def _contract() -> FrozenAdditiveProfileContract:
    profiles = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "three_pa_per_100": [10.0, 5.0, 8.0, 2.0],
            "three_pm_per_100": [4.0, 2.0, 3.0, 0.5],
            "assists_per_100": [8.0, 4.0, 5.0, 2.0],
            "turnovers_per_100": [3.0, 2.0, 2.5, 1.0],
            "usage_pct": [30.0, 20.0, 25.0, 15.0],
            "steals_per_100": [2.0, 1.0, 1.5, 0.5],
            "blocks_per_100": [0.5, 1.0, 1.5, 2.0],
            "offensive_rebound_pct": [2.0, 5.0, 8.0, 12.0],
        }
    )
    coefficients = {feature: (index + 1) / 10.0 for index, feature in enumerate(PROFILE_FEATURES)}
    references = {
        feature: float(profiles[column].mean())
        for feature, column in FEATURE_TO_PROFILE_COLUMN.items()
    }
    published = {}
    for row in profiles.itertuples(index=False):
        published[int(row.player_id)] = sum(
            coefficients[feature]
            * (float(getattr(row, column)) - references[feature])
            for feature, column in FEATURE_TO_PROFILE_COLUMN.items()
        )
    return FrozenAdditiveProfileContract(
        season="2026-27",
        core_model_version="NAIL-RAPM v1.2.1.3",
        source_run_id="test-run",
        profiles=profiles,
        coefficients=coefficients,
        reference_values=references,
        published_adjustments=published,
    )


def _players(*, extra_three: float = 0.0) -> pd.DataFrame:
    rows = []
    for side, team_id, ids in [("home", 10, (1, 2)), ("away", 20, (3, 4))]:
        for offset, player_id in enumerate(ids):
            rows.append(
                {
                    "game_id": "g1",
                    "team_side": side,
                    "team_id": team_id,
                    "personId": player_id,
                    "statistics_minutes": "PT24M",
                    "statistics_fieldGoalsAttempted": 10 + offset,
                    "statistics_freeThrowsAttempted": 4,
                    "statistics_threePointersAttempted": 5 + extra_three,
                    "statistics_threePointersMade": 2,
                    "statistics_assists": 3,
                    "statistics_turnovers": 1,
                    "statistics_steals": 1,
                    "statistics_blocks": 1,
                    "statistics_reboundsOffensive": 2,
                    "statistics_reboundsDefensive": 4,
                }
            )
    return pd.DataFrame(rows)


def _stints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "home_player_ids": [(1, 2), (1, 2)],
            "away_player_ids": [(3, 4), (3, 4)],
            "possessions": [20.0, 30.0],
        }
    )


def _contribution(*, extra_three: float = 0.0, source: str = "a"):
    return build_profile_game_contribution(
        _players(extra_three=extra_three),
        _stints(),
        season="2026-27",
        game_id="g1",
        game_time_utc=GAME_TIME,
        finalized_at=FINALIZED_AT,
        source_sha256=source * 64,
        source_uri="local://g1",
    )


def test_zero_evidence_exactly_preserves_every_preseason_profile() -> None:
    contract = _contract()
    state = solve_additive_profile_state(
        contract,
        (),
        cutoff_time=FINALIZED_AT + timedelta(seconds=1),
    )

    for feature in PROFILE_FEATURES:
        np.testing.assert_array_equal(
            state[f"updated_{feature}"], state[f"preseason_{feature}"]
        )
        assert state[f"evidence_weight_{feature}"].eq(0.0).all()
        assert state[f"rating_delta_{feature}"].eq(0.0).all()
    np.testing.assert_array_equal(
        state["updated_additive_profile"], state["preseason_additive_profile"]
    )


def test_game_contribution_uses_box_counts_and_lineup_possessions() -> None:
    contribution = _contribution()
    evidence = contribution.evidence.set_index("player_id")

    assert evidence.loc[1, "on_court_possessions"] == pytest.approx(50.0)
    assert evidence.loc[1, "minutes"] == pytest.approx(24.0)
    assert evidence.loc[1, "three_pointers_attempted"] == pytest.approx(5.0)
    assert evidence.loc[1, "usage_events"] == pytest.approx(10 + 0.44 * 4 + 1)
    assert evidence.loc[1, "usage_opportunities"] > 0
    assert evidence.loc[1, "offensive_rebound_pct_minutes"] > 0


def test_one_game_updates_all_eight_profiles_and_reconstructs_rating_delta() -> None:
    contract = _contract()
    state = solve_additive_profile_state(
        contract,
        (_contribution(),),
        cutoff_time=FINALIZED_AT + timedelta(seconds=1),
    ).set_index("player_id")

    for feature in PROFILE_FEATURES:
        assert state.loc[1, f"evidence_weight_{feature}"] > 0
        assert np.isfinite(state.loc[1, f"posterior_standard_error_{feature}"])
    expected = sum(state.loc[1, f"rating_delta_{feature}"] for feature in PROFILE_FEATURES)
    assert state.loc[1, "additive_profile_delta"] == pytest.approx(expected)
    assert profile_delta_map(state.reset_index())[1] == pytest.approx(expected)


def test_three_point_makes_update_volume_and_accuracy_separately() -> None:
    config = AdditiveProfileUpdaterConfig(
        prior_multipliers={
            "three_pa_per_100": 2.0,
            "three_point_percentage": 3.0,
            "assists_per_100": 1.0,
            "turnovers_per_100": 1.0,
            "usage_pct": 1.0,
            "steals_per_100": 1.0,
            "blocks_per_100": 1.0,
            "offensive_rebound_claim_total": 1.0,
        }
    )
    state = solve_additive_profile_state(
        _contract(),
        (_contribution(),),
        cutoff_time=FINALIZED_AT + timedelta(seconds=1),
        config=config,
    ).set_index("player_id")

    prior_pa = 10.0
    prior_pct = 0.4
    pa_prior_exposure = 29.29 * 2.0
    pct_prior_attempts = 242.61 * 3.0
    expected_pa = (pa_prior_exposure * prior_pa + 100.0 * 5.0) / (
        pa_prior_exposure + 50.0
    )
    expected_pct = (pct_prior_attempts * prior_pct + 2.0) / (
        pct_prior_attempts + 5.0
    )
    assert state.loc[1, "updated_three_pa_per_100"] == pytest.approx(expected_pa)
    assert state.loc[1, "updated_three_pm_per_100"] == pytest.approx(
        expected_pa * expected_pct
    )


def test_strict_finalization_cutoff_and_revision_ledger() -> None:
    contract = _contract()
    contribution = _contribution()
    at_final = solve_additive_profile_state(
        contract, (contribution,), cutoff_time=FINALIZED_AT
    )
    after_final = solve_additive_profile_state(
        contract,
        (contribution,),
        cutoff_time=FINALIZED_AT + timedelta(microseconds=1),
    )
    assert at_final["additive_profile_delta"].eq(0.0).all()
    assert not after_final["additive_profile_delta"].eq(0.0).all()

    corrected = _contribution(extra_three=2.0, source="b")
    ledger = ProfileContributionLedger()
    assert ledger.upsert(contribution) == "added"
    assert ledger.upsert(contribution) == "unchanged"
    assert ledger.upsert(corrected) == "revised"
    assert ledger.values() == (corrected,)


def test_profile_contribution_and_state_publication_are_idempotent(tmp_path: Path) -> None:
    contribution = _contribution()
    path, created = publish_profile_game_contribution(contribution, tmp_path / "games")
    repeated, repeated_created = publish_profile_game_contribution(
        contribution, tmp_path / "games"
    )
    loaded = read_profile_game_contribution(path)

    assert created is True
    assert repeated_created is False
    assert repeated == path
    assert loaded.contribution_sha256 == contribution.contribution_sha256
    pd.testing.assert_frame_equal(loaded.evidence, contribution.evidence)

    state = solve_additive_profile_state(
        _contract(),
        (contribution,),
        cutoff_time=FINALIZED_AT + timedelta(seconds=1),
    )
    state_path, state_created = publish_additive_profile_state(state, tmp_path / "states")
    state_repeated, state_repeated_created = publish_additive_profile_state(
        state, tmp_path / "states"
    )
    assert state_created is True
    assert state_repeated_created is False
    assert state_repeated == state_path


def test_published_contract_reconstructs_breakdown_and_feature_aliases(
    tmp_path: Path,
) -> None:
    contract = _contract()
    profiles_path = tmp_path / "profiles.parquet"
    rankings_path = tmp_path / "rankings.parquet"
    profiles = contract.profiles.assign(target_season="2026-27")
    profiles.to_parquet(profiles_path, index=False)
    rows = []
    for profile in contract.profiles.itertuples(index=False):
        breakdown = []
        total = 0.0
        for feature, column in FEATURE_TO_PROFILE_COLUMN.items():
            value = float(getattr(profile, column))
            reference = float(contract.reference_values[feature])
            contribution = float(contract.coefficients[feature]) * (value - reference)
            total += contribution
            breakdown.append(
                {
                    "feature": feature,
                    "player_value": value,
                    "reference_value": reference,
                    "contribution": contribution,
                }
            )
        rows.append(
            {
                "season": "2026-27",
                "player_id": int(profile.player_id),
                "additive_profile_adjustment": total,
                "additive_profile_breakdown_json": json.dumps(breakdown),
            }
        )
    pd.DataFrame(rows).to_parquet(rankings_path, index=False)
    rankings_path.with_suffix(".json").write_text(json.dumps({"run_id": "published"}))

    loaded = load_published_profile_contract(rankings_path, profiles_path)

    assert loaded.player_ids == contract.player_ids
    assert loaded.coefficients == pytest.approx(contract.coefficients)
    assert loaded.reference_values == pytest.approx(contract.reference_values)
    assert "offensive_rebound_pct" in loaded.profiles
