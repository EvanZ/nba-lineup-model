from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from nba_lineup_model.season.current_serving import (
    REPORT_FILENAME,
    SERVING_STATE_FILENAME,
    ServingPromotionConfig,
    build_serving_state,
    promote_serving_bundle,
    publish_serving_bundle,
    resolve_latest_bundle,
    rollback_serving_bundle,
    verify_serving_bundle,
)

SEASON = "2026-27"
CUTOFF = datetime(2026, 10, 27, 7, 0, tzinfo=UTC)
CONFIG = ServingPromotionConfig(preseason_run_id="promoted-preseason-run")


def _rating_state() -> pd.DataFrame:
    games = json.dumps(["0022600001", "0022600002"])
    lineage = json.dumps(["a" * 64, "b" * 64])
    return pd.DataFrame(
        {
            "schema_version": 1,
            "season": SEASON,
            "core_model_version": "nail-v1.2.1.3",
            "state_updater_version": "in-season-nail-v1",
            "player_id": [1, 2, 3],
            "team_at_cutoff": ["GSW", "GSW", "LAL"],
            "cutoff_time": [CUTOFF] * 3,
            "cutoff_game_id": ["0022600002"] * 3,
            "frozen_preseason_nail": [3.0, 1.0, -0.5],
            "current_season_residual": [0.2, -0.1, 0.0],
            "updated_nail": [3.2, 0.9, -0.5],
            "raw_on_court_possessions": [140.0, 110.0, 0.0],
            "effective_information": [70.0, 55.0, 0.0],
            "prior_equivalent_possessions": [6000.0] * 3,
            "prior_weight": [0.98, 0.985, 1.0],
            "evidence_weight": [0.02, 0.015, 0.0],
            "posterior_standard_error": [0.1, 0.1, 0.1],
            "included_game_ids_json": [games] * 3,
            "contribution_lineage_json": [lineage] * 3,
        }
    )


def _minutes_state() -> pd.DataFrame:
    games = json.dumps(["0022600001", "0022600002"])
    lineage = json.dumps(["c" * 64, "d" * 64])
    return pd.DataFrame(
        {
            "schema_version": [2] * 3,
            "season": [SEASON] * 3,
            "updater_version": ["in-season-minutes-v1"] * 3,
            "player_id": [1, 2, 3],
            "player_name": ["Alpha", "Beta", "Gamma"],
            "team": ["GSW", "GSW", "LAL"],
            "preseason_participation_prior": [0.9, 0.8, 0.7],
            "current_participation_estimate": [0.95, 0.9, 0.7],
            "participation_observations": [2, 2, 0],
            "preseason_conditional_minutes_prior": [34.0, 25.0, 18.0],
            "preseason_normalized_minutes": [30.6, 20.0, 12.6],
            "current_conditional_minutes_estimate": [35.0, 22.0, 18.0],
            "conditional_observations": [2, 2, 0],
            "conditional_minutes_weighted_sum": [70.0, 44.0, 0.0],
            "effective_conditional_weight": [2.0, 2.0, 0.0],
            "conditional_window_observations": [2, 2, 0],
            "conditional_history_json": ["[]"] * 3,
            "included_game_ids_json": [games] * 3,
            "contribution_lineage_json": [lineage] * 3,
            "cutoff_time": [CUTOFF.isoformat()] * 3,
            "prior_participation_weight": [4.0] * 3,
            "prior_conditional_weight": [0.2] * 3,
        }
    )


def _write_sources(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rating = tmp_path / "rating.parquet"
    minutes = tmp_path / "minutes.parquet"
    _rating_state().to_parquet(rating, index=False)
    _minutes_state().to_parquet(minutes, index=False)
    return rating, minutes


def _publish(tmp_path: Path, *, root: Path | None = None):
    rating, minutes = _write_sources(tmp_path)
    return publish_serving_bundle(
        rating,
        minutes,
        root=root or (tmp_path / "serving"),
        season=SEASON,
        evidence_through_date=date(2026, 10, 26),
        as_of_cutoff_time=CUTOFF,
        config=CONFIG,
        rating_source_uri="s3://example/rating.parquet",
        minutes_source_uri="s3://example/minutes.parquet",
        generated_at=datetime(2026, 10, 27, 8, 0, tzinfo=UTC),
    )


def test_build_serving_state_preserves_zero_evidence_priors() -> None:
    state, audit = build_serving_state(
        _rating_state(),
        _minutes_state(),
        season=SEASON,
        evidence_through_date=date(2026, 10, 26),
        as_of_cutoff_time=CUTOFF,
        config=CONFIG,
    )

    gamma = state.loc[state["player_id"].eq(3)].iloc[0]
    assert gamma["updated_nail"] == gamma["frozen_preseason_nail"]
    assert (
        gamma["current_conditional_minutes_estimate"]
        == gamma["preseason_conditional_minutes_prior"]
    )
    assert audit["source_game_ids"] == ["0022600001", "0022600002"]
    assert audit["team_count"] == 2


def test_build_serving_state_rejects_unexplained_desynchronized_games() -> None:
    minutes = _minutes_state()
    minutes["included_game_ids_json"] = json.dumps(["0022600001"])

    with pytest.raises(ValueError, match="unexplained game gaps"):
        build_serving_state(
            _rating_state(),
            minutes,
            season=SEASON,
            evidence_through_date=date(2026, 10, 26),
            as_of_cutoff_time=CUTOFF,
            config=CONFIG,
        )


def test_build_serving_state_allows_named_game_quarantine() -> None:
    minutes = _minutes_state()
    minutes["included_game_ids_json"] = json.dumps(["0022600001"])

    _, audit = build_serving_state(
        _rating_state(),
        minutes,
        season=SEASON,
        evidence_through_date=date(2026, 10, 26),
        as_of_cutoff_time=CUTOFF,
        config=CONFIG,
        source_exclusions={
            "ratings": [],
            "minutes": [
                {
                    "game_id": "0022600002",
                    "reason": "invalid substitution sequence",
                }
            ],
        },
    )

    assert audit["source_game_ids"] == ["0022600001", "0022600002"]
    assert audit["quality_issue_count"] == 1


def test_daily_report_surfaces_quarantined_game(tmp_path: Path) -> None:
    rating, minutes = _write_sources(tmp_path)
    frame = pd.read_parquet(minutes)
    frame["included_game_ids_json"] = json.dumps(["0022600001"])
    frame.to_parquet(minutes, index=False)

    bundle = publish_serving_bundle(
        rating,
        minutes,
        root=tmp_path / "serving",
        season=SEASON,
        evidence_through_date=date(2026, 10, 26),
        as_of_cutoff_time=CUTOFF,
        config=CONFIG,
        rating_source_uri="rating",
        minutes_source_uri="minutes",
        source_exclusions={
            "ratings": [],
            "minutes": [
                {
                    "game_id": "0022600002",
                    "reason": "invalid substitution sequence",
                }
            ],
        },
        source_coverage={
            "ratings": {"games": 2, "excluded": 0},
            "minutes": {"games": 2, "excluded": 1},
        },
    )

    report = (bundle.path / REPORT_FILENAME).read_text()
    assert "Daily pipeline quality" in report
    assert "0022600002" in report
    assert "invalid substitution sequence" in report
    assert bundle.manifest["identity"]["data_quality"]["status"] == "warnings"


def test_publish_serving_bundle_is_content_addressed_and_idempotent(
    tmp_path: Path,
) -> None:
    first = _publish(tmp_path)
    second = publish_serving_bundle(
        tmp_path / "rating.parquet",
        tmp_path / "minutes.parquet",
        root=tmp_path / "serving",
        season=SEASON,
        evidence_through_date=date(2026, 10, 26),
        as_of_cutoff_time=CUTOFF,
        config=CONFIG,
        rating_source_uri="s3://example/rating.parquet",
        minutes_source_uri="s3://example/minutes.parquet",
        generated_at=datetime(2026, 10, 27, 9, 0, tzinfo=UTC),
    )

    assert first.created is True
    assert second.created is False
    assert second.path == first.path
    assert first.path.name == f"bundle_sha256={first.manifest['bundle_sha256']}"
    assert (first.path / SERVING_STATE_FILENAME).is_file()
    assert (first.path / REPORT_FILENAME).is_file()
    assert verify_serving_bundle(first.path) == first.manifest


def test_manifest_records_team_audit_difference(tmp_path: Path) -> None:
    minutes = _minutes_state()
    minutes.loc[minutes["player_id"].eq(2), "team"] = "PHX"
    rating_path = tmp_path / "rating.parquet"
    minutes_path = tmp_path / "minutes.parquet"
    _rating_state().to_parquet(rating_path, index=False)
    minutes.to_parquet(minutes_path, index=False)

    bundle = publish_serving_bundle(
        rating_path,
        minutes_path,
        root=tmp_path / "serving",
        season=SEASON,
        evidence_through_date=date(2026, 10, 26),
        as_of_cutoff_time=CUTOFF,
        config=CONFIG,
        rating_source_uri="rating",
        minutes_source_uri="minutes",
    )

    assert bundle.manifest["identity"]["coverage"]["rating_team_mismatch_count"] == 1


def test_promote_and_rollback_serving_bundle(tmp_path: Path) -> None:
    root = tmp_path / "serving"
    first = _publish(tmp_path / "first", root=root)
    promote_serving_bundle(
        first.path,
        root=root,
        promoted_at=datetime(2026, 10, 27, 8, 0, tzinfo=UTC),
    )
    assert resolve_latest_bundle(root) == first.path.resolve()

    second_root = tmp_path / "second"
    second_root.mkdir()
    rating, minutes = _write_sources(second_root)
    frame = pd.read_parquet(rating)
    frame.loc[frame["player_id"].eq(1), "current_season_residual"] = 0.3
    frame.loc[frame["player_id"].eq(1), "updated_nail"] = 3.3
    frame.to_parquet(rating, index=False)
    second = publish_serving_bundle(
        rating,
        minutes,
        root=root,
        season=SEASON,
        evidence_through_date=date(2026, 10, 26),
        as_of_cutoff_time=CUTOFF,
        config=CONFIG,
        rating_source_uri="s3://example/rating-v2.parquet",
        minutes_source_uri="s3://example/minutes.parquet",
    )
    _, changed = promote_serving_bundle(second.path, root=root)
    assert changed is True
    assert resolve_latest_bundle(root) == second.path.resolve()

    restored = rollback_serving_bundle(root=root)
    assert restored["bundle_sha256"] == first.manifest["bundle_sha256"]
    assert resolve_latest_bundle(root) == first.path.resolve()


def test_verify_serving_bundle_detects_file_mutation(tmp_path: Path) -> None:
    bundle = _publish(tmp_path)
    with (bundle.path / REPORT_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write("changed")

    with pytest.raises(ValueError, match="checksum changed"):
        verify_serving_bundle(bundle.path)
