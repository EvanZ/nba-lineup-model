from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.modeling.nail_centering_audit import (
    build_centering_audit,
    write_centering_audit,
)


def _catalogs() -> tuple[pd.DataFrame, pd.DataFrame]:
    player_ids = np.arange(1, 13)
    completed_total = np.linspace(-2.0, 3.5, len(player_ids))
    completed_additive = np.linspace(-0.6, 0.5, len(player_ids))
    forward_total = 0.8 * completed_total - 0.4
    forward_additive = 0.7 * completed_additive - 0.1
    completed = pd.DataFrame(
        {
            "season": "2025-26",
            "player_id": player_ids,
            "player_name": [f"Player {value}" for value in player_ids],
            "rapm": completed_total,
            "additive_profile_adjustment": completed_additive,
            "on_court_possessions": np.arange(100.0, 1300.0, 100.0),
        }
    )
    forward = pd.DataFrame(
        {
            "season": "2026-27",
            "player_id": player_ids,
            "player_name": [f"Player {value}" for value in player_ids],
            "rapm": forward_total,
            "prior_rating": forward_total - forward_additive,
            "additive_profile_adjustment": forward_additive,
        }
    )
    return completed, forward


def test_common_reference_centers_both_vectors_and_reconciles_components() -> None:
    completed, forward = _catalogs()

    audit = build_centering_audit(completed, forward, invariance_samples=100)
    bridge = audit.player_bridge
    weights = bridge["reference_weight"]

    assert np.average(bridge["completed_common_total"], weights=weights) == pytest.approx(0.0)
    assert np.average(bridge["forward_common_total"], weights=weights) == pytest.approx(0.0)
    assert bridge["completed_reconciliation_error"].abs().max() < 1e-12
    assert bridge["forward_reconciliation_error"].abs().max() < 1e-12
    assert audit.diagnostics["passed"].all()


def test_reference_correction_accounts_for_center_difference() -> None:
    completed, forward = _catalogs()

    audit = build_centering_audit(completed, forward, invariance_samples=50)
    bridge = audit.player_bridge

    expected = bridge["published_total_change"] + bridge["reference_frame_correction"]
    assert bridge["common_reference_total_change"].to_numpy() == pytest.approx(expected)
    assert bridge["reference_frame_correction"].nunique() == 1


def test_uniform_translation_does_not_change_five_on_five_margin() -> None:
    completed, forward = _catalogs()

    audit = build_centering_audit(completed, forward, invariance_samples=500)

    assert audit.invariance["max_abs_margin_change"].max() < 1e-12


def test_missing_additive_profile_is_folded_into_scalar() -> None:
    completed, forward = _catalogs()
    completed.loc[0, "additive_profile_adjustment"] = np.nan
    forward.loc[0, "additive_profile_adjustment"] = np.nan

    audit = build_centering_audit(completed, forward, invariance_samples=20)
    player = audit.player_bridge.loc[audit.player_bridge["player_id"].eq(1)].iloc[0]

    assert player["completed_additive"] == 0.0
    assert player["completed_scalar"] == player["completed_total"]
    assert player["forward_additive"] == 0.0
    assert player["forward_scalar"] == player["forward_total"]


def test_duplicate_catalog_rows_are_rejected() -> None:
    completed, forward = _catalogs()
    completed = pd.concat([completed, completed.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicates"):
        build_centering_audit(completed, forward)


def test_writer_produces_browser_report_and_machine_readable_tables(tmp_path) -> None:
    completed, forward = _catalogs()
    completed.loc[0, "player_id"] = 201939
    completed.loc[0, "player_name"] = "Stephen Curry"
    forward.loc[0, "player_id"] = 201939
    forward.loc[0, "player_name"] = "Stephen Curry"
    audit = build_centering_audit(completed, forward, invariance_samples=20)

    result = write_centering_audit(
        audit,
        completed_path=tmp_path / "completed.parquet",
        forward_path=tmp_path / "forward.parquet",
        output_dir=tmp_path / "audit",
    )

    assert (result.run_dir / "player_bridge.parquet").is_file()
    assert (result.run_dir / "centers.parquet").is_file()
    assert (result.run_dir / "diagnostics.parquet").is_file()
    report = (result.run_dir / "report.html").read_text()
    assert "Did the ruler move?" in report
    assert "Stephen Curry bridge" in report
