"""Tests for the constrained Split NAIL precision-surface renderer."""

from __future__ import annotations

import json

import pandas as pd

from nba_lineup_model.modeling.constrained_split_nail_precision_audit import (
    render_precision_surface,
)


def test_render_precision_surface_writes_svg_from_persisted_grid(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "r_player": [0.1, 0.1, 0.5, 0.5],
            "r_context": [1.0, 2.0, 1.0, 2.0],
            "side_scoring_rmse": [2.1, 2.0, 1.9, 2.2],
        }
    ).to_parquet(run_dir / "side_scoring_selection_summary.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps({"selected_r_player": 0.5, "selected_r_context": 1.0})
    )

    chart_path = render_precision_surface(
        run_dir=run_dir,
        chart_path=tmp_path / "precision-selection-surface.svg",
    )

    assert chart_path.is_file()
    assert "Selected: r_player=0.5, r_context=1" in chart_path.read_text()
