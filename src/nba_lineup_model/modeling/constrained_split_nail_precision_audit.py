"""Render the joint constrained-split NAIL precision-selection surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.constrained_split_nail import MODEL_NAME

DEFAULT_SEASON = "2025-26"
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_CHART_PATH = Path(
    "docs/assets/images/constrained-split-nail/precision-selection-surface.svg"
)


def _latest_run_dir(*, artifacts_dir: Path, season: str) -> Path:
    root = artifacts_dir / MODEL_NAME / season
    latest = root / "latest.json"
    if not latest.is_file():
        raise FileNotFoundError(f"No latest constrained-split run at {latest}")
    run_id = str(json.loads(latest.read_text())["run_id"])
    return root / run_id


def render_precision_surface(*, run_dir: Path | str, chart_path: Path | str) -> Path:
    """Render the observed joint precision grid without interpolating new candidates."""

    source = Path(run_dir)
    output = Path(chart_path)
    summary = pd.read_parquet(source / "side_scoring_selection_summary.parquet")
    metadata = json.loads((source / "metadata.json").read_text())

    r_player = np.sort(summary["r_player"].unique().astype(float))
    r_context = np.sort(summary["r_context"].unique().astype(float))
    matrix = (
        summary.pivot(index="r_player", columns="r_context", values="side_scoring_rmse")
        .reindex(index=r_player, columns=r_context)
        .to_numpy(dtype=float)
    )
    selected_player = float(metadata["selected_r_player"])
    selected_context = float(metadata["selected_r_context"])
    selected_rmse = float(
        summary.loc[
            summary["r_player"].eq(selected_player) & summary["r_context"].eq(selected_context),
            "side_scoring_rmse",
        ].iloc[0]
    )

    figure, axis = plt.subplots(figsize=(9.4, 6.6), layout="constrained")
    mesh = axis.pcolormesh(
        np.log10(r_context),
        np.log10(r_player),
        matrix,
        shading="nearest",
        cmap="Blues_r",
    )
    contour = axis.contour(
        np.log10(r_context),
        np.log10(r_player),
        matrix,
        colors="#17231c",
        linewidths=0.8,
        levels=6,
    )
    axis.clabel(contour, inline=True, fontsize=8, fmt="%.4f")
    axis.scatter(
        np.log10(selected_context),
        np.log10(selected_player),
        color="#e8502f",
        edgecolors="white",
        linewidths=1.4,
        s=90,
        zorder=3,
        label=(
            f"Selected: r_player={selected_player:g}, "
            f"r_context={selected_context:g}"
        ),
    )
    axis.set_xticks(np.log10(r_context), [f"{value:g}" for value in r_context])
    axis.set_yticks(np.log10(r_player), [f"{value:g}" for value in r_player])
    axis.set_xlabel("Context specialization precision (r_context)")
    axis.set_ylabel("Player specialization precision (r_player)")
    axis.set_title(
        "Constrained Split NAIL joint precision selection",
        loc="left",
        fontweight="bold",
    )
    axis.text(
        0,
        1.02,
        (
            f"Observed held-out scoring-side RMSE. Best grid point: {selected_rmse:.4f}; "
            "axes are log-spaced."
        ),
        transform=axis.transAxes,
        fontsize=9,
        color="#5f6860",
    )
    axis.legend(loc="upper right", frameon=False, fontsize=8.5)
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.02)
    colorbar.set_label("Scoring-side RMSE (lower is better)")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render constrained Split NAIL precision surface")
    parser.add_argument("--run-dir")
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--chart-path", default=str(DEFAULT_CHART_PATH))
    args = parser.parse_args()
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else _latest_run_dir(artifacts_dir=Path(args.artifacts_dir), season=args.season)
    )
    chart_path = render_precision_surface(run_dir=run_dir, chart_path=args.chart_path)
    print(f"Constrained Split NAIL precision surface: {chart_path}")


if __name__ == "__main__":
    main()
