"""Diagnostic charts for the forward component-native Four Factors state."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FACTOR_LABELS = {
    "efg_pct": "eFG%",
    "tov_pct": "Turnover avoidance",
    "orb_pct": "ORB%",
    "ftr": "FTR",
}
OFFENSE_COLOR = "#2C6E9A"
DEFENSE_COLOR = "#D9772A"


def build_forward_four_factor_od_diagnostics(
    run_dir: Path | str,
    *,
    output_dir: Path | str = Path("docs/assets/images/four-factors"),
) -> dict[str, Path]:
    """Render parameter, dispersion, and schedule-control time series."""

    root = Path(run_dir)
    effects = pd.read_parquet(root / "season_factor_side_effects.parquet")
    controls = pd.read_parquet(root / "season_factor_side_controls.parquet")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return {
        "penalties": _render_penalties(effects, destination / "four-factor-od-penalties.png"),
        "dispersion": _render_dispersion(effects, destination / "four-factor-od-dispersion.png"),
        "controls": _render_controls(controls, destination / "four-factor-od-controls.png"),
    }


def _render_penalties(effects: pd.DataFrame, output: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for axis, factor in zip(axes.flat, FACTOR_LABELS, strict=True):
        frame = effects.loc[effects["factor"].eq(factor)].drop_duplicates("season")
        axis.plot(frame["season"], frame["selected_offense_lambda"], color=OFFENSE_COLOR, label="Offense")
        axis.plot(frame["season"], frame["selected_defense_lambda"], color=DEFENSE_COLOR, label="Defense")
        axis.set_title(FACTOR_LABELS[factor])
        axis.set_yscale("log")
        axis.set_ylabel("Selected ridge penalty")
        axis.tick_params(axis="x", rotation=45, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Forward Four Factors O/D: independently selected penalties")
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def _render_dispersion(effects: pd.DataFrame, output: Path) -> Path:
    rows: list[dict[str, object]] = []
    for (season, factor), frame in effects.groupby(["season", "factor"], sort=True):
        for side in ("offense", "defense"):
            value = frame[f"{side}_effect_percentage_points"].to_numpy(dtype=float)
            weight = frame[f"{side}_opportunities"].to_numpy(dtype=float)
            mean = np.average(value, weights=weight)
            rows.append(
                {
                    "season": season,
                    "factor": factor,
                    "side": side,
                    "weighted_sd": float(np.sqrt(np.average((value - mean) ** 2, weights=weight))),
                }
            )
    dispersion = pd.DataFrame(rows)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for axis, factor in zip(axes.flat, FACTOR_LABELS, strict=True):
        frame = dispersion.loc[dispersion["factor"].eq(factor)]
        for side, color in (("offense", OFFENSE_COLOR), ("defense", DEFENSE_COLOR)):
            series = frame.loc[frame["side"].eq(side)]
            axis.plot(series["season"], series["weighted_sd"], color=color, label=side.title())
        axis.set_title(FACTOR_LABELS[factor])
        axis.set_ylabel("Opportunity-weighted player SD (pp)")
        axis.tick_params(axis="x", rotation=45, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Forward Four Factors O/D: player-effect dispersion")
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def _render_controls(controls: pd.DataFrame, output: Path) -> Path:
    columns = (
        ("home_court_percentage_points", "Home court"),
        ("own_back_to_back_percentage_points", "Own back-to-back"),
        ("opponent_back_to_back_percentage_points", "Opponent back-to-back"),
    )
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for axis, (column, label) in zip(axes, columns, strict=True):
        for factor, frame in controls.groupby("factor", sort=False):
            axis.plot(frame["season"], frame[column], label=FACTOR_LABELS[factor])
        axis.axhline(0.0, color="#66716e", linewidth=0.8)
        axis.set_ylabel(f"{label}\n(pp)")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=4, frameon=False, fontsize=8)
    axes[-1].tick_params(axis="x", rotation=45, labelsize=8)
    figure.suptitle("Forward Four Factors O/D: factor-side schedule controls")
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render forward Four Factors O/D diagnostics")
    parser.add_argument("run_dir")
    parser.add_argument("--output-dir", default="docs/assets/images/four-factors")
    args = parser.parse_args()
    for path in build_forward_four_factor_od_diagnostics(
        args.run_dir, output_dir=args.output_dir
    ).values():
        print(path)


if __name__ == "__main__":
    main()
