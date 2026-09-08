"""Discover stable five-man residual archetypes without changing NAIL."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
)
from nba_lineup_model.modeling.frozen_feature_screen import (
    DEFAULT_SEASONS,
    build_frozen_feature_screen,
)
from nba_lineup_model.modeling.replacement_level import prepare_player_exposure_cohort
from nba_lineup_model.modeling.stints import read_rapm_stints

DEFAULT_OUTPUT_ROOT = Path("artifacts/models/analysis/lineup_residual_archetypes")
DEFAULT_CHART_PATH = Path("docs/assets/images/lineup-residual-archetypes/profile-differences.svg")
MINIMUM_POSSESSIONS = 250.0


def build_lineup_residual_archetype_audit(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    minimum_possessions: float = MINIMUM_POSSESSIONS,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    chart_path: Path | str = DEFAULT_CHART_PATH,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
) -> Path:
    """Persist exposure-shrunk observed-unit residuals and profile contrasts."""

    if minimum_possessions <= 0:
        raise ValueError("minimum_possessions must be positive")
    screen = build_frozen_feature_screen(
        "primary_usage_shooting_alignment", seasons=seasons
    )
    residuals = pd.read_parquet(screen.run_dir / "stint_residuals.parquet")
    panel = pd.read_parquet(panel_path)
    cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(max(seasons))],
        through_season=max(seasons),
        analytical_dir=analytical_dir,
    )
    profile_builder = partial(
        build_contextual_player_profiles,
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
        use_last_observed_profile=True,
    )
    rows: list[pd.DataFrame] = []
    for season in seasons:
        stints = read_rapm_stints(season, analytical_dir=analytical_dir)
        scored = residuals.loc[residuals["season"].eq(season)].merge(
            stints.loc[
                :,
                [
                    "game_id", "stint_index", "home_team_tricode", "away_team_tricode",
                    "home_player_ids", "away_player_ids",
                ],
            ],
            on=["game_id", "stint_index"],
            how="inner",
            validate="one_to_one",
        )
        for side, sign in (("home", 1.0), ("away", -1.0)):
            frame = scored.loc[:, ["season", "possessions", "frozen_residual_net_rating"]].copy()
            frame["team"] = scored[f"{side}_team_tricode"].astype(str)
            frame["lineup_player_ids"] = scored[f"{side}_player_ids"].map(_lineup_key)
            frame["lineup_residual_net_rating"] = sign * frame["frozen_residual_net_rating"]
            rows.append(frame)
    observations = pd.concat(rows, ignore_index=True)
    lineups = _shrink_lineup_residuals(observations, minimum_possessions=minimum_possessions)

    profile_rows: list[pd.DataFrame] = []
    for season, group in lineups.groupby("season", sort=True):
        player_ids = sorted(
            {player_id for lineup in group["lineup_player_ids"] for player_id in lineup}
        )
        profiles = profile_builder(
            panel,
            target_season=season,
            target_player_ids=player_ids,
            analytical_dir=str(analytical_dir),
            curated_dir=str(curated_dir),
            exposure_cohort=cohort,
        ).set_index("player_id")
        profile_rows.append(_lineup_profile_summary(group, profiles))
    lineups = pd.concat(profile_rows, ignore_index=True)
    contrasts = _archetype_contrasts(lineups)
    _render_contrasts(contrasts, Path(chart_path))

    root = Path(output_root)
    run_id = f"lineup-residual-archetypes-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    lineups.to_parquet(run_dir / "lineup_residuals.parquet", index=False)
    contrasts.to_parquet(run_dir / "archetype_profile_contrasts.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "seasons": list(seasons),
                "minimum_possessions": minimum_possessions,
                "residual_source": str(screen.run_dir),
                "contract": (
                    "Frozen production residuals aggregated by realized five-man unit; "
                    "empirical-Bayes shrinkage ranks descriptive archetypes only."
                ),
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    return run_dir


def _lineup_key(values: object) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in values))  # type: ignore[union-attr]


def _shrink_lineup_residuals(
    observations: pd.DataFrame, *, minimum_possessions: float
) -> pd.DataFrame:
    grouped = observations.groupby(["season", "team", "lineup_player_ids"], sort=False)
    output = grouped.apply(
        lambda frame: pd.Series(
            {
                "possessions": float(frame["possessions"].sum()),
                "games": int(frame.index.size),
                "raw_residual_net_rating": float(
                    np.average(frame["lineup_residual_net_rating"], weights=frame["possessions"])
                ),
            }
        ),
        include_groups=False,
    ).reset_index()
    output = output.loc[output["possessions"].ge(minimum_possessions)].copy()
    residual_values = observations["lineup_residual_net_rating"]
    possession_weights = observations["possessions"]
    baseline = float(np.average(residual_values, weights=possession_weights))
    variance = float(
        np.average((residual_values - baseline) ** 2, weights=possession_weights)
    )
    raw_variance = float(
        np.average(
            (output["raw_residual_net_rating"] - baseline) ** 2,
            weights=output["possessions"],
        )
    )
    noise = variance / float(output["possessions"].mean())
    between = max(raw_variance - noise, 1e-6)
    output["shrinkage_weight"] = output["possessions"] / (
        output["possessions"] + variance / between
    )
    output["shrunk_residual_net_rating"] = baseline + output["shrinkage_weight"] * (
        output["raw_residual_net_rating"] - baseline
    )
    output["residual_decile"] = pd.qcut(
        output["shrunk_residual_net_rating"], 10, labels=False, duplicates="drop"
    ).astype(int) + 1
    return output


def _lineup_profile_summary(lineups: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in lineups.itertuples(index=False):
        unit = profiles.loc[list(row.lineup_player_ids)]
        usage = np.sort(unit["usage_pct"].to_numpy(dtype=float))
        shooting = np.sort(unit["three_pm_per_100"].to_numpy(dtype=float))
        records.append(
            {
                **row._asdict(),
                "max_usage_pct": float(usage[-1]),
                "top_two_usage_share": float(usage[-2:].sum() / usage.sum()),
                "usage_gap": float(usage[-1] - usage[-2]),
                "top_two_three_pm_per_100": float(shooting[-2:].mean()),
                "bottom_two_three_pm_per_100": float(shooting[:2].mean()),
                "top_two_assists_per_100": float(np.sort(unit["assists_per_100"])[-2:].sum()),
                "max_blocks_per_100": float(unit["blocks_per_100"].max()),
            }
        )
    return pd.DataFrame(records)


def _archetype_contrasts(lineups: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "max_usage_pct", "top_two_usage_share", "usage_gap", "top_two_three_pm_per_100",
        "bottom_two_three_pm_per_100", "top_two_assists_per_100", "max_blocks_per_100",
    ]
    rows: list[dict[str, object]] = []
    for season, frame in (*lineups.groupby("season", sort=True), ("pooled", lineups)):
        high = frame.loc[frame["residual_decile"].eq(frame["residual_decile"].max())]
        low = frame.loc[frame["residual_decile"].eq(frame["residual_decile"].min())]
        for column in columns:
            scale = float(frame[column].std(ddof=0))
            rows.append(
                {
                    "season": season,
                    "feature": column,
                    "high_mean": float(high[column].mean()),
                    "low_mean": float(low[column].mean()),
                    "standardized_difference": (
                        float((high[column].mean() - low[column].mean()) / scale)
                        if scale
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _render_contrasts(contrasts: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pooled = contrasts.loc[contrasts["season"].eq("pooled")].sort_values("standardized_difference")
    figure, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")
    colors = np.where(pooled["standardized_difference"].ge(0), "#2f6ea8", "#e8793e")
    axis.barh(pooled["feature"], pooled["standardized_difference"], color=colors)
    axis.axvline(0, color="#1c2522", linewidth=0.9)
    axis.set_xlabel("High-residual minus low-residual lineup mean (pooled SD units)")
    axis.set_title("Exposure-shrunk frozen residual lineup archetypes", weight="bold")
    figure.savefig(path, bbox_inches="tight", format="svg")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exposure-shrunk frozen lineup residual archetypes"
    )
    parser.add_argument("--minimum-possessions", type=float, default=MINIMUM_POSSESSIONS)
    args = parser.parse_args()
    run_dir = build_lineup_residual_archetype_audit(minimum_possessions=args.minimum_possessions)
    print(f"Lineup residual archetype audit: run={run_dir}")


if __name__ == "__main__":
    main()
