"""Forward-only shrinkage selection for assisted-shot player profile rates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nba_lineup_model.season.schema import validate_season

DEFAULT_ASSISTED_SHOT_PATH = Path(
    "data/analytical/assisted_shot_taxonomy/player_season_assisted_shot_profiles.parquet"
)
DEFAULT_OUTPUT_PATH = Path("artifacts/reports/assisted_shot_padding_selection.parquet")
PSEUDO_POSSESSION_GRID = (0.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0)
QUALIFIED_POSSESSIONS = 500.0
FEATURE_COLUMNS = (
    "unassisted_rim_makes",
    "unassisted_three_makes",
)
CENTER_MODES = ("possession_weighted_mean", "qualified_player_median", "qualified_weighted_median")


@dataclass(frozen=True)
class AssistedShotPaddingSelection:
    """Frozen shrinkage specification for a new assisted-shot rate feature."""

    feature: str
    center_mode: str
    pseudo_possessions: float
    weighted_rmse: float
    source_season_count: int
    player_season_count: int


def select_assisted_shot_padding(
    *,
    assisted_shot_path: Path | str = DEFAULT_ASSISTED_SHOT_PATH,
    through_source_season: str = "2022-23",
) -> pd.DataFrame:
    """Select source-only centers and pads by next-season rate prediction."""

    through = validate_season(through_source_season)
    profiles = pd.read_parquet(assisted_shot_path)
    required = {"season", "player_id", "on_court_possessions", *FEATURE_COLUMNS}
    missing = required - set(profiles)
    if missing:
        raise ValueError(f"Assisted shot profiles missing columns: {sorted(missing)}")
    profiles = profiles.loc[:, ["season", "player_id", "on_court_possessions", *FEATURE_COLUMNS]].copy()
    profiles["season"] = profiles["season"].astype(str)
    profiles["player_id"] = profiles["player_id"].astype("int64")
    profiles["on_court_possessions"] = pd.to_numeric(
        profiles["on_court_possessions"], errors="raise"
    ).astype(float)
    rows: list[dict[str, object]] = []
    source_seasons = sorted(
        (season for season in profiles["season"].unique() if season <= through),
        key=lambda season: int(season[:4]),
    )
    for feature in FEATURE_COLUMNS:
        observations = _forward_observations(profiles, feature, source_seasons)
        for center_mode in CENTER_MODES:
            mode_observations = observations.loc[observations["center_mode"].eq(center_mode)].copy()
            for pseudo_possessions in PSEUDO_POSSESSION_GRID:
                predicted = _shrunk_rate(
                    mode_observations["source_count"].to_numpy(dtype=float),
                    mode_observations["source_possessions"].to_numpy(dtype=float),
                    mode_observations["reference_rate"].to_numpy(dtype=float),
                    pseudo_possessions,
                )
                actual = 100.0 * mode_observations["target_count"].to_numpy(dtype=float) / mode_observations[
                    "target_possessions"
                ].to_numpy(dtype=float)
                weights = mode_observations["target_possessions"].to_numpy(dtype=float)
                rows.append(
                    {
                        "feature": feature,
                        "center_mode": center_mode,
                        "pseudo_possessions": pseudo_possessions,
                        "weighted_rmse": float(np.sqrt(np.average((predicted - actual) ** 2, weights=weights))),
                        "source_season_count": int(mode_observations["source_season"].nunique()),
                        "player_season_count": int(len(mode_observations)),
                    }
                )
            mode_rows = rows[-len(PSEUDO_POSSESSION_GRID) :]
            reference = _references_by_season(profiles, feature, center_mode)
            for row in mode_rows:
                row["reference_rate_summary"] = float(reference.mean())
    results = pd.DataFrame(rows).sort_values(
        ["feature", "weighted_rmse", "pseudo_possessions", "center_mode"], kind="stable"
    )
    results["rank"] = results.groupby("feature", sort=False).cumcount() + 1
    return results.reset_index(drop=True)


def _forward_observations(
    profiles: pd.DataFrame,
    feature: str,
    source_seasons: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for source_season in source_seasons:
        target_season = _next_season(source_season)
        source = profiles.loc[profiles["season"].eq(source_season)].copy()
        target = profiles.loc[profiles["season"].eq(target_season)].copy()
        if target.empty:
            continue
        references = {
            mode: _reference_rate(source, feature, mode) for mode in CENTER_MODES
        }
        joined = source.merge(
            target.loc[:, ["player_id", "on_court_possessions", feature]],
            on="player_id",
            how="inner",
            suffixes=("_source", "_target"),
            validate="one_to_one",
        )
        if joined.empty:
            continue
        for mode, reference in references.items():
            rows.append(
                pd.DataFrame(
                    {
                        "source_season": source_season,
                        "center_mode": mode,
                        "source_count": joined[f"{feature}_source"].to_numpy(dtype=float),
                        "source_possessions": joined["on_court_possessions_source"].to_numpy(dtype=float),
                        "target_count": joined[f"{feature}_target"].to_numpy(dtype=float),
                        "target_possessions": joined["on_court_possessions_target"].to_numpy(dtype=float),
                        "reference_rate": reference,
                    }
                )
            )
    if not rows:
        raise ValueError(f"No forward observations for {feature}")
    return pd.concat(rows, ignore_index=True)


def _references_by_season(profiles: pd.DataFrame, feature: str, center_mode: str) -> pd.Series:
    return profiles.groupby("season", sort=False).apply(
        lambda members: _reference_rate(members, feature, center_mode), include_groups=False
    )


def _reference_rate(frame: pd.DataFrame, feature: str, center_mode: str) -> float:
    counts = pd.to_numeric(frame[feature], errors="raise").to_numpy(dtype=float)
    possessions = pd.to_numeric(frame["on_court_possessions"], errors="raise").to_numpy(dtype=float)
    rates = 100.0 * counts / possessions
    if center_mode == "possession_weighted_mean":
        return float(100.0 * counts.sum() / possessions.sum())
    qualified = possessions >= QUALIFIED_POSSESSIONS
    if not qualified.any():
        raise ValueError(f"No qualified players for {feature} reference")
    if center_mode == "qualified_player_median":
        return float(np.median(rates[qualified]))
    if center_mode == "qualified_weighted_median":
        return _weighted_median(rates[qualified], possessions[qualified])
    raise ValueError(f"Unknown assisted-shot center mode: {center_mode}")


def _shrunk_rate(
    counts: np.ndarray,
    possessions: np.ndarray,
    reference_rate: np.ndarray,
    pseudo_possessions: float,
) -> np.ndarray:
    return 100.0 * (counts + pseudo_possessions * reference_rate / 100.0) / (
        possessions + pseudo_possessions
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2.0, side="left")])


def _next_season(season: str) -> str:
    start = int(validate_season(season)[:4])
    return f"{start + 1}-{str(start + 2)[-2:]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Select assisted-shot profile shrinkage")
    parser.add_argument("--assisted-shot-path", default=str(DEFAULT_ASSISTED_SHOT_PATH))
    parser.add_argument("--through-source-season", default="2022-23")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    results = select_assisted_shot_padding(
        assisted_shot_path=args.assisted_shot_path,
        through_source_season=args.through_source_season,
    )
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output, index=False)
    print(results.loc[results["rank"].eq(1)].to_string(index=False))


if __name__ == "__main__":
    main()
