"""Audited Four Factors targets aligned to the published RAPM stint partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.stints import modeling_code_fingerprint
from nba_lineup_model.season.compact import (
    read_curated_partition_manifest,
    validate_curated_partition,
)
from nba_lineup_model.season.layout import CuratedDatasetLayout, CuratedPartition
from nba_lineup_model.season.schema import validate_season


FOUR_FACTORS = ("efg_pct", "tov_pct", "orb_pct", "ftr")
COUNT_COLUMNS = (
    "field_goal_attempts",
    "field_goals_made",
    "two_point_makes",
    "three_point_makes",
    "free_throw_attempts",
    "free_throws_made",
    "turnovers",
    "offensive_rebounds",
    "defensive_rebounds",
)
REQUIRED_STINT_COLUMNS = {
    "game_id",
    "stint_index",
    "start_event_index",
    "end_event_index",
    "points_home",
    "points_away",
    "catalog_home_team_id",
    "catalog_away_team_id",
}
REQUIRED_EVENT_COLUMNS = {
    "game_id",
    "event_index",
    "event_type",
    "event_subtype",
    "shot_result",
    "team_id",
}


def build_four_factor_stint_frame(
    lineup_stints: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach event-count Four Factors targets to canonical lineup stints.

    Events are assigned using the same event-index boundaries that define a
    lineup stint. The result intentionally retains count numerators and
    denominators beside rates, so later RAPM fits can choose factor-specific
    opportunity weights without reconstructing the event join.
    """

    missing_stints = REQUIRED_STINT_COLUMNS - set(lineup_stints)
    missing_events = REQUIRED_EVENT_COLUMNS - set(events)
    if missing_stints:
        raise ValueError(f"Lineup stints missing columns: {sorted(missing_stints)}")
    if missing_events:
        raise ValueError(f"Events missing columns: {sorted(missing_events)}")
    if lineup_stints.empty:
        raise ValueError("Four Factors requires at least one lineup stint")
    if lineup_stints[["game_id", "stint_index"]].duplicated().any():
        raise ValueError("Lineup stint keys must be unique")

    output = lineup_stints.copy()
    for side in ("home", "away"):
        for count in COUNT_COLUMNS:
            output[f"{side}_{count}"] = 0

    event_rows = events.copy()
    event_rows["event_index"] = pd.to_numeric(event_rows["event_index"], errors="coerce")
    event_type = event_rows["event_type"].astype("string").str.casefold()
    subtype = event_rows["event_subtype"].astype("string").str.casefold()
    result = event_rows["shot_result"].astype("string").str.casefold()
    final_shot = result.isin(["made", "missed"])
    event_rows["_is_fga"] = event_type.isin(["2pt", "3pt"]) & final_shot
    event_rows["_is_fgm"] = event_rows["_is_fga"] & result.eq("made")
    event_rows["_is_two_made"] = event_type.eq("2pt") & result.eq("made")
    event_rows["_is_three_made"] = event_type.eq("3pt") & result.eq("made")
    event_rows["_is_fta"] = event_type.eq("freethrow") & final_shot
    event_rows["_is_ftm"] = event_rows["_is_fta"] & result.eq("made")
    event_rows["_is_turnover"] = event_type.eq("turnover")
    event_rows["_is_orb"] = event_type.eq("rebound") & subtype.eq("offensive")
    event_rows["_is_drb"] = event_type.eq("rebound") & subtype.eq("defensive")
    flag_columns = {
        "field_goal_attempts": "_is_fga",
        "field_goals_made": "_is_fgm",
        "two_point_makes": "_is_two_made",
        "three_point_makes": "_is_three_made",
        "free_throw_attempts": "_is_fta",
        "free_throws_made": "_is_ftm",
        "turnovers": "_is_turnover",
        "offensive_rebounds": "_is_orb",
        "defensive_rebounds": "_is_drb",
    }
    factor_events = event_rows.loc[event_rows[list(flag_columns.values())].any(axis=1)].copy()

    game_event_audits: dict[str, dict[str, int]] = {}
    grouped_events = {
        str(game_id): frame for game_id, frame in factor_events.groupby("game_id", sort=False)
    }
    for game_id, game_stints in output.groupby("game_id", sort=False):
        game_events = grouped_events.get(str(game_id))
        counts = {
            "factor_event_count": 0 if game_events is None else len(game_events),
            "assigned_factor_event_count": 0,
            "unassigned_factor_event_count": 0,
            "unknown_team_factor_event_count": 0,
        }
        game_event_audits[str(game_id)] = counts
        if game_events is None or game_events.empty:
            continue
        ordered = game_stints.sort_values("start_event_index", kind="stable")
        starts = pd.to_numeric(ordered["start_event_index"], errors="raise").to_numpy()
        ends = pd.to_numeric(ordered["end_event_index"], errors="raise").to_numpy()
        event_index = game_events["event_index"].to_numpy(dtype=float)
        positions = np.searchsorted(starts, event_index, side="right") - 1
        valid = (
            np.isfinite(event_index)
            & (positions >= 0)
            & (positions < len(ordered))
            & (event_index <= ends[np.clip(positions, 0, len(ordered) - 1)])
        )
        counts["unassigned_factor_event_count"] = int((~valid).sum())
        if not valid.any():
            continue

        assigned = game_events.loc[valid].copy()
        selected = ordered.iloc[positions[valid]]
        assigned["_stint_row"] = selected.index.to_numpy()
        team_ids = pd.to_numeric(assigned["team_id"], errors="coerce").to_numpy()
        home_ids = pd.to_numeric(selected["catalog_home_team_id"], errors="raise").to_numpy()
        away_ids = pd.to_numeric(selected["catalog_away_team_id"], errors="raise").to_numpy()
        assigned["_side"] = np.where(team_ids == home_ids, "home", np.where(team_ids == away_ids, "away", ""))
        counts["unknown_team_factor_event_count"] = int(assigned["_side"].eq("").sum())
        assigned = assigned.loc[assigned["_side"].ne("")]
        counts["assigned_factor_event_count"] = len(assigned)
        for side in ("home", "away"):
            side_events = assigned.loc[assigned["_side"].eq(side)]
            for count, flag in flag_columns.items():
                totals = side_events.loc[side_events[flag]].groupby("_stint_row").size()
                if not totals.empty:
                    output.loc[totals.index, f"{side}_{count}"] = totals.astype("int64")

    for side, opponent in (("home", "away"), ("away", "home")):
        output[f"{side}_efg_numerator"] = (
            output[f"{side}_field_goals_made"]
            + 0.5 * output[f"{side}_three_point_makes"]
        )
        output[f"{side}_efg_denominator"] = output[f"{side}_field_goal_attempts"]
        output[f"{side}_tov_numerator"] = output[f"{side}_turnovers"]
        output[f"{side}_tov_denominator"] = (
            output[f"{side}_field_goal_attempts"]
            + 0.44 * output[f"{side}_free_throw_attempts"]
            + output[f"{side}_turnovers"]
        )
        output[f"{side}_orb_numerator"] = output[f"{side}_offensive_rebounds"]
        output[f"{side}_orb_denominator"] = (
            output[f"{side}_offensive_rebounds"]
            + output[f"{opponent}_defensive_rebounds"]
        )
        output[f"{side}_ftr_numerator"] = output[f"{side}_free_throw_attempts"]
        output[f"{side}_ftr_denominator"] = output[f"{side}_field_goal_attempts"]
        output[f"{side}_efg_pct"] = _safe_rate(
            output[f"{side}_efg_numerator"], output[f"{side}_efg_denominator"]
        )
        output[f"{side}_tov_pct"] = _safe_rate(
            output[f"{side}_tov_numerator"], output[f"{side}_tov_denominator"]
        )
        output[f"{side}_orb_pct"] = _safe_rate(
            output[f"{side}_orb_numerator"], output[f"{side}_orb_denominator"]
        )
        output[f"{side}_ftr"] = _safe_rate(
            output[f"{side}_ftr_numerator"], output[f"{side}_ftr_denominator"]
        )
        output[f"{side}_points_from_events"] = (
            2 * output[f"{side}_two_point_makes"]
            + 3 * output[f"{side}_three_point_makes"]
            + output[f"{side}_free_throws_made"]
        )
        output[f"{side}_point_residual"] = (
            pd.to_numeric(output[f"points_{side}"], errors="raise")
            - output[f"{side}_points_from_events"]
        )

    audit = (
        output.groupby("game_id", as_index=False)
        .agg(
            home_points=("points_home", "sum"),
            away_points=("points_away", "sum"),
            home_points_from_events=("home_points_from_events", "sum"),
            away_points_from_events=("away_points_from_events", "sum"),
            home_point_residual=("home_point_residual", "sum"),
            away_point_residual=("away_point_residual", "sum"),
            stint_count=("stint_index", "size"),
        )
        .sort_values("game_id", kind="stable")
        .reset_index(drop=True)
    )
    for column in (
        "factor_event_count",
        "assigned_factor_event_count",
        "unassigned_factor_event_count",
        "unknown_team_factor_event_count",
    ):
        audit[column] = audit["game_id"].astype(str).map(
            lambda game_id: game_event_audits[game_id][column]
        )
    audit["is_four_factor_eligible"] = (
        audit["home_point_residual"].eq(0)
        & audit["away_point_residual"].eq(0)
        & audit["unassigned_factor_event_count"].eq(0)
        & audit["unknown_team_factor_event_count"].eq(0)
    )
    audit["four_factor_quality_status"] = np.select(
        [
            audit["is_four_factor_eligible"],
            audit["unassigned_factor_event_count"].gt(0),
            audit["unknown_team_factor_event_count"].gt(0),
        ],
        ["clean", "unassigned_factor_event", "unknown_factor_event_team"],
        default="point_reconciliation_failure",
    )
    quality = audit.set_index("game_id")[
        ["is_four_factor_eligible", "four_factor_quality_status"]
    ]
    output = output.join(quality, on="game_id")
    return output, audit


def build_four_factor_stint_dataset(
    season: str,
    *,
    curated_dir: Path | str = Path("data/curated"),
    analytical_dir: Path | str = Path("data/analytical"),
) -> Path:
    """Materialize one season's audited Four Factors stint mart atomically."""

    season = validate_season(season)
    curated_root = Path(curated_dir)
    layout = CuratedDatasetLayout(curated_root)
    partition_kwargs = {"season": season, "season_type": "regular"}
    event_partition = CuratedPartition(table="events", **partition_kwargs)
    stint_partition = CuratedPartition(table="lineup_stints", **partition_kwargs)
    event_manifest = read_curated_partition_manifest(event_partition, curated_root)
    stint_manifest = read_curated_partition_manifest(stint_partition, curated_root)
    validate_curated_partition(event_manifest, curated_root)
    validate_curated_partition(stint_manifest, curated_root)
    events = _read_partition(
        layout.partition_dir(event_partition),
        columns=sorted(REQUIRED_EVENT_COLUMNS),
    )
    stints = _read_partition(layout.partition_dir(stint_partition))
    frame, audit = build_four_factor_stint_frame(stints, events)

    target = Path(analytical_dir) / "four_factor_stints" / season / "regular"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid4().hex}"
    backup = target.parent / f".{target.name}.bak-{uuid4().hex}"
    temporary.mkdir()
    try:
        frame.to_parquet(temporary / "part-00000.parquet", index=False)
        audit.to_parquet(temporary / "game_audit.parquet", index=False)
        manifest = {
            "schema_version": 1,
            "season": season,
            "created_at": datetime.now(UTC).isoformat(),
            "builder_code_version": modeling_code_fingerprint((Path(__file__),)),
            "four_factors": FOUR_FACTORS,
            "event_assignment": "lineup_stint_event_index_boundaries",
            "source_event_manifest_sha256": _sha256_file(
                layout.partition_dir(event_partition) / "_manifest.json"
            ),
            "source_lineup_stint_manifest_sha256": _sha256_file(
                layout.partition_dir(stint_partition) / "_manifest.json"
            ),
            "row_count": len(frame),
            "game_count": int(frame["game_id"].nunique()),
            "factor_event_count": int(audit["factor_event_count"].sum()),
            "eligible_game_count": int(audit["is_four_factor_eligible"].sum()),
            "excluded_game_count": int((~audit["is_four_factor_eligible"]).sum()),
            "unassigned_factor_event_count": int(audit["unassigned_factor_event_count"].sum()),
            "unknown_team_factor_event_count": int(audit["unknown_team_factor_event_count"].sum()),
        }
        (temporary / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        validate_four_factor_stint_dataset(temporary)
        if target.exists():
            target.replace(backup)
        try:
            temporary.replace(target)
        except Exception:
            if backup.exists() and not target.exists():
                backup.replace(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not target.exists():
            backup.replace(target)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)
    return target


def validate_four_factor_stint_dataset(path: Path | str) -> dict[str, object]:
    """Validate the materialized Four Factors mart's basic contract."""

    root = Path(path)
    manifest = json.loads((root / "_manifest.json").read_text())
    frame = pd.read_parquet(root / "part-00000.parquet")
    audit = pd.read_parquet(root / "game_audit.parquet")
    required = {
        "home_efg_pct",
        "away_efg_pct",
        "home_tov_pct",
        "away_tov_pct",
        "home_orb_pct",
        "away_orb_pct",
        "home_ftr",
        "away_ftr",
        "is_four_factor_eligible",
        "four_factor_quality_status",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Four Factors mart missing columns: {sorted(missing)}")
    if len(frame) != int(manifest["row_count"]):
        raise ValueError("Four Factors mart row count does not match manifest")
    if frame.empty or audit.empty:
        raise ValueError("Four Factors mart cannot be empty")
    if int(manifest["game_count"]) != int(frame["game_id"].nunique()):
        raise ValueError("Four Factors mart game count does not match manifest")
    return manifest


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="raise").astype(float)
    numerator = pd.to_numeric(numerator, errors="raise").astype(float)
    return numerator.div(denominator.where(denominator.gt(0)))


def _read_partition(root: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Read curated parts individually to tolerate nullable Arrow type drift."""

    files = sorted(root.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Four Factors mart has no parquet parts in {root}")
    return pd.concat(
        [pd.read_parquet(file, columns=columns) for file in files],
        ignore_index=True,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one Four Factors RAPM stint mart")
    parser.add_argument("season")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--analytical-dir", default="data/analytical")
    args = parser.parse_args()
    print(
        build_four_factor_stint_dataset(
            args.season,
            curated_dir=args.curated_dir,
            analytical_dir=args.analytical_dir,
        )
    )


if __name__ == "__main__":
    main()
