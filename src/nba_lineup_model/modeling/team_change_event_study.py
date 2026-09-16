"""Study sustained completed NAIL movement after clean offseason team changes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_PANEL_PATH
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import MODEL_NAME
from nba_lineup_model.web_api.inference import published_player_ratings_path


DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_ARTIFACT_SEASON = "2025-26"
DEFAULT_OUTPUT_ROOT = Path("artifacts/models/analysis/team_change_event_study")
MINIMUM_AGE = 25.0
MINIMUM_POSSESSIONS = 1_000.0
POST_MOVE_SEASONS = 3

VALIDATION_EVENTS = (
    ("LeBron James", "2010-11", "CLE", "MIA"),
    ("Steve Nash", "2004-05", "DAL", "PHX"),
    ("Shaquille O'Neal", "2004-05", "LAL", "MIA"),
    ("Kevin Durant", "2016-17", "OKC", "GSW"),
)


@dataclass(frozen=True)
class TeamChangeEventStudyRun:
    """Immutable outputs from one first-pass team-change event study."""

    run_dir: Path
    source_run_dir: Path
    strict_event_count: int
    broad_event_count: int


def build_team_change_event_mart(
    completed_ratings: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    minimum_age: float = MINIMUM_AGE,
    minimum_possessions: float = MINIMUM_POSSESSIONS,
    post_move_seasons: int = POST_MOVE_SEASONS,
) -> pd.DataFrame:
    """Return player-year changes with immediate and sustained NAIL movement.

    The event season is the first completed season on the new team.  A clean
    offseason change has exactly one primary team in the previous and target
    season, with different team tricodes.  The broad cohort permits a later
    move; the strict cohort requires the acquiring team for all post-move
    seasons.
    """

    if minimum_age <= 0:
        raise ValueError("minimum_age must be positive")
    if minimum_possessions <= 0:
        raise ValueError("minimum_possessions must be positive")
    if post_move_seasons < 1:
        raise ValueError("post_move_seasons must be at least one")

    ratings = _normalize_completed_ratings(completed_ratings)
    player_seasons = _normalize_player_seasons(panel).merge(
        ratings,
        on=["season", "player_id"],
        how="inner",
        validate="one_to_one",
    )
    if player_seasons.empty:
        raise ValueError("No completed ratings map to the player-season panel")
    if player_seasons.duplicated(["player_id", "season_start_year"]).any():
        raise ValueError("Player-season inputs must be unique by player and season year")

    player_seasons = player_seasons.sort_values(
        ["player_id", "season_start_year"], kind="stable"
    ).reset_index(drop=True)
    current = player_seasons.copy()
    previous = player_seasons.copy()
    previous["season_start_year"] = previous["season_start_year"] + 1
    previous = previous.rename(columns={
        "season": "previous_season",
        "primary_team_tricode": "old_team",
        "team_count": "previous_team_count",
        "rapm_possessions": "previous_possessions",
        "completed_nail_rating": "previous_completed_nail_rating",
        "preseason_player_prior": "previous_preseason_player_prior",
        "completed_nail_rank": "previous_completed_nail_rank",
    })
    keep_previous = [
        "player_id", "season_start_year", "previous_season", "old_team",
        "previous_team_count", "previous_possessions", "previous_completed_nail_rating",
        "previous_preseason_player_prior", "previous_completed_nail_rank",
    ]
    events = current.merge(
        previous.loc[:, keep_previous],
        on=["player_id", "season_start_year"],
        how="inner",
        validate="one_to_one",
    ).rename(columns={
        "season_start_year": "event_season_start_year",
        "season": "event_season",
        "primary_team_tricode": "new_team",
        "team_count": "event_team_count",
        "rapm_possessions": "event_possessions",
        "age": "event_age",
        "completed_nail_rating": "event_completed_nail_rating",
        "preseason_player_prior": "event_preseason_player_prior",
        "completed_nail_rank": "event_completed_nail_rank",
    })
    events["clean_offseason_move"] = (
        events["event_team_count"].eq(1)
        & events["previous_team_count"].eq(1)
        & events["new_team"].notna()
        & events["old_team"].notna()
        & events["new_team"].ne(events["old_team"])
    )
    events["age_eligible"] = events["event_age"].ge(minimum_age)
    events["entry_exposure_eligible"] = (
        events["previous_possessions"].ge(minimum_possessions)
        & events["event_possessions"].ge(minimum_possessions)
    )
    for offset in range(post_move_seasons):
        events, _ = _attach_relative_season(
            events, player_seasons, offset, minimum_possessions=minimum_possessions
        )
    for offset in range(1, 4):
        events, _ = _attach_relative_season(
            events, player_seasons, -offset, minimum_possessions=minimum_possessions
        )

    post_available = [f"relative_{offset}_eligible" for offset in range(post_move_seasons)]
    events["broad_cohort_eligible"] = (
        events["clean_offseason_move"]
        & events["age_eligible"]
        & events["entry_exposure_eligible"]
        & events.loc[:, post_available].all(axis=1)
    )
    later_team_changed = pd.Series(False, index=events.index, dtype=bool)
    for offset in range(1, post_move_seasons):
        later_team_changed |= events[f"relative_{offset}_team"].ne(events["new_team"])
    events["later_team_change"] = later_team_changed.where(
        events["broad_cohort_eligible"], pd.NA
    )
    pre_available = [f"relative_{offset}_eligible" for offset in (-3, -2, -1)]
    pre_move_team_stable = pd.Series(True, index=events.index, dtype=bool)
    for offset in (-3, -2, -1):
        pre_move_team_stable &= events[f"relative_{offset}_team"].eq(events["old_team"])
    events["pre_move_team_stable"] = pre_move_team_stable.where(
        events.loc[:, pre_available].all(axis=1), pd.NA
    )
    events["strict_cohort_eligible"] = (
        events["broad_cohort_eligible"]
        & ~later_team_changed
        & events.loc[:, pre_available].all(axis=1)
        & pre_move_team_stable
    )

    post_weight_columns = [f"relative_{offset}_possessions" for offset in range(post_move_seasons)]
    post_rating_columns = [
        f"relative_{offset}_completed_nail_rating" for offset in range(post_move_seasons)
    ]
    weights = events.loc[:, post_weight_columns].to_numpy(dtype=float)
    weight_total = weights.sum(axis=1)
    events["post_move_possessions"] = weight_total
    events["post_move_completed_nail_rating"] = events.loc[:, post_rating_columns].mean(axis=1)

    pre_weight_columns = [f"relative_{offset}_possessions" for offset in (-3, -2, -1)]
    pre_rating_columns = [
        f"relative_{offset}_completed_nail_rating" for offset in (-3, -2, -1)
    ]
    events["pre_move_possessions"] = events.loc[:, pre_weight_columns].sum(axis=1)
    events["pre_move_completed_nail_rating"] = events.loc[:, pre_rating_columns].mean(axis=1)
    events["immediate_completed_nail_change"] = (
        events["event_completed_nail_rating"] - events["previous_completed_nail_rating"]
    )
    events["sustained_completed_nail_change"] = (
        events["post_move_completed_nail_rating"] - events["pre_move_completed_nail_rating"]
    )
    return events.sort_values(
        ["event_season_start_year", "player_name", "player_id"], kind="stable"
    ).reset_index(drop=True)


def validation_examples(event_mart: pd.DataFrame) -> pd.DataFrame:
    """Return recognizable historical moves to audit event-season alignment."""

    required = {"player_name", "event_season", "old_team", "new_team", "strict_cohort_eligible"}
    missing = required - set(event_mart)
    if missing:
        raise ValueError(f"Event mart missing validation columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for player_name, season, old_team, new_team in VALIDATION_EVENTS:
        matches = event_mart.loc[
            event_mart["player_name"].eq(player_name)
            & event_mart["event_season"].eq(season)
            & event_mart["old_team"].eq(old_team)
            & event_mart["new_team"].eq(new_team)
        ]
        rows.append(
            {
                "player_name": player_name,
                "event_season": season,
                "expected_old_team": old_team,
                "expected_new_team": new_team,
                "found": len(matches) == 1,
                "strict_cohort_eligible": (
                    bool(matches["strict_cohort_eligible"].iloc[0]) if len(matches) == 1 else False
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_event_mart(event_mart: pd.DataFrame) -> pd.DataFrame:
    """Report cohort counts without making a causal comparison."""

    definitions = (
        ("All consecutive player-season transitions", pd.Series(True, index=event_mart.index)),
        ("Clean offseason moves", event_mart["clean_offseason_move"]),
        (
            "Age 25+ with 1,000 possessions before and during move",
            event_mart["clean_offseason_move"]
            & event_mart["age_eligible"]
            & event_mart["entry_exposure_eligible"],
        ),
        ("Broad three-season mover cohort", event_mart["broad_cohort_eligible"]),
        ("Strict three-season acquiring-team cohort", event_mart["strict_cohort_eligible"]),
    )
    return pd.DataFrame(
        [
            {
                "cohort": label,
                "events": int(mask.sum()),
                "players": int(event_mart.loc[mask, "player_id"].nunique()),
                "event_season_first": event_mart.loc[mask, "event_season"].min() if mask.any() else pd.NA,
                "event_season_last": event_mart.loc[mask, "event_season"].max() if mask.any() else pd.NA,
            }
            for label, mask in definitions
        ]
    )


def run_team_change_event_study(
    *,
    source_run_dir: Path | str | None = None,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    minimum_age: float = MINIMUM_AGE,
    minimum_possessions: float = MINIMUM_POSSESSIONS,
) -> TeamChangeEventStudyRun:
    """Materialize the first-pass production NAIL team-change study."""

    source = (
        Path(source_run_dir)
        if source_run_dir is not None
        else _latest_run(Path(artifacts_dir) / MODEL_NAME / DEFAULT_ARTIFACT_SEASON)
    )
    ratings_path = published_player_ratings_path(MODEL_NAME, source.name)
    if not ratings_path.is_file():
        raise FileNotFoundError(f"Published completed player ratings are missing: {ratings_path}")
    event_mart = build_team_change_event_mart(
        pd.read_parquet(ratings_path),
        pd.read_parquet(player_season_panel_path),
        minimum_age=minimum_age,
        minimum_possessions=minimum_possessions,
    )
    broad = event_mart.loc[event_mart["broad_cohort_eligible"]].copy()
    strict = event_mart.loc[event_mart["strict_cohort_eligible"]].copy()
    ranked = strict.sort_values(
        ["sustained_completed_nail_change", "post_move_possessions", "player_name"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    summary = summarize_event_mart(event_mart)
    examples = validation_examples(event_mart)

    root = Path(output_root)
    run_id = f"team-change-event-study-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    event_mart.to_parquet(run_dir / "team_change_events.parquet", index=False)
    broad.to_parquet(run_dir / "broad_cohort.parquet", index=False)
    strict.to_parquet(run_dir / "strict_cohort.parquet", index=False)
    ranked.to_parquet(run_dir / "strict_sustained_movers.parquet", index=False)
    ranked.loc[:, _mover_table_columns(ranked)].to_csv(
        run_dir / "strict_sustained_movers.csv", index=False
    )
    summary.to_parquet(run_dir / "cohort_summary.parquet", index=False)
    examples.to_parquet(run_dir / "validation_examples.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_run_dir": str(source),
                "completed_rating_source": str(ratings_path),
                "outcome_contract": (
                    "The primary outcome is completed display NAIL-RAPM movement: the "
                    "simple three-season completed-rating mean on the acquiring team "
                    "minus the simple three-season completed-rating mean on the departing team. "
                    "Every included season must clear the possession threshold. Immediate change "
                    "is retained as a secondary first-new-season minus final-old-season measure."
                ),
                "event_definition": (
                    "A clean offseason move has one primary team in consecutive seasons "
                    "and a changed primary-team tricode."
                ),
                "minimum_age": minimum_age,
                "minimum_possessions": minimum_possessions,
                "post_move_seasons": POST_MOVE_SEASONS,
                "strict_cohort": "The player has one primary departing team for all three pre-move seasons and one primary acquiring team for all three post-move seasons.",
                "broad_cohort": "The player has three qualified post-move seasons; later moves are flagged.",
                "causal_interpretation": False,
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return TeamChangeEventStudyRun(
        run_dir=run_dir,
        source_run_dir=source,
        strict_event_count=len(strict),
        broad_event_count=len(broad),
    )


def _attach_relative_season(
    events: pd.DataFrame,
    player_seasons: pd.DataFrame,
    offset: int,
    *,
    minimum_possessions: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach one completed player-season relative to the event season."""

    suffix = f"relative_{offset}"
    target = player_seasons.copy()
    target["event_season_start_year"] = target["season_start_year"] - offset
    target = target.rename(columns={
        "season": f"{suffix}_season",
        "primary_team_tricode": f"{suffix}_team",
        "team_count": f"{suffix}_team_count",
        "rapm_possessions": f"{suffix}_possessions",
        "completed_nail_rating": f"{suffix}_completed_nail_rating",
        "preseason_player_prior": f"{suffix}_preseason_player_prior",
        "completed_nail_rank": f"{suffix}_completed_nail_rank",
    })
    columns = [
        f"{suffix}_season", f"{suffix}_team", f"{suffix}_team_count", f"{suffix}_possessions",
        f"{suffix}_completed_nail_rating", f"{suffix}_preseason_player_prior",
        f"{suffix}_completed_nail_rank",
    ]
    joined = events.merge(
        target.loc[:, ["player_id", "event_season_start_year", *columns]],
        on=["player_id", "event_season_start_year"],
        how="left",
        validate="one_to_one",
    )
    joined[f"{suffix}_residual"] = (
        joined[f"{suffix}_completed_nail_rating"]
        - joined[f"{suffix}_preseason_player_prior"]
    )
    joined[f"{suffix}_eligible"] = (
        joined[f"{suffix}_team_count"].eq(1)
        & joined[f"{suffix}_team"].notna()
        & joined[f"{suffix}_possessions"].ge(minimum_possessions)
    )
    return joined, [*columns, f"{suffix}_residual", f"{suffix}_eligible"]


def _normalize_completed_ratings(completed_ratings: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "player_id", "player_name", "rapm", "prior_rapm"}
    missing = required - set(completed_ratings)
    if missing:
        raise ValueError(f"Completed ratings missing columns: {sorted(missing)}")
    ratings = completed_ratings.loc[:, ["season", "player_id", "player_name", "rapm", "prior_rapm"]].copy()
    if "rank_all_players" in completed_ratings:
        ratings["completed_nail_rank"] = completed_ratings["rank_all_players"]
    else:
        ratings["completed_nail_rank"] = ratings.groupby("season")["rapm"].rank(
            method="first", ascending=False
        )
    ratings = ratings.rename(columns={
        "rapm": "completed_nail_rating",
        "prior_rapm": "preseason_player_prior",
    })
    ratings["player_id"] = ratings["player_id"].astype("int64")
    ratings["season"] = ratings["season"].astype(str)
    if ratings.duplicated(["season", "player_id"]).any():
        raise ValueError("Completed ratings must be unique by player and season")
    return ratings


def _mover_table_columns(movers: pd.DataFrame) -> list[str]:
    """Keep the human-readable export focused on the first-pass contract."""

    preferred = [
        "player_name",
        "event_season",
        "old_team",
        "new_team",
        "event_age",
        "relative_-3_season",
        "relative_-3_possessions",
        "relative_-3_completed_nail_rating",
        "relative_-3_completed_nail_rank",
        "relative_-2_season",
        "relative_-2_possessions",
        "relative_-2_completed_nail_rating",
        "relative_-2_completed_nail_rank",
        "relative_-1_season",
        "relative_-1_possessions",
        "relative_-1_completed_nail_rating",
        "relative_-1_completed_nail_rank",
        "pre_move_possessions",
        "pre_move_completed_nail_rating",
        "relative_0_season",
        "relative_0_possessions",
        "relative_0_completed_nail_rating",
        "relative_0_completed_nail_rank",
        "relative_1_season",
        "relative_1_possessions",
        "relative_1_completed_nail_rating",
        "relative_1_completed_nail_rank",
        "relative_2_season",
        "relative_2_possessions",
        "relative_2_completed_nail_rating",
        "relative_2_completed_nail_rank",
        "post_move_possessions",
        "post_move_completed_nail_rating",
        "immediate_completed_nail_change",
        "sustained_completed_nail_change",
    ]
    return [column for column in preferred if column in movers]


def _normalize_player_seasons(panel: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season", "season_start_year", "player_id", "primary_team_tricode", "team_count",
        "rapm_possessions", "age",
    }
    missing = required - set(panel)
    if missing:
        raise ValueError(f"Player-season panel missing columns: {sorted(missing)}")
    output = panel.loc[:, list(required)].copy()
    output["season"] = output["season"].astype(str)
    output["player_id"] = output["player_id"].astype("int64")
    output["season_start_year"] = output["season_start_year"].astype(int)
    if output.duplicated(["season", "player_id"]).any():
        raise ValueError("Player-season panel must be unique by player and season")
    return output


def _latest_run(root: Path) -> Path:
    pointer = root / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"No latest artifact pointer: {pointer}")
    return root / str(json.loads(pointer.read_text())["run_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the NAIL team-change event study")
    parser.add_argument("--minimum-age", type=float, default=MINIMUM_AGE)
    parser.add_argument("--minimum-possessions", type=float, default=MINIMUM_POSSESSIONS)
    args = parser.parse_args()
    run = run_team_change_event_study(
        minimum_age=args.minimum_age,
        minimum_possessions=args.minimum_possessions,
    )
    print(
        "Team-change event study: "
        f"run={run.run_dir} broad_events={run.broad_event_count} strict_events={run.strict_event_count}"
    )


if __name__ == "__main__":
    main()
