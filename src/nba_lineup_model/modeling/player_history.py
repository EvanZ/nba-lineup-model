from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nba_lineup_model.modeling.schema import (
    CODE_VERSION_PATTERN,
    ArtifactRecord,
    BaselineRunManifest,
)
from nba_lineup_model.modeling.train import validate_baseline_run
from nba_lineup_model.players.storage import (
    read_player_catalog,
    read_player_season_bios,
    validate_player_season_partition,
)
from nba_lineup_model.season.compact import (
    read_curated_partition_manifest,
    validate_curated_partition,
)
from nba_lineup_model.season.layout import CuratedDatasetLayout, CuratedPartition
from nba_lineup_model.season.schema import (
    SEASON_PATTERN,
    SHA256_PATTERN,
    validate_season,
)

_COUNTING_STATS = {
    "assists": "statistics_assists",
    "blocks": "statistics_blocks",
    "field_goals_attempted": "statistics_fieldGoalsAttempted",
    "field_goals_made": "statistics_fieldGoalsMade",
    "free_throws_attempted": "statistics_freeThrowsAttempted",
    "free_throws_made": "statistics_freeThrowsMade",
    "fouls_offensive": "statistics_foulsOffensive",
    "fouls_drawn": "statistics_foulsDrawn",
    "fouls_personal": "statistics_foulsPersonal",
    "plus_minus": "statistics_plusMinusPoints",
    "points": "statistics_points",
    "rebounds_defensive": "statistics_reboundsDefensive",
    "rebounds_offensive": "statistics_reboundsOffensive",
    "rebounds_total": "statistics_reboundsTotal",
    "steals": "statistics_steals",
    "three_pointers_attempted": "statistics_threePointersAttempted",
    "three_pointers_made": "statistics_threePointersMade",
    "turnovers": "statistics_turnovers",
    "two_pointers_attempted": "statistics_twoPointersAttempted",
    "two_pointers_made": "statistics_twoPointersMade",
}
_PER_36_STATS = (
    "assists",
    "blocks",
    "field_goals_attempted",
    "free_throws_attempted",
    "fouls_drawn",
    "fouls_personal",
    "plus_minus",
    "points",
    "rebounds_defensive",
    "rebounds_offensive",
    "rebounds_total",
    "steals",
    "three_pointers_attempted",
    "turnovers",
)
_PRIOR_FEATURE_COLUMNS = (
    "age",
    "nba_experience_years",
    "games",
    "games_started",
    "minutes",
    "rapm",
    "rapm_possessions",
    "raw_on_court_net_rating",
    "points_per_36",
    "field_goals_attempted_per_36",
    "three_pointers_attempted_per_36",
    "free_throws_attempted_per_36",
    "assists_per_36",
    "turnovers_per_36",
    "rebounds_offensive_per_36",
    "rebounds_defensive_per_36",
    "steals_per_36",
    "blocks_per_36",
    "fouls_personal_per_36",
    "plus_minus_per_36",
    "field_goal_percentage",
    "three_point_percentage",
    "free_throw_percentage",
    "effective_field_goal_percentage",
    "true_shooting_percentage",
)
_RAPM_COLUMNS = (
    "player_id",
    "rapm",
    "raw_on_court_net_rating",
    "stint_count",
    "possessions",
    "seconds",
    "exposure_eligible",
    "primary_team_id",
    "primary_team_tricode",
)


class PlayerSeasonPanelSource(BaseModel):
    """Exact model and curated inputs for one season of the panel."""

    model_config = ConfigDict(strict=True, extra="forbid")

    season: str = Field(pattern=SEASON_PATTERN)
    rapm_run_id: str
    rapm_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    curated_players_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    player_bios_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    player_count: int = Field(ge=1)
    curated_player_row_count: int = Field(ge=1)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)


class PlayerSeasonPanelManifest(BaseModel):
    """Integrity and temporal contract for reusable player-season features."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    created_at: datetime
    builder_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    seasons: tuple[str, ...] = Field(min_length=1)
    season_type: Literal["regular"] = "regular"
    source_count: int = Field(ge=1)
    player_season_row_count: int = Field(ge=1)
    transition_row_count: int = Field(ge=0)
    cold_start_transition_count: int = Field(ge=0)
    prior_feature_columns: tuple[str, ...] = Field(min_length=1)
    sources: tuple[PlayerSeasonPanelSource, ...] = Field(min_length=1)
    artifacts: tuple[ArtifactRecord, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_panel(self) -> PlayerSeasonPanelManifest:
        if len(self.seasons) != len(set(self.seasons)):
            raise ValueError("Player-season panel seasons must be unique")
        if tuple(source.season for source in self.sources) != self.seasons:
            raise ValueError("Player-season panel sources must match season order")
        if self.source_count != len(self.sources):
            raise ValueError("Player-season panel source count changed")
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Player-season panel artifact filenames must be unique")
        return self


def aggregate_box_score_features(players: pd.DataFrame) -> pd.DataFrame:
    """Aggregate played game boxscore rows into stable season-level features."""

    players = _normalize_legacy_boxscore_schema(players)
    required = {
        "game_id",
        "personId",
        "name",
        "team_id",
        "team_tricode",
        "played",
        "starter",
        "statistics_minutes",
        *_COUNTING_STATS.values(),
    }
    missing = required - set(players)
    if missing:
        raise ValueError(f"Player boxscores missing columns: {sorted(missing)}")
    played = players.loc[players["played"].astype(str).eq("1")].copy()
    if played.empty:
        raise ValueError("Player boxscore features require at least one played row")
    if played.duplicated(["game_id", "personId"]).any():
        raise ValueError("Player boxscores contain duplicate game-player rows")
    played["player_id"] = pd.to_numeric(
        played["personId"],
        errors="raise",
    ).astype("int64")
    played["minutes"] = (
        pd.to_timedelta(
            played["statistics_minutes"],
            errors="coerce",
        ).dt.total_seconds()
        / 60.0
    )
    if played["minutes"].isna().any():
        raise ValueError("Played player boxscores require valid ISO minutes")
    played = played.loc[played["minutes"].gt(0)].copy()
    if played.empty:
        raise ValueError("Player boxscore features require positive recorded minutes")
    played["is_starter"] = played["starter"].astype(str).eq("1").astype("int64")
    for output, source in _COUNTING_STATS.items():
        played[output] = pd.to_numeric(played[source], errors="raise").astype(float)

    grouped = played.groupby("player_id", sort=True)
    output = grouped.agg(
        player_name=("name", "first"),
        games=("game_id", "nunique"),
        games_started=("is_starter", "sum"),
        minutes=("minutes", "sum"),
        team_count=("team_id", "nunique"),
        **{feature: (feature, "sum") for feature in _COUNTING_STATS},
    ).reset_index()
    primary_teams = (
        played.groupby(
            ["player_id", "team_id", "team_tricode"],
            as_index=False,
            sort=True,
        )["minutes"]
        .sum()
        .sort_values(
            ["player_id", "minutes", "team_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("player_id")
        .rename(
            columns={
                "team_id": "box_primary_team_id",
                "team_tricode": "box_primary_team_tricode",
            }
        )
        .loc[
            :,
            [
                "player_id",
                "box_primary_team_id",
                "box_primary_team_tricode",
            ],
        ]
    )
    output = output.merge(primary_teams, on="player_id", validate="one_to_one")
    output["games"] = output["games"].astype("int64")
    output["games_started"] = output["games_started"].astype("int64")
    output["team_count"] = output["team_count"].astype("int64")
    for feature in _PER_36_STATS:
        output[f"{feature}_per_36"] = _safe_rate(
            36.0 * output[feature],
            output["minutes"],
        )
    output["field_goal_percentage"] = _safe_rate(
        output["field_goals_made"],
        output["field_goals_attempted"],
    )
    output["three_point_percentage"] = _safe_rate(
        output["three_pointers_made"],
        output["three_pointers_attempted"],
    )
    output["free_throw_percentage"] = _safe_rate(
        output["free_throws_made"],
        output["free_throws_attempted"],
    )
    output["effective_field_goal_percentage"] = _safe_rate(
        output["field_goals_made"] + 0.5 * output["three_pointers_made"],
        output["field_goals_attempted"],
    )
    output["true_shooting_percentage"] = _safe_rate(
        output["points"],
        2.0 * (output["field_goals_attempted"] + 0.44 * output["free_throws_attempted"]),
    )
    return output.sort_values("player_id", kind="stable").reset_index(drop=True)


def _normalize_legacy_boxscore_schema(players: pd.DataFrame) -> pd.DataFrame:
    """Fill historical NBA Stats fields absent before the modern player schema.

    Older box scores provide player identity, minutes, total field goals, and
    three-point totals but omit a modern ``played`` flag, full name, two-point
    totals, and foul-drawn/offensive-foul counts. Positive minutes remain the
    inclusion rule, so treating an absent played flag as present cannot include
    a DNP. Unavailable foul counts are explicit structural zeros; two-point
    totals are identities derived from the recorded aggregate fields.
    """

    frame = players.copy()
    if "name" not in frame:
        first = frame.get("firstName", pd.Series("", index=frame.index)).fillna("")
        family = frame.get("familyName", pd.Series("", index=frame.index)).fillna("")
        full_name = first.astype(str).str.strip() + " " + family.astype(str).str.strip()
        full_name = full_name.str.strip()
        initials = frame.get("nameI", pd.Series("", index=frame.index)).fillna("")
        frame["name"] = full_name.where(full_name.ne(""), initials.astype(str).str.strip())
    inferred_played = (
        pd.to_timedelta(frame["statistics_minutes"], errors="coerce")
        .dt.total_seconds()
        .gt(0)
        .astype("int64")
        .astype(str)
    )
    if "played" not in frame:
        frame["played"] = inferred_played
    else:
        # Stats V3 retains the field but leaves it null; preserve explicit DNPs.
        frame["played"] = frame["played"].where(frame["played"].notna(), inferred_played)
    for column in ("statistics_foulsOffensive", "statistics_foulsDrawn"):
        if column not in frame:
            frame[column] = 0
    if "statistics_twoPointersAttempted" not in frame and {
        "statistics_fieldGoalsAttempted",
        "statistics_threePointersAttempted",
    } <= set(frame):
        frame["statistics_twoPointersAttempted"] = pd.to_numeric(
            frame["statistics_fieldGoalsAttempted"], errors="raise"
        ) - pd.to_numeric(frame["statistics_threePointersAttempted"], errors="raise")
    if "statistics_twoPointersMade" not in frame and {
        "statistics_fieldGoalsMade",
        "statistics_threePointersMade",
    } <= set(frame):
        frame["statistics_twoPointersMade"] = pd.to_numeric(
            frame["statistics_fieldGoalsMade"], errors="raise"
        ) - pd.to_numeric(frame["statistics_threePointersMade"], errors="raise")
    return frame


def player_season_frame(
    season: str,
    boxscore_features: pd.DataFrame,
    rapm_rankings: pd.DataFrame,
    player_bios: pd.DataFrame,
    player_catalog: pd.DataFrame,
    *,
    rapm_run_id: str,
) -> pd.DataFrame:
    """Combine same-season outcomes, box features, and static player context."""

    season = validate_season(season)
    required_rankings = set(_RAPM_COLUMNS)
    missing = required_rankings - set(rapm_rankings)
    if missing:
        raise ValueError(f"RAPM rankings missing columns: {sorted(missing)}")
    catalog_ids = set(player_catalog["player_id"].astype(int))
    # Some pre-modern Stats feeds emit numeric roster placeholders that are not
    # NBA player identities. They cannot be joined to a bio or carried across
    # seasons, so exclude them from the player-level modeling universe.
    rapm_rankings = rapm_rankings.loc[
        rapm_rankings["player_id"].astype(int).isin(catalog_ids)
    ].copy()
    ranking_ids = set(rapm_rankings["player_id"].astype(int))
    bio_ids = set(player_bios["player_id"].astype(int))
    if not ranking_ids <= bio_ids:
        raise ValueError(
            "RAPM rankings contain players absent from season bios: "
            f"rapm={len(ranking_ids)}, bios={len(bio_ids)}"
        )
    # A partial historical RAPM season can exclude a game that still has a
    # positive-minute box score. The panel's outcome universe is RAPM players.
    boxscore_features = boxscore_features.loc[
        boxscore_features["player_id"].astype(int).isin(ranking_ids)
    ].copy()

    rankings = rapm_rankings.loc[
        :,
        list(_RAPM_COLUMNS),
    ].rename(
        columns={
            "stint_count": "rapm_stint_count",
            "possessions": "rapm_possessions",
            "seconds": "rapm_seconds",
            "exposure_eligible": "rapm_exposure_eligible",
        }
    )
    bios = player_bios.loc[
        :,
        [
            "player_id",
            "player_name",
            "age",
            "listed_position",
            "height_inches",
            "weight_pounds",
            "college",
            "country",
            "draft_year",
            "draft_round",
            "draft_number",
            "is_undrafted",
        ],
    ]
    catalog = player_catalog.loc[
        :,
        ["player_id", "from_year", "to_year"],
    ]
    panel = (
        rankings.merge(
            boxscore_features.drop(columns="player_name"),
            on="player_id",
            how="left",
            validate="one_to_one",
        )
        .merge(bios, on="player_id", validate="one_to_one")
        .merge(catalog, on="player_id", validate="many_to_one")
    )
    start_year = int(season[:4])
    panel.insert(0, "schema_version", 1)
    panel.insert(1, "season", season)
    panel.insert(2, "season_start_year", start_year)
    panel.insert(3, "rapm_run_id", rapm_run_id)
    panel["boxscore_features_available"] = panel["games"].notna()
    panel["nba_experience_years"] = (
        (start_year - pd.to_numeric(panel["from_year"], errors="raise"))
        .clip(lower=0)
        .astype("int64")
    )
    panel["is_rookie"] = panel["nba_experience_years"].eq(0)
    panel["years_since_draft"] = (start_year - panel["draft_year"].astype("Float64")).astype(
        "Float64"
    )
    if panel.duplicated(["season", "player_id"]).any():
        raise ValueError("Player-season panel keys must be unique")
    return panel.sort_values("player_id", kind="stable").reset_index(drop=True)


def player_transition_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Create target-season rows containing only lagged performance features."""

    required = {
        "season",
        "season_start_year",
        "player_id",
        "player_name",
        "age",
        "nba_experience_years",
        "listed_position",
        "height_inches",
        "weight_pounds",
        "draft_year",
        "draft_round",
        "draft_number",
        "is_undrafted",
        "is_rookie",
        "rapm",
        "rapm_possessions",
        "rapm_seconds",
        "rapm_exposure_eligible",
        *_PRIOR_FEATURE_COLUMNS,
    }
    missing = required - set(panel)
    if missing:
        raise ValueError(f"Player-season panel missing transition columns: {sorted(missing)}")
    ordered_seasons = (
        panel.loc[:, ["season", "season_start_year"]]
        .drop_duplicates()
        .sort_values("season_start_year", kind="stable")
    )
    transitions: list[pd.DataFrame] = []
    season_rows = list(ordered_seasons.itertuples(index=False))
    for prior, target in zip(season_rows, season_rows[1:], strict=False):
        if int(target.season_start_year) != int(prior.season_start_year) + 1:
            continue
        target_rows = panel.loc[
            panel["season"].eq(target.season),
            [
                "player_id",
                "player_name",
                "age",
                "nba_experience_years",
                "listed_position",
                "height_inches",
                "weight_pounds",
                "draft_year",
                "draft_round",
                "draft_number",
                "is_undrafted",
                "is_rookie",
                "rapm",
                "rapm_possessions",
                "rapm_seconds",
                "rapm_exposure_eligible",
            ],
        ].rename(
            columns={
                "age": "target_age",
                "nba_experience_years": "target_nba_experience_years",
                "rapm": "target_rapm",
                "rapm_possessions": "target_rapm_possessions",
                "rapm_seconds": "target_rapm_seconds",
                "rapm_exposure_eligible": "target_rapm_exposure_eligible",
            }
        )
        prior_rows = panel.loc[
            panel["season"].eq(prior.season),
            ["player_id", *_PRIOR_FEATURE_COLUMNS],
        ].rename(columns={column: f"prior_{column}" for column in _PRIOR_FEATURE_COLUMNS})
        transition = target_rows.merge(
            prior_rows,
            on="player_id",
            how="left",
            validate="one_to_one",
        )
        transition.insert(0, "schema_version", 1)
        transition.insert(1, "target_season", target.season)
        transition.insert(2, "prior_season", prior.season)
        transition.insert(
            4,
            "has_prior_season",
            transition["prior_rapm"].notna(),
        )
        transitions.append(transition)
    if not transitions:
        return pd.DataFrame()
    output = pd.concat(transitions, ignore_index=True)
    if output.duplicated(["target_season", "player_id"]).any():
        raise ValueError("Player transition keys must be unique")
    return output.sort_values(
        ["target_season", "player_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_player_season_panel(
    seasons: Sequence[str],
    *,
    rapm_run_ids: Mapping[str, str] | None = None,
    curated_dir: Path | str = Path("data/curated"),
    artifacts_dir: Path | str = Path("artifacts/models"),
    player_catalog_path: Path | str = Path("data/catalog/players.parquet"),
    analytical_dir: Path | str = Path("data/analytical"),
) -> tuple[PlayerSeasonPanelManifest, Path]:
    """Build an atomic multi-season panel and leakage-safe lag-one transitions."""

    normalized = tuple(
        sorted(
            {validate_season(season) for season in seasons},
            key=lambda value: int(value[:4]),
        )
    )
    if not normalized:
        raise ValueError("Player-season panel requires at least one season")
    catalog = read_player_catalog(player_catalog_path)
    catalog_frame = pd.DataFrame([player.model_dump(mode="python") for player in catalog.players])
    frames: list[pd.DataFrame] = []
    sources: list[PlayerSeasonPanelSource] = []
    for season in normalized:
        run_dir, rapm_manifest = _resolve_rapm_run(
            season,
            (rapm_run_ids or {}).get(season),
            artifacts_dir,
        )
        players_partition = CuratedPartition(
            table="players",
            season=season,
            season_type="regular",
        )
        curated_manifest = read_curated_partition_manifest(
            players_partition,
            curated_dir,
        )
        validate_curated_partition(curated_manifest, curated_dir)
        validate_player_season_partition(
            season,
            "regular",
            curated_dir,
        )
        players = _read_curated_players(
            players_partition,
            curated_manifest,
            curated_dir,
        )
        boxscores = aggregate_box_score_features(players)
        rankings = pd.read_parquet(run_dir / "player_rankings.parquet")
        bios = pd.DataFrame(
            [
                player.model_dump(mode="python")
                for player in read_player_season_bios(
                    season,
                    "regular",
                    curated_dir,
                ).players
            ]
        )
        frame = player_season_frame(
            season,
            boxscores,
            rankings,
            bios,
            catalog_frame,
            rapm_run_id=rapm_manifest.run_id,
        )
        frames.append(frame)
        layout = CuratedDatasetLayout(Path(curated_dir))
        sources.append(
            PlayerSeasonPanelSource(
                season=season,
                rapm_run_id=rapm_manifest.run_id,
                rapm_manifest_sha256=_sha256_file(run_dir / "manifest.json"),
                curated_players_manifest_sha256=_sha256_file(
                    layout.manifest_path(players_partition)
                ),
                player_bios_manifest_sha256=_sha256_file(
                    Path(curated_dir) / "player_seasons" / season / "regular" / "_manifest.json"
                ),
                player_count=len(frame),
                curated_player_row_count=len(players),
            )
        )
    panel = (
        pd.concat(frames, ignore_index=True)
        .sort_values(
            ["season_start_year", "player_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    transitions = player_transition_frame(panel)
    return _write_panel(
        panel,
        transitions,
        tuple(sources),
        normalized,
        analytical_dir,
    )


def validate_player_season_panel(
    panel_dir: Path | str,
) -> PlayerSeasonPanelManifest:
    """Validate exact panel artifacts, hashes, rows, and temporal columns."""

    root = Path(panel_dir)
    manifest = PlayerSeasonPanelManifest.model_validate_json((root / "_manifest.json").read_text())
    expected = {artifact.filename for artifact in manifest.artifacts}
    actual = {
        path.name for path in root.iterdir() if path.is_file() and path.name != "_manifest.json"
    }
    if actual != expected:
        raise ValueError("Player-season panel files do not match the manifest")
    frames: dict[str, pd.DataFrame] = {}
    for artifact in manifest.artifacts:
        path = root / artifact.filename
        if path.stat().st_size != artifact.byte_count:
            raise ValueError(f"Player-season panel byte count changed: {path.name}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"Player-season panel hash changed: {path.name}")
        frame = pd.read_parquet(path)
        frames[path.name] = frame
        if artifact.row_count is not None and len(frame) != artifact.row_count:
            raise ValueError(f"Player-season panel rows changed: {path.name}")
    panel = frames["player_seasons.parquet"]
    transitions = frames["transitions.parquet"]
    if panel.duplicated(["season", "player_id"]).any():
        raise ValueError("Player-season panel keys changed")
    if not transitions.empty and transitions.duplicated(["target_season", "player_id"]).any():
        raise ValueError("Player transition keys changed")
    if len(panel) != manifest.player_season_row_count:
        raise ValueError("Player-season manifest row count changed")
    if len(transitions) != manifest.transition_row_count:
        raise ValueError("Player transition manifest row count changed")
    return manifest


def player_history_code_fingerprint(
    source_paths: Sequence[Path | str] | None = None,
) -> str:
    """Hash sources that define player-season and transition features."""

    paths = (
        (Path(__file__),) if source_paths is None else tuple(Path(path) for path in source_paths)
    )
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _read_curated_players(
    partition: CuratedPartition,
    manifest: Any,
    curated_dir: Path | str,
) -> pd.DataFrame:
    root = CuratedDatasetLayout(Path(curated_dir)).partition_dir(partition)
    return pd.concat(
        [pd.read_parquet(root / part.filename) for part in manifest.parts],
        ignore_index=True,
    )


def _resolve_rapm_run(
    season: str,
    run_id: str | None,
    artifacts_dir: Path | str,
) -> tuple[Path, BaselineRunManifest]:
    season_dir = Path(artifacts_dir) / "rapm" / season
    if run_id is None:
        latest = season_dir / "latest.json"
        if not latest.is_file():
            raise ValueError(f"No RAPM run exists for {season}")
        run_id = str(json.loads(latest.read_text())["run_id"])
    run_dir = season_dir / run_id
    manifest = validate_baseline_run(run_dir)
    if manifest.season != season:
        raise ValueError("Player-season RAPM source has the wrong season")
    return run_dir, manifest


def _write_panel(
    panel: pd.DataFrame,
    transitions: pd.DataFrame,
    sources: tuple[PlayerSeasonPanelSource, ...],
    seasons: tuple[str, ...],
    analytical_dir: Path | str,
) -> tuple[PlayerSeasonPanelManifest, Path]:
    target = Path(analytical_dir) / "player_season_panel"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid4().hex}"
    backup = target.parent / f".{target.name}.bak-{uuid4().hex}"
    temporary.mkdir()
    try:
        outputs = {
            "player_seasons.parquet": panel,
            "transitions.parquet": transitions,
        }
        for filename, frame in outputs.items():
            frame.to_parquet(temporary / filename, index=False)
        artifacts = tuple(
            ArtifactRecord(
                filename=filename,
                row_count=len(frame),
                byte_count=(temporary / filename).stat().st_size,
                sha256=_sha256_file(temporary / filename),
            )
            for filename, frame in outputs.items()
        )
        manifest = PlayerSeasonPanelManifest(
            created_at=datetime.now(UTC),
            builder_code_version=player_history_code_fingerprint(),
            seasons=seasons,
            source_count=len(sources),
            player_season_row_count=len(panel),
            transition_row_count=len(transitions),
            cold_start_transition_count=(
                int((~transitions["has_prior_season"]).sum()) if not transitions.empty else 0
            ),
            prior_feature_columns=_PRIOR_FEATURE_COLUMNS,
            sources=sources,
            artifacts=artifacts,
        )
        (temporary / "_manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        validate_player_season_panel(temporary)
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
        validate_player_season_panel(target)
        return manifest, target
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not target.exists():
            backup.replace(target)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


def _safe_rate(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    values = np.divide(
        numerator.to_numpy(dtype=float),
        denominator.to_numpy(dtype=float),
        out=np.full(len(numerator), np.nan, dtype=float),
        where=denominator.to_numpy(dtype=float) > 0,
    )
    return pd.Series(values, index=numerator.index, dtype=float)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_rapm_run_ids(values: list[str] | None) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values or []:
        season, separator, run_id = value.partition("=")
        if not separator or not run_id:
            raise argparse.ArgumentTypeError("RAPM run IDs must use SEASON=RUN_ID")
        output[validate_season(season)] = run_id
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build same-season player outcomes and leakage-safe prior-season "
            "features for RAPM and neural models."
        )
    )
    parser.add_argument(
        "seasons",
        nargs="+",
        help="Ordered seasons in YYYY-YY format",
    )
    parser.add_argument(
        "--rapm-run-id",
        action="append",
        dest="rapm_run_ids",
        help="Pin one source as SEASON=RUN_ID; repeat by season",
    )
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument(
        "--player-catalog",
        default="data/catalog/players.parquet",
    )
    parser.add_argument("--analytical-dir", default="data/analytical")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest, path = build_player_season_panel(
        args.seasons,
        rapm_run_ids=_parse_rapm_run_ids(args.rapm_run_ids),
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        player_catalog_path=args.player_catalog,
        analytical_dir=args.analytical_dir,
    )
    print(
        f"Player-season panel: seasons={len(manifest.seasons)}, "
        f"rows={manifest.player_season_row_count}, "
        f"transitions={manifest.transition_row_count}, "
        f"cold_starts={manifest.cold_start_transition_count}; path={path}"
    )


if __name__ == "__main__":
    main()
