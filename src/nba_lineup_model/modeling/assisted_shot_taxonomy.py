"""Validated assisted and unassisted made-shot profiles from canonical play-by-play."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nba_lineup_model.modeling.player_history import (
    player_history_code_fingerprint,
    validate_player_season_panel,
)
from nba_lineup_model.modeling.schema import CODE_VERSION_PATTERN, ArtifactRecord
from nba_lineup_model.modeling.shot_taxonomy import SHOT_FAMILIES, classify_shot_events
from nba_lineup_model.season.schema import SHA256_PATTERN, validate_season

DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_PROCESSED_PLAYERS_DIR = Path("data/processed/players")
DEFAULT_OUTPUT_DIR = Path("data/analytical/assisted_shot_taxonomy")

ASSIST_STATUSES = ("assisted", "unassisted", "unknown")
ASSIST_MARKER_PATTERN = r"\([^)]*\bAST\b[^)]*\)"


class AssistedShotTaxonomyManifest(BaseModel):
    """Integrity and semantics contract for assisted made-shot profiles."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    created_at: datetime
    builder_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    source_player_season_panel_path: str
    source_player_season_panel_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    source_processed_players_dir: str
    seasons: tuple[str, ...] = Field(min_length=1)
    row_count: int = Field(ge=1)
    shot_families: tuple[str, ...]
    assist_statuses: tuple[str, ...]
    source_event_manifest_sha256: dict[str, str]
    artifacts: tuple[ArtifactRecord, ...] = Field(min_length=4)

    @field_validator("seasons")
    @classmethod
    def validate_season_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_season(value) for value in values)

    @model_validator(mode="after")
    def validate_contract(self) -> AssistedShotTaxonomyManifest:
        if self.seasons != tuple(sorted(self.seasons, key=_season_start_year)):
            raise ValueError("Assisted shot taxonomy seasons must be chronological")
        if self.shot_families != SHOT_FAMILIES:
            raise ValueError("Assisted shot taxonomy family contract changed")
        if self.assist_statuses != ASSIST_STATUSES:
            raise ValueError("Assisted shot taxonomy assist-status contract changed")
        if set(self.source_event_manifest_sha256) != set(self.seasons):
            raise ValueError("Assisted shot taxonomy must record every source event manifest")
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Assisted shot taxonomy artifact filenames must be unique")
        return self


def build_assisted_shot_taxonomy(
    *,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    processed_players_dir: Path | str = DEFAULT_PROCESSED_PLAYERS_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    seasons: tuple[str, ...] | None = None,
) -> tuple[AssistedShotTaxonomyManifest, Path]:
    """Build atomic player-season profiles and a box-score reconciliation audit."""

    panel_path = Path(player_season_panel_path)
    panel_root = panel_path.parent
    validate_player_season_panel(panel_root)
    panel = pd.read_parquet(panel_path)
    _validate_panel(panel)
    available_seasons = tuple(sorted(panel["season"].astype(str).unique(), key=_season_start_year))
    selected_seasons = _resolve_seasons(available_seasons, seasons)
    curated_root = Path(curated_dir)
    players_root = Path(processed_players_dir)

    profiles: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    game_reconciliations: list[pd.DataFrame] = []
    season_reconciliations: list[dict[str, object]] = []
    event_manifests: dict[str, str] = {}
    for index, season in enumerate(selected_seasons, start=1):
        print(
            f"Assisted shot taxonomy: [{index}/{len(selected_seasons)}] reconciling {season}",
            flush=True,
        )
        event_root = curated_root / "events" / season / "regular"
        manifest_path = event_root / "_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Assisted shot taxonomy is missing events manifest for {season}")
        event_manifests[season] = _sha256_file(manifest_path)
        events = _read_event_partition(event_root)
        classified = classify_assisted_shot_events(events)
        season_panel = panel.loc[panel["season"].eq(season)].copy()
        season_profiles, season_coverage = build_season_assisted_shot_profiles(
            season, season_panel, classified
        )
        reconciliation = reconcile_season_assisted_shots(
            season, classified, players_root
        )
        profiles.append(season_profiles)
        coverage.append(season_coverage)
        game_reconciliations.append(reconciliation)
        season_reconciliations.append(_summarize_reconciliation(season, reconciliation))

    outputs = {
        "player_season_assisted_shot_profiles.parquet": pd.concat(profiles, ignore_index=True),
        "season_assisted_shot_coverage.parquet": pd.DataFrame(coverage),
        "game_assisted_shot_reconciliation.parquet": pd.concat(game_reconciliations, ignore_index=True),
        "season_assisted_shot_reconciliation.parquet": pd.DataFrame(season_reconciliations),
    }
    return _publish_assisted_shot_outputs(
        output_dir=Path(output_dir),
        outputs=outputs,
        panel_path=panel_path,
        panel_manifest_sha256=_sha256_file(panel_root / "_manifest.json"),
        processed_players_dir=players_root,
        seasons=selected_seasons,
        event_manifests=event_manifests,
    )


def combine_assisted_shot_taxonomy_parts(
    *,
    parts_root: Path | str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[AssistedShotTaxonomyManifest, Path]:
    """Atomically combine independently validated seasonal assisted-shot builds."""

    root = Path(parts_root)
    part_roots = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: _season_start_year(path.name))
    if not part_roots:
        raise FileNotFoundError(f"No assisted shot taxonomy parts found in {root}")
    manifests = [validate_assisted_shot_taxonomy(path) for path in part_roots]
    if any(len(manifest.seasons) != 1 or manifest.seasons[0] != path.name for manifest, path in zip(manifests, part_roots, strict=True)):
        raise ValueError("Assisted shot taxonomy parts must be one validated season per directory")
    seasons = tuple(manifest.seasons[0] for manifest in manifests)
    if len(seasons) != len(set(seasons)):
        raise ValueError("Assisted shot taxonomy parts duplicate a season")
    reference = manifests[0]
    if any(
        manifest.source_player_season_panel_manifest_sha256
        != reference.source_player_season_panel_manifest_sha256
        or manifest.source_processed_players_dir != reference.source_processed_players_dir
        for manifest in manifests[1:]
    ):
        raise ValueError("Assisted shot taxonomy parts use inconsistent source contracts")
    outputs = {
        "player_season_assisted_shot_profiles.parquet": pd.concat(
            [pd.read_parquet(path / "player_season_assisted_shot_profiles.parquet") for path in part_roots],
            ignore_index=True,
        ),
        "season_assisted_shot_coverage.parquet": pd.concat(
            [pd.read_parquet(path / "season_assisted_shot_coverage.parquet") for path in part_roots],
            ignore_index=True,
        ),
        "game_assisted_shot_reconciliation.parquet": pd.concat(
            [pd.read_parquet(path / "game_assisted_shot_reconciliation.parquet") for path in part_roots],
            ignore_index=True,
        ),
        "season_assisted_shot_reconciliation.parquet": pd.concat(
            [pd.read_parquet(path / "season_assisted_shot_reconciliation.parquet") for path in part_roots],
            ignore_index=True,
        ),
    }
    event_manifests = {
        manifest.seasons[0]: manifest.source_event_manifest_sha256[manifest.seasons[0]]
        for manifest in manifests
    }
    return _publish_assisted_shot_outputs(
        output_dir=Path(output_dir),
        outputs=outputs,
        panel_path=Path(reference.source_player_season_panel_path),
        panel_manifest_sha256=reference.source_player_season_panel_manifest_sha256,
        processed_players_dir=Path(reference.source_processed_players_dir),
        seasons=seasons,
        event_manifests=event_manifests,
    )


def build_assisted_shot_taxonomy_parallel(
    *,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    processed_players_dir: Path | str = DEFAULT_PROCESSED_PLAYERS_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    workers: int = 4,
) -> tuple[AssistedShotTaxonomyManifest, Path]:
    """Build one validated seasonal part per worker, then atomically combine them."""

    if workers < 1:
        raise ValueError("Assisted shot taxonomy workers must be positive")
    panel_path = Path(player_season_panel_path)
    panel = pd.read_parquet(panel_path, columns=["season"])
    seasons = tuple(sorted(panel["season"].astype(str).unique(), key=_season_start_year))
    output = Path(output_dir)
    parts_root = output.parent / f".{output.name}_parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    pending = [
        season
        for season in seasons
        if not _is_valid_single_season_part(parts_root / season, season)
    ]
    if pending:
        print(
            f"Assisted shot taxonomy: building {len(pending)} seasonal parts with {workers} workers",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _build_assisted_shot_taxonomy_part,
                    season,
                    str(panel_path),
                    str(curated_dir),
                    str(processed_players_dir),
                    str(parts_root / season),
                ): season
                for season in pending
            }
            completed = 0
            for future in as_completed(futures):
                season = futures[future]
                future.result()
                completed += 1
                print(
                    f"Assisted shot taxonomy: [{completed}/{len(pending)}] completed {season}",
                    flush=True,
                )
    return combine_assisted_shot_taxonomy_parts(parts_root=parts_root, output_dir=output)


def _publish_assisted_shot_outputs(
    *,
    output_dir: Path,
    outputs: dict[str, pd.DataFrame],
    panel_path: Path,
    panel_manifest_sha256: str,
    processed_players_dir: Path,
    seasons: tuple[str, ...],
    event_manifests: dict[str, str],
) -> tuple[AssistedShotTaxonomyManifest, Path]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{uuid4().hex}"
    backup = output_dir.parent / f".{output_dir.name}.bak-{uuid4().hex}"
    temporary.mkdir()
    try:
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
        manifest = AssistedShotTaxonomyManifest(
            created_at=datetime.now(UTC),
            builder_code_version=player_history_code_fingerprint((Path(__file__),)),
            source_player_season_panel_path=str(panel_path),
            source_player_season_panel_manifest_sha256=panel_manifest_sha256,
            source_processed_players_dir=str(processed_players_dir),
            seasons=seasons,
            row_count=len(outputs["player_season_assisted_shot_profiles.parquet"]),
            shot_families=SHOT_FAMILIES,
            assist_statuses=ASSIST_STATUSES,
            source_event_manifest_sha256=event_manifests,
            artifacts=artifacts,
        )
        (temporary / "_manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        validate_assisted_shot_taxonomy(temporary)
        if output_dir.exists():
            output_dir.replace(backup)
        try:
            temporary.replace(output_dir)
        except Exception:
            if backup.exists() and not output_dir.exists():
                backup.replace(output_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        validate_assisted_shot_taxonomy(output_dir)
        return manifest, output_dir
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not output_dir.exists():
            backup.replace(output_dir)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


def _build_assisted_shot_taxonomy_part(
    season: str,
    panel_path: str,
    curated_dir: str,
    processed_players_dir: str,
    output_dir: str,
) -> None:
    build_assisted_shot_taxonomy(
        player_season_panel_path=panel_path,
        curated_dir=curated_dir,
        processed_players_dir=processed_players_dir,
        output_dir=output_dir,
        seasons=(season,),
    )


def _is_valid_single_season_part(path: Path, season: str) -> bool:
    if not path.is_dir():
        return False
    try:
        manifest = validate_assisted_shot_taxonomy(path)
    except (FileNotFoundError, ValueError):
        return False
    return manifest.seasons == (season,)


def classify_assisted_shot_events(events: pd.DataFrame) -> pd.DataFrame:
    """Classify made shots by family and explicit play-by-play assist status."""

    required = {
        "game_id",
        "team_id",
        "team_tricode",
        "event_type",
        "event_subtype",
        "player_id",
        "shot_result",
        "description",
    }
    missing = required - set(events)
    if missing:
        raise ValueError(f"Assisted shot taxonomy event frame missing columns: {sorted(missing)}")
    base = classify_shot_events(events)
    event_type = events["event_type"].astype("string").str.casefold()
    metadata = events.loc[
        event_type.isin(["2pt", "3pt"]),
        ["game_id", "team_id", "team_tricode", "description"],
    ].reset_index(drop=True)
    output = pd.concat([metadata, base], axis=1)
    description = output["description"].astype("string")
    output["has_description"] = description.notna() & description.str.strip().ne("")
    output["is_assisted"] = output["has_description"] & description.str.contains(
        ASSIST_MARKER_PATTERN, case=False, regex=True, na=False
    )
    output["assist_status"] = np.select(
        [output["is_assisted"], output["has_description"]],
        ["assisted", "unassisted"],
        default="unknown",
    )
    output["team_id"] = pd.to_numeric(output["team_id"], errors="coerce").astype("Int64")
    return output.reset_index(drop=True)


def build_season_assisted_shot_profiles(
    season: str,
    player_season_panel: pd.DataFrame,
    classified_shots: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return player-season made-shot counts split by family and assist status."""

    validate_season(season)
    _validate_season_panel(season, player_season_panel)
    profiles = player_season_panel.loc[
        :, ["season", "season_start_year", "player_id", "player_name", "rapm_possessions"]
    ].copy()
    profiles["player_id"] = pd.to_numeric(profiles["player_id"], errors="raise").astype("int64")
    profiles["on_court_possessions"] = pd.to_numeric(
        profiles.pop("rapm_possessions"), errors="raise"
    ).astype(float)
    if not profiles["on_court_possessions"].gt(0).all():
        raise ValueError(f"Assisted shot profiles require positive possessions in {season}")

    made = classified_shots.loc[
        classified_shots["is_made"] & classified_shots["player_id"].notna()
    ].copy()
    made["player_id"] = made["player_id"].astype("int64")
    matched = made.loc[made["player_id"].isin(set(profiles["player_id"]))].copy()
    counts = (
        matched.groupby(["player_id", "shot_family", "assist_status"], as_index=False)
        .size()
        .rename(columns={"size": "makes"})
        .astype({"makes": "int64"})
    )
    made_columns: list[str] = []
    for family in SHOT_FAMILIES:
        for status in ASSIST_STATUSES:
            column = f"{status}_{family}_makes"
            made_columns.append(column)
            subset = counts.loc[
                counts["shot_family"].eq(family) & counts["assist_status"].eq(status),
                ["player_id", "makes"],
            ].rename(columns={"makes": column})
            profiles = profiles.merge(subset, on="player_id", how="left", validate="one_to_one")
            profiles[column] = profiles[column].fillna(0).astype("int64")
            profiles[f"{column}_per_100"] = (
                100.0 * profiles[column] / profiles["on_court_possessions"]
            )
    for family in SHOT_FAMILIES:
        family_columns = [f"{status}_{family}_makes" for status in ASSIST_STATUSES]
        profiles[f"{family}_makes"] = profiles[family_columns].sum(axis=1).astype("int64")
    profiles["assisted_makes"] = profiles[
        [f"assisted_{family}_makes" for family in SHOT_FAMILIES]
    ].sum(axis=1).astype("int64")
    profiles["unassisted_makes"] = profiles[
        [f"unassisted_{family}_makes" for family in SHOT_FAMILIES]
    ].sum(axis=1).astype("int64")
    profiles["unknown_assist_status_makes"] = profiles[
        [f"unknown_{family}_makes" for family in SHOT_FAMILIES]
    ].sum(axis=1).astype("int64")
    profiles["total_makes"] = profiles[["assisted_makes", "unassisted_makes", "unknown_assist_status_makes"]].sum(axis=1).astype("int64")
    profiles["assist_status_coverage"] = np.divide(
        profiles["assisted_makes"] + profiles["unassisted_makes"],
        profiles["total_makes"],
        out=np.full(len(profiles), np.nan, dtype=float),
        where=profiles["total_makes"].to_numpy(dtype=float) > 0,
    )

    coverage = {
        "season": season,
        "source_shot_event_count": int(len(classified_shots)),
        "made_shot_event_count": int(classified_shots["is_made"].sum()),
        "player_attributed_made_shot_count": int(len(made)),
        "panel_matched_made_shot_count": int(len(matched)),
        "unmatched_player_made_shot_count": int(len(made) - len(matched)),
        "described_made_shot_count": int(classified_shots.loc[classified_shots["is_made"], "has_description"].sum()),
        "unknown_assist_status_made_shot_count": int(
            (classified_shots["is_made"] & classified_shots["assist_status"].eq("unknown")).sum()
        ),
        "profile_player_count": int(len(profiles)),
        "profile_player_with_made_shot_count": int(profiles["total_makes"].gt(0).sum()),
    }
    return profiles, coverage


def reconcile_season_assisted_shots(
    season: str,
    classified_shots: pd.DataFrame,
    processed_players_dir: Path | str,
) -> pd.DataFrame:
    """Reconcile team-game PBP makes and assist markers with official player boxes."""

    validate_season(season)
    made = classified_shots.loc[
        classified_shots["is_made"] & classified_shots["team_id"].notna()
    ].copy()
    event = (
        made.groupby(["game_id", "team_id", "team_tricode"], dropna=False, as_index=False)
        .agg(
            event_fgm=("is_made", "size"),
            event_two_pm=("event_type", lambda values: int(values.eq("2pt").sum())),
            event_three_pm=("event_type", lambda values: int(values.eq("3pt").sum())),
            event_assisted_fgm=("assist_status", lambda values: int(values.eq("assisted").sum())),
            event_unassisted_fgm=("assist_status", lambda values: int(values.eq("unassisted").sum())),
            event_unknown_assist_fgm=("assist_status", lambda values: int(values.eq("unknown").sum())),
        )
    )
    official = _read_official_team_boxes(sorted(event["game_id"].unique()), Path(processed_players_dir))
    result = event.merge(
        official,
        on=["game_id", "team_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    result["season"] = season
    for column in [
        "event_fgm",
        "event_two_pm",
        "event_three_pm",
        "event_assisted_fgm",
        "event_unassisted_fgm",
        "event_unknown_assist_fgm",
    ]:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    for column in ["official_fgm", "official_two_pm", "official_three_pm", "official_assists"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    result["fgm_delta"] = result["event_fgm"] - result["official_fgm"]
    result["two_pm_delta"] = result["event_two_pm"] - result["official_two_pm"]
    result["three_pm_delta"] = result["event_three_pm"] - result["official_three_pm"]
    result["assists_delta"] = result["event_assisted_fgm"] - result["official_assists"]
    result["box_available"] = result["_merge"].ne("left_only")
    result["pbp_available"] = result["_merge"].ne("right_only")
    result["fgm_exact"] = result["fgm_delta"].eq(0)
    result["two_pm_exact"] = result["two_pm_delta"].eq(0)
    result["three_pm_exact"] = result["three_pm_delta"].eq(0)
    result["assists_exact"] = result["assists_delta"].eq(0)
    return result.loc[
        :,
        [
            "season",
            "game_id",
            "team_id",
            "team_tricode",
            "box_team_tricode",
            "pbp_available",
            "box_available",
            "event_fgm",
            "official_fgm",
            "fgm_delta",
            "fgm_exact",
            "event_two_pm",
            "official_two_pm",
            "two_pm_delta",
            "two_pm_exact",
            "event_three_pm",
            "official_three_pm",
            "three_pm_delta",
            "three_pm_exact",
            "event_assisted_fgm",
            "event_unassisted_fgm",
            "event_unknown_assist_fgm",
            "official_assists",
            "assists_delta",
            "assists_exact",
        ],
    ].sort_values(["game_id", "team_id"], kind="stable").reset_index(drop=True)


def validate_assisted_shot_taxonomy(path: Path | str) -> AssistedShotTaxonomyManifest:
    """Validate published assisted-shot artifacts, keys, hashes, and arithmetic."""

    root = Path(path)
    manifest = AssistedShotTaxonomyManifest.model_validate_json((root / "_manifest.json").read_text())
    expected = {artifact.filename for artifact in manifest.artifacts}
    actual = {item.name for item in root.iterdir() if item.is_file() and item.name != "_manifest.json"}
    if actual != expected:
        raise ValueError("Assisted shot taxonomy artifacts do not match manifest")
    for artifact in manifest.artifacts:
        output = root / artifact.filename
        if output.stat().st_size != artifact.byte_count or _sha256_file(output) != artifact.sha256:
            raise ValueError(f"Assisted shot taxonomy artifact integrity changed: {artifact.filename}")
    profiles = pd.read_parquet(root / "player_season_assisted_shot_profiles.parquet")
    coverage = pd.read_parquet(root / "season_assisted_shot_coverage.parquet")
    reconciliation = pd.read_parquet(root / "game_assisted_shot_reconciliation.parquet")
    if len(profiles) != manifest.row_count or profiles.duplicated(["season", "player_id"]).any():
        raise ValueError("Assisted shot taxonomy player-season keys changed")
    if tuple(profiles["season"].drop_duplicates()) != manifest.seasons:
        raise ValueError("Assisted shot taxonomy profile seasons changed")
    if set(coverage["season"].astype(str)) != set(manifest.seasons):
        raise ValueError("Assisted shot taxonomy coverage seasons changed")
    if reconciliation.duplicated(["season", "game_id", "team_id"]).any():
        raise ValueError("Assisted shot reconciliation keys changed")
    status_total = profiles[["assisted_makes", "unassisted_makes", "unknown_assist_status_makes"]].sum(axis=1)
    if not status_total.eq(profiles["total_makes"]).all():
        raise ValueError("Assisted shot profile make statuses do not sum to total makes")
    return manifest


def _read_event_partition(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Assisted shot taxonomy has no event parquet files in {root}")
    columns = [
        "game_id",
        "team_id",
        "team_tricode",
        "event_type",
        "event_subtype",
        "player_id",
        "shot_result",
        "description",
    ]
    return pd.concat([pd.read_parquet(file, columns=columns) for file in files], ignore_index=True)


def _read_official_team_boxes(game_ids: list[str], root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    columns = [
        "game_id",
        "team_id",
        "team_tricode",
        "statistics_fieldGoalsMade",
        "statistics_threePointersMade",
        "statistics_assists",
    ]
    for game_id in game_ids:
        path = root / f"{game_id}.parquet"
        if not path.is_file():
            continue
        players = pd.read_parquet(path, columns=columns)
        statistics = [column for column in columns if column.startswith("statistics_")]
        for column in statistics:
            players[column] = pd.to_numeric(players[column], errors="coerce").fillna(0).astype("int64")
        summary = (
            players.groupby(["game_id", "team_id", "team_tricode"], as_index=False)[statistics]
            .sum()
            .rename(
                columns={
                    "team_tricode": "box_team_tricode",
                    "statistics_fieldGoalsMade": "official_fgm",
                    "statistics_threePointersMade": "official_three_pm",
                    "statistics_assists": "official_assists",
                }
            )
        )
        summary["official_two_pm"] = summary["official_fgm"] - summary["official_three_pm"]
        rows.append(summary)
    if not rows:
        return pd.DataFrame(
            columns=[
                "game_id",
                "team_id",
                "box_team_tricode",
                "official_fgm",
                "official_two_pm",
                "official_three_pm",
                "official_assists",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _summarize_reconciliation(season: str, reconciliation: pd.DataFrame) -> dict[str, object]:
    comparable = reconciliation.loc[
        reconciliation["pbp_available"] & reconciliation["box_available"]
    ].copy()
    if comparable.empty:
        raise ValueError(f"Assisted shot reconciliation has no comparable team-games in {season}")
    output: dict[str, object] = {
        "season": season,
        "team_game_count": int(len(reconciliation)),
        "comparable_team_game_count": int(len(comparable)),
        "box_missing_team_game_count": int((~reconciliation["box_available"]).sum()),
        "pbp_missing_team_game_count": int((~reconciliation["pbp_available"]).sum()),
        "unknown_assist_status_make_count": int(comparable["event_unknown_assist_fgm"].sum()),
        "unknown_assist_status_make_rate": float(
            comparable["event_unknown_assist_fgm"].sum() / comparable["event_fgm"].sum()
        ),
    }
    for name in ("fgm", "two_pm", "three_pm", "assists"):
        delta = pd.to_numeric(comparable[f"{name}_delta"], errors="raise")
        output[f"{name}_exact_team_game_rate"] = float(comparable[f"{name}_exact"].mean())
        output[f"{name}_mean_delta"] = float(delta.mean())
        output[f"{name}_max_abs_delta"] = int(delta.abs().max())
    return output


def _validate_panel(panel: pd.DataFrame) -> None:
    required = {"season", "season_start_year", "player_id", "player_name", "rapm_possessions"}
    missing = required - set(panel)
    if missing:
        raise ValueError(f"Assisted shot taxonomy player panel missing columns: {sorted(missing)}")
    if panel.duplicated(["season", "player_id"]).any():
        raise ValueError("Assisted shot taxonomy player panel must be unique by season and player")


def _validate_season_panel(season: str, panel: pd.DataFrame) -> None:
    _validate_panel(panel)
    if panel["season"].nunique() != 1 or not panel["season"].eq(season).all():
        raise ValueError("Assisted shot taxonomy season panel must contain exactly one matching season")


def _resolve_seasons(
    available_seasons: tuple[str, ...], selected_seasons: tuple[str, ...] | None
) -> tuple[str, ...]:
    if selected_seasons is None:
        return available_seasons
    selected = tuple(sorted({validate_season(season) for season in selected_seasons}, key=_season_start_year))
    missing = set(selected) - set(available_seasons)
    if missing:
        raise ValueError(f"Assisted shot taxonomy requested unavailable seasons: {sorted(missing)}")
    return selected


def _season_start_year(season: str) -> int:
    return int(validate_season(season)[:4])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validated assisted made-shot profiles")
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--curated-dir", default=str(DEFAULT_CURATED_DIR))
    parser.add_argument("--processed-players-dir", default=str(DEFAULT_PROCESSED_PLAYERS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--season",
        action="append",
        dest="seasons",
        help="Build one or more named seasons; repeat for a resumable historical build.",
    )
    parser.add_argument(
        "--combine-parts-dir",
        help="Combine independently validated one-season outputs from this directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Build all seasons concurrently as validated parts, then combine them.",
    )
    args = parser.parse_args()
    if args.combine_parts_dir:
        manifest, output = combine_assisted_shot_taxonomy_parts(
            parts_root=args.combine_parts_dir,
            output_dir=args.output_dir,
        )
        print(
            "Assisted shot taxonomy combined: "
            f"seasons={len(manifest.seasons):,} rows={manifest.row_count:,} output={output}"
        )
        return
    if args.workers:
        if args.seasons:
            raise ValueError("--workers cannot be combined with --season")
        manifest, output = build_assisted_shot_taxonomy_parallel(
            player_season_panel_path=args.player_season_panel_path,
            curated_dir=args.curated_dir,
            processed_players_dir=args.processed_players_dir,
            output_dir=args.output_dir,
            workers=args.workers,
        )
        print(
            "Assisted shot taxonomy: "
            f"seasons={len(manifest.seasons):,} rows={manifest.row_count:,} output={output}"
        )
        return
    manifest, output = build_assisted_shot_taxonomy(
        player_season_panel_path=args.player_season_panel_path,
        curated_dir=args.curated_dir,
        processed_players_dir=args.processed_players_dir,
        output_dir=args.output_dir,
        seasons=tuple(args.seasons) if args.seasons else None,
    )
    print(
        "Assisted shot taxonomy: "
        f"seasons={len(manifest.seasons):,} rows={manifest.row_count:,} output={output}"
    )


if __name__ == "__main__":
    main()
