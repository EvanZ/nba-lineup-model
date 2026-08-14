"""Canonical historical shot families and possession-based player profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
from nba_lineup_model.season.schema import SHA256_PATTERN, validate_season

DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_OUTPUT_DIR = Path("data/analytical/shot_taxonomy")

SHOT_FAMILIES = ("rim", "non_rim_two", "three")
RIM_SUBTYPE_TOKENS = ("layup", "dunk", "tip", "finger roll", "alley oop")
RATE_PSEUDO_POSSESSIONS = 300.0
EFFICIENCY_PSEUDO_ATTEMPTS = 75.0


class ShotTaxonomyManifest(BaseModel):
    """Integrity and semantics contract for the player-season shot mart."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    created_at: datetime
    builder_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    source_player_season_panel_path: str
    source_player_season_panel_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    seasons: tuple[str, ...] = Field(min_length=1)
    row_count: int = Field(ge=1)
    shot_families: tuple[str, ...]
    rate_pseudo_possessions: float = Field(gt=0)
    efficiency_pseudo_attempts: float = Field(gt=0)
    source_event_manifest_sha256: dict[str, str]
    artifacts: tuple[ArtifactRecord, ...] = Field(min_length=3)

    @field_validator("seasons")
    @classmethod
    def validate_season_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_season(value) for value in values)

    @model_validator(mode="after")
    def validate_contract(self) -> ShotTaxonomyManifest:
        if self.seasons != tuple(sorted(self.seasons, key=_season_start_year)):
            raise ValueError("Shot taxonomy seasons must be chronological")
        if self.shot_families != SHOT_FAMILIES:
            raise ValueError("Shot taxonomy family contract changed")
        if set(self.source_event_manifest_sha256) != set(self.seasons):
            raise ValueError("Shot taxonomy must record every source event manifest")
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Shot taxonomy artifact filenames must be unique")
        return self


def build_shot_taxonomy(
    *,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[ShotTaxonomyManifest, Path]:
    """Build an atomic, historical player-season shot taxonomy mart."""

    panel_path = Path(player_season_panel_path)
    panel_root = panel_path.parent
    validate_player_season_panel(panel_root)
    panel = pd.read_parquet(panel_path)
    _validate_panel(panel)
    seasons = tuple(sorted(panel["season"].astype(str).unique(), key=_season_start_year))
    curated_root = Path(curated_dir)

    profiles: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    subtype_coverage: list[pd.DataFrame] = []
    references: list[pd.DataFrame] = []
    event_manifests: dict[str, str] = {}
    for season in seasons:
        event_root = curated_root / "events" / season / "regular"
        manifest_path = event_root / "_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Shot taxonomy is missing events manifest for {season}")
        event_manifests[season] = _sha256_file(manifest_path)
        events = _read_event_partition(event_root)
        classified = classify_shot_events(events)
        season_panel = panel.loc[panel["season"].eq(season)].copy()
        season_profiles, season_references, season_coverage = build_season_shot_profiles(
            season,
            season_panel,
            classified,
        )
        profiles.append(season_profiles)
        references.append(season_references)
        coverage.append(season_coverage)
        subtype_coverage.append(_subtype_coverage(season, classified))

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    backup = output.parent / f".{output.name}.bak-{uuid4().hex}"
    temporary.mkdir()
    try:
        outputs = {
            "player_season_shot_profiles.parquet": pd.concat(profiles, ignore_index=True),
            "season_shot_taxonomy_coverage.parquet": pd.DataFrame(coverage),
            "subtype_shot_taxonomy_coverage.parquet": pd.concat(subtype_coverage, ignore_index=True),
            "league_shot_taxonomy_references.parquet": pd.concat(references, ignore_index=True),
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
        manifest = ShotTaxonomyManifest(
            created_at=datetime.now(UTC),
            builder_code_version=player_history_code_fingerprint((Path(__file__),)),
            source_player_season_panel_path=str(panel_path),
            source_player_season_panel_manifest_sha256=_sha256_file(panel_root / "_manifest.json"),
            seasons=seasons,
            row_count=len(outputs["player_season_shot_profiles.parquet"]),
            shot_families=SHOT_FAMILIES,
            rate_pseudo_possessions=RATE_PSEUDO_POSSESSIONS,
            efficiency_pseudo_attempts=EFFICIENCY_PSEUDO_ATTEMPTS,
            source_event_manifest_sha256=event_manifests,
            artifacts=artifacts,
        )
        (temporary / "_manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        validate_shot_taxonomy(temporary)
        if output.exists():
            output.replace(backup)
        try:
            temporary.replace(output)
        except Exception:
            if backup.exists() and not output.exists():
                backup.replace(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        validate_shot_taxonomy(output)
        return manifest, output
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not output.exists():
            backup.replace(output)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


def classify_shot_events(events: pd.DataFrame) -> pd.DataFrame:
    """Classify canonical two- and three-point events into stable shot families."""

    required = {"event_type", "event_subtype", "player_id", "shot_result"}
    missing = required - set(events)
    if missing:
        raise ValueError(f"Shot taxonomy event frame missing columns: {sorted(missing)}")
    output = events.loc[:, ["event_type", "event_subtype", "player_id", "shot_result"]].copy()
    output["event_type"] = output["event_type"].astype("string").str.casefold()
    output = output.loc[output["event_type"].isin(["2pt", "3pt"])].copy()
    subtype = output["event_subtype"].astype("string").fillna("")
    normalized = subtype.str.casefold().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    rim = normalized.str.contains("|".join(RIM_SUBTYPE_TOKENS), regex=True, na=False)
    output["shot_family"] = np.where(
        output["event_type"].eq("3pt"),
        "three",
        np.where(rim, "rim", "non_rim_two"),
    )
    output["event_subtype_display"] = subtype.replace("", "<missing>")
    result = output["shot_result"].astype("string").str.casefold()
    output["is_made"] = result.eq("made")
    output["has_final_result"] = result.isin(["made", "missed"])
    output["player_id"] = pd.to_numeric(output["player_id"], errors="coerce").astype("Int64")
    return output.reset_index(drop=True)


def build_season_shot_profiles(
    season: str,
    player_season_panel: pd.DataFrame,
    classified_shots: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return one season's raw and stabilized possession-native shot profiles."""

    validate_season(season)
    required_panel = {"season", "player_id", "player_name", "rapm_possessions"}
    missing_panel = required_panel - set(player_season_panel)
    if missing_panel:
        raise ValueError(f"Shot taxonomy panel missing columns: {sorted(missing_panel)}")
    if player_season_panel["season"].nunique() != 1 or not player_season_panel["season"].eq(season).all():
        raise ValueError("Shot taxonomy season panel must contain exactly one matching season")
    if player_season_panel["player_id"].duplicated().any():
        raise ValueError("Shot taxonomy season panel must be unique by player")
    profiles = player_season_panel.loc[
        :, ["season", "season_start_year", "player_id", "player_name", "rapm_possessions"]
    ].copy()
    profiles["player_id"] = pd.to_numeric(profiles["player_id"], errors="raise").astype("int64")
    profiles["on_court_possessions"] = pd.to_numeric(
        profiles.pop("rapm_possessions"), errors="raise"
    ).astype(float)
    if not profiles["on_court_possessions"].gt(0).all():
        raise ValueError(f"Shot taxonomy profiles require positive possessions in {season}")

    attributed = classified_shots.loc[
        classified_shots["has_final_result"] & classified_shots["player_id"].notna()
    ].copy()
    attributed["player_id"] = attributed["player_id"].astype("int64")
    panel_ids = set(profiles["player_id"])
    matched = attributed.loc[attributed["player_id"].isin(panel_ids)].copy()
    counts = (
        matched.groupby(["player_id", "shot_family"], as_index=False)
        .agg(attempts=("is_made", "size"), makes=("is_made", "sum"))
        .astype({"attempts": "int64", "makes": "int64"})
    )
    for family in SHOT_FAMILIES:
        family_counts = counts.loc[counts["shot_family"].eq(family), ["player_id", "attempts", "makes"]]
        family_counts = family_counts.rename(
            columns={"attempts": f"{family}_attempts", "makes": f"{family}_makes"}
        )
        profiles = profiles.merge(family_counts, on="player_id", how="left", validate="one_to_one")
        profiles[[f"{family}_attempts", f"{family}_makes"]] = profiles[
            [f"{family}_attempts", f"{family}_makes"]
        ].fillna(0).astype("int64")
    profiles["total_shot_attempts"] = profiles[[f"{family}_attempts" for family in SHOT_FAMILIES]].sum(axis=1)
    profiles["total_shot_makes"] = profiles[[f"{family}_makes" for family in SHOT_FAMILIES]].sum(axis=1)

    references = _league_references(season, profiles)
    reference_by_family = references.set_index("shot_family")
    for family in SHOT_FAMILIES:
        attempts = profiles[f"{family}_attempts"].to_numpy(dtype=float)
        makes = profiles[f"{family}_makes"].to_numpy(dtype=float)
        denominator = profiles["on_court_possessions"].to_numpy(dtype=float)
        league_rate = float(reference_by_family.loc[family, "attempts_per_100"])
        league_pct = float(reference_by_family.loc[family, "fg_pct"])
        profiles[f"{family}_attempts_per_100"] = 100.0 * (
            attempts + RATE_PSEUDO_POSSESSIONS * league_rate / 100.0
        ) / (denominator + RATE_PSEUDO_POSSESSIONS)
        profiles[f"{family}_fg_pct"] = np.divide(
            makes,
            attempts,
            out=np.full(len(profiles), np.nan, dtype=float),
            where=attempts > 0,
        )
        profiles[f"{family}_fg_pct_shrunk"] = (
            makes + EFFICIENCY_PSEUDO_ATTEMPTS * league_pct
        ) / (attempts + EFFICIENCY_PSEUDO_ATTEMPTS)
    profiles["effective_field_goal_pct"] = np.divide(
        profiles["total_shot_makes"] + 0.5 * profiles["three_makes"],
        profiles["total_shot_attempts"],
        out=np.full(len(profiles), np.nan, dtype=float),
        where=profiles["total_shot_attempts"].to_numpy(dtype=float) > 0,
    )
    profiles["shot_profile_available"] = profiles["total_shot_attempts"].gt(0)

    coverage = {
        "season": season,
        "source_shot_event_count": int(len(classified_shots)),
        "final_result_shot_event_count": int(classified_shots["has_final_result"].sum()),
        "player_attributed_shot_event_count": int(
            (classified_shots["has_final_result"] & classified_shots["player_id"].notna()).sum()
        ),
        "panel_matched_shot_event_count": int(len(matched)),
        "unmatched_player_shot_event_count": int(
            len(attributed) - len(matched)
        ),
        "profile_player_count": int(len(profiles)),
        "profile_player_with_shot_count": int(profiles["shot_profile_available"].sum()),
        **{
            f"{family}_shot_event_count": int(
                classified_shots["shot_family"].eq(family).sum()
            )
            for family in SHOT_FAMILIES
        },
    }
    return profiles, references, coverage


def validate_shot_taxonomy(path: Path | str) -> ShotTaxonomyManifest:
    """Validate published taxonomy artifacts, keys, hashes, and coverage."""

    root = Path(path)
    manifest = ShotTaxonomyManifest.model_validate_json((root / "_manifest.json").read_text())
    expected = {artifact.filename for artifact in manifest.artifacts}
    actual = {item.name for item in root.iterdir() if item.is_file() and item.name != "_manifest.json"}
    if actual != expected:
        raise ValueError("Shot taxonomy artifacts do not match manifest")
    for artifact in manifest.artifacts:
        output = root / artifact.filename
        if output.stat().st_size != artifact.byte_count or _sha256_file(output) != artifact.sha256:
            raise ValueError(f"Shot taxonomy artifact integrity changed: {artifact.filename}")
    profiles = pd.read_parquet(root / "player_season_shot_profiles.parquet")
    coverage = pd.read_parquet(root / "season_shot_taxonomy_coverage.parquet")
    if len(profiles) != manifest.row_count or profiles.duplicated(["season", "player_id"]).any():
        raise ValueError("Shot taxonomy player-season keys changed")
    if tuple(profiles["season"].drop_duplicates()) != manifest.seasons:
        raise ValueError("Shot taxonomy profile seasons changed")
    if set(coverage["season"].astype(str)) != set(manifest.seasons):
        raise ValueError("Shot taxonomy coverage seasons changed")
    return manifest


def _read_event_partition(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Shot taxonomy has no event parquet files in {root}")
    return pd.concat(
        [
            pd.read_parquet(
                file,
                columns=["event_type", "event_subtype", "player_id", "shot_result"],
            )
            for file in files
        ],
        ignore_index=True,
    )


def _league_references(season: str, profiles: pd.DataFrame) -> pd.DataFrame:
    possessions = float(profiles["on_court_possessions"].sum())
    rows = []
    for family in SHOT_FAMILIES:
        attempts = float(profiles[f"{family}_attempts"].sum())
        makes = float(profiles[f"{family}_makes"].sum())
        if attempts <= 0:
            raise ValueError(f"Shot taxonomy found no {family} attempts in {season}")
        rows.append(
            {
                "season": season,
                "shot_family": family,
                "attempts": int(attempts),
                "makes": int(makes),
                "attempts_per_100": 100.0 * attempts / possessions,
                "fg_pct": makes / attempts,
            }
        )
    return pd.DataFrame(rows)


def _subtype_coverage(season: str, classified: pd.DataFrame) -> pd.DataFrame:
    return (
        classified.groupby(
            ["event_type", "event_subtype_display", "shot_family"], dropna=False, as_index=False
        )
        .agg(
            shot_event_count=("shot_family", "size"),
            final_result_count=("has_final_result", "sum"),
            made_count=("is_made", "sum"),
        )
        .assign(season=season)
        .loc[:, ["season", "event_type", "event_subtype_display", "shot_family", "shot_event_count", "final_result_count", "made_count"]]
        .sort_values(["event_type", "shot_event_count"], ascending=[True, False], kind="stable")
        .reset_index(drop=True)
    )


def _validate_panel(panel: pd.DataFrame) -> None:
    required = {"season", "season_start_year", "player_id", "player_name", "rapm_possessions"}
    missing = required - set(panel)
    if missing:
        raise ValueError(f"Shot taxonomy player panel missing columns: {sorted(missing)}")
    if panel.duplicated(["season", "player_id"]).any():
        raise ValueError("Shot taxonomy player panel must be unique by season and player")


def _season_start_year(season: str) -> int:
    return int(validate_season(season)[:4])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical player-season shot taxonomy profiles")
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--curated-dir", default=str(DEFAULT_CURATED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest, output = build_shot_taxonomy(
        player_season_panel_path=args.player_season_panel_path,
        curated_dir=args.curated_dir,
        output_dir=args.output_dir,
    )
    print(
        f"Shot taxonomy: seasons={len(manifest.seasons):,} rows={manifest.row_count:,} output={output}"
    )


if __name__ == "__main__":
    main()
