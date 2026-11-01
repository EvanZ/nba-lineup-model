"""Build a local current-season serving bundle from completed updater states."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_lineup_model.season.current_serving import (
    ServingPromotionConfig,
    promote_serving_bundle,
    publish_serving_bundle,
)
from nba_lineup_model.season.daily_data_artifact import verify_daily_data_artifact
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact
from nba_lineup_model.season.slate_time import PACIFIC

DEFAULT_RATING_ROOT = Path("/var/lib/nba-gestalt/in-season-nail")
DEFAULT_MINUTES_ROOT = Path("/var/lib/nba-gestalt/in-season-minutes")
DEFAULT_RELEASE_ROOT = Path("/opt/nba-gestalt/current")
DEFAULT_OUTPUT_ROOT = Path("/var/lib/nba-gestalt-dev/current-season-serving")
DEFAULT_DAILY_MODEL_ROOT = Path("/var/lib/nba-gestalt-dev/daily-model-nba-v1")
DEFAULT_DAILY_DATA_ROOT = Path("/var/lib/nba-gestalt-dev/daily-data-nba-v2")


def materialize_current_season_serving(
    *,
    rating_root: Path | str = DEFAULT_RATING_ROOT,
    minutes_root: Path | str = DEFAULT_MINUTES_ROOT,
    release_root: Path | str = DEFAULT_RELEASE_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Combine synchronized rating/minutes publications without writing to S3."""

    rating_root = Path(rating_root)
    minutes_root = Path(minutes_root)
    release_root = Path(release_root)
    output_root = Path(output_root)
    rating_publication = _load_json(rating_root / "latest.json")
    minutes_publication = _load_json(minutes_root / "latest.json")
    through_date = str(rating_publication.get("through_date", ""))
    if not through_date or through_date != str(minutes_publication.get("through_date", "")):
        raise ValueError("Rating and minutes publication dates do not match")
    for label, publication in (
        ("rating", rating_publication),
        ("minutes", minutes_publication),
    ):
        coverage = publication.get("coverage")
        if not isinstance(coverage, dict) or int(coverage.get("excluded", -1)) < 0:
            raise ValueError(f"{label.title()} publication coverage is invalid")
    source_exclusions = {
        "ratings": _publication_exclusions(
            rating_publication,
            root=rating_root,
            summary_key="excluded_contributions",
        ),
        "minutes": _publication_exclusions(
            minutes_publication,
            root=minutes_root,
            summary_key="excluded_games",
        ),
    }

    season = str(rating_publication.get("season", ""))
    if season != str(minutes_publication.get("season", "")):
        raise ValueError("Rating and minutes publication seasons do not match")
    rating_uri = _artifact_uri(rating_publication, "player_state")
    rating_trajectory_uri = _artifact_uri(
        rating_publication, "player_game_trajectory"
    )
    minutes_uri = _artifact_uri(minutes_publication, "player_state")
    rating_digest = _artifact_digest(rating_uri)
    rating_trajectory_digest = _artifact_digest(rating_trajectory_uri)
    minutes_digest = _artifact_digest(minutes_uri)
    rating_matches = sorted(
        rating_root.glob(
            f"player_states/season={season}/cutoff=*/"
            f"state_sha256={rating_digest}/player_state.parquet"
        )
    )
    if len(rating_matches) != 1:
        raise ValueError(
            f"Expected one local rating state for {rating_digest}, found {len(rating_matches)}"
        )
    if _file_sha256(rating_matches[0]) != rating_digest:
        raise ValueError("Local rating state does not match its published checksum")
    rating_run_id = str(rating_publication.get("dagster_run_id", ""))
    rating_trajectory_path = (
        rating_root
        / "runs"
        / f"through_date={through_date}"
        / rating_run_id
        / "player_game_trajectory.parquet"
    )
    if not rating_run_id or not rating_trajectory_path.is_file():
        raise FileNotFoundError(
            f"Local player-game trajectory is missing: {rating_trajectory_path}"
        )
    if _file_sha256(rating_trajectory_path) != rating_trajectory_digest:
        raise ValueError(
            "Local player-game trajectory does not match its published checksum"
        )
    minutes_run_id = str(minutes_publication.get("dagster_run_id", ""))
    minutes_path = (
        minutes_root
        / "runs"
        / f"through_date={through_date}"
        / minutes_run_id
        / "player_state.parquet"
    )
    if not minutes_path.is_file():
        raise FileNotFoundError(f"Local minutes state is missing: {minutes_path}")
    if _file_sha256(minutes_path) != minutes_digest:
        raise ValueError("Local minutes state does not match its published checksum")

    release = _load_json(release_root / "release.json")
    run_id = str(release.get("run_id", ""))
    if not run_id:
        raise ValueError("Production release metadata has no preseason run ID")
    evidence_date = datetime.fromisoformat(through_date).date()
    next_midnight_pacific = datetime.fromisoformat(
        f"{through_date}T00:00:00"
    ).replace(tzinfo=PACIFIC) + timedelta(days=1)
    bundle = publish_serving_bundle(
        rating_matches[0],
        minutes_path,
        root=output_root,
        season=season,
        evidence_through_date=evidence_date,
        as_of_cutoff_time=next_midnight_pacific,
        config=ServingPromotionConfig(preseason_run_id=run_id),
        rating_source_uri=rating_uri,
        minutes_source_uri=minutes_uri,
        rating_trajectory_path=rating_trajectory_path,
        rating_trajectory_source_uri=rating_trajectory_uri,
        source_exclusions=source_exclusions,
        source_coverage={
            "ratings": dict(rating_publication["coverage"]),
            "minutes": dict(minutes_publication["coverage"]),
        },
    )
    promote_serving_bundle(bundle.path, root=output_root)
    return bundle.path


def materialize_daily_model_serving(
    *,
    daily_model_root: Path | str = DEFAULT_DAILY_MODEL_ROOT,
    daily_data_root: Path | str = DEFAULT_DAILY_DATA_ROOT,
    release_root: Path | str = DEFAULT_RELEASE_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Promote one serving bundle from the verified daily-model chain tip.

    Unlike the legacy shadow-publication adapter above, this path consumes the
    ratings, minutes, and trajectory components from one content-addressed
    model checkpoint.  Those three products therefore cannot advance on
    different dates or against different frozen-prior universes.
    """

    model_root = Path(daily_model_root).resolve()
    data_root = Path(daily_data_root).resolve()
    release_root = Path(release_root)
    output_root = Path(output_root)
    pointer = _load_json(model_root / "latest.json")
    raw_bundle_path = str(pointer.get("bundle_path", "")).strip()
    if not raw_bundle_path:
        raise ValueError("Daily-model pointer has no bundle path")
    model_bundle = (model_root / raw_bundle_path).resolve()
    try:
        model_bundle.relative_to(model_root)
    except ValueError as error:
        raise ValueError("Daily-model pointer escapes its root") from error
    manifest = verify_daily_model_artifact(model_bundle)
    if str(pointer.get("bundle_sha256", "")) != str(manifest["bundle_sha256"]):
        raise ValueError("Daily-model pointer checksum does not match its bundle")

    identity = manifest["identity"]
    components = identity["components"]
    rating_path = _daily_component_path(model_bundle, components, "player_ratings")
    minutes_path = _daily_component_path(model_bundle, components, "minutes_state")
    trajectory_path = _daily_component_path(
        model_bundle, components, "rating_trajectory"
    )
    through_date = str(identity["game_date_pacific"])
    evidence_date = datetime.fromisoformat(through_date).date()
    next_midnight_pacific = datetime.fromisoformat(
        f"{through_date}T00:00:00"
    ).replace(tzinfo=PACIFIC) + timedelta(days=1)

    release = _load_json(release_root / "release.json")
    run_id = str(release.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("Production release metadata has no preseason run ID")
    provenance = identity["model_provenance"]
    model_sha = str(manifest["bundle_sha256"])
    game_count = int(identity["cumulative_game_count"])
    coverage = {"games": game_count, "excluded": 0}
    with tempfile.TemporaryDirectory(prefix="daily-serving-trajectory-") as temp_dir:
        adapted_trajectory = Path(temp_dir) / "rating_trajectory.parquet"
        adapted_minutes = Path(temp_dir) / "minutes_state.parquet"
        _adapt_daily_rating_trajectory(
            source_path=trajectory_path,
            rating_state_path=rating_path,
            daily_data_root=data_root,
            evidence_date=evidence_date,
            output_path=adapted_trajectory,
        )
        _adapt_daily_minutes_state(
            source_path=minutes_path,
            output_path=adapted_minutes,
            cutoff_time=next_midnight_pacific,
            participation_prior_games=4.0,
            conditional_prior_games=0.2,
        )
        bundle = publish_serving_bundle(
            rating_path,
            adapted_minutes,
            root=output_root,
            season=str(identity["season"]),
            evidence_through_date=evidence_date,
            as_of_cutoff_time=next_midnight_pacific,
            config=ServingPromotionConfig(
                preseason_run_id=run_id,
                rating_prior_possessions=6000.0,
                minutes_participation_prior_games=4.0,
                minutes_conditional_prior_games=0.2,
                minutes_conditional_lookback_games=3,
                minutes_role_cutoff_minutes=44,
            ),
            rating_source_uri=f"artifact://daily-model/{model_sha}/player_ratings",
            minutes_source_uri=(
                f"artifact://daily-model/{model_sha}/minutes_state#serving-v1"
            ),
            rating_trajectory_path=adapted_trajectory,
            rating_trajectory_source_uri=(
                f"artifact://daily-model/{model_sha}/rating_trajectory#serving-v1"
            ),
            source_exclusions={"ratings": [], "minutes": []},
            source_coverage={"ratings": coverage, "minutes": coverage},
        )
    promoted_pointer, _ = promote_serving_bundle(bundle.path, root=output_root)
    if str(promoted_pointer["evidence_through_date"]) != through_date:
        raise ValueError("Serving pointer did not retain the daily-model date")
    serving_manifest = _load_json(bundle.path / "manifest.json")
    source_uris = serving_manifest["identity"]["sources"]
    if not all(model_sha in str(source["uri"]) for source in source_uris.values()):
        raise ValueError("Serving components do not share one daily-model identity")
    if str(provenance.get("prior_sha256", "")) == "":
        raise ValueError("Daily-model serving state has no coupled-prior identity")
    return bundle.path


def _adapt_daily_minutes_state(
    *,
    source_path: Path,
    output_path: Path,
    cutoff_time: datetime,
    participation_prior_games: float,
    conditional_prior_games: float,
) -> None:
    """Add serving-contract config fields to the immutable daily state."""

    state = pd.read_parquet(source_path).copy()
    if state.empty or state["player_id"].duplicated().any():
        raise ValueError("Daily minutes state must contain one row per player")
    state["cutoff_time"] = cutoff_time
    state["prior_participation_weight"] = float(participation_prior_games)
    state["prior_conditional_weight"] = float(conditional_prior_games)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state.to_parquet(output_path, index=False)


def _adapt_daily_rating_trajectory(
    *,
    source_path: Path,
    rating_state_path: Path,
    daily_data_root: Path,
    evidence_date: date,
    output_path: Path,
) -> None:
    """Translate the official daily trajectory into the public web schema."""

    trajectory = pd.read_parquet(source_path).copy()
    ratings = pd.read_parquet(rating_state_path).copy()
    required = {
        "season",
        "game_date_pacific",
        "game_id",
        "game_time_utc",
        "player_id",
        "team",
        "opponent",
        "played",
        "injury_status",
        "injury_reason",
        "rating_after",
    }
    missing = sorted(required - set(trajectory))
    if missing:
        raise ValueError("Daily rating trajectory is missing: " + ", ".join(missing))
    rating_required = {
        "player_id",
        "frozen_preseason_nail",
        "raw_on_court_possessions",
        "prior_equivalent_possessions",
        "state_updater_version",
    }
    missing_ratings = sorted(rating_required - set(ratings))
    if missing_ratings:
        raise ValueError("Daily rating state is missing: " + ", ".join(missing_ratings))

    game_possessions: dict[tuple[str, int], float] = {}
    for game_date, bundle, manifest in _daily_data_chain(daily_data_root):
        if game_date > evidence_date:
            continue
        stints = pd.read_parquet(
            _daily_data_component_path(
                bundle, manifest["identity"]["components"], "observed_stints"
            )
        )
        for row in stints.itertuples(index=False):
            for column in ("home_player_ids", "away_player_ids"):
                for player_id in getattr(row, column):
                    key = (str(row.game_id), int(player_id))
                    game_possessions[key] = game_possessions.get(key, 0.0) + float(
                        row.possessions
                    )

    trajectory["player_id"] = trajectory["player_id"].astype(int)
    trajectory["game_id"] = trajectory["game_id"].astype(str)
    trajectory["game_on_court_possessions"] = [
        game_possessions.get((game_id, player_id), 0.0)
        for game_id, player_id in zip(
            trajectory["game_id"], trajectory["player_id"], strict=True
        )
    ]
    played = trajectory["played"].astype(bool)
    if (trajectory.loc[played, "game_on_court_possessions"] <= 0).any():
        missing_keys = trajectory.loc[
            played & trajectory["game_on_court_possessions"].le(0),
            ["game_id", "player_id"],
        ]
        raise ValueError(
            "Played trajectory rows lack official stint exposure: "
            + ", ".join(
                f"{row.game_id}/{row.player_id}"
                for row in missing_keys.itertuples(index=False)
            )
        )
    trajectory = trajectory.sort_values(
        ["player_id", "game_time_utc", "game_id"], kind="stable"
    ).reset_index(drop=True)
    grouped = trajectory.groupby("player_id", sort=False)
    trajectory["game_number"] = grouped.cumcount() + 1
    appearances = grouped["played"].cumsum()
    trajectory["appearance_number"] = appearances.where(trajectory["played"], np.nan)
    trajectory["cumulative_on_court_possessions"] = grouped[
        "game_on_court_possessions"
    ].cumsum()

    baseline = ratings.set_index("player_id")["frozen_preseason_nail"].astype(float)
    prior = ratings.set_index("player_id")["prior_equivalent_possessions"].astype(float)
    updater = ratings["state_updater_version"].astype(str).unique().tolist()
    if len(updater) != 1:
        raise ValueError("Daily rating state has multiple updater versions")
    trajectory["updater_version"] = updater[0]
    trajectory["game_date"] = trajectory["game_date_pacific"].astype(str)
    trajectory["pregame_status"] = trajectory["injury_status"]
    trajectory["pregame_reason"] = trajectory["injury_reason"]
    trajectory["preseason_rating"] = trajectory["player_id"].map(baseline)
    trajectory["updated_rating"] = trajectory["rating_after"].astype(float)
    trajectory["in_season_delta"] = (
        trajectory["updated_rating"] - trajectory["preseason_rating"]
    )
    trajectory["effective_information"] = trajectory[
        "cumulative_on_court_possessions"
    ]
    prior_values = trajectory["player_id"].map(prior)
    trajectory["evidence_weight"] = trajectory["effective_information"] / (
        trajectory["effective_information"] + prior_values
    )

    final_exposure = (
        trajectory.groupby("player_id", sort=False)["cumulative_on_court_possessions"]
        .last()
        .reindex(ratings["player_id"].astype(int), fill_value=0.0)
        .to_numpy(dtype=float)
    )
    expected_exposure = ratings["raw_on_court_possessions"].to_numpy(dtype=float)
    if not np.allclose(final_exposure, expected_exposure, atol=1e-8):
        raise ValueError("Adapted trajectory does not reconstruct final player exposure")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_parquet(output_path, index=False)


def _daily_data_chain(root: Path) -> list[tuple[date, Path, dict[str, Any]]]:
    pointer = _load_json(root / "latest.json")
    bundle = (root / str(pointer["bundle_path"])).resolve()
    rows: list[tuple[date, Path, dict[str, Any]]] = []
    while True:
        try:
            bundle.relative_to(root)
        except ValueError as error:
            raise ValueError("Daily-data chain escapes its root") from error
        manifest = verify_daily_data_artifact(bundle)
        identity = manifest["identity"]
        game_date = date.fromisoformat(str(identity["game_date_pacific"]))
        rows.append((game_date, bundle, manifest))
        parent_sha = identity.get("parent_data_sha256")
        if parent_sha is None:
            break
        prior_date = game_date - timedelta(days=1)
        bundle = (
            root
            / f"season={identity['season']}"
            / f"game_date_pacific={prior_date.isoformat()}"
            / f"data_sha256={parent_sha}"
        ).resolve()
    rows.reverse()
    if str(pointer.get("bundle_sha256", "")) != str(rows[-1][2]["bundle_sha256"]):
        raise ValueError("Daily-data pointer checksum does not match its chain tip")
    return rows


def _daily_data_component_path(
    bundle: Path, components: dict[str, Any], name: str
) -> Path:
    entry = components.get(name)
    raw_path = str(entry.get("path", "")).strip() if isinstance(entry, dict) else ""
    if not raw_path:
        raise ValueError(f"Daily-data component is missing: {name}")
    path = (bundle / raw_path).resolve()
    try:
        path.relative_to(bundle)
    except ValueError as error:
        raise ValueError(f"Daily-data component escapes its bundle: {name}") from error
    return path


def _daily_component_path(
    bundle: Path, components: dict[str, Any], name: str
) -> Path:
    entry = components.get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"Daily-model component is missing: {name}")
    raw_path = str(entry.get("path", "")).strip()
    if not raw_path:
        raise ValueError(f"Daily-model component path is missing: {name}")
    path = (bundle / raw_path).resolve()
    try:
        path.relative_to(bundle)
    except ValueError as error:
        raise ValueError(f"Daily-model component escapes its bundle: {name}") from error
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object: {path}")
    return payload


def _publication_exclusions(
    publication: dict[str, Any],
    *,
    root: Path,
    summary_key: str,
) -> list[dict[str, str]]:
    raw = publication.get("exclusions")
    if raw is None:
        through_date = str(publication.get("through_date", ""))
        run_id = str(publication.get("dagster_run_id", ""))
        summary_path = root / "runs" / f"through_date={through_date}" / run_id / "summary.json"
        if through_date and run_id and summary_path.is_file():
            summary = _load_json(summary_path)
            raw = summary.get(summary_key, [])
        else:
            raw = []
    if not isinstance(raw, list):
        raise ValueError("Publication exclusions must be a list")
    exclusions: list[dict[str, str]] = []
    for issue in raw:
        if not isinstance(issue, dict):
            raise ValueError("Publication exclusion must be an object")
        game_id = str(issue.get("game_id", "")).strip()
        reason = str(issue.get("reason", "")).strip()
        if not game_id or not reason:
            raise ValueError("Publication exclusions require game_id and reason")
        exclusions.append({"game_id": game_id, "reason": reason})
    coverage = publication.get("coverage")
    expected = int(coverage.get("excluded", -1)) if isinstance(coverage, dict) else -1
    if expected < 0 or expected != len(exclusions):
        raise ValueError(
            "Publication exclusion count does not match its named quarantines"
        )
    return sorted(exclusions, key=lambda issue: (issue["game_id"], issue["reason"]))


def _artifact_uri(publication: dict[str, Any], key: str) -> str:
    artifacts = publication.get("artifacts")
    uri = artifacts.get(key) if isinstance(artifacts, dict) else None
    if not isinstance(uri, str) or not uri:
        raise ValueError(f"Publication is missing artifact URI: {key}")
    return uri


def _artifact_digest(uri: str) -> str:
    stem = Path(uri).stem
    digest = stem.rsplit("-", 1)[-1]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"Artifact URI has no SHA-256 identity: {uri}")
    return digest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rating-root", type=Path, default=DEFAULT_RATING_ROOT)
    parser.add_argument("--minutes-root", type=Path, default=DEFAULT_MINUTES_ROOT)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--daily-model-root", type=Path)
    parser.add_argument("--daily-data-root", type=Path, default=DEFAULT_DAILY_DATA_ROOT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.daily_model_root is not None:
        path = materialize_daily_model_serving(
            daily_model_root=args.daily_model_root,
            daily_data_root=args.daily_data_root,
            release_root=args.release_root,
            output_root=args.output_root,
        )
    else:
        path = materialize_current_season_serving(
            rating_root=args.rating_root,
            minutes_root=args.minutes_root,
            release_root=args.release_root,
            output_root=args.output_root,
        )
    print(path)


if __name__ == "__main__":
    main()
