"""Build a local current-season serving bundle from completed updater states."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from nba_lineup_model.season.current_serving import (
    ServingPromotionConfig,
    promote_serving_bundle,
    publish_serving_bundle,
)
from nba_lineup_model.season.slate_time import PACIFIC

DEFAULT_RATING_ROOT = Path("/var/lib/nba-gestalt/in-season-nail")
DEFAULT_MINUTES_ROOT = Path("/var/lib/nba-gestalt/in-season-minutes")
DEFAULT_RELEASE_ROOT = Path("/opt/nba-gestalt/current")
DEFAULT_OUTPUT_ROOT = Path("/var/lib/nba-gestalt-dev/current-season-serving")


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
        if not isinstance(coverage, dict) or int(coverage.get("excluded", -1)) != 0:
            raise ValueError(f"{label.title()} publication is not exclusion-free")

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
    )
    promote_serving_bundle(bundle.path, root=output_root)
    return bundle.path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object: {path}")
    return payload


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
    return parser


def main() -> None:
    args = _parser().parse_args()
    path = materialize_current_season_serving(
        rating_root=args.rating_root,
        minutes_root=args.minutes_root,
        release_root=args.release_root,
        output_root=args.output_root,
    )
    print(path)


if __name__ == "__main__":
    main()
