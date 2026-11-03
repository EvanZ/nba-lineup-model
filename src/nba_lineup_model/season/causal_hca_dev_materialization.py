"""Materialize a complete dev Schedule from one causal HCA model chain.

The source Schedule is read-only.  Its frozen minutes, injury assumptions,
market snapshots, B2B values, scores, and official NBA box scores are copied
byte-for-byte before only model-derived projections, ledgers, grading, and
retrospectives are replaced in the isolated output tree.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from nba_lineup_model.season.daily_artifact_common import canonical_json, file_sha256
from nba_lineup_model.season.daily_model_replay import _component_path, _read_model_chain
from nba_lineup_model.season.game_prediction_artifact import (
    build_spread_ledger,
    complete_projected_minutes_attribution,
    require_projection_reconciliation,
)
from nba_lineup_model.season.official_schedule_runtime import (
    _build_official_actual_minutes_retrospective,
    _publish_pregame_candidate,
)
from nba_lineup_model.web_api.win_projections import _build_final_game_archive


@dataclass(frozen=True)
class PregameState:
    as_of: date
    ratings: dict[int, float]
    home_court: float
    bundle_sha256: str
    rating_source_uri: str
    home_court_source_uri: str
    home_court_sha256: str
    cumulative_game_count: int


def materialize_causal_hca_dev_schedule(
    *,
    source_schedule_root: Path | str,
    output_schedule_root: Path | str,
    candidate_model_root: Path | str,
    preseason_rankings_path: Path | str,
    start_date: date,
    evidence_through_date: date,
    slate_through_date: date,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    """Write a side-by-side candidate Schedule without mutating source data."""

    source = Path(source_schedule_root).resolve()
    output = Path(output_schedule_root).resolve()
    model_root = Path(candidate_model_root).resolve()
    if output.exists():
        raise FileExistsError(f"Candidate Schedule output already exists: {output}")
    if slate_through_date < evidence_through_date:
        raise ValueError("Slate coverage cannot end before model evidence")
    source_sha_before = _tree_sha256(source)
    shutil.copytree(source, output)
    source_manifest = _read_json(source / "manifest.json")
    source_manifest_identity = str(source_manifest.get("identity", ""))
    source_pregame_hashes = source_manifest.get("pregame_sha256_by_game")
    candidate_pregame_hashes = (
        dict(source_pregame_hashes)
        if isinstance(source_pregame_hashes, dict)
        else {}
    )
    states, preseason_ratings, prior_hca, prior_sha, preseason_sha = _load_states(
        model_root=model_root,
        preseason_rankings_path=Path(preseason_rankings_path).resolve(),
    )
    if not states or states[-1].as_of != evidence_through_date:
        raise ValueError("Candidate model tip does not match the requested evidence date")
    completed_frozen = [
        _read_json(path)
        for path in sorted((source / "frozen").glob("*.json"))
        if start_date <= _game_date(_read_json(path)) <= evidence_through_date
    ]
    probability_scale = _probability_scale(completed_frozen)
    now = captured_at or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Candidate capture time must be timezone-aware")

    rewritten_games = 0
    completed_games = 0
    pending_candidates = 0
    game_ids: list[str] = []
    completed_game_ids: list[str] = []
    for latest_path in sorted((output / "latest").glob("*.json")):
        game_date = date.fromisoformat(latest_path.stem)
        if not start_date <= game_date <= slate_through_date:
            continue
        payload = _read_json(latest_path)
        games = payload.get("games")
        if not isinstance(games, list):
            raise ValueError(f"Schedule day has no games: {latest_path}")
        state = _pregame_state(
            game_date=game_date,
            states=states,
            preseason_ratings=preseason_ratings,
            prior_hca=prior_hca,
            prior_sha=prior_sha,
            preseason_sha=preseason_sha,
        )
        for index, game in enumerate(games):
            if not isinstance(game, dict):
                raise ValueError(f"Schedule day has an invalid game: {latest_path}")
            game_id = str(game.get("game_id", ""))
            frozen_path = output / "frozen" / f"{game_id}.json"
            is_completed = bool(game.get("status", {}).get("completed"))
            source_pregame = _read_json(frozen_path) if frozen_path.is_file() else game
            candidate_pregame = _candidate_pregame(
                source_pregame,
                state=state,
                probability_scale=probability_scale,
            )
            candidate_pregame_hashes[game_id] = hashlib.sha256(
                canonical_json(candidate_pregame)
            ).hexdigest()
            served = copy.deepcopy(game)
            served["projection"] = copy.deepcopy(candidate_pregame["projection"])
            served["projected_minutes"] = copy.deepcopy(
                candidate_pregame["projected_minutes"]
            )
            # A frozen pregame is immutable evidence of the inputs that were in
            # play, but the candidate tree must contain the candidate model's
            # projection over those same inputs even before the game is final.
            # Never leave a copied baseline-HCA frozen artifact beside a
            # candidate-HCA latest payload.
            if frozen_path.is_file():
                _write_json(frozen_path, candidate_pregame)
            if is_completed:
                if not frozen_path.is_file():
                    raise FileNotFoundError(f"Completed game has no frozen input: {game_id}")
                boxscore_path = output / "final" / f"{game_id}.nba.json"
                summary_path = output / "final" / f"{game_id}.summary.json"
                if not boxscore_path.is_file() or not summary_path.is_file():
                    raise FileNotFoundError(f"Completed game artifacts are missing: {game_id}")
                old_summary = _read_json(summary_path)
                projection_payload = _retrospective_rating_payload(state)
                retrospective = _build_official_actual_minutes_retrospective(
                    served,
                    pregame=candidate_pregame,
                    boxscore=_read_json(boxscore_path),
                    projection_payload=projection_payload,
                )
                archived_at = datetime.fromisoformat(
                    str(old_summary["archived_at"]).replace("Z", "+00:00")
                )
                final_summary = _build_final_game_archive(
                    served,
                    pregame=candidate_pregame,
                    retrospective=retrospective,
                    archived_at=archived_at,
                )
                _write_json(summary_path, final_summary)
                served["retrospective"] = copy.deepcopy(retrospective)
                completed_games += 1
                completed_game_ids.append(game_id)
            else:
                start = datetime.fromisoformat(
                    str(served["start_time"]).replace("Z", "+00:00")
                )
                if now < start:
                    _, created = _publish_pregame_candidate(
                        served,
                        state_root=output,
                        captured_at=now,
                    )
                    pending_candidates += int(created)
            games[index] = served
            rewritten_games += 1
            game_ids.append(game_id)
        _write_json(latest_path, payload)

    if source_sha_before != _tree_sha256(source):
        raise RuntimeError("Source Schedule changed during candidate materialization")
    expected_completed = states[-1].cumulative_game_count
    if completed_games != expected_completed:
        raise ValueError(
            "Candidate completed-game coverage is "
            f"{completed_games}, expected {expected_completed}"
        )
    candidate_manifest = copy.deepcopy(source_manifest)
    candidate_manifest.update(
        {
            "candidate_id": "causal_hca_replacement_candidate_v2",
            "model_track": "causal_hca_replacement_candidate_v2",
            "evaluation_cell": "causal_hca_replacement_candidate_v2",
            "through_date": evidence_through_date.isoformat(),
            "game_ids": sorted(completed_game_ids),
            "game_count": completed_games,
            "source_manifest_identity": source_manifest_identity,
            "source_schedule_sha256": source_sha_before,
            "source_model_tip_sha256": source_manifest.get("model_tip_sha256"),
            "model_tip_sha256": states[-1].bundle_sha256,
            "pregame_sha256_by_game": {
                game_id: candidate_pregame_hashes[game_id]
                for game_id in sorted(candidate_pregame_hashes)
            },
        }
    )
    projection_controls = copy.deepcopy(
        candidate_manifest.get("projection_controls") or {}
    )
    projection_controls.update(
        {
            "home_court": states[-1].home_court,
            "home_court_candidate_id": "causal_hca_replacement_candidate_v2",
            "home_court_contract": "strictly-prior daily posterior",
        }
    )
    candidate_manifest["projection_controls"] = projection_controls
    candidate_identity = {
        key: value
        for key, value in candidate_manifest.items()
        if key != "identity"
    }
    candidate_manifest["identity"] = hashlib.sha256(
        canonical_json(candidate_identity)
    ).hexdigest()
    _write_json(output / "manifest.json", candidate_manifest)
    report = {
        "status": "complete",
        "candidate_id": "causal_hca_replacement_candidate_v2",
        "start_date": start_date.isoformat(),
        "evidence_through_date": evidence_through_date.isoformat(),
        "slate_through_date": slate_through_date.isoformat(),
        "completed_game_count": completed_games,
        "served_game_count": rewritten_games,
        "pending_candidate_count": pending_candidates,
        "game_ids": sorted(game_ids),
        "candidate_model_tip_sha256": states[-1].bundle_sha256,
        "source_schedule_sha256": source_sha_before,
        "output_schedule_sha256": _tree_sha256(output),
        "probability_scale": probability_scale,
        "source_data_mutations": 0,
        "runtime_forbidden_provider_dependencies": 0,
    }
    _write_json(output.parent / f"{output.name}.materialization.json", report)
    return report


def _load_states(
    *, model_root: Path, preseason_rankings_path: Path
) -> tuple[
    list[PregameState],
    dict[int, float],
    float,
    str,
    str,
]:
    chain = _read_model_chain(model_root)
    prior_path = model_root / "prior_selection.json"
    prior = _read_json(prior_path)
    preseason = pd.read_parquet(
        preseason_rankings_path, columns=["player_id", "rapm"]
    )
    preseason_ratings = dict(
        zip(
            preseason["player_id"].astype(int),
            preseason["rapm"].astype(float),
            strict=True,
        )
    )
    states: list[PregameState] = []
    for state_date, bundle, manifest in chain:
        ratings = pd.read_parquet(
            _component_path(bundle, manifest, "player_ratings"),
            columns=["player_id", "updated_nail"],
        )
        hca_path = _component_path(bundle, manifest, "home_court_state")
        hca = _read_json(hca_path)
        states.append(
            PregameState(
                as_of=state_date,
                ratings=dict(
                    zip(
                        ratings["player_id"].astype(int),
                        ratings["updated_nail"].astype(float),
                        strict=True,
                    )
                ),
                home_court=float(hca["home_court_estimate"]),
                bundle_sha256=str(manifest["bundle_sha256"]),
                rating_source_uri=(
                    f"artifact://daily-model/{manifest['bundle_sha256']}"
                    "/player_ratings"
                ),
                home_court_source_uri=(
                    f"artifact://daily-model/{manifest['bundle_sha256']}"
                    "/home_court_state"
                ),
                home_court_sha256=file_sha256(hca_path),
                cumulative_game_count=int(
                    manifest["identity"]["cumulative_game_count"]
                ),
            )
        )
    return (
        states,
        preseason_ratings,
        float(prior["prior_mean"]),
        file_sha256(prior_path),
        file_sha256(preseason_rankings_path),
    )


def _pregame_state(
    *,
    game_date: date,
    states: list[PregameState],
    preseason_ratings: dict[int, float],
    prior_hca: float,
    prior_sha: str,
    preseason_sha: str,
) -> PregameState:
    eligible = [state for state in states if state.as_of < game_date]
    if eligible:
        return eligible[-1]
    return PregameState(
        as_of=game_date - timedelta(days=1),
        ratings=preseason_ratings,
        home_court=prior_hca,
        bundle_sha256=prior_sha,
        rating_source_uri=f"artifact://preseason-rankings/{preseason_sha}",
        home_court_source_uri=f"artifact://causal-hca-prior/{prior_sha}",
        home_court_sha256=prior_sha,
        cumulative_game_count=0,
    )


def _candidate_pregame(
    game: dict[str, Any], *, state: PregameState, probability_scale: float
) -> dict[str, Any]:
    game_id = str(game["game_id"])
    projection = game.get("projection")
    if not isinstance(projection, dict):
        raise ValueError(f"Game has no projection: {game_id}")
    old_ledger = projection.get("ledger")
    back_to_back = old_ledger.get("back_to_back") if isinstance(old_ledger, dict) else None
    if not isinstance(back_to_back, dict):
        raise ValueError(f"Game has no frozen B2B context: {game_id}")
    projected = complete_projected_minutes_attribution(
        game["projected_minutes"],
        injury_report=game.get("injury_report"),
        away_team=str(game["away_team"]),
        home_team=str(game["home_team"]),
        ratings=state.ratings,
        game_id=game_id,
    )
    provenance = copy.deepcopy(projection.get("input_provenance") or {})
    provenance.update(
        {
            "candidate_id": "causal_hca_replacement_candidate_v2",
            "rating_as_of": state.as_of.isoformat(),
            "rating_source_uri": state.rating_source_uri,
            "rating_state_sha256": state.bundle_sha256,
            "home_court_source_uri": state.home_court_source_uri,
            "home_court_state_sha256": state.home_court_sha256,
            "source_projection_sha256": hashlib.sha256(
                canonical_json(projection)
            ).hexdigest(),
        }
    )
    ledger = build_spread_ledger(
        game_id=game_id,
        away_team=str(game["away_team"]),
        home_team=str(game["home_team"]),
        projected_minutes=projected,
        home_court_adjustment=state.home_court,
        back_to_back_coefficient=float(back_to_back["coefficient"]),
        home_back_to_back=bool(back_to_back["home_team_is_back_to_back"]),
        away_back_to_back=bool(back_to_back["away_team_is_back_to_back"]),
        provenance=provenance,
    )
    margin = float(ledger["home_margin"])
    candidate = copy.deepcopy(game)
    candidate["projected_minutes"] = projected
    candidate["projection"] = {
        **projection,
        "model": "Causal HCA replacement candidate v2",
        "model_track": "causal_hca_replacement_candidate_v2",
        "state_kind": "causal_reconstruction",
        "rating_as_of": state.as_of.isoformat(),
        "serving_bundle_sha256": state.bundle_sha256,
        "input_provenance": provenance,
        "home_margin": margin,
        "fair_home_spread": -margin,
        "home_win_probability": 1.0 / (1.0 + math.exp(-probability_scale * margin)),
        "ledger": ledger,
    }
    require_projection_reconciliation(candidate["projection"], game_id=game_id)
    return candidate


def _retrospective_rating_payload(state: PregameState) -> dict[str, object]:
    """Expose the complete causal rating-before state for actual participants."""

    return {
        "players": [
            {"player_id": player_id, "projected_nail": rating}
            for player_id, rating in sorted(state.ratings.items())
        ]
    }


def _probability_scale(games: list[dict[str, Any]]) -> float:
    values = []
    for game in games:
        projection = game.get("projection")
        if not isinstance(projection, dict):
            continue
        margin = float(projection["home_margin"])
        probability = float(projection["home_win_probability"])
        if abs(margin) > 1e-8 and 0.0 < probability < 1.0:
            values.append(math.log(probability / (1.0 - probability)) / margin)
    if not values:
        raise ValueError("Frozen games do not identify a probability scale")
    return float(pd.Series(values).median())


def _game_date(game: dict[str, Any]) -> date:
    start = datetime.fromisoformat(str(game["start_time"]).replace("Z", "+00:00"))
    from nba_lineup_model.season.slate_time import PACIFIC

    return start.astimezone(PACIFIC).date()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-schedule-root", type=Path, required=True)
    parser.add_argument("--output-schedule-root", type=Path, required=True)
    parser.add_argument("--candidate-model-root", type=Path, required=True)
    parser.add_argument("--preseason-rankings-path", type=Path, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--evidence-through-date", type=date.fromisoformat, required=True)
    parser.add_argument("--slate-through-date", type=date.fromisoformat, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = materialize_causal_hca_dev_schedule(**vars(args))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
