"""Build auditable possession-start states from canonical event boundaries."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nba_lineup_model.modeling.schema import ArtifactRecord
from nba_lineup_model.modeling.stints import modeling_code_fingerprint
from nba_lineup_model.season.compact import (
    read_curated_partition_manifest,
    validate_curated_partition,
)
from nba_lineup_model.season.layout import CuratedDatasetLayout, CuratedPartition
from nba_lineup_model.season.schema import validate_season


DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")

START_STATES = (
    "period_start",
    "after_made_2pt",
    "after_made_3pt",
    "after_final_free_throw",
    "after_live_ball_turnover_steal",
    "after_other_live_ball_turnover",
    "after_dead_ball_turnover",
    "after_defensive_rebound",
    "after_held_ball",
    "audited_fallback",
)
COARSE_START_STATES = ("live_ball", "dead_ball", "unknown")
LIVE_BALL_STATES = frozenset(
    {
        "after_live_ball_turnover_steal",
        "after_other_live_ball_turnover",
        "after_defensive_rebound",
    }
)
DEAD_BALL_STATES = frozenset(set(START_STATES) - set(LIVE_BALL_STATES) - {"audited_fallback"})

EVENT_COLUMNS = (
    "game_id",
    "event_id",
    "event_index",
    "event_type",
    "event_subtype",
    "descriptor",
    "team_id",
    "player_id",
    "source_order_number",
    "elapsed_game_seconds",
    "shot_result",
    "description",
)
POSSESSION_COLUMNS = (
    "season",
    "season_type",
    "game_id",
    "possession_id",
    "possession_index",
    "period_possession_index",
    "period",
    "offense_team_id",
    "defense_team_id",
    "start_event_id",
    "end_event_id",
    "start_event_index",
    "end_event_index",
    "terminal_reason",
)
MART_COLUMNS = (
    "season",
    "season_type",
    "game_id",
    "possession_id",
    "possession_index",
    "period_possession_index",
    "period",
    "offense_team_id",
    "defense_team_id",
    "start_event_id",
    "start_event_index",
    "start_state",
    "coarse_start_state",
    "is_live_ball",
    "is_dead_ball",
    "is_period_start",
    "state_classification_method",
    "state_classification_detail",
    "prior_possession_id",
    "prior_possession_index",
    "prior_offense_team_id",
    "prior_defense_team_id",
    "prior_terminal_reason",
    "prior_end_event_id",
    "prior_end_event_index",
    "prior_terminal_event_type",
    "prior_terminal_event_subtype",
    "prior_terminal_event_descriptor",
    "prior_terminal_event_shot_result",
    "prior_terminal_event_team_id",
    "prior_terminal_event_player_id",
    "state_trigger_event_id",
    "state_trigger_event_index",
    "state_trigger_event_type",
    "state_trigger_event_subtype",
    "state_trigger_team_id",
    "state_trigger_player_id",
    "turnover_has_credited_steal",
    "intervening_event_count",
    "has_intervening_timeout_or_review",
    "has_intervening_substitution",
    "is_audited_fallback",
    "fallback_reason",
)
_DEAD_BALL_TURNOVER_TOKENS = (
    "out of bounds",
    "offensive foul",
    "shot clock",
    "travel",
    "backcourt",
    "goaltending",
    "violation",
    "double dribble",
    "palming",
    "discontinued dribble",
    "inbound",
    "lane",
    "illegal assist",
    "excess timeout",
    "too many players",
    "kicked ball",
    "back to the basket",
    "freethrow shooter",
)
_LIVE_BALL_TURNOVER_TOKENS = ("bad pass", "lost ball", "punched ball")


@dataclass(frozen=True)
class _StateClassification:
    state: str
    coarse_state: str
    method: str
    detail: str
    trigger: Mapping[str, object] | None
    turnover_has_credited_steal: bool
    fallback_reason: str | None = None


def build_possession_start_state_frame(
    possessions: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Return one state row per possession without target-possession outcomes.

    The state is determined from the previous possession's terminal event and
    events between that boundary and the current possession's first event. A
    credited steal is permitted when it is the current possession's opening
    event because it is simultaneous with the live-ball change of control.
    """

    _validate_source_frames(possessions, events)
    source = possessions.loc[:, POSSESSION_COLUMNS].copy()
    source["game_id"] = source["game_id"].astype(str)
    source["possession_id"] = source["possession_id"].astype(str)
    source["possession_index"] = pd.to_numeric(source["possession_index"], errors="raise").astype(int)
    source["period"] = pd.to_numeric(source["period"], errors="raise").astype(int)
    source["start_event_index"] = pd.to_numeric(
        source["start_event_index"], errors="raise"
    ).astype(int)
    source["end_event_index"] = pd.to_numeric(source["end_event_index"], errors="raise").astype(int)
    if source.duplicated(["game_id", "possession_id"]).any():
        raise ValueError("Possession start-state mart requires unique game/possession keys")

    event_frame = events.loc[:, EVENT_COLUMNS].copy()
    event_frame["game_id"] = event_frame["game_id"].astype(str)
    event_frame["event_index"] = pd.to_numeric(event_frame["event_index"], errors="raise").astype(int)
    if event_frame.duplicated(["game_id", "event_index"]).any():
        raise ValueError("Possession start-state mart requires unique game/event indexes")

    rows: list[dict[str, object]] = []
    events_by_game = {
        game_id: frame.sort_values("event_index", kind="stable").reset_index(drop=True)
        for game_id, frame in event_frame.groupby("game_id", sort=False)
    }
    for game_id, game_possessions in source.groupby("game_id", sort=False):
        game_events = events_by_game.get(game_id)
        if game_events is None:
            raise ValueError(f"Possession start-state mart is missing events for game {game_id}")
        ordered = game_possessions.sort_values("possession_index", kind="stable").reset_index(drop=True)
        _validate_game_possessions(ordered)
        event_records = game_events.to_dict("records")
        event_indexes = [int(event["event_index"]) for event in event_records]
        event_by_index = {int(event["event_index"]): event for event in event_records}
        events_by_elapsed: dict[float, list[dict[str, object]]] = defaultdict(list)
        for event in event_records:
            events_by_elapsed[float(event["elapsed_game_seconds"])].append(event)
        possession_records = ordered.to_dict("records")
        for position, current in enumerate(possession_records):
            previous = possession_records[position - 1] if position else None
            is_period_start = previous is None or int(current["period"]) != int(previous["period"])
            current_start_index = int(current["start_event_index"])
            if previous is None:
                intervening: Sequence[Mapping[str, object]] = ()
                handoff: Sequence[Mapping[str, object]] = ()
                terminal_event = None
            else:
                prior_end_index = int(previous["end_event_index"])
                lower = bisect.bisect_right(event_indexes, prior_end_index)
                upper = bisect.bisect_left(event_indexes, current_start_index)
                intervening = event_records[lower:upper]
                terminal_event = event_by_index.get(prior_end_index)
                if terminal_event is None:
                    handoff = ()
                else:
                    terminal_elapsed = float(terminal_event["elapsed_game_seconds"])
                    handoff = tuple(
                        event
                        for event in events_by_elapsed[terminal_elapsed]
                        if int(event["event_index"]) >= prior_end_index
                    )
            classification = _classify_start_state(
                current=current,
                previous=previous,
                terminal_event=terminal_event,
                handoff_events=handoff,
                is_period_start=is_period_start,
            )
            rows.append(
                _mart_row(
                    current=current,
                    previous=previous,
                    terminal_event=terminal_event,
                    classification=classification,
                    is_period_start=is_period_start,
                    intervening=intervening,
                )
            )
    output = pd.DataFrame(rows, columns=MART_COLUMNS)
    _validate_start_state_frame(output)
    return output.sort_values(["season", "game_id", "possession_index"], kind="stable").reset_index(
        drop=True
    )


def build_possession_start_state_dataset(
    season: str,
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
) -> Path:
    """Materialize one audited regular-season possession-start-state partition."""

    season = validate_season(season)
    curated_root = Path(curated_dir)
    layout = CuratedDatasetLayout(curated_root)
    partition_args = {"season": season, "season_type": "regular"}
    event_partition = CuratedPartition(table="events", **partition_args)
    possession_partition = CuratedPartition(table="possessions", **partition_args)
    event_manifest = read_curated_partition_manifest(event_partition, curated_root)
    possession_manifest = read_curated_partition_manifest(possession_partition, curated_root)
    validate_curated_partition(event_manifest, curated_root)
    validate_curated_partition(possession_manifest, curated_root)
    events = _read_partition(layout.partition_dir(event_partition), columns=list(EVENT_COLUMNS))
    possessions = _read_partition(
        layout.partition_dir(possession_partition), columns=list(POSSESSION_COLUMNS)
    )
    frame = build_possession_start_state_frame(possessions, events)
    coverage = _state_coverage(frame)
    game_audit = _game_audit(frame, possessions)
    fallback_audit = _fallback_audit(frame)

    target = Path(analytical_dir) / "possession_start_states" / season / "regular"
    return _write_dataset(
        target=target,
        frame=frame,
        coverage=coverage,
        game_audit=game_audit,
        fallback_audit=fallback_audit,
        source_event_manifest_path=layout.partition_dir(event_partition) / "_manifest.json",
        source_possession_manifest_path=layout.partition_dir(possession_partition) / "_manifest.json",
    )


def build_all_possession_start_state_datasets(
    *,
    seasons: tuple[str, ...] | None = None,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
) -> tuple[Path, ...]:
    """Build every requested regular-season start-state partition locally."""

    curated_root = Path(curated_dir)
    selected = seasons or _available_seasons(curated_root)
    if not selected:
        raise ValueError("No curated regular-season possession partitions are available")
    outputs: list[Path] = []
    for index, season in enumerate(selected, start=1):
        print(f"Possession start-state mart: [{index}/{len(selected)}] {season}", flush=True)
        outputs.append(
            build_possession_start_state_dataset(
                season,
                curated_dir=curated_root,
                analytical_dir=analytical_dir,
            )
        )
    return tuple(outputs)


def validate_possession_start_state_dataset(path: Path | str) -> dict[str, object]:
    """Validate mart schema, internal accounting, and output checksums."""

    root = Path(path)
    manifest = json.loads((root / "_manifest.json").read_text())
    frame = pd.read_parquet(root / "part-00000.parquet")
    coverage = pd.read_parquet(root / "state_coverage.parquet")
    game_audit = pd.read_parquet(root / "game_audit.parquet")
    fallback_audit = pd.read_parquet(root / "fallback_audit.parquet")
    _validate_start_state_frame(frame)
    if len(frame) != int(manifest["row_count"]):
        raise ValueError("Start-state mart row count does not match manifest")
    if frame["game_id"].nunique() != int(manifest["game_count"]):
        raise ValueError("Start-state mart game count does not match manifest")
    if tuple(manifest["start_states"]) != START_STATES:
        raise ValueError("Start-state mart taxonomy does not match the code contract")
    if set(coverage["start_state"]) != set(START_STATES):
        raise ValueError("Start-state coverage must include every taxonomy state")
    if int(coverage["possession_count"].sum()) != len(frame):
        raise ValueError("Start-state coverage does not conserve possessions")
    if len(game_audit) != frame["game_id"].nunique():
        raise ValueError("Start-state game audit must contain one row per game")
    if not game_audit["source_possession_count"].eq(game_audit["state_row_count"]).all():
        raise ValueError("Start-state game audit lost or duplicated possessions")
    if len(fallback_audit) != int(frame["is_audited_fallback"].sum()):
        raise ValueError("Start-state fallback audit does not match mart fallback rows")
    for artifact in manifest["artifacts"]:
        artifact_path = root / artifact["filename"]
        if not artifact_path.is_file() or _sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError(f"Start-state mart checksum mismatch: {artifact['filename']}")
    return manifest


def _classify_start_state(
    *,
    current: Mapping[str, object],
    previous: Mapping[str, object] | None,
    terminal_event: Mapping[str, object] | None,
    handoff_events: Sequence[Mapping[str, object]],
    is_period_start: bool,
) -> _StateClassification:
    if is_period_start:
        return _StateClassification(
            state="period_start",
            coarse_state="dead_ball",
            method="period_boundary",
            detail="opening_possession_of_period",
            trigger=None,
            turnover_has_credited_steal=False,
        )
    if previous is None or terminal_event is None:
        return _fallback("missing_prior_terminal_event")

    terminal_reason = str(previous["terminal_reason"])
    event_type = _text(terminal_event, "event_type")
    shot_result = _text(terminal_event, "shot_result")
    if terminal_reason == "made_field_goal":
        if event_type == "2pt" and shot_result == "made":
            return _dead("after_made_2pt", "terminal_event", "made_2pt", terminal_event)
        if event_type == "3pt" and shot_result == "made":
            return _dead("after_made_3pt", "terminal_event", "made_3pt", terminal_event)
        return _fallback("made_field_goal_terminal_not_2pt_or_3pt")
    if terminal_reason == "made_final_free_throw":
        if event_type == "freethrow" and shot_result == "made":
            return _dead(
                "after_final_free_throw",
                "terminal_event",
                "made_final_nontechnical_free_throw",
                terminal_event,
            )
        return _fallback("made_final_free_throw_terminal_not_matching_event")
    if terminal_reason == "defensive_rebound":
        if event_type == "rebound" and _text(terminal_event, "event_subtype") == "defensive":
            if _same_team(terminal_event, current, "team_id", "offense_team_id"):
                return _live(
                    "after_defensive_rebound",
                    "terminal_event",
                    "defensive_rebound",
                    terminal_event,
                )
            return _fallback("defensive_rebound_team_does_not_match_next_offense")
        return _fallback("defensive_rebound_terminal_not_matching_event")
    if terminal_reason == "turnover":
        return _classify_turnover(current, terminal_event, handoff_events)
    if terminal_reason == "held_ball":
        return _dead("after_held_ball", "terminal_event", "held_ball_recovery", terminal_event)
    if terminal_reason == "period_end":
        return _fallback("period_end_without_period_boundary")
    if terminal_reason in {"source_change", "feed_end"}:
        return _fallback(f"unresolved_terminal_reason:{terminal_reason}")
    return _fallback(f"unsupported_terminal_reason:{terminal_reason}")


def _classify_turnover(
    current: Mapping[str, object],
    terminal_event: Mapping[str, object],
    handoff_events: Sequence[Mapping[str, object]],
) -> _StateClassification:
    offense_team_id = int(current["offense_team_id"])
    steals = []
    for event in handoff_events:
        event_team_id = _integer_or_na(event.get("team_id"))
        if (
            _text(event, "event_type") == "steal"
            and not pd.isna(event_team_id)
            and event_team_id == offense_team_id
        ):
            steals.append(event)
    if steals:
        trigger = min(steals, key=lambda event: int(event["event_index"]))
        return _live(
            "after_live_ball_turnover_steal",
            "handoff_event",
            "credited_steal",
            trigger,
            turnover_has_credited_steal=True,
        )
    detail = _turnover_detail(terminal_event)
    if any(token in detail for token in _DEAD_BALL_TURNOVER_TOKENS):
        return _dead("after_dead_ball_turnover", "turnover_subtype", detail, terminal_event)
    if any(token in detail for token in _LIVE_BALL_TURNOVER_TOKENS):
        return _live(
            "after_other_live_ball_turnover",
            "turnover_subtype",
            detail,
            terminal_event,
        )
    return _fallback("unclassified_turnover_without_credited_steal")


def _mart_row(
    *,
    current: Mapping[str, object],
    previous: Mapping[str, object] | None,
    terminal_event: Mapping[str, object] | None,
    classification: _StateClassification,
    is_period_start: bool,
    intervening: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    trigger = classification.trigger
    return {
        "season": _value(current, "season"),
        "season_type": _value(current, "season_type"),
        "game_id": str(current["game_id"]),
        "possession_id": str(current["possession_id"]),
        "possession_index": int(current["possession_index"]),
        "period_possession_index": _integer_or_na(current.get("period_possession_index")),
        "period": int(current["period"]),
        "offense_team_id": _integer_or_na(current["offense_team_id"]),
        "defense_team_id": _integer_or_na(current["defense_team_id"]),
        "start_event_id": _value(current, "start_event_id"),
        "start_event_index": int(current["start_event_index"]),
        "start_state": classification.state,
        "coarse_start_state": classification.coarse_state,
        "is_live_ball": classification.coarse_state == "live_ball",
        "is_dead_ball": classification.coarse_state == "dead_ball",
        "is_period_start": is_period_start,
        "state_classification_method": classification.method,
        "state_classification_detail": classification.detail,
        "prior_possession_id": _value(previous, "possession_id"),
        "prior_possession_index": _integer_or_na(
            previous.get("possession_index") if previous is not None else None
        ),
        "prior_offense_team_id": _integer_or_na(
            previous.get("offense_team_id") if previous is not None else None
        ),
        "prior_defense_team_id": _integer_or_na(
            previous.get("defense_team_id") if previous is not None else None
        ),
        "prior_terminal_reason": _value(previous, "terminal_reason"),
        "prior_end_event_id": _value(previous, "end_event_id"),
        "prior_end_event_index": _integer_or_na(
            previous.get("end_event_index") if previous is not None else None
        ),
        "prior_terminal_event_type": _value(terminal_event, "event_type"),
        "prior_terminal_event_subtype": _value(terminal_event, "event_subtype"),
        "prior_terminal_event_descriptor": _value(terminal_event, "descriptor"),
        "prior_terminal_event_shot_result": _value(terminal_event, "shot_result"),
        "prior_terminal_event_team_id": _integer_or_na(
            terminal_event.get("team_id") if terminal_event is not None else None
        ),
        "prior_terminal_event_player_id": _integer_or_na(
            terminal_event.get("player_id") if terminal_event is not None else None
        ),
        "state_trigger_event_id": _value(trigger, "event_id"),
        "state_trigger_event_index": _integer_or_na(
            trigger.get("event_index") if trigger is not None else None
        ),
        "state_trigger_event_type": _value(trigger, "event_type"),
        "state_trigger_event_subtype": _value(trigger, "event_subtype"),
        "state_trigger_team_id": _integer_or_na(
            trigger.get("team_id") if trigger is not None else None
        ),
        "state_trigger_player_id": _integer_or_na(
            trigger.get("player_id") if trigger is not None else None
        ),
        "turnover_has_credited_steal": classification.turnover_has_credited_steal,
        "intervening_event_count": int(len(intervening)),
        "has_intervening_timeout_or_review": _has_timeout_or_review(intervening),
        "has_intervening_substitution": _has_event_type(intervening, "substitution"),
        "is_audited_fallback": classification.state == "audited_fallback",
        "fallback_reason": classification.fallback_reason,
    }


def _state_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    observed = frame.groupby("start_state", sort=False).size().reindex(START_STATES, fill_value=0)
    rows = []
    for state, count in observed.items():
        coarse = _coarse_for_state(state)
        rows.append(
            {
                "start_state": state,
                "coarse_start_state": coarse,
                "possession_count": int(count),
                "possession_share": float(count / len(frame)),
                "is_live_ball": coarse == "live_ball",
                "is_dead_ball": coarse == "dead_ball",
            }
        )
    return pd.DataFrame(rows)


def _game_audit(frame: pd.DataFrame, possessions: pd.DataFrame) -> pd.DataFrame:
    source_counts = possessions.assign(game_id=possessions["game_id"].astype(str)).groupby("game_id").size()
    audit = (
        frame.groupby("game_id", as_index=False, sort=False)
        .agg(
            state_row_count=("possession_id", "size"),
            period_start_count=("is_period_start", "sum"),
            fallback_count=("is_audited_fallback", "sum"),
            live_ball_count=("is_live_ball", "sum"),
            dead_ball_count=("is_dead_ball", "sum"),
        )
        .sort_values("game_id", kind="stable")
        .reset_index(drop=True)
    )
    audit["source_possession_count"] = audit["game_id"].map(source_counts).astype(int)
    return audit.loc[
        :,
        [
            "game_id",
            "source_possession_count",
            "state_row_count",
            "period_start_count",
            "fallback_count",
            "live_ball_count",
            "dead_ball_count",
        ],
    ]


def _fallback_audit(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "season",
        "game_id",
        "possession_id",
        "possession_index",
        "period",
        "offense_team_id",
        "defense_team_id",
        "prior_possession_id",
        "prior_terminal_reason",
        "prior_end_event_id",
        "prior_terminal_event_type",
        "prior_terminal_event_subtype",
        "prior_terminal_event_descriptor",
        "state_classification_method",
        "state_classification_detail",
        "fallback_reason",
    ]
    return frame.loc[frame["is_audited_fallback"], columns].reset_index(drop=True)


def _write_dataset(
    *,
    target: Path,
    frame: pd.DataFrame,
    coverage: pd.DataFrame,
    game_audit: pd.DataFrame,
    fallback_audit: pd.DataFrame,
    source_event_manifest_path: Path,
    source_possession_manifest_path: Path,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid4().hex}"
    backup = target.parent / f".{target.name}.bak-{uuid4().hex}"
    temporary.mkdir()
    try:
        outputs = {
            "part-00000.parquet": frame,
            "state_coverage.parquet": coverage,
            "game_audit.parquet": game_audit,
            "fallback_audit.parquet": fallback_audit,
        }
        for filename, output in outputs.items():
            output.to_parquet(temporary / filename, index=False)
        artifacts = [
            ArtifactRecord(
                filename=filename,
                row_count=len(output),
                byte_count=(temporary / filename).stat().st_size,
                sha256=_sha256_file(temporary / filename),
            ).model_dump()
            for filename, output in outputs.items()
        ]
        manifest = {
            "schema_version": 1,
            "season": str(frame["season"].iloc[0]),
            "season_type": "regular",
            "created_at": datetime.now(UTC).isoformat(),
            "builder_code_version": modeling_code_fingerprint((Path(__file__),)),
            "state_derivation": "prior_terminal_event_plus_handoff_events",
            "start_states": list(START_STATES),
            "coarse_start_states": list(COARSE_START_STATES),
            "source_event_manifest_sha256": _sha256_file(source_event_manifest_path),
            "source_possession_manifest_sha256": _sha256_file(source_possession_manifest_path),
            "row_count": len(frame),
            "game_count": int(frame["game_id"].nunique()),
            "fallback_count": int(frame["is_audited_fallback"].sum()),
            "artifacts": artifacts,
        }
        (temporary / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        validate_possession_start_state_dataset(temporary)
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


def _validate_source_frames(possessions: pd.DataFrame, events: pd.DataFrame) -> None:
    missing_possessions = set(POSSESSION_COLUMNS) - set(possessions)
    missing_events = set(EVENT_COLUMNS) - set(events)
    if missing_possessions:
        raise ValueError(f"Start-state mart possession frame missing columns: {sorted(missing_possessions)}")
    if missing_events:
        raise ValueError(f"Start-state mart event frame missing columns: {sorted(missing_events)}")
    if possessions.empty or events.empty:
        raise ValueError("Start-state mart requires non-empty possession and event frames")


def _validate_game_possessions(possessions: pd.DataFrame) -> None:
    indexes = possessions["possession_index"].to_numpy(dtype=int)
    if len(indexes) != len(set(indexes)):
        raise ValueError("Possession indexes must be unique within a game")
    if (indexes[1:] <= indexes[:-1]).any():
        raise ValueError("Possession indexes must be strictly increasing within a game")


def _validate_start_state_frame(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != MART_COLUMNS:
        raise ValueError("Start-state mart column contract changed")
    if frame.empty or frame.duplicated(["game_id", "possession_id"]).any():
        raise ValueError("Start-state mart must have unique non-empty possession rows")
    if not frame["start_state"].isin(START_STATES).all():
        raise ValueError("Start-state mart has an unsupported granular state")
    if not frame["coarse_start_state"].isin(COARSE_START_STATES).all():
        raise ValueError("Start-state mart has an unsupported coarse state")
    expected_coarse = frame["start_state"].map(_coarse_for_state)
    if not frame["coarse_start_state"].eq(expected_coarse).all():
        raise ValueError("Start-state mart coarse states do not match granular states")
    if not frame["is_live_ball"].eq(frame["coarse_start_state"].eq("live_ball")).all():
        raise ValueError("Start-state mart live-ball indicator is inconsistent")
    if not frame["is_dead_ball"].eq(frame["coarse_start_state"].eq("dead_ball")).all():
        raise ValueError("Start-state mart dead-ball indicator is inconsistent")
    if not frame.loc[frame["is_period_start"], "start_state"].eq("period_start").all():
        raise ValueError("Every period-opening possession must have the period_start label")
    if frame.loc[~frame["is_period_start"], "prior_possession_id"].isna().any():
        raise ValueError("Non-period-start rows require a prior possession key")
    fallback = frame["start_state"].eq("audited_fallback")
    if not frame["is_audited_fallback"].eq(fallback).all():
        raise ValueError("Fallback indicator does not match start-state label")
    if frame.loc[fallback, "fallback_reason"].isna().any():
        raise ValueError("Fallback rows require an audit reason")
    target_outcome_columns = {"offense_points", "points_home", "points_away", "duration_seconds"}
    if target_outcome_columns & set(frame):
        raise ValueError("Start-state mart must not persist target-possession outcome fields")


def _available_seasons(curated_root: Path) -> tuple[str, ...]:
    possession_root = curated_root / "possessions"
    seasons = [
        directory.name
        for directory in possession_root.iterdir()
        if directory.is_dir()
        and (directory / "regular" / "_manifest.json").is_file()
        and (curated_root / "events" / directory.name / "regular" / "_manifest.json").is_file()
    ]
    return tuple(sorted((validate_season(season) for season in seasons), key=lambda season: int(season[:4])))


def _read_partition(root: Path, *, columns: list[str]) -> pd.DataFrame:
    files = sorted(root.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Start-state mart has no curated parquet files in {root}")
    return pd.concat([pd.read_parquet(file, columns=columns) for file in files], ignore_index=True)


def _coarse_for_state(state: str) -> str:
    if state in LIVE_BALL_STATES:
        return "live_ball"
    if state in DEAD_BALL_STATES:
        return "dead_ball"
    if state == "audited_fallback":
        return "unknown"
    raise ValueError(f"Unsupported start state: {state}")


def _dead(
    state: str,
    method: str,
    detail: str,
    trigger: Mapping[str, object],
) -> _StateClassification:
    return _StateClassification(state, "dead_ball", method, detail, trigger, False)


def _live(
    state: str,
    method: str,
    detail: str,
    trigger: Mapping[str, object],
    *,
    turnover_has_credited_steal: bool = False,
) -> _StateClassification:
    return _StateClassification(
        state,
        "live_ball",
        method,
        detail,
        trigger,
        turnover_has_credited_steal,
    )


def _fallback(reason: str) -> _StateClassification:
    return _StateClassification(
        "audited_fallback",
        "unknown",
        "fallback",
        reason,
        None,
        False,
        fallback_reason=reason,
    )


def _turnover_detail(event: Mapping[str, object]) -> str:
    detail = " ".join(
        value
        for value in (
            _text(event, "event_subtype"),
            _text(event, "descriptor"),
            _text(event, "description"),
        )
        if value
    )
    return detail.replace("-", " ")


def _same_team(
    event: Mapping[str, object],
    row: Mapping[str, object],
    event_column: str,
    row_column: str,
) -> bool:
    event_team = _integer_or_na(event.get(event_column))
    row_team = _integer_or_na(row.get(row_column))
    return not pd.isna(event_team) and event_team == row_team


def _has_timeout_or_review(events: Sequence[Mapping[str, object]]) -> bool:
    return any(
        _text(event, "event_type") == "timeout" or "review" in _text(event, "description")
        for event in events
    )


def _has_event_type(events: Sequence[Mapping[str, object]], event_type: str) -> bool:
    return any(_text(event, "event_type") == event_type for event in events)


def _text(event: Mapping[str, object], column: str) -> str:
    value = _value(event, column)
    return "" if value is None else str(value).casefold()


def _value(event: Mapping[str, object] | None, column: str) -> object:
    if event is None or column not in event:
        return None
    value = event[column]
    return None if pd.isna(value) else value


def _integer_or_na(value: object) -> int | object:
    if value is None or pd.isna(value):
        return pd.NA
    return int(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the possession start-state mart")
    parser.add_argument("--season", action="append", dest="seasons")
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--analytical-dir", type=Path, default=DEFAULT_ANALYTICAL_DIR)
    args = parser.parse_args()
    seasons = tuple(validate_season(value) for value in args.seasons) if args.seasons else None
    outputs = build_all_possession_start_state_datasets(
        seasons=seasons,
        curated_dir=args.curated_dir,
        analytical_dir=args.analytical_dir,
    )
    print(f"Possession start-state mart: built {len(outputs)} season partitions")


if __name__ == "__main__":
    main()
