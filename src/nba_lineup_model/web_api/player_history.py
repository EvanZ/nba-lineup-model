"""Consume selected player_game_history without reconstructing game evidence."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import date, datetime
from functools import lru_cache
from time import time

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, HTTPException, Path, Response

from .schedule import BUCKET, PACIFIC, ScheduleReader, _number

router = APIRouter()
SEASON = "2026-27"
MODEL = "v1"
ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, BotoCoreError, ClientError)
STAT_FIELDS = (
    "points",
    "rebounds",
    "offensive_rebounds",
    "defensive_rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "personal_fouls",
    "field_goals_made",
    "field_goals_attempted",
    "field_goal_percentage",
    "three_pointers_made",
    "three_pointers_attempted",
    "three_point_percentage",
    "free_throws_made",
    "free_throws_attempted",
    "free_throw_percentage",
    "plus_minus",
)
STATUSES = {"played", "inactive", "did_not_play", "not_listed", "unknown"}
_lock = threading.Lock()


def reference(ref, root, *, suffix=".json"):
    sha = ref["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError("Invalid history input checksum")
    key = root + sha + suffix
    if ref["uri"] != f"s3://{BUCKET}/{key}":
        raise ValueError("History reference escapes its player/game/date/model scope")
    return key, sha


def ratings_reference(ref, day):
    reference(
        ref,
        f"lake/artifacts/player_ratings/season={SEASON}/model_version={MODEL}/game_date_pacific={day}/",
    )


def team_id(value, *, nullable=False):
    if value is None and nullable:
        return
    if type(value) is not int or value <= 0:
        raise ValueError("Invalid saved history team ID")


def scalar(record):
    _number(record["nail"])
    if _number(record["cumulative_possessions"]) < 0:
        raise ValueError("Negative saved cumulative possessions")
    team_id(record["team_id"], nullable=True)


class HistoryReader(ScheduleReader):
    @staticmethod
    def identity(doc, player_id, kind):
        if (
            type(doc["schema_version"]) is not int
            or doc["schema_version"] != 1
            or doc["artifact_kind"] != kind
            or doc["season"] != SEASON
            or doc["model_version"] != MODEL
            or type(doc["player_id"]) is not int
            or doc["player_id"] != player_id
            or any(
                not isinstance(doc[k], str) or not doc[k]
                for k in ("model_id", "model_run_id", "player_name")
            )
        ):
            raise ValueError("History player/season/model/schema identity mismatch")

    def history(self, player_id):
        root = (
            f"lake/artifacts/player_game_history/season={SEASON}/model_version={MODEL}/"
            f"player_id={player_id}/"
        )
        pair = self._get(root + "latest.json", missing=True)
        if pair is None:
            raise HTTPException(
                status_code=404,
                detail="No saved in-season game history is published for this player.",
                headers={"Cache-Control": "no-store"},
            )
        raw, _ = pair
        index = json.loads(raw)
        self.identity(index, player_id, "player_game_history_index")
        start, current, refs = index["starting_point"], index["current"], index["points"]
        if start["kind"] not in {"opening_prior", "introduced_prior"} or not isinstance(refs, list):
            raise ValueError("Invalid history prior/point list")
        _number(start["nail"])
        first, initialized, through = (
            date.fromisoformat(start["first_state_date"]),
            date.fromisoformat(start["initialized_on"]),
            date.fromisoformat(index["through_game_date_pacific"]),
        )
        if initialized > first or first > through or through > datetime.now(PACIFIC).date():
            raise ValueError("Invalid history initialization/through date")
        days = [ref["game_date_pacific"] for ref in refs]
        if days != sorted(set(days)) or any(
            str(date.fromisoformat(d)) != d or not first <= date.fromisoformat(d) <= through
            for d in days
        ):
            raise ValueError("History game dates are duplicated, unordered or outside coverage")
        if index["last_game"] != (refs[-1] if refs else None):
            raise ValueError("History last-game reference differs from accepted point list")
        if "uri" in current or current["game_date_pacific"] != str(through):
            raise ValueError(
                "History current must be the through-date scalar, not a point reference"
            )
        scalar(current)
        if current["source_ratings"] != index["source_ratings"]:
            raise ValueError("Current history scalar refers to another accepted ratings head")
        ratings_reference(current["source_ratings"], through)
        ratings_reference(start["source_ratings"], first)
        reference(
            index["initialization_input"],
            f"lake/artifacts/player_ratings/season={SEASON}/model_version={MODEL}/priors/",
        )
        records = []
        for ref in refs:
            day = ref["game_date_pacific"]
            key, sha = reference(ref, root + f"game_date_pacific={day}/")
            point = self.immutable(key, sha)
            self.identity(point, player_id, "player_game_history")
            if any(point[k] != index[k] for k in ("model_id", "model_run_id")):
                raise ValueError("History model identity changed between game points")
            if (
                point["game_date_pacific"] != day
                or point["date_contract"] != "completed_game_date_v1"
                or point["calculation_contract"] != "frozen_priors_stints_joint_hca_fixed_b2b_v1"
                or point["inputs"]["ratings_state"] != ref["source_ratings"]
            ):
                raise ValueError("Game-history point date/source contract mismatch")
            ratings_reference(ref["source_ratings"], day)
            if (
                point["starting_nail"] != start["nail"]
                or point["initialized_on"] != start["initialized_on"]
                or point["initialization"] != start["initialization"]
                or point["inputs"]["priors"] != index["initialization_input"]
            ):
                raise ValueError("Game history initialization provenance changed")
            scalar(point)
            if not isinstance(point["games"], list) or not point["games"]:
                raise ValueError("Game-date history record has no games")
            ids = [g["game_id"] for g in point["games"]]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate game in a player game-date record")
            for game in point["games"]:
                self.validate_game(game, day)
            is_current = current["game_date_pacific"] == day
            if is_current and (
                point["nail"] != current["nail"]
                or point["cumulative_possessions"] != current["cumulative_possessions"]
                or ref["source_ratings"] != current["source_ratings"]
            ):
                raise ValueError("Current scalar disagrees with same-date game point")
            records.append(
                {**point, "source": {"uri": ref["uri"], "sha256": sha}, "is_current": is_current}
            )
        return {
            **index,
            "kind": "saved_player_game_history",
            "points": records,
            "complete": True,
            "index": {
                "uri": f"s3://{BUCKET}/{root}latest.json",
                "observed_sha256": hashlib.sha256(raw).hexdigest(),
            },
        }

    @staticmethod
    def validate_game(game, day):
        gid = game["game_id"]
        if not isinstance(gid, str) or not re.fullmatch(r"\d{10}", gid):
            raise ValueError("Invalid player-game NBA ID")
        team_id(game["team_id"])
        team_id(game["opponent_team_id"])
        if (
            game["team_id"] == game["opponent_team_id"]
            or game["home_away"] not in {"home", "away"}
            or game["result"] not in {"W", "L"}
            or any(not isinstance(game[k], str) or not game[k] for k in ("team", "opponent"))
        ):
            raise ValueError("Invalid saved player-game identities/result")
        for field in ("actual_minutes", "possessions", "team_score", "opponent_score"):
            if _number(game[field]) < 0:
                raise ValueError("Negative saved game observation")
        participation = game["participation"]
        if (
            participation["status"] not in STATUSES
            or type(participation["box_score_row_present"]) is not bool
            or (
                participation["reason"] is not None and not isinstance(participation["reason"], str)
            )
        ):
            raise ValueError("Invalid producer participation classification")
        if game["prediction_purpose"] not in {"pregame_projection", "historical_replay"}:
            raise ValueError("Invalid saved prediction purpose")
        if game["pregame"] is not None:
            pregame = game["pregame"]
            _number(pregame["nail"])
            if _number(pregame["projected_minutes"]) < 0:
                raise ValueError("Negative projected minutes")
            for field in ("injury_status", "injury_reason"):
                if pregame[field] is not None and not isinstance(pregame[field], str):
                    raise ValueError("Invalid saved injury information")
        stats = game["stats"]
        if (
            participation["status"] in {"inactive", "not_listed", "did_not_play"}
            and stats is not None
        ):
            raise ValueError("Nonparticipant stats must be unavailable, not a fabricated zero line")
        if stats is not None:
            for field in STAT_FIELDS:
                if stats[field] is None:
                    continue
                value = _number(stats[field])
                if field != "plus_minus" and value < 0:
                    raise ValueError("Negative saved box statistic")
                if field.endswith("percentage") and not 0 <= value <= 1:
                    raise ValueError("Saved shooting percentages must be fractions")
        for name, kind in (
            ("actual_minutes", "game_predictions_actual_minutes"),
            ("prediction", "game_predictions"),
        ):
            reference(
                game["inputs"][name],
                f"lake/artifacts/{kind}/season={SEASON}/model_version={MODEL}/game_date_pacific={day}/game_id={gid}/",
            )
        box = game["inputs"]["box_score"]
        box_root = f"lake/raw/cdn/boxscore/season={SEASON}/game_id={gid}/"
        reference(box, box_root)
        if box[
            "metadata_uri"
        ] != f"s3://{BUCKET}/{box_root}{box['sha256']}.meta.json" or not re.fullmatch(
            r"[a-f0-9]{64}", box["metadata_sha256"]
        ):
            raise ValueError("Invalid pinned box sidecar reference")


@lru_cache(maxsize=1)
def _reader():
    import boto3

    return HistoryReader(boto3.client("s3"))


@lru_cache(maxsize=16)
def _history(player_id, _hour):
    return _reader().history(player_id)


@router.get("/api/players/{player_id}/rating-history")
def rating_history(response: Response, player_id: int = Path(gt=0)):
    hour = int(time() // 3600)
    try:
        with _lock:
            result = _history(player_id, hour)
        response.headers["Cache-Control"] = (
            f"private, max-age={max(0, int((hour + 1) * 3600 - time()))}"
        )
        return result
    except ERRORS as error:
        logging.getLogger(__name__).exception("Saved player game history unavailable")
        raise HTTPException(
            status_code=503,
            detail="Saved game history is incomplete or failed validation.",
            headers={"Cache-Control": "no-store"},
        ) from error
