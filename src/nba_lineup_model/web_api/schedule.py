"""Read-only selected-date Schedule adapters for saved official pipeline outputs."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from time import time
from zoneinfo import ZoneInfo

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, HTTPException, Response

BUCKET = "nba-gestalt-data"
PACIFIC = ZoneInfo("America/Los_Angeles")
router = APIRouter()


def _instant(value):
    if not isinstance(value, str):
        raise ValueError("Saved time is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Saved schedule time has no timezone")
    return parsed


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Saved value is not a finite number")
    return value


class ScheduleReader:
    """Only S3 get/list operations are used; immutable JSON is cached by checksum."""

    def __init__(self, client):
        self.client = client
        self.immutable = lru_cache(maxsize=128)(self._immutable)

    def _get(self, key, *, missing=False):
        try:
            result = self.client.get_object(Bucket=BUCKET, Key=key)
        except ClientError as error:
            if missing and error.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                return None
            raise
        raw = result["Body"].read()
        return raw, result

    def _immutable(self, key, sha):
        raw, _ = self._get(key)
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("Saved Schedule object failed checksum verification")
        return json.loads(raw)

    def selection(self, key, root, identity, *, missing=False):
        result = self._get(key, missing=missing)
        if result is None:
            return None
        pointer = json.loads(result[0])
        sha = pointer["sha256"]
        if not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise ValueError("Invalid saved selection checksum")
        selected_key = root + sha + ".json"
        if pointer["uri"] != f"s3://{BUCKET}/{selected_key}":
            raise ValueError("Saved selection escapes its game/date/version")
        payload = self.immutable(selected_key, sha)
        for record in (pointer, payload):
            for name, expected in identity.items():
                if record[name] != expected:
                    raise ValueError("Saved prediction identity mismatch: " + name)
        return payload, {"uri": pointer["uri"], "sha256": sha}

    def official_schedule(self, season):
        root = f"lake/raw/scheduleleaguev2/season={season}/"
        raw, _ = self._get(root + "latest.json")
        pointer = json.loads(raw)
        sha = pointer["sha256"]
        if not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise ValueError("Invalid official schedule checksum")
        key = root + f"sha256={sha}/schedule.json"
        if pointer["uri"] != f"s3://{BUCKET}/{key}":
            raise ValueError("Official schedule selection escapes its season")
        document = self.immutable(key, sha)["leagueSchedule"]
        if str(document["seasonYear"]) != season or str(document["leagueId"]) != "00":
            raise ValueError("Wrong official season/league")
        games, teams, seen = [], {}, set()
        for group in document["gameDates"]:
            for row in group["games"]:
                home, away = row["homeTeam"], row["awayTeam"]
                if (
                    str(row.get("gameStatus")) == "1"
                    and str(home["teamId"]) == str(away["teamId"]) == "0"
                ):
                    continue
                stamp = row.get("gameDateTimeUTC") or ""
                reported = (row.get("gameDateEst") or group["gameDate"])[:10]
                reported_day = (
                    datetime.strptime(reported, "%m/%d/%Y").date()
                    if "/" in reported
                    else date.fromisoformat(reported)
                )
                start = None if stamp.upper() in {"", "TBD", "TBA"} else _instant(stamp)
                game_id = str(row["gameId"])
                if not re.fullmatch(r"\d{10}", game_id) or game_id in seen:
                    raise ValueError("Invalid/duplicate official NBA game ID")
                seen.add(game_id)
                broadcasts = []
                for kind in (
                    "nationalTvBroadcasters",
                    "nationalOttBroadcasters",
                    "homeTvBroadcasters",
                    "awayTvBroadcasters",
                ):
                    for broadcaster in row.get("broadcasters", {}).get(kind, []):
                        label = broadcaster.get("broadcasterDisplay") or broadcaster.get(
                            "broadcasterAbbreviation"
                        )
                        if label and label not in broadcasts:
                            broadcasts.append(label)
                game = {
                    "game_id": game_id,
                    "start_time": stamp if start else None,
                    "game_date": str(start.astimezone(PACIFIC).date() if start else reported_day),
                    "home_team": home["teamTricode"],
                    "away_team": away["teamTricode"],
                    "home_team_name": (
                        home.get("teamCity", "") + " " + home.get("teamName", "")
                    ).strip()
                    or None,
                    "away_team_name": (
                        away.get("teamCity", "") + " " + away.get("teamName", "")
                    ).strip()
                    or None,
                    "home_team_id": int(home["teamId"]),
                    "away_team_id": int(away["teamId"]),
                    "broadcasts": broadcasts,
                }
                if not game["home_team"] or not game["away_team"]:
                    raise ValueError("Official schedule has no team identity")
                games.append(game)
                if game_id.startswith("002"):
                    teams[game["home_team_id"]] = game["home_team"]
                    teams[game["away_team_id"]] = game["away_team"]
        return (
            games,
            [{"team_id": k, "team": v} for k, v in sorted(teams.items(), key=lambda kv: kv[1])],
            {"uri": pointer["uri"], "sha256": sha},
        )

    def scoreboard(self, season, day, scheduled):
        key = f"lake/raw/cdn/scoreboard/season={season}/game_date_pacific={day}/latest.json"
        result = self._get(key, missing=True)
        if result is None:
            return {}, None
        raw, response = result
        metadata = response.get("Metadata", {})
        sha = hashlib.sha256(raw).hexdigest()
        if sha != metadata.get("sha256"):
            raise ValueError("Scoreboard checksum mismatch")
        board = json.loads(raw)["scoreboard"]
        if board["gameDate"] != day:
            raise ValueError("Scoreboard belongs to a different Pacific slate")
        captured = metadata["fetched-at"]
        _instant(captured)
        expected = {g["game_id"]: g for g in scheduled}
        rows = {}
        for row in board["games"]:
            gid = str(row["gameId"])
            if gid not in expected or gid in rows:
                raise ValueError("Scoreboard game IDs differ from official schedule")
            self.validate_teams(row, expected[gid])
            rows[gid] = row
        return rows, {"uri": f"s3://{BUCKET}/{key}", "sha256": sha, "captured_at": captured}

    @staticmethod
    def validate_teams(row, scheduled):
        for side in ("home", "away"):
            team = row[side + "Team"]
            if (
                str(team["teamId"]) != str(scheduled[side + "_team_id"])
                or team["teamTricode"] != scheduled[side + "_team"]
            ):
                raise ValueError("Official observation team identity mismatch")

    def final_box(self, season, scheduled):
        root = f"lake/raw/cdn/boxscore/season={season}/game_id={scheduled['game_id']}/"
        candidates = []
        for page in self.client.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=root
        ):
            for item in page.get("Contents", []):
                name = item["Key"][len(root) :]
                if "espn" in name.casefold():
                    raise ValueError("Prohibited basketball source in box-score archive")
                if re.fullmatch(r"[a-f0-9]{64}\.json", name):
                    candidates.append(item)
        if not candidates:
            return None
        selected = max(candidates, key=lambda item: (item["LastModified"], item["Key"]))
        key = selected["Key"]
        sha = key[len(root) : -5]
        game, reference = self._read_final_box(season, scheduled, key, sha)
        return game, {**reference, "captured_at": selected["LastModified"].isoformat()}

    def pinned_final_box(self, season, scheduled, reference):
        """Verify the exact published input; never replace it with a newer capture."""
        root = f"lake/raw/cdn/boxscore/season={season}/game_id={scheduled['game_id']}/"
        sha, metadata_sha = reference["sha256"], reference["metadata_sha256"]
        if not all(
            isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v) for v in (sha, metadata_sha)
        ):
            raise ValueError("Invalid pinned final-box checksums")
        key = root + sha + ".json"
        if (
            reference["uri"] != f"s3://{BUCKET}/{key}"
            or reference["metadata_uri"] != f"s3://{BUCKET}/{root}{sha}.meta.json"
        ):
            raise ValueError("Pinned final-box reference escapes its season/game")
        return self._read_final_box(season, scheduled, key, sha, metadata_sha)

    def _read_final_box(self, season, scheduled, key, sha, metadata_sha=None):
        document = self.immutable(key, sha)
        game = document["game"]
        if str(game["gameId"]) != scheduled["game_id"] or str(game["gameStatus"]) != "3":
            raise ValueError("Saved NBA box score is mismatched or not final")
        self.validate_teams(game, scheduled)
        metadata_key = key[:-5] + ".meta.json"
        if metadata_sha is None:
            raw_metadata, _ = self._get(metadata_key)
            metadata_sha = hashlib.sha256(raw_metadata).hexdigest()
            metadata = json.loads(raw_metadata)
        else:
            metadata = self.immutable(metadata_key, metadata_sha)
        expected = {"season": season, "game_id": game["gameId"], "box_score_sha256": sha}
        for side in ("home", "away"):
            team = game[side + "Team"]
            _number(team["score"])
            players = team["players"]
            if not players:
                raise ValueError("Final box score has no player roster")
            roster, ids = [], set()
            for player in players:
                pid, name = player["personId"], player["name"]
                if (
                    type(pid) is not int
                    or pid <= 0
                    or pid in ids
                    or not isinstance(name, str)
                    or not name.strip()
                ):
                    raise ValueError("Invalid saved final-box roster identity")
                ids.add(pid)
                roster.append({"player_id": pid, "player_name": name.strip()})
            encoded = json.dumps(
                sorted(roster, key=lambda r: r["player_id"]),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
            expected[side + "_team_id"] = team["teamId"]
            expected[side + "_roster_sha256"] = hashlib.sha256(encoded).hexdigest()
        if metadata != expected:
            raise ValueError("Final-box sidecar differs from its saved box score")
        return game, {
            "uri": f"s3://{BUCKET}/{key}",
            "sha256": sha,
            "metadata_uri": f"s3://{BUCKET}/{metadata_key}",
            "metadata_sha256": metadata_sha,
        }

    def prediction(self, season, day, scheduled):
        root = (
            f"lake/artifacts/game_predictions/season={season}/model_version=v1/"
            f"game_date_pacific={day}/game_id={scheduled['game_id']}/"
        )
        selected = self.selection(
            root + "latest.json",
            root,
            {
                "season": season,
                "model_version": "v1",
                "game_date_pacific": day,
                "game_id": scheduled["game_id"],
                "schema_version": 1,
            },
            missing=True,
        )
        if selected is None:
            return None
        record, reference = selected
        if record["artifact_kind"] != "game_prediction" or record["purpose"] not in {
            "pregame_projection",
            "historical_replay",
        }:
            raise ValueError("Wrong saved prediction kind/purpose")
        if _instant(record["scheduled_start_utc"]) != _instant(scheduled["start_time"]):
            raise ValueError("Prediction tipoff differs from official schedule")
        for side in ("home", "away"):
            if (
                record[side]["team_id"] != scheduled[side + "_team_id"]
                or record[side]["team"] != scheduled[side + "_team"]
            ):
                raise ValueError("Prediction team identity mismatch")
        return record, reference

    def game_view(self, season, day, scheduled, board, board_ref, now):
        selected = self.prediction(season, day, scheduled)
        observation = board.get(scheduled["game_id"])
        observation_ref = board_ref if observation else None
        box_missing = False
        if observation is None and _instant(scheduled["start_time"]) <= now:
            box = self.final_box(season, scheduled)
            if box is not None:
                observation, observation_ref = box
            else:
                box_missing = True
        state = "pre" if _instant(scheduled["start_time"]) > now else "unknown"
        status = {
            "state": state,
            "detail": "Scheduled" if state == "pre" else "Score unavailable",
            "period": None,
            "display_clock": None,
            "completed": False,
            "description": "Scheduled" if state == "pre" else "Score unavailable",
        }
        home_score = away_score = None
        if observation is not None:
            raw_status = observation["gameStatus"]
            if raw_status not in (1, 2, 3):
                raise ValueError("Unknown official game status")
            status = {
                "state": {1: "pre", 2: "in", 3: "post"}[raw_status],
                "detail": observation.get("gameStatusText") or "",
                "period": observation.get("period"),
                "display_clock": observation.get("gameClock"),
                "completed": raw_status == 3,
                "description": observation.get("gameStatusText") or "",
            }
            if raw_status in (2, 3):
                home_score = _number(observation["homeTeam"]["score"])
                away_score = _number(observation["awayTeam"]["score"])
                if home_score is None or away_score is None:
                    raise ValueError("Official score missing for live/final game")
        result = {
            **scheduled,
            "status": status,
            "home_score": home_score,
            "away_score": away_score,
            "projection": None,
            "market": None,
            "prediction_market": None,
            "injury_report": None,
            "projected_minutes": None,
            "prediction_purpose": None,
            "sources": {"prediction": None, "observation": observation_ref},
            "availability": {
                "prediction": "missing",
                "observation": "available" if observation else "missing",
                "market": "not_migrated",
                "postgame_details": "not_migrated",
            },
        }
        if selected is not None:
            record, reference = selected
            result.update(_prediction_view(record))
            result["sources"]["prediction"] = reference
            result["availability"]["prediction"] = "available"
        return result, box_missing

    def view(self, season, day, *, now=None):
        now = now or datetime.now(UTC)
        target = date.fromisoformat(day)
        scheduled, teams, schedule_ref = self.official_schedule(season)
        calendar = {
            str(target + timedelta(days=offset)): {
                "game_date": str(target + timedelta(days=offset)),
                "game_count": sum(
                    g["game_date"] == str(target + timedelta(days=offset)) for g in scheduled
                ),
            }
            for offset in range(-3, 4)
        }
        slate = [g for g in scheduled if g["game_date"] == day]
        if any(g["start_time"] is None for g in slate):
            raise ValueError("Selected slate has an unverified tipoff")
        board, board_ref = self.scoreboard(season, day, slate) if slate else ({}, None)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(
                pool.map(lambda g: self.game_view(season, day, g, board, board_ref, now), slate)
            )
        games = sorted([g for g, _ in results], key=lambda g: (g["start_time"], g["game_id"]))
        from .schedule_details import DetailsReader

        DetailsReader(self, season, day, now).enrich(games)
        coverage = {
            "scheduled_game_ids": [g["game_id"] for g in games],
            "prediction_game_ids": [g["game_id"] for g in games if g["projection"] is not None],
            "missing_prediction_game_ids": [g["game_id"] for g in games if g["projection"] is None],
            "missing_observation_game_ids": [
                g["game_id"] for g in games if g["status"]["state"] == "unknown"
            ],
            "missing_scoreboard_game_ids": [
                g["game_id"] for g in games if g["game_id"] not in board
            ],
            "missing_final_box_game_ids": [g["game_id"] for g, missing in results if missing],
        }
        coverage["complete"] = (
            not coverage["missing_prediction_game_ids"]
            and not coverage["missing_observation_game_ids"]
        )
        coverage["markets"] = {
            state: [g["game_id"] for g in games if g["availability"]["market"] == state]
            for state in ("available", "missing", "stale", "invalid")
        }
        coverage["grading"] = {
            state: [g["game_id"] for g in games if g["availability"]["grading"] == state]
            for state in ("available", "missing", "stale", "invalid", "not_applicable")
        }
        coverage["postgame_details"] = {
            state: [g["game_id"] for g in games if g["availability"]["postgame_details"] == state]
            for state in ("available", "missing", "stale", "invalid", "not_applicable")
        }
        captures = [
            g["sources"]["observation"]["captured_at"] for g in games if g["sources"]["observation"]
        ]
        collected = max(captures, key=_instant) if captures else None
        return {
            "season": season,
            "date": day,
            "calendar": calendar,
            "teams": teams,
            "day": {
                "available": True,
                "game_date": day,
                "source": "official_nba_saved_pipeline_data",
                "games": games,
                "collected_at": collected,
                "live_score_collected_at": board_ref["captured_at"] if board_ref else None,
                "coverage": coverage,
                "sources": {"schedule": schedule_ref, "scoreboard": board_ref},
            },
        }


def _prediction_view(record):
    components = record["components"]
    margin = _number(record["predicted_home_margin"])
    probability = _number(record["home_win_probability"])
    if not 0 <= probability <= 1:
        raise ValueError("Saved win probability outside [0,1]")
    provenance = {
        "schema_version": 1,
        "rating_as_of": record.get("ratings_observed_through_game_date_pacific"),
        "as_of_cutoff_time": None,
        "inputs": record["inputs"],
    }
    ledger = {
        "schema_version": 1,
        "artifact_kind": "saved_pipeline_prediction_attribution",
        "contract": "Saved game prediction components; no recalculation",
        "assumed_possessions": _number(record["assumed_possessions"]),
        "away": {
            "team": record["away"]["team"],
            "team_strength": _number(components["away_player_total"]),
            "projected_minutes": _number(record["away"]["total_projected_minutes"]),
        },
        "home": {
            "team": record["home"]["team"],
            "team_strength": _number(components["home_player_total"]),
            "projected_minutes": _number(record["home"]["total_projected_minutes"]),
        },
        "team_strength_difference": components["home_player_total"]
        - components["away_player_total"],
        "home_court_adjustment": _number(components["hca"]),
        "back_to_back": {
            "away_team_is_back_to_back": components["away_is_b2b"],
            "home_team_is_back_to_back": components["home_is_b2b"],
            "coefficient": _number(components["b2b_coefficient"]),
            "adjustment": _number(components["b2b_adjustment"]),
        },
        "home_margin": margin,
        "fair_home_spread": -margin,
        "provenance": provenance,
    }
    minutes = {}
    for side in ("home", "away"):
        minutes[side] = []
        for row in record[side]["players"]:

            def nullable(name, row=row):
                return None if row[name] is None else _number(row[name])

            minutes[side].append(
                {
                    "player_id": row["player_id"],
                    "player_name": row["name"],
                    "status": row["injury_status"],
                    "reason": row["injury_reason"],
                    "play_probability": nullable("play_probability"),
                    "conditional_minutes": nullable("conditional_minutes"),
                    "projected_minutes": nullable("projected_minutes"),
                    "projected_nail": nullable("nail"),
                    "expected_contribution": nullable("rating_contribution"),
                    "selected_rotation": row["selected_rotation"],
                }
            )
    injury = record["injury_report"]
    return {
        "projection": {
            "model": record["model_version"],
            "home_margin": margin,
            "home_win_probability": probability,
            "fair_home_spread": -margin,
            "input_provenance": provenance,
            "ledger": ledger,
        },
        "prediction_purpose": record["purpose"],
        "prediction_generated_at": record["generated_at_utc"],
        "projected_minutes": minutes,
        "injury_report": {
            "source": "official_nba_pinned_prediction_input",
            "source_url": injury.get("source_url"),
            "pdf_uri": injury.get("pdf_uri"),
            "published_at": injury.get("published_at"),
            "team_reports": {
                record[side]["team"]: {"status": record[side]["submission_status"]}
                for side in ("home", "away")
            },
        },
    }


@lru_cache(maxsize=1)
def _reader():
    import boto3

    return ScheduleReader(boto3.client("s3"))


@lru_cache(maxsize=1)
def _cached_view(season, day, _minute):
    return _reader().view(season, day)


def _serve(day, response):
    selected = day or datetime.now(PACIFIC).date()
    first_year = selected.year if selected.month >= 7 else selected.year - 1
    season = f"{first_year}-{str(first_year + 1)[-2:]}"
    try:
        minute = int(time() // 60)
        result = _cached_view(season, str(selected), minute)
        response.headers["Cache-Control"] = (
            f"private, max-age={max(0, math.floor((minute + 1) * 60 - time()))}"
        )
        return result
    except (OSError, ValueError, KeyError, TypeError, BotoCoreError, ClientError) as error:
        logging.getLogger(__name__).exception("Saved Schedule unavailable for %s", selected)
        raise HTTPException(
            status_code=503,
            detail="Saved Schedule data is unavailable for this date.",
            headers={"Cache-Control": "no-store"},
        ) from error


@router.get("/api/schedule-view")
def schedule_view(response: Response, date: date | None = None):
    return _serve(date, response)


@router.get("/api/schedule")
def schedule_day(response: Response, date: date | None = None):
    return _serve(date, response)["day"]
