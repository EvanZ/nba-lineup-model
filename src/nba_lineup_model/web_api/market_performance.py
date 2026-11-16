"""Read saved market evaluations without model/release or Dagster dependencies."""

from __future__ import annotations

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from functools import lru_cache
from time import time
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, Response

BUCKET = "nba-gestalt-data"
SERIES = ("model", "actual_minutes_counterfactual", "polymarket", "bookmaker_consensus")
ATS_FIELDS = ("wins", "losses", "pushes", "no_picks", "not_applicable", "unavailable")
router = APIRouter()


def _object(client, key):
    raw = client.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return raw, json.loads(raw)


def _day(client, prefix, day, season):
    root = prefix + f"game_date_pacific={day}/"
    _, pointer = _object(client, root + "latest.json")
    sha = pointer["sha256"]
    if not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError("Invalid market-performance checksum")
    uri = f"s3://{BUCKET}/{root}{sha}.json"
    if pointer["uri"] != uri:
        raise ValueError("Market selection escapes its date/version")
    raw, payload = _object(client, root + sha + ".json")
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Market-performance checksum mismatch")
    for record in (pointer, payload):
        if (
            record["season"] != season
            or record["model_version"] != "v1"
            or record["game_date_pacific"] != day
            or record["schema_version"] != 1
        ):
            raise ValueError("Market-performance identity mismatch")
    if payload["artifact_kind"] != "market_performance":
        raise ValueError("Wrong market-performance artifact kind")
    games = payload["games"]
    ids = [str(g["game_id"]) for g in games]
    coverage = payload["coverage"]
    if (
        len(set(ids)) != len(ids)
        or len(ids) != coverage["scheduled_game_count"]
        or coverage["completed_game_count"] != len(ids)
    ):
        raise ValueError("Saved market evaluation has incomplete game coverage")
    payload["selection"] = {"uri": uri, "sha256": sha}
    return payload


def edge_bins(days):
    """Bin paired model/book margins; retain the producer's saved ATS grading."""
    definitions = (
        ("zero_to_one", "0–1 pts", 0, True, 1),
        ("over_one_to_three", ">1–3 pts", 1, False, 3),
        ("over_three", ">3 pts", 3, False, None),
    )
    bins = [
        {
            "key": key,
            "label": label,
            "minimum_edge": minimum,
            "minimum_inclusive": inclusive,
            "maximum_edge": maximum,
            "maximum_inclusive": maximum is not None,
            "n": 0,
            **{field: 0 for field in ATS_FIELDS},
        }
        for key, label, minimum, inclusive, maximum in definitions
    ]
    result_fields = {
        "win": "wins",
        "loss": "losses",
        "push": "pushes",
        "no_pick": "no_picks",
        "not_applicable": "not_applicable",
        "unavailable": "unavailable",
    }
    missing = 0
    for day in days:
        for game in day["games"]:
            model = game["comparisons"]["model"]
            market = game["comparisons"]["bookmaker_consensus"]
            prediction = model.get("predicted_home_margin")
            benchmark = market.get("predicted_home_margin")
            if prediction is None or benchmark is None:
                missing += 1
                continue
            edge = abs(float(prediction) - float(benchmark))
            if not math.isfinite(edge):
                raise ValueError("Non-finite model/market edge")
            ats = model["ats"]
            if ats["benchmark"] != "bookmaker_consensus":
                raise ValueError("Unexpected ATS benchmark for edge bins")
            if ats["result"] not in result_fields:
                raise ValueError("Unknown saved ATS result")
            bucket = bins[0 if edge <= 1 else 1 if edge <= 3 else 2]
            bucket["n"] += 1
            bucket[result_fields[ats["result"]]] += 1
    paired = sum(bucket["n"] for bucket in bins)
    for bucket in bins:
        graded = bucket["wins"] + bucket["losses"]
        bucket["graded_games"] = graded
        bucket["cover_rate"] = bucket["wins"] / graded if graded else None
        bucket["share"] = bucket["n"] / paired if paired else None
    return {
        "bins": bins,
        "paired_games": paired,
        "missing_pairs": missing,
        "cohort": "games_with_model_and_bookmaker_consensus_margins",
    }


def aggregate_totals(days):
    """Preserve the producer's count-weighted metrics and ATS definitions."""
    result = {}
    for name in SERIES:
        rows = [day["series"][name] for day in days]
        count = sum(row["probability_game_count"] for row in rows)
        margins = sum(row["margin_game_count"] for row in rows)
        correct = sum(row["correct_winner_count"] for row in rows)
        ats = {key: sum(row["ats"][key] for row in rows) for key in ATS_FIELDS}
        bets = ats["wins"] + ats["losses"]
        ats["win_rate_excluding_pushes"] = ats["wins"] / bets if bets else None
        result[name] = {
            "games": count,
            "brier": math.fsum(r["brier_loss_sum"] for r in rows) / count if count else None,
            "winner_accuracy": correct / count if count else None,
            "correct_winners": correct,
            "margin_game_count": margins,
            "margin_rmse": math.sqrt(
                math.fsum(r["squared_margin_error_sum"] for r in rows) / margins
            )
            if margins
            else None,
            "margin_mae": math.fsum(r["absolute_margin_error_sum"] for r in rows) / margins
            if margins
            else None,
            "ats": ats,
        }
        for key in ("brier", "winner_accuracy", "margin_rmse", "margin_mae"):
            if result[name][key] is not None and not math.isfinite(result[name][key]):
                raise ValueError("Non-finite market-performance metric")
    return result


def _scheduled_games(client, season, through):
    from urllib.parse import urlparse

    root = f"lake/raw/scheduleleaguev2/season={season}/"
    _, pointer = _object(client, root + "latest.json")
    uri = pointer["uri"]
    sha = pointer["sha256"]
    expected = f"s3://{BUCKET}/{root}sha256={sha}/schedule.json"
    if uri != expected:
        raise ValueError("Official schedule selection escapes its season")
    raw, payload = _object(client, urlparse(uri).path.lstrip("/"))
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Official schedule checksum mismatch")
    schedule = payload["leagueSchedule"]
    if str(schedule["seasonYear"]) != season or str(schedule["leagueId"]) != "00":
        raise ValueError("Official schedule identity mismatch")
    dated_games = []
    for group in schedule["gameDates"]:
        for game in group["games"]:
            game_id = str(game["gameId"])
            if (
                str(game.get("gameStatus")) == "1"
                and str(game["homeTeam"]["teamId"]) == str(game["awayTeam"]["teamId"]) == "0"
            ):
                continue
            timestamp = game.get("gameDateTimeUTC") or ""
            reported = (game.get("gameDateEst") or group["gameDate"])[:10]
            reported_date = (
                datetime.strptime(reported, "%m/%d/%Y").date()
                if "/" in reported
                else date.fromisoformat(reported)
            )
            if timestamp.upper() in {"", "TBD", "TBA"}:
                if reported_date <= through:
                    raise ValueError("Completed schedule date has no verified UTC tipoff")
                continue
            start = datetime.fromisoformat(timestamp)
            if start.tzinfo is None:
                raise ValueError("Official schedule time lacks timezone")
            day = start.astimezone(ZoneInfo("America/Los_Angeles")).date()
            if day <= through:
                dated_games.append((day.isoformat(), game_id))
    # Match the producer's season boundary without excluding later Cup/postseason games.
    regular_dates = [day for day, game_id in dated_games if game_id.startswith("002")]
    games = {}
    if regular_dates:
        season_start = min(regular_dates)
        for day, game_id in dated_games:
            if day >= season_start:
                games.setdefault(day, set()).add(game_id)
    return games, {"uri": uri, "sha256": sha}


def read_market_performance(client, *, season, through):
    if not re.fullmatch(r"\d{4}-\d{2}", season):
        raise ValueError("Invalid season")
    if int(season[-2:]) != (int(season[:4]) + 1) % 100:
        raise ValueError("Invalid season range")
    prefix = f"lake/artifacts/market_performance/season={season}/model_version=v1/"
    scheduled, schedule_input = _scheduled_games(client, season, through)
    dates = []
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix, Delimiter="/"
    ):
        for item in page.get("CommonPrefixes", []):
            pattern = re.escape(prefix) + r"game_date_pacific=(\d{4}-\d{2}-\d{2})/"
            match = re.fullmatch(pattern, item["Prefix"])
            if match and date.fromisoformat(match[1]) <= through:
                dates.append(match[1])
    dates = sorted(set(dates))
    if not dates:
        raise ValueError("No saved market-performance reports are available")
    with ThreadPoolExecutor(max_workers=8) as pool:
        days = list(pool.map(lambda day: _day(client, prefix, day, season), dates))
    for day in days:
        actual = {str(g["game_id"]) for g in day["games"]}
        if actual != scheduled.get(day["game_date_pacific"], set()):
            raise ValueError("Saved market game IDs differ from the official slate")
    missing_dates = sorted(set(scheduled) - set(dates))
    cohorts = {day["daily_totals"]["cohort"] for day in days}
    if len(cohorts) != 1:
        raise ValueError("Cannot combine different market-performance cohorts")
    windows = []
    for key, label, length in (
        ("season", "Season", None),
        ("7d", "Last 7 days", 7),
        ("14d", "Last 14 days", 14),
        ("30d", "Last 30 days", 30),
    ):
        start = (
            through - timedelta(days=length - 1) if length else date.fromisoformat(min(scheduled))
        )
        selected = [d for d in days if date.fromisoformat(d["game_date_pacific"]) >= start]
        windows.append(
            {
                "key": key,
                "label": label,
                "start_date": start.isoformat(),
                "end_date": through.isoformat(),
                "cohort": next(iter(cohorts)),
                "game_count": sum(d["coverage"]["completed_game_count"] for d in selected),
                "comparison_game_count": sum(d["daily_totals"]["game_count"] for d in selected),
                "series": aggregate_totals([d["daily_totals"] for d in selected]),
                "edge_bins": edge_bins(selected),
            }
        )
    return {
        "season": season,
        "model_version": "v1",
        "through": through.isoformat(),
        "as_of": dates[-1],
        "updated_at": max(d["generated_at"] for d in days),
        "source": "market_performance",
        "windows": windows,
        "coverage": {
            "saved_dates": dates,
            "dates_without_saved_reports": missing_dates,
            "missing_game_ids_by_date": {d: sorted(scheduled[d]) for d in missing_dates},
            "expected_game_count": sum(len(v) for v in scheduled.values()),
            "complete": not missing_dates,
            "missing_market_fields": {
                d["game_date_pacific"]: d["coverage"]["missing_market_fields"]
                for d in days
                if d["coverage"]["missing_market_fields"]
            },
        },
        "inputs": [d["selection"] for d in days],
        "schedule_input": schedule_input,
    }


@lru_cache(maxsize=8)
def _cached_report(season, through, _hour):
    import boto3

    return read_market_performance(boto3.client("s3"), season=season, through=through)


@router.get("/api/market-performance")
def market_performance(
    response: Response,
    season: str = Query(pattern=r"^\d{4}-\d{2}$"),
    through: date | None = None,
):
    """Read accepted saved selections; never launch or materialize work."""
    from botocore.exceptions import BotoCoreError, ClientError

    cutoff = through or (datetime.now(ZoneInfo("America/Los_Angeles")).date() - timedelta(days=1))
    try:
        hour = int(time() // 3600)
        report = _cached_report(season, cutoff, hour)
        # Bound browser caching to the same hour as the server cache entry.
        remaining = max(0, math.floor((hour + 1) * 3600 - time()))
        response.headers["Cache-Control"] = f"private, max-age={remaining}"
        return report
    except (OSError, ValueError, KeyError, TypeError, BotoCoreError, ClientError) as error:
        import logging

        logging.getLogger(__name__).exception("Saved market report unavailable")
        raise HTTPException(
            status_code=503,
            detail="The saved market report is unavailable.",
            headers={"Cache-Control": "no-store"},
        ) from error
