import copy
import hashlib
import io
import json
from datetime import date, timedelta
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from nba_lineup_model.web_api import market_performance as api
from nba_lineup_model.web_api.gateway import MarketGateway


def totals(count, brier, squared, absolute):
    row = {
        "probability_game_count": count,
        "margin_game_count": count,
        "correct_winner_count": count,
        "brier_loss_sum": brier,
        "squared_margin_error_sum": squared,
        "absolute_margin_error_sum": absolute,
        "ats": {key: (count if key == "wins" else 0) for key in api.ATS_FIELDS},
    }
    return {
        "cohort": "shared",
        "game_count": count,
        "series": {name: copy.deepcopy(row) for name in api.SERIES},
    }


class SavedS3:
    """Only read operations exist; any write in the consumer fails the test."""

    def __init__(self, days, season_start=None):
        self.objects = {}
        self.prefix = "lake/artifacts/market_performance/season=2026-27/model_version=v1/"
        self.dates = sorted(days)
        game_dates = []
        start, end = (
            date.fromisoformat(season_start or self.dates[0]),
            date.fromisoformat(self.dates[-1]),
        )
        for offset in range((end - start).days + 1):
            day = (start + timedelta(days=offset)).isoformat()
            count = days.get(day, {"game_count": 1})["game_count"]
            game_dates.append(
                {
                    "gameDate": day,
                    "games": [
                        {
                            "gameId": "002" + day.replace("-", "") + str(n),
                            "gameDateTimeUTC": day + "T20:00:00Z",
                            "gameStatus": 3,
                            "homeTeam": {"teamId": 1},
                            "awayTeam": {"teamId": 2},
                        }
                        for n in range(count)
                    ],
                }
            )
        schedule = json.dumps(
            {
                "leagueSchedule": {
                    "seasonYear": "2026-27",
                    "leagueId": "00",
                    "gameDates": game_dates,
                }
            }
        ).encode()
        sha = hashlib.sha256(schedule).hexdigest()
        root = "lake/raw/scheduleleaguev2/season=2026-27/"
        uri = f"s3://{api.BUCKET}/{root}sha256={sha}/schedule.json"
        self.objects[root + f"sha256={sha}/schedule.json"] = schedule
        self.objects[root + "latest.json"] = json.dumps({"uri": uri, "sha256": sha}).encode()
        for day, total in days.items():
            payload = {
                "schema_version": 1,
                "artifact_kind": "market_performance",
                "season": "2026-27",
                "model_version": "v1",
                "game_date_pacific": day,
                "generated_at": day + "T12:00:00Z",
                "daily_totals": total,
                "games": [
                    {
                        "game_id": "002" + day.replace("-", "") + str(n),
                        "comparisons": {
                            "model": {
                                "predicted_home_margin": 1,
                                "ats": {"benchmark": "bookmaker_consensus", "result": "win"},
                            },
                            "bookmaker_consensus": {"predicted_home_margin": 0},
                        },
                    }
                    for n in range(total["game_count"])
                ],
                "coverage": {
                    "scheduled_game_count": total["game_count"],
                    "completed_game_count": total["game_count"],
                    "missing_market_fields": [],
                },
            }
            raw = json.dumps(payload).encode()
            sha = hashlib.sha256(raw).hexdigest()
            root = self.prefix + f"game_date_pacific={day}/"
            self.objects[root + sha + ".json"] = raw
            pointer = {
                k: payload[k]
                for k in ("schema_version", "season", "model_version", "game_date_pacific")
            }
            pointer.update(uri=f"s3://{api.BUCKET}/{root}{sha}.json", sha256=sha)
            self.objects[root + "latest.json"] = json.dumps(pointer).encode()

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, **kwargs):
        assert kwargs["Prefix"] == self.prefix
        yield {
            "CommonPrefixes": [
                {"Prefix": self.prefix + f"game_date_pacific={d}/"} for d in self.dates
            ]
        }

    def get_object(self, *, Bucket, Key):
        assert Bucket == api.BUCKET
        return {"Body": io.BytesIO(self.objects[Key])}


def test_weighted_metrics_windows_and_explicit_date_gap():
    client = SavedS3({"2026-11-01": totals(1, 0.1, 4, 2), "2026-11-03": totals(3, 0.9, 12, 6)})
    out = api.read_market_performance(client, season="2026-27", through=date(2026, 11, 3))
    score = out["windows"][0]["series"]["model"]
    assert score["brier"] == pytest.approx(0.25)
    assert score["margin_rmse"] == 2
    assert score["margin_mae"] == 2
    assert score["games"] == 4
    assert out["coverage"]["dates_without_saved_reports"] == ["2026-11-02"]
    assert len(out["inputs"]) == 2


def test_corrupt_snapshot_fails_without_substituting_other_data():
    client = SavedS3({"2026-11-01": totals(1, 0.1, 4, 2)})
    key = next(
        k for k in client.objects if "market_performance" in k and not k.endswith("latest.json")
    )
    client.objects[key] += b" "
    with pytest.raises(ValueError, match="checksum"):
        api.read_market_performance(client, season="2026-27", through=date(2026, 11, 1))


def test_empty_denominators_are_null():
    score = api.aggregate_totals([totals(0, 0, 0, 0)])["model"]
    assert score["brier"] is None
    assert score["margin_rmse"] is None
    assert score["ats"]["win_rate_excluding_pushes"] is None


def test_market_route_does_not_initialize_legacy_backend(monkeypatch):
    reader = Mock(return_value={"source": "market_performance"})
    monkeypatch.setattr(api, "_cached_report", reader)
    legacy = Mock(side_effect=AssertionError("legacy initialized"))
    with TestClient(MarketGateway(legacy)) as client:
        response = client.get("/api/market-performance?season=2026-27&through=2026-11-14")
    assert response.status_code == 200
    legacy.assert_not_called()
    assert reader.call_args.args[:2] == ("2026-27", date(2026, 11, 14))


def test_legacy_failure_does_not_break_market_route(monkeypatch):
    monkeypatch.setattr(api, "_cached_report", lambda *args: {"source": "market_performance"})
    legacy = Mock(side_effect=ValueError("missing legacy manifest"))
    with TestClient(MarketGateway(legacy)) as client:
        assert client.get("/api/health").status_code == 503
        assert client.get("/api/market-performance?season=2026-27").status_code == 200
    legacy.assert_called_once()


def test_missing_opening_date_is_reported_not_hidden_by_first_saved_date():
    client = SavedS3({"2026-11-02": totals(1, 0.1, 4, 2)}, season_start="2026-11-01")
    out = api.read_market_performance(client, season="2026-27", through=date(2026, 11, 2))
    assert out["windows"][0]["start_date"] == "2026-11-01"
    assert out["coverage"]["dates_without_saved_reports"] == ["2026-11-01"]
    assert out["coverage"]["expected_game_count"] == 2
    assert out["coverage"]["complete"] is False
    assert out["windows"][0]["game_count"] == 1


def test_foreign_snapshot_uri_is_rejected():
    client = SavedS3({"2026-11-01": totals(1, 0.1, 4, 2)})
    key = next(k for k in client.objects if "market_performance" in k and k.endswith("latest.json"))
    pointer = json.loads(client.objects[key])
    pointer["uri"] = "s3://another-bucket/private.json"
    client.objects[key] = json.dumps(pointer).encode()
    with pytest.raises(ValueError, match="escapes"):
        api.read_market_performance(client, season="2026-27", through=date(2026, 11, 1))


def edge_game(margin, result, benchmark=0):
    return {
        "comparisons": {
            "model": {
                "predicted_home_margin": margin,
                "ats": {"benchmark": "bookmaker_consensus", "result": result},
            },
            "bookmaker_consensus": {"predicted_home_margin": benchmark},
        }
    }


def test_edge_bin_boundaries_volume_and_saved_outcomes():
    games = [
        edge_game(0, "no_pick"),
        edge_game(1, "win"),
        edge_game(1.000001, "loss"),
        edge_game(3, "push"),
        edge_game(3.000001, "win"),
        edge_game(-4, "loss"),
        edge_game(None, "unavailable"),
    ]
    out = api.edge_bins([{"games": games}])
    assert out["paired_games"] == 6
    assert out["missing_pairs"] == 1
    low, middle, high = out["bins"]
    assert [b["n"] for b in out["bins"]] == [2, 2, 2]
    assert low["wins"] == 1 and low["no_picks"] == 1
    assert middle["losses"] == 1 and middle["pushes"] == 1
    assert high["wins"] == 1 and high["losses"] == 1
    assert [b["cover_rate"] for b in out["bins"]] == [1, 0, 0.5]
    assert all(b["share"] == pytest.approx(1 / 3) for b in out["bins"])


def test_edge_uses_model_market_difference_and_not_prediction_magnitude():
    out = api.edge_bins([{"games": [edge_game(10, "win", benchmark=9)]}])
    assert [b["n"] for b in out["bins"]] == [1, 0, 0]


def test_empty_edge_bins_have_null_rates():
    out = api.edge_bins([{"games": [edge_game(None, "unavailable")]}])
    assert out["paired_games"] == 0
    assert all(b["cover_rate"] is None and b["share"] is None for b in out["bins"])


def test_hourly_cache_reuses_reads_and_expires_at_next_hour(monkeypatch):
    import boto3

    clock = [360_123.0]
    monkeypatch.setattr(api, "time", lambda: clock[0])
    monkeypatch.setattr(boto3, "client", lambda name: object())
    reader = Mock(return_value={"source": "market_performance"})
    monkeypatch.setattr(api, "read_market_performance", reader)
    api._cached_report.cache_clear()
    try:
        with TestClient(MarketGateway(None)) as client:
            url = "/api/market-performance?season=2026-27&through=2026-11-14"
            first = client.get(url)
            assert first.status_code == 200
            assert first.headers["cache-control"] == "private, max-age=3477"
            clock[0] += 60
            assert client.get(url).status_code == 200
            assert reader.call_count == 1
            clock[0] = 363_601
            refreshed = client.get(url)
            assert refreshed.headers["cache-control"] == "private, max-age=3599"
            assert reader.call_count == 2
            assert client.get(url.replace("2026-11-14", "2026-11-13")).status_code == 200
            assert reader.call_count == 3
    finally:
        api._cached_report.cache_clear()


def test_market_errors_are_not_cached(monkeypatch):
    import boto3

    monkeypatch.setattr(api, "time", lambda: 360_123)
    monkeypatch.setattr(boto3, "client", lambda name: object())
    reader = Mock(
        side_effect=[ValueError("missing saved selection"), {"source": "market_performance"}]
    )
    monkeypatch.setattr(api, "read_market_performance", reader)
    api._cached_report.cache_clear()
    try:
        with TestClient(MarketGateway(None)) as client:
            url = "/api/market-performance?season=2026-27"
            failed = client.get(url)
            assert failed.status_code == 503
            assert failed.headers["cache-control"] == "no-store"
            assert client.get(url).status_code == 200
            assert client.get(url).status_code == 200
            assert reader.call_count == 2
    finally:
        api._cached_report.cache_clear()
