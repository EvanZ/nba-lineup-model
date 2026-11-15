import json
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from types import SimpleNamespace

import pytest
from dagster import Definitions
from ratings.conftest import S3
from gestalt_pipelines.box_scores.resources import NBABoxScoreArchive
from gestalt_pipelines.market_performance.assets import market_performance_by_date
from gestalt_pipelines.market_performance.jobs import evaluate_market_performance
from gestalt_pipelines.market_performance.resources import MarketPerformanceArchive
from gestalt_pipelines.market_performance.utils import json_bytes
from gestalt_pipelines.predictions.resources import GamePredictionArchive
from gestalt_pipelines.resources import NBAScheduleResource

DAY = "2026-10-21"
START = datetime(2026, 10, 22, 0, tzinfo=UTC)


class Store(S3):
    def get_paginator(self, operation):
        return SimpleNamespace(paginate=lambda **kw: [{"Contents": [
            {"Key": key, "ETag": sha256(raw).hexdigest()} for key, raw in self.objects.items() if key.startswith(kw["Prefix"])]}])

    def put_object(self, **kwargs):
        assert kwargs["Key"].startswith(("lake/artifacts/market_performance/", "lake/artifacts/errors/pipeline=market_performance/"))
        return super().put_object(**kwargs)


def snapshot(collected="2026-10-21T23:20:00Z"):
    books = []
    for key, spread in (("polymarket", -7), ("fanduel", -4.5), ("espnbet", -5.5)):
        books.append({"key": key, "title": key, "last_update": "2026-10-21T23:15:00Z", "markets": [
            {"key": "h2h", "outcomes": [{"name": "Boston Celtics", "price": -200 if key == "polymarket" else -150},
                                          {"name": "Philadelphia 76ers", "price": 150}]},
            {"key": "spreads", "outcomes": [{"name": "Boston Celtics", "price": -110, "point": spread},
                                              {"name": "Philadelphia 76ers", "price": -110, "point": -spread}]},
        ]})
    return {"provider": "The Odds API", "sport_key": "basketball_nba", "season": "2026-27", "collected_at": collected,
            "events": [{"id": "example", "home_team": "Boston Celtics", "away_team": "Philadelphia 76ers",
                        "commence_time": START.isoformat(), "bookmakers": books}]}


@pytest.fixture
def setup(monkeypatch):
    s3 = Store()
    monkeypatch.setattr("boto3.client", lambda *a, **kw: s3)
    game = {"game_id": "0022600001", "game_time_utc": START, "home_team_id": 1610612738,
            "away_team_id": 1610612755, "home_team_tricode": "BOS", "away_team_tricode": "PHI"}
    refs = {name: {"uri": f"s3://nba-gestalt-data/test/{name}.json", "sha256": letter*64}
            for name, letter in (("prediction", "a"), ("counter", "b"), ("box", "c"), ("schedule", "d"))}
    prediction = {"purpose": "historical_replay", "predicted_home_margin": 6, "home_win_probability": .681083,
                  "scheduled_start_utc": START.isoformat(), "home": {"team_id": game["home_team_id"]}, "away": {"team_id": game["away_team_id"]}}
    counter = {"counterfactual_home_margin": 7, "counterfactual_home_win_probability": .711333, "actual_home_margin": 8,
               "inputs": {"prediction": refs["prediction"], "box_score": refs["box"]}}
    box = {"gameStatus": 3, "homeTeam": {"score": 108, "teamName": "Celtics"}, "awayTeam": {"score": 100, "teamName": "76ers"}}
    raw_schedule = json_bytes({"leagueSchedule": {"gameDates": [{"games": [{"gameId": game["game_id"],
        "homeTeam": {"teamName": "Celtics"}, "awayTeam": {"teamName": "76ers"}}]}]}})
    data = {"games": [game], "prediction": prediction, "counter": counter, "box": box, "refs": refs, "missing": None}
    monkeypatch.setattr(NBAScheduleResource, "games_for_date", lambda *a, **kw: (data["games"], refs["schedule"]))
    monkeypatch.setattr(NBAScheduleResource, "load", lambda *a, **kw: (raw_schedule, refs["schedule"]))
    monkeypatch.setattr(NBABoxScoreArchive, "read_saved", lambda *a: (json_bytes({"game": box}), refs["box"]))
    def selection(self, day, gid, actual_minutes=False):
        name = "counter" if actual_minutes else "prediction"
        return (None, None) if data["missing"] == name else (refs[name], "etag")
    monkeypatch.setattr(GamePredictionArchive, "selection", selection)
    monkeypatch.setattr(GamePredictionArchive, "load", lambda self, p, actual_minutes=False: deepcopy(data["counter"] if actual_minutes else data["prediction"]))
    key = "lake/raw/odds-api/season=2026-27/snapshots/20261021T232000Z.json"
    s3.objects[key] = json_bytes(snapshot())
    store = MarketPerformanceArchive(season="2026-27")
    definitions = Definitions(assets=[market_performance_by_date], jobs=[evaluate_market_performance], resources={
        "market_performance_archive": store, "game_prediction_archive": GamePredictionArchive(season="2026-27"),
        "nba_box_score_archive": NBABoxScoreArchive(), "nba_schedule": NBAScheduleResource(season="2026-27")})
    return definitions.resolve_job_def("evaluate_market_performance"), store, s3, data
