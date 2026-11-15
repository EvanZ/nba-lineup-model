"""Synthetic accepted upstream artifacts and an in-memory S3 service."""
from datetime import UTC, datetime

import pytest

from minutes.test_projection import example
from ratings.conftest import S3
from gestalt_pipelines.minutes.utils import build_game_projection
from gestalt_pipelines.predictions.resources import GamePredictionArchive
from gestalt_pipelines.predictions.utils import build_prediction

DAY = "2026-10-21"
NOW = datetime(2026, 10, 21, 18, tzinfo=UTC)


@pytest.fixture
def data():
    source = example()
    allocation = build_game_projection(**source)
    players = [p for t in allocation["teams"].values() for p in t["players"]]
    ratings = {"season": "2026-27", "game_date_pacific": "2026-10-20",
               "observed_through_game_date_pacific": "2026-10-20",
               "date_contract": "completed_game_date_v1", "effective_from_game_date_pacific": DAY,
               "players": [{"player_id": p["player_id"], "nail": p["player_id"]/19} for p in players],
               "hca": {"estimate": 1.8073343269384732},
               "settings": {"fixed_back_to_back": -1.5087106592522943, "replacement_prior": -5.109488431706889},
               "inputs": {"priors": {"uri": "s3://nba-gestalt-data/test/priors.json", "sha256": "c"*64}}}
    return dict(game=source["game"], day=DAY, season="2026-27", ratings=ratings,
                minutes=allocation, priors=None, report={"published_at": "2026-10-21T17:00:00Z",
                "source_url": "https://ak-static.cms.nba.com/referee/injury/Injury-Report.pdf",
                "pdf_uri": "s3://nba-gestalt-data/test/report.pdf"},
                b2b={"home": True, "away": False},
                inputs={"player_ratings": {"uri": "s3://nba-gestalt-data/test/ratings.json", "sha256": "a"*64},
                        "game_minutes": {"uri": "s3://nba-gestalt-data/test/minutes.json", "sha256": "b"*64},
                        "schedule": {"uri": "s3://nba-gestalt-data/test/schedule.json", "sha256": "d"*64}},
                generated_at=NOW, purpose="pregame_projection")


@pytest.fixture
def record(data):
    return build_prediction(**data)


@pytest.fixture
def archive(monkeypatch):
    s3 = S3()
    monkeypatch.setattr("gestalt_pipelines.predictions.resources.archive.boto3.client", lambda _: s3)
    return GamePredictionArchive(season="2026-27"), s3


@pytest.fixture
def clock(monkeypatch):
    class Clock(datetime):
        value = NOW
        @classmethod
        def now(cls, tz=None):
            return cls.value.astimezone(tz) if tz else cls.value.replace(tzinfo=None)
    monkeypatch.setattr("gestalt_pipelines.predictions.resources.archive.datetime", Clock)
    monkeypatch.setattr("gestalt_pipelines.predictions.assets.predict.datetime", Clock)
    return Clock
