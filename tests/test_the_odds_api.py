from __future__ import annotations

import json
import urllib.parse
from datetime import UTC, datetime
from email.message import Message

import pytest

import nba_lineup_model.web_api.the_odds_api as odds_api
from nba_lineup_model.web_api.the_odds_api import (
    OddsApiQuotaGuard,
    collect_nba_odds_snapshot,
    estimated_request_cost,
    normalize_bookmaker_lines,
    normalize_market_consensus,
)


def _events() -> list[dict[str, object]]:
    return [
        {
            "id": "nba-bos-nyk",
            "commence_time": "2026-10-27T23:30:00Z",
            "home_team": "New York Knicks",
            "away_team": "Boston Celtics",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "spreads",
                            "last_update": "2026-10-27T18:00:00Z",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -110, "point": 2.5},
                                {"name": "New York Knicks", "price": -110, "point": -2.5},
                            ],
                        },
                        {
                            "key": "h2h",
                            "last_update": "2026-10-27T18:01:00Z",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": 120},
                                {"name": "New York Knicks", "price": -142},
                            ],
                        },
                    ],
                }
            ],
        }
    ]


class _Response:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.headers = Message()
        self.headers["x-requests-used"] = "12"
        self.headers["x-requests-remaining"] = "488"
        self.headers["x-requests-last"] = "2"

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_collect_snapshot_archives_quota_without_api_key(tmp_path, monkeypatch) -> None:
    requested_urls: list[str] = []

    def fake_urlopen(request, timeout):
        assert timeout == 20
        requested_urls.append(request.full_url)
        return _Response(_events())

    monkeypatch.setattr(odds_api.urllib.request, "urlopen", fake_urlopen)
    snapshot = collect_nba_odds_snapshot(
        api_key="secret-key",
        state_dir=tmp_path,
        s3_uri=None,
        now=datetime(2026, 10, 27, 18, 5, tzinfo=UTC),
    )

    assert len(snapshot["events"]) == 1
    assert snapshot["quota"] == {
        "requests_used": 12,
        "requests_remaining": 488,
        "request_cost": 2,
        "minimum_remaining_credits": 50,
    }
    latest = (tmp_path / "odds-api" / "latest.json").read_text()
    assert "secret-key" not in latest
    assert "secret-key" in requested_urls[0]
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(requested_urls[0]).query)
    assert query["bookmakers"] == [
        "draftkings,fanduel,betmgm,betrivers,espnbet,hardrockbet,betparx,ballybet,betonlineag,polymarket"
    ]
    assert not list((tmp_path / "pending" / "odds-api").glob("*.json"))


def test_normalize_draftkings_lines_by_game_date() -> None:
    snapshot = {
        "collected_at": "2026-10-27T18:05:00Z",
        "events": _events(),
    }

    lines = normalize_bookmaker_lines(snapshot)

    assert lines == {
        "2026-10-27|BOS@NYK": {
            "provider": "DraftKings",
            "provider_key": "draftkings",
            "source": "The Odds API",
            "source_event_id": "nba-bos-nyk",
            "captured_at": "2026-10-27T18:05:00Z",
            "last_update": "2026-10-27T18:01:00Z",
            "home_spread": -2.5,
            "away_spread": 2.5,
            "home_spread_open": None,
            "away_spread_open": None,
            "home_moneyline": -142,
            "away_moneyline": 120,
            "home_moneyline_open": None,
            "away_moneyline_open": None,
            "total": None,
            "total_open": None,
        }
    }


def test_normalize_market_consensus_uses_median_spread_and_devigged_books() -> None:
    event = _events()[0]
    event["bookmakers"] = [
        *event["bookmakers"],
        {
            "key": "fanduel",
            "title": "FanDuel",
            "markets": [
                {
                    "key": "spreads",
                    "last_update": "2026-10-27T18:02:00Z",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": -110, "point": 3.5},
                        {"name": "New York Knicks", "price": -110, "point": -3.5},
                    ],
                },
                {
                    "key": "h2h",
                    "last_update": "2026-10-27T18:03:00Z",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": 125},
                        {"name": "New York Knicks", "price": -150},
                    ],
                },
            ],
        },
        {
            "key": "betmgm",
            "title": "BetMGM",
            "markets": [
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": -110, "point": 2.5},
                        {"name": "New York Knicks", "price": -110, "point": -2.5},
                    ],
                },
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": 124},
                        {"name": "New York Knicks", "price": -146},
                    ],
                },
            ],
        },
        {
            "key": "betrivers",
            "title": "BetRivers",
            "markets": [
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": -110, "point": 3.5},
                        {"name": "New York Knicks", "price": -110, "point": -3.5},
                    ],
                },
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": 128},
                        {"name": "New York Knicks", "price": -155},
                    ],
                },
            ],
        },
        {
            "key": "polymarket",
            "title": "Polymarket",
            "markets": [
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": -110, "point": 10.5},
                        {"name": "New York Knicks", "price": -110, "point": -10.5},
                    ],
                },
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": 400},
                        {"name": "New York Knicks", "price": -500},
                    ],
                },
            ],
        },
    ]

    lines = normalize_market_consensus(
        {"collected_at": "2026-10-27T18:05:00Z", "events": [event]}
    )

    consensus = lines["2026-10-27|BOS@NYK"]
    assert consensus["provider"] == "Market consensus"
    assert consensus["provider_key"] == "market_consensus"
    assert consensus["bookmaker_keys"] == [
        "draftkings",
        "fanduel",
        "betmgm",
        "betrivers",
    ]
    assert consensus["bookmaker_count"] == 4
    assert consensus["bookmakers"][0] == {
        "key": "draftkings",
        "title": "DraftKings",
        "home_spread": -2.5,
        "away_spread": 2.5,
        "home_moneyline": -142,
        "away_moneyline": 120,
        "home_win_probability": pytest.approx(0.5634920634920635),
        "away_win_probability": pytest.approx(0.4365079365079365),
        "last_update": "2026-10-27T18:01:00Z",
    }
    assert consensus["home_spread"] == pytest.approx(-3.0)
    assert consensus["away_spread"] == pytest.approx(3.0)
    assert consensus["home_moneyline"] is None
    assert consensus["away_moneyline"] is None
    assert consensus["home_win_probability"] == pytest.approx(0.57238465)
    assert consensus["away_win_probability"] == pytest.approx(0.42761535)


def test_quota_guard_preserves_reserve_before_request(tmp_path, monkeypatch) -> None:
    quota_path = tmp_path / "odds-api" / "quota.json"
    quota_path.parent.mkdir(parents=True)
    quota_path.write_text(
        json.dumps(
            {
                "collected_at": "2026-10-27T18:00:00Z",
                "requests_remaining": 51,
            }
        )
    )
    monkeypatch.setattr(
        odds_api.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("quota guard should prevent HTTP request"),
    )

    with pytest.raises(OddsApiQuotaGuard, match="preserve the configured quota reserve"):
        collect_nba_odds_snapshot(
            api_key="secret-key",
            state_dir=tmp_path,
            s3_uri=None,
            now=datetime(2026, 10, 27, 19, 0, tzinfo=UTC),
        )


def test_retry_reuses_staged_response_instead_of_refetching(tmp_path, monkeypatch) -> None:
    fetches = 0

    def fake_fetch(**_kwargs):
        nonlocal fetches
        fetches += 1
        return _events(), {
            "x-requests-used": "2",
            "x-requests-remaining": "498",
            "x-requests-last": "2",
        }

    uploads = 0

    def flaky_upload(_source, _destination):
        nonlocal uploads
        uploads += 1
        if uploads == 1:
            raise RuntimeError("temporary S3 failure")

    monkeypatch.setattr(odds_api, "_fetch_odds", fake_fetch)
    monkeypatch.setattr(odds_api, "_aws_s3_cp_exclusive", flaky_upload)
    monkeypatch.setattr(odds_api, "_aws_s3_cp", lambda *_args: None)
    kwargs = {
        "api_key": "secret-key",
        "state_dir": tmp_path,
        "s3_uri": "s3://archive/schedule",
        "request_id": "same-dagster-run",
        "now": datetime(2026, 10, 27, 19, 0, tzinfo=UTC),
    }
    with pytest.raises(RuntimeError, match="temporary S3 failure"):
        collect_nba_odds_snapshot(**kwargs)

    collect_nba_odds_snapshot(**kwargs)

    assert fetches == 1
    assert uploads == 2


def test_default_request_cost_is_two_credits() -> None:
    assert len(odds_api.DEFAULT_BOOKMAKERS) == 10
    assert odds_api.CONSENSUS_BOOKMAKERS == odds_api.DEFAULT_BOOKMAKERS[:-1]
    assert odds_api.DEFAULT_BOOKMAKERS[-1] == "polymarket"
    assert estimated_request_cost() == 2
