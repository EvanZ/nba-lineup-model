"""Regression coverage for dated moves sourced from official transaction rows."""

import io
import json
from datetime import date
from types import SimpleNamespace

from fastapi.testclient import TestClient

from nba_lineup_model.web_api import saved_roster_moves as moves
from nba_lineup_model.web_api.gateway import MarketGateway


def test_trade_classification_and_date_refresh(monkeypatch):
    day = date(2026, 11, 8)
    key = f"{moves.TRANSACTION_ROOT}game_date_pacific={day}/latest.json"
    transaction = {
        "PLAYER_ID": 1642363.0,
        "TEAM_ID": 1610612760.0,
        "PLAYER_SLUG": "nique-clifford",
        "TRANSACTION_DATE": f"{day}T00:00:00",
        "TRANSACTION_DESCRIPTION": "Oklahoma City received Nique Clifford from Sacramento.",
        "Transaction_Type": "Trade",
    }

    class TransactionStore:
        etag = '"original"'
        reads = 0
        rows = [transaction]

        def get_paginator(self, method):
            assert method == "list_objects_v2"
            return self

        def paginate(self, **params):
            assert params == {"Bucket": moves.BUCKET, "Prefix": moves.TRANSACTION_ROOT}
            return iter([{"Contents": [{"Key": key, "ETag": self.etag}]}])

        def get_object(self, **params):
            assert params == {"Bucket": moves.BUCKET, "Key": key, "IfMatch": self.etag}
            self.reads += 1
            document = {
                "season": moves.SEASON,
                "game_date_pacific": str(day),
                "source_url": moves.TRANSACTION_SOURCE,
                "transactions": self.rows,
            }
            return {"Body": io.BytesIO(json.dumps(document).encode())}

    store = TransactionStore()
    teams = {1610612758: "SAC", 1610612760: "OKC"}
    monkeypatch.setattr(moves, "_reader", lambda: SimpleNamespace(client=store))
    monkeypatch.setattr(moves, "offseason", lambda: ({}, teams, {"1642363": "Nique Clifford"}, {}))
    monkeypatch.setattr(moves, "OPENING_DATE", day)
    monkeypatch.setattr(
        moves,
        "accepted_roster",
        lambda _, roster_day: {str(1610612760 if roster_day == day else 1610612758): [1642363]},
    )
    moves.in_season.cache_clear()
    moves.transactions_on_day.cache_clear()
    try:
        clock = iter((3601, 3601, 7201, 7201, 10801, 10801))
        monkeypatch.setattr(moves, "time", lambda: next(clock))
        client = TestClient(MarketGateway(None))
        url = "/api/roster-moves?period=in-season"
        original = client.get(url)
        assert original.status_code == 200
        (trade,) = original.json()["transactions"]
        assert (trade["move_type"], trade["source_team"], trade["target_team"]) == (
            "trade",
            "SAC",
            "OKC",
        )
        assert store.reads == 1

        assert client.get(url).json() == original.json()
        assert store.reads == 1

        store.etag = '"updated"'
        store.rows = [transaction, {**transaction, "Transaction_Type": "Waive"}]
        trade, waiver = client.get(url).json()["transactions"]
        assert (waiver["move_type"], waiver["source_team"], waiver["target_team"]) == (
            "waiver",
            "OKC",
            None,
        )
        assert trade["event_id"] != waiver["event_id"]
        assert store.reads == 2
    finally:
        moves.in_season.cache_clear()
        moves.transactions_on_day.cache_clear()


def test_trade_without_accepted_destination_has_no_origin():
    teams = {1: "SAC", 2: "OKC"}
    assert moves.trade_origin(2, 123, {"1": [123]}, {}, teams) is None
