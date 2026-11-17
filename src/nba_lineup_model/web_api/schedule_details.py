"""Read-only quote and postgame adapters. No pipeline imports or source API calls."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC
from statistics import median

from botocore.exceptions import BotoCoreError, ClientError

from .market_calendar import saved_ats_result
from .schedule import BUCKET, PACIFIC, _instant, _number

SOURCE_ERRORS = (OSError, ValueError, KeyError, TypeError, BotoCoreError, ClientError)
BOX_FIELDS = ("uri", "sha256", "metadata_uri", "metadata_sha256")


class StaleSelection(ValueError):
    """A saved derivative refers to a different accepted baseline or box score."""


def _ref(reference):
    return {k: reference[k] for k in ("uri", "sha256")}


def _box_ref(reference):
    return {k: reference[k] for k in BOX_FIELDS}


def _identity(record, game):
    if record["game_id"] != game["game_id"] or _instant(record["scheduled_start_utc"]) != _instant(
        game["start_time"]
    ):
        raise ValueError("Saved detail game/tipoff mismatch")
    for side in ("home", "away"):
        row = record[side]
        if (
            row.get("team_id") != game[side + "_team_id"]
            or row.get("team", row.get("team_abbreviation")) != game[side + "_team"]
        ):
            raise ValueError("Saved detail team mismatch")


def _names(game):
    names = [
        "Los Angeles Clippers"
        if game[side + "_team"] == "LAC" and game.get(side + "_team_name") == "LA Clippers"
        else game.get(side + "_team_name")
        for side in ("home", "away")
    ]
    if not all(isinstance(n, str) and n for n in names) or names[0] == names[1]:
        raise ValueError("Official full team identities unavailable for odds mapping")
    return names


def _outcomes(outcomes, game):
    home, away = _names(game)
    if len(outcomes) != 2 or {v["name"] for v in outcomes} != {home, away}:
        raise ValueError("Odds outcomes do not match official team identities")
    return next(v for v in outcomes if v["name"] == home), next(
        v for v in outcomes if v["name"] == away
    )


def _probability(price):
    price = _number(price)
    if price == 0:
        raise ValueError("Invalid American moneyline")
    return -price / (100 - price) if price < 0 else 100 / (price + 100)


def _quote_value(kind, outcomes, game):
    home, away = _outcomes(outcomes, game)
    if kind == "spreads":
        h, a = _number(home["point"]), _number(away["point"])
        if abs(h + a) > 1e-9:
            raise ValueError("Spread sides disagree")
        return -h
    hp, ap = _probability(home["price"]), _probability(away["price"])
    return hp / (hp + ap)


def _validate_quote(quote, kind, game):
    cutoff = min(_instant(game["start_time"]), _instant(quote["provider_commence_time"]))
    if _instant(quote["observed_at"]) >= cutoff or _instant(quote["last_updated_at"]) >= cutoff:
        raise ValueError("Post-tipoff quote cannot be used as pregame odds")
    if kind != quote["market"]:
        raise ValueError("Wrong quote market")
    observed = _instant(quote["observed_at"])
    snapshot = quote["snapshot"]
    stamp = observed.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[a-f0-9]{64}", snapshot["sha256"]) or not re.fullmatch(
        r"s3://nba-gestalt-data/lake/raw/odds-api/season=\d{4}-\d{2}/snapshots/"
        + stamp
        + r"\.json",
        snapshot["uri"],
    ):
        raise ValueError("Invalid saved odds snapshot reference")
    _number(quote["value"])
    home, away = _outcomes(quote["outcomes"], game)
    if kind == "spreads" and (
        abs(_number(home["point"]) + _number(away["point"])) > 1e-9
        or quote["value"] != -home["point"]
    ):
        raise ValueError("Saved spread values disagree")
    if kind == "h2h" and not 0 <= quote["value"] <= 1:
        raise ValueError("Saved quote probability outside [0,1]")
    if kind == "spreads":
        _number(quote["outcomes"][0]["point"])
        _number(quote["outcomes"][1]["point"])
    else:
        for outcome in quote["outcomes"]:
            if _number(outcome["price"]) == 0:
                raise ValueError("Invalid saved moneyline")


def market_view(markets, game, label):
    """Use saved aggregates for historical markets; latest aggregates use pipeline conventions."""
    details, poly = {}, {}
    times = {}
    for kind in ("spreads", "h2h"):
        selection = markets.get(kind)
        if not selection:
            continue
        quotes = list(selection["books"])
        if selection.get("polymarket"):
            quotes.append(selection["polymarket"])
        seen = set()
        for q in quotes:
            _validate_quote(q, kind, game)
            key = q["bookmaker_key"]
            if key in seen:
                raise ValueError("Duplicate saved bookmaker quote")
            seen.add(key)
            home, away = _outcomes(q["outcomes"], game)
            target = (
                poly
                if key == "polymarket"
                else details.setdefault(
                    key,
                    {
                        "key": key,
                        "title": q["title"],
                        "home_spread": None,
                        "away_spread": None,
                        "home_moneyline": None,
                        "away_moneyline": None,
                        "home_win_probability": None,
                        "away_win_probability": None,
                    },
                )
            )
            if kind == "spreads":
                target.update(home_spread=home["point"], away_spread=away["point"])
            else:
                target.update(
                    home_moneyline=home["price"],
                    away_moneyline=away["price"],
                    home_win_probability=q["value"],
                    away_win_probability=1 - q["value"],
                )
            target[kind + "_captured_at"] = q["observed_at"]
        times[kind] = quotes[0]["observed_at"] if quotes else None
    spreads, h2h = markets.get("spreads") or {}, markets.get("h2h") or {}
    margin, probability = spreads.get("consensus"), h2h.get("consensus")
    if margin is not None:
        _number(margin)
    if probability is not None and not 0 <= _number(probability) <= 1:
        raise ValueError("Invalid saved consensus probability")
    market = {
        "provider": "Bookmaker consensus",
        "provider_key": "bookmaker_consensus",
        "selection": label,
        "home_spread": -margin if margin is not None else None,
        "away_spread": margin,
        "home_moneyline": None,
        "away_moneyline": None,
        "total": None,
        "home_win_probability": probability,
        "away_win_probability": 1 - probability if probability is not None else None,
        "bookmakers": list(details.values()),
        "bookmaker_keys": list(details),
        "bookmaker_count": len(details),
        "spread_bookmaker_count": len(spreads.get("books", [])),
        "moneyline_bookmaker_count": len(h2h.get("books", [])),
        "spread_captured_at": times.get("spreads"),
        "moneyline_captured_at": times.get("h2h"),
    }
    return market if details else None, (
        {"provider": "Polymarket", "selection": label, **poly} if poly else None
    )


def selected_event_view(event, game, snapshot):
    """Format exactly the event selected by the pipeline, without selecting a capture."""
    books = event["bookmakers"]
    if len({b["key"] for b in books}) != len(books):
        raise ValueError("Duplicate selected bookmaker")
    # Published #85 contract. No pipeline imports or external-provider calls.
    approved = {
        "draftkings",
        "fanduel",
        "betmgm",
        "betrivers",
        "hardrockbet",
        "betparx",
        "ballybet",
        "betonlineag",
        "polymarket",
        "espnbet",
    }
    cutoff = min(_instant(game["start_time"]), _instant(event["commence_time"]))
    markets = {}
    for kind in ("spreads", "h2h"):
        quotes = []
        for book in books:
            if book["key"] not in approved:
                continue
            rows = [m for m in book["markets"] if m["key"] == kind]
            if len(rows) > 1:
                raise ValueError("Duplicate selected bookmaker market")
            if not rows:
                continue
            market = rows[0]
            updated = market.get("last_update") or book["last_update"]
            if _instant(updated) >= cutoff:
                raise ValueError("Selected provider update is not pregame")
            if any(_number(row["price"]) == 0 for row in market["outcomes"]):
                raise ValueError("Invalid selected American odds")
            quotes.append(
                {
                    "bookmaker_key": book["key"],
                    "title": book.get("title") or book["key"],
                    "market": kind,
                    "value": _quote_value(kind, market["outcomes"], game),
                    "outcomes": market["outcomes"],
                    "last_updated_at": updated,
                    "observed_at": snapshot["collected_at"],
                    "provider_commence_time": event["commence_time"],
                    "snapshot": _ref(snapshot),
                }
            )
        others = [q for q in quotes if q["bookmaker_key"] != "polymarket"]
        markets[kind] = {
            "books": others,
            "polymarket": next((q for q in quotes if q["bookmaker_key"] == "polymarket"), None),
            "consensus": median(q["value"] for q in others) if others else None,
        }
    market, poly = market_view(markets, game, "Saved pregame")
    if market is None and poly is None:
        raise ValueError("Selected odds event has no approved quotes")
    return market, poly


class DetailsReader:
    def __init__(self, reader, season, day, now):
        self.reader, self.season, self.day, self.now = reader, season, day, now
        self.boxes = {}

    def game_odds(self, game):
        """Consume the pipeline's selection; never search or freeze raw captures."""
        import hashlib

        key = (
            f"lake/artifacts/game_odds/season={self.season}/"
            f"game_date_pacific={self.day}/game_id={game['game_id']}/latest.json"
        )
        result = self.reader._get(key, missing=True)
        if result is None:
            return None
        raw, _ = result
        pointer = json.loads(raw)
        if (
            type(pointer["schema_version"]) is not int
            or pointer["schema_version"] != 1
            or pointer["selection_rule"] != "latest_saved_pregame_event_v1"
            or pointer["season"] != self.season
            or pointer["game_date_pacific"] != self.day
            or pointer["game_id"] != game["game_id"]
            or _instant(pointer["scheduled_start_utc"]) != _instant(game["start_time"])
            or pointer["purpose"] not in {"pregame_selection", "historical_replay"}
            or type(pointer["frozen"]) is not bool
        ):
            raise ValueError("Invalid per-game odds selection identity/version")
        for side, name in zip(("home", "away"), _names(game), strict=True):
            if (
                pointer[side + "_team_id"] != game[side + "_team_id"]
                or pointer[side + "_team"] != name
            ):
                raise ValueError("Per-game odds team identity mismatch")
        selected_at = _instant(pointer["selected_at"])
        cutoff = _instant(pointer["cutoff_utc"])
        if selected_at > self.now or (self.now >= cutoff and not pointer["frozen"]):
            raise ValueError("Per-game odds selection is not finalized or is in the future")
        if pointer["frozen"]:
            frozen_at = _instant(pointer["frozen_at"])
            if not cutoff <= frozen_at <= self.now or frozen_at < selected_at:
                raise ValueError("Invalid per-game odds freeze time")
        elif pointer.get("frozen_at") is not None or selected_at >= cutoff:
            raise ValueError("Invalid unfrozen per-game odds metadata")
        snapshot = pointer["snapshot"]
        captured = _instant(snapshot["collected_at"])
        stamp = captured.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        snapshot_key = f"lake/raw/odds-api/season={self.season}/snapshots/{stamp}.json"
        if (
            snapshot["uri"] != f"s3://{BUCKET}/{snapshot_key}"
            or not re.fullmatch(r"[a-f0-9]{64}", snapshot["sha256"])
            or not captured <= selected_at <= self.now
            or captured >= cutoff
        ):
            raise ValueError("Invalid per-game odds immutable snapshot reference")
        doc = self.reader.immutable(snapshot_key, snapshot["sha256"])
        if (
            doc["schema_version"] != 1
            or doc["provider"] != "The Odds API"
            or doc["season"] != self.season
            or doc["sport_key"] != "basketball_nba"
            or doc["collected_at"] != snapshot["collected_at"]
        ):
            raise ValueError("Wrong selected odds snapshot identity")
        events = [e for e in doc["events"] if e["id"] == pointer["provider_event_id"]]
        if len(events) != 1:
            raise ValueError("Missing or duplicate selected odds event")
        event = events[0]
        home, away = _names(game)
        provider_start = _instant(event["commence_time"])
        if (
            event["home_team"] != home
            or event["away_team"] != away
            or event["sport_key"] != "basketball_nba"
            or str(provider_start.astimezone(PACIFIC).date()) != self.day
            or cutoff != min(_instant(game["start_time"]), provider_start)
        ):
            raise ValueError("Selected odds event identity/cutoff mismatch")
        reference = {**_ref(snapshot), "captured_at": snapshot["collected_at"]}
        market, poly = selected_event_view(event, game, snapshot)
        label = "Frozen pregame" if pointer["frozen"] else "Latest pregame"
        for view in (market, poly):
            if view is not None:
                view["selection"] = label
        return (
            market,
            poly,
            {
                **reference,
                "provider_event_id": pointer["provider_event_id"],
                "selection": {
                    "uri": f"s3://{BUCKET}/{key}",
                    "observed_sha256": hashlib.sha256(raw).hexdigest(),
                    "document": pointer,
                },
            },
        )

    def historical(self, games):
        root = (
            f"lake/artifacts/market_performance/season={self.season}/model_version=v1/"
            f"game_date_pacific={self.day}/"
        )
        pair = self.reader.selection(
            root + "latest.json",
            root,
            {
                "schema_version": 1,
                "season": self.season,
                "model_version": "v1",
                "game_date_pacific": self.day,
            },
            missing=True,
        )
        if pair is None:
            return None
        doc, ref = pair
        if doc["artifact_kind"] != "market_performance":
            raise ValueError("Wrong historical market artifact kind")
        ids = [g["game_id"] for g in doc["games"]]
        if len(set(ids)) != len(ids) or set(ids) != {g["game_id"] for g in games}:
            raise ValueError("Historical market artifact does not cover the official slate")
        return {g["game_id"]: g for g in doc["games"]}, ref

    def pinned_box(self, game, reference):
        reference = _box_ref(reference)
        key = (game["game_id"], *(reference[k] for k in BOX_FIELDS))
        if key not in self.boxes:
            self.boxes[key] = self.reader.pinned_final_box(self.season, game, reference)
        return self.boxes[key]

    def historical_game(self, game, historical):
        if historical is None:
            return None
        rows, ref = historical
        row = rows[game["game_id"]]
        _identity(row, game)
        markets = row["markets"]
        if (
            markets["provider"] != "The Odds API"
            or markets["selection_rule"] != "latest_pregame_polymarket_and_same_snapshot_books"
        ):
            raise ValueError("Unsupported historical market selection rule")
        market, poly = market_view(markets, game, "Saved pregame")
        return market, poly, ref

    def historical_grading(self, game, historical):
        if historical is None:
            return None
        self.historical_game(game, historical)
        rows, ref = historical
        row = rows[game["game_id"]]
        if row["inputs"]["prediction"] != game["sources"]["prediction"]:
            raise StaleSelection("Historical market report baseline changed")
        box, _ = self.pinned_box(game, row["inputs"]["final_box_score"])
        if any(
            row[side]["final_score"] != box[side + "Team"]["score"] for side in ("home", "away")
        ):
            raise ValueError("Historical market result does not match its pinned final box")
        comparison = row["comparisons"]["model"]
        audit = None
        if comparison is not None:
            if (
                game["projection"] is None
                or comparison["predicted_home_margin"] != game["projection"]["home_margin"]
                or comparison["home_win_probability"] != game["projection"]["home_win_probability"]
            ):
                raise StaleSelection("Historical grading baseline values changed")
            if type(comparison["winner_correct"]) is not bool:
                raise ValueError("Invalid saved winner grade")
            ats_result = saved_ats_result(comparison)
            audit = {
                "inputs": deepcopy(row["inputs"]),
                "recorded_outcome": {
                    side + "_score": box[side + "Team"]["score"] for side in ("home", "away")
                },
                "grading": {
                    "model_favorite_won": comparison["winner_correct"],
                    "market": {"model_ats_result": ats_result}
                    if ats_result in {"win", "loss", "push"}
                    else None,
                },
            }
        return (audit, ref) if audit is not None else None

    def actual_minutes(self, game):
        root = (
            f"lake/artifacts/game_predictions_actual_minutes/season={self.season}/model_version=v1/"
            f"game_date_pacific={self.day}/game_id={game['game_id']}/"
        )
        pair = self.reader.selection(
            root + "latest.json",
            root,
            {
                "schema_version": 1,
                "season": self.season,
                "model_version": "v1",
                "game_date_pacific": self.day,
                "game_id": game["game_id"],
            },
            missing=True,
        )
        if pair is None:
            return None
        doc, ref = pair
        _identity(doc, game)
        if (
            doc["artifact_kind"] != "game_prediction_actual_minutes"
            or doc["purpose"] != "postgame_actual_minutes_counterfactual"
        ):
            raise ValueError("Wrong postgame artifact purpose")
        if (
            doc["inputs"]["prediction"] != game["sources"]["prediction"]
            or game["projection"] is None
        ):
            raise StaleSelection("Actual-minutes baseline prediction changed")
        if doc["baseline_prediction_purpose"] != game["prediction_purpose"]:
            raise StaleSelection("Actual-minutes baseline purpose changed")
        for saved, shown in [
            ("original_predicted_home_margin", "home_margin"),
            ("original_home_win_probability", "home_win_probability"),
        ]:
            if doc[saved] != game["projection"][shown]:
                raise StaleSelection("Actual-minutes baseline values changed")
        box, _ = self.pinned_box(game, doc["inputs"]["box_score"])
        if any(doc["actual_" + s + "_score"] != box[s + "Team"]["score"] for s in ("home", "away")):
            raise ValueError("Counterfactual final scores differ from its pinned box")
        _instant(doc["generated_at_utc"])
        margin = _number(doc["counterfactual_home_margin"])
        probability = _number(doc["counterfactual_home_win_probability"])
        if not 0 <= probability <= 1:
            raise ValueError("Invalid counterfactual probability")
        ledger = deepcopy(game["projection"]["ledger"])
        components = doc["components"]
        ledger.update(
            artifact_kind="retrospective_actual_minutes_attribution",
            home_margin=margin,
            fair_home_spread=-margin,
            team_strength_difference=components["home_player_total"]
            - components["away_player_total"],
            home_court_adjustment=_number(components["hca"]),
            back_to_back={
                "away_team_is_back_to_back": components["away_is_b2b"],
                "home_team_is_back_to_back": components["home_is_b2b"],
                "coefficient": _number(components["b2b_coefficient"]),
                "adjustment": _number(components["b2b_adjustment"]),
            },
        )
        actual = {}
        seen = set()
        for side in ("home", "away"):
            team = doc[side]
            ledger[side] = {
                "team": team["team"],
                "actual_minutes": _number(team["total_actual_minutes"]),
                "projected_minutes": _number(team["total_projected_minutes"]),
                "team_strength": _number(team["actual_player_total"]),
            }
            actual[side] = []
            for p in team["players"]:
                pid = p["player_id"]
                if type(pid) is not int or pid <= 0 or pid in seen:
                    raise ValueError("Invalid/duplicate actual player identity")
                seen.add(pid)
                values = {
                    k: _number(p[k])
                    for k in (
                        "actual_minutes",
                        "projected_minutes",
                        "nail",
                        "projected_rating_contribution",
                        "actual_rating_contribution",
                        "contribution_delta",
                    )
                }
                if values["actual_minutes"] < 0:
                    raise ValueError("Negative actual minutes")
                actual[side].append(
                    {
                        "player_id": pid,
                        "player_name": p["name"],
                        "actual_minutes": values["actual_minutes"],
                        "frozen_nail": values["nail"],
                        "actual_contribution": values["actual_rating_contribution"],
                        "projected_minutes": values["projected_minutes"],
                        "projected_contribution": values["projected_rating_contribution"],
                        "contribution_delta": values["contribution_delta"],
                        "in_baseline_prediction": p["in_baseline_prediction"],
                        "rating_source": p["rating_source"],
                        "rating_matched": p["rating_source"] != "replacement",
                    }
                )
        for side in ("home", "away"):
            baseline = {p["player_id"]: p for p in game["projected_minutes"][side]}
            selected_baseline = {
                p["player_id"] for p in actual[side] if p["in_baseline_prediction"]
            }
            if selected_baseline != set(baseline):
                raise StaleSelection("Actual-minutes baseline player coverage changed")
            for row in actual[side]:
                before = baseline.get(row["player_id"])
                if before and (
                    row["projected_minutes"] != before["projected_minutes"]
                    or row["frozen_nail"] != before["projected_nail"]
                    or row["projected_contribution"] != before["expected_contribution"]
                ):
                    raise StaleSelection("Actual-minutes baseline player values changed")
        return {
            "computed_at": doc["generated_at_utc"],
            "method": doc["purpose"],
            "baseline_prediction_purpose": doc["baseline_prediction_purpose"],
            "home_margin": margin,
            "home_win_probability": probability,
            "actual_minutes": actual,
            "ledger": ledger,
            "inputs": doc["inputs"],
            "recorded_outcome": {
                "home_score": doc["actual_home_score"],
                "away_score": doc["actual_away_score"],
            },
        }, ref

    @staticmethod
    def failure(game, component, error):
        import logging

        state = "stale" if isinstance(error, StaleSelection) else "invalid"
        game["availability"][component] = state
        game.setdefault("detail_errors", {})[component] = (
            "Saved data refers to a different baseline prediction."
            if state == "stale"
            else "Saved data failed validation."
        )
        logging.getLogger(__name__).warning("Schedule %s %s: %s", game["game_id"], component, error)

    def enrich_game(self, game, historical):
        game["availability"].update(
            market="missing",
            postgame_details="missing" if game["status"]["state"] == "post" else "not_applicable",
            grading="missing" if game["status"]["state"] == "post" else "not_applicable",
        )
        try:
            selected = self.game_odds(game)
            # Existing published reports retain original odds/provenance for
            # historical games without pointers. Invalid pointers never fall through.
            if selected is None:
                if isinstance(historical, Exception):
                    raise historical
                selected = self.historical_game(game, historical)
            if selected is not None:
                game["market"], game["prediction_market"], game["sources"]["market"] = selected[:3]
                game["availability"]["market"] = (
                    "available" if game["market"] or game["prediction_market"] else "missing"
                )
        except SOURCE_ERRORS as error:
            self.failure(game, "market", error)
        if game["status"]["state"] == "post":
            try:
                if isinstance(historical, Exception):
                    raise historical
                selected = self.historical_grading(game, historical)
                if selected is not None:
                    game["final_audit"], game["sources"]["grading"] = selected
                    game["availability"]["grading"] = "available"
            except SOURCE_ERRORS as error:
                self.failure(game, "grading", error)
            try:
                selected = self.actual_minutes(game)
                if selected is not None:
                    game["retrospective"], game["sources"]["postgame_details"] = selected
                    game["availability"]["postgame_details"] = "available"
            except SOURCE_ERRORS as error:
                self.failure(game, "postgame_details", error)
        return game

    def enrich(self, games):
        if not games:
            return

        def safely(call):
            try:
                return call()
            except SOURCE_ERRORS as error:
                return error

        historical = (
            safely(lambda: self.historical(games))
            if any(_instant(g["start_time"]) <= self.now for g in games)
            else None
        )
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda g: self.enrich_game(g, historical), games))
