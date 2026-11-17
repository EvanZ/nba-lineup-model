"""The pre-tipoff Polymarket selection and same-snapshot book median used by the audited reports."""
import math
from statistics import median

from ...utils import PACIFIC
from ...odds.utils import instant, matches as _matches


def probability(price):
    if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price == 0:
        raise ValueError("Invalid American moneyline.")
    return -price / (100-price) if price < 0 else 100 / (price+100)


def _quote(book, kind, event, collected, cutoff, reference):
    markets = [m for m in book.get("markets", []) if m["key"] == kind]
    if len(markets) > 1:
        raise ValueError("Duplicate bookmaker market.")
    if not markets:
        return None
    market = markets[0]
    updated = instant(market.get("last_update") or book["last_update"])
    if updated >= cutoff:
        return None
    outcomes = {item["name"]: item for item in market["outcomes"]}
    if len(market["outcomes"]) != 2 or set(outcomes) != {event["home_team"], event["away_team"]}:
        return None
    home, away = outcomes[event["home_team"]], outcomes[event["away_team"]]
    if kind == "h2h":
        hp, ap = probability(home["price"]), probability(away["price"])
        value = hp / (hp+ap)
    else:
        if (not all(isinstance(row.get("point"), (int, float)) and math.isfinite(row["point"]) for row in (home, away))
                or not math.isclose(home["point"], -away["point"], abs_tol=1e-9)):
            raise ValueError("Market spread sides differ or are invalid.")
        value = -home["point"]
    return {"bookmaker_key": book["key"], "title": book.get("title"), "market": kind, "value": value,
            "observed_at": collected.isoformat(), "last_updated_at": updated.isoformat(),
            "provider_commence_time": event["commence_time"], "snapshot": reference,
            "outcomes": [{k: row[k] for k in ("name", "price", "point") if k in row} for row in (home, away)]}


def select_markets(games, names, snapshots):
    """Consume snapshots newest first; each market is anchored to its latest eligible Polymarket quote."""
    selected = {game["game_id"]: {} for game in games}
    for snapshot, reference in snapshots:
        collected = instant(snapshot["collected_at"])
        for game in games:
            gid = game["game_id"]
            if len(selected[gid]) == 2 or collected >= game["game_time_utc"]:
                continue
            candidates = [event for event in snapshot["events"]
                if instant(event["commence_time"]).astimezone(PACIFIC).date() == game["game_time_utc"].astimezone(PACIFIC).date()
                and _matches(event["home_team"], names[gid]["home"])
                and _matches(event["away_team"], names[gid]["away"])]
            if len(candidates) > 1:
                raise ValueError(f"Ambiguous Odds API event for {gid}.")
            if not candidates:
                continue
            event = candidates[0]
            cutoff = min(game["game_time_utc"], instant(event["commence_time"]))
            if collected >= cutoff:
                continue
            books = event.get("bookmakers", [])
            keys = [book["key"] for book in books]
            if len(keys) != len(set(keys)):
                raise ValueError("Duplicate bookmaker in an Odds API event.")
            poly = next((book for book in books if book["key"] == "polymarket"), None)
            if poly is None:
                continue
            for kind in ("h2h", "spreads"):
                if kind in selected[gid]:
                    continue
                quote = _quote(poly, kind, event, collected, cutoff, reference)
                if quote is None:
                    continue
                others = [value for book in books if book["key"] != "polymarket"
                          if (value := _quote(book, kind, event, collected, cutoff, reference)) is not None]
                selected[gid][kind] = {"polymarket": quote, "books": others,
                                      "consensus": median(value["value"] for value in others) if others else None}
        if all(len(markets) == 2 for markets in selected.values()):
            break
    return selected
