"""Choose one real pregame event; leave all prices in the original capture."""
import math
from datetime import datetime

from .. import BOOKMAKERS, MARKETS
from ...utils import PACIFIC


def instant(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Odds timestamps must include a timezone.")
    return result


def matches(name, nickname):
    return name.casefold() == nickname.casefold() or name.casefold().endswith(" " + nickname.casefold())


def eligible_event(snapshot, game, names):
    collected = instant(snapshot["collected_at"])
    if collected >= game["game_time_utc"]:
        return None
    candidates = [event for event in snapshot["events"]
        if instant(event["commence_time"]).astimezone(PACIFIC).date() == game["game_time_utc"].astimezone(PACIFIC).date()
        and matches(event["home_team"], names["home"])
        and matches(event["away_team"], names["away"])]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous Odds API event for {game['game_id']}.")
    if not candidates:
        return None
    event = candidates[0]
    cutoff = min(game["game_time_utc"], instant(event["commence_time"]))
    if collected >= cutoff:
        return None
    if not event.get("id") or event.get("sport_key") != "basketball_nba":
        raise ValueError("The selected event has an invalid identity or sport.")
    books = event.get("bookmakers", [])
    if len({book["key"] for book in books}) != len(books):
        raise ValueError("Duplicate bookmaker in the selected event.")
    usable = False
    for book in books:
        if book["key"] not in BOOKMAKERS:
            continue
        markets = [market for market in book.get("markets", []) if market["key"] in MARKETS]
        if len({m["key"] for m in markets}) != len(markets):
            raise ValueError("Duplicate market in the selected event.")
        for market in markets:
            if instant(market.get("last_update") or book["last_update"]) >= cutoff:
                return None
            rows = market["outcomes"]
            if len(rows) != 2 or {row["name"] for row in rows} != {event["home_team"], event["away_team"]}:
                raise ValueError("The selected market does not contain the two scheduled teams.")
            def number(value):
                return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
            if any(not number(row.get("price")) or row["price"] == 0 for row in rows):
                raise ValueError("Invalid American odds in the selected event.")
            if market["key"] == "spreads" and (any(not number(row.get("point")) for row in rows)
                    or not math.isclose(rows[0]["point"], -rows[1]["point"], abs_tol=1e-9)):
                raise ValueError("Invalid spread sides in the selected event.")
            usable = True
    return (event, cutoff) if usable else None
