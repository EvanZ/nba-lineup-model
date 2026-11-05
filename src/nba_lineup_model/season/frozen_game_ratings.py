"""Resolve a game's exact rating source within the release's accepted model chain."""
from datetime import date, datetime
from pathlib import Path
import math
import re
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq

from nba_lineup_model.season.daily_artifact_common import file_sha256
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact


def load_frozen_game_ratings(selected: dict, *, model_bundle: Path,
                            model_sha256: str, preseason_rankings: Path) -> dict[int, float]:
    provenance = selected["projection"]["input_provenance"]
    source = str(provenance.get("rating_source_uri", ""))
    tip = datetime.fromisoformat(str(selected["start_time"]).replace("Z", "+00:00"))
    if tip.tzinfo is None:
        raise ValueError("Frozen game start time has no timezone")
    game_date = tip.astimezone(ZoneInfo("America/Los_Angeles")).date()
    evidence_date = date.fromisoformat(str(provenance["rating_as_of"]))
    if evidence_date >= game_date:
        raise ValueError("Frozen ratings contain same-day or future evidence")
    daily = re.fullmatch(r"artifact://daily-model/([0-9a-f]{64})/player_ratings", source)
    preseason = re.fullmatch(r"artifact://preseason-rankings/([0-9a-f]{64})", source)
    if daily:
        wanted = daily.group(1)
        current = Path(model_bundle)
        root = current.parents[2]
        expected = model_sha256
        seen = set()
        while expected:
            if expected in seen:
                raise ValueError("Accepted model chain contains a cycle")
            seen.add(expected)
            manifest = verify_daily_model_artifact(current)
            if manifest["bundle_sha256"] != expected:
                raise ValueError("Accepted model chain reference does not match its contents")
            identity = manifest["identity"]
            if expected == wanted:
                if identity["game_date_pacific"] != evidence_date.isoformat():
                    raise ValueError("Frozen rating evidence date differs from its model")
                component = identity["components"]["player_ratings"]
                path = current / component["path"]
                if file_sha256(path) != component["sha256"]:
                    raise ValueError("Frozen player ratings checksum mismatch")
                column = "updated_nail"
                break
            expected = identity["parent_model_sha256"]
            if expected:
                current = (root / f"season={identity['season']}"
                           / f"game_date_pacific={identity['parent_game_date_pacific']}"
                           / f"model_sha256={expected}")
        else:
            raise ValueError("Game's exact model is absent from the accepted chain")
    elif preseason:
        path = Path(preseason_rankings)
        if file_sha256(path) != preseason.group(1):
            raise ValueError("Game's exact preseason ratings are unavailable")
        column = "rapm"
    else:
        raise ValueError("Game has no supported immutable rating source")
    table = pq.ParquetFile(path).read()
    if "player_id" not in table.column_names or column not in table.column_names:
        raise ValueError("Frozen player ratings have an invalid schema")
    ratings = {}
    for row in table.select(["player_id", column]).to_pylist():
        player_id, rating = int(row["player_id"]), float(row[column])
        if player_id in ratings or not math.isfinite(rating):
            raise ValueError("Frozen player ratings contain duplicate IDs or nonfinite values")
        ratings[player_id] = rating
    return ratings
