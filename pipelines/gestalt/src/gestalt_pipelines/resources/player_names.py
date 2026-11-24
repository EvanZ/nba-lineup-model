"""Read official NBA player identities from saved PlayerIndex and transactions."""

import json
import re
import unicodedata
from datetime import date, datetime
from hashlib import sha256
from urllib.parse import parse_qsl, urlparse

import boto3
from dagster import ConfigurableResource

from ..utils import DATA_BUCKET


TRANSACTIONS_URL = "https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json"


def _name_key(value):
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]", "", text)


def _transaction_name(row):
    # Extract the actual name text from the official description. The slug only
    # identifies that span; never invent a display name by title-casing a slug.
    target = _name_key(row.get("PLAYER_SLUG") or "")
    text = row.get("TRANSACTION_DESCRIPTION") or ""
    if not target:
        return None
    words = list(re.finditer(r"\S+", text))
    for start in range(len(words)):
        for end in range(start, len(words)):
            candidate = text[words[start].start():words[end].end()].strip(" ,;:")
            key = _name_key(candidate)
            if key == target:
                return candidate.rstrip(".")
            if len(key) > len(target):
                break
    return None


class NBAPlayerNamesResource(ConfigurableResource):
    season: str
    bucket: str = DATA_BUCKET

    def load(self) -> tuple[dict[int, str], dict]:
        """Return NBA ID -> name and the exact raw/metadata references used."""
        client = boto3.client("s3")
        key = f"lake/raw/playerindex/{self.season}.json"
        with client.get_object(Bucket=self.bucket, Key=key)["Body"] as body:
            raw = body.read()
        meta_key = key.replace(".json", ".meta.json")
        with client.get_object(Bucket=self.bucket, Key=meta_key)["Body"] as body:
            metadata_raw = body.read()
        metadata = json.loads(metadata_raw)
        url = urlparse(metadata["url"])
        params = dict(parse_qsl(url.query, keep_blank_values=True))
        digest = sha256(raw).hexdigest()
        if (
            metadata["endpoint"] != "playerindex" or metadata["season"] != self.season
            or metadata["sha256"] != digest
            or (url.scheme, url.netloc, url.path) != ("https", "stats.nba.com", "/stats/playerindex")
            or metadata["params"] != params
            or params.get("Season") != self.season or params.get("Historical") != "1"
            or params.get("LeagueID") != "00" or params.get("TeamID") != "0"
        ):
            raise ValueError("Invalid source identity or checksum in NBA PlayerIndex metadata.")
        if "espn" in raw.decode().casefold() or "espn" in metadata_raw.decode().casefold():
            raise ValueError("Prohibited source text in NBA PlayerIndex.")
        tables = [table for table in json.loads(raw)["resultSets"] if table["name"] == "PlayerIndex"]
        if len(tables) != 1:
            raise ValueError("Expected one NBA PlayerIndex table.")
        table = tables[0]
        players = {}
        for values in table["rowSet"]:
            if len(values) != len(table["headers"]):
                raise ValueError("Malformed NBA PlayerIndex row.")
            row = dict(zip(table["headers"], values))
            pid = str(row["PERSON_ID"])
            name = " ".join(part.strip() for part in (
                row["PLAYER_FIRST_NAME"], row["PLAYER_LAST_NAME"],
            ) if part)
            if not re.fullmatch(r"[1-9][0-9]*", pid) or not name:
                raise ValueError("Invalid NBA player identity in PlayerIndex.")
            player_id = int(pid)
            if player_id in players and players[player_id] != name:
                raise ValueError(f"Conflicting NBA names for player {player_id}.")
            players[player_id] = name
        if not players:
            raise ValueError("The NBA PlayerIndex contains no players.")
        indexed_ids = set(players)
        transaction_refs, learned, unresolved = {}, set(), set()
        prefix = f"lake/raw/nba/transactions/season={self.season}/"
        keys = sorted(obj["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix)
                      for obj in page.get("Contents", [])
                      if re.fullmatch(re.escape(prefix)+r"game_date_pacific=\d{4}-\d{2}-\d{2}/latest\.json", obj["Key"]))
        for transaction_key in keys:
            with client.get_object(Bucket=self.bucket, Key=transaction_key)["Body"] as body:
                transaction_raw = body.read()
            record = json.loads(transaction_raw)
            day = transaction_key.split("game_date_pacific=", 1)[1][:10]
            if (date.fromisoformat(day).isoformat() != day or record["season"] != self.season
                    or record["game_date_pacific"] != day or record["source_url"] != TRANSACTIONS_URL
                    or not isinstance(record["transactions"], list) or "espn" in transaction_raw.decode().casefold()):
                raise ValueError("Invalid official NBA transaction identity source.")
            used = False
            for row in record["transactions"]:
                value = row["PLAYER_ID"]
                if isinstance(value, bool):
                    raise ValueError("Invalid player ID in official NBA transactions.")
                pid = int(value)
                if pid <= 0 or pid != float(value):
                    raise ValueError("Invalid player ID in official NBA transactions.")
                if (date.fromisoformat(row["TRANSACTION_DATE"][:10]).isoformat() != day
                        or datetime.fromisoformat(row["first_observed_at"]).tzinfo is None):
                    raise ValueError("Invalid transaction date/observation identity.")
                if pid in indexed_ids or pid in learned:
                    continue
                used = True
                name = _transaction_name(row)
                if name is None:
                    unresolved.add(pid)
                    continue
                players[pid] = name
                learned.add(pid);unresolved.discard(pid)
            if used:
                transaction_refs[day] = {"uri": f"s3://{self.bucket}/{transaction_key}", "sha256": sha256(transaction_raw).hexdigest()}
        return players, {
            "uri": f"s3://{self.bucket}/{key}", "sha256": digest,
            "metadata": {"uri": f"s3://{self.bucket}/{meta_key}", "sha256": sha256(metadata_raw).hexdigest()},
            "transactions": transaction_refs, "transaction_player_ids": sorted(learned),
            "unresolved_transaction_player_ids": sorted(unresolved),
        }
