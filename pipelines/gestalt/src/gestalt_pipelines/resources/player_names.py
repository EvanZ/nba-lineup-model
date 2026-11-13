"""Read official NBA player names from the saved PlayerIndex in S3."""

import json
import re
from hashlib import sha256
from urllib.parse import parse_qsl, urlparse

import boto3
from dagster import ConfigurableResource

from ..utils import DATA_BUCKET


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
        return players, {
            "uri": f"s3://{self.bucket}/{key}", "sha256": digest,
            "metadata": {"uri": f"s3://{self.bucket}/{meta_key}", "sha256": sha256(metadata_raw).hexdigest()},
        }
