"""Read operator-approved starting values and their exact S3 identity."""
import json
import math
import re
from contextlib import closing
from datetime import date
from hashlib import sha256

from botocore.exceptions import ClientError


def load_prior_repairs(client, *, bucket, key, season, model_version, day, value_field, maximum=None):
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
            return {}, None
        raise
    with closing(response["Body"]) as body:
        raw = body.read()
    doc = json.loads(raw)
    if (not isinstance(doc, dict) or doc.get("schema_version") != 1
            or doc.get("season") != season or doc.get("model_version") != model_version
            or not isinstance(doc.get("players"), dict)):
        raise ValueError(f"Invalid starting-prior repair file identity: {key}")
    eligible = {}
    for player_id, entry in doc["players"].items():
        if not re.fullmatch(r"[1-9][0-9]*", player_id) or not isinstance(entry, dict):
            raise ValueError(f"Invalid player entry in {key}: {player_id}")
        effective = date.fromisoformat(entry["eligible_from_game_date_pacific"])
        name, value, source = entry["player_name"], entry[value_field], entry["source"]
        if (not isinstance(name, str) or not name.strip()
                or type(value) not in (float, int) or not math.isfinite(value)
                or (maximum is not None and not 0 <= value <= maximum)
                or not re.fullmatch(r"[0-9]{4}-[0-9]{2}", source["season"]) or source["season"] >= season
                or not source["uri"].startswith(f"s3://{bucket}/")
                or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
                or re.search(r"\bespn\b|espncdn|espn\.com", json.dumps(entry), re.I)):
            raise ValueError(f"Invalid starting value or source in {key} for player {player_id}")
        if effective <= date.fromisoformat(day):
            eligible[player_id] = entry
    if not eligible:
        return {}, None
    ref = {"uri": f"s3://{bucket}/{key}", "sha256": sha256(raw).hexdigest()}
    if response.get("VersionId") not in (None, "null"):
        ref["version_id"] = response["VersionId"]
    return {pid: {**entry, "repair_input": ref} for pid, entry in eligible.items()}, ref
