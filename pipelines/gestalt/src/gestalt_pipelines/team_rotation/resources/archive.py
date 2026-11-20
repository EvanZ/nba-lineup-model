"""Verified saved inputs and immutable team rotation publication."""
import re
import json
from contextlib import closing
from datetime import UTC, datetime
from hashlib import sha256

import boto3
from botocore.exceptions import ClientError
from dagster import ConfigurableResource

from ...utils import DATA_BUCKET, ERRORS_PREFIX, NBA_TEAM_IDS, PACIFIC
from ..utils import json_bytes, official_json, summarize_team, validate_rows


class TeamRotationArchive(ConfigurableResource):
    bucket: str = DATA_BUCKET

    def _read(self, key):
        response = boto3.client("s3").get_object(Bucket=self.bucket, Key=key)
        with closing(response["Body"]) as stream:
            return stream.read(), response["ETag"]

    def _optional(self, key):
        try:
            raw, etag = self._read(key)
            return official_json(raw), etag
        except ClientError as error:
            if error.response["Error"]["Code"] in {"404", "NoSuchKey"}:
                return None, None
            raise

    def _verified(self, reference, prefix, *, suffix=".json"):
        digest = reference.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Expected a SHA-256 input checksum.")
        key = reference.get("uri", "").removeprefix(f"s3://{self.bucket}/")
        if key != prefix+digest+suffix:
            raise ValueError("Input reference is outside its canonical immutable path.")
        raw, _ = self._read(key)
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch at {reference['uri']}.")
        return official_json(raw)

    def prepare(self, *, season, day, first_day, nba_schedule):
        """Read each saved stint date and each pinned box once for all 30 teams."""
        raw, schedule = nba_schedule.load()
        catalog = json.loads(raw)["leagueSchedule"]
        if str(catalog["seasonYear"]) != season or str(catalog["leagueId"]) != "00":
            raise ValueError("Wrong NBA schedule season or league.")
        teams = {}
        for group in catalog["gameDates"]:
            for game in group["games"]:
                if not str(game["gameId"]).startswith("002"):
                    continue
                for side in ("homeTeam", "awayTeam"):
                    item = game[side]
                    tid = int(item["teamId"])
                    if tid in NBA_TEAM_IDS:
                        label = item["teamTricode"].upper()
                        if tid in teams and teams[tid] != label:
                            raise ValueError("Inconsistent official team identity.")
                        teams[tid] = label
        if set(teams) != set(NBA_TEAM_IDS) or len(set(teams.values())) != 30:
            raise ValueError("The official schedule must identify all 30 NBA teams.")
        games, _ = nba_schedule.games_between(first_day, day, schedule, allow_empty=True)
        all_games = [dict(g, game_date_pacific=g["game_time_utc"].astimezone(PACIFIC).date().isoformat())
                 for g in games]
        games = [g for g in all_games if g["game_id"].startswith("002")]
        dates = sorted({g["game_date_pacific"] for g in games})
        selection, schedule_etag = self._optional(f"lake/raw/scheduleleaguev2/season={season}/latest.json")
        if selection != schedule:
            raise ValueError("Schedule selection changed during reading.")
        rows, selections, guards, boxes, names = [], {}, [(f"lake/raw/scheduleleaguev2/season={season}/latest.json", schedule_etag)], {}, {}
        for date_key in dates:
            prefix = f"lake/artifacts/stints/season={season}/game_date_pacific={date_key}/"
            pointer, etag = self._optional(prefix+"latest.json")
            if pointer is None:
                raise FileNotFoundError(f"No accepted stints for {date_key}.")
            if pointer.get("season") != season or pointer.get("game_date_pacific") != date_key:
                raise ValueError("Stints selection has the wrong season or date.")
            doc = self._verified(pointer, prefix)
            expected = {g["game_id"] for g in games if g["game_date_pacific"] == date_key}
            all_expected = {g["game_id"] for g in all_games if g["game_date_pacific"] == date_key}
            if (doc.get("season") != season or doc.get("game_date_pacific") != date_key
                or doc.get("validation", {}).get("complete") is not True
                or set(doc["validation"]["expected_game_ids"]) != all_expected
                or set(doc["validation"]["processed_game_ids"]) != all_expected
                or set(doc["inputs"]["games"]) != all_expected):
                raise ValueError(f"Incomplete accepted stints for {date_key}.")
            guards.append((prefix+"latest.json", etag))
            selections[date_key] = pointer
            rows.extend(r for r in doc["rows"] if r["game_id"] in expected)
            if any(r["game_id"] not in all_expected for r in doc["rows"]):
                raise ValueError("Stints contain an unscheduled game.")
            for gid in sorted(expected):
                ref = doc["inputs"]["games"][gid]["box_score"]
                prefix_box = f"lake/raw/cdn/boxscore/season={season}/game_id={gid}/"
                box = self._verified(ref, prefix_box)["game"]
                meta = {"uri": ref["metadata_uri"], "sha256": ref["metadata_sha256"]}
                if meta["uri"] != f"s3://{self.bucket}/{prefix_box}{ref['sha256']}.meta.json":
                    raise ValueError("Box metadata reference has the wrong path.")
                meta_raw, _ = self._read(meta["uri"].removeprefix(f"s3://{self.bucket}/"))
                if sha256(meta_raw).hexdigest() != meta["sha256"]:
                    raise ValueError("Box metadata checksum mismatch.")
                official_json(meta_raw)
                game = next(g for g in games if g["game_id"] == gid)
                if box["gameId"] != gid or int(box["gameStatus"]) != 3:
                    raise ValueError(f"Pinned box is not final for {gid}.")
                for side in ("home", "away"):
                    team = box[side+"Team"]
                    if int(team["teamId"]) != game[side+"_team_id"]:
                        raise ValueError("Pinned box team differs from schedule.")
                    for player in team["players"]:
                        pid = int(player["personId"])
                        name = player.get("name") or " ".join(filter(None, (player.get("firstName"), player.get("familyName"))))
                        if name.strip():
                            names[pid] = name.strip()
                boxes[gid] = ref
        validate_rows(rows, games)
        records = []
        for tid, label in sorted(teams.items()):
            own = [g for g in games if tid in (g["home_team_id"], g["away_team_id"])]
            inputs = {"schedule": schedule,
                      "stints": {d: selections[d] for d in sorted({g["game_date_pacific"] for g in own})},
                      "box_scores": {g["game_id"]: boxes[g["game_id"]] for g in own}}
            records.append(summarize_team(season=season, day=day.isoformat(), team_id=tid,
                                         team=label, games=games, rows=rows, names=names, inputs=inputs))
        return records, guards, schedule

    def _immutable(self, key, raw):
        try:
            boto3.client("s3").put_object(Bucket=self.bucket, Key=key, Body=raw,
                                         ContentType="application/json", IfNoneMatch="*")
            created = True
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                raise
            created = False
        if self._read(key)[0] != raw:
            raise ValueError(f"Immutable rotation object failed readback: {key}.")
        return created

    def publish(self, records, guards):
        """Validate every team before writes; retain newer selections on older backfills."""
        if {r["team_id"] for r in records} != set(NBA_TEAM_IDS) or len(records) != 30:
            raise ValueError("Publication requires exactly all 30 team summaries.")
        identities = {(r["season"], r["game_date_pacific"]) for r in records}
        if len(identities) != 1 or any(r["coverage"]["complete"] is not True for r in records):
            raise ValueError("Publication requires complete summaries for one season/date.")
        staged = []
        for record in records:
            raw = json_bytes(record)
            digest = sha256(raw).hexdigest()
            base = f"lake/artifacts/team_rotation/season={record['season']}/team_id={record['team_id']}/"
            key = f"{base}game_date_pacific={record['game_date_pacific']}/{digest}.json"
            pointer = {k: record[k] for k in ("schema_version", "season", "team_id", "game_date_pacific", "processing_version")}
            pointer.update(uri=f"s3://{self.bucket}/{key}", sha256=digest)
            prior, etag = self._optional(base+"latest.json")
            if prior:
                prefix = base+f"game_date_pacific={prior['game_date_pacific']}/"
                previous = self._verified(prior, prefix)
                if any(previous.get(k) != prior.get(k) for k in ("season", "team_id", "game_date_pacific")):
                    raise ValueError("Invalid accepted rotation identity.")
            staged.append((key, raw, base+"latest.json", pointer, prior, etag))
        for key, etag in guards:
            if self._read(key)[1] != etag:
                raise ValueError("An accepted input changed; rerun with the corrected inputs.")
        created = sum(self._immutable(key, raw) for key, raw, *_ in staged)
        # Recheck after staging, before moving any accepted team selection.
        for key, etag in guards:
            if self._read(key)[1] != etag:
                raise ValueError("An accepted input changed during staging; no selection advanced.")
        advanced = 0
        for _, _, key, pointer, prior, etag in staged:
            if prior and prior["game_date_pacific"] > pointer["game_date_pacific"]:
                continue
            if pointer != prior:
                condition = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
                boto3.client("s3").put_object(Bucket=self.bucket, Key=key, Body=json_bytes(pointer),
                                             ContentType="application/json", **condition)
                advanced += 1
            if self._optional(key)[0] != pointer:
                raise ValueError("Rotation selection failed readback.")
        return {"team_count": 30, "created_count": created, "reused_count": 30-created,
                "advanced_count": advanced, "selections": [item[3] for item in staged]}

    def save_event(self, *, season, day, run_id, message, original_failure=None):
        status = "recovered" if original_failure else "failed"
        identity = [run_id, "team_rotation", status, original_failure]
        now = datetime.now(UTC)
        event = {"schema_version": 1, "event_id": sha256(json_bytes(identity)).hexdigest(),
                 "stage": "team_rotation", "season": season, "status": status,
                 "timestamp": now.isoformat(), "message": message, "dagster_run_id": run_id,
                 "game_date_pacific": day, "original_failure": original_failure}
        suffix = "recovery" if original_failure else "failure"
        key = (f"{ERRORS_PREFIX}/pipeline=team_rotation/season={season}/"
               f"occurred_date_pacific={now.astimezone(PACIFIC).date()}/{event['event_id']}.{suffix}.json")
        old, _ = self._optional(key)
        if old:
            if {k:v for k,v in old.items() if k != "timestamp"} != {k:v for k,v in event.items() if k != "timestamp"}:
                raise ValueError("Saved rotation error differs from this run.")
            event = old
        raw = json_bytes(event)
        self._immutable(key, raw)
        return {"uri": f"s3://{self.bucket}/{key}", "sha256": sha256(raw).hexdigest()}

    def read_failure(self, ref, *, season, day):
        key = ref["uri"].removeprefix(f"s3://{self.bucket}/")
        prefix = f"{ERRORS_PREFIX}/pipeline=team_rotation/season={season}/"
        if not key.startswith(prefix) or not key.endswith(".failure.json"):
            raise ValueError("Failure reference is outside the rotation error path.")
        raw, _ = self._read(key)
        doc = official_json(raw)
        if (sha256(raw).hexdigest() != ref["sha256"] or doc["season"] != season
            or doc["game_date_pacific"] != day or doc["stage"] != "team_rotation" or doc["status"] != "failed"):
            raise ValueError("Rotation failure differs from the selected failed run.")
        return ref
