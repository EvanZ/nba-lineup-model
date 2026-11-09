import json
from datetime import date, datetime
from hashlib import sha256
from types import SimpleNamespace

import pytest

from gestalt_pipelines.injury_reports.utils import read_roster, reconcile_games

DAY = date(2026, 11, 8)
EARLY_GAME = {
    "game_id": "early",
    "game_date": DAY,
    "game_time_utc": datetime.fromisoformat("2026-11-08T18:00:00Z"),
    "away_team_id": 1,
    "away_team_tricode": "AAA",
    "home_team_id": 2,
    "home_team_tricode": "BBB",
}
LATE_GAME = {
    "game_id": "late",
    "game_date": DAY,
    "game_time_utc": datetime.fromisoformat("2026-11-08T22:00:00Z"),
    "away_team_id": 3,
    "away_team_tricode": "CCC",
    "home_team_id": 4,
    "home_team_tricode": "DDD",
}


def test_reconcile_applies_game_cutoffs_and_keeps_unresolved_names():
    report = {
        "published_at": "2026-11-08T18:00:00Z",
        "pdf_uri": "s3://source/report.pdf",
        "game_team_reports": {
            "2026-11-08|AAA@BBB": {"AAA": {"status": "submitted"}},
            "2026-11-08|CCC@DDD": {
                "CCC": {"status": "submitted"}, "DDD": {"status": "not_yet_submitted"},
            },
        },
        "players": [
            {"player_name": name, "game_date": "2026-11-08", "matchup": "CCC@DDD",
             "team": "CCC", "status": "Out", "reason": "Injury"}
            for name in ("Jr., Test Ápple", "Test Berry", "Test Cherry")
        ],
    }
    rosters = {
        1: {10: "Home One"},
        2: {20: "Home Two"},
        3: {30: "Test Apple, Jr.", 31: "Test Berry Sr.", 32: "Test Berry Jr."},
        4: {40: "Home Four"},
    }

    games = reconcile_games([EARLY_GAME, LATE_GAME], rosters, report)

    assert set(games) == {"early", "late"}
    early = games["early"]
    assert not early["report_used"]
    assert {team["submission_status"] for team in early["teams"].values()} == {"missing"}
    late = games["late"]
    assert late["report_used"]
    assert late["teams"]["4"]["submission_status"] == "not_yet_submitted"
    assert not late["complete"]
    away = late["teams"]["3"]
    assert set(away["players"]) == {"30"}
    assert away["players"]["30"]["source_row_index"] == 0
    assert [row["identity_status"] for row in away["unresolved_players"]] == ["ambiguous", "unmatched"]
    assert all(row["pdf_uri"] == report["pdf_uri"] for row in away["unresolved_players"])


def test_roster_metadata_retains_capture_date_and_checks_official_team_id():
    team = 1
    season = "2026-27"
    prefix = f"lake/raw/commonteamroster/{season}/"
    root = f"s3://nba-gestalt-data/{prefix}{team}"
    roster = {
        "resultSets": [{
            "name": "CommonTeamRoster",
            "headers": ["PLAYER", "TeamID", "PLAYER_ID"],
            "rowSet": [["Test Player", team, 101]],
        }],
    }
    metadata = {
        "endpoint": "commonteamroster", "season": season, "team_id": team,
        "url": "https://stats.nba.com/stats/commonteamroster",
        "fetched_at": "2026-10-20T04:58:02Z",
    }

    def load():
        raw = json.dumps(roster).encode()
        sidecar = json.dumps({**metadata, "sha256": sha256(raw).hexdigest()}).encode()
        objects = {f"{root}.json": raw, f"{root}.meta.json": sidecar}

        def read_pinned(uri, expected, *, prefix):
            assert prefix == f"lake/raw/commonteamroster/{season}/"
            assert sha256(objects[uri]).hexdigest() == expected
            return objects[uri]

        return read_roster(
            SimpleNamespace(read_pinned=read_pinned),
            {"uri": f"{root}.meta.json", "sha256": sha256(sidecar).hexdigest()},
            season,
        )

    team_id, players, evidence = load()
    assert team_id == team and players == {101: "Test Player"}
    assert evidence["roster_date_pacific"] == "2026-10-19"
    roster["resultSets"][0]["rowSet"][0][1] = 2
    with pytest.raises(ValueError, match="Invalid official roster evidence"):
        load()


def test_operator_reexecutes_failed_partition_after_updating_s3_repairs(monkeypatch):
    from datetime import UTC, timedelta
    from io import BytesIO

    from dagster import DagsterInstance, in_process_executor
    from dagster._core.definitions.job_base import InMemoryJob
    from dagster._core.execution.api import execute_run
    from gestalt_pipelines.definitions import defs
    from gestalt_pipelines.injury_reports.assets import reconcile as asset
    from gestalt_pipelines.injury_reports.resources import archive as archive_module

    base = "s3://nba-gestalt-data/lake/raw/scheduleleaguev2/season=2026-27"
    latest_uri = f"{base}/latest.json"
    objects = {}

    def put(uri, raw):
        objects[uri] = raw
        return {"uri": uri, "sha256": sha256(raw).hexdigest()}

    def read_pinned(_archive, uri, digest, *, prefix):
        assert sha256(objects[uri]).hexdigest() == digest
        return objects[uri]

    def save_json(_archive, key, record):
        return put(f"s3://nba-gestalt-data/{key}", json.dumps(record).encode())

    roster_refs = [put(f"s3://fixture/{team}", b"roster") for team in (1, 2)]
    game = {**EARLY_GAME, "game_time_utc": datetime.now(UTC) + timedelta(hours=1)}
    report = {
        "published_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        "pdf_uri": "s3://fixture/report.pdf",
        "game_team_reports": {"2026-11-08|AAA@BBB": {
            "AAA": {"status": "submitted"}, "BBB": {"status": "submitted"},
        }},
        "players": [
            {"player_name": name, "game_date": DAY.isoformat(), "matchup": "AAA@BBB",
             "team": team, "status": "Out", "reason": "Injury"}
            for name, team in (("Original Name", "AAA"), ("Report Spelling", "AAA"), ("New Player", "BBB"))
        ],
    }
    repairs_uri = (
        "s3://nba-gestalt-data/lake/artifacts/injury_status_by_game_date/"
        "season=2026-27/player_repairs.json"
    )
    put(repairs_uri, b"{}")

    def get_object(*, Bucket, Key):
        return {"Body": BytesIO(objects[f"s3://{Bucket}/{Key}"])}

    monkeypatch.setattr(archive_module.boto3, "client", lambda _: SimpleNamespace(get_object=get_object))
    monkeypatch.setattr(asset.InjuryReportArchive, "read_pinned", read_pinned)
    monkeypatch.setattr(asset.InjuryReportArchive, "save_json", save_json)
    def games_for_date(resource, day, reference):
        payload, reference = resource.load(reference)
        assert day == DAY and payload == b"schedule"
        return [game], reference

    monkeypatch.setattr(asset.NBAScheduleResource, "games_for_date", games_for_date)
    monkeypatch.setattr(asset, "read_extraction", lambda *_: report)

    def roster(_archive, ref, _season):
        team = int(ref["uri"].rsplit("/", 1)[1])
        players = {101: "Original Name", 102: "Roster Spelling"} if team == 1 else {201: "Other Player"}
        return team, players, {
            "metadata": ref, "fetched_at": "2026-10-20T04:58:02Z", "roster_date_pacific": "2026-10-19",
        }

    monkeypatch.setattr(asset, "read_roster", roster)
    job = defs.resolve_job_def("reconcile_injury_date").with_executor_def(in_process_executor)

    def run(instance, config):
        return job.execute_in_process(
            instance=instance, partition_key=DAY.isoformat(), raise_on_error=False,
            run_config={"ops": {"injury_status_by_game_date": {"config": config}}},
        )

    digest = sha256(b"schedule").hexdigest()
    schedule_ref = put(f"{base}/sha256={digest}/schedule.json", b"schedule")
    put(latest_uri, json.dumps(schedule_ref).encode())
    with DagsterInstance.ephemeral() as instance:
        failed = run(instance, {
            "roster_metadata": roster_refs, "extraction": put("s3://fixture/extraction", b"report"),
        })
        assert not failed.success
        assert sum(uri.endswith(".failure.json") for uri in objects) == 2
        repairs = {"Original Name": 101, "Report Spelling": 102, "New Player": 202}
        repaired_ref = put(repairs_uri, json.dumps(repairs).encode())
        digest = sha256(b"new schedule").hexdigest()
        newer = put(f"{base}/sha256={digest}/schedule.json", b"new schedule")
        put(latest_uri, json.dumps(newer).encode())

        parent = instance.get_run_by_id(failed.run_id)
        child = instance.create_run_for_job(
            job, run_config=parent.run_config, tags=parent.tags,
            parent_run_id=parent.run_id, root_run_id=parent.run_id,
        )
        replay = execute_run(InMemoryJob(job), child, instance)
        assert any("Original Name resolved directly from the roster" in log.message
                   for log in instance.all_logs(replay.run_id))

    assert replay.success
    assert sum(uri.endswith(".recovery.json") for uri in objects) == 2
    corrected = [json.loads(raw) for uri, raw in objects.items() if uri.endswith("/statuses.json")][-1]
    assert corrected["complete"]
    assert corrected["inputs"]["schedule"] == schedule_ref
    assert corrected["inputs"]["player_repairs"] == {**repaired_ref, "entries": repairs}
    teams = corrected["games"]["early"]["teams"]
    assert teams["1"]["players"]["101"]["identity_source"] == "roster"
    assert teams["1"]["players"]["102"]["identity_source"] == "repair_dictionary"
    assert teams["2"]["players"]["202"]["identity_source"] == "repair_dictionary"
