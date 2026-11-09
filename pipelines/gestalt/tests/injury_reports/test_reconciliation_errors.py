import json
from datetime import datetime
from hashlib import sha256
from types import SimpleNamespace

import pytest

from gestalt_pipelines.injury_reports.utils import select_prior_failures, write_identity_events
from gestalt_pipelines.injury_reports.utils.errors import row_key

DAY = "2026-11-08"
EXTRACTION = {"uri": "s3://nba-gestalt-data/extraction.json", "sha256": "f" * 64}
STATUS_PATH = "s3://nba-gestalt-data/statuses.json"


def test_replay_recovers_only_the_original_error_for_each_resolved_row():
    objects = {}

    def save_json(key, record):
        uri = f"s3://nba-gestalt-data/{key}"
        objects[uri] = json.dumps(record).encode()
        return {"uri": uri, "sha256": sha256(objects[uri]).hexdigest()}

    def read_pinned(uri, digest, *, prefix):
        raw = objects[uri]
        assert prefix == "lake/artifacts/errors/pipeline=injury_reports/season=2026-27/" and sha256(raw).hexdigest() == digest
        return raw

    archive = SimpleNamespace(season="2026-27", save_json=save_json, read_pinned=read_pinned)
    first = {
        "game_id": "game", "team_id": "10", "source_row_index": 2,
        "team": "NBA", "player_name": "First Player", "pdf_uri": "s3://nba-gestalt-data/report.pdf",
    }
    second = {**first, "source_row_index": 3, "player_name": "Second Player"}
    unresolved = {row_key(row): {**row, "identity_status": "unmatched"} for row in (first, second)}
    common = {"game_date": DAY, "extraction": EXTRACTION, "artifact_path": STATUS_PATH}
    failures, recoveries = write_identity_events(
        archive, unresolved, {}, run_id="failure-run",
        occurred_at=datetime.fromisoformat("2026-11-08T15:00:00Z"), **common,
    )
    assert len(failures) == 2 and not recoveries
    assert all("/season=2026-27/occurred_date_pacific=2026-11-08/" in ref["uri"] for ref in failures)
    assert all(ref["uri"].endswith(".failure.json") for ref in failures)
    refs = failures
    first_replay = {**unresolved, row_key(first): {**first, "player_id": 101}}
    prior = select_prior_failures(archive, refs, first_replay, DAY, EXTRACTION)

    still_open, recovered = write_identity_events(
        archive, first_replay, prior, run_id="recovery-run",
        occurred_at=datetime.fromisoformat("2026-11-10T02:00:00Z"), **common,
    )

    assert still_open == [failures[1]] and len(recovered) == 1
    assert "/occurred_date_pacific=2026-11-09/" in recovered[0]["uri"]
    recovery = json.loads(objects[recovered[0]["uri"]])
    assert recovery["original_failure"]["uri"] == failures[0]["uri"]
    assert recovery["row"]["player_id"] == 101
    assert json.loads(objects[failures[0]["uri"]])["status"] == "failed"
    with pytest.raises(ValueError, match="selected date and extraction"):
        select_prior_failures(archive, refs, first_replay, DAY, {**EXTRACTION, "sha256": "e" * 64})
