import base64
import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.injury_reports.resources import archive, official_nba
from gestalt_pipelines.injury_reports.utils import parse_pdf

SOURCE = (
    "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-11-07_11_45PM.pdf"
)
PDF = b"%PDF-1.7\nfixture PDF bytes\n%%EOF\n"
SHA = sha256(PDF).hexdigest()
PREFIX = (
    "lake/raw/nba/injury-reports/season=2026-27/publication_date=2026-11-07/"
    f"published_at=20261108T044500Z/sha256={SHA}"
)
OBSERVED = datetime(2026, 11, 8, 4, 47, tzinfo=UTC)
CAPTURED = OBSERVED + timedelta(seconds=1)


class MemoryArchive:
    def __init__(self):
        self.objects = {}
        self.fail_suffix = None
        self.corrupt_pdf = False

    def put_object(self, **request):
        key = request["Key"]
        assert request["Bucket"] == "nba-gestalt-data"
        assert request["IfNoneMatch"] == "*"
        checksum = base64.b64encode(sha256(request["Body"]).digest()).decode("ascii")
        assert request["ChecksumSHA256"] == checksum
        if self.fail_suffix and key.endswith(self.fail_suffix):
            raise ClientError({"Error": {"Code": "503", "Message": "unavailable"}}, "PutObject")
        if key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[key] = (request["Body"], request["Metadata"])
        return {}

    def get_object(self, *, Bucket, Key):
        assert Bucket == "nba-gestalt-data"
        data, metadata = self.objects[Key]
        if self.corrupt_pdf and Key.endswith("report.pdf"):
            data += b"unexpected"
        return {"Body": BytesIO(data), "Metadata": metadata, "LastModified": CAPTURED}


@pytest.fixture
def capture(monkeypatch):
    storage = MemoryArchive()
    observed = [OBSERVED]

    def download(_source, url):
        return official_nba.OfficialInjuryReport(
            url, official_nba.publication_time(url), observed[0], PDF
        )

    monkeypatch.setattr(archive.boto3, "client", lambda service: storage)
    monkeypatch.setattr(official_nba.OfficialInjuryReports, "download", download)
    job = defs.resolve_job_def("capture_injury_pdfs")

    def run(*, checksum=SHA, raise_on_error=True):
        return job.execute_in_process(
            run_config={"ops": {"injury_pdf_archive": {"config": {
                "source_url": SOURCE,
                "expected_sha256": checksum,
            }}}},
            raise_on_error=raise_on_error,
        )

    return storage, observed, run


def test_capture_uses_agreed_paths_and_reuses_an_existing_revision(capture):
    storage, observed, run = capture
    assert run().success
    assert set(storage.objects) == {f"{PREFIX}/report.pdf", f"{PREFIX}/capture.json"}
    pdf_bytes, _ = storage.objects[f"{PREFIX}/report.pdf"]
    assert pdf_bytes == PDF
    initial, _ = storage.objects[f"{PREFIX}/capture.json"]
    assert json.loads(initial) == {
        "schema_version": 1,
        "season": "2026-27",
        "source_url": SOURCE,
        "immutable_uri": f"s3://nba-gestalt-data/{PREFIX}/report.pdf",
        "sha256": SHA,
        "published_at": "2026-11-08T04:45:00Z",
        "first_observed_at": "2026-11-08T04:47:00Z",
        "captured_at": "2026-11-08T04:47:01Z",
    }
    observed[0] += timedelta(minutes=15)
    assert run().success
    assert storage.objects[f"{PREFIX}/capture.json"][0] == initial
    assert len(storage.objects) == 2


def test_failed_pdf_upload_does_not_commit_a_capture(capture):
    storage, _, run = capture
    storage.fail_suffix = "report.pdf"
    assert not run(raise_on_error=False).success
    assert storage.objects == {}


def test_pdf_verification_failure_does_not_commit_a_capture(capture):
    storage, _, run = capture
    storage.corrupt_pdf = True
    assert not run(raise_on_error=False).success
    assert set(storage.objects) == {f"{PREFIX}/report.pdf"}


def test_capture_record_can_be_retried_after_the_pdf_was_saved(capture):
    storage, observed, run = capture
    storage.fail_suffix = "capture.json"
    assert not run(raise_on_error=False).success
    assert set(storage.objects) == {f"{PREFIX}/report.pdf"}
    storage.fail_suffix = None
    observed[0] += timedelta(minutes=15)
    assert run().success
    record = json.loads(storage.objects[f"{PREFIX}/capture.json"][0])
    assert record["first_observed_at"] == "2026-11-08T04:47:00Z"


def test_unapproved_pdf_bytes_are_rejected_before_storage(capture):
    storage, _, run = capture
    assert not run(checksum="0" * 64, raise_on_error=False).success
    assert storage.objects == {}


@pytest.mark.parametrize("source", [
    "https://example.com/referee/injury/Injury-Report_2026-11-07_11_45PM.pdf",
    "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-11-01_01_15AM.pdf",
])
def test_source_or_publication_time_must_be_unambiguous(source):
    with pytest.raises(ValueError):
        official_nba.publication_time(source)


def test_download_rejects_a_truncated_pdf(monkeypatch):
    truncated = BytesIO(b"%PDF-1.7\npartial response")
    opener = SimpleNamespace(open=lambda *_args, **_kwargs: truncated)
    monkeypatch.setattr(official_nba, "build_opener", lambda *_args: opener)

    with pytest.raises(ValueError, match="not a supported PDF"):
        official_nba.OfficialInjuryReports().download(SOURCE)


def _injury_pdf(*pages):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for commands in pages:
        page = writer.add_blank_page(width=850, height=600)
        contents = DecodedStreamObject()
        contents.set_data(commands)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page["/Resources"][NameObject("/Font")] = DictionaryObject({NameObject("/F1"): font})
        page.replace_contents(contents)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("corrupt_pdf", [False, True])
def test_extract_exact_captured_pdf(monkeypatch, corrupt_pdf):
    pdf = _injury_pdf(b"""
        BT 1 0 0 1 20 100 Tm /F1 10 Tf (11/07/2026) Tj ET
        BT 1 0 0 1 190 100 Tm /F1 10 Tf (BOS@NY) Tj ET
        BT 1 0 0 1 250 100 Tm /F1 10 Tf (Boston Celtics) Tj ET
        BT 1 0 0 1 420 100 Tm /F1 10 Tf (Example, Player) Tj ET
        BT 1 0 0 1 580 100 Tm /F1 10 Tf (Questionable) Tj ET
        BT 1 0 0 1 666 100 Tm /F1 10 Tf (Left Ankle) Tj ET
        BT 1 0 0 1 250 120 Tm /F1 10 Tf (New York Knicks) Tj ET
        BT 1 0 0 1 420 120 Tm /F1 10 Tf (NOT YET SUBMITTED) Tj ET
    """)
    digest = sha256(pdf).hexdigest()

    storage = MemoryArchive()
    monkeypatch.setattr(archive.boto3, "client", lambda service: storage)
    capture = archive.InjuryReportArchive(season="2026-27").save(
        official_nba.OfficialInjuryReport(
            SOURCE, official_nba.publication_time(SOURCE), OBSERVED, pdf
        ),
        digest,
    )
    storage.corrupt_pdf = corrupt_pdf
    job = defs.resolve_job_def("extract_injury_pdf")
    config = {"ops": {"injury_report_extractions": {"config": {
        "capture_uri": capture["capture_uri"],
    }}}}
    result = job.execute_in_process(run_config=config, raise_on_error=False)
    key = (
        "lake/artifacts/injury_report_extraction/season=2026-27/publication_date=2026-11-07/"
        f"published_at=20261108T044500Z/sha256={digest}/parser_version=v2/report.json"
    )

    if corrupt_pdf:
        assert not result.success
        assert key not in storage.objects
        error = result.get_step_failure_events()[0].event_specific_data.error
        assert "does not match its capture" in error.cause.message
        return

    assert result.success
    original = storage.objects[key][0]
    record = json.loads(original)
    assert record["pdf_uri"] == capture["pdf_uri"]
    assert record["capture_uri"] == capture["capture_uri"]
    assert record["reported_game_dates"] == ["2026-11-07"]
    assert record["players"] == [{
        "player_name": "Player Example",
        "status": "Questionable",
        "reason": "Left Ankle",
        "game_date": "2026-11-07",
        "matchup": "BOS@NYK",
        "team": "BOS",
    }]
    assert record["game_team_reports"]["2026-11-07|BOS@NYK"]["NYK"]["status"] == "not_yet_submitted"
    retried = job.execute_in_process(run_config=config)
    event = retried.get_asset_materialization_events()[0]
    materialization = event.event_specific_data.materialization
    assert not materialization.metadata["created"].value
    assert storage.objects[key][0] == original


@pytest.mark.parametrize(("evidence", "status"), [
    ("NO INJURIES TO REPORT", "submitted"),
    ("NOT YET SUBMITTED", "not_yet_submitted"),
])
def test_zero_player_report_keeps_only_evidenced_team_statuses(evidence, status):
    pdf = _injury_pdf(b"""
        BT 1 0 0 1 20 100 Tm /F1 10 Tf (11/07/2026) Tj ET
        BT 1 0 0 1 190 100 Tm /F1 10 Tf (BOS@NY) Tj ET
        BT 1 0 0 1 250 100 Tm /F1 10 Tf (Boston Celtics) Tj ET
    """, f"""
        BT 1 0 0 1 250 50 Tm /F1 10 Tf (Injury Report) Tj ET
        BT 1 0 0 1 250 65 Tm /F1 10 Tf (Page) Tj ET
        BT 1 0 0 1 250 80 Tm /F1 10 Tf (Team) Tj ET
        BT 1 0 0 1 420 110 Tm /F1 10 Tf ({evidence}) Tj ET
        BT 1 0 0 1 190 130 Tm /F1 10 Tf (LAL@PHX) Tj ET
    """.encode("ascii"))

    report = parse_pdf(pdf)

    assert report["players"] == []
    assert report["game_team_reports"] == {
        "2026-11-07|BOS@NYK": {
            "BOS": {"team": "BOS", "matchup": "BOS@NYK", "game_date": "2026-11-07", "status": status}
        },
        "2026-11-07|LAL@PHX": {},
    }


def test_report_without_readable_submission_evidence_fails():
    pdf = _injury_pdf(b"""
        BT 1 0 0 1 20 100 Tm /F1 10 Tf (11/07/2026) Tj ET
        BT 1 0 0 1 190 100 Tm /F1 10 Tf (BOS@NY) Tj ET
        BT 1 0 0 1 250 100 Tm /F1 10 Tf (Boston Celtics) Tj ET
    """)

    with pytest.raises(ValueError, match="no readable team submission evidence"):
        parse_pdf(pdf)
