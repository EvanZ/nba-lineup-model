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

SOURCE = (
    "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-11-07_11_45PM.pdf"
)
PDF = b"%PDF-1.7\nfixture PDF bytes\n%%EOF\n"
SHA = sha256(PDF).hexdigest()
PREFIX = (
    "lake/raw/nba/injury-reports/publication_date=2026-11-07/"
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
