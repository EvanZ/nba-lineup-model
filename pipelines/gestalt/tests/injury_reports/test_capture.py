import base64
import json
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from botocore.exceptions import ClientError
from dagster import DagsterInvalidConfigError, build_schedule_context, build_sensor_context, validate_run_config

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.injury_reports.resources import archive, official_nba
from gestalt_pipelines.injury_reports.utils import parse_pdf, read_extraction

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
        self.commit_times = {}
        self.now = CAPTURED
        self.fail_suffix = None
        self.corrupt_pdf = False

    def put_object(self, **request):
        key = request["Key"]
        assert request["Bucket"] == "nba-gestalt-data"
        checksum = base64.b64encode(sha256(request["Body"]).digest()).decode("ascii")
        assert request["ChecksumSHA256"] == checksum
        if self.fail_suffix and key.endswith(self.fail_suffix):
            raise ClientError({"Error": {"Code": "503", "Message": "unavailable"}}, "PutObject")
        if "IfMatch" in request:
            matches = key in self.objects and request["IfMatch"] == self.etag(key)
        else:
            assert request["IfNoneMatch"] == "*"
            matches = key not in self.objects
        if not matches:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[key] = (request["Body"], request["Metadata"])
        self.commit_times[key] = self.now
        return {}

    def etag(self, key):
        return f'"{sha256(self.objects[key][0]).hexdigest()}"'

    def get_object(self, *, Bucket, Key):
        assert Bucket == "nba-gestalt-data"
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        data, metadata = self.objects[Key]
        if self.corrupt_pdf and Key.endswith("report.pdf"):
            data += b"unexpected"
        return {
            "Body": BytesIO(data), "Metadata": metadata, "LastModified": self.commit_times.get(Key, CAPTURED),
            "ETag": self.etag(Key),
        }


    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return SimpleNamespace(paginate=self._list_objects)

    def _list_objects(self, *, Bucket, Prefix):
        assert Bucket == "nba-gestalt-data"
        return [{"Contents": [
            {"Key": key, "LastModified": self.commit_times.get(key, CAPTURED)}
            for key in self.objects if key.startswith(Prefix)
        ]}]


@pytest.fixture
def capture(monkeypatch):
    storage = MemoryArchive()
    observed = [OBSERVED]

    def download(_source, url):
        if url != SOURCE:
            raise HTTPError(url, 404, "Unavailable", None, None)
        return official_nba.OfficialInjuryReport(
            url, official_nba.publication_time(url), observed[0], PDF
        )

    monkeypatch.setattr(archive.boto3, "client", lambda service: storage)
    monkeypatch.setattr(official_nba.OfficialInjuryReports, "download", download)
    job = defs.resolve_job_def("capture_injury_pdfs")

    def run(partition_key="2026-11-07", *, raise_on_error=True):
        return job.execute_in_process(partition_key=partition_key, raise_on_error=raise_on_error)

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


def test_capture_partitions_fill_each_date_and_skip_saved_slots(capture, monkeypatch):
    storage, _, run = capture
    late = SOURCE.replace("11_45PM", "11_30PM")
    first_day = {SOURCE, late}
    second_day = {url.replace("2026-11-07", "2026-11-06") for url in first_day}
    available = first_day | second_day
    attempted = []

    def download(_source, url):
        attempted.append(url)
        if url not in available:
            raise HTTPError(url, 404, "Unavailable", None, None)
        return official_nba.OfficialInjuryReport(
            url, official_nba.publication_time(url), OBSERVED, PDF,
        )

    monkeypatch.setattr(official_nba.OfficialInjuryReports, "download", download)

    def metadata(day):
        attempted.clear()
        result = run(day)
        assert result.success
        [event] = result.get_asset_materialization_events()
        assert event.partition == day
        return event.event_specific_data.materialization.metadata

    first = metadata("2026-11-07")
    assert first["recovered_count"].value == 2
    assert set(attempted) & available == first_day

    overlapping = metadata("2026-11-07")
    assert overlapping["reused_count"].value == 2
    assert overlapping["recovered_capture_uris"].value == []
    assert set(attempted).isdisjoint(available)

    second = metadata("2026-11-06")
    assert second["recovered_count"].value == 2
    assert set(attempted) & available == second_day
    assert len(storage.objects) == 8


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


def test_capture_skips_unavailable_quarter_hour_slots(capture, monkeypatch):
    storage, _, run = capture
    attempted = []

    def download(_source, url):
        attempted.append(url)
        if url != SOURCE.replace("11_45PM", "11_15PM"):
            code = 403 if url.endswith("11_30PM.pdf") else 404
            raise HTTPError(url, code, "Unavailable", None, None)
        return official_nba.OfficialInjuryReport(
            url, official_nba.publication_time(url), OBSERVED, PDF,
        )

    monkeypatch.setattr(official_nba.OfficialInjuryReports, "download", download)
    assert run().success
    assert attempted[:3] == [
        SOURCE,
        SOURCE.replace("11_45PM", "11_30PM"),
        SOURCE.replace("11_45PM", "11_15PM"),
    ]
    [capture_record] = [json.loads(value[0]) for key, value in storage.objects.items() if key.endswith("capture.json")]
    assert len(attempted) == 96
    assert capture_record["source_url"] == attempted[2]
    assert capture_record["sha256"] == SHA


def test_capture_reports_the_date_when_no_pdf_is_available(capture, monkeypatch):
    storage, _, run = capture
    attempted = []

    def download(_source, url):
        attempted.append(url)
        raise HTTPError(url, 404, "Unavailable", None, None)

    monkeypatch.setattr(official_nba.OfficialInjuryReports, "download", download)
    result = run(raise_on_error=False)
    assert not result.success and not storage.objects
    [failure] = result.get_step_failure_events()
    details = failure.event_specific_data.user_failure_data
    assert "No official NBA injury report found for 2026-11-07" in details.description
    assert details.metadata["unavailable_count"].value == 96
    assert len(attempted) == 96
    assert attempted[-1] == SOURCE.replace("11_45PM", "12_00AM")


@pytest.mark.parametrize("source", [
    "https://example.com/referee/injury/Injury-Report_2026-11-07_11_45PM.pdf",
    "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-11-01_01_15AM.pdf",
])
def test_source_or_publication_time_must_be_unambiguous(source):
    with pytest.raises(ValueError):
        official_nba.publication_time(source)


def test_spring_forward_skips_nonexistent_eastern_slots():
    slots = list(official_nba.OfficialInjuryReports().publication_slots(date(2026, 3, 8)))
    assert len(slots) == 92
    assert all("_02_" not in url or not url.endswith("AM.pdf") for _, url in slots)


def test_download_rejects_a_truncated_pdf(monkeypatch):
    truncated = BytesIO(b"%PDF-1.7\npartial response")
    opener = SimpleNamespace(open=lambda *_args, **_kwargs: truncated)
    monkeypatch.setattr(official_nba, "build_opener", lambda *_args: opener)

    with pytest.raises(ValueError, match="not a supported PDF"):
        official_nba.OfficialInjuryReports().download(SOURCE)


def test_latest_capture_prefers_publication_time_then_saved_revision(monkeypatch):
    day_prefix = PREFIX.split("/published_at=", 1)[0] + "/"
    first_revision = f"{PREFIX}/capture.json"
    corrected = f"{PREFIX.removesuffix(SHA)}{'a' * 64}/capture.json"
    older_report = first_revision.replace("T044500Z", "T043000Z")
    incomplete = f"{PREFIX.removesuffix(SHA)}{'b' * 64}/report.pdf"

    def paginate(*, Bucket, Prefix):
        assert Bucket == "nba-gestalt-data" and Prefix == day_prefix
        return [
            {"Contents": [
                {"Key": older_report, "LastModified": CAPTURED + timedelta(minutes=1)},
                {"Key": first_revision, "LastModified": CAPTURED},
            ]},
            {"Contents": [
                {"Key": corrected, "LastModified": CAPTURED + timedelta(seconds=1)},
                {"Key": incomplete, "LastModified": CAPTURED + timedelta(minutes=2)},
            ]},
        ]

    client = SimpleNamespace(get_paginator=lambda _: SimpleNamespace(paginate=paginate))
    monkeypatch.setattr(archive.boto3, "client", lambda _: client)

    resource = archive.InjuryReportArchive(season="2026-27")
    assert resource.latest_capture_uri(date(2026, 11, 7)) == f"s3://nba-gestalt-data/{corrected}"


@pytest.mark.parametrize("config", [
    {},
    {"publication_date": "2026-11-07", "capture_uri": f"s3://nba-gestalt-data/{PREFIX}/capture.json"},
])
def test_extract_config_requires_exactly_one_input(config):
    job = defs.resolve_job_def("extract_injury_pdf")
    run_config = {"ops": {"injury_report_extractions": {"config": config}}}
    with pytest.raises(DagsterInvalidConfigError):
        validate_run_config(job, run_config)


def _injury_pdf(*pages, creation_date=None):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    if creation_date is not None:
        writer.add_metadata({"/CreationDate": creation_date})
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


@pytest.mark.parametrize(("source", "creation_date", "published_at", "corrupt_pdf"), [
    (SOURCE, None, "2026-11-08T04:45:00Z", False),
    (SOURCE, None, "2026-11-08T04:45:00Z", True),
    (
        "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-11-01_01_45AM.pdf",
        "D:20261101014502-05'00'",
        "2026-11-01T06:45:00Z",
        False,
    ),
])
def test_extract_exact_captured_pdf(monkeypatch, source, creation_date, published_at, corrupt_pdf):
    pdf = _injury_pdf(b"""
        BT 1 0 0 1 20 100 Tm /F1 10 Tf (11/07/2026) Tj ET
        BT 1 0 0 1 190 100 Tm /F1 10 Tf (BOS@NY) Tj ET
        BT 1 0 0 1 250 100 Tm /F1 10 Tf (Boston Celtics) Tj ET
        BT 1 0 0 1 420 100 Tm /F1 10 Tf (Example, Player) Tj ET
        BT 1 0 0 1 580 100 Tm /F1 10 Tf (Questionable) Tj ET
        BT 1 0 0 1 666 100 Tm /F1 10 Tf (Left Ankle) Tj ET
        BT 1 0 0 1 250 120 Tm /F1 10 Tf (New York Knicks) Tj ET
        BT 1 0 0 1 420 120 Tm /F1 10 Tf (NOT YET SUBMITTED) Tj ET
    """, creation_date=creation_date)
    digest = sha256(pdf).hexdigest()
    verified_time = official_nba.publication_time(source, pdf)
    expected_time = datetime.fromisoformat(published_at)
    assert verified_time == expected_time

    storage = MemoryArchive()
    monkeypatch.setattr(archive.boto3, "client", lambda service: storage)
    capture = archive.InjuryReportArchive(season="2026-27").save(
        official_nba.OfficialInjuryReport(
            source, verified_time, OBSERVED, pdf
        ),
    )
    storage.corrupt_pdf = corrupt_pdf
    job = defs.resolve_job_def("extract_injury_pdf")
    config = {"ops": {"injury_report_extractions": {"config": {
        "capture_uri": capture["capture_uri"],
    }}}}
    publication_date = expected_time.astimezone(official_nba.EASTERN).date()
    result = job.execute_in_process(
        partition_key=publication_date.isoformat(), run_config=config, raise_on_error=False,
    )
    key = (
        f"lake/artifacts/injury_report_extraction/season=2026-27/publication_date={publication_date}/"
        f"published_at={expected_time:%Y%m%dT%H%M%SZ}/sha256={digest}/parser_version=v2/report.json"
    )

    if corrupt_pdf:
        assert not result.success
        assert key not in storage.objects
        error = result.get_step_failure_events()[0].event_specific_data.error
        assert "does not match its capture" in error.cause.message
        return

    assert result.success
    original = storage.objects[key][0]
    record = read_extraction(original, f"s3://nba-gestalt-data/{key}", "2026-27")
    assert record["published_at"] == published_at
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
    encoded_uri = capture["capture_uri"].replace("=", "%3D", 1).replace("=", "%3d")
    config["ops"]["injury_report_extractions"]["config"]["capture_uri"] = encoded_uri
    retried = job.execute_in_process(partition_key=publication_date.isoformat(), run_config=config)
    event = retried.get_asset_materialization_events()[0]
    materialization = event.event_specific_data.materialization
    assert materialization.metadata["capture_uri"].value == capture["capture_uri"]
    assert not materialization.metadata["created"].value
    assert storage.objects[key][0] == original


def test_extract_partition_processes_all_captures_and_reuses_current_outputs(monkeypatch):
    from gestalt_pipelines.injury_reports.assets import extract

    storage = MemoryArchive()
    monkeypatch.setattr(archive.boto3, "client", lambda service: storage)
    resource = archive.InjuryReportArchive(season="2026-27")
    for url in (SOURCE, SOURCE.replace("11_45PM", "11_30PM")):
        resource.save(official_nba.OfficialInjuryReport(url, official_nba.publication_time(url), OBSERVED, PDF))

    parsed = []

    def parse(pdf):
        parsed.append(pdf)
        return {"game_team_reports": {"2026-11-07|BOS@NYK": {}}, "players": []}

    monkeypatch.setattr(extract, "parse_pdf", parse)
    job = defs.resolve_job_def("extract_injury_pdf")
    assert validate_run_config(job, {})
    for created in (2, 0):
        result = job.execute_in_process(partition_key="2026-11-07")
        materialization = result.get_asset_materialization_events()[0].event_specific_data.materialization
        assert materialization.metadata["created_count"].value == created
        assert materialization.metadata["reused_count"].value == 2 - created
    assert parsed == [PDF, PDF]


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


def test_scheduled_capture_spans_midnight_without_scanning_the_day(capture, monkeypatch):
    from gestalt_pipelines.injury_reports import schedules

    injury_pdf_poll = schedules.injury_pdf_poll
    tick = datetime.fromisoformat("2026-11-08T05:02:00Z")
    monkeypatch.setattr(schedules, "datetime", SimpleNamespace(now=lambda _: tick))
    missed = build_schedule_context(
        scheduled_execution_time=tick - timedelta(minutes=15), repository_def=defs.get_repository_def(),
    )
    assert not injury_pdf_poll.evaluate_tick(missed).run_requests
    context = build_schedule_context(scheduled_execution_time=tick, repository_def=defs.get_repository_def())
    requests = injury_pdf_poll.evaluate_tick(context).run_requests
    assert [request.partition_key for request in requests] == ["2026-11-07", "2026-11-08"]

    checked = []

    def download(_, url):
        checked.append(official_nba.publication_time(url))
        return None

    monkeypatch.setattr(official_nba.OfficialInjuryReports, "download_if_available", download)
    job = defs.resolve_job_def("capture_injury_pdfs")
    for request in requests:
        assert job.execute_in_process(
            partition_key=request.partition_key, run_config=request.run_config,
        ).success
    assert sorted(checked) == [
        datetime.fromisoformat(value) for value in (
            "2026-11-08T04:15:00Z", "2026-11-08T04:30:00Z",
            "2026-11-08T04:45:00Z", "2026-11-08T05:00:00Z",
        )
    ]


def test_live_sensors_follow_committed_objects_and_pin_reconciliation_inputs(capture, monkeypatch):
    from gestalt_pipelines.injury_reports import schedules
    from gestalt_pipelines.injury_reports.assets import extract, reconcile
    from gestalt_pipelines.injury_reports.schedules import injury_pdf_poll
    from gestalt_pipelines.injury_reports.sensors import injury_report_dates_ready, live, new_injury_pdf
    from gestalt_pipelines.resources import NBAScheduleResource

    storage, _, run_capture = capture
    archive_resource = archive.InjuryReportArchive(season="2026-27")
    schedule = NBAScheduleResource(season="2026-27")
    now = [datetime.fromisoformat("2026-11-08T05:02:00Z")]
    day = date(2026, 11, 8)
    game = {
        "game_id": "1", "game_date": day, "game_time_utc": datetime.fromisoformat("2026-11-08T18:00:00Z"),
        "away_team_id": 1, "home_team_id": 2, "away_team_tricode": "BOS", "home_team_tricode": "NYK",
    }
    schedule_ref = {"uri": "s3://fixture/schedule.json", "sha256": "a" * 64}

    class Clock(datetime):
        @classmethod
        def now(cls, tz):
            return now[0].astimezone(tz)

    def slate(_, requested, reference, *, allow_empty):
        assert reference == schedule_ref
        return ([game] if requested == day else []), reference

    from gestalt_pipelines.rosters.resources import NBARosterArchive
    from gestalt_pipelines.resources import NBAPlayerNamesResource
    roster_refs = [{"uri": "s3://fixture/dated_roster", "sha256": "a"*64}]
    monkeypatch.setattr(NBARosterArchive, "latest_selection", lambda *_: (day.isoformat(), roster_refs[0]))
    monkeypatch.setattr(NBARosterArchive, "load", lambda *_: (
        {"game_date_pacific": day.isoformat(), "teams": {str(game["away_team_id"]): [1], str(game["home_team_id"]): [2]}}, roster_refs[0],
    ))
    monkeypatch.setattr(NBAPlayerNamesResource, "load", lambda _: (
        {1: "Fixture One", 2: "Fixture Two"}, {"uri": "s3://fixture/names", "sha256": "b"*64},
    ))
    monkeypatch.setattr(live, "datetime", Clock)
    monkeypatch.setattr(schedules, "datetime", Clock)
    monkeypatch.setattr(NBAScheduleResource, "load", lambda *_: (b"{}", schedule_ref))
    monkeypatch.setattr(NBAScheduleResource, "games_for_date", slate)
    monkeypatch.setattr(extract, "parse_pdf", lambda _: {
        "game_team_reports": {f"{day}|BOS@NYK": {
            "BOS": {"status": "submitted"}, "NYK": {"status": "not_yet_submitted"},
        }},
        "players": [],
    })
    storage.objects["lake/artifacts/injury_status_by_game_date/season=2026-27/player_repairs.json"] = (b"{}", {})

    def evaluate(sensor, cursor=None):
        resources = {"injury_report_archive": archive_resource}
        if sensor is injury_report_dates_ready:
            resources["nba_schedule"] = schedule
            resources["nba_roster_archive"] = NBARosterArchive()
            resources["nba_player_names"] = NBAPlayerNamesResource(season="2026-27")
        with build_sensor_context(
            repository_def=defs.get_repository_def(), resources=resources, cursor=cursor,
        ) as context:
            return sensor.evaluate_tick(context)

    assert run_capture().success
    extract_tick = evaluate(new_injury_pdf)
    request, = extract_tick.run_requests
    assert request.partition_key == "2026-11-07"
    assert evaluate(new_injury_pdf).run_requests[0].run_key == request.run_key
    extraction_job = defs.resolve_job_def("extract_injury_pdf")
    storage.corrupt_pdf = True
    assert not extraction_job.execute_in_process(
        partition_key=request.partition_key, run_config=request.run_config, tags=request.tags, raise_on_error=False,
    ).success
    storage.corrupt_pdf = False

    later_url = SOURCE.replace("2026-11-07_11_45PM", "2026-11-08_12_15AM")
    storage.now = datetime.fromisoformat("2026-11-08T05:17:00Z")
    now[0] = storage.now + timedelta(minutes=1)

    def download(_, url):
        if url == later_url:
            return official_nba.OfficialInjuryReport(url, official_nba.publication_time(url), storage.now, PDF)
        return None

    monkeypatch.setattr(official_nba.OfficialInjuryReports, "download_if_available", download)
    schedule_context = build_schedule_context(
        scheduled_execution_time=storage.now, repository_def=defs.get_repository_def(),
    )
    capture_request = next(
        request for request in injury_pdf_poll.evaluate_tick(schedule_context).run_requests
        if request.partition_key == "2026-11-08"
    )
    captured = defs.resolve_job_def("capture_injury_pdfs").execute_in_process(
        partition_key=capture_request.partition_key, run_config=capture_request.run_config,
    )
    capture_meta = captured.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    later_uri, = capture_meta["recovered_capture_uris"].value
    assert capture_meta["recovered_count"].value == 1

    assert extraction_job.execute_in_process(
        partition_key=request.partition_key, run_config=request.run_config, tags=request.tags,
    ).success
    remaining, = evaluate(new_injury_pdf, extract_tick.cursor).run_requests
    assert remaining.run_config["ops"]["injury_report_extractions"]["config"]["capture_uri"] == later_uri

    reconcile_tick = evaluate(injury_report_dates_ready)
    request, = reconcile_tick.run_requests
    assert request.partition_key == "2026-11-08"
    config = request.run_config["ops"]["injury_status_by_game_date"]["config"]
    assert config["live"] and config["schedule"] == schedule_ref
    assert "roster_metadata" not in config
    assert "/published_at=20261108T044500Z/" in config["extractions"]["1"]["uri"]
    reconcile_job = defs.resolve_job_def("reconcile_injury_date")
    first = reconcile_job.execute_in_process(
        partition_key=request.partition_key, run_config=request.run_config,
    )
    assert first.success
    first_metadata = first.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    first_pointer, first_etag = archive_resource.read_latest_status(day.isoformat())
    assert first_metadata["latest_update"].value == "published"
    assert first_pointer["changed_game_ids"] == ["1"]
    assert first_pointer["uri"] == first_metadata["statuses"].value["uri"]
    assert evaluate(injury_report_dates_ready, reconcile_tick.cursor).run_requests[0].run_key == request.run_key
    roster_refs[0] = {**roster_refs[0], "sha256": "c"*64}
    changed_roster = evaluate(injury_report_dates_ready, reconcile_tick.cursor).run_requests[0]
    assert changed_roster.run_key != request.run_key
    roster_refs[0] = {**roster_refs[0], "sha256": "a"*64}

    storage.now = now[0] + timedelta(minutes=1)
    now[0] = storage.now + timedelta(minutes=1)
    assert remaining.partition_key == "2026-11-08"
    assert extraction_job.execute_in_process(
        partition_key=remaining.partition_key, run_config=remaining.run_config, tags=remaining.tags,
    ).success
    assert not evaluate(new_injury_pdf).run_requests
    updated, = evaluate(injury_report_dates_ready, reconcile_tick.cursor).run_requests
    assert updated.partition_key == request.partition_key and updated.run_key != request.run_key
    pinned = updated.run_config["ops"]["injury_status_by_game_date"]["config"]["extractions"]["1"]
    assert pinned["uri"] == archive_resource.extraction_uri(later_uri, "v2")
    newer = reconcile_job.execute_in_process(
        partition_key=updated.partition_key, run_config=updated.run_config,
    )
    assert newer.success
    newer_pointer, newer_etag = archive_resource.read_latest_status(day.isoformat())
    assert newer_etag != first_etag
    assert newer_pointer["changed_game_ids"] == ["1"]
    assert not archive_resource.save_latest_status(day.isoformat(), first_pointer, first_etag)

    older = reconcile_job.execute_in_process(
        partition_key=request.partition_key, run_config=request.run_config,
    )
    assert older.success
    older_metadata = older.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    assert older_metadata["latest_update"].value == "reports_changed"
    assert archive_resource.read_latest_status(day.isoformat()) == (newer_pointer, newer_etag)


@pytest.mark.parametrize("upgrade_parser", [False, True])
def test_extraction_failure_recovery_keeps_the_exact_pdf(capture, monkeypatch, upgrade_parser):
    from dagster import DagsterInstance, in_process_executor
    from dagster._core.definitions.job_base import InMemoryJob
    from dagster._core.execution.api import execute_run
    from gestalt_pipelines.injury_reports.assets import extract

    storage, _, run_capture = capture
    assert run_capture().success
    capture_uri = f"s3://nba-gestalt-data/{PREFIX}/capture.json"
    waits = []
    reads = []
    failing = True
    load_capture = archive.InjuryReportArchive.load

    def load(resource, uri):
        reads.append(uri)
        if failing and not upgrade_parser:
            raise ClientError(
                {"Error": {"Code": "ServiceUnavailable"}, "ResponseMetadata": {"HTTPStatusCode": 503}},
                "GetObject",
            )
        return load_capture(resource, uri)

    def parse(_):
        if failing and upgrade_parser:
            raise ValueError("The PDF layout is unreadable.")
        return {"game_team_reports": {"2026-11-07|BOS@NYK": {}}, "players": []}

    monkeypatch.setattr(archive.InjuryReportArchive, "load", load)
    monkeypatch.setattr(extract, "parse_pdf", parse)
    monkeypatch.setattr(extract, "sleep", waits.append)
    job = defs.resolve_job_def("extract_injury_pdf").with_executor_def(in_process_executor)
    config = {"ops": {"injury_report_extractions": {"config": {"capture_uri": capture_uri}}}}

    with DagsterInstance.ephemeral() as instance:
        failed = job.execute_in_process(
            instance=instance, partition_key="2026-11-07", run_config=config,
            tags={"injury_reports/parser_version": "v2"}, raise_on_error=False,
        )
        assert not failed.success
        assert waits == ([] if upgrade_parser else [15, 15])
        assert reads == [capture_uri] * (1 if upgrade_parser else 3)
        metadata = failed.get_step_failure_events()[0].event_specific_data.user_failure_data.metadata
        failure_ref = metadata["extraction_failure"].value
        failure_raw = storage.objects[failure_ref["uri"].removeprefix("s3://nba-gestalt-data/")][0]
        failure = json.loads(failure_raw)
        assert sha256(failure_raw).hexdigest() == failure_ref["sha256"]
        assert failure["stage"] == "injury_report_extractions"
        assert failure["status"] == "failed" and failure["reported_game_dates"] is None
        assert failure["input"] == {
            "capture_uri": capture_uri, "pdf_uri": f"s3://nba-gestalt-data/{PREFIX}/report.pdf",
            "sha256": SHA, "parser_version": "v2",
        }
        assert sum(key.endswith(".failure.json") for key in storage.objects) == 1

        failing = False
        if upgrade_parser:
            monkeypatch.setattr(extract, "PARSER_VERSION", "v3")
            config["ops"]["injury_report_extractions"]["config"] = {"repair_failed_run_id": failed.run_id}
            retry = job.execute_in_process(instance=instance, partition_key="2026-11-07", run_config=config)
        else:
            parent = instance.get_run_by_id(failed.run_id)
            child = instance.create_run_for_job(
                job, run_config=parent.run_config, tags=parent.tags,
                parent_run_id=parent.run_id, root_run_id=parent.run_id,
            )
            retry = execute_run(InMemoryJob(job), child, instance)

    assert retry.success and reads[-1] == capture_uri
    metadata = retry.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    recovery_ref, = metadata["recoveries"].value
    recovery = json.loads(storage.objects[recovery_ref["uri"].removeprefix("s3://nba-gestalt-data/")][0])
    assert recovery["status"] == "recovered"
    assert recovery["original_failure"] == {**failure_ref, "event_id": failure["event_id"]}
    assert recovery["input"]["capture_uri"] == capture_uri
    assert recovery["reported_game_dates"] == ["2026-11-07"]
    version = "v3" if upgrade_parser else "v2"
    assert f"/parser_version={version}/report.json" in recovery["artifact_path"]
