"""Download one official NBA injury report without parsing it."""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from io import BytesIO
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dagster import ConfigurableResource
from pypdf import PdfReader

from ...utils import EASTERN

REPORT_URL_PREFIX = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_"
REPORT_URL = re.compile(
    re.escape(REPORT_URL_PREFIX)
    + r"(?P<published>[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}_[0-9]{2}(?:AM|PM))\.pdf"
)
MAX_PDF_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class OfficialInjuryReport:
    source_url: str
    published_at: datetime
    first_observed_at: datetime
    pdf: bytes


def _utc_instants(wall_time: datetime) -> tuple[datetime, ...]:
    instants = set()
    for fold in (0, 1):
        instant = wall_time.replace(tzinfo=EASTERN, fold=fold).astimezone(UTC)
        if instant.astimezone(EASTERN).replace(tzinfo=None) == wall_time:
            instants.add(instant)
    return tuple(sorted(instants))


def publication_instants(source_url: str) -> tuple[datetime, ...]:
    match = REPORT_URL.fullmatch(source_url)
    if match is None:
        raise ValueError("Expected an official NBA injury report PDF URL.")
    wall_time = datetime.strptime(match["published"], "%Y-%m-%d_%I_%M%p")
    return _utc_instants(wall_time)


def publication_time(source_url: str, pdf: bytes | None = None) -> datetime:
    instants = publication_instants(source_url)
    if len(instants) == 1:
        return instants[0]
    if not instants:
        raise ValueError("The report URL names a nonexistent Eastern publication time.")
    if pdf is not None:
        metadata = PdfReader(BytesIO(pdf)).metadata
        created_at = metadata.creation_date if metadata else None
        if created_at is not None and created_at.utcoffset() is not None:
            created_slot = created_at.astimezone(UTC).replace(second=0, microsecond=0)
            if created_slot in instants:
                return created_slot
    raise ValueError("The report PDF does not verify a unique Eastern publication instant.")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OfficialInjuryReports(ConfigurableResource):
    def publication_slots(self, publication_date: date) -> Iterator[tuple[tuple[datetime, ...], str]]:
        now = datetime.now(EASTERN)
        if publication_date > now.date():
            raise ValueError(f"Official NBA injury reports are not available yet for {publication_date}.")

        latest_slot = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        last_slot_of_day = datetime.combine(publication_date, time(23, 45), tzinfo=EASTERN)
        cursor = min(latest_slot, last_slot_of_day)
        while cursor.date() == publication_date:
            instants = _utc_instants(cursor.replace(tzinfo=None))
            if instants:
                yield instants, f"{REPORT_URL_PREFIX}{publication_date}_{cursor:%I_%M%p}.pdf"
            cursor -= timedelta(minutes=15)

    def download_if_available(self, source_url: str) -> OfficialInjuryReport | None:
        try:
            return self.download(source_url)
        except HTTPError as error:
            if error.code not in {403, 404}:
                raise
            error.close()
            return None

    def download(self, source_url: str) -> OfficialInjuryReport:
        if REPORT_URL.fullmatch(source_url) is None:
            raise ValueError("Expected an official NBA injury report PDF URL.")
        request = Request(source_url, headers={"User-Agent": "nba-gestalt/1.0"})
        with build_opener(_NoRedirect()).open(request, timeout=20) as response:
            pdf = response.read(MAX_PDF_BYTES + 1)
        if (
            len(pdf) > MAX_PDF_BYTES
            or not pdf.startswith(b"%PDF-")
            or not pdf.rstrip().endswith(b"%%EOF")
        ):
            raise ValueError("The official response is not a supported PDF.")
        return OfficialInjuryReport(source_url, publication_time(source_url, pdf), datetime.now(UTC), pdf)
