"""Download one official NBA injury report without parsing it."""

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

from dagster import ConfigurableResource

EASTERN = ZoneInfo("America/New_York")
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


def publication_time(source_url: str) -> datetime:
    match = REPORT_URL.fullmatch(source_url)
    if match is None:
        raise ValueError("Expected an official NBA injury report PDF URL.")
    wall_time = datetime.strptime(match["published"], "%Y-%m-%d_%I_%M%p")
    eastern = wall_time.replace(tzinfo=EASTERN)
    if eastern.utcoffset() != eastern.replace(fold=1).utcoffset():
        raise ValueError("The report URL does not identify a unique Eastern publication instant.")
    return eastern.astimezone(UTC)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OfficialInjuryReports(ConfigurableResource):
    def download_latest(self, publication_date: date) -> OfficialInjuryReport:
        now = datetime.now(EASTERN)
        if publication_date > now.date():
            raise ValueError(f"Official NBA injury reports are not available yet for {publication_date}.")

        latest_slot = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        last_slot_of_day = datetime.combine(publication_date, time(23, 45), tzinfo=EASTERN)
        cursor = min(latest_slot, last_slot_of_day)
        while cursor.date() == publication_date:
            url = f"{REPORT_URL_PREFIX}{publication_date}_{cursor:%I_%M%p}.pdf"
            try:
                return self.download(url)
            except HTTPError as error:
                if error.code not in {403, 404}:
                    raise
                error.close()
            cursor -= timedelta(minutes=15)

        raise FileNotFoundError(f"No official NBA injury report found for {publication_date}.")

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
        return OfficialInjuryReport(source_url, publication_time(source_url), datetime.now(UTC), pdf)
