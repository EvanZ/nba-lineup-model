"""Download one official NBA injury report without parsing it."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

from dagster import ConfigurableResource

EASTERN = ZoneInfo("America/New_York")
REPORT_URL = re.compile(
    r"https://ak-static\.cms\.nba\.com/referee/injury/Injury-Report_"
    r"(?P<published>[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}_[0-9]{2}(?:AM|PM))\.pdf"
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
    def download(self, source_url: str) -> OfficialInjuryReport:
        published_at = publication_time(source_url)
        request = Request(source_url, headers={"User-Agent": "nba-gestalt/1.0"})
        with build_opener(_NoRedirect()).open(request, timeout=20) as response:
            pdf = response.read(MAX_PDF_BYTES + 1)
        if (
            len(pdf) > MAX_PDF_BYTES
            or not pdf.startswith(b"%PDF-")
            or not pdf.rstrip().endswith(b"%%EOF")
        ):
            raise ValueError("The official response is not a supported PDF.")
        return OfficialInjuryReport(source_url, published_at, datetime.now(UTC), pdf)
