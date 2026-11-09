"""Dagster resources for injury reports."""

from .archive import InjuryReportArchive
from .official_nba import OfficialInjuryReports

__all__ = ["InjuryReportArchive", "OfficialInjuryReports"]
