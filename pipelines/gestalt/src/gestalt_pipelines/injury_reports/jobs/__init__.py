"""Dagster jobs for injury reports."""

from dagster import AssetSelection, define_asset_job

from ..assets import injury_pdf_archive, injury_report_extractions, injury_status_by_game_date


capture_injury_pdfs = define_asset_job(
    "capture_injury_pdfs",
    selection=AssetSelection.assets(injury_pdf_archive),
    description="Fill missing official NBA PDF captures for an Eastern publication-date partition.",
)


extract_injury_pdf = define_asset_job(
    "extract_injury_pdf",
    selection=AssetSelection.assets(injury_report_extractions),
    description="Extract the latest archived report for an Eastern publication date or an exact capture URI.",
)


reconcile_injury_date = define_asset_job(
    "reconcile_injury_date",
    selection=AssetSelection.assets(injury_status_by_game_date),
    description="Reconcile one Pacific slate from selected S3 injury, schedule, and roster evidence.",
)
