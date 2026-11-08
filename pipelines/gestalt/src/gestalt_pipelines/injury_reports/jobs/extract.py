"""Manual injury-report extraction job."""

from dagster import AssetSelection, define_asset_job

from ..assets.extract import injury_report_extractions

extract_injury_pdf = define_asset_job(
    "extract_injury_pdf",
    selection=AssetSelection.assets(injury_report_extractions),
    description="Extract the official NBA injury report identified by one archived capture URI.",
)
