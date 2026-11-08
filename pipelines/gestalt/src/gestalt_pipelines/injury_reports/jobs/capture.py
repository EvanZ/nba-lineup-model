"""Manual injury-report capture job."""

from dagster import AssetSelection, define_asset_job

from ..assets.capture import injury_pdf_archive

capture_injury_pdfs = define_asset_job(
    "capture_injury_pdfs",
    selection=AssetSelection.assets(injury_pdf_archive),
    description="Capture one official NBA injury report PDF with an approved checksum.",
)
