"""Manual initialization from approved frozen artifacts; never fits a model."""

from collections import Counter

from dagster import AssetExecutionContext, Failure, MaterializeResult, MetadataValue, asset

from ..resources import RatingsArchive
from ..utils import implementation_identity, prepare_initialization, source_references


@asset(
    group_name="ratings", code_version=implementation_identity()["source_sha256"],
    description="Preserve the approved numerical priors, joint HCA prior, and fixed B2B control.",
)
def player_rating_priors(context: AssetExecutionContext, ratings_archive: RatingsArchive) -> MaterializeResult:
    archive = ratings_archive
    inputs = {}
    metadata = {"season": archive.season, "model_version": archive.model_version}
    try:
        inputs.update(source_references(archive.season, archive.model_version))
        prior, etag = archive.selection()
        if prior is not None:
            record, reference = archive.load_priors()
            if reference != prior or archive.selection() != (prior, etag):
                raise ValueError("The initialization selection changed during this run.")
            saved = {**reference, "reused": True}
        else:
            blobs = archive.read_sources()
            record = prepare_initialization(blobs, season=archive.season)
            saved = archive.save_priors(record, prior=prior, expected_etag=etag)
    except Exception as error:
        message = str(error)
        try:
            failure = archive.save_event(run_id=context.run.run_id, message=message, inputs=inputs)
            metadata["ratings_failure"] = MetadataValue.json(failure)
        except Exception as write_error:
            message += f" Could not save the S3 failure record: {write_error}"
        raise Failure(message, metadata=metadata, allow_retries=False) from error
    metadata.update({
        "priors_uri": saved["uri"], "sha256": saved["sha256"], "reused": saved["reused"],
        "player_count": len(record["players"]),
        "forecast_source_counts": MetadataValue.json(dict(Counter(p["forecast_source"] for p in record["players"]))),
        "hca_prior_mean": record["settings"]["hca"]["prior_mean"],
        "fixed_back_to_back": record["settings"]["fixed_back_to_back"],
        "inputs": MetadataValue.json(inputs),
    })
    try:
        original = archive.parent_failure(context)
        if original:
            recovery = archive.save_event(
                run_id=context.run.run_id, message="Ratings initialization validated and published.",
                inputs=inputs, original=original,
            )
            metadata["ratings_recovery"] = MetadataValue.json(recovery)
    except Exception as error:
        raise Failure(f"Initialization published, but linked recovery logging failed: {error}", metadata=metadata) from error
    return MaterializeResult(metadata=metadata)
