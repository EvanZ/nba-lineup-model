"""Small synthetic saved artifacts; tests never fit models or contact S3."""

import io
import json
from hashlib import sha256

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from gestalt_pipelines.ratings.resources import RatingsArchive

from gestalt_pipelines.ratings.utils import initialization as init
def parquet(rows):
    stream = io.BytesIO()
    pd.DataFrame(rows).to_parquet(stream, index=False)
    return stream.getvalue()


@pytest.fixture
def inputs(monkeypatch):
    labels = [label for label, count in init.EXPECTED_SOURCES.items() for _ in range(count)]
    catalog = [{"player_id": i+1, "season": "2026-27", "forecast_source": label,
                "rapm": init.REPLACEMENT_PRIOR if label == "replacement_cold_start_prior" else i/19,
                "prior_rating": i/19, "player_name": f"Player {i}", "optional_value": None}
               for i, label in enumerate(labels)]
    blobs = {
        "catalog": parquet(catalog),
        "catalog_metadata": json.dumps({"player_count": 524, "target_season": "2026-27"}).encode(),
        "schedule_metadata": parquet([
            {"season": "2024-25", "schedule_control_raw_weight": -1.0, "schedule_training_stint_count": 10},
            {"season": "2025-26", "schedule_control_raw_weight": -2.0, "schedule_training_stint_count": 30},
        ]),
        "hca": json.dumps({"target_season": "2026-27", "source_through_season": "2025-26",
                           "prior_mean": 1.69, "prior_variance": 0.33,
                           "prior_equivalent_possessions": 42883.0,
                           "observation_variance_per_100": 14358.0}).encode(),
    }
    references = {name: {"uri": f"s3://nba-gestalt-data/test/{name}", "sha256": sha256(raw).hexdigest()}
                  for name, raw in blobs.items()}
    monkeypatch.setattr(init, "source_references", lambda season, version: references)
    return blobs, references, catalog


@pytest.fixture
def record(inputs):
    return init.prepare_initialization(inputs[0], season="2026-27")


class S3:
    def __init__(self):
        self.objects = {}
        self.race = False

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        raw = self.objects[Key]
        return {"Body": io.BytesIO(raw), "ETag": sha256(raw).hexdigest()}

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None, **kwargs):
        if (IfNoneMatch == "*" and Key in self.objects) or (self.race and Key.endswith("latest.json")):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if IfMatch is not None and (Key not in self.objects or sha256(self.objects[Key]).hexdigest() != IfMatch):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body
        return {}


@pytest.fixture
def archive(monkeypatch):
    s3 = S3()
    monkeypatch.setattr("gestalt_pipelines.ratings.resources.archive.boto3.client", lambda service: s3)
    return RatingsArchive(season="2026-27"), s3
