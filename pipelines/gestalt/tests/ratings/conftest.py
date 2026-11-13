"""Small synthetic saved artifacts; tests never fit models or contact S3."""

import io
import json
import tarfile
from hashlib import sha256
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from gestalt_pipelines.ratings.utils import initialization as init
from gestalt_pipelines.ratings.utils.context import FEATURES, NONADDITIVE_FEATURES


@pytest.fixture
def context_values():
    return {"source_season": "2025-26", "feature_set": "nail_v12_1_1_standard_usage",
            "features": list(FEATURES), "nonadditive_features": list(NONADDITIVE_FEATURES),
            "coefficients": [0.25*(i+1) for i in range(10)], "intercept": 0.7,
            "scaler": {"mean": [0.3]*10, "scale": [2.0]*10, "variance": [4.0]*10,
                       "with_mean": True, "with_std": True}}


def parquet(rows):
    stream = io.BytesIO()
    pd.DataFrame(rows).to_parquet(stream, index=False)
    return stream.getvalue()


@pytest.fixture
def inputs(monkeypatch, context_values):
    labels = [label for label, count in init.EXPECTED_SOURCES.items() for _ in range(count)]
    catalog = [{"player_id": i+1, "season": "2026-27", "forecast_source": label,
                "rapm": init.REPLACEMENT_PRIOR if label == "replacement_cold_start_prior" else i/19,
                "prior_rating": i/19, "player_name": f"Player {i}", "optional_value": None}
               for i, label in enumerate(labels)]
    profiles = [{"player_id": r["player_id"], "season": "2026-27", "usage_pct": 0.2} for r in catalog]
    comparison = [{"player_id": r["player_id"], "prior_rating": r["prior_rating"],
                   "raw_prediction": r["prior_rating"]+0.125}
                  for r in catalog if r["forecast_source"].startswith("forward_")]
    members = {"generation_catalog_comparison.parquet": parquet(comparison),
               "last_observed_states.parquet": parquet([
                   {"player_id": 9999, "season": "2025-26", "rapm": -1.5, "on_court_possessions": 1200},
               ])}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, raw in members.items():
            item = tarfile.TarInfo(name); item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
    model = io.BytesIO()
    joblib.dump(SimpleNamespace(feature_names_in_=["target_age"]), model)
    blobs = {
        "catalog": parquet(catalog), "profiles": parquet(profiles), "context": b"saved context",
        "catalog_metadata": json.dumps({"player_count": 524, "target_season": "2026-27"}).encode(),
        "schedule_metadata": parquet([
            {"season": "2024-25", "schedule_control_raw_weight": -1.0, "schedule_training_stint_count": 10},
            {"season": "2025-26", "schedule_control_raw_weight": -2.0, "schedule_training_stint_count": 30},
        ]),
        "hca": json.dumps({"target_season": "2026-27", "source_through_season": "2025-26",
                           "prior_mean": 1.69, "prior_variance": 0.33,
                           "prior_equivalent_possessions": 42883.0,
                           "observation_variance_per_100": 14358.0}).encode(),
        "aging_metadata": json.dumps({"prediction_season": "2026-27", "completed_through": "2025-26",
                                      "features": ["target_age"]}).encode(),
        "aging_predictor": model.getvalue(), "aging_evidence": stream.getvalue(),
    }
    references = {name: {"uri": f"s3://nba-gestalt-data/test/{name}", "sha256": sha256(raw).hexdigest()}
                  for name, raw in blobs.items()}
    monkeypatch.setattr(init, "source_references", lambda season, version: references)
    monkeypatch.setattr(init, "export_context", lambda raw: context_values)
    return blobs, references, catalog


@pytest.fixture
def record(inputs):
    return init.prepare_initialization(inputs[0], season="2026-27")
