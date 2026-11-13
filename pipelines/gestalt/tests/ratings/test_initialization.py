"""Coverage, byte identity, source policy, and saved-value preservation."""

import json

import pytest

from gestalt_pipelines.ratings.utils import initialization as init
from gestalt_pipelines.ratings.utils.sources import source_references


def test_preserves_original_values_and_frozen_alignment(inputs, monkeypatch):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    def no_fit(*args, **kwargs):
        raise AssertionError("Initialization must not fit any model")
    for cls in (Ridge, Pipeline, StandardScaler):
        monkeypatch.setattr(cls, "fit", no_fit)
    result = init.prepare_initialization(inputs[0], season="2026-27")
    decoded = json.loads(init.json_bytes(result))
    assert decoded["players"] == inputs[2]
    assert len(decoded["profiles"]) == 524
    assert decoded["aging"]["alignment"]["offset"] == pytest.approx(-0.125, abs=1e-14)
    assert decoded["aging"]["alignment"]["anchor_count"] == 451
    assert decoded["settings"]["fixed_back_to_back"] == -1.75
    assert decoded["aging"]["original_1e_8_check_passed"] is False


def test_checksum_mismatch_is_rejected_before_unpickling(inputs, monkeypatch):
    import joblib
    blobs, _, _ = inputs
    def no_load(*args, **kwargs):
        raise AssertionError("Unverified bytes must never be unpickled")
    monkeypatch.setattr(joblib, "load", no_load)
    with pytest.raises(ValueError, match="checksum"):
        init.prepare_initialization({**blobs, "aging_predictor": b"changed"}, season="2026-27")


def test_missing_source_is_explicit(inputs):
    blobs = dict(inputs[0]); del blobs["profiles"]
    with pytest.raises(ValueError, match="incomplete"):
        init.prepare_initialization(blobs, season="2026-27")


def test_missing_profile_does_not_publish(record):
    record["profiles"].pop()
    with pytest.raises(ValueError, match="coverage"):
        init.validate_initialization(record, "2026-27", "v1")


def test_source_policy_and_nonfinite_values_are_rejected(record):
    record["players"][0]["source"] = "https://sports.api.espn.com/"
    with pytest.raises(ValueError, match="Unapproved"):
        init.validate_initialization(record, "2026-27", "v1")
    del record["players"][0]["source"]
    record["players"][0]["rapm"] = float("nan")
    with pytest.raises(ValueError):
        init.validate_initialization(record, "2026-27", "v1")


def test_unknown_season_has_no_substitute_inputs():
    with pytest.raises(ValueError, match="No approved"):
        source_references("2027-28", "v1")
