"""Coverage, byte identity, source policy, and saved-value preservation."""

import json

import pytest

from gestalt_pipelines.ratings.utils import initialization as init
from gestalt_pipelines.ratings.utils.sources import source_references


def test_preserves_original_values_without_loading_models(inputs, monkeypatch):
    import builtins
    original = builtins.__import__
    def checked(name, *args, **kwargs):
        if name.split(".")[0] in {"joblib", "sklearn", "nba_lineup_model"}:
            raise AssertionError("Initialization must use numerical priors only")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked)
    result = init.prepare_initialization(inputs[0], season="2026-27")
    decoded = json.loads(init.json_bytes(result))
    assert decoded["players"] == inputs[2]
    assert set(decoded["inputs"]) == {"catalog", "catalog_metadata", "hca", "schedule_metadata"}
    assert decoded["settings"]["fixed_back_to_back"] == -1.75
    assert decoded["settings"]["hca"]["prior_mean"] == 1.69
    assert not {"profiles", "context", "aging"}.intersection(decoded)


def test_checksum_mismatch_is_rejected_before_parsing(inputs, monkeypatch):
    def no_parse(*args, **kwargs):
        raise AssertionError("Unverified bytes must never be parsed")
    monkeypatch.setattr(init, "_table", no_parse)
    with pytest.raises(ValueError, match="checksum"):
        init.prepare_initialization({**inputs[0], "catalog": b"changed"}, season="2026-27")


def test_missing_source_is_explicit(inputs):
    blobs = dict(inputs[0]); del blobs["catalog"]
    with pytest.raises(ValueError, match="incomplete"):
        init.prepare_initialization(blobs, season="2026-27")


def test_missing_player_does_not_publish(record):
    record["players"].pop()
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


def test_saved_priors_can_retain_unused_provenance(record):
    record["inputs"]["context"] = {"uri": "s3://nba-gestalt-data/unused.joblib", "sha256": "a"*64}
    record["profiles"] = [{"unused": True}]
    record["context"] = {"unused": True}
    record["aging"] = {"unused": True}
    init.validate_initialization(record, "2026-27", "v1")
