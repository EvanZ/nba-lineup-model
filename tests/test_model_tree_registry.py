from __future__ import annotations

import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).parents[1] / "docs" / "assets" / "data" / "model-tree.json"


def test_model_tree_registry_has_valid_primary_lineage_and_metrics() -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    metric_ids = {metric["id"] for metric in registry["metrics"]}
    models = registry["models"]
    model_ids = [model["id"] for model in models]

    assert registry["default_metric"] in metric_ids
    assert len(model_ids) == len(set(model_ids))

    by_id = {model["id"]: model for model in models}
    for model in models:
        parent = model["parent"]
        assert parent is None or parent in by_id
        assert model["docs"].startswith("./")
        assert model["change"]
        assert set(model["metrics"]) <= metric_ids
        assert all(isinstance(value, (int, float)) for value in model["metrics"].values())

    for model in models:
        visited: set[str] = set()
        current = model
        while current["parent"] is not None:
            assert current["id"] not in visited
            visited.add(current["id"])
            current = by_id[current["parent"]]
