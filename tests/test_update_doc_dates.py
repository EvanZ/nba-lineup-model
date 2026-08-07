from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "update_doc_dates.py"
    spec = importlib.util.spec_from_file_location("update_doc_dates", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_with_last_updated_adds_and_replaces_front_matter() -> None:
    module = _load_module()

    added = module.with_last_updated("# Example\n", "2026-08-07")
    replaced = module.with_last_updated(
        "---\ntitle: Example\nlast_updated: \"2000-01-01\"\n---\n# Example\n",
        "2026-08-07",
    )

    assert added == '---\nlast_updated: "2026-08-07"\n---\n\n# Example\n'
    assert replaced == '---\ntitle: Example\nlast_updated: "2026-08-07"\n---\n# Example\n'
