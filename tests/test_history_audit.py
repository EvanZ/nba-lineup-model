from __future__ import annotations

import json

from nba_lineup_model.audit.history import audit_cached_history


def test_cached_history_audit_persists_empty_output_for_no_games(tmp_path) -> None:
    results, paths = audit_cached_history([], raw_dir=tmp_path, output_dir=tmp_path / "audit")

    assert results == []
    assert paths["games"].exists()
    assert paths["summary"].exists()
    assert paths["sources"].exists()
    assert json.loads(paths["manifest"].read_text())["game_count"] == 0
