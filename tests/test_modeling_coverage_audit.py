from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nba_lineup_model.audit.modeling_coverage import build_modeling_coverage_audit


def test_modeling_coverage_audit_persists_season_team_and_manifest(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.parquet"
    pd.DataFrame(
        {
            "season": ["2024-25", "2024-25"],
            "season_type": ["regular", "regular"],
            "game_id": ["001", "002"],
            "home_team_id": [1, 3],
            "home_team_tricode": ["AAA", "CCC"],
            "away_team_id": [2, 1],
            "away_team_tricode": ["BBB", "AAA"],
        }
    ).to_parquet(catalog_path, index=False)
    stints_dir = tmp_path / "analytical" / "rapm_stints" / "2024-25" / "regular"
    stints_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "game_id": ["001", "001"],
            "home_team_id": [1, 1],
            "home_team_tricode": ["AAA", "AAA"],
            "away_team_id": [2, 2],
            "away_team_tricode": ["BBB", "BBB"],
        }
    ).to_parquet(stints_dir / "part-00000.parquet", index=False)
    (stints_dir / "_manifest.json").write_text("{}\n")
    players_dir = tmp_path / "curated" / "players" / "2024-25" / "regular"
    players_dir.mkdir(parents=True)
    (players_dir / "_manifest.json").write_text("{}\n")

    result = build_modeling_coverage_audit(
        catalog_path=catalog_path,
        analytical_dir=tmp_path / "analytical",
        curated_dir=tmp_path / "curated",
        output_dir=tmp_path / "audit",
    )

    season = result.season_coverage.iloc[0]
    assert season.catalog_games == 2
    assert season.modeled_games == 1
    assert season.missing_games == 1
    assert season.coverage_pct == 50.0
    assert season.coverage_status == "critical"
    teams = result.team_coverage.set_index("team")
    assert teams.loc["AAA", "coverage_pct"] == 50.0
    assert teams.loc["BBB", "coverage_pct"] == 100.0
    assert teams.loc["CCC", "coverage_pct"] == 0.0
    assert teams.loc["CCC", "coverage_status"] == "critical"
    assert all(path.is_file() for path in result.paths.values())
    manifest = json.loads(result.paths["manifest"].read_text())
    assert manifest["thresholds"] == {
        "warning_coverage_pct": 95.0,
        "critical_coverage_pct": 90.0,
    }
    assert manifest["inputs"]["catalog_sha256"]
    assert manifest["outputs"]["team_coverage"]["sha256"]
