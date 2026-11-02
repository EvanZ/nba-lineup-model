import ast
import json
from pathlib import Path

import pytest

from nba_lineup_model.season.source_policy import assert_allowed_url, require_clean_artifact_sources


@pytest.mark.parametrize("payload", [{"espn_game_id":"401900001"},{"game_id":"401900001"},{"source":"https://site.api.espn.com/test"}])
def test_source_policy_rejects_legacy_fields_ids_and_provenance(tmp_path,payload):
    (tmp_path/"sample.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="Forbidden source"):
        require_clean_artifact_sources(tmp_path)


def test_source_policy_accepts_official_nba_and_independent_markets(tmp_path):
    (tmp_path/"sample.json").write_text(json.dumps({"game_id":"0022600001","source":"NBA official CDN","market_source":"The Odds API"}))
    require_clean_artifact_sources(tmp_path)
    assert_allowed_url("https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json")
    with pytest.raises(ValueError):
        assert_allowed_url("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard")


def test_runtime_source_contains_no_forbidden_provider_urls_or_adapter_modules():
    root=Path(__file__).parents[1]
    for path in (root/"src").rglob("*.py"):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith(("http://","https://")):
                assert "espn" not in node.value.casefold(), str(path)
    assert not (root/"src/nba_lineup_model/modeling/current_season_schedule_materialization.py").exists()


def test_source_policy_rejects_parquet_provenance_values(tmp_path):
    import pandas as pd
    pd.DataFrame({"player_id":[1], "identity_source":["espn_roster"]}).to_parquet(tmp_path/"roster.parquet")
    with pytest.raises(ValueError, match="Forbidden source"):
        require_clean_artifact_sources(tmp_path)


@pytest.mark.parametrize("active", [False, True])
def test_promotion_restores_operator_service_state_without_starting_stopped_services(active):
    import subprocess
    script=(Path(__file__).parents[1]/"scripts/promote_gestalt_dev_to_prod.sh").read_text()
    fragment=script[script.index("declare -A dagster_was_active"):script.index("rollback() {")]
    initial="active" if active else "inactive"
    shell='systemctl() { echo '+initial+'; }; sudo() { echo "$*"; };\n'+fragment+'\nrestore_dagster_service_states\n'
    result=subprocess.run(["bash","-c",shell],check=True,capture_output=True,text=True)
    expected="start" if active else "stop"
    assert result.stdout.splitlines()==[f"systemctl {expected} {name}" for name in ("gestalt-dagster-code-server","gestalt-dagster-webserver","gestalt-dagster-daemon")]
