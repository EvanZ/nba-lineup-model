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


def test_promotion_never_restarts_deprecated_shared_dagster():
    script=(Path(__file__).parents[1]/"scripts/promote_gestalt_dev_to_prod.sh").read_text()
    assert "Production promotion blocked: deprecated service is active" in script
    assert "legacy_dagster_services" in script
    assert "restore_dagster_service_states" not in script
    assert 'systemctl start "$service"' not in script


def test_legacy_shared_dagster_installer_is_a_fail_closed_tombstone():
    root = Path(__file__).parents[1]
    script = (root / "scripts/deploy_gestalt_orchestration.sh").read_text()
    assert "shared NBA GESTALT Dagster pipeline is deprecated" in script
    assert "exit 64" in script
    assert "systemctl enable" not in script
    assert "dagster-daemon run" not in script

    for name in (
        "gestalt-dagster-code-server.service",
        "gestalt-dagster-webserver.service",
        "gestalt-dagster-daemon.service",
    ):
        unit = (root / "deploy" / name).read_text()
        assert "DEPRECATED" in unit
        assert "RefuseManualStart=true" in unit


def test_promotion_uses_exact_reviewed_dev_runtime_and_clean_schedule_overlay():
    script = (Path(__file__).parents[1] / "scripts/promote_gestalt_dev_to_prod.sh").read_text()
    assert 'DEV_RUNTIME="$(readlink -f "${GESTALT_APPROVED_DEV_RUNTIME:-$DEV_STATE_ROOT/live-release}")"' in script
    assert 'DEV_RELEASE="$DEV_RUNTIME/code"' in script
    assert "dev_commit=\"$(jq -er '.identity.code_revision' \"$DEV_RUNTIME/runtime-manifest.json\")\"" in script
    assert 'DEV_SCHEDULE_ROOT="$DEV_RUNTIME/schedule-snapshot"' in script
    assert 'FRONTEND_DIST="${GESTALT_PROD_FRONTEND_DIST:-$DEV_RUNTIME/web}"' in script
    assert 'sudo systemctl stop "${collector_services[@]}"' not in script
    assert 'sudo systemctl restart "${collector_services[@]}"' not in script
    assert 'SOURCE_REPO/deploy/$unit' not in script



def test_production_release_cannot_replace_canonical_live_observations():
    root = Path(__file__).parents[1]
    script = (root / "scripts" / "promote_gestalt_dev_to_prod.sh").read_text()
    unit = (root / "deploy" / "gestalt-web.service").read_text()
    assert 'atomic_symlink "$schedule_overlay" "$SCHEDULE_CURRENT"' not in script
    assert 'canonical_schedule_target="$(readlink -f "$SCHEDULE_CURRENT"' in script
    assert "Production promotion changed the canonical live-observation pointer" in script
    assert "GESTALT_SCHEDULE_STATE_DIR=/var/lib/nba-gestalt/live-release/schedule" in unit
    assert "GESTALT_LIVE_SCORE_STATE_DIR=/var/lib/nba-gestalt/schedule-serving/current" in unit

def test_protected_collectors_write_only_to_clean_schedule_overlay():
    root = Path(__file__).parents[1]
    for name in (
        "gestalt-collection-code-server.service",
        "gestalt-collection-webserver.service",
        "gestalt-collection-daemon.service",
    ):
        unit = (root / "deploy" / name).read_text()
        assert "GESTALT_SCHEDULE_STATE_DIR=/var/lib/nba-gestalt/schedule-serving/current" in unit
        assert "GESTALT_MODEL_INPUT_SCHEDULE_STATE_DIR=/var/lib/nba-gestalt/schedule-serving/current" in unit
        assert "GESTALT_SCHEDULE_STATE_DIR=/var/lib/nba-gestalt/schedule\n" not in unit
