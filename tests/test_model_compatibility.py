import ast
import dataclasses
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from nba_lineup_model.orchestration.model_compatibility import (
    calculation_inventory, require_compatible_calculations,
)

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED = "1459c83f82b1fa50d901048e65bb39c9fd678204"
POLICY = Path("src/nba_lineup_model/orchestration/model_compatibility") / (ACCEPTED + ".json")


def test_sealed_copy_needs_no_git_and_unrelated_reports_do_not_block(tmp_path, monkeypatch):
    for name in [*calculation_inventory(ROOT), str(POLICY)]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("Git must not run"))
    assert require_compatible_calculations(tmp_path, ACCEPTED)["verified_file_count"] > 50
    (tmp_path / "src/nba_lineup_model/modeling/unrelated_report.py").write_text("VALUE = 99\n")
    require_compatible_calculations(tmp_path, ACCEPTED)
    dependency = tmp_path / "src/nba_lineup_model/models/baselines.py"
    dependency.write_text(dependency.read_text() + "\nCHANGED_CALCULATION = True\n")
    with pytest.raises(ValueError, match="explicitly approved causal replay"):
        require_compatible_calculations(tmp_path, ACCEPTED)


def test_unknown_model_cannot_receive_an_implicit_compatibility_exception():
    with pytest.raises(ValueError, match="No reviewed"):
        require_compatible_calculations(ROOT, "0" * 40)


def old_source(relative):
    source = subprocess.check_output(["git", "-C", str(ROOT), "show", ACCEPTED + ":" + relative])
    policy = json.loads((ROOT / POLICY).read_text())
    assert hashlib.sha256(source).hexdigest() == policy["accepted_files"][relative]
    return source


def test_updater_arithmetic_and_context_implementation_are_byte_identical():
    names = ["modeling/in_season_nail.py",
             "modeling/in_season_minutes.py", "modeling/causal_hca.py",
             "modeling/game_script_minutes.py", "modeling/in_season_nail_monitor.py",
             "models/baselines.py"]
    for name in names:
        relative = "src/nba_lineup_model/" + name
        assert (ROOT / relative).read_bytes() == old_source(relative), name


@pytest.mark.parametrize("name", ["cold_start_exposure", "replacement_level"])
def test_plot_import_relocation_does_not_change_any_calculation(name):
    relative = f"src/nba_lineup_model/modeling/{name}.py"
    class WithoutPlotImport(ast.NodeTransformer):
        def visit_Import(self, node):
            if len(node.names) == 1 and node.names[0].name == "matplotlib.pyplot":
                return None
            return node
    def tree(source):
        return ast.dump(WithoutPlotImport().visit(ast.parse(source)), include_attributes=False)
    assert tree(old_source(relative)) == tree((ROOT / relative).read_bytes())


def baseline_module(relative, tmp_path):
    name = "compat_baseline_" + Path(relative).stem
    path = tmp_path / (name + ".py")
    path.write_bytes(old_source(relative))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def signature(value):
    """Compare all loaded values, independent of baseline/current class names."""
    import joblib
    if dataclasses.is_dataclass(value):
        return {f.name:signature(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k):signature(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [signature(v) for v in value]
    return joblib.hash(value)


def test_ec2_bound_loader_and_selected_data_equivalence(tmp_path, monkeypatch):
    selected = os.environ.get("GESTALT_COMPATIBILITY_RELEASE")
    if not selected:
        pytest.skip("Set GESTALT_COMPATIBILITY_RELEASE to verified EC2 inputs")
    from nba_lineup_model.web_api.release_serving import ReleaseServing
    from nba_lineup_model.web_api.inference import LineupEvaluator, MODEL_ARTIFACT, DEFAULT_ARTIFACTS_DIR
    from nba_lineup_model.orchestration.forward_daily import accepted_chain
    from nba_lineup_model.season.daily_data_artifact import verify_daily_data_artifact
    release = ReleaseServing.load(selected)
    assert release is not None
    baseline = baseline_module("src/nba_lineup_model/web_api/inference.py", tmp_path)
    monkeypatch.chdir(release.bindings["historical_runtime"])
    kwargs = release.evaluator_kwargs()
    for field, variable in {
        "preseason_rankings_override":"GESTALT_PRESEASON_RANKINGS_PATH",
        "preseason_profiles_override":"GESTALT_PRESEASON_PROFILES_PATH",
        "current_serving_bundle":"GESTALT_CURRENT_SEASON_SERVING_BUNDLE",
        "current_lineups_bundle":"GESTALT_CURRENT_SEASON_LINEUPS_BUNDLE",
    }.items():
        monkeypatch.setenv(variable, str(kwargs[field]))
    latest = json.loads((Path(DEFAULT_ARTIFACTS_DIR) / MODEL_ARTIFACT /
                        baseline.DISPLAY_SEASON / "latest.json").read_text())
    assert latest["run_id"] == kwargs["run_id"], "Baseline must select identical inputs"
    old = baseline.LineupEvaluator.from_latest_artifact()
    expected = signature(old)
    del old
    gc.collect()
    current = LineupEvaluator.from_latest_artifact(**kwargs)
    actual = signature(current)
    assert actual == expected, [k for k in actual if actual[k] != expected.get(k)]
    del current, actual, expected
    gc.collect()
    old_data = baseline_module("src/nba_lineup_model/season/daily_data_artifact.py", tmp_path)
    _, data = accepted_chain(release.root)
    games = set()
    for path in data:
        before = old_data.verify_daily_data_artifact(path)
        after = verify_daily_data_artifact(path)
        assert before == after
        games.update(after["identity"]["game_ids"])
    expected_games = release.manifest["approved_manifest"]["identity"]["game_count"]
    assert len(games) == expected_games
    print(f"Verified loader-state equality and {len(data)} dates / {len(games)} games")


def test_reviewed_policy_matches_current_calculation_closure():
    result = require_compatible_calculations(ROOT, ACCEPTED)
    policy = json.loads((ROOT / POLICY).read_text())
    changed = {p for p in policy["compatible_files"]
               if policy["compatible_files"][p] != policy["accepted_files"].get(p)}
    assert changed == set(policy["reviewed_changes"])
    assert result["verified_file_count"] == len(policy["compatible_files"])


def test_identity_mapping_and_reconciliation_leave_other_daily_calculations_unchanged():
    # Reverse only the four reviewed identity-matching edits, then require every
    # other byte (including updater arithmetic and observation weights) to match.
    relative = "src/nba_lineup_model/season/daily_model_replay.py"
    source = (ROOT / relative).read_text()
    reviewed_edits = [('        names = {normalized_name}\n'
      '        # Official NBA injury PDF (2026-11-04, MIL@BOS) uses Christopher;\n'
      '        # official liveData identifies Chris Cenac Jr. as NBA player 1643416.\n'
      '        # Bind this reviewed alias to that ID and canonical name, never to a\n'
      '        # general Chris/Christopher substitution or a fuzzy name match.\n'
      '        if player_id == 1643416 and normalized_name == "chriscenac":\n'
      '            names.add("christophercenac")\n'
      '        for name in names:\n'
      '            identity.setdefault((team, name), set()).add(player_id)\n'
      '            identity_by_name.setdefault(name, set()).add((team, player_id))',
      '        identity.setdefault((team, normalized_name), set()).add(player_id)\n'
      '        identity_by_name.setdefault(normalized_name, set()).add((team, player_id))'),
     ('mapped_report: dict[tuple[str, int], dict[str, object]] = {}',
      'mapped_report: dict[tuple[str, str], dict[str, object]] = {}'),
     ('mapped_report[(team, candidates[0])] = normalized', 'mapped_report[key] = normalized'),
     ('report = mapped_report.get((team, player_id), {})',
      'report = mapped_report.get((team, _normalize_name(player["player_name"])), {})')]
    for current, accepted in reviewed_edits:
        assert source.count(current) == 1
        source = source.replace(current, accepted)
    # #65 changes only state-adoption orchestration and its compatibility checks.
    # Every other calculation, including observation weights and the solver,
    # remains the accepted implementation after reversing the reviewed alias.
    def functions(raw):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(raw).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    before, after = functions(old_source(relative)), functions(source)
    permitted = {'replay_daily_model_chain', '_load_parent_state', '_longest_compatible_prefix'}
    assert set(before) == set(after)
    assert {name for name in before if before[name] != after[name]} <= permitted
