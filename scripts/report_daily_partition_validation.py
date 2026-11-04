"""Render verified partition results and complete affected-player coverage."""
from __future__ import annotations
import argparse
import html
import json
import re
import subprocess
import urllib.request
from pathlib import Path
import pandas as pd
from nba_lineup_model.orchestration.daily_work import atomic_json, read_object, verify_reference
from nba_lineup_model.season.production_site_release import verify_production_site_release
from nba_lineup_model.season.serving_selection import verify_schedule_selection

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "live", "browser-results", "before", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--api", required=True)
    p.add_argument("--test-unit", required=True)
    p.add_argument("--finding", action="append", default=[], help="Verified unresolved finding for the human review")
    args = p.parse_args()
    spec = read_object(args.work / "work.json")
    status = read_object(args.work / "status.json")
    browser = read_object(args.browser_results)
    runtime = verify_production_site_release(args.live.resolve())
    approved = runtime["approved_manifest"]["identity"]
    validation = approved["api_validation"]
    model_ref = spec["identity"]["execution"]["target_model_manifest"]
    model_path = verify_reference(model_ref).parent
    model = read_object(model_path / "manifest.json")
    assert model["bundle_sha256"] == approved["model"]["bundle_sha256"]
    assert status["status"] == "serving_validated" and len(status["reused_stages"]) == 5
    assert browser["status"] == "passed" and not browser["errors"]
    assert set(browser["viewports"]) == {1440, 768, 390}
    assert Path(browser["web_root"]).resolve() == (args.live / "web").resolve()
    assert browser["api"].rstrip("/") == args.api.rstrip("/")
    observations = pd.read_parquet(model_path / model["identity"]["components"]["minutes_observations"]["path"])
    expected = {(int(r.player_id),str(r.game_id),bool(r.played)) for r in observations.itertuples()}
    checked = {(int(r["player_id"]),str(r["game_id"]),bool(r["played"])) for r in browser["checks"]}
    assert expected == checked and len(expected) == len(browser["checks"])
    assert browser["players"] == len(checked)
    assert browser["played"] == sum(item[2] for item in expected)
    assert browser["missed"] == sum(not item[2] for item in expected)
    with urllib.request.urlopen(args.api.rstrip("/")+"/api/schedule-performance", timeout=60) as response:
        performance = json.load(response)
    before = read_object(args.before)
    assert performance["windows"] == before["performance"]["windows"]
    selection = verify_schedule_selection(args.live / "schedule")
    prior_ref = spec["identity"]["execution"]["schedule_files"]["forecast-selections.json"]
    assert selection["selection_map_sha256"] == read_object(verify_reference(prior_ref))["selection_map_sha256"]
    log = subprocess.check_output(["journalctl","-u",args.test_unit,"-o","cat","--no-pager"],text=True)
    result = subprocess.check_output(["systemctl","show",args.test_unit,"--property=Result","--value"],text=True).strip()
    assert result == "success", "Supplied test unit did not finish successfully"
    tests = re.findall(r"(\d+) passed",log)
    assert tests, "No successful test result in supplied journal"
    layout_warnings = [row["width"] for row in browser["layout"] if row["overflow"]]
    checks = {**validation, "api_validation_status":validation["status"],
              "status":"passed_with_findings" if layout_warnings or args.finding else "passed",
              "review_findings":args.finding, "browser_player_pages":len(checked), "browser_played":browser["played"],
              "browser_missed":browser["missed"], "reused_stages":status["reused_stages"],
              "tests_passed":int(tests[-1]), "historical_forecasts_and_markets_unchanged":True,
              "responsive_overflow_widths":layout_warnings,
              "historical_performance_unchanged":True, "runtime_release":runtime["runtime_sha256"]}
    atomic_json(args.output / "validation.json", checks, immutable=True)
    esc = lambda value: html.escape(str(value))
    games = []
    for gid, frame in observations.groupby("game_id",sort=True):
        rows = "".join(
            f'<tr><td><a href="/#player/{int(r.player_id)}">{esc(r.player_name)}</a><br><small>{esc(r.team)}</small></td>'
            f'<td>{float(r.actual_minutes):.1f}</td><td>{"Played" if r.played else "Did not play"}</td></tr>'
            for r in frame.sort_values(["team","player_name"]).itertuples())
        teams = " / ".join(sorted(frame.team.unique()))
        games.append(f'<details><summary>{esc(teams)} · {esc(gid)} · {len(frame)} players verified</summary>'
                     '<table><thead><tr><th>Player</th><th>Actual min.</th><th>Game status</th></tr></thead><tbody>'
                     + rows + '</tbody></table></details>')
    artifact_rows = "".join(f'<tr><td>{esc(k)}</td><td class="hash">{esc(approved[k]["bundle_sha256"])}</td></tr>'
                            for k in ("data","model","serving","lineups"))
    text = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Daily partition verification</title><style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:1rem;color:#173c31;background:#faf9f5}'
            'table{width:100%;border-collapse:collapse;table-layout:fixed}td,th{text-align:left;vertical-align:top;padding:.6rem;border-bottom:1px solid #ddd;overflow-wrap:anywhere}'
            'th:first-child{width:48%}summary{padding:1rem 0;cursor:pointer}small{color:#626960}.hash{font:12px monospace;overflow-wrap:anywhere}li{margin:.6rem 0}a{color:#126352}</style></head><body>'
            f'<h1>Verified daily update — {esc(approved["evidence_through_date"])}</h1>'
            '<p><strong>Active in dev only. Production was not changed.</strong></p>'
            '<p>The existing official data and model checkpoint were reused, not retrained. Serving views now include the missing day.</p><ul>'
            f'<li>{validation["evidence_game_count"]} games in the updater; {validation["season_performance_games"]} completed games in Schedule.</li>'
            f'<li>{validation["verified_player_opportunities"]} player-game records verified by API, including {validation["verified_missed_game_logs"]} missed games.</li>'
            f'<li>Browser: all {len(checked)} affected player pages ({browser["played"]} played, {browser["missed"]} did not play) have the game-log row and chart point.</li>'
            f'<li>Responsive checks: 1440, 768, and 390 pixels. Horizontal overflow found at: {esc(layout_warnings or "none")}.</li>'
            f'<li>Wins: {validation["wins_completed_team_games"]} completed team-games, as of {esc(validation["win_projection_as_of"])}.</li>'
            '<li>All historical forecasts, market selections, and performance metrics unchanged.</li>'
            f'<li>Unchanged retry reused all five stage checkpoints. {tests[-1]} focused tests passed.</li></ul>'
            '<h2>Review findings</h2><ul>'+ "".join(f"<li>{esc(finding)}</li>" for finding in args.finding)+'</ul>'+
            ('<p>Responsive overflow remains unresolved; this is not a fully clean UI sign-off.</p>' if layout_warnings else '')+
            '<h2>Every affected game and player</h2>'+"".join(games)+
            '<h2>Verified artifacts</h2><table><tbody>'+artifact_rows+'</tbody></table>'
            f'<p>Work: <code class="hash">{esc(spec["work_id"])}</code></p></body></html>')
    args.output.mkdir(parents=True,exist_ok=True)
    report = args.output / "report.html"
    if report.exists():
        assert report.read_text() == text, "Refusing to overwrite a different validation report"
    else:
        report.write_text(text)
    print(json.dumps(checks,indent=2))

if __name__ == "__main__":
    main()
