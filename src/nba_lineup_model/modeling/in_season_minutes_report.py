"""Self-contained HTML report for the in-season minutes shadow evaluation."""

# ruff: noqa: E501 -- HTML and CSS are intentionally kept as a portable template.

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba_lineup_model.modeling.in_season_minutes import CONDITIONAL_OUTCOMES
from nba_lineup_model.season.slate_time import PACIFIC

CELL_LABELS = {
    "updated_frozen": "Frozen minutes",
    "updated_updated": "Updated minutes",
}


def _number(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _percent(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{100.0 * float(value):.1f}%"


def _player_summary(evaluation: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    available = evaluation.loc[
        evaluation["outcome_classification"].isin(CONDITIONAL_OUTCOMES)
    ].copy()
    if available.empty:
        return pd.DataFrame()
    target = (
        "early_game_role_minutes"
        if "early_game_role_minutes" in available
        else "regulation_equivalent_minutes"
    )
    available["frozen_error"] = (
        available["frozen_conditional_minutes"].astype(float)
        - available[target].astype(float)
    ).abs()
    available["updated_error"] = (
        available["updated_conditional_minutes"].astype(float)
        - available[target].astype(float)
    ).abs()
    players = (
        available.groupby(["player_id", "player_name"], as_index=False)
        .agg(
            team=("team", "last"),
            games=("game_id", "nunique"),
            actual_minutes=(target, "mean"),
            frozen_minutes=("frozen_conditional_minutes", "mean"),
            updated_minutes=("updated_conditional_minutes", "mean"),
            frozen_mae=("frozen_error", "mean"),
            updated_mae=("updated_error", "mean"),
        )
        .assign(improvement=lambda frame: frame["frozen_mae"] - frame["updated_mae"])
    )
    state_columns = [
        "player_id",
        "preseason_conditional_minutes_prior",
        "current_conditional_minutes_estimate",
        "conditional_observations",
        "conditional_window_observations",
    ]
    available = [column for column in state_columns if column in state.columns]
    if "player_id" in available:
        players = players.merge(
            state[available].drop_duplicates("player_id", keep="last"),
            on="player_id",
            how="left",
        )
    return players.sort_values(["improvement", "games"], ascending=[False, False])


def _player_rows(players: pd.DataFrame) -> str:
    rows: list[str] = []
    for row in players.itertuples(index=False):
        improvement = float(row.improvement)
        change_class = "positive" if improvement > 0 else "negative" if improvement < 0 else ""
        current = getattr(row, "current_conditional_minutes_estimate", None)
        prior = getattr(row, "preseason_conditional_minutes_prior", None)
        observations = getattr(
            row,
            "conditional_window_observations",
            getattr(row, "conditional_observations", None),
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.player_name))}</td>"
            f"<td>{html.escape(str(row.team))}</td>"
            f"<td>{int(row.games)}</td>"
            f"<td>{_number(row.actual_minutes, 1)}</td>"
            f"<td>{_number(row.frozen_minutes, 1)}</td>"
            f"<td>{_number(row.updated_minutes, 1)}</td>"
            f"<td>{_number(row.frozen_mae, 2)}</td>"
            f"<td>{_number(row.updated_mae, 2)}</td>"
            f'<td class="{change_class}">{improvement:+.2f}</td>'
            f"<td>{_number(prior, 1)}</td>"
            f"<td>{_number(current, 1)}</td>"
            f"<td>{_number(observations, 0)}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_minutes_shadow_report(
    *,
    summary: dict[str, Any],
    evaluation: pd.DataFrame,
    games: pd.DataFrame,
    state: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write one portable HTML report containing metrics and player detail."""
    del games  # Game-level detail remains available in the linked CSV artifact.
    players = _player_summary(evaluation, state)
    eligible = players.loc[players["games"].ge(2)] if not players.empty else players
    top = eligible.head(12)
    bottom = eligible.tail(12).sort_values("improvement")
    metrics = summary["player_metrics"]
    frozen = metrics["frozen"]
    updated = metrics["updated"]
    conditional_delta = float(frozen["conditional_minutes_mae"]) - float(
        updated["conditional_minutes_mae"]
    )
    unconditional_delta = float(frozen["unconditional_minutes_mae"]) - float(
        updated["unconditional_minutes_mae"]
    )
    generated = datetime.fromisoformat(str(summary["generated_at"]).replace("Z", "+00:00"))
    generated_label = generated.astimezone(PACIFIC).strftime("%b %-d, %Y · %-I:%M %p %Z")
    game_comparison = summary.get(
        "minutes_ab_game_metrics", summary["four_cell_game_metrics"]
    )
    four_cell_rows: list[str] = []
    for key, label in CELL_LABELS.items():
        row = game_comparison[key]
        four_cell_rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{int(row['games'])}</td>"
            f"<td>{_number(row['margin_mae'], 2)}</td>"
            f"<td>{_number(row['margin_rmse'], 2)}</td>"
            f"<td>{_number(row['brier_score'], 5)}</td>"
            f"<td>{_number(row['log_loss'], 5)}</td>"
            f"<td>{_percent(row['winner_accuracy'])}</td>"
            f"<td>{int(row['ats_wins'])}-{int(row['ats_losses'])}-{int(row['ats_pushes'])}</td>"
            "</tr>"
        )
    exclusions = len(summary.get("excluded_games", []))
    coverage = summary.get("early_role_coverage", {})
    reconstructed = int(coverage.get("reconstructed_games", 0))
    fallback = int(coverage.get("fallback_games", int(summary["games"])))
    role_config = summary.get("config", {}).get("early_game_role", {})
    cutoff_minutes = int(role_config.get("cutoff_minutes", 44))
    lookback_games = int(summary.get("config", {}).get("conditional_lookback_games", 3))
    table_header = """
      <thead><tr><th>Player</th><th>Team</th><th>Available games</th><th>Actual avg.</th>
      <th>Frozen</th><th>Updated</th><th>Frozen MAE</th><th>Updated MAE</th>
      <th>Improvement</th><th>Prior role</th><th>Current role</th><th>Obs.</th></tr></thead>
    """
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NBA GESTALT · Minutes Shadow Report</title>
<style>
:root{{--paper:#f5f2e9;--ink:#18231f;--muted:#68736d;--line:#d6d2c5;--green:#175c48;--blue:#4f8798;--red:#a84a36;--amber:#b77a18;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;line-height:1.45}}
main{{max-width:1480px;margin:auto;padding:32px}} h1,h2{{font-family:Inter,system-ui,sans-serif;letter-spacing:-.025em}} h1{{font-size:34px;margin:0 0 6px}} h2{{font-size:20px;margin:34px 0 12px}} p{{margin:6px 0;color:var(--muted)}}
.eyebrow{{color:var(--green);font-weight:800;text-transform:uppercase;letter-spacing:.08em;font-size:12px}} .note{{border-left:4px solid var(--amber);background:#fff5de;padding:12px 16px;margin:18px 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:22px 0}} .card{{background:#fffdf7;border:1px solid var(--line);padding:16px}} .label{{color:var(--muted);font-size:11px;text-transform:uppercase;font-weight:800}} .value{{font:800 27px/1.2 Inter,system-ui,sans-serif;margin-top:6px}} .sub{{font-size:12px;color:var(--muted);margin-top:4px}}
.table-wrap{{overflow:auto;border:1px solid var(--line);background:#fffdf7}} table{{border-collapse:collapse;width:100%;font-size:12px}} th{{position:sticky;top:0;background:#ece8dc;text-align:right;text-transform:uppercase;font-size:10px;letter-spacing:.04em}} th:first-child,td:first-child{{text-align:left}} td,th{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} tbody tr:hover{{background:#f0f5f1}} .positive{{color:var(--green);font-weight:800}} .negative{{color:var(--red);font-weight:800}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .grid>section{{min-width:0}} input{{width:100%;max-width:420px;padding:10px;border:1px solid var(--line);background:#fffdf7;font:inherit;margin:0 0 10px}} footer{{margin:36px 0;color:var(--muted);font-size:11px}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}main{{padding:18px}}}}
</style></head><body><main>
<div class="eyebrow">NBA GESTALT · Shadow evaluation</div>
<h1>Early-game minutes updater</h1>
<p>Season {html.escape(str(summary['season']))} · through {html.escape(str(summary['through_date']))} Pacific · generated {html.escape(generated_label)} · first {cutoff_minutes} minutes · {lookback_games}-game lookback · preseason weight {float(summary['config']['conditional_prior_games']):g} games</p>
<div class="note"><strong>Research only.</strong> Official injury reports control availability. Both game-table rows use the same possession-updated NAIL ratings, so the comparison isolates frozen versus updated minutes. The updated role uses only earlier completed games.</div>
<section class="cards">
  <div class="card"><div class="label">Games</div><div class="value">{int(summary['games'])}</div><div class="sub">{int(summary['player_game_rows']):,} player-game rows · {exclusions} exclusions</div></div>
  <div class="card"><div class="label">Conditional MAE</div><div class="value">{_number(updated['conditional_minutes_mae'], 2)}</div><div class="sub">Frozen {_number(frozen['conditional_minutes_mae'], 2)} · <span class="positive">better by {conditional_delta:.2f}</span></div></div>
  <div class="card"><div class="label">Unconditional MAE</div><div class="value">{_number(updated['unconditional_minutes_mae'], 2)}</div><div class="sub">Frozen {_number(frozen['unconditional_minutes_mae'], 2)} · <span class="positive">better by {unconditional_delta:.2f}</span></div></div>
  <div class="card"><div class="label">Available-player observations</div><div class="value">{int(updated['conditional_minutes_n']):,}</div><div class="sub">Played games plus healthy coach DNPs</div></div>
  <div class="card"><div class="label">{cutoff_minutes}-minute role coverage</div><div class="value">{reconstructed}/{int(summary['games'])}</div><div class="sub">{fallback} realized-minute fallbacks</div></div>
</section>
<h2>Updated-ratings minutes A/B</h2><p>Both rows use the same possession-updated NAIL ratings. Only the minutes allocation changes. Lower is better for MAE, RMSE, Brier, and log loss.</p>
<div class="table-wrap"><table><thead><tr><th>Model inputs</th><th>Games</th><th>Margin MAE</th><th>RMSE</th><th>Brier</th><th>Log loss</th><th>Winner</th><th>ATS</th></tr></thead><tbody>{''.join(four_cell_rows)}</tbody></table></div>
<div class="grid">
 <section><h2>Most improved players</h2><p>At least two available games; improvement is frozen MAE minus updated MAE.</p><div class="table-wrap"><table>{table_header}<tbody>{_player_rows(top)}</tbody></table></div></section>
 <section><h2>Players to monitor</h2><p>Negative improvement means the updater has been less accurate so far.</p><div class="table-wrap"><table>{table_header}<tbody>{_player_rows(bottom)}</tbody></table></div></section>
</div>
<h2>All evaluated players</h2><input id="filter" aria-label="Filter players" placeholder="Filter player or team…">
<div class="table-wrap"><table id="players">{table_header}<tbody>{_player_rows(players)}</tbody></table></div>
<footer>Frozen = immutable pregame minutes projection. Updated = the last {lookback_games} eligible early-game roles, measured through {cutoff_minutes} elapsed minutes and normalized to a 48-minute role, with a {float(summary['config']['conditional_prior_games']):g}-game preseason prior. Each injury-conditioned team allocation is normalized to 240 minutes. Medical and inactive-list absences are excluded; known active coach DNPs remain zero-minute role observations. No realized score or margin is used as a forecast feature.</footer>
</main><script>const q=document.getElementById('filter'),rows=[...document.querySelectorAll('#players tbody tr')];q.addEventListener('input',()=>{{const v=q.value.toLowerCase();rows.forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(v));}});</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document)
    return output_path
