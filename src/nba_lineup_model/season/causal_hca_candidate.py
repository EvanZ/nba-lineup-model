"""Build and audit the isolated causal HCA replacement candidate chain."""

from __future__ import annotations

import argparse
import html
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from nba_lineup_model.modeling.causal_hca import (
    CANDIDATE_ID,
    HomeCourtPriorSelection,
    select_home_court_prior,
)
from nba_lineup_model.season.daily_artifact_common import file_sha256
from nba_lineup_model.season.daily_model_replay import (
    CausalHcaReplayConfig,
    _component_path,
    _http_context_provider,
    _read_model_chain,
    replay_daily_model_chain,
)


def build_candidate(
    *,
    start_date: date,
    end_date: date,
    preseason_rankings_path: Path,
    preseason_minutes_path: Path,
    historical_model_path: Path,
    data_root: Path,
    output_root: Path,
    baseline_model_root: Path,
    matchup_url: str,
    code_revision: str,
) -> dict[str, Any]:
    """Select the historical prior, replay the chain, and render its audit."""

    context_path = historical_model_path / "season_context_metadata.parquet"
    possessions_path = historical_model_path / "possession_predictions.parquet"
    selection = select_home_court_prior(
        pd.read_parquet(context_path),
        pd.read_parquet(possessions_path, columns=["season", "residual_offense_margin"]),
        target_season="2026-27",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    prior_payload = {
        **selection.as_dict(),
        "context_metadata_sha256": file_sha256(context_path),
        "possession_predictions_sha256": file_sha256(possessions_path),
        "lookahead_target_season_rows_used": 0,
        "back_to_back_policy": "unchanged_fixed_existing_schedule_control",
    }
    prior_path = output_root / "prior_selection.json"
    prior_path.write_text(json.dumps(prior_payload, indent=2, sort_keys=True) + "\n")
    config = CausalHcaReplayConfig(
        home_court_prior_mean=selection.prior_mean,
        home_court_prior_equivalent_possessions=(selection.prior_equivalent_possessions),
        home_court_prior_source=(f"artifact://causal-hca-prior/{file_sha256(prior_path)}"),
        matchup_url=matchup_url,
    )
    provider = _http_context_provider(matchup_url, court="neutral")

    def strict_context(stints, home_b2b, away_b2b):
        offsets, warnings = provider(stints, home_b2b, away_b2b)
        if warnings:
            raise ValueError("Pinned neutral context inputs incomplete: " + "; ".join(warnings))
        return offsets, warnings

    replay_daily_model_chain(
        start_date=start_date,
        end_date=end_date,
        preseason_rankings_path=preseason_rankings_path,
        preseason_minutes_path=preseason_minutes_path,
        data_root=data_root,
        output_root=output_root,
        config=config,
        code_revision=code_revision,
        context_provider=strict_context,
    )
    result = summarize_candidate(
        output_root=output_root,
        baseline_model_root=baseline_model_root,
        selection=selection,
        start_date=start_date,
        end_date=end_date,
    )
    (output_root / "candidate_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    render_candidate_report(result, output_root / "candidate_report.html")
    return result


def summarize_candidate(
    *,
    output_root: Path,
    baseline_model_root: Path,
    selection: HomeCourtPriorSelection,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    candidate_chain = _read_model_chain(output_root)
    baseline_chain = _read_model_chain(baseline_model_root)
    if not candidate_chain:
        raise ValueError("Causal HCA candidate produced no model chain")
    candidate_dates = [row[0] for row in candidate_chain]
    if candidate_dates[0] != start_date or candidate_dates[-1] != end_date:
        raise ValueError("Candidate chain does not cover the exact requested date range")
    baseline_by_date = {
        game_date: (bundle, manifest)
        for game_date, bundle, manifest in baseline_chain
    }
    hca_rows: list[dict[str, object]] = []
    for game_date, bundle, manifest in candidate_chain:
        hca = json.loads(_component_path(bundle, manifest, "home_court_state").read_text())
        hca_rows.append({"game_date": game_date.isoformat(), **hca})
    candidate_bundle, candidate_manifest = candidate_chain[-1][1:]
    baseline_entry = baseline_by_date.get(end_date)
    if baseline_entry is None:
        raise ValueError("Baseline model chain is missing the requested comparison date")
    baseline_bundle, baseline_manifest = baseline_entry
    candidate_ratings = pd.read_parquet(
        _component_path(candidate_bundle, candidate_manifest, "player_ratings")
    )
    baseline_ratings = pd.read_parquet(
        _component_path(baseline_bundle, baseline_manifest, "player_ratings")
    )
    comparison = candidate_ratings.loc[
        :, ["player_id", "player_name", "team_at_cutoff", "updated_nail"]
    ].merge(
        baseline_ratings.loc[:, ["player_id", "updated_nail"]],
        on="player_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_baseline"),
    )
    comparison["delta"] = comparison["updated_nail_candidate"] - comparison["updated_nail_baseline"]
    comparison["absolute_delta"] = comparison["delta"].abs()
    top_changes = comparison.nlargest(30, "absolute_delta").drop(columns="absolute_delta")
    final_identity = candidate_manifest["identity"]
    baseline_identity = baseline_manifest["identity"]
    return {
        "candidate_id": CANDIDATE_ID,
        "status": "state_chain_complete_forward_evaluation_pending",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "date_count": len(candidate_chain),
        "game_count": int(final_identity["cumulative_game_count"]),
        "candidate_tip_sha256": candidate_manifest["bundle_sha256"],
        "baseline_tip_sha256": baseline_manifest["bundle_sha256"],
        "baseline_game_count": int(baseline_identity["cumulative_game_count"]),
        "hca_prior": selection.as_dict(),
        "hca_trajectory": hca_rows,
        "top_player_changes": _records(top_changes),
        "player_overlap_count": len(comparison),
        "b2b_policy": "unchanged from production",
        "published_projection_mutations": 0,
        "production_pointer_changes": 0,
        "runtime_espn_dependencies": 0,
        "forward_evaluation": {
            "status": "pending_canonical_pregame_input_component",
            "reason": (
                "The official daily data/model chain does not yet contain an immutable "
                "pregame input component. Legacy schedule snapshots are intentionally "
                "not used because the repository source policy prohibits ESPN-backed "
                "identifiers and data."
            ),
        },
    }


def render_candidate_report(summary: Mapping[str, Any], output: Path) -> None:
    prior = summary["hca_prior"]
    trajectory_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['game_date']))}</td>"
        f"<td>{float(row['home_court_estimate']):+.3f}</td>"
        f"<td>{100 * float(row['home_court_evidence_weight']):.1f}%</td>"
        f"<td>{float(row['home_court_standard_error']):.3f}</td>"
        "</tr>"
        for row in summary["hca_trajectory"]
    )
    player_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['player_name']))}</td>"
        f"<td>{html.escape(str(row['team_at_cutoff']))}</td>"
        f"<td>{float(row['updated_nail_baseline']):+.3f}</td>"
        f"<td>{float(row['updated_nail_candidate']):+.3f}</td>"
        f"<td>{float(row['delta']):+.3f}</td>"
        "</tr>"
        for row in summary["top_player_changes"]
    )
    output.write_text(
        "<!doctype html><meta charset='utf-8'><title>Causal HCA candidate v2</title>"
        "<style>body{font:15px system-ui;max-width:1180px;margin:32px auto;padding:0 20px;"
        "color:#17201b}h1,h2{letter-spacing:-.02em}section{border:1px solid #d8ddd8;"
        "padding:20px;margin:18px 0;border-radius:10px}table{border-collapse:collapse;"
        "width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #e5e9e5}"
        ".ok{color:#176b3a;font-weight:700}.warn{color:#985d00;font-weight:700}"
        "code{background:#f2f4f2;padding:2px 5px}</style>"
        f"<h1>{html.escape(str(summary['candidate_id']))}</h1>"
        f"<p class='ok'>Joint state chain complete: {summary['game_count']} games over "
        f"{summary['date_count']} Pacific dates. Production changes: 0.</p>"
        "<section><h2>Historically selected HCA prior</h2>"
        f"<p><strong>{html.escape(str(prior['transition_method']))}</strong> won the "
        f"expanding-window comparison ({prior['forward_validation_rows']} forecasts): "
        f"RMSE {float(prior['forward_rmse']):.3f}, MAE {float(prior['forward_mae']):.3f}.</p>"
        f"<p>2026-27 prior mean: <strong>{float(prior['prior_mean']):+.3f}</strong> · "
        f"prior precision: {float(prior['prior_equivalent_possessions']):,.0f} equivalent "
        f"possessions · source through {html.escape(str(prior['source_through_season']))}. "
        "B2B remains unchanged.</p></section>"
        "<section><h2>Daily HCA trajectory</h2><table><thead><tr><th>Date</th>"
        "<th>HCA</th><th>Current-season evidence</th><th>Posterior SE</th></tr></thead>"
        f"<tbody>{trajectory_rows}</tbody></table></section>"
        "<section><h2>Largest player-rating changes vs fixed-HCA chain</h2>"
        "<table><thead><tr><th>Player</th><th>Team</th><th>Fixed HCA</th>"
        f"<th>Joint HCA</th><th>Delta</th></tr></thead><tbody>{player_rows}</tbody>"
        "</table></section>"
        "<section><h2>Forward evaluation gate</h2><p class='warn'>Pending.</p>"
        f"<p>{html.escape(str(summary['forward_evaluation']['reason']))}</p>"
        "<p>No legacy schedule snapshot was substituted, and no published historical "
        "projection was modified.</p></section>",
        encoding="utf-8",
    )


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records", double_precision=12))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--preseason-rankings-path", type=Path, required=True)
    parser.add_argument("--preseason-minutes-path", type=Path, required=True)
    parser.add_argument("--historical-model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-model-root", type=Path, required=True)
    parser.add_argument("--matchup-url", required=True)
    parser.add_argument("--code-revision", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = build_candidate(**vars(args))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
