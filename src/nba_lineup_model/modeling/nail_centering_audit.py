# ruff: noqa: E501
"""Audit cross-season NAIL display centering without changing the model.

Player ratings are only identified up to a uniform translation: adding the
same constant to every player cancels from a five-on-five matchup.  Published
completed and forward ratings can nevertheless use different player pools for
their display centers.  This module places both vectors on one explicit
returning-player reference and separates that translation from the actual
player-level transformation.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.in_season_nail_anchor_calibration import (
    DEFAULT_COMPLETED_RATINGS_PATH,
    DEFAULT_CURRENT_PRESEASON_PATH,
)

DEFAULT_SOURCE_SEASON = "2025-26"
DEFAULT_TARGET_SEASON = "2026-27"
DEFAULT_OUTPUT_DIR = Path("artifacts/models/nail_centering_audit")
AUDIT_VERSION = "nail-centering-audit-v0.1.0"


@dataclass(frozen=True)
class CenteringAudit:
    """One transition-specific common-reference centering audit."""

    source_season: str
    target_season: str
    player_bridge: pd.DataFrame
    centers: pd.DataFrame
    diagnostics: pd.DataFrame
    invariance: pd.DataFrame


@dataclass(frozen=True)
class CenteringAuditRun:
    """Published local audit artifact."""

    run_id: str
    run_dir: Path


def build_centering_audit(
    completed: pd.DataFrame,
    forward: pd.DataFrame,
    *,
    source_season: str = DEFAULT_SOURCE_SEASON,
    target_season: str = DEFAULT_TARGET_SEASON,
    invariance_samples: int = 2_000,
    random_seed: int = 40,
) -> CenteringAudit:
    """Put completed and forward totals on the same returning-player origin.

    The common cohort is the intersection of the two catalogs with positive
    source-season possessions.  The exact same source-season weights center
    both vectors.  Component values are not independently re-centered; the
    final-total translation is retained as a separate reference term so that
    ``common total = scalar + additive - total center`` remains exact.
    """

    completed_frame = _completed_frame(completed, source_season=source_season)
    forward_frame = _forward_frame(forward, target_season=target_season)
    bridge = completed_frame.merge(
        forward_frame,
        on="player_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_completed", "_forward"),
    )
    bridge = bridge.loc[bridge["reference_weight"].gt(0)].copy()
    if len(bridge) < 10:
        raise ValueError("Centering audit requires at least ten returning players")
    if not np.isfinite(bridge["reference_weight"]).all():
        raise ValueError("Reference weights must be finite")

    center_rows: list[dict[str, object]] = []
    for state in ("completed", "forward"):
        for component in ("total", "scalar", "additive"):
            column = f"{state}_{component}"
            center_rows.append(
                {
                    "state": state,
                    "component": component,
                    "common_reference_center": _weighted_mean(
                        bridge[column], bridge["reference_weight"]
                    ),
                    "player_count": int(len(bridge)),
                    "reference_weight": float(bridge["reference_weight"].sum()),
                }
            )
    centers = pd.DataFrame(center_rows)
    center_by_key = {
        (str(row.state), str(row.component)): float(row.common_reference_center)
        for row in centers.itertuples(index=False)
    }
    completed_center = center_by_key[("completed", "total")]
    forward_center = center_by_key[("forward", "total")]

    bridge["completed_reference_offset"] = -completed_center
    bridge["forward_reference_offset"] = -forward_center
    bridge["completed_common_total"] = bridge["completed_total"] - completed_center
    bridge["forward_common_total"] = bridge["forward_total"] - forward_center
    bridge["published_total_change"] = bridge["forward_total"] - bridge["completed_total"]
    bridge["scalar_change_published_coordinates"] = (
        bridge["forward_scalar"] - bridge["completed_scalar"]
    )
    bridge["additive_change_published_coordinates"] = (
        bridge["forward_additive"] - bridge["completed_additive"]
    )
    bridge["reference_frame_correction"] = completed_center - forward_center
    bridge["common_reference_total_change"] = (
        bridge["forward_common_total"] - bridge["completed_common_total"]
    )
    bridge["completed_reconciliation_error"] = bridge["completed_common_total"] - (
        bridge["completed_scalar"]
        + bridge["completed_additive"]
        + bridge["completed_reference_offset"]
    )
    bridge["forward_reconciliation_error"] = bridge["forward_common_total"] - (
        bridge["forward_scalar"] + bridge["forward_additive"] + bridge["forward_reference_offset"]
    )
    bridge.insert(0, "source_season", source_season)
    bridge.insert(1, "target_season", target_season)
    bridge = bridge.sort_values(
        ["forward_common_total", "player_name_forward", "player_id"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    invariance = pd.concat(
        [
            _lineup_invariance(
                bridge,
                state=state,
                center=center_by_key[(state, "total")],
                samples=invariance_samples,
                random_seed=random_seed,
            )
            for state in ("completed", "forward")
        ],
        ignore_index=True,
    )
    diagnostics = pd.DataFrame(
        [
            {
                "check": "completed_total_reconciliation_max_abs",
                "value": float(bridge["completed_reconciliation_error"].abs().max()),
                "tolerance": 1e-10,
            },
            {
                "check": "forward_total_reconciliation_max_abs",
                "value": float(bridge["forward_reconciliation_error"].abs().max()),
                "tolerance": 1e-10,
            },
            {
                "check": "lineup_translation_invariance_max_abs",
                "value": float(invariance["max_abs_margin_change"].max()),
                "tolerance": 1e-10,
            },
            {
                "check": "common_completed_weighted_mean_abs",
                "value": abs(
                    _weighted_mean(bridge["completed_common_total"], bridge["reference_weight"])
                ),
                "tolerance": 1e-10,
            },
            {
                "check": "common_forward_weighted_mean_abs",
                "value": abs(
                    _weighted_mean(bridge["forward_common_total"], bridge["reference_weight"])
                ),
                "tolerance": 1e-10,
            },
        ]
    )
    diagnostics["passed"] = diagnostics["value"].le(diagnostics["tolerance"])
    return CenteringAudit(
        source_season=source_season,
        target_season=target_season,
        player_bridge=bridge,
        centers=centers,
        diagnostics=diagnostics,
        invariance=invariance,
    )


def run(
    *,
    completed_path: Path | str = DEFAULT_COMPLETED_RATINGS_PATH,
    forward_path: Path | str = DEFAULT_CURRENT_PRESEASON_PATH,
    source_season: str = DEFAULT_SOURCE_SEASON,
    target_season: str = DEFAULT_TARGET_SEASON,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> CenteringAuditRun:
    """Execute and publish the local centering audit."""

    completed = pd.read_parquet(completed_path)
    forward = pd.read_parquet(forward_path)
    audit = build_centering_audit(
        completed,
        forward,
        source_season=source_season,
        target_season=target_season,
    )
    return write_centering_audit(
        audit,
        completed_path=Path(completed_path),
        forward_path=Path(forward_path),
        output_dir=Path(output_dir),
    )


def write_centering_audit(
    audit: CenteringAudit,
    *,
    completed_path: Path,
    forward_path: Path,
    output_dir: Path,
) -> CenteringAuditRun:
    """Atomically write the audit tables, manifest, and HTML report."""

    payload_hash = hashlib.sha256(
        pd.util.hash_pandas_object(audit.player_bridge, index=False).values.tobytes()
    ).hexdigest()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"centering-{stamp}-{payload_hash[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / run_id
    staging = output_dir / f".{run_id}.tmp-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        audit.player_bridge.to_parquet(staging / "player_bridge.parquet", index=False)
        audit.centers.to_parquet(staging / "centers.parquet", index=False)
        audit.diagnostics.to_parquet(staging / "diagnostics.parquet", index=False)
        audit.invariance.to_parquet(staging / "invariance.parquet", index=False)
        manifest = {
            "audit_version": AUDIT_VERSION,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "source_season": audit.source_season,
            "target_season": audit.target_season,
            "completed_path": str(completed_path),
            "forward_path": str(forward_path),
            "common_reference": "returning players weighted by source-season on-court possessions",
            "player_count": int(len(audit.player_bridge)),
            "reference_weight": float(audit.player_bridge["reference_weight"].sum()),
            "all_checks_passed": bool(audit.diagnostics["passed"].all()),
            "payload_sha256": payload_hash,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        _write_html_report(staging / "report.html", audit)
        staging.rename(run_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return CenteringAuditRun(run_id=run_id, run_dir=run_dir)


def _completed_frame(frame: pd.DataFrame, *, source_season: str) -> pd.DataFrame:
    required = {
        "season",
        "player_id",
        "player_name",
        "rapm",
        "additive_profile_adjustment",
        "on_court_possessions",
    }
    _validate_catalog(frame, required=required, label="completed")
    selected = frame.loc[frame["season"].astype(str).eq(source_season)].copy()
    if selected.empty:
        raise ValueError(f"Completed catalog has no rows for {source_season}")
    additive = pd.to_numeric(selected["additive_profile_adjustment"], errors="coerce").fillna(0.0)
    total = pd.to_numeric(selected["rapm"], errors="raise")
    output = pd.DataFrame(
        {
            "player_id": selected["player_id"].astype(int),
            "player_name": selected["player_name"].astype(str),
            "completed_total": total,
            "completed_scalar": total - additive,
            "completed_additive": additive,
            "reference_weight": pd.to_numeric(
                selected["on_court_possessions"], errors="raise"
            ).clip(lower=0.0),
        }
    )
    _validate_finite(output, ("completed_total", "completed_scalar", "completed_additive"))
    return output


def _forward_frame(frame: pd.DataFrame, *, target_season: str) -> pd.DataFrame:
    required = {
        "season",
        "player_id",
        "player_name",
        "rapm",
        "additive_profile_adjustment",
    }
    _validate_catalog(frame, required=required, label="forward")
    selected = frame.loc[frame["season"].astype(str).eq(target_season)].copy()
    if selected.empty:
        raise ValueError(f"Forward catalog has no rows for {target_season}")
    additive = pd.to_numeric(selected["additive_profile_adjustment"], errors="coerce").fillna(0.0)
    total = pd.to_numeric(selected["rapm"], errors="raise")
    output = pd.DataFrame(
        {
            "player_id": selected["player_id"].astype(int),
            "player_name": selected["player_name"].astype(str),
            "forward_total": total,
            "forward_scalar": total - additive,
            "forward_additive": additive,
        }
    )
    _validate_finite(output, ("forward_total", "forward_scalar", "forward_additive"))
    return output


def _validate_catalog(frame: pd.DataFrame, *, required: set[str], label: str) -> None:
    if missing := required - set(frame.columns):
        raise ValueError(f"{label.title()} catalog missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{label.title()} catalog is empty")
    keys = frame.loc[:, ["season", "player_id"]].copy()
    if keys.duplicated().any():
        raise ValueError(f"{label.title()} catalog duplicates season/player rows")


def _validate_finite(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    values = frame.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Rating components must be finite")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value_array = pd.to_numeric(values, errors="raise").to_numpy(float)
    weight_array = pd.to_numeric(weights, errors="raise").to_numpy(float)
    if not np.isfinite(value_array).all() or not np.isfinite(weight_array).all():
        raise ValueError("Weighted mean inputs must be finite")
    if float(weight_array.sum()) <= 0:
        raise ValueError("Weighted mean requires positive total weight")
    return float(np.average(value_array, weights=weight_array))


def _lineup_invariance(
    bridge: pd.DataFrame,
    *,
    state: str,
    center: float,
    samples: int,
    random_seed: int,
) -> pd.DataFrame:
    if samples <= 0:
        raise ValueError("Invariance sample count must be positive")
    ratings = bridge[f"{state}_total"].to_numpy(float)
    if len(ratings) < 10:
        raise ValueError("Lineup invariance requires ten players")
    rng = np.random.default_rng(random_seed)
    differences = np.empty(samples, dtype=float)
    for index in range(samples):
        chosen = rng.choice(len(ratings), size=10, replace=False)
        home = chosen[:5]
        away = chosen[5:]
        before = float(ratings[home].sum() - ratings[away].sum())
        centered = ratings - center
        after = float(centered[home].sum() - centered[away].sum())
        differences[index] = after - before
    return pd.DataFrame(
        [
            {
                "state": state,
                "samples": samples,
                "translation": -center,
                "max_abs_margin_change": float(np.abs(differences).max()),
                "mean_abs_margin_change": float(np.abs(differences).mean()),
            }
        ]
    )


def _write_html_report(path: Path, audit: CenteringAudit) -> None:
    bridge = audit.player_bridge
    centers = audit.centers
    diagnostics = audit.diagnostics
    curry = bridge.loc[bridge["player_id"].eq(201939)]
    if curry.empty:
        curry = bridge.iloc[[0]]
    row = curry.iloc[0]
    published_change = float(row["published_total_change"])
    common_change = float(row["common_reference_total_change"])
    correction = float(row["reference_frame_correction"])
    fraction = (
        abs(correction) / abs(published_change) if not np.isclose(published_change, 0.0) else np.nan
    )
    reference_weight = float(bridge["reference_weight"].sum())
    completed_center = float(
        centers.loc[
            centers["state"].eq("completed") & centers["component"].eq("total"),
            "common_reference_center",
        ].iloc[0]
    )
    forward_center = float(
        centers.loc[
            centers["state"].eq("forward") & centers["component"].eq("total"),
            "common_reference_center",
        ].iloc[0]
    )
    verdict = (
        "Centering explains only a minority of Curry's published decline. "
        "Most of the gap remains after both seasons use the identical reference."
    )
    if abs(common_change) <= 0.5 * abs(published_change):
        verdict = (
            "Centering explains at least half of Curry's published decline; repair the "
            "cross-season display frame before interpreting the player transformation."
        )
    if not bool(diagnostics["passed"].all()):
        verdict = "A mechanical centering check failed; block rollout and repair the pipeline."

    curry_table = pd.DataFrame(
        [
            {
                "State": audit.source_season + " completed",
                "Published total": row["completed_total"],
                "Scalar": row["completed_scalar"],
                "Additive": row["completed_additive"],
                "Common-reference center": completed_center,
                "Common-reference total": row["completed_common_total"],
            },
            {
                "State": audit.target_season + " annual forward",
                "Published total": row["forward_total"],
                "Scalar": row["forward_scalar"],
                "Additive": row["forward_additive"],
                "Common-reference center": forward_center,
                "Common-reference total": row["forward_common_total"],
            },
        ]
    ).to_html(index=False, classes="metrics", float_format=lambda value: f"{value:+.3f}")
    change_table = pd.DataFrame(
        [
            {"Quantity": "Published total change", "Value": published_change},
            {"Quantity": "Reference-frame correction", "Value": correction},
            {"Quantity": "Change after common centering", "Value": common_change},
            {
                "Quantity": "Share of published change attributable to centering",
                "Value": fraction,
            },
        ]
    )
    change_table["Display"] = [
        f"{published_change:+.3f}",
        f"{correction:+.3f}",
        f"{common_change:+.3f}",
        f"{fraction:.1%}",
    ]
    change_html = change_table.loc[:, ["Quantity", "Display"]].to_html(
        index=False, classes="metrics", escape=True
    )
    center_display = centers.copy()
    center_display["State"] = center_display["state"].str.title()
    center_display["Component"] = center_display["component"].str.title()
    center_display = center_display.rename(
        columns={
            "common_reference_center": "Center",
            "player_count": "Players",
            "reference_weight": "Possession weight",
        }
    )
    center_html = center_display.loc[
        :, ["State", "Component", "Center", "Players", "Possession weight"]
    ].to_html(index=False, classes="metrics", float_format=lambda value: f"{value:,.6f}")
    diagnostic_display = diagnostics.copy()
    diagnostic_display["Check"] = diagnostic_display["check"].str.replace("_", " ").str.title()
    diagnostic_display["Status"] = np.where(diagnostic_display["passed"], "PASS", "FAIL")
    diagnostic_html = (
        diagnostic_display.loc[:, ["Check", "value", "tolerance", "Status"]]
        .rename(columns={"value": "Observed", "tolerance": "Tolerance"})
        .to_html(index=False, classes="metrics", float_format=lambda value: f"{value:.3e}")
    )
    top_changes = bridge.loc[
        :,
        [
            "player_name_forward",
            "completed_common_total",
            "forward_common_total",
            "common_reference_total_change",
        ],
    ].copy()
    top_changes = top_changes.reindex(
        top_changes["common_reference_total_change"].abs().sort_values(ascending=False).index
    ).head(20)
    top_changes = top_changes.rename(
        columns={
            "player_name_forward": "Player",
            "completed_common_total": "Completed",
            "forward_common_total": "Forward",
            "common_reference_total_change": "Change",
        }
    ).to_html(index=False, classes="metrics", float_format=lambda value: f"{value:+.3f}")

    path.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NAIL centering reconciliation</title>
<style>
:root{{--ink:#17231f;--muted:#65716c;--paper:#f7f5ee;--card:#fffdf7;--green:#174f3e;--gold:#b86f18;--line:#d9d7cc}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:36px 28px 72px}} h1{{font-size:34px;letter-spacing:-.03em;margin:0 0 8px}} h2{{margin:34px 0 12px;font-size:21px}} p{{line-height:1.55}} .eyebrow{{color:var(--green);font-size:12px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}}
.lede{{color:var(--muted);max-width:850px}} .verdict{{border-left:5px solid var(--green);background:#edf4ef;padding:18px 20px;margin:24px 0;font-size:18px;font-weight:650}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:24px 0}} .card{{background:var(--card);border:1px solid var(--line);padding:16px}} .label{{font-size:11px;color:var(--muted);font-weight:800;text-transform:uppercase;letter-spacing:.08em}} .value{{font:700 25px ui-monospace,SFMono-Regular,Menlo,monospace;margin-top:7px}} .note{{color:var(--muted);font-size:13px}}
.table-wrap{{overflow:auto;background:var(--card);border:1px solid var(--line)}} table.metrics{{width:100%;border-collapse:collapse;font-size:14px}} table.metrics th,table.metrics td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} table.metrics th:first-child,table.metrics td:first-child{{text-align:left}} table.metrics th{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}} code{{background:#eceae2;padding:2px 5px}} .warning{{border-left:4px solid var(--gold);padding:10px 14px;background:#fbf0dd}}
@media(max-width:760px){{.cards{{grid-template-columns:1fr 1fr}} main{{padding:24px 14px 56px}}}}
</style></head><body><main>
<div class="eyebrow">Offline blocking audit · {html.escape(AUDIT_VERSION)}</div>
<h1>Did the ruler move?</h1>
<p class="lede">Completed {html.escape(audit.source_season)} and annual-forward {html.escape(audit.target_season)} NAIL totals are re-expressed against the identical {len(bridge):,}-player returning cohort, weighted by {reference_weight:,.0f} source-season on-court possessions. No model coefficient or game prediction is changed.</p>
<div class="verdict">{html.escape(verdict)}</div>
<div class="cards">
  <div class="card"><div class="label">Published Curry change</div><div class="value">{published_change:+.3f}</div></div>
  <div class="card"><div class="label">Centering correction</div><div class="value">{correction:+.3f}</div></div>
  <div class="card"><div class="label">Common-reference change</div><div class="value">{common_change:+.3f}</div></div>
  <div class="card"><div class="label">Explained by centering</div><div class="value">{fraction:.1%}</div></div>
</div>
<h2>Stephen Curry bridge</h2><div class="table-wrap">{curry_table}</div>
<p class="note">The component columns retain their published allocation. The common-reference translation is applied once to the final total, not independently to scalar and additive pieces.</p>
<div class="table-wrap">{change_html}</div>
<h2>Reference-frame diagnostics</h2><div class="table-wrap">{center_html}</div>
<p>The completed vector is {completed_center:+.3f} and the forward vector is {forward_center:+.3f} when measured on the same returning-player weights. Their {completed_center - forward_center:+.3f} difference is the entire cross-season centering correction for every returning player.</p>
<h2>Mechanical verification</h2><div class="table-wrap">{diagnostic_html}</div>
<p class="note">The lineup test samples five players against five players. A uniform translation must cancel exactly. The small observed residual is floating-point arithmetic, not a model change.</p>
<h2>Largest common-reference changes</h2><div class="table-wrap">{top_changes}</div>
<h2>Interpretation</h2>
<p>A player total has no absolute zero until a reference population is declared. This audit makes that population identical across the transition. It can diagnose a display-frame discontinuity, but it cannot decide whether the remaining scalar and additive changes are optimally predictive. That is the next temporal-calibration question.</p>
<p class="warning"><strong>Decision gate:</strong> centering does not explain most of Curry's gap if the common-reference decline remains close to the published decline. In that case, keep production unchanged and proceed to separate forward validation of the scalar and additive transformations.</p>
</main></body></html>"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-path", type=Path, default=DEFAULT_COMPLETED_RATINGS_PATH)
    parser.add_argument("--forward-path", type=Path, default=DEFAULT_CURRENT_PRESEASON_PATH)
    parser.add_argument("--source-season", default=DEFAULT_SOURCE_SEASON)
    parser.add_argument("--target-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(
        completed_path=args.completed_path,
        forward_path=args.forward_path,
        source_season=args.source_season,
        target_season=args.target_season,
        output_dir=args.output_dir,
    )
    print(result.run_dir)


if __name__ == "__main__":
    main()
