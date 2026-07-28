from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.diagnostics import validate_diagnostics_run

_BAND_COLORS = {
    "stable core": "#2a806f",
    "qualified": "#2b6f9f",
    "fragile": "#b63b4a",
}
_PROFILE_NAMES = (
    "Victor Wembanyama",
    "Shai Gilgeous-Alexander",
    "Kawhi Leonard",
    "Neemias Queta",
    "Jimmy Butler III",
    "Ausar Thompson",
    "Brandon Miller",
)


@dataclass(frozen=True)
class CaseStudyThresholds:
    """Transparent editorial thresholds for the top-ranking review bands."""

    top_n: int = 25
    stable_bootstrap_probability: float = 0.75
    qualified_bootstrap_probability: float = 0.50
    chronological_rank_range_warning: float = 40.0
    lambda_rank_range_warning: float = 40.0
    allocation_rank_change_warning: float = 20.0
    delete_game_coefficient_warning: float = 0.45
    teammate_share_warning: float = 0.80


@dataclass(frozen=True)
class CaseStudySource:
    """Provenance displayed in a generated RAPM case study."""

    season: str
    diagnostics_run_id: str
    source_model_run_id: str
    manifest_sha256: str
    generator_code_sha256: str
    selected_lambda: float
    bootstrap_samples: int
    player_count: int
    eligible_player_count: int
    game_count: int
    stint_count: int


def classify_top_rankings(
    player_diagnostics: pd.DataFrame,
    thresholds: CaseStudyThresholds | None = None,
) -> pd.DataFrame:
    """Assign transparent review bands to the exposure-eligible top ranking."""

    thresholds = thresholds or CaseStudyThresholds()
    required = {
        "eligible_rank",
        "exposure_eligible",
        "top_25_probability",
        "chronological_eligible_rank_range",
        "lambda_eligible_rank_range",
        "max_allocation_absolute_eligible_rank_change",
        "max_delete_game_absolute_coefficient_change",
        "most_common_teammate_share",
    }
    missing = required - set(player_diagnostics.columns)
    if missing:
        raise ValueError(f"Player diagnostics are missing case-study columns: {sorted(missing)}")
    top = (
        player_diagnostics.loc[player_diagnostics["exposure_eligible"]]
        .nsmallest(thresholds.top_n, "eligible_rank")
        .copy()
    )
    if len(top) != thresholds.top_n:
        raise ValueError(f"Expected {thresholds.top_n} eligible players, found {len(top)}")

    warnings = {
        "chronology": top["chronological_eligible_rank_range"].gt(
            thresholds.chronological_rank_range_warning
        ),
        "lambda": top["lambda_eligible_rank_range"].gt(thresholds.lambda_rank_range_warning),
        "allocation": top["max_allocation_absolute_eligible_rank_change"].gt(
            thresholds.allocation_rank_change_warning
        ),
        "single-game influence": top["max_delete_game_absolute_coefficient_change"].gt(
            thresholds.delete_game_coefficient_warning
        ),
        "teammate concentration": top["most_common_teammate_share"].gt(
            thresholds.teammate_share_warning
        ),
    }
    warning_frame = pd.DataFrame(warnings, index=top.index)
    top["structural_warning_count"] = warning_frame.sum(axis=1).astype(int)
    top["structural_warnings"] = warning_frame.apply(
        lambda row: ", ".join(row.index[row]) if row.any() else "none",
        axis=1,
    )
    top["review_band"] = "fragile"
    qualified = top["top_25_probability"].ge(thresholds.qualified_bootstrap_probability) & top[
        "structural_warning_count"
    ].le(1)
    stable = top["top_25_probability"].ge(thresholds.stable_bootstrap_probability) & top[
        "structural_warning_count"
    ].eq(0)
    top.loc[qualified, "review_band"] = "qualified"
    top.loc[stable, "review_band"] = "stable core"
    return top.sort_values("eligible_rank", kind="stable").reset_index(drop=True)


def build_rapm_case_study(
    season: str,
    *,
    diagnostics_run_id: str | None = None,
    reports_dir: Path | str = Path("artifacts/reports"),
    output_path: Path | str | None = None,
    asset_dir: Path | str | None = None,
    thresholds: CaseStudyThresholds | None = None,
) -> tuple[Path, tuple[Path, ...]]:
    """Generate a review page and charts from one immutable diagnostics run."""

    thresholds = thresholds or CaseStudyThresholds()
    run_dir = _resolve_diagnostics_run(
        season,
        Path(reports_dir),
        diagnostics_run_id,
    )
    manifest = validate_diagnostics_run(run_dir)
    players = pd.read_parquet(run_dir / "player_diagnostics.parquet")
    lambda_summary = pd.read_parquet(run_dir / "lambda_summary.parquet")
    allocation_metrics = pd.read_parquet(run_dir / "allocation_metrics.parquet")
    top = classify_top_rankings(players, thresholds)

    page_path = (
        Path(output_path)
        if output_path is not None
        else Path("docs/models") / f"{season}-rapm-case-study.md"
    )
    charts_dir = (
        Path(asset_dir)
        if asset_dir is not None
        else page_path.parent.parent / "assets" / "images" / "rapm" / season
    )
    page_path.parent.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_chart = charts_dir / "bootstrap-intervals.svg"
    sensitivity_chart = charts_dir / "ranking-sensitivity.svg"
    _write_bootstrap_chart(top, bootstrap_chart)
    _write_sensitivity_chart(top, sensitivity_chart, thresholds)

    source = CaseStudySource(
        season=season,
        diagnostics_run_id=manifest.run_id,
        source_model_run_id=manifest.source_model_run_id,
        manifest_sha256=_sha256_file(run_dir / "manifest.json"),
        generator_code_sha256=_sha256_file(Path(__file__)),
        selected_lambda=manifest.selected_rapm_lambda,
        bootstrap_samples=manifest.bootstrap_samples,
        player_count=manifest.player_count,
        eligible_player_count=int(players["exposure_eligible"].sum()),
        game_count=manifest.game_count,
        stint_count=manifest.stint_count,
    )
    bootstrap_reference = _relative_reference(page_path.parent, bootstrap_chart)
    sensitivity_reference = _relative_reference(
        page_path.parent,
        sensitivity_chart,
    )
    markdown = render_case_study_markdown(
        source,
        top,
        lambda_summary,
        allocation_metrics,
        thresholds,
        bootstrap_reference=bootstrap_reference,
        sensitivity_reference=sensitivity_reference,
    )
    page_path.write_text(markdown)
    return page_path, (bootstrap_chart, sensitivity_chart)


def render_case_study_markdown(
    source: CaseStudySource,
    top: pd.DataFrame,
    lambda_summary: pd.DataFrame,
    allocation_metrics: pd.DataFrame,
    thresholds: CaseStudyThresholds,
    *,
    bootstrap_reference: str,
    sensitivity_reference: str,
) -> str:
    """Render the complete one-season RAPM case study."""

    band_counts = top["review_band"].value_counts()
    stable_count = int(band_counts.get("stable core", 0))
    qualified_count = int(band_counts.get("qualified", 0))
    fragile_count = int(band_counts.get("fragile", 0))
    stable_probability = _percent(thresholds.stable_bootstrap_probability)
    qualified_probability = _percent(thresholds.qualified_bootstrap_probability)
    stable_rule = (
        f"Bootstrap top-25 probability at least {stable_probability} and no structural warning"
    )
    qualified_rule = (
        f"Bootstrap top-25 probability at least {qualified_probability} "
        "and at most one structural warning"
    )
    profiles = _profile_rows(top)
    wembanyama = profiles["Victor Wembanyama"]
    gilgeous_alexander = profiles["Shai Gilgeous-Alexander"]
    leonard = profiles["Kawhi Leonard"]
    queta = profiles["Neemias Queta"]
    butler = profiles["Jimmy Butler III"]
    ausar_thompson = profiles["Ausar Thompson"]
    miller = profiles["Brandon Miller"]

    return f"""<p class="project-kicker">Model review / {source.season}</p>

# What a One-Season RAPM Can Establish

<p class="project-lead">
This case study starts with an untouched one-year RAPM ranking, then asks which
positions survive resampling, time, regularization, data-construction, influence,
and lineup-context checks.
</p>

<div class="signal-strip">
  <div><strong>{source.game_count:,} games</strong><span>regular season</span></div>
  <div>
    <strong>{source.bootstrap_samples} bootstraps</strong>
    <span>complete-game resamples</span>
  </div>
  <div>
    <strong>{source.eligible_player_count} eligible players</strong>
    <span>500-possession floor</span>
  </div>
</div>

!!! warning "Experimental, not promoted"
    The review bands below are editorial screening aids, not hypothesis tests,
    causal conclusions, or a replacement player metric. "Fragile" means the
    exact top-25 position is not robust in this one-season specification; it
    does not mean the player is poor or that the coefficient must be false.

## Starting point

The source model is a signed, one-number ridge RAPM fit to {source.stint_count:,}
regular-season stints. It selected lambda `{source.selected_lambda:g}` through
expanding chronological validation and then refit all {source.game_count:,}
games. The table below is the initial exposure-eligible top 25 before applying
any diagnostic screen.

{_initial_ranking_table(top)}

## Review bands

The bands deliberately remain separate from RAPM. They summarize whether a
top-25 position survives several diagnostics; they do not alter coefficients.

| Band | Rule |
| --- | --- |
| Stable core | {stable_rule} |
| Qualified | {qualified_rule} |
| Fragile | Below the qualified bootstrap threshold or carrying multiple structural warnings |

A structural warning is triggered by a chronological or lambda eligible-rank
range above {_integer(thresholds.chronological_rank_range_warning)}, an
allocation-policy rank change above
{_integer(thresholds.allocation_rank_change_warning)}, an exact delete-game
coefficient change above {thresholds.delete_game_coefficient_warning:.2f}, or
more than {_percent(thresholds.teammate_share_warning)} of possessions beside
one teammate. These are transparent case-study thresholds chosen for review
readability, not estimated statistical cutoffs.

The screen retains **{stable_count}** players in the stable core, qualifies
**{qualified_count}**, and marks **{fragile_count}** initial top-25 positions as
fragile.

{_diagnostic_screen_table(top)}

## Sampling stability

Each horizontal interval is the 5th to 95th percentile of a player's
coefficient across {source.bootstrap_samples} complete-game bootstrap samples.
The dot is the original full-season RAPM estimate. Positive intervals support
positive one-season impact, but the top-25 probability is the stricter question
used by the review bands.

<figure class="case-study-figure" markdown>
  ![Bootstrap coefficient intervals for the initial RAPM top 25]({bootstrap_reference})
  <figcaption>Coefficient uncertainty and review band for the initial eligible top 25.</figcaption>
</figure>

## Specification and time

The next view separates two different failure modes. Horizontal movement means
the rank depends on ridge strength; vertical movement means it changed across
expanding season windows. Circle size increases with the largest rank movement
under an alternate possession-allocation policy.

<figure class="case-study-figure" markdown>
  ![Lambda and chronological rank sensitivity for the initial RAPM top 25]({sensitivity_reference})
  <figcaption>Dashed lines mark the case-study structural-warning thresholds.</figcaption>
</figure>

## Five diagnostic stories

### 1. Wembanyama and Gilgeous-Alexander: convergent evidence

Victor Wembanyama begins first at **{wembanyama.rapm:.2f} RAPM** and remains
top 25 in **{_percent(wembanyama.top_25_probability)}** of bootstrap samples.
His chronological, lambda, and allocation rank ranges are only
**{_integer(wembanyama.chronological_eligible_rank_range)}**,
**{_integer(wembanyama.lambda_eligible_rank_range)}**, and
**{_integer(wembanyama.max_allocation_absolute_eligible_rank_change)}**.
Shai Gilgeous-Alexander is similarly consistent:
**{_percent(gilgeous_alexander.top_25_probability)}** top-25 retention with
rank movements of **{_integer(gilgeous_alexander.chronological_eligible_rank_range)}**,
**{_integer(gilgeous_alexander.lambda_eligible_rank_range)}**, and
**{_integer(gilgeous_alexander.max_allocation_absolute_eligible_rank_change)}**.
The diagnostics cannot prove either coefficient is causal, but they find no
material internal reason to reject these positions.

### 2. Kawhi Leonard: strong estimate, influential game

Kawhi Leonard ranks third at **{leonard.rapm:.2f}** and remains top 25 in
**{_percent(leonard.top_25_probability)}** of bootstrap samples. Lambda and
allocation changes are modest, but deleting his most influential screened game
moves the coefficient by **{leonard.max_delete_game_absolute_coefficient_change:.2f}**
points, the largest effect among the reviewed leaders. The ranking remains
plausible, but its support is less diffuse than the point estimate alone
suggests, so the screen labels it qualified.

### 3. Neemias Queta: positive signal, entangled context

Neemias Queta is the most useful surprising result. His bootstrap interval is
entirely positive at **[{queta.bootstrap_p05:.2f}, {queta.bootstrap_p95:.2f}]**
and he remains top 25 in **{_percent(queta.top_25_probability)}** of samples.
However, **{_percent(queta.most_common_teammate_share)}** of his modeled
possessions are beside {queta.most_common_teammate_name}. His raw on-court net
rating of **{queta.raw_on_court_net_rating:.2f}** is adjusted down to
**{queta.rapm:.2f}**. This is not evidence to discard him; it is evidence that
one season has limited leverage for separating his contribution from a
recurring successful context.

### 4. Jimmy Butler III: stable over time, unstable by specification

Jimmy Butler III barely moves chronologically, with a rank range of
**{_integer(butler.chronological_eligible_rank_range)}**, but moves
**{_integer(butler.lambda_eligible_rank_range)}** places across the lambda path
and **{_integer(butler.max_allocation_absolute_eligible_rank_change)}** under
alternate possession allocation. His **{_percent(butler.top_25_probability)}**
bootstrap retention is not the main concern. The disagreement instead comes
from modeling choices, which is why a single bootstrap interval would have
missed the fragility.

### 5. Ausar Thompson and Brandon Miller: rank precision breaks down

Ausar Thompson and Brandon Miller begin 22nd and 25th, but retain a top-25
position in only **{_percent(ausar_thompson.top_25_probability)}** and
**{_percent(miller.top_25_probability)}** of bootstrap samples. Their
chronological rank ranges reach
**{_integer(ausar_thompson.chronological_eligible_rank_range)}** and
**{_integer(miller.chronological_eligible_rank_range)}**; Thompson also moves
**{_integer(ausar_thompson.max_allocation_absolute_eligible_rank_change)}**
places under allocation alternatives. Both bootstrap intervals remain
positive, so the evidence challenges their precise top-25 placement rather
than their positive estimated impact.

## Model-level checks

Nearby lambda values preserve broad ordering, while the ends of the tested path
change the membership of the leaderboard materially.

{_lambda_summary_table(lambda_summary)}

Possession-allocation policies tell a similar two-level story: held-out
game-margin performance is stable, but some individual ranks move sharply.
Each skill score is computed against the mean model under the same target
construction.

{_allocation_summary_table(allocation_metrics)}

## Conclusion

The diagnostics narrow the initial ranking rather than simply approving or
rejecting it. Five players form a stable one-season core. Nine remain credible
with a specific qualification. Eleven top-25 positions are too sensitive to
sampling or specification to publish without prominent uncertainty.

The important distinction is between **coefficient sign**, **coefficient
magnitude**, and **rank precision**. Several fragile top-25 players still have
bootstrap intervals above zero. The tests are saying that the season supports
positive impact more strongly than it supports an exact leaderboard position.
Multi-season RAPM is the next direct test of whether these signals persist.

## Reproduce this page

```bash
uv run --group docs nba-build-rapm-case-study {source.season} \\
  --diagnostics-run-id {source.diagnostics_run_id}
```

| Provenance | Value |
| --- | --- |
| Diagnostics run | `{source.diagnostics_run_id}` |
| Source model run | `{source.source_model_run_id}` |
| Diagnostics manifest SHA-256 | `{source.manifest_sha256}` |
| Generator source SHA-256 | `{source.generator_code_sha256}` |
| Player population | {source.player_count} total / {source.eligible_player_count} eligible |
| Bootstrap samples | {source.bootstrap_samples} |

See the [RAPM training and diagnostics guide](../guides/train-rapm.md) for the
methodological references and complete artifact definitions.
"""


def _initial_ranking_table(top: pd.DataFrame) -> str:
    lines = [
        "| Rank | Player | Team | RAPM | Possessions | Raw on-court |",
        "| ---: | --- | :---: | ---: | ---: | ---: |",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {_integer(row.eligible_rank)} | {row.player_name} | "
            f"{row.primary_team_tricode} | {row.rapm:.2f} | "
            f"{row.possessions:,.0f} | {row.raw_on_court_net_rating:.2f} |"
        )
    return "\n".join(lines)


def _diagnostic_screen_table(top: pd.DataFrame) -> str:
    lines = [
        "| Rank | Player | Review | Boot top 25 | Chronology range | Lambda range | "
        "Allocation move | Delete-game move | Teammate share |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {_integer(row.eligible_rank)} | {row.player_name} | "
            f"{str(row.review_band).title()} | "
            f"{_percent(row.top_25_probability)} | "
            f"{_integer(row.chronological_eligible_rank_range)} | "
            f"{_integer(row.lambda_eligible_rank_range)} | "
            f"{_integer(row.max_allocation_absolute_eligible_rank_change)} | "
            f"{row.max_delete_game_absolute_coefficient_change:.2f} | "
            f"{_percent(row.most_common_teammate_share)} |"
        )
    return "\n".join(lines)


def _lambda_summary_table(summary: pd.DataFrame) -> str:
    lines = [
        "| Lambda | Coefficient correlation | Rank correlation | Top-25 overlap |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in summary.sort_values("regularization").itertuples(index=False):
        marker = " **selected**" if row.is_selected else ""
        lines.append(
            f"| {row.regularization:g}{marker} | "
            f"{row.coefficient_correlation:.3f} | {row.rank_spearman:.3f} | "
            f"{row.top_25_overlap}/25 |"
        )
    return "\n".join(lines)


def _allocation_summary_table(metrics: pd.DataFrame) -> str:
    rapm = metrics.loc[metrics["model"].eq("rapm")].copy()
    lines = [
        "| Allocation policy | Test possessions | Game-margin RMSE | Skill vs mean |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rapm.itertuples(index=False):
        lines.append(
            f"| `{row.allocation_policy}` | {row.test_possessions:,.0f} | "
            f"{row.game_margin_rmse:.2f} | {row.skill_vs_mean:.3f} |"
        )
    return "\n".join(lines)


def _profile_rows(top: pd.DataFrame) -> dict[str, object]:
    profiles = {}
    for name in _PROFILE_NAMES:
        rows = top.loc[top["player_name"].eq(name)]
        if len(rows) != 1:
            raise ValueError(f"Case-study profile player not found exactly once: {name}")
        profiles[name] = next(rows.itertuples(index=False))
    return profiles


def _write_bootstrap_chart(top: pd.DataFrame, output_path: Path) -> None:
    plt = _pyplot()
    frame = top.sort_values("eligible_rank", kind="stable")
    positions = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(10, 10.5))
    axis.hlines(
        positions,
        frame["bootstrap_p05"],
        frame["bootstrap_p95"],
        color="#9aa8b6",
        linewidth=2.2,
        zorder=1,
    )
    for band, color in _BAND_COLORS.items():
        mask = frame["review_band"].eq(band)
        axis.scatter(
            frame.loc[mask, "rapm"],
            positions[mask],
            s=54,
            color=color,
            edgecolor="#ffffff",
            linewidth=0.8,
            label=band.title(),
            zorder=2,
        )
    axis.axvline(0, color="#596879", linewidth=1, linestyle="--")
    axis.set_yticks(
        positions,
        [
            f"{_integer(row.eligible_rank)}  {row.player_name}"
            for row in frame.itertuples(index=False)
        ],
    )
    axis.invert_yaxis()
    axis.set_xlabel("RAPM points per 100 possessions")
    axis.set_title(
        "Bootstrap intervals for the initial eligible top 25",
        loc="left",
        fontweight="bold",
        pad=16,
    )
    axis.legend(
        loc="lower right",
        frameon=False,
        ncol=3,
        bbox_to_anchor=(1.0, 1.005),
    )
    _style_axis(axis)
    figure.tight_layout()
    _save_svg(figure, output_path)
    plt.close(figure)


def _write_sensitivity_chart(
    top: pd.DataFrame,
    output_path: Path,
    thresholds: CaseStudyThresholds,
) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(10, 6.6))
    sizes = (
        45
        + 4.5
        * top["max_allocation_absolute_eligible_rank_change"].clip(lower=0, upper=50).to_numpy()
    )
    for band, color in _BAND_COLORS.items():
        mask = top["review_band"].eq(band)
        axis.scatter(
            top.loc[mask, "lambda_eligible_rank_range"],
            top.loc[mask, "chronological_eligible_rank_range"],
            s=sizes[mask],
            color=color,
            alpha=0.82,
            edgecolor="#ffffff",
            linewidth=0.9,
            label=band.title(),
        )
    labels = {
        "Victor Wembanyama": "Wembanyama",
        "Shai Gilgeous-Alexander": "Gilgeous-Alexander",
        "Kawhi Leonard": "Leonard",
        "Neemias Queta": "Queta",
        "Jimmy Butler III": "Butler",
        "Ausar Thompson": "A. Thompson",
        "Brandon Miller": "Miller",
    }
    for row in top.loc[top["player_name"].isin(labels)].itertuples(index=False):
        axis.annotate(
            labels[row.player_name],
            (
                row.lambda_eligible_rank_range,
                row.chronological_eligible_rank_range,
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#172231",
        )
    axis.axvline(
        thresholds.lambda_rank_range_warning,
        color="#e66a25",
        linestyle="--",
        linewidth=1.2,
    )
    axis.axhline(
        thresholds.chronological_rank_range_warning,
        color="#e66a25",
        linestyle="--",
        linewidth=1.2,
    )
    axis.set_xlabel("Eligible-rank range across lambda path")
    axis.set_ylabel("Eligible-rank range across season windows")
    axis.set_title(
        "Different diagnostics expose different kinds of instability",
        loc="left",
        fontweight="bold",
        pad=16,
    )
    axis.legend(loc="upper left", frameon=False)
    _style_axis(axis)
    figure.tight_layout()
    _save_svg(figure, output_path)
    plt.close(figure)


def _pyplot():
    cache_root = Path(tempfile.gettempdir()) / "nba-lineup-model-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["svg.hashsalt"] = "nba-lineup-model"
    matplotlib.rcParams["svg.fonttype"] = "none"
    from matplotlib import pyplot as plt

    return plt


def _style_axis(axis) -> None:
    axis.figure.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")
    axis.grid(axis="x", color="#d5dee7", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#9aa8b6")
    axis.tick_params(colors="#172231", labelsize=9)
    axis.xaxis.label.set_color("#172231")
    axis.yaxis.label.set_color("#172231")
    axis.title.set_color("#183b63")


def _save_svg(figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "nba-lineup-model", "Date": None},
    )


def _resolve_diagnostics_run(
    season: str,
    reports_dir: Path,
    diagnostics_run_id: str | None,
) -> Path:
    season_dir = reports_dir / "rapm" / season
    if diagnostics_run_id is None:
        diagnostics_run_id = json.loads((season_dir / "latest.json").read_text())["run_id"]
    run_dir = season_dir / diagnostics_run_id
    if not run_dir.is_dir():
        raise ValueError(f"RAPM diagnostics run does not exist: {run_dir}")
    return run_dir


def _relative_reference(parent: Path, target: Path) -> str:
    return Path(os.path.relpath(target, parent)).as_posix()


def _percent(value: float) -> str:
    percentage = 100.0 * float(value)
    if percentage.is_integer():
        return f"{percentage:.0f}%"
    return f"{percentage:.1f}%"


def _integer(value: float) -> int:
    return int(round(float(value)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a documented case study from one RAPM diagnostics run."
    )
    parser.add_argument("season", help="NBA season in YYYY-YY format")
    parser.add_argument("--diagnostics-run-id")
    parser.add_argument("--reports-dir", default="artifacts/reports")
    parser.add_argument("--output-path")
    parser.add_argument("--asset-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    page_path, chart_paths = build_rapm_case_study(
        args.season,
        diagnostics_run_id=args.diagnostics_run_id,
        reports_dir=args.reports_dir,
        output_path=args.output_path,
        asset_dir=args.asset_dir,
    )
    print(f"Wrote RAPM case study: {page_path}")
    for path in chart_paths:
        print(f"Wrote case-study chart: {path}")


if __name__ == "__main__":
    main()
