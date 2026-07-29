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

from nba_lineup_model.modeling.bayesian import validate_bayesian_rapm_run
from nba_lineup_model.modeling.diagnostics import validate_diagnostics_run

_POSTERIOR_COLOR = "#2b6f9f"
_BOOTSTRAP_COLOR = "#e66a25"
_POINT_COLOR = "#183b63"


@dataclass(frozen=True)
class BayesianCaseStudySource:
    """Provenance displayed in a generated Bayesian RAPM case study."""

    season: str
    bayesian_run_id: str
    source_model_run_id: str
    diagnostics_run_id: str
    bayesian_manifest_sha256: str
    diagnostics_manifest_sha256: str
    generator_code_sha256: str
    selected_lambda: float
    posterior_draws: int
    minimum_ranking_possessions: float
    player_count: int
    eligible_player_count: int
    game_count: int
    stint_count: int


def prepare_case_study_players(
    posterior_rankings: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    *,
    top_n: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join posterior and bootstrap uncertainty and return all eligible plus initial top N."""

    required_posterior = {
        "player_id",
        "player_name",
        "eligible_rank",
        "exposure_eligible",
        "ridge_rapm",
        "posterior_lower",
        "posterior_upper",
        "probability_positive",
        "posterior_top_25_probability",
        "posterior_rank_p05",
        "posterior_rank_median",
        "posterior_rank_p95",
        "possessions",
    }
    required_bootstrap = {
        "player_id",
        "bootstrap_p05",
        "bootstrap_p95",
        "positive_probability",
        "top_25_probability",
        "median_eligible_rank",
    }
    missing_posterior = required_posterior - set(posterior_rankings.columns)
    missing_bootstrap = required_bootstrap - set(bootstrap_summary.columns)
    if missing_posterior:
        raise ValueError(
            f"Posterior rankings missing case-study columns: {sorted(missing_posterior)}"
        )
    if missing_bootstrap:
        raise ValueError(
            f"Bootstrap summary missing case-study columns: {sorted(missing_bootstrap)}"
        )
    bootstrap = bootstrap_summary.loc[:, sorted(required_bootstrap)].rename(
        columns={
            "positive_probability": "bootstrap_positive_probability",
            "top_25_probability": "bootstrap_top_25_probability",
            "median_eligible_rank": "bootstrap_median_eligible_rank",
        }
    )
    eligible = posterior_rankings.loc[posterior_rankings["exposure_eligible"]].merge(
        bootstrap,
        on="player_id",
        how="inner",
        validate="one_to_one",
    )
    eligible["posterior_interval_width_90"] = (
        eligible["posterior_upper"] - eligible["posterior_lower"]
    )
    eligible["bootstrap_interval_width_90"] = (
        eligible["bootstrap_p95"] - eligible["bootstrap_p05"]
    )
    eligible["posterior_to_bootstrap_width_ratio"] = (
        eligible["posterior_interval_width_90"]
        / eligible["bootstrap_interval_width_90"]
    )
    eligible["top_25_probability_gap"] = (
        eligible["posterior_top_25_probability"]
        - eligible["bootstrap_top_25_probability"]
    )
    eligible = eligible.sort_values("eligible_rank", kind="stable").reset_index(drop=True)
    top = eligible.head(top_n).copy()
    if len(top) != top_n:
        raise ValueError(f"Expected {top_n} eligible players, found {len(top)}")
    return eligible, top


def build_bayesian_case_study(
    season: str,
    *,
    bayesian_run_id: str | None = None,
    diagnostics_run_id: str | None = None,
    model_artifacts_dir: Path | str = Path("artifacts/models"),
    reports_dir: Path | str = Path("artifacts/reports"),
    output_path: Path | str | None = None,
    asset_dir: Path | str | None = None,
) -> tuple[Path, tuple[Path, ...]]:
    """Generate a Bayesian-versus-ridge case study from immutable model runs."""

    bayesian_dir = _resolve_run(
        Path(model_artifacts_dir) / "bayesian_rapm" / season,
        bayesian_run_id,
        "Bayesian RAPM",
    )
    diagnostics_dir = _resolve_run(
        Path(reports_dir) / "rapm" / season,
        diagnostics_run_id,
        "RAPM diagnostics",
    )
    bayesian_manifest = validate_bayesian_rapm_run(bayesian_dir)
    diagnostics_manifest = validate_diagnostics_run(diagnostics_dir)
    if bayesian_manifest.source_model_run_id != diagnostics_manifest.source_model_run_id:
        raise ValueError("Bayesian and diagnostics runs do not share the same ridge source")
    if bayesian_manifest.dataset_part_sha256 != diagnostics_manifest.dataset_part_sha256:
        raise ValueError("Bayesian and diagnostics runs do not share the same stint data")
    if not np.isclose(bayesian_manifest.credible_interval_probability, 0.90):
        raise ValueError("The comparison case study requires 90% Bayesian intervals")

    posterior_rankings = pd.read_parquet(bayesian_dir / "posterior_rankings.parquet")
    bootstrap_summary = pd.read_parquet(diagnostics_dir / "bootstrap_summary.parquet")
    comparison = pd.read_parquet(bayesian_dir / "comparison_metrics.parquet")
    calibration = pd.read_parquet(bayesian_dir / "predictive_calibration.parquet")
    eligible, top = prepare_case_study_players(
        posterior_rankings,
        bootstrap_summary,
    )

    page_path = (
        Path(output_path)
        if output_path is not None
        else Path("docs/models") / f"{season}-bayesian-rapm-case-study.md"
    )
    charts_dir = (
        Path(asset_dir)
        if asset_dir is not None
        else page_path.parent.parent / "assets" / "images" / "bayesian-rapm" / season
    )
    page_path.parent.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    interval_chart = charts_dir / "posterior-bootstrap-intervals.svg"
    rank_chart = charts_dir / "top-25-probabilities.svg"
    _write_interval_chart(top, interval_chart)
    _write_rank_probability_chart(top, rank_chart)

    source = BayesianCaseStudySource(
        season=season,
        bayesian_run_id=bayesian_manifest.run_id,
        source_model_run_id=bayesian_manifest.source_model_run_id,
        diagnostics_run_id=diagnostics_manifest.run_id,
        bayesian_manifest_sha256=_sha256_file(bayesian_dir / "manifest.json"),
        diagnostics_manifest_sha256=_sha256_file(diagnostics_dir / "manifest.json"),
        generator_code_sha256=_sha256_file(Path(__file__)),
        selected_lambda=bayesian_manifest.selected_rapm_lambda,
        posterior_draws=bayesian_manifest.posterior_draws,
        minimum_ranking_possessions=bayesian_manifest.minimum_ranking_possessions,
        player_count=bayesian_manifest.player_count,
        eligible_player_count=len(eligible),
        game_count=bayesian_manifest.game_count,
        stint_count=bayesian_manifest.stint_count,
    )
    markdown = render_bayesian_case_study_markdown(
        source,
        eligible,
        top,
        comparison.iloc[0],
        calibration,
        interval_reference=_relative_reference(page_path.parent, interval_chart),
        rank_reference=_relative_reference(page_path.parent, rank_chart),
    )
    page_path.write_text(markdown)
    return page_path, (interval_chart, rank_chart)


def render_bayesian_case_study_markdown(
    source: BayesianCaseStudySource,
    eligible: pd.DataFrame,
    top: pd.DataFrame,
    comparison: pd.Series,
    calibration: pd.DataFrame,
    *,
    interval_reference: str,
    rank_reference: str,
) -> str:
    """Render the complete Bayesian-versus-ridge case study."""

    positive_interval_count = int(top["posterior_lower"].gt(0).sum())
    posterior_majority_top_25 = int(top["posterior_top_25_probability"].ge(0.5).sum())
    bootstrap_majority_top_25 = int(top["bootstrap_top_25_probability"].ge(0.5).sum())
    median_width_ratio = float(eligible["posterior_to_bootstrap_width_ratio"].median())
    largest_gap = top.loc[top["top_25_probability_gap"].abs().idxmax()]
    leaders = top.nlargest(2, "posterior_top_25_probability")
    coverage_90 = calibration.loc[calibration["nominal_coverage"].eq(0.90)].iloc[0]
    maximum_prediction_difference = (
        f"{comparison.max_absolute_test_prediction_difference:.2e}"
    )
    possession_floor = f"{source.minimum_ranking_possessions:g}"

    return f"""<p class="project-kicker">Bayesian model review / {source.season}</p>

# What Bayesian RAPM Adds to the Same Ridge Ranking

<p class="project-lead">
The conjugate Bayesian model deliberately reproduces the selected ridge RAPM
point estimates, then replaces a single leaderboard with joint uncertainty over
player effects, ranks, and held-out stint predictions.
</p>

<div class="signal-strip">
  <div><strong>{source.stint_count:,} stints</strong><span>same regular-season mart</span></div>
  <div><strong>{source.posterior_draws:,} draws</strong><span>exact joint posterior</span></div>
  <div>
    <strong>{source.eligible_player_count} eligible</strong>
    <span>{possession_floor}-possession floor</span>
  </div>
</div>

!!! warning "Conditional model uncertainty"
    These credible intervals condition on the signed linear model, selected
    lambda, possession weights, and equal-segment allocation. They do not cover
    uncertainty about lambda selection, possession construction, omitted
    context, or whether the Gaussian likelihood is the right data-generating
    model.

## Same center, new distribution

The model uses the same signed player matrix and normalized possession weights
as ridge:

\\[
y_i \\mid \\alpha, \\beta, \\sigma^2
\\sim
\\mathcal{{N}}\\left(
\\alpha + x_i^\\mathsf{{T}}\\beta,
\\frac{{\\sigma^2}}{{\\widetilde{{w}}_i}}
\\right).
\\]

The player prior is

\\[
\\beta \\mid \\sigma^2
\\sim
\\mathcal{{N}}\\left(
0,
\\frac{{\\sigma^2}}{{n\\lambda}}I
\\right),
\\]

with a flat home-court intercept and
\\(p(\\sigma^2) \\propto 1/\\sigma^2\\). Because this is conjugate Gaussian
regression, the posterior is exact: there is no MCMC convergence or
variational-approximation error.

The posterior location and ridge solution are the same linear solve. The small
differences below are numerical solver tolerance, not modeling disagreement.

| Check | Result |
| --- | ---: |
| Maximum player-coefficient difference | {comparison.max_absolute_coefficient_difference:.2e} |
| Coefficient correlation | {comparison.coefficient_correlation:.9f} |
| Eligible-rank correlation | {comparison.eligible_rank_spearman:.9f} |
| Initial top-25 overlap | {_integer(comparison.top_25_overlap)}/25 |
| Maximum held-out point-prediction difference | {maximum_prediction_difference} |
| Ridge game-margin RMSE | {comparison.ridge_game_margin_rmse:.3f} |
| Bayesian-mean game-margin RMSE | {comparison.posterior_mean_game_margin_rmse:.3f} |

This is therefore not a fourth competing leaderboard. It is the probabilistic
interpretation of the chosen ridge model.

## The initial top 25

Only **{positive_interval_count} of 25** initial leaders have a 90% marginal
credible interval entirely above zero. Only **{posterior_majority_top_25}**
have at least a 50% posterior probability of remaining top 25, compared with
**{bootstrap_majority_top_25}** under the complete-game bootstrap.

{_top_ranking_table(top)}

The rank columns come from joint draws, so they retain posterior covariance
between teammates and opponents. Marginal coefficient intervals alone cannot
answer a rank question.

## Two uncertainty questions

<figure class="case-study-figure" markdown>
  ![Bayesian and game-bootstrap intervals for the initial RAPM top 25]({interval_reference})
  <figcaption>
    Paired 5th-to-95th percentile intervals around the same ridge point estimate.
  </figcaption>
</figure>

Across all {source.eligible_player_count} eligible players, the Bayesian
interval is **{median_width_ratio:.2f} times** as wide as the complete-game
bootstrap interval at the median. This does not make one interval correct and
the other wrong:

- The bootstrap resamples complete games and refits ridge. It measures the
  sampling stability of the regularized estimation procedure.
- The Bayesian posterior conditions on this season's design and asks which
  latent coefficient vectors remain plausible under the likelihood and prior.
- Ridge can be stable under resampling while several correlated player effects
  remain weakly identified. Shrinkage reduces estimator variance without
  making the underlying parameters equally precise.

This distinction is visible in rank probabilities:

<figure class="case-study-figure" markdown>
  ![Bayesian versus bootstrap top-25 probabilities]({rank_reference})
  <figcaption>
    Points below the diagonal receive less top-25 support from the posterior
    than from game resampling.
  </figcaption>
</figure>

{leaders.iloc[0].player_name} and {leaders.iloc[1].player_name} have the
strongest posterior top-25 support at
**{_percent(leaders.iloc[0].posterior_top_25_probability)}** and
**{_percent(leaders.iloc[1].posterior_top_25_probability)}**. The largest
top-25 probability difference is {largest_gap.player_name}:
**{_percent(largest_gap.posterior_top_25_probability)}** posterior versus
**{_percent(largest_gap.bootstrap_top_25_probability)}** bootstrap. The
disagreement concerns rank precision, not the shared point estimate.

## Held-out predictive calibration

The posterior was separately fit on the original 1,044-game final training
window. Coverage below is measured on the untouched final 186 games and 5,789
stints.

{_calibration_table(calibration)}

The 90% predictive interval covers
**{_percent(coverage_90.unweighted_coverage)}** of held-out stints and
**{_percent(coverage_90.possession_weighted_coverage)}** after possession
weighting. That is close to nominal, with modest conservatism under the weighted
view. These are intervals for noisy stint net rating, not game margin or a
player coefficient.

## What this baseline establishes

The Bayesian model adds three things the ridge leaderboard cannot provide:

1. uncertainty about coefficient sign and magnitude;
2. joint rank and top-N probabilities;
3. posterior predictive intervals with measurable held-out coverage.

It also clarifies the next modeling requirement. A hierarchical Bayesian RAPM
should estimate prior scales rather than inheriting one cross-validated lambda,
and it can partially pool by season, age, position, or draft information. That
is where PyMC becomes useful. The exact conjugate model remains the reference
implementation because any richer model should justify its additional
complexity against these closed-form results.

## Reproduce this page

```bash
uv run nba-train-bayesian-rapm {source.season} \\
  --source-run-id {source.source_model_run_id}

uv run --group docs nba-build-bayesian-rapm-case-study {source.season} \\
  --bayesian-run-id {source.bayesian_run_id} \\
  --diagnostics-run-id {source.diagnostics_run_id}
```

| Provenance | Value |
| --- | --- |
| Bayesian run | `{source.bayesian_run_id}` |
| Source ridge run | `{source.source_model_run_id}` |
| Diagnostics run | `{source.diagnostics_run_id}` |
| Bayesian manifest SHA-256 | `{source.bayesian_manifest_sha256}` |
| Diagnostics manifest SHA-256 | `{source.diagnostics_manifest_sha256}` |
| Generator source SHA-256 | `{source.generator_code_sha256}` |
| Selected lambda | `{source.selected_lambda:g}` |
| Player population | {source.player_count} total / {source.eligible_player_count} eligible |

The ridge/Bayesian equivalence follows the standard Gaussian-prior
interpretation summarized by
[van Wieringen (2015)](https://arxiv.org/abs/1509.09169). Posterior and
posterior-predictive interpretation follows
[Gelman et al., *Bayesian Data Analysis*](https://sites.stat.columbia.edu/gelman/book/).
See the [Bayesian RAPM methodology](bayesian-rapm.md) for the full artifact
contract and the [original RAPM case study]({source.season}-rapm-case-study.md)
for the broader stability diagnostics.
"""


def _top_ranking_table(top: pd.DataFrame) -> str:
    lines = [
        "| Rank | Player | RAPM | Bayesian 90% interval | P(positive) | "
        "Bayes top 25 | Bootstrap top 25 | Posterior rank 90% |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {_integer(row.eligible_rank)} | {row.player_name} | "
            f"{row.ridge_rapm:.2f} | [{row.posterior_lower:.2f}, "
            f"{row.posterior_upper:.2f}] | {_percent(row.probability_positive)} | "
            f"{_percent(row.posterior_top_25_probability)} | "
            f"{_percent(row.bootstrap_top_25_probability)} | "
            f"{_integer(row.posterior_rank_p05)}-{_integer(row.posterior_rank_p95)} |"
        )
    return "\n".join(lines)


def _calibration_table(calibration: pd.DataFrame) -> str:
    lines = [
        "| Nominal interval | Stint coverage | Possession-weighted coverage | "
        "Weighted mean width |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in calibration.sort_values("nominal_coverage").itertuples(index=False):
        lines.append(
            f"| {_percent(row.nominal_coverage)} | "
            f"{_percent(row.unweighted_coverage)} | "
            f"{_percent(row.possession_weighted_coverage)} | "
            f"{row.possession_weighted_mean_interval_width:.1f} |"
        )
    return "\n".join(lines)


def _write_interval_chart(top: pd.DataFrame, output_path: Path) -> None:
    plt = _pyplot()
    frame = top.sort_values("eligible_rank", kind="stable")
    positions = np.arange(len(frame), dtype=float)
    figure, axis = plt.subplots(figsize=(10, 11))
    axis.hlines(
        positions - 0.13,
        frame["posterior_lower"],
        frame["posterior_upper"],
        color=_POSTERIOR_COLOR,
        linewidth=2.2,
        label="Bayesian posterior",
    )
    axis.hlines(
        positions + 0.13,
        frame["bootstrap_p05"],
        frame["bootstrap_p95"],
        color=_BOOTSTRAP_COLOR,
        linewidth=2.2,
        label="Complete-game bootstrap",
    )
    axis.scatter(
        frame["ridge_rapm"],
        positions,
        color=_POINT_COLOR,
        edgecolor="#ffffff",
        linewidth=0.7,
        s=35,
        zorder=3,
        label="Ridge estimate",
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
    axis.set_xlabel("Points per 100 possessions")
    axis.set_title(
        "Same point estimate, different uncertainty questions",
        loc="left",
        fontweight="bold",
        pad=52,
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


def _write_rank_probability_chart(top: pd.DataFrame, output_path: Path) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8.2, 7.2))
    axis.scatter(
        top["bootstrap_top_25_probability"],
        top["posterior_top_25_probability"],
        s=55,
        color=_POSTERIOR_COLOR,
        edgecolor="#ffffff",
        linewidth=0.8,
    )
    axis.plot([0, 1], [0, 1], color="#8b98a6", linewidth=1.1, linestyle="--")
    labels = (
        top.assign(
            absolute_gap=top["top_25_probability_gap"].abs(),
        )
        .nlargest(7, "absolute_gap")
        .sort_values("posterior_top_25_probability", ascending=False, kind="stable")
        .copy()
    )
    previous_probability: float | None = None
    close_label_index = 0
    for row in labels.itertuples(index=False):
        short_name = str(row.player_name).split()[-1]
        if (
            previous_probability is None
            or abs(row.posterior_top_25_probability - previous_probability) >= 0.035
        ):
            close_label_index = 0
            y_offset = 6
        else:
            close_label_index += 1
            y_offset = -12 if close_label_index % 2 else 10
        right_edge = row.bootstrap_top_25_probability > 0.90
        axis.annotate(
            short_name,
            (
                row.bootstrap_top_25_probability,
                row.posterior_top_25_probability,
            ),
            xytext=(-7 if right_edge else 7, y_offset),
            textcoords="offset points",
            fontsize=8,
            color="#172231",
            horizontalalignment="right" if right_edge else "left",
        )
        previous_probability = row.posterior_top_25_probability
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Complete-game bootstrap top-25 probability")
    axis.set_ylabel("Bayesian posterior top-25 probability")
    axis.set_title(
        "The posterior is less certain about exact top-25 membership",
        loc="left",
        fontweight="bold",
        pad=16,
    )
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
    import matplotlib.pyplot as plt

    return plt


def _style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9aa8b6")
    axis.spines["bottom"].set_color("#9aa8b6")
    axis.grid(axis="both", color="#dbe2e9", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)
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


def _resolve_run(season_dir: Path, run_id: str | None, label: str) -> Path:
    if run_id is None:
        run_id = json.loads((season_dir / "latest.json").read_text())["run_id"]
    run_dir = season_dir / run_id
    if not run_dir.is_dir():
        raise ValueError(f"{label} run does not exist: {run_dir}")
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
        description="Generate a Bayesian-versus-ridge RAPM case study."
    )
    parser.add_argument("season", help="NBA season in YYYY-YY format")
    parser.add_argument("--bayesian-run-id")
    parser.add_argument("--diagnostics-run-id")
    parser.add_argument("--model-artifacts-dir", default="artifacts/models")
    parser.add_argument("--reports-dir", default="artifacts/reports")
    parser.add_argument("--output-path")
    parser.add_argument("--asset-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    page_path, chart_paths = build_bayesian_case_study(
        args.season,
        bayesian_run_id=args.bayesian_run_id,
        diagnostics_run_id=args.diagnostics_run_id,
        model_artifacts_dir=args.model_artifacts_dir,
        reports_dir=args.reports_dir,
        output_path=args.output_path,
        asset_dir=args.asset_dir,
    )
    print(f"Wrote Bayesian RAPM case study: {page_path}")
    for path in chart_paths:
        print(f"Wrote case-study chart: {path}")


if __name__ == "__main__":
    main()
