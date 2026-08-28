"""Persist an explicit model-contract audit for production NAIL-RAPM v1.2.1.2.

The audit does not refit or score NAIL.  It records the actual contract used by
the promoted scalar model and makes a sharp distinction between choices with
historical empirical support and choices that remain fixed, inherited, or
unvalidated in the current release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_CONTEXT_ALPHA
from nba_lineup_model.modeling.train import DEFAULT_LAMBDA_GRID


MODEL_NAME = "forward_nail_rapm_v1212_back_to_back"
RELEASE = "NAIL-RAPM v1.2.1.2"
DEFAULT_AUDIT_ROOT = Path("artifacts/audits/nail_model_contract")
SCHEMA_VERSION = 1
RETAINED_NONADDITIVE_FEATURES = ("top_two_assists", "usage_concentration")

STATUS_SELECTED = "historically_selected_for_scalar_release"
STATUS_INHERITED = "inherited_not_jointly_revalidated_current_release"
STATUS_UNVALIDATED = "fixed_choice_not_validated_current_release"
STATUS_STRUCTURAL = "structural_identifiability_or_attribution_limit"
STATUS_BOUNDARY = "data_or_evaluation_boundary"
STATUS_ESTABLISHED = "established_data_contract"


def contract_records() -> list[dict[str, object]]:
    """Return the complete production-NAIL assumption inventory."""

    additive = ", ".join(LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES)
    nonadditive = ", ".join(RETAINED_NONADDITIVE_FEATURES)
    return [
        _record(
            "objective.stint_target",
            "Objective",
            "Possession-weighted stint home net-rating target",
            "RAPM fits target_home_net_rating on reconstructed stints, weighted by possessions.",
            STATUS_ESTABLISHED,
            "Core RAPM data contract; not a release-specific tuned hyperparameter.",
            "Keep reconstruction and possession-conservation tests as a release invariant.",
            False,
            "prior_rapm.py:fit_forward_lagged_rapm_season",
        ),
        _record(
            "objective.regular_training",
            "Objective",
            "Regular-season-only recursive training",
            "Historical regular seasons build the state; target-season playoffs are evaluation only.",
            STATUS_INHERITED,
            "The release inherits the accepted regular-only contract, but it was not jointly reselected with v1.2.1.2 B2B.",
            "Pre-register a regular-only versus regular-plus-playoffs comparison on a fresh confirmation window.",
            False,
            "forward_portable_matchup_contextual_rapm.py:train_forward_portable_matchup_contextual_rapm",
        ),
        _record(
            "objective.oracle_exposure",
            "Evaluation",
            "Frozen scoring uses observed target-season lineups and possessions",
            "The leaderboard is an oracle-exposure value-model evaluation, not a rotation or injury forecast.",
            STATUS_BOUNDARY,
            "Intentional scope boundary.",
            "Do not treat it as a full-season forecast without a separate rotation/minutes model.",
            False,
            "frozen_multiseason_backtest.py",
        ),
        _record(
            "prior.value_conditioned_aging",
            "Player prior",
            "Value-conditioned aging transition",
            "A forward aging model updates the scalar player prior before a new season.",
            STATUS_INHERITED,
            "Accepted earlier in the NAIL lineage, but not jointly revalidated with the v1.2.1.2 profile, context, and B2B release.",
            "Treat as a distinct registered component in the next locked-release validation; do not silently retune it with feature changes.",
            True,
            "gap_returner_prior.py:build_centered_value_conditioned_aging_gap_returner_priors",
        ),
        _record(
            "prior.exposure_gated_cold_start",
            "Player prior",
            "Exposure-gated draft/replacement cold-start prior",
            "Cold players blend a draft profile and forward replacement token using an exposure classifier.",
            STATUS_INHERITED,
            "Prior NAIL evidence exists, but not a release-level joint validation with the current context contract.",
            "Evaluate on a locked rookie/low-exposure cohort alongside the next release candidate.",
            True,
            "forward_exposure_gated_rapm.py:_cold_start_priors",
        ),
        _record(
            "prior.replacement_token",
            "Player prior",
            "Low-exposure replacement token",
            "Forward replacement rating derives from a low-exposure player pool each completed season.",
            STATUS_INHERITED,
            "The candidate definition and cutoff are not jointly revalidated for v1.2.1.2.",
            "Freeze the cutoff/eligibility policy, then test it only within a dedicated cold-start validation.",
            True,
            "forward_exposure_gated_rapm.py:_fit_replacement_token",
        ),
        _record(
            "prior.gap_returner_bridge",
            "Player prior",
            "Last-observed annual aging bridge across gap seasons",
            "A returning player's last observed state advances through each unobserved season with the aging transition and receives no fabricated RAPM update.",
            STATUS_SELECTED,
            "v1.2 directly compared gap handling and promoted this branch, but it has not been jointly revalidated with later B2B addition.",
            "Keep as the registered default; revisit only through a forward gap-returner ablation.",
            False,
            "gap_returner_prior.py:GAP_RETURNER_METHOD",
        ),
        _record(
            "prior.exposure_centering",
            "Player prior",
            "Prior-season exposure centering",
            "The prior vector is centered using prior-season on-court possession exposure.",
            STATUS_ESTABLISHED,
            "An identifiability convention that fixes the arbitrary player-rating origin.",
            "Keep exact centering/reconstruction tests; do not interpret zero as a literal replacement threshold.",
            False,
            "forward_aging_player_prior.py:center_player_priors",
        ),
        _record(
            "player.lambda_source",
            "Player RAPM",
            "Player lambda schedule is imported from a separate forward exposure-gated RAPM run",
            "Each season uses the reference run's selected lambda rather than reselecting lambda after source context/B2B offsets are applied.",
            STATUS_UNVALIDATED,
            "This is a consequential inherited schedule: its optimality for the production residualized target has not been demonstrated.",
            "Primary scalar audit experiment: compare imported lambdas with nested, residualized-target chronological lambda selection on all frozen seasons.",
            True,
            "forward_portable_matchup_contextual_rapm.py:lambda_schedule",
        ),
        _record(
            "player.lambda_grid",
            "Player RAPM",
            "Chronological Ridge lambda grid",
            f"Reference selection grid: {list(DEFAULT_LAMBDA_GRID)} with expanding game-level folds.",
            STATUS_SELECTED,
            "Selected in the forward reference RAPM only; conditional on its target and priors.",
            "If lambda is reselected for residualized targets, retain the same grid initially and report boundary selections.",
            False,
            "train.py:DEFAULT_LAMBDA_GRID; schema.py:ChronologicalSplitConfig",
        ),
        _record(
            "player.uniform_prior_precision",
            "Player RAPM",
            "Uniform player-prior precision",
            "Production v1.2.1.2 does not use posterior-variance or exposure-varying player-state precision.",
            STATUS_SELECTED,
            "A state-precision candidate was tested and not promoted; this keeps the uniform Ridge contract.",
            "Retain as the baseline unless a forward state-precision candidate beats it on a locked window.",
            False,
            "forward_portable_matchup_contextual_rapm.py:use_player_state_precision=False",
        ),
        _record(
            "profiles.lagged_source",
            "Profiles",
            "Strictly lagged player profiles with last-observed returner profiles",
            "Lineup features use prior completed information; cold players receive a draft/replacement profile blend.",
            STATUS_INHERITED,
            "Forward timing is valid, but the exact returner-profile policy was not jointly revalidated in v1.2.1.2.",
            "Test only in a dedicated gap/cold-start study; keep target-season profile outcomes out of frozen scoring.",
            True,
            "contextual_profiles.py:build_contextual_player_profiles",
        ),
        _record(
            "profiles.padding",
            "Profiles",
            "Stat-specific Medvedovsky-style padding",
            "Rate features use the published stat-specific shrinkage contract before lineup aggregation.",
            STATUS_SELECTED,
            "v1.1 compared padding approaches and promoted this contract; later additions did not retune it.",
            "Treat it as fixed for current release; any new padding scheme requires a separate frozen comparison.",
            False,
            "contextual_profiles.py:MEDVEDOVSKY_2020_PROFILE_PADDING",
        ),
        _record(
            "profiles.feature_set",
            "Profiles",
            "Eight additive basketball profile coordinates",
            additive,
            STATUS_INHERITED,
            "The compact feature set predates v1.2.1.2; individual current-release marginal contributions have not been reselected as a bundle.",
            "Use the feature screen for candidates and confirm any changed bundle against a locked scalar baseline.",
            True,
            "contextual_features.py:LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES",
        ),
        _record(
            "context.nonadditive_feature_set",
            "Lineup context",
            "Two retained non-additive unit features",
            nonadditive,
            STATUS_SELECTED,
            "v1.2.1 pruned the earlier bundle and retained these two through frozen comparison and coefficient diagnostics.",
            "Continue to require residual screens, coefficient stability, and a frozen ablation for any feature change.",
            False,
            "contextual_features.py:CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE",
        ),
        _record(
            "context.linear_antisymmetry",
            "Lineup context",
            "Linear Ridge on home-minus-away unit features with explicit antisymmetric augmentation",
            "The same coefficient weights z(home)-z(away); reversed orientation is added during fitting and enforced during scoring.",
            STATUS_SELECTED,
            "The spline/temporal variants were separately tested and not promoted for this release.",
            "Keep linear antisymmetry as the scalar baseline; test new nonlinear families as explicit candidates.",
            False,
            "matchup_contextual.py:fit_linear_ridge_matchup_contextual_model",
        ),
        _record(
            "context.single_alpha",
            "Lineup context",
            "One common context Ridge alpha",
            f"context_alpha={DEFAULT_CONTEXT_ALPHA:,.0f} for all eight additive and two non-additive columns after standardization.",
            STATUS_UNVALIDATED,
            "10,000 was selected in an earlier context-regularization study, but not jointly with the current B2B release or separately by feature block.",
            "Re-evaluate only after player-lambda provenance is resolved; then compare a small pre-registered common-alpha grid.",
            True,
            "forward_contextual_rapm.py:DEFAULT_CONTEXT_ALPHA; matchup_contextual.py:fit_linear_ridge_matchup_contextual_model",
        ),
        _record(
            "context.shared_block_penalty",
            "Lineup context",
            "Additive and non-additive terms share the same context alpha",
            "No separate penalty distinguishes player-compilable profile sums from residual non-additive features.",
            STATUS_UNVALIDATED,
            "This controls how much additive box-score signal can enter the context layer and is central to attribution.",
            "After validating the common-alpha baseline, test one pre-specified two-block penalty split with frozen confirmation.",
            True,
            "matchup_contextual.py:fit_linear_ridge_matchup_contextual_model",
        ),
        _record(
            "context.sequential_residualization",
            "Lineup context",
            "One-pass recursive residualization",
            "Prior-season context is subtracted before player RAPM; current context is then fit to the player-adjusted residual. The two blocks are not jointly iterated to convergence.",
            STATUS_UNVALIDATED,
            "This is the defining NAIL training protocol, but it is not equivalent to a jointly penalized optimum and has no current controlled comparison.",
            "Construct a no-new-feature alternating/joint-fit control with identical priors and penalties before changing attribution claims.",
            True,
            "forward_portable_matchup_contextual_rapm.py:offset and _fit_matchup_contextual_season",
        ),
        _record(
            "context.additive_attribution",
            "Attribution",
            "Additive profile contribution is post-fit reattribution, not a separately fit player prior",
            "The production fit learns all ten context coefficients at lineup level; display-level additive player terms are derived after the fit and do not alter fitted predictions.",
            STATUS_STRUCTURAL,
            "This is a valid decomposition convention when reconstruction holds, but it is not evidence that a player's profile term is a causal individual effect.",
            "Keep reconstruction audits and label the component as attributable model credit, not a separately identified causal estimate.",
            False,
            "forward_portable_matchup_contextual_rapm.py:_player_season_ratings",
        ),
        _record(
            "context.reference_distribution",
            "Attribution",
            "Possession-weighted completed-season reference unit distribution",
            "The reference field identifies portable unit scores h(U) and matchup residual q(U,V); total context prediction is unchanged by that decomposition.",
            STATUS_STRUCTURAL,
            "Reference choice affects explanatory h/q values but not total context C(U,V).",
            "Keep the reference distribution persisted and label any h/q comparison with its source season.",
            False,
            "matchup_contextual.py:_reference_distribution",
        ),
        _record(
            "schedule.back_to_back_definition",
            "Schedule",
            "Calendar-day home-minus-away B2B feature",
            "A team is B2B when the last cataloged competitive game in the same season occurred exactly one calendar day earlier.",
            STATUS_SELECTED,
            "The B2B candidate was directly compared and promoted under the stated no-material-harm gate.",
            "Keep the definition fixed; test additional rest variables as separate candidates rather than silently expanding this feature.",
            False,
            "schedule_controls.py:build_back_to_back_game_features",
        ),
        _record(
            "schedule.alpha",
            "Schedule",
            "B2B Ridge alpha equals context alpha",
            f"schedule_alpha defaults to {DEFAULT_CONTEXT_ALPHA:,.0f}; it was not independently selected for the binary B2B column.",
            STATUS_UNVALIDATED,
            "The B2B feature itself was tested, but its regularization was inherited from context rather than tuned or justified separately.",
            "Run a small, pre-registered B2B-alpha sensitivity after resolving player lambda provenance.",
            True,
            "forward_portable_matchup_contextual_rapm.py:resolved_schedule_alpha",
        ),
        _record(
            "schedule.sequential_fit",
            "Schedule",
            "B2B fits after player RAPM and before context residual fit",
            "The completed B2B coefficient is carried forward and subtracted from the next season's player/context targets.",
            STATUS_UNVALIDATED,
            "The ordering avoids direct target leakage but is another one-pass block allocation choice.",
            "Include schedule in the controlled alternating/joint-fit comparison rather than testing its ordering in isolation.",
            True,
            "forward_portable_matchup_contextual_rapm.py:_fit_back_to_back_schedule_season",
        ),
        _record(
            "schedule.home_court",
            "Schedule",
            "Season-specific home-court intercept recovered from player residuals",
            "Home advantage is a scalar intercept in each source-season score decomposition, not a player rating or a carried B2B-like state.",
            STATUS_ESTABLISHED,
            "Identifiability treatment is explicit, though its annual rather than pooled form has not been separately optimized.",
            "Keep as an intercept; any time-varying HCA study should be evaluated separately from player ratings.",
            False,
            "frozen_prior_evaluation.py:_recover_home_intercept",
        ),
        _record(
            "evaluation.repeated_model_selection",
            "Evaluation",
            "Three frozen seasons repeatedly informed candidate design",
            "The 2023-24 through 2025-26 leaderboard is no longer a pristine final holdout.",
            STATUS_BOUNDARY,
            "This limits strength of promotion claims but does not invalidate correctly implemented replays.",
            "Use a newly completed season or a locked historical confirmation window before a major release claim.",
            True,
            "three-season-frozen-backtest.md methodology",
        ),
        _record(
            "uncertainty.player_intervals",
            "Uncertainty",
            "No player-level uncertainty intervals in production output",
            "Model-comparison bootstrap intervals exist, but player NAIL ratings remain point estimates.",
            STATUS_BOUNDARY,
            "Ranking precision and close-player comparisons are not quantified.",
            "Add clustered-game bootstrap or posterior intervals once the training-block contract is locked.",
            True,
            "nail_v1212_back_to_back_bootstrap artifacts",
        ),
    ]


def build_nail_model_contract_audit(*, audit_root: Path | str = DEFAULT_AUDIT_ROOT) -> Path:
    """Write an immutable production-NAIL contract audit and update latest."""

    frame = pd.DataFrame(contract_records())
    _validate_contract(frame)
    now = datetime.now(UTC)
    run_id = f"nail-model-contract-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = Path(audit_root)
    run_dir = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        frame.to_parquet(temporary / "contract_matrix.parquet", index=False)
        (temporary / "contract_matrix.json").write_text(frame.to_json(orient="records", indent=2))
        metadata = {
            "model": MODEL_NAME,
            "release": RELEASE,
            "schema_version": SCHEMA_VERSION,
            "generated_at": now.isoformat(),
            "record_count": int(len(frame)),
            "promotion_blocker_count": int(frame["promotion_blocker"].sum()),
            "status_counts": {
                str(status): int(count)
                for status, count in frame["status"].value_counts().sort_index().items()
            },
            "contract_matrix_sha256": hashlib.sha256(
                (temporary / "contract_matrix.json").read_bytes()
            ).hexdigest(),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(run_dir)
        (root / "latest.json").write_text(
            json.dumps({"run_id": run_id, "run_dir": str(run_dir)}, indent=2) + "\n"
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return run_dir


def _record(
    choice_id: str,
    layer: str,
    choice: str,
    current_setting: str,
    status: str,
    evidence_or_reason: str,
    required_validation: str,
    promotion_blocker: bool,
    source: str,
) -> dict[str, object]:
    return {
        "choice_id": choice_id,
        "layer": layer,
        "choice": choice,
        "current_setting": current_setting,
        "status": status,
        "evidence_or_reason": evidence_or_reason,
        "required_validation": required_validation,
        "promotion_blocker": promotion_blocker,
        "source": source,
    }


def _validate_contract(frame: pd.DataFrame) -> None:
    required = {
        "choice_id",
        "layer",
        "choice",
        "current_setting",
        "status",
        "evidence_or_reason",
        "required_validation",
        "promotion_blocker",
        "source",
    }
    if set(frame) != required:
        raise ValueError("NAIL contract audit columns changed unexpectedly")
    if frame["choice_id"].duplicated().any():
        raise ValueError("NAIL contract audit contains duplicate choice IDs")
    allowed = {
        STATUS_SELECTED,
        STATUS_INHERITED,
        STATUS_UNVALIDATED,
        STATUS_STRUCTURAL,
        STATUS_BOUNDARY,
        STATUS_ESTABLISHED,
    }
    if not set(frame["status"]).issubset(allowed):
        raise ValueError("NAIL contract audit has an unknown status")
    if frame["promotion_blocker"].dtype != bool:
        raise ValueError("NAIL contract audit blocker flag must be boolean")


def main() -> None:
    parser = argparse.ArgumentParser(description="Persist the production NAIL model-contract audit")
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    args = parser.parse_args()
    run_dir = build_nail_model_contract_audit(audit_root=args.audit_root)
    print(f"NAIL model-contract audit: run={run_dir}")


if __name__ == "__main__":
    main()
