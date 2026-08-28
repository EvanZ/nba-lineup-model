"""Persist an explicit model-contract audit for standalone Split NAIL.

This module deliberately performs no fit and no replay.  Its purpose is to
make every consequential modeling choice inspectable before Split NAIL is ever
treated as a promotable rating model.  In particular, it distinguishes
accepted scalar NAIL choices from new O/D assumptions that have not been
selected or validated in the Split model.
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

from nba_lineup_model.modeling.forward_split_nail import (
    DEFAULT_FEATURE_RELATIVE_PRECISION,
    FIRST_SEASON,
)
from nba_lineup_model.modeling.split_nail import (
    SPLIT_NAIL_ADDITIVE_FEATURES,
    SPLIT_NAIL_NONADDITIVE_FEATURES,
)
from nba_lineup_model.modeling.train import DEFAULT_LAMBDA_GRID


MODEL_NAME = "forward_split_nail_rapm"
DEFAULT_AUDIT_ROOT = Path("artifacts/audits/split_nail_model_contract")
SCHEMA_VERSION = 1

STATUS_SELECTED = "selected_within_split_contract"
STATUS_INHERITED = "inherited_scalar_contract_not_revalidated_for_split"
STATUS_UNVALIDATED = "fixed_split_choice_not_validated"
STATUS_STRUCTURAL = "structural_identifiability_or_attribution_limit"
STATUS_BOUNDARY = "data_or_evaluation_boundary"


def contract_records() -> list[dict[str, object]]:
    """Return the full, reviewable Split NAIL contract inventory.

    ``promotion_blocker`` is intentionally conservative: a model cannot be
    promoted as an O/D rating system while it depends on an unchecked O/D
    allocation choice or reports an unidentifiable O/D parameter as evidence.
    """

    return [
        _record(
            "objective.two_scoring_rows",
            "Objective",
            "One home-offense and one away-offense target row per stint",
            "Two points-per-100 targets weighted by each side's offensive possessions.",
            STATUS_STRUCTURAL,
            "The O/D decomposition needs both scoring directions, but this is a modeling parameterization rather than a separately selected target contract.",
            "Compare against an equivalent net-margin formulation with identical priors and penalties; require frozen prediction parity or improvement.",
            True,
            "split_nail.py:build_split_nail_design_from_side_features",
        ),
        _record(
            "objective.regular_training",
            "Objective",
            "Regular-season stints train the recursive state",
            "Playoffs are held out for frozen evaluation, not included in annual fitting.",
            STATUS_INHERITED,
            "Established production NAIL season-type contract, but not separately revalidated under Split NAIL.",
            "Run a pre-registered regular-only versus regular-plus-playoffs Split comparison on a future untouched evaluation period.",
            False,
            "forward_split_nail.py:train_forward_split_nail",
        ),
        _record(
            "objective.oracle_exposure",
            "Evaluation",
            "Frozen possession and game scoring use realized target-season lineups and possessions",
            "This is an oracle-exposure value-model evaluation, not a rotation forecast.",
            STATUS_BOUNDARY,
            "Intentional evaluation scope; it does not test roster, injury, or minutes projection.",
            "Keep separate from rotation-model evaluation. Do not interpret as a next-season win forecast without an exposure model.",
            False,
            "split_nail_shared_support_replay.py",
        ),
        _record(
            "scalar.value_conditioned_aging",
            "Scalar prior",
            "Value-conditioned aging curve for total player prior",
            "Applied only to total prior P; it has no offense/defense state.",
            STATUS_INHERITED,
            "Previously accepted scalar NAIL component; its interaction with O/D specialization has not been tested.",
            "Ablate or reselect the aging state jointly with O/D specialization on a pre-registered forward validation schedule.",
            True,
            "gap_returner_prior.py:build_centered_value_conditioned_aging_gap_returner_priors",
        ),
        _record(
            "scalar.exposure_gated_cold_start",
            "Scalar prior",
            "Exposure-gated draft/replacement cold-start prior",
            "New-player total P blends a draft profile with a forward replacement token through predicted exposure.",
            STATUS_INHERITED,
            "Previously accepted scalar NAIL component; no evidence yet that its total-only form is appropriate for O/D.",
            "Evaluate O/D cold-start alternatives, including neutral split and any pre-specified side-information split, on rookies only.",
            True,
            "forward_exposure_gated_rapm.py:_cold_start_priors",
        ),
        _record(
            "scalar.replacement_token",
            "Scalar prior",
            "Replacement-level token",
            "One forward token is fit from low-exposure players and enters the scalar cold-start state.",
            STATUS_INHERITED,
            "Previously selected in scalar NAIL; total-only replacement value is not an O/D allocation.",
            "Validate a neutral O/D replacement split versus an O/D replacement state on a pre-registered cold-start holdout.",
            True,
            "forward_exposure_gated_rapm.py:_fit_replacement_token",
        ),
        _record(
            "scalar.gap_returner_bridge",
            "Scalar prior",
            "Last-observed annual aging bridge across absences",
            "An absent player's total state advances one aging transition per missed season; no missed season gets a RAPM update.",
            STATUS_INHERITED,
            "Previously accepted scalar NAIL component; it is not a side-specific state transition.",
            "Evaluate gap-returner total and specialization transitions jointly on players with one-plus missing seasons.",
            True,
            "gap_returner_prior.py:GAP_RETURNER_METHOD",
        ),
        _record(
            "scalar.prior_centering",
            "Scalar prior",
            "Prior exposure centering",
            "Total prior P is centered using prior-season player exposure.",
            STATUS_INHERITED,
            "Established scalar identifiability convention; no Split-specific sensitivity test.",
            "Verify O+D is centered under the same convention and test no change in frozen predictions after algebraically equivalent recentering.",
            False,
            "forward_aging_player_prior.py:center_player_priors",
        ),
        _record(
            "state.immediate_side_carry",
            "O/D state",
            "Immediate-prior-season specialization carry",
            "s(i,t-1) = O(i,t-1) - D(i,t-1) is carried only when the player appeared in the immediately preceding completed Split fit.",
            STATUS_UNVALIDATED,
            "Persistence rho=1 for the immediate state was chosen by construction, not selected in Split NAIL.",
            "Compare a small pre-registered grid of specialization persistence/forgetting values using all frozen seasons.",
            True,
            "forward_split_nail.py:side_differences state update",
        ),
        _record(
            "state.gap_specialization_reset",
            "O/D state",
            "Gap returners reset O/D specialization to zero",
            "The scalar total is bridged, but a player absent from the immediate prior Split fit receives s=0 on return.",
            STATUS_UNVALIDATED,
            "This is an implicit consequence of storing only the immediate season's side map, not a validated gap policy.",
            "Implement and compare last-observed specialization aging/decay versus neutral reset on a gap-returner cohort.",
            True,
            "forward_split_nail.py:split_nail_prior_vector inputs",
        ),
        _record(
            "state.neutral_cold_split",
            "O/D state",
            "Cold starts use O=D=P/2",
            "Every player without prior side state, including rookies, receives zero specialization.",
            STATUS_UNVALIDATED,
            "Reasonable symmetry default, but not an empirically selected O/D cold-start policy.",
            "Test neutral split against narrowly defined, lag-only O/D cold-start priors on a rookie/low-exposure holdout.",
            True,
            "split_nail.py:split_nail_prior_vector",
        ),
        _record(
            "state.no_od_aging",
            "O/D state",
            "No offense/defense-specific aging transition",
            "Aging changes P only; specialization is neither aged nor value-conditioned.",
            STATUS_UNVALIDATED,
            "An omission with direct consequences for players who age differently by side.",
            "First establish whether specialization persistence survives regularization sensitivity; only then test simple O/D aging transitions.",
            True,
            "forward_split_nail.py + gap_returner_prior.py",
        ),
        _record(
            "profile.lagged_profile_source",
            "Player profiles",
            "Strictly lagged player profiles, with last-observed profile for returning players",
            "Target-season lineup features use profiles known at the prior completed season, or a cold-start/replacement profile.",
            STATUS_INHERITED,
            "Forward timing was accepted for scalar NAIL profile work, but not independently revalidated under O/D Split.",
            "Retain as a forward information invariant; test last-observed profile handling only within a pre-registered gap-returner study.",
            True,
            "contextual_profiles.py:build_contextual_player_profiles",
        ),
        _record(
            "profile.padding_contract",
            "Player profiles",
            "Medvedovsky-style shrinkage/padding contract",
            "Rate statistics are padded before player profiles are used in lineup features.",
            STATUS_INHERITED,
            "Published/previously selected scalar profile contract, not jointly optimized with O/D Split.",
            "Use the published contract as the registered default; any alternate padding requires its own frozen comparison.",
            False,
            "contextual_profiles.py:MEDVEDOVSKY_2020_PROFILE_PADDING",
        ),
        _record(
            "profile.feature_set",
            "Player profiles",
            "Eight additive profile features",
            ", ".join(SPLIT_NAIL_ADDITIVE_FEATURES),
            STATUS_INHERITED,
            "Feature list comes from the production NAIL v1.2.1 standard-usage contract; not reselected jointly with O/D Split.",
            "Screen feature additions/removals with the diagnostic harness, then confirm any selected bundle in a frozen O/D comparison.",
            True,
            "contextual_features.py:LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES",
        ),
        _record(
            "profile.side_specific_coefficients",
            "Player profiles",
            "Every additive profile feature gets both O and D coefficients",
            "No basketball-semantic zero constraints are imposed on profile sides.",
            STATUS_UNVALIDATED,
            "This maximizes flexibility but was not selected against side-constrained alternatives.",
            "Compare all-sides-free to a small, pre-specified semantic constraint set; use the same O/D penalty grid.",
            True,
            "split_nail.py:build_split_nail_design_from_side_features",
        ),
        _record(
            "context.nonadditive_feature_set",
            "Lineup context",
            "Two retained non-additive lineup terms",
            ", ".join(SPLIT_NAIL_NONADDITIVE_FEATURES),
            STATUS_INHERITED,
            "They were selected for production scalar NAIL, but their O/D decomposition has not been revalidated.",
            "Run an O/D ablation for each retained term under a pre-registered shared penalty schedule.",
            True,
            "split_nail.py:SPLIT_NAIL_NONADDITIVE_FEATURES",
        ),
        _record(
            "context.side_specific_coefficients",
            "Lineup context",
            "Each non-additive term gets both O and D coefficients",
            "The model does not assume a term belongs to only one side.",
            STATUS_UNVALIDATED,
            "The no-refit audit measures correlation but cannot validate the allocation.",
            "Compare shared, O-only, D-only, and O/D versions only after resolving the common specialization penalty.",
            True,
            "split_nail.py:build_split_nail_design_from_side_features",
        ),
        _record(
            "scaling.season_sd",
            "Scaling",
            "Completed-season possession-weighted standard-deviation scaling",
            "Each profile/context feature is divided by its own completed-season weighted SD; degenerate columns use scale 1.",
            STATUS_UNVALIDATED,
            "A sensible regularization normalization, but no alternative scaling was tested in Split NAIL.",
            "Check raw-unit versus weighted-SD scaling on a limited pre-registered sensitivity run; keep fallback behavior unit-tested.",
            False,
            "split_nail.py:build_split_nail_design_from_side_features",
        ),
        _record(
            "regularization.lambda_grid",
            "Regularization",
            "Annual player-state lambda selected on chronological within-season folds",
            f"Grid: {list(DEFAULT_LAMBDA_GRID)}; three expanding folds, 10% validation each, 15% final within-season test reserved by default.",
            STATUS_SELECTED,
            "Lambda is selected conditionally on the fixed O/D coordinate penalty and fixed feature precisions.",
            "When any paired-precision rule changes, rerun lambda selection inside each season; do not reuse selected lambdas.",
            False,
            "forward_split_nail.py:_select_standalone_lambda; schema.py:ChronologicalSplitConfig",
        ),
        _record(
            "regularization.player_specialization_ratio",
            "Regularization",
            "Player specialization precision is 4x total precision",
            "For m=O+D and s=O-D, precision(m)=1 and precision(s)=4.",
            STATUS_UNVALIDATED,
            "The 4x ratio was chosen heuristically and has never been searched or justified by a prior.",
            "Primary next validation: pre-register a small ratio grid, reselect lambda inside each season, and score all three frozen seasons.",
            True,
            "split_nail.py:_constrained_precision",
        ),
        _record(
            "regularization.feature_relative_precision",
            "Regularization",
            "Profile/context total precision is 1.5x player total precision",
            f"feature_relative_precision={DEFAULT_FEATURE_RELATIVE_PRECISION}; feature specialization precision is therefore 6x player total precision.",
            STATUS_UNVALIDATED,
            "The 1.5 multiplier was fixed, not selected. It applies to both additive and non-additive blocks.",
            "Jointly validate a small feature-precision grid with the player specialization ratio; do not tune it on one target season.",
            True,
            "forward_split_nail.py:DEFAULT_FEATURE_RELATIVE_PRECISION; split_nail.py:_constrained_precision",
        ),
        _record(
            "regularization.shared_feature_precision",
            "Regularization",
            "One common precision for every additive and non-additive feature pair",
            "All ten feature pairs share the same total/specialization precision multiplier after scaling.",
            STATUS_UNVALIDATED,
            "No evidence says the two non-additive terms should be shrunk identically to the eight additive profile terms.",
            "After global ratio validation, test at most one pre-specified two-block precision split: additive profiles versus non-additive context.",
            True,
            "split_nail.py:_constrained_precision",
        ),
        _record(
            "schedule.back_to_back_definition",
            "Schedule controls",
            "Calendar-day back-to-back flag",
            "A team is flagged when its prior cataloged competitive game in the same season was exactly one calendar day earlier.",
            STATUS_INHERITED,
            "B2B improved scalar NAIL and is known before tipoff; O/D treatment is not separately validated.",
            "Keep the calendar definition fixed; validate scalar versus O/D B2B parameterization with the O/D penalty grid.",
            True,
            "schedule_controls.py:build_back_to_back_game_features",
        ),
        _record(
            "schedule.back_to_back_od_penalty",
            "Schedule controls",
            "B2B total precision 1.5 and specialization precision 6",
            "B2B shares the feature-relative multiplier and 4x specialization ratio.",
            STATUS_UNVALIDATED,
            "No separate evidence supports treating a binary schedule control like the player-profile block.",
            "Test scalar B2B first; retain an O/D B2B split only if it clears the pre-registered frozen gate.",
            True,
            "split_nail.py:_constrained_precision",
        ),
        _record(
            "schedule.home_court_od_split",
            "Schedule controls",
            "Separate home-offense and home-defense coefficients",
            "Home court has m precision 1 and s precision 4, but the source data only identifies its net home scoring effect.",
            STATUS_STRUCTURAL,
            "The O/D HCA split is not separately identified; equal-looking O/D coefficients are a penalty artifact, not evidence.",
            "Collapse HCA to one scalar net home-court term before any O/D rating promotion.",
            True,
            "split_nail.py:home_court_column; split_nail.py:_constrained_precision",
        ),
        _record(
            "outputs.reference_centering",
            "Outputs",
            "Display additive profile component uses reference-centered profile exposure",
            "Completed-season lineup exposure defines a reference profile so displayed additive contributions are interpretable and sum with prior plus season update.",
            STATUS_STRUCTURAL,
            "This is an output attribution convention, not a predictive training feature.",
            "Require algebraic reconstruction tests; do not use display centering to claim an independently estimated causal component.",
            False,
            "forward_split_nail.py:materialized player-season ratings",
        ),
        _record(
            "evaluation.shared_support_replay",
            "Evaluation",
            "No-refit frozen replay on shared 2023-24 through 2025-26 support",
            "The audit verifies term reconstruction equals persisted Split predictions to floating-point precision.",
            STATUS_SELECTED,
            "This validates replay provenance, not the correctness of unselected O/D assumptions.",
            "Keep as a release gate for every future Split candidate.",
            False,
            "split_nail_shared_support_replay.py; split_nail_attribution_audit.py",
        ),
        _record(
            "evaluation.repeated_model_selection",
            "Evaluation",
            "The three frozen seasons have informed iterative model design",
            "2023-24 through 2025-26 are not a pristine final holdout after repeated candidate comparisons.",
            STATUS_BOUNDARY,
            "This is an unavoidable current evidence limitation, not a code bug.",
            "Before a public O/D model claim, reserve a new untouched season or an earlier locked replay window for confirmation.",
            True,
            "three-season-frozen-backtest.md methodology",
        ),
        _record(
            "uncertainty.player_od_intervals",
            "Uncertainty",
            "No O/D player-rating uncertainty intervals",
            "Published O and D estimates currently have point values only.",
            STATUS_BOUNDARY,
            "This limits interpretation of close rankings and specialization claims.",
            "Add clustered-game bootstrap or a Bayesian posterior interval after the core O/D contract is selected.",
            True,
            "No current artifact",
        ),
    ]


def build_split_nail_model_contract_audit(
    *, audit_root: Path | str = DEFAULT_AUDIT_ROOT
) -> Path:
    """Write an immutable audit artifact and update ``latest.json``."""

    frame = pd.DataFrame(contract_records())
    _validate_contract(frame)
    now = datetime.now(UTC)
    run_id = f"split-nail-model-contract-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = Path(audit_root)
    run_dir = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        frame.to_parquet(temporary / "contract_matrix.parquet", index=False)
        (temporary / "contract_matrix.json").write_text(
            frame.to_json(orient="records", indent=2)
        )
        summary = {
            "model": MODEL_NAME,
            "schema_version": SCHEMA_VERSION,
            "generated_at": now.isoformat(),
            "record_count": int(len(frame)),
            "promotion_blocker_count": int(frame["promotion_blocker"].sum()),
            "status_counts": {
                str(key): int(value)
                for key, value in frame["status"].value_counts().sort_index().items()
            },
            "contract_matrix_sha256": hashlib.sha256(
                (temporary / "contract_matrix.json").read_bytes()
            ).hexdigest(),
        }
        (temporary / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
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
        raise ValueError("Split NAIL contract audit columns changed unexpectedly")
    if frame["choice_id"].duplicated().any():
        raise ValueError("Split NAIL contract audit contains duplicate choice IDs")
    allowed = {
        STATUS_SELECTED,
        STATUS_INHERITED,
        STATUS_UNVALIDATED,
        STATUS_STRUCTURAL,
        STATUS_BOUNDARY,
    }
    if not set(frame["status"]).issubset(allowed):
        raise ValueError("Split NAIL contract audit has an unknown status")
    if not frame["promotion_blocker"].dtype == bool:
        raise ValueError("Split NAIL contract audit blocker flag must be boolean")


def main() -> None:
    parser = argparse.ArgumentParser(description="Persist the Split NAIL model-contract audit")
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    args = parser.parse_args()
    run_dir = build_split_nail_model_contract_audit(audit_root=args.audit_root)
    print(f"Split NAIL model-contract audit: run={run_dir}")


if __name__ == "__main__":
    main()
