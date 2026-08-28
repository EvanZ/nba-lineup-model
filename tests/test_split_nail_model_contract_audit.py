"""Regression tests for the explicit Split NAIL model-contract inventory."""

import pandas as pd

from nba_lineup_model.modeling.split_nail_model_contract_audit import (
    STATUS_STRUCTURAL,
    STATUS_UNVALIDATED,
    _validate_contract,
    contract_records,
)


def test_contract_audit_marks_all_known_od_heuristics_as_blockers() -> None:
    records = contract_records()
    by_id = {record["choice_id"]: record for record in records}

    _validate_contract(pd.DataFrame(records))
    assert by_id["regularization.player_specialization_ratio"]["status"] == STATUS_UNVALIDATED
    assert by_id["regularization.player_specialization_ratio"]["promotion_blocker"]
    assert by_id["regularization.feature_relative_precision"]["promotion_blocker"]
    assert by_id["state.gap_specialization_reset"]["promotion_blocker"]
    assert by_id["schedule.home_court_od_split"]["status"] == STATUS_STRUCTURAL
    assert by_id["schedule.home_court_od_split"]["promotion_blocker"]
