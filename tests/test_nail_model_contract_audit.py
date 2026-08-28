"""Regression tests for the production NAIL model-contract inventory."""

import pandas as pd

from nba_lineup_model.modeling.nail_model_contract_audit import (
    STATUS_UNVALIDATED,
    _validate_contract,
    contract_records,
)


def test_production_contract_exposes_residualization_and_lambda_provenance() -> None:
    records = contract_records()
    by_id = {record["choice_id"]: record for record in records}

    _validate_contract(pd.DataFrame(records))
    assert by_id["player.lambda_source"]["status"] == STATUS_UNVALIDATED
    assert by_id["player.lambda_source"]["promotion_blocker"]
    assert by_id["context.sequential_residualization"]["promotion_blocker"]
    assert by_id["schedule.alpha"]["promotion_blocker"]
