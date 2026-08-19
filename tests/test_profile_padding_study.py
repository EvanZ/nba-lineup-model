from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.profile_padding_study import (
    METRICS,
    contract_from_estimates,
)


def test_contract_from_estimates_maps_primitive_statistics_to_profile_formula() -> None:
    estimates = pd.DataFrame(
        {
            "metric": [spec.name for spec in METRICS],
            "selected_pseudo_sample_size": [float(index + 1) for index, _ in enumerate(METRICS)],
        }
    )

    contract = contract_from_estimates(
        estimates,
        through_target_season="2022-23",
    )

    values = dict(
        zip(
            estimates["metric"],
            estimates["selected_pseudo_sample_size"],
            strict=True,
        )
    )
    assert contract.reference_mode == "season"
    assert contract.rate_pseudo_possessions["three_pa"] == values["three_pa"]
    assert contract.three_point_percentage_attempts == values["three_point_pct"]
    assert (
        contract.usage_component_pseudo_possessions["field_goals_attempted"]
        == values["field_goals_attempted"]
    )
    assert (
        contract.rebound_percentage_pseudo_possessions["offensive_rebound_pct"]
        == values["offensive_rebound_pct"]
    )
