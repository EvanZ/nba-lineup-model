from __future__ import annotations

from catboost import CatBoostRegressor


def make_catboost_regressor(**kwargs: object) -> CatBoostRegressor:
    defaults = {
        "loss_function": "RMSE",
        "verbose": False,
        "random_seed": 7,
    }
    return CatBoostRegressor(**(defaults | kwargs))
