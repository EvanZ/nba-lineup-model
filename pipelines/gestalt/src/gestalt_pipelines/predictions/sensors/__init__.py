"""Input-driven game prediction automation."""
from .margins import predict_game_margins_sensor
from .actual_minutes import predict_game_margins_actual_minutes_sensor

__all__ = ["predict_game_margins_sensor", "predict_game_margins_actual_minutes_sensor"]
