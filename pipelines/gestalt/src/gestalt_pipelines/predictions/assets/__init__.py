"""Game prediction assets."""
from .predict import game_predictions

__all__ = ["game_predictions"]

from .actual_minutes import game_predictions_actual_minutes
__all__ += ["game_predictions_actual_minutes"]
