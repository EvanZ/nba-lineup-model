"""Canonical modeling datasets and reproducible training runs."""

from nba_lineup_model.modeling.schema import (
    AgingModelRunManifest,
    AgingSeasonFold,
    BaselineRunManifest,
    BayesianRapmRunManifest,
    ChronologicalFold,
    ChronologicalSplitConfig,
    ModelEvaluationManifest,
    NeuralPossessionManifest,
    NeuralRapmRunManifest,
    RapmDiagnosticsManifest,
    RapmStintManifest,
)
from nba_lineup_model.modeling.stints import (
    build_rapm_stint_dataset,
    modeling_code_fingerprint,
    rapm_stints_frame,
    read_rapm_stints,
    validate_rapm_stint_partition,
)

__all__ = [
    "AgingModelRunManifest",
    "AgingSeasonFold",
    "BaselineRunManifest",
    "BayesianRapmRunManifest",
    "ChronologicalFold",
    "ChronologicalSplitConfig",
    "ModelEvaluationManifest",
    "NeuralPossessionManifest",
    "NeuralRapmRunManifest",
    "RapmStintManifest",
    "RapmDiagnosticsManifest",
    "build_rapm_stint_dataset",
    "modeling_code_fingerprint",
    "rapm_stints_frame",
    "read_rapm_stints",
    "validate_rapm_stint_partition",
]
