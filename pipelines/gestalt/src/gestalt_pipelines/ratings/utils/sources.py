"""Approved frozen inputs. New seasons/versions require explicit selections."""

MODEL_VERSION = "v1"
MODEL_ID = "forward_nail_rapm_v1212_residualized_lambda"
RUN_ID = "forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25"
ROOT = "nba-gestalt/releases/gestalt-web-d739c28/artifacts"
CORE = f"{ROOT}/models/{MODEL_ID}/2025-26/{RUN_ID}"
AGING = "nba-gestalt/handoffs/2026-11-11/model-rebuild/aging-predictor-2026-27/2026-11-13-aed0818eea96"
HCA = (
    "lake/artifacts/site_release/season=2026-27/game_date_pacific=2026-11-02/"
    "runtime_sha256=042869e115b05d0895f14db3f85568dabb72b0788ae97425e7a1f5df21688e3f/"
    "code/deploy/causal-hca-replacement-candidate-v2.json"
)

# Hashes identify the approved bytes, not parsed/reformatted copies.
SOURCES = {
    "catalog": (f"{ROOT}/web/preseason_rankings/{MODEL_ID}/{RUN_ID}/2026-27.parquet", "a849a831e172c89f24c81935a673278236a9f10a482811df7e6457e6dad1db13"),
    "catalog_metadata": (f"{ROOT}/web/preseason_rankings/{MODEL_ID}/{RUN_ID}/2026-27.json", "f79eee0f9591af0b5e13d051ae5660eeae621327bc0f2c1a173ab9478ea22e89"),
    "profiles": (f"{ROOT}/web/preseason_profiles/{MODEL_ID}/{RUN_ID}/2026-27.parquet", "b6a2d23ea3cbdb03d491b04ed2d16b830e4d35b074a3279bcc5829003fdd8ee9"),
    "context": (f"{CORE}/season_context_models.joblib", "16155585a285da2cc7d7bc061cffac2e3a9e4694e8f8876b307c222ac793e174"),
    "schedule_metadata": (f"{CORE}/season_schedule_control_metadata.parquet", "09fcd9d94208d99ed720294cb039e9de8c89518fa8ce94bb3beb550edf03ea53"),
    "hca": (HCA, "b45c9c63f932d036c9e4ae0f59f5e83d1195f592788c78033928bc7668a86d75"),
    "aging_predictor": (f"{AGING}/aging_predictor.joblib", "aed0818eea96cf456160dc649fffd49e0cf57aedf2beefbeb3492e4dbd0266d0"),
    "aging_metadata": (f"{AGING}/metadata.json", "62d1625ba8f1ff00d79493176fc98b2e852f0c6d6d28fe3884429168c5bdf69a"),
    "aging_evidence": (f"{AGING}/rebuild_evidence.tar.gz", "c1cc7f1963954f8092c199cc4c7cad5b27e05dda7e3c90f9a8d8689eeab6ec33"),
}


def source_references(season: str, version: str) -> dict:
    if (season, version) != ("2026-27", MODEL_VERSION):
        raise ValueError("No approved ratings initialization exists for this season/model version.")
    return {name: {"uri": f"s3://nba-gestalt-data/{key}", "sha256": digest}
            for name, (key, digest) in SOURCES.items()}
