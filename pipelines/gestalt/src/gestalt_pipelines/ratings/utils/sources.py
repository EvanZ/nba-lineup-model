"""Approved frozen inputs. New seasons/versions require explicit selections."""

MODEL_VERSION = "v1"
MODEL_ID = "forward_nail_rapm_v1212_residualized_lambda"
RUN_ID = "forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25"
ROOT = "nba-gestalt/releases/gestalt-web-d739c28/artifacts"
CORE = f"{ROOT}/models/{MODEL_ID}/2025-26/{RUN_ID}"
HCA = (
    "lake/artifacts/site_release/season=2026-27/game_date_pacific=2026-11-02/"
    "runtime_sha256=042869e115b05d0895f14db3f85568dabb72b0788ae97425e7a1f5df21688e3f/"
    "code/deploy/causal-hca-replacement-candidate-v2.json"
)

# Hashes identify the approved bytes, not parsed/reformatted copies.
SOURCES = {
    "catalog": (f"{ROOT}/web/preseason_rankings/{MODEL_ID}/{RUN_ID}/2026-27.parquet", "a849a831e172c89f24c81935a673278236a9f10a482811df7e6457e6dad1db13"),
    "catalog_metadata": (f"{ROOT}/web/preseason_rankings/{MODEL_ID}/{RUN_ID}/2026-27.json", "f79eee0f9591af0b5e13d051ae5660eeae621327bc0f2c1a173ab9478ea22e89"),
    "schedule_metadata": (f"{CORE}/season_schedule_control_metadata.parquet", "09fcd9d94208d99ed720294cb039e9de8c89518fa8ce94bb3beb550edf03ea53"),
    "hca": (HCA, "b45c9c63f932d036c9e4ae0f59f5e83d1195f592788c78033928bc7668a86d75"),
}


HISTORICAL_INITIALIZATION_KIND = "approved_historical_rating"


def source_references(season: str, version: str) -> dict:
    if (season, version) != ("2026-27", MODEL_VERSION):
        raise ValueError("No approved ratings initialization exists for this season/model version.")
    return {name: {"uri": f"s3://nba-gestalt-data/{key}", "sha256": digest}
            for name, (key, digest) in SOURCES.items()}
