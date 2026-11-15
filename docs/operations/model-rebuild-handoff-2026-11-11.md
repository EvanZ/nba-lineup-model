# Accepted model reconstruction and causal replay handoff

Inspected on EC2 on 2026-11-11. Documentation only: no training, replay, service restart, forecast rewrite, or deployment was performed.

Read this with `model-rebuild-handoff-2026-11-11-evidence.json` in the same directory. That file records exact release identities, bundle paths, component hashes, frozen input paths, and reconciliation location. Paths in this document are on EC2, not the laptop.

## 1. The accepted model updates HCA

The accepted in-season model is `causal_hca_replacement_candidate_v2`, with ratings updater `possession_residual_ridge_joint_hca_v2`, on historical core `forward_nail_rapm_v1212_residualized_lambda`.

HCA is a jointly estimated coefficient, not a fixed serving adjustment. Start from the historically selected prior, subtract only neutral-court frozen context, and jointly solve player residual ratings and HCA. The prior mean is 1.6915705750166747, equivalent possessions 42883.87482466509, and observation variance per 100 is 14358.128916355508. Player residual prior equivalent possessions are 6000. Back-to-back remains the existing fixed schedule control. Do not carry the historical fixed HCA into the context offset as well as estimating HCA: that would double count it.

The accepted prior selection is `/var/lib/nba-gestalt/daily-pipeline-v2/prior/causal_hca_selection.json`. Its file SHA256 is `b45c9c63f932d036c9e4ae0f59f5e83d1195f592788c78033928bc7668a86d75`. The repository file `deploy/causal-hca-replacement-candidate-v2.json` is useful context; the accepted file and its exact bytes are what the recorded prior-source identity references. Preserve the source URI as well as numeric parameters.

| Selected environment | Evidence through | Cumulative games | Players in posterior | Posterior HCA | HCA standard error |
|---|---|---:|---:|---:|---:|
| Production release | 2026-11-01 | 92 | 528 | 1.5707818047038942 | 0.5259803247557175 |
| Dev release | 2026-11-04 | 121 | 529 | 1.516446759666651 | 0.5124202075932959 |

These are artifact reference values, not a claim of current live API health. `gestalt-web.service` and `gestalt-web-dev.service` were inactive at inspection. Do not restart them merely to read this handoff.

## 2. Resolve the actual selected artifacts

Start from `/var/lib/nba-gestalt/live-release/runtime-manifest.json` or `/var/lib/nba-gestalt-dev/live-release/runtime-manifest.json`. For these releases, `approved_manifest.identity` contains the selected `model.bundle_path`, `data.bundle_path`, and provenance. Do not select a model by a directory timestamp or a generic latest pointer.

Production model bundle SHA256:
`2fb91ebfcb125162bb35fad27111eaba4b20c8eb95d0505cacf10a2ec231dba0`

Dev model bundle SHA256:
`6821950c77eb9943f3fc8493f56313847a53d811fedd9a027e49c0080f1708f6`

Both record accepted model revision `1459c83f82b1fa50d901048e65bb39c9fd678204` and configuration SHA256 `558385560bbb2b2d518bceb847468e2f87900a33964191f54a4ecc2bd07534c3`. That model revision differs from frontend/release revision. Both accepted revision and production release revision exist in EC2 Git.

The inspected checkout is `/home/ubuntu/nba-lineup-model`, revision `39bae56a89def256663308994e8d99cd1561b175`, branch `codex/72-box-scores`. `require_model_implementation_compatible(checkout, accepted_revision)` passed at inspection. This is the repository's calculation compatibility gate, not a completed numerical replay. Do not merely stamp an accepted revision onto incompatible code.

## 3. Exact frozen inputs: production versus dev

Production:

- Rankings: `/var/lib/nba-gestalt/live-release/preseason/rankings.parquet`.
- Minutes: `/var/lib/nba-gestalt/daily-pipeline-v2/prior/preseason_minutes.json`.
- Paired prior SHA256: `2d92682a63b485d1e47111a31e42b2e2437d866cb97ba73318822feb057c693e`.
- Both inputs contain 528 players. `_load_frozen_priors` reproduced this hash.
- Exact selected data chain is rooted at `/var/lib/nba-gestalt-dev/daily-data-collection-v2`; take the selected bundle path from the production manifest to confirm identity. Do not substitute the similarly named production collection root without comparing artifacts.
- Exact selected model chain is under `/var/lib/nba-gestalt-dev/candidates/causal_hca_replacement_candidate_v2/revision=1459c83f82b1fa50d901048e65bb39c9fd678204`.

Dev:

- Derive `WORKING` as the ancestor named `working` of the selected dev model bundle; its exact long path is saved in the evidence JSON.
- Original frozen rankings: `WORKING/inputs/rankings.parquet`.
- Original frozen rankings sidecar: `WORKING/inputs/rankings.json`; preserve it because the prior hash includes metadata when present.
- Original frozen minutes: `WORKING/inputs/minutes.json`.
- Paired prior SHA256: `641b187e08756ceab4adde6f170101bdca108094b6e1ef9fc7a2e5c78c5a3229`.
- These original priors contain 528 players and reproduce the recorded dev hash.
- Reconciliation root: `WORKING/inputs/reconciliation`. It contains the introduction of player `1631207` and the extended 529-player inputs. Pass this root through the canonical replay integration; preserve the introduction's date/provenance.
- The 529-player dev serving rankings are a display/catalog result, not the original 528-player frozen prior. Pairing them with original minutes fails the player-universe gate. Pairing them with extended minutes changes the prior identity. Neither is an exact replay of the recorded dev chain.
- Data and model roots are `WORKING/daily-data` and `WORKING/daily-model`.

Rankings require player_id, player_name, team, rapm, and the correct season if a season column is present. The minutes JSON contains the players under its minutes payload, with player_id, availability_probability, conditional_minutes_per_game, and baseline_minutes_per_game. Use `_load_frozen_priors` for the actual schema/universe/hash gate instead of reconstructing the fingerprint manually.

## 4. Code map

All source paths below are relative to `/home/ubuntu/nba-lineup-model/src/nba_lineup_model`.

| Purpose | Code to use |
|---|---|
| Accepted configuration and integrated model step | `orchestration/daily_pipeline.py`: `DailyPipelineConfig`, `causal_hca_replay_config`, `require_model_implementation_compatible`, `CanonicalPipelineSteps.model` |
| Chronological state replay, compatible-prefix reuse, parent restoration | `season/daily_model_replay.py`: `replay_daily_model_chain`, `CausalHcaReplayConfig`, `_build_one_date`, `_load_parent_state`, `_load_frozen_priors` |
| Joint player/HCA design and solve | `modeling/causal_hca.py`: `augment_game_contribution_with_home_court`, `solve_joint_home_court_state` |
| Player possession contribution | `modeling/in_season_nail.py`: contribution/state routines called by replay |
| Minutes updater | `modeling/in_season_minutes.py`: `initialize_minutes_state`, `advance_minutes_state`; updater `early_game_role_v3_direct_posterior` |
| Historical HCA-prior research | `season/causal_hca_candidate.py`: `build_candidate`; `modeling/causal_hca.py`: `select_home_court_prior` |
| Verified daily observed-data construction | `season/daily_data_replay.py` and canonical daily pipeline |
| Historical core training | `modeling/forward_nail_v1212_residualized_lambda.py`: `train_nail_v1212_residualized_lambda` |
| Frozen preseason catalogs/minutes generation | `web_api/preseason_rankings_cache.py`, `web_api/preseason_minutes.py` |

**Do not invoke the generic `python -m nba_lineup_model.season.daily_model_replay` CLI and assume it selects the accepted HCA model.** Its main builds `DailyModelReplayConfig`, whose model version is `official_nba_daily_model_v1` and updater is `possession_residual_ridge_v1`. The accepted route uses the causal subclass through the orchestrator or an explicitly configured Python invocation.

The minutes configuration is participation prior 4 games, conditional prior 0.2 games, conditional lookback 3, clip delta 12 minutes, rotation size 15, early-role cutoff 44 minutes. Use the accepted configuration; changing these is a model change.

## 5. Replay inputs and causal order

Each selected daily data bundle supplies these components:

`final_games.json`, `observed_stints.parquet`, `observed_lineups.parquet`, `actual_minutes.parquet`, `nba_boxscores.json`, `nba_inactive_lists.json`, `injuries.json`, `rosters.json`, and `player_identity.parquet`.

Keep their manifests, hashes, source provenance, schedule coverage, and quality reports. Consume official NBA artifacts; never fabricate stints from box scores or substitute another source. Enumerating date directories is not proof of complete game coverage.

Genesis is 2026-10-20. The selected production chain has 13 date directories through November 1; dev has 16 through November 4. For each chronological date, the replay consumes the verified observed data and the appropriate earlier state, accumulates possession information, updates the joint player/HCA state, advances minutes state, and writes a versioned daily bundle. Preserve pregame state before incorporating the game's observed evidence. Same-game or later observations must not feed a claimed pregame prediction.

A suffix replay requires a compatible parent and matching data/provenance. If no compatible parent exists, the code requires starting at data genesis. A repaired earlier input invalidates downstream state. Scope and approve that suffix explicitly before running it. Preserve historical frozen forecasts; revised state and retrospective outputs do not authorize replacing those forecasts.

For a resumable joint solve, `rating_gram.npz` is an (N+1)x(N+1) matrix: N player residual parameters plus HCA. `rating_vectors.npz` supplies score and exposure. Posterior player residuals are added to frozen preseason ratings. The HCA prior enters the same regularized solve. At the selected tips the Gram shapes are 529x529 (production) and 530x530 (dev). Keep the cross terms; `home_court_state.json` alone cannot resume the joint model.

## 6. Configuration and invocation recipe

Use `deploy/gestalt-daily-production-config.json` as a wiring reference, not as permission to execute against live roots. Its current values identify the accepted model revision, HCA selection file, provenance URL, and frozen inputs. For an approved replay, resolve immutable inputs first and redirect all outputs to an isolated candidate root.

Construct the configuration with `causal_hca_replay_config(pipeline_config)`. It was checked to reproduce the accepted configuration hash. In particular, the stored provenance URL is `http://127.0.0.1:18013/api/matchups`; replacing it in the config changes the hash even when computations are equivalent. The actual runtime context provider can use a separate pinned isolated endpoint.

`CanonicalPipelineSteps.api` launches an isolated API from its bound runtime and environment. `CanonicalPipelineSteps.model` calls its neutral-court provider with `court='neutral'`, rejects context warnings, and calls replay with the selected frozen inputs, accepted provenance, and reconciliation root. This is the integration to reuse. Do not query an unrelated live API or silently accept missing profile/context inputs.

The essential Python call shape is below. The capitalized values must be resolved from the evidence and pinned runtime; this is a handoff recipe, not a command that was executed:

```python
from nba_lineup_model.season.daily_model_replay import replay_daily_model_chain
from nba_lineup_model.orchestration.daily_pipeline import (
    causal_hca_replay_config, require_model_implementation_compatible,
)

require_model_implementation_compatible(CODE_ROOT, ACCEPTED_MODEL_REVISION)
config = causal_hca_replay_config(PIPELINE_CONFIG)
assert config.sha256 == '558385560bbb2b2d518bceb847468e2f87900a33964191f54a4ecc2bd07534c3'

bundles = replay_daily_model_chain(
    start_date=START_DATE,
    end_date=END_DATE,
    preseason_rankings_path=FROZEN_RANKINGS,
    preseason_minutes_path=FROZEN_MINUTES,
    data_root=VERIFIED_DATA_CHAIN_ROOT,
    output_root=ISOLATED_CANDIDATE_MODEL_ROOT,
    config=config,
    code_revision=ACCEPTED_MODEL_REVISION,
    context_provider=STRICT_PINNED_NEUTRAL_CONTEXT_PROVIDER,
    reconciliation_root=RECONCILIATION_ROOT,  # dev introduction history
    approved_historical_replay_from=APPROVED_EARLIEST_DATE,
)
```

No approval is supplied by this document. The exact replay dates, affected downstream state, and forecast immutability must be settled before execution.

Runtime note: checkout `.venv/bin/python` lacked pandas during inspection. Read-only prior/config/schema checks worked with `PYTHONPATH=src /opt/nba-gestalt/releases/gestalt-web-d739c28/.venv/bin/python`. This establishes a usable inspection interpreter, not a fully validated training environment. Use the lockfile and canonical deployment environment for actual jobs; do not assume the checkout venv is complete.

## 7. Expected output contract

The replay produces content-addressed daily-model bundles with manifest lineage, input-data hash, parent-model hash, model provenance, daily/cumulative game IDs, and validation status. Components are under `files/`:

| Output | Meaning |
|---|---|
| `player_ratings.parquet` | frozen_preseason_nail, current_season_residual, updated_nail, exposure/information, prior/evidence weights, posterior uncertainty |
| `home_court_state.json` | posterior HCA, prior source and precision, information, prior/evidence weights, standard error |
| `rating_gram.npz`, `rating_vectors.npz` | complete joint sufficient statistics for continuation |
| `minutes_state.parquet` | participation/conditional-minute posterior, counts, effective weights, conditional history |
| `minutes_observations.parquet` | per-player/game classification, played status, actual/regulation-equivalent/early-role minutes, injury and roster evidence |
| `minute_allocations.parquet` | allocated team/player minutes from current state |
| `rating_trajectory.parquet` | player-game before/after ratings, deltas and observation context |
| `lineup_values.parquet` | lineup identities, possessions and additive values |
| `game_projections.parquet` | descriptive game outputs including projection_kind and eligible_for_forward_evaluation; do not assume these are immutable pregame forecasts |
| `replay_state.json` | team mapping, last team dates, contribution lineage, included games |
| `evaluation_metrics.json` | state/evaluation diagnostics |
| `player_introductions.json` | present in dev: reconciliation introduction lineage |

Selected tip row counts: production ratings/minutes 528, trajectory 3152, game projections 4; dev ratings/minutes 529, trajectory 4164, game projections 14. The 4/14 game projection counts are for those dates, whereas 92/121 are cumulative model evidence counts.

## 8. Rebuilding the historical core from training data

For ordinary in-season continuation, reuse the accepted historical core. Its root is:

`/opt/nba-gestalt/releases/gestalt-web-d739c28/artifacts/models/forward_nail_rapm_v1212_residualized_lambda/2025-26/forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25`

The dev serving bindings also name historical runtime `/opt/nba-gestalt/releases/gestalt-web-d739c28`; `durable-serving.json` identifies S3 objects for historical files. Keep these on EC2/S3.

Key artifacts include `season_context_models.joblib`, `season_context_metadata.parquet`, `season_aging_models.joblib`, `player_season_ratings.parquet`, `historical_player_coefficients.parquet`, `season_model_metadata.parquet`, `season_player_cv_results.parquet`, and `possession_predictions.parquet`. The saved context metadata hash is `d287b289d9139ddaa3e3a85024394843197f01a1c08bacdbcaa30e4d544cd2b7`; possession predictions hash is `cfc98ef4345e14079106baf44574823f68edbe578fd5cfd051bfec6f6105e158`. These are the historical inputs used to select the HCA prior. `causal_hca_candidate.build_candidate` reselects the prior and generates evaluation reports; that research operation is distinct from continuing accepted state with the already accepted selection.

To retrain the historical core, use `train_nail_v1212_residualized_lambda(through_season='2025-26', context_alpha=10000.0, schedule_alpha=10000.0, player_season_panel_path=..., analytical_dir=..., curated_dir=..., artifacts_dir=...)`. It delegates through `forward_nail_v1212_back_to_back` to `train_forward_portable_matchup_contextual_rapm`. The selected run uses residualized chronological lambda CV, context curvature/temporal penalties 0, and lambda grid `[1e-5,3e-5,1e-4,3e-4,.001,.003,.01,.03,.1,.3,1,3,10]`.

Required training inputs are the player-season panel, historical RAPM stints/possession data, curated game/schedule data for back-to-back features, and profile/prior inputs used by that trainer. Defaults point to `data/analytical/player_season_panel/player_seasons.parquet`, `data/analytical`, and `data/curated`; these are lookup defaults, not verified immutable training snapshots. Supply explicit pinned paths.

An exact from-raw historical retrain remains unproven: the inspected metadata records code_version `sha256:a1ae9fd4b87638e7af6b85059f4ab32a3679199505f2f1063333adc285cf47a4`, rather than a Git commit, and this inspection did not establish a complete immutable inventory of the original historical training inputs. Recover and verify those snapshots before promising bit-for-bit reproduction. Do not infer them from today's analytical directory, or replace accepted preseason inputs by rebuilding them opportunistically.

## 9. Verification performed and remaining gates

Performed: current implementation compatibility check passed; accepted configuration hash reproduced; both exact frozen-prior hashes reproduced; all selected production model/data component hashes passed (12/12 and 9/9); all selected dev model/data component hashes passed (13/13 and 9/9); output schemas, tip row counts, joint Gram dimensions, HCA states, and available date-directory bounds inspected.

Not performed: full historical training, numerical replay, every-date/every-game schedule coverage validation, full-chain continuity audit, API-response validation, or rendered UI validation. Tip component integrity does not establish complete season coverage. Release metadata contains coverage counts from earlier checks; do not treat those as fresh verification.

For acceptance of future work: validate every expected game/date against canonical schedule and source coverage; validate all data/model parent and hash links; check reconciled identities and model posterior values against reference outputs; verify forecast-time cutoffs and immutable past forecasts; use canonical dev materialization/publication; verify API data and rendered dev UI. Production promotion requires explicit approval. Do not revive the deprecated shared Dagster services or use low-level pointer edits.

The requested artifact is this handoff. No model or site changes are included.
