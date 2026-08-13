from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from nba_lineup_model.audit import (
    AuditGameSpec,
    AuditManifest,
    audit_game_payloads,
    audit_reconstruction,
    run_audit_manifest,
    sample_audit_manifest,
)
from nba_lineup_model.build_game import reconstruct_game_payloads
from nba_lineup_model.ingest.nba_cdn import CachedResponse, NbaCdnEndpoint

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def audit_boxscore_fixture() -> dict:
    payload = deepcopy(load_fixture("boxscore_lineup_scenario.json"))
    game = payload["game"]
    game.update(
        {
            "gameStatus": 3,
            "gameTimeUTC": "2020-01-01T00:00:00Z",
            "period": 2,
        }
    )
    for side, score in (("homeTeam", 2), ("awayTeam", 2)):
        game[side]["score"] = score
        game[side]["statistics"] = {
            "fieldGoalsAttempted": 1,
            "freeThrowsAttempted": 0,
            "reboundsOffensive": 0,
            "turnoversTotal": 0,
        }
    return payload


class StubPayloadSource:
    def __init__(self, game_id: str, play_by_play: dict, boxscore: dict) -> None:
        self.game_id = game_id
        self.play_by_play = play_by_play
        self.boxscore = boxscore

    def fetch_play_by_play(
        self,
        game_id: str,
        *,
        use_cache: bool = True,
    ) -> CachedResponse:
        if game_id != self.game_id:
            raise RuntimeError("fixture game is unavailable")
        return CachedResponse(
            endpoint=NbaCdnEndpoint.PLAY_BY_PLAY,
            game_id=game_id,
            url="https://example.test/playbyplay.json",
            payload=self.play_by_play,
        )

    def fetch_boxscore(
        self,
        game_id: str,
        *,
        use_cache: bool = True,
    ) -> CachedResponse:
        if game_id != self.game_id:
            raise RuntimeError("fixture game is unavailable")
        return CachedResponse(
            endpoint=NbaCdnEndpoint.BOXSCORE,
            game_id=game_id,
            url="https://example.test/boxscore.json",
            payload=self.boxscore,
        )


def test_audit_game_payloads_checks_full_pipeline():
    spec = AuditGameSpec(
        game_id="0020000001",
        season="2020-21",
        season_type="regular",
        sample_group="baseline",
        expected_overtime=False,
    )

    result = audit_game_payloads(
        spec,
        load_fixture("playbyplay_lineup_scenario.json"),
        audit_boxscore_fixture(),
    )

    assert result.status == "pass"
    assert result.event_count == 13
    assert result.possession_count == 2
    assert result.home_possession_count == 1
    assert result.away_possession_count == 1
    assert result.possession_segment_count == 3
    assert result.score_matches_boxscore is True
    assert result.possession_score_conserved is True
    assert result.segment_score_conserved is True
    assert result.segment_duration_conserved is True
    assert result.issue_codes == ()


def test_audit_fails_when_overtime_expectation_is_wrong():
    spec = AuditGameSpec(
        game_id="0020000001",
        season="2020-21",
        season_type="regular",
        expected_overtime=True,
    )

    result = audit_game_payloads(
        spec,
        load_fixture("playbyplay_lineup_scenario.json"),
        audit_boxscore_fixture(),
    )

    assert result.status == "fail"
    assert "audit:overtime_expectation_mismatch" in result.issue_codes


def test_audit_warns_when_period_possession_counts_are_unbalanced_but_conserve():
    spec = AuditGameSpec(
        game_id="0020000001",
        season="2020-21",
        season_type="regular",
    )
    play_by_play = load_fixture("playbyplay_lineup_scenario.json")
    reconstruction = reconstruct_game_payloads(play_by_play, audit_boxscore_fixture())
    first = reconstruction.possessions.possessions[0]
    second = reconstruction.possessions.possessions[1]
    reconstruction.possessions.possessions[1] = second.model_copy(
        update={
            "offense_team_id": first.offense_team_id,
            "defense_team_id": first.defense_team_id,
            "period": first.period,
            "period_possession_index": 1,
        }
    )

    result = audit_reconstruction(spec, reconstruction, audit_boxscore_fixture())

    assert result.status == "warning"
    assert result.possession_score_conserved is True
    assert result.segment_score_conserved is True
    assert "audit:unbalanced_period_possession_counts" in result.issue_codes


def test_audit_runner_records_fetch_errors_and_continues(tmp_path: Path):
    valid = AuditGameSpec(
        game_id="0020000001",
        season="2020-21",
        season_type="regular",
    )
    unavailable = AuditGameSpec(
        game_id="0020000010",
        season="2020-21",
        season_type="regular",
    )
    source = StubPayloadSource(
        valid.game_id,
        load_fixture("playbyplay_lineup_scenario.json"),
        audit_boxscore_fixture(),
    )

    run = run_audit_manifest(
        AuditManifest(games=[valid, unavailable]),
        client=source,
        output_dir=tmp_path,
    )

    assert [result.status for result in run.results] == ["pass", "error"]
    assert run.results[1].error_stage == "fetch"
    assert all(path.exists() for path in run.output_paths.values())
    summary = pd.read_parquet(run.output_paths["summary"])
    assert summary.loc[0, "game_count"] == 2
    assert summary.loc[0, "pass_count"] == 1
    assert summary.loc[0, "error_count"] == 1


def test_manifest_rejects_duplicate_game_ids():
    game = {
        "game_id": "0020000001",
        "season": "2020-21",
        "season_type": "regular",
    }

    with pytest.raises(ValidationError, match="duplicate game IDs"):
        AuditManifest(games=[game, game])


def test_stratified_sampling_is_deterministic_and_round_trips(tmp_path: Path):
    catalog = pd.DataFrame(
        [
            {
                "game_id": f"002{season_code}{index:05d}",
                "season": season,
                "season_type": "regular",
                "sample_group": "baseline",
                "expected_overtime": False,
            }
            for season, season_code in (("2020-21", "20"), ("2021-22", "21"))
            for index in range(1, 5)
        ]
    )

    first = sample_audit_manifest(catalog, games_per_stratum=2, random_seed=7)
    second = sample_audit_manifest(catalog, games_per_stratum=2, random_seed=7)
    output_path = first.write(tmp_path / "manifest.json")

    assert len(first.games) == 4
    assert [game.game_id for game in first.games] == [
        game.game_id for game in second.games
    ]
    assert AuditManifest.read(output_path) == first


def test_stratified_sampling_rejects_null_strata():
    catalog = pd.DataFrame(
        [
            {
                "game_id": "0022000001",
                "season": "2020-21",
                "season_type": None,
            }
        ]
    )

    with pytest.raises(ValueError, match="null values"):
        sample_audit_manifest(catalog, games_per_stratum=1)
