"""Dagster definitions for NBA GESTALT pipelines."""

from dagster import Definitions, EnvVar

from .box_scores.assets import nba_box_scores
from .box_scores.jobs import capture_nba_box_scores
from .box_scores.resources import NBABoxScoreArchive
from .box_scores.schedules import box_score_poll
from .injury_reports.assets import injury_pdf_archive, injury_report_extractions, injury_status_by_game_date
from .injury_reports.jobs import capture_injury_pdfs, extract_injury_pdf, reconcile_injury_date
from .injury_reports.partitions import INJURY_REPORT_SEASON
from .injury_reports.resources import InjuryReportArchive, OfficialInjuryReports
from .injury_reports.schedules import injury_pdf_poll
from .injury_reports.sensors import injury_report_dates_ready, new_injury_pdf
from .minutes.assets import game_minutes, player_minutes
from .minutes.jobs import project_game_minutes, update_player_minutes
from .minutes.resources import MinutesArchive
from .minutes.sensors import game_minutes_ready, minutes_ready
from .odds.assets import nba_market_odds
from .odds.jobs import archive_nba_market_odds
from .odds.resources import OddsApi
from .odds.schedules import nba_market_odds_game_day
from .play_by_play.assets import nba_play_by_play, nba_stints
from .play_by_play.jobs import capture_nba_play_by_play, parse_nba_games
from .play_by_play.resources import NBAPlayByPlayArchive
from .play_by_play.schedules import play_by_play_morning
from .play_by_play.sensors import play_by_play_ready_for_stints
from .resources import NBAPlayerNamesResource, NBAScheduleResource
from .rosters.assets import rosters_by_date
from .rosters.jobs import build_nba_rosters
from .rosters.resources import NBARosterArchive
from .rosters.schedules import rosters_morning
from .rosters.sensors import roster_transactions_ready
from .scoreboard.assets import nba_scoreboard
from .scoreboard.jobs import capture_nba_scoreboard
from .scoreboard.resources import NBAScoreboardArchive
from .scoreboard.schedules import scoreboard_poll
from .season_preparation.assets import nba_season_schedule
from .season_preparation.jobs import capture_nba_season_schedule
from .transactions.assets import transactions_by_date
from .transactions.jobs import capture_nba_transactions
from .transactions.resources import NBATransactions
from .transactions.schedules import transactions_hourly

defs = Definitions(
    assets=[
        injury_pdf_archive, injury_report_extractions, injury_status_by_game_date, player_minutes, game_minutes,
        nba_season_schedule, nba_market_odds, nba_box_scores, nba_scoreboard, transactions_by_date, nba_play_by_play, nba_stints, rosters_by_date,
    ],
    jobs=[
        capture_injury_pdfs, extract_injury_pdf, reconcile_injury_date, update_player_minutes, project_game_minutes,
        capture_nba_season_schedule, archive_nba_market_odds, capture_nba_box_scores, capture_nba_scoreboard,
        capture_nba_transactions, capture_nba_play_by_play, parse_nba_games, build_nba_rosters,
    ],
    schedules=[injury_pdf_poll, nba_market_odds_game_day, box_score_poll, scoreboard_poll, transactions_hourly, play_by_play_morning, rosters_morning],
    sensors=[new_injury_pdf, injury_report_dates_ready, roster_transactions_ready, play_by_play_ready_for_stints, minutes_ready, game_minutes_ready],
    resources={
        "official_injury_reports": OfficialInjuryReports(),
        "injury_report_archive": InjuryReportArchive(season=INJURY_REPORT_SEASON),
        "nba_box_score_archive": NBABoxScoreArchive(),
        "nba_play_by_play_archive": NBAPlayByPlayArchive(),
        "nba_roster_archive": NBARosterArchive(),
        "minutes_archive": MinutesArchive(season=EnvVar("GESTALT_NBA_SEASON")),
        "nba_transactions": NBATransactions(season=EnvVar("GESTALT_NBA_SEASON")),
        "odds_api": OddsApi(api_key=EnvVar("GESTALT_ODDS_API_KEY")),
        "nba_scoreboard_archive": NBAScoreboardArchive(),
        "nba_player_names": NBAPlayerNamesResource(season=EnvVar("GESTALT_NBA_SEASON")),
        "nba_schedule": NBAScheduleResource(season=EnvVar("GESTALT_NBA_SEASON")),
    },
)
