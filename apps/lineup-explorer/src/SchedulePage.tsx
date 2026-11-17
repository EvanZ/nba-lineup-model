import { releaseFetch as fetch } from "./releaseFetch";
import { useEffect, useMemo, useState } from "react";
import { CalendarDays, ChevronLeft, ChevronRight, CircleAlert, Radio, TrendingUp } from "lucide-react";

import { teamLogoUrl } from "./teamLogos";
import { MarketPerformanceView, useMarketPerformanceReport, type CalendarSummary, type DailyRecord, type Report } from "./MarketPerformanceReport";

type MarketLine = {
  provider: string;
  provider_key?: string;
  selection: string;
  spread_captured_at: string | null;
  moneyline_captured_at: string | null;
  spread_bookmaker_count: number;
  moneyline_bookmaker_count: number;
  bookmaker_keys?: string[];
  bookmaker_count?: number;
  bookmakers?: BookmakerLineDetail[];
  home_spread: number | null;
  away_spread: number | null;
  total: number | null;
  away_moneyline: number | null;
  home_moneyline: number | null;
  away_win_probability?: number | null;
  home_win_probability?: number | null;
};

type BookmakerLineDetail = {
  key: string;
  title: string;
  home_spread: number | null;
  away_spread: number | null;
  home_moneyline: number | null;
  away_moneyline: number | null;
  home_win_probability: number | null;
  away_win_probability: number | null;
  last_update: string | null;
};

const BOOKMAKER_BADGES: Record<string, { short: string; label: string }> = {
  draftkings: { short: "DK", label: "DraftKings" },
  fanduel: { short: "FD", label: "FanDuel" },
  betmgm: { short: "MGM", label: "BetMGM" },
  betrivers: { short: "BR", label: "BetRivers" },
  hardrockbet: { short: "HR", label: "Hard Rock Bet" },
  betparx: { short: "PX", label: "betPARX" },
  ballybet: { short: "BB", label: "Bally Bet" },
  betonlineag: { short: "BO", label: "BetOnline" },
};

const BOOKMAKER_ORDER = Object.keys(BOOKMAKER_BADGES);

type PredictionMarket = {
  provider: string;
  selection: string;
  home_spread?: number;
  away_spread?: number;
  home_win_probability?: number;
  away_win_probability?: number;
  spreads_captured_at?: string;
  h2h_captured_at?: string;
};

type ProjectedMinute = {
  player_id: number;
  player_name: string | null;
  status: string | null;
  play_probability: number | null;
  conditional_minutes: number | null;
  projected_minutes: number | null;
  projected_nail?: number | null;
  expected_contribution?: number | null;
  reason: string | null;
};

type ProjectionProvenance = {
  schema_version: number;
  rating_as_of: string | null;
  as_of_cutoff_time: string | null;
  inputs: Record<string, unknown>;
};

type SpreadLedgerTeam = {
  team: string;
  projected_minutes?: number;
  actual_minutes?: number;
  team_strength: number;
};

type SpreadLedger = {
  schema_version: number;
  artifact_kind?: "saved_pipeline_prediction_attribution" | "immutable_game_prediction_attribution" | "retrospective_actual_minutes_attribution";
  contract: string;
  assumed_possessions?: number;
  away: SpreadLedgerTeam;
  home: SpreadLedgerTeam;
  team_strength_difference: number;
  home_court_adjustment: number;
  back_to_back: {
    away_team_is_back_to_back: boolean;
    home_team_is_back_to_back: boolean;
    coefficient: number;
    adjustment: number;
  };
  home_margin: number;
  fair_home_spread: number;
  provenance: ProjectionProvenance;
};

type ActualMinute = {
  player_id: number;
  player_name: string | null;
  projected_minutes: number;
  projected_contribution: number;
  contribution_delta: number;
  in_baseline_prediction: boolean;
  rating_source: string;
  actual_minutes: number;
  frozen_nail?: number | null;
  actual_contribution?: number | null;
  rating_matched: boolean;
};

type FinalAudit = {
  recorded_outcome: { home_score: number; away_score: number };
  inputs: Record<string, unknown>;
  grading: {
    model_favorite_won: boolean;
    market: { model_ats_result: "win" | "loss" | "push" } | null;
  };
};

type ArchivedGame = {
  game_id: string;
  game_date: string;
  prediction_purpose: "pregame_projection" | "historical_replay" | null;
  start_time: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  broadcasts: string[];
  availability: { market: string; grading: string; postgame_details: string };
  detail_errors?: Record<string, string>;
  status: { state: string; completed: boolean; description: string; detail: string | null; display_clock?: string | null; period?: number | null };
  projection: {
    model: string;
    state_kind?: string;
    model_track?: string;
    rating_as_of?: string | null;
    home_margin: number;
    home_win_probability: number;
    fair_home_spread?: number | null;
    input_provenance?: ProjectionProvenance;
    ledger?: SpreadLedger;
  } | null;
  market: MarketLine | null;
  prediction_market?: PredictionMarket | null;
  injury_report: {
    source: string;
    source_url: string | null;
    published_at: string | null;
    team_reports?: Record<string, { status: string }>;
  } | null;
  projected_minutes: { away: ProjectedMinute[]; home: ProjectedMinute[] } | null;
  retrospective?: {
    recorded_outcome: { home_score: number; away_score: number };
    computed_at: string;
    method: string;
    home_margin: number;
    home_win_probability: number;
    actual_minutes: { away: ActualMinute[]; home: ActualMinute[] };
    ledger?: SpreadLedger;
  };
  final_audit?: FinalAudit | null;
  frozen_at?: string;
  data_quality_errors?: string[];
};

type ScheduleDay = {
  available: boolean;
  game_date: string;
  collected_at?: string | null;
  live_score_collected_at?: string | null;
  source: string;
  forecast_version?: { model_track?: string; evaluation_cell?: string; mode?: string; through_date?: string; identity?: string };
  games: ArchivedGame[];
  coverage: { missing_prediction_game_ids: string[]; missing_observation_game_ids: string[] };
};

type CalendarDay = CalendarSummary;
type CalendarCount = Pick<CalendarSummary, "game_date" | "game_count">;

function dateKey(date: Date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function parseDate(value: string) {
  const [year, month, day] = value.split("-").map(Number);
  return new Date(year, month - 1, day, 12);
}

function isDateKey(value: string | undefined): value is string {
  if (!value || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  return dateKey(parseDate(value)) === value;
}

function scheduleHref(value: string) {
  return `#schedule/${value}`;
}

function playerHref(playerId: number) {
  return `#player/${playerId}`;
}

function teamHref(team: string) {
  return `#team/${team}`;
}

function addDays(value: string, amount: number) {
  const date = parseDate(value);
  date.setDate(date.getDate() + amount);
  return dateKey(date);
}

function longDate(value: string) {
  return new Intl.DateTimeFormat("en-US", { weekday: "long", month: "long", day: "numeric" }).format(parseDate(value));
}

function shortDay(value: string) {
  return new Intl.DateTimeFormat("en-US", { weekday: "short" }).format(parseDate(value));
}

function shortMonth(value: string) {
  return new Intl.DateTimeFormat("en-US", { month: "short" }).format(parseDate(value));
}


function formatSpread(value: number) {
  if (Math.abs(value) < 0.05) return "PK";
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}`;
}

function formatMoneyline(value: number | null) {
  if (value === null) return "—";
  return `${value > 0 ? "+" : ""}${value}`;
}

function americanImpliedProbability(value: number | null) {
  if (value === null || value === 0) return null;
  return value > 0 ? 100 / (value + 100) : -value / (-value + 100);
}

function marketProbabilities(market: MarketLine) {
  if (market.home_win_probability != null && market.away_win_probability != null) {
    return { home: market.home_win_probability, away: market.away_win_probability };
  }
  const home = americanImpliedProbability(market.home_moneyline);
  const away = americanImpliedProbability(market.away_moneyline);
  if (home === null || away === null || home + away <= 0) return null;
  return { home: home / (home + away), away: away / (home + away) };
}

function gameTime(value?: string) {
  if (!value) return "Time TBD";
  return new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/New_York",
    timeZoneName: "short",
  }).format(new Date(value));
}

function archiveAge(value?: string | null) {
  if (!value) return "Awaiting first archive";
  return `Updated ${new Intl.DateTimeFormat("en-US", { hour: "numeric", minute: "2-digit", timeZoneName: "short" }).format(new Date(value))}`;
}

function dailyRecordLabel(record: DailyRecord) {
  const atsGames = record.atsWins + record.atsLosses + record.atsPushes;
  const ats = atsGames
    ? `${record.atsWins}-${record.atsLosses}${record.atsPushes ? `-${record.atsPushes}` : ""}`
    : "—";
  const winners = record.wins + record.losses ? `${record.wins}-${record.losses}` : "—";
  return `W-L ${winners} · ATS ${ats}`;
}

export function CalendarResult({ day }: { day?: CalendarDay }) {
  if (!day || day.record_status === "pending" || day.record_status === "no_games") return null;
  if (!day.record) {
    const reason = day.record_status === "invalid" ? "Saved report failed validation." : "No saved model results are available.";
    return <small className="schedule-date-record unavailable" title={reason}>Results unavailable</small>;
  }
  const coverage = day.record_coverage;
  const detail = `${day.record_status === "partial" ? "Partial saved results" : "Saved results"}: winner grades ${coverage?.model_graded_game_count ?? "—"}/${day.game_count}; ATS grades ${coverage?.ats_graded_game_count ?? "—"}/${day.game_count}${coverage?.ats_no_pick_game_count ? `; ${coverage.ats_no_pick_game_count} no pick` : ""}.`;
  const [winner, ats] = dailyRecordLabel(day.record).split(" · ");
  return <small className="schedule-date-record" title={detail} aria-label={`${dailyRecordLabel(day.record)}. ${detail}`}><span>{winner}</span>{" · "}<span>{ats}{day.record_status === "partial" ? " *" : ""}</span></small>;
}

export function calendarSummary(day: CalendarCount | undefined, report: Report | null, error: string | null): CalendarSummary | undefined {
  if (!day) return undefined;
  const empty = { ...day, record: null };
  if (!day.game_count) return { ...empty, record_status: "no_games" };
  if (!report) return error ? { ...empty, record_status: "unavailable" } : undefined;
  const saved = report.calendar[day.game_date];
  if (saved) return saved.game_count === day.game_count ? saved : { ...empty, record_status: "invalid" };
  return { ...empty, record_status: day.game_date > report.through ? "pending" : "missing" };
}

function reportTime(value?: string | null) {
  if (!value) return "Report time unavailable";
  return new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/New_York",
    timeZoneName: "short",
  }).format(new Date(value));
}

function formatGameClock(value?: string | null) {
  if (!value) return "";
  const match = /^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$/i.exec(value.trim());
  if (!match) return value;
  const minutes = Number(match[1] ?? 0) * 60 + Number(match[2] ?? 0);
  const seconds = Number(match[3] ?? 0);
  const secondsText = Number.isInteger(seconds)
    ? String(seconds).padStart(2, "0")
    : seconds.toFixed(1).padStart(4, "0");
  return `${minutes}:${secondsText}`;
}

function liveStatus(game: ArchivedGame | undefined) {
  if (!game) return "Pregame";
  if (game.status.completed) return "Final";
  if (game.status.state !== "in") return game.status.description || "Pregame";
  if (game.status.description.toLowerCase().includes("halftime")) return "Halftime";
  const period = game.status.period ?? 0;
  const periodText = period <= 4 ? `Q${period}` : period === 5 ? "OT" : `${period - 4}OT`;
  return `${periodText} · ${formatGameClock(game.status.display_clock)}`.replace(/ · $/, "");
}


function signedValue(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (Math.abs(value) < 0.5 * 10 ** -digits) return (0).toFixed(digits);
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function RotationTable({
  team,
  players,
  teamLedger,
  actualTeamLedger,
  actualPlayers,
  reportStatus,
}: {
  team: string;
  players: ProjectedMinute[];
  teamLedger?: SpreadLedgerTeam;
  actualTeamLedger?: SpreadLedgerTeam;
  actualPlayers?: ActualMinute[];
  reportStatus?: string;
}) {
  const reportPending = reportStatus !== undefined && reportStatus !== "submitted";
  const hasUnknownPlayers = reportPending && players.some((player) => player.status === "Report pending");
  const hasAttribution = Boolean(
    teamLedger
    && players.some((player) => player.projected_nail != null && player.expected_contribution != null),
  );
  const actualByPlayer = new Map((actualPlayers ?? []).map((player) => [player.player_id, player]));
  const predictedIds = new Set(players.map((player) => player.player_id));
  const rows = [
    ...players,
    ...(actualPlayers ?? []).filter((player) => !predictedIds.has(player.player_id)).map((player) => ({
      player_id: player.player_id,
      player_name: player.player_name,
      status: null,
      play_probability: null,
      conditional_minutes: null,
      projected_minutes: player.projected_minutes,
      projected_nail: player.frozen_nail ?? null,
      expected_contribution: player.projected_contribution,
      reason: player.rating_source === "replacement" ? "Not in baseline prediction · saved replacement rating" : "Not in baseline prediction",
    })),
  ];
  const teamContributionDifference = actualTeamLedger
    ? actualTeamLedger.team_strength - (teamLedger?.team_strength ?? 0)
    : null;
  const teamMinutesDifference = actualTeamLedger?.actual_minutes != null
    && teamLedger?.projected_minutes != null
    ? actualTeamLedger.actual_minutes - teamLedger.projected_minutes
    : null;
  return <div className="schedule-rotation-team">
    <h4><a href={teamHref(team)}>{team}</a> {actualPlayers ? "rotation: projected & actual" : "projected rotation"}</h4>
    {hasUnknownPlayers && <p className="schedule-report-pending">Unlisted availability remains unknown — team report coverage is incomplete. Listed statuses are shown below.</p>}
    <div className={`schedule-rotation-table ${hasAttribution ? "has-attribution" : ""} ${actualPlayers ? "has-actual" : ""}`}>
      {(hasAttribution || actualPlayers) && <div className="schedule-rotation-header">
        <span>Player</span>
        {hasAttribution && <><small>NAIL</small><small>Proj. min.</small><small>Proj. contrib.</small></>}
        {!hasAttribution && actualPlayers && <small>Pred.</small>}
        {actualPlayers && <><small>Act. min.</small><small>Act. contrib.</small><small>Δ contrib.</small><small>Δ min.</small></>}
      </div>}
      {rows.map((player) => {
        const actual = actualByPlayer.get(player.player_id);
        const difference = actual && player.projected_minutes != null ? actual.actual_minutes - player.projected_minutes : null;
        const actualContribution = actual?.actual_contribution ?? null;
        const contributionDifference = actual?.contribution_delta ?? null;
        const displayedRating = player.projected_nail ?? actual?.frozen_nail;
        const availability = player.status === "Report pending"
          ? "Status unknown — team report pending"
          : `${player.status ?? "Unknown"}${player.reason ? ` · ${player.reason}` : ""}`;
        return <div key={player.player_id} className={player.status !== "Available" ? "injury-listed" : ""}>
        <span className="schedule-rotation-player">{player.player_id > 0 ? <a className="schedule-player-link" href={playerHref(player.player_id)}>{player.player_name ?? `NBA player ${player.player_id}`}</a> : player.player_name ?? "Name unavailable"}<small>{availability}</small></span>
        {hasAttribution && <>
          <strong data-label="NAIL">{signedValue(displayedRating)}</strong>
          <strong data-label="Minutes">{player.projected_minutes?.toFixed(1) ?? "—"}</strong>
          <strong data-label="Contribution" className={player.expected_contribution != null && player.expected_contribution < 0 ? "negative" : "positive"}>{signedValue(player.expected_contribution)}</strong>
        </>}
        {!hasAttribution && <strong data-label="Predicted">{player.projected_minutes?.toFixed(1) ?? "—"}</strong>}
        {actualPlayers && <><strong data-label="Actual minutes">{actual?.actual_minutes.toFixed(1) ?? "—"}</strong><strong data-label="Actual contribution" className={actualContribution !== null && actualContribution < 0 ? "negative" : "positive"}>{signedValue(actualContribution)}</strong><strong data-label="Contribution delta" className={contributionDifference !== null && contributionDifference < 0 ? "negative" : "positive"}>{signedValue(contributionDifference)}</strong><strong data-label="Minutes delta" className={difference !== null && difference < 0 ? "negative" : "positive"}>{difference === null ? "—" : `${difference > 0 ? "+" : ""}${difference.toFixed(1)}`}</strong></>}
      </div>;})}
      {hasAttribution && teamLedger && <div className="schedule-rotation-total">
        <span>Team total</span>
        <strong data-label="NAIL">—</strong>
        <strong data-label="Minutes">{teamLedger.projected_minutes?.toFixed(1) ?? "—"}</strong>
        <strong data-label="Contribution">{signedValue(teamLedger.team_strength)}</strong>
        {actualPlayers && <><strong data-label="Actual minutes">{actualTeamLedger?.actual_minutes?.toFixed(1) ?? "—"}</strong><strong data-label="Actual contribution">{signedValue(actualTeamLedger?.team_strength)}</strong><strong data-label="Contribution delta" className={teamContributionDifference !== null && teamContributionDifference < 0 ? "negative" : "positive"}>{signedValue(teamContributionDifference)}</strong><strong data-label="Minutes delta" className={teamMinutesDifference !== null && teamMinutesDifference < 0 ? "negative" : "positive"}>{signedValue(teamMinutesDifference, 1)}</strong></>}
      </div>}
    </div>
  </div>;
}

function SpreadLedgerView({ ledger, actualLedger, postgame, homeWinProbability }: {
  ledger: SpreadLedger;
  actualLedger?: SpreadLedger;
  postgame?: ArchivedGame["retrospective"];
  homeWinProbability: number;
}) {
  const awayTeam = ledger.away.team;
  const homeTeam = ledger.home.team;
  const backToBackTeams = [
    ledger.back_to_back.away_team_is_back_to_back ? awayTeam : null,
    ledger.back_to_back.home_team_is_back_to_back ? homeTeam : null,
  ].filter((team): team is string => Boolean(team));
  const backToBackNote = backToBackTeams.length
    ? `${backToBackTeams.join(" & ")} ${backToBackTeams.length > 1 ? "are" : "is"} on a back-to-back`
    : "Neither team is on a back-to-back";
  return <section className={`schedule-spread-ledger ${actualLedger ? "has-actual" : ""}`} aria-label="Model spread ledger">
    <div className="schedule-spread-ledger-heading">
      <div><span>Model spread ledger</span><strong>From player value to the line</strong></div>
      <small>{ledger.assumed_possessions == null ? "Possessions unavailable" : `${ledger.assumed_possessions} possessions per game`}</small>
    </div>
    <div className="schedule-spread-ledger-rows">
      {actualLedger && <div className="columns"><span></span><small>Projected</small><small title={postgame ? `Postgame actual-minutes counterfactual · Saved ${quoteTime(postgame.computed_at)}` : undefined}>Actual minutes</small></div>}
      <div><span><a href={teamHref(awayTeam)}>{awayTeam}</a> player contributions</span><strong>{signedValue(ledger.away.team_strength)}</strong>{actualLedger && <strong>{signedValue(actualLedger.away.team_strength)}</strong>}</div>
      <div><span><a href={teamHref(homeTeam)}>{homeTeam}</a> player contributions</span><strong>{signedValue(ledger.home.team_strength)}</strong>{actualLedger && <strong>{signedValue(actualLedger.home.team_strength)}</strong>}</div>
      <div><span>{homeTeam} minus {awayTeam}</span><strong>{signedValue(ledger.team_strength_difference)}</strong>{actualLedger && <strong>{signedValue(actualLedger.team_strength_difference)}</strong>}</div>
      <div><span>Home court</span><strong>{signedValue(ledger.home_court_adjustment)}</strong>{actualLedger && <strong>{signedValue(actualLedger.home_court_adjustment)}</strong>}</div>
      <div><span>Back-to-back adjustment<small>{backToBackNote}</small></span><strong>{signedValue(ledger.back_to_back.adjustment)}</strong>{actualLedger && <strong>{signedValue(actualLedger.back_to_back.adjustment)}</strong>}</div>
      <div className="result final"><span>Model spread</span><strong>{homeTeam} {formatSpread(ledger.fair_home_spread)}</strong>{actualLedger && <strong>{homeTeam} {formatSpread(actualLedger.fair_home_spread)}</strong>}</div>
      {postgame && <div className="result"><span>{homeTeam} win probability</span><strong>{(homeWinProbability * 100).toFixed(1)}%</strong><strong>{(postgame.home_win_probability * 100).toFixed(1)}%</strong></div>}
    </div>
    <p>{postgame ? "Actual-minutes column: saved postgame counterfactual." : "Saved prediction ratings and minutes."}{postgame?.recorded_outcome && <> Recorded outcome: {awayTeam} {postgame.recorded_outcome.away_score} · {homeTeam} {postgame.recorded_outcome.home_score}</>}</p>
  </section>;
}

function gameCountLabel(count: number) {
  return `${count} ${count === 1 ? "game" : "games"}`;
}




function bookmakerTooltip(line: BookmakerLineDetail | undefined, awayTeam: string, homeTeam: string, fallbackLabel: string) {
  if (!line) return fallbackLabel;
  const awayLine = [
    line.away_spread === null ? null : formatSpread(line.away_spread),
    line.away_moneyline === null ? null : formatMoneyline(line.away_moneyline),
  ].filter(Boolean).join(" · ");
  const homeLine = [
    line.home_spread === null ? null : formatSpread(line.home_spread),
    line.home_moneyline === null ? null : formatMoneyline(line.home_moneyline),
  ].filter(Boolean).join(" · ");
  return `${line.title}\n${awayTeam} ${awayLine || "line unavailable"}\n${homeTeam} ${homeLine || "line unavailable"}`;
}

function BookmakerAvatars({ market, awayTeam, homeTeam }: { market?: MarketLine | null; awayTeam: string; homeTeam: string }) {
  const bookmakerKeys = market?.bookmaker_keys;
  if (!bookmakerKeys?.length) return null;
  const uniqueKeys = [...new Set(bookmakerKeys)];
  const orderedKeys = [
    ...BOOKMAKER_ORDER.filter((key) => uniqueKeys.includes(key)),
    ...uniqueKeys.filter((key) => !BOOKMAKER_BADGES[key]).sort(),
  ];
  const visibleKeys = orderedKeys.slice(0, 6);
  const names = orderedKeys.map((key) => BOOKMAKER_BADGES[key]?.label ?? key);
  const linesByKey = new Map((market?.bookmakers ?? []).map((line) => [line.key, line]));
  return <div className="schedule-bookmaker-avatars" aria-label={`Contributing books: ${names.join(", ")}`}>
    {visibleKeys.map((key) => {
      const badge = BOOKMAKER_BADGES[key] ?? { short: key.slice(0, 2).toUpperCase(), label: key };
      return <span key={key} className="schedule-bookmaker-avatar" data-bookmaker={key} title={bookmakerTooltip(linesByKey.get(key), awayTeam, homeTeam, badge.label)}>{badge.short}</span>;
    })}
    {orderedKeys.length > visibleKeys.length && <span className="schedule-bookmaker-avatar overflow" title={orderedKeys.slice(visibleKeys.length).map((key) => bookmakerTooltip(linesByKey.get(key), awayTeam, homeTeam, BOOKMAKER_BADGES[key]?.label ?? key)).join("\n\n")}>+{orderedKeys.length - visibleKeys.length}</span>}
  </div>;
}


function quoteTime(value?: string | null) {
  if (!value) return "Unavailable";
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZone: "America/Los_Angeles", timeZoneName: "short" }).format(new Date(value));
}

function ScheduleMarketPanel({ game }: { game: ArchivedGame }) {
  const market = game.market;
  const poly = game.prediction_market;
  const probability = market ? marketProbabilities(market) : null;
  const percent = (value?: number | null) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
  return <div className="schedule-market-panel">
    <div className="schedule-market-heading">
      <p>{market?.provider ?? "Market line"}</p>
      <div className="schedule-market-meta">
        <span className="schedule-market-tag" title={market ? `Spread · ${market.spread_bookmaker_count} books · ${quoteTime(market.spread_captured_at)}
Moneyline · ${market.moneyline_bookmaker_count} books · ${quoteTime(market.moneyline_captured_at)}` : undefined}>{market?.selection ?? poly?.selection ?? "Pregame"}</span>
        <BookmakerAvatars market={market} awayTeam={game.away_team} homeTeam={game.home_team} />
      </div>
    </div>
    {market ? <>
      <div className="schedule-market-teams">
        <div><span>{game.away_team}</span><strong>{market.away_spread == null ? "—" : formatSpread(market.away_spread)}</strong><small>{percent(probability?.away)}</small></div>
        <div><span>{game.home_team}</span><strong>{market.home_spread == null ? "—" : formatSpread(market.home_spread)}</strong><small>{percent(probability?.home)}</small></div>
      </div>
    </> : <p className="schedule-market-empty">Market data unavailable</p>}
    {poly && <div className="schedule-polymarket-row" title={`Spread · ${quoteTime(poly.spreads_captured_at)}
Moneyline · ${quoteTime(poly.h2h_captured_at)}`} aria-label={`Polymarket for ${game.away_team} at ${game.home_team}`}>
      <span>Polymarket</span>
      <small>{game.away_team} {poly.away_spread == null ? "—" : formatSpread(poly.away_spread)} · {percent(poly.away_win_probability)}</small>
      <small>{game.home_team} {poly.home_spread == null ? "—" : formatSpread(poly.home_spread)} · {percent(poly.home_win_probability)}</small>
    </div>}
  </div>;
}

export function ScheduleGameCard({ game }: { game: ArchivedGame }) {
  const archived = game;
  const pregame = game;
  if (!pregame?.projection) return <article className="schedule-game-card">
    <div className="schedule-game-context">
      <a href={`https://www.nba.com/game/${game.away_team.toLowerCase()}-vs-${game.home_team.toLowerCase()}-${game.game_id}`} target="_blank" rel="noreferrer">NBA ↗</a>
      <span>{gameTime(game.start_time)}</span>
      <span>{game.broadcasts.join(" · ") || "Broadcast unavailable"}</span>
      <span>{liveStatus(archived)}</span>
    </div>
    <div className="schedule-matchup">
      <a className="schedule-team away" href={teamHref(game.away_team)}><img alt="" src={teamLogoUrl(game.away_team)} /><strong>{game.away_team} {archived?.away_score ?? ""}</strong></a>
      <span>at</span>
      <a className="schedule-team home" href={teamHref(game.home_team)}><img alt="" src={teamLogoUrl(game.home_team)} /><strong>{game.home_team} {archived?.home_score ?? ""}</strong></a>
    </div>
    <div className="schedule-projection-panel"><p>Forecast unavailable</p><small>No saved forecast is available for this game.</small></div>
    <ScheduleMarketPanel game={game} />
  </article>;
  const projection = { margin: pregame.projection.home_margin, winProbability: pregame.projection.home_win_probability };
  const market = pregame.market;
  const injuryReport = pregame.injury_report;
  const projectedMinutes = pregame?.projected_minutes ?? archived?.projected_minutes;
  const spreadLedger = pregame.projection.ledger;
  const displayedMinutes = projectedMinutes;
  const pendingReportTeams = [game.away_team, game.home_team].filter((team) => (
    injuryReport?.team_reports?.[team]?.status !== undefined
    && injuryReport.team_reports[team].status !== "submitted"
  ));
  const retrospective = archived?.retrospective;
  const marketHomeMargin = market?.home_spread == null ? null : -market.home_spread;
  const modelEdge = marketHomeMargin === null ? null : projection.margin - marketHomeMargin;
  const edgeTeam = modelEdge === null ? null : modelEdge >= 0 ? game.home_team : game.away_team;
  const gameState = archived?.status.state ?? "pre";
  const isPregame = gameState === "pre";
  const statusLabel = liveStatus(archived);
  const finalAudit = archived?.final_audit;
  const atsResult = finalAudit?.grading.market?.model_ats_result;
  const nbaGameUrl = "https://www.nba.com/game/"
    + game.away_team.toLowerCase()
    + "-vs-"
    + game.home_team.toLowerCase()
    + "-"
    + game.game_id;

  return (
    <article className="schedule-game-card">
      <div className="schedule-game-context">
        <a
          className="schedule-game-time schedule-nba-game-link"
          href={nbaGameUrl}
          target="_blank"
          rel="noreferrer"
          title="Open game on NBA.com"
          aria-label={"Open " + game.away_team + " at " + game.home_team + " on NBA.com"}
        >
          <span>{gameTime(archived?.start_time)}</span>
          <span className="schedule-nba-game-link-label" aria-hidden="true">NBA ↗</span>
        </a>
        <span>{(archived?.broadcasts ?? []).join(" · ") || "Broadcast unavailable"}</span>
        <span className="schedule-game-status"><Radio size={11} aria-hidden="true" /> {statusLabel}</span>
        {archived?.status.completed && finalAudit && <div className="schedule-result-badges" aria-label="Saved model results">
          <span className={finalAudit.grading.model_favorite_won ? "win" : "loss"} title={finalAudit.recorded_outcome ? `Saved result: ${game.away_team} ${finalAudit.recorded_outcome.away_score}, ${game.home_team} ${finalAudit.recorded_outcome.home_score}` : "Saved model winner result"}>
            Winner {finalAudit.grading.model_favorite_won ? "✓" : "✕"}
          </span>
          <span className={atsResult ?? "unavailable"} title="Saved ATS result against the report’s pinned final score and pregame spread">
            {atsResult === "win" ? "ATS ✓" : atsResult === "loss" ? "ATS ✕" : atsResult === "push" ? "ATS push" : "ATS n/a"}
          </span>
        </div>}
      </div>

      <div className="schedule-matchup">
        <a className="schedule-team away" href={teamHref(game.away_team)} aria-label={`Open ${game.away_team} team page`}>
          <img alt="" src={teamLogoUrl(game.away_team)} />
          <div><strong>{game.away_team}</strong><span>{isPregame ? "Away" : archived?.away_score ?? "—"}</span></div>
        </a>
        <span className="schedule-at">at</span>
        <a className="schedule-team home" href={teamHref(game.home_team)} aria-label={`Open ${game.home_team} team page`}>
          <img alt="" src={teamLogoUrl(game.home_team)} />
          <div><strong>{game.home_team}</strong><span>{isPregame ? "Home" : archived?.home_score ?? "—"}</span></div>
        </a>
      </div>

      <div className="schedule-projection-panel">
        <p>GESTALT projection{game.prediction_purpose === "historical_replay" ? " · Historical replay" : ""}{pregame?.projection.model_track === "updated_ratings_updated_minutes" || pregame?.projection.state_kind === "in_season_updated" ? " · Updated" : ""}</p>
        <div className="schedule-probability-bar" aria-label={`${game.home_team} ${(projection.winProbability * 100).toFixed(0)} percent win probability`}>
          <span style={{ width: `${(1 - projection.winProbability) * 100}%` }} />
          <span style={{ width: `${projection.winProbability * 100}%` }} />
        </div>
        <div className="schedule-probability-labels">
          <strong>{game.away_team} {((1 - projection.winProbability) * 100).toFixed(0)}%</strong>
          <strong>{game.home_team} {(projection.winProbability * 100).toFixed(0)}%</strong>
        </div>
        <div className="schedule-model-line">
          <span>Model spread</span>
          <strong>{game.home_team} {formatSpread(-projection.margin)}</strong>
        </div>
      </div>

      <ScheduleMarketPanel game={game} />

      <div className={`schedule-edge ${modelEdge === null || Math.abs(modelEdge) < 1 ? "muted" : ""}`}>
        <TrendingUp size={15} aria-hidden="true" />
        <span>{modelEdge === null ? "Market unavailable" : `${edgeTeam} edge`}</span>
        <strong>{modelEdge === null ? "—" : `${Math.abs(modelEdge).toFixed(1)} pts`}</strong>
      </div>

      {archived?.data_quality_errors?.length ? <p className="error schedule-error" role="alert">Data quality: {archived.data_quality_errors.join("; ")}</p> : null}
      {game.detail_errors && <p className="schedule-report-warning" role="status">{Object.entries(game.detail_errors).map(([key, value]) => `${key === "market" ? "Market" : key === "grading" ? "Results" : "Postgame"}: ${value}`).join(" ")}</p>}
      {game.status.state === "post" && !finalAudit && <p className="schedule-market-empty">Saved result grades unavailable</p>}
      {game.status.state === "post" && !retrospective && <p className="schedule-market-empty">Postgame detail unavailable</p>}
      {injuryReport && displayedMinutes && <details className="schedule-assumptions">
        <summary>
          <span>{spreadLedger ? "Rotation, ratings & spread ledger" : "Rotation & injury assumptions"}</span>
          <small>{pendingReportTeams.length ? `${pendingReportTeams.join(" & ")} pending · saved prediction input` : `saved prediction input`} · NBA report {reportTime(injuryReport.published_at)}{spreadLedger ? " · ledger available" : ""}</small>
        </summary>
        <div className="schedule-assumptions-body">
          {pendingReportTeams.length > 0 && <p className="schedule-report-warning"><strong>{pendingReportTeams.join(" & ")} report coverage is incomplete.</strong> Statuses and projected minutes are shown exactly as saved with this prediction.</p>}
          {!spreadLedger && <p className="schedule-attribution-unavailable">Saved rating attribution is unavailable.</p>}
          {spreadLedger && <SpreadLedgerView ledger={spreadLedger} actualLedger={retrospective?.ledger} postgame={retrospective} homeWinProbability={pregame.projection.home_win_probability} />}
          <div className="schedule-rotations">
            <RotationTable team={game.away_team} players={displayedMinutes.away} teamLedger={spreadLedger?.away} actualTeamLedger={retrospective?.ledger?.away} actualPlayers={retrospective?.actual_minutes.away} reportStatus={injuryReport.team_reports?.[game.away_team]?.status} />
            <RotationTable team={game.home_team} players={displayedMinutes.home} teamLedger={spreadLedger?.home} actualTeamLedger={retrospective?.ledger?.home} actualPlayers={retrospective?.actual_minutes.home} reportStatus={injuryReport.team_reports?.[game.home_team]?.status} />
          </div>
          {injuryReport.source_url && <a href={injuryReport.source_url} target="_blank" rel="noreferrer">Open official NBA injury report</a>}
        </div>
      </details>}
    </article>
  );
}

export function SchedulePage({ initialDate }: { initialDate?: string }) {
  const [calendar, setCalendar] = useState<Record<string, CalendarCount>>({});
  const { report: marketReport, error: marketError } = useMarketPerformanceReport();
  const [teams, setTeams] = useState<string[]>([]);
  const [scheduleDay, setScheduleDay] = useState<ScheduleDay | null>(null);
  const [selectedDate, setSelectedDate] = useState(
    isDateKey(initialDate) ? initialDate : dateKey(new Date()),
  );
  const [teamFilter, setTeamFilter] = useState("all");
  const [error, setError] = useState<string | null>(null);

  function selectDate(nextDate: string) {
    if (!isDateKey(nextDate)) return;
    setSelectedDate(nextDate);
    const nextHash = scheduleHref(nextDate);
    if (window.location.hash !== nextHash) window.location.hash = nextHash;
  }

  useEffect(() => {
    const nextDate = isDateKey(initialDate) ? initialDate : dateKey(new Date());
    setSelectedDate((currentDate) => currentDate === nextDate ? currentDate : nextDate);
  }, [initialDate]);

  const dateRail = useMemo(() => Array.from({ length: 7 }, (_, index) => addDays(selectedDate, index - 3)), [selectedDate]);
  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    setScheduleDay(null);
    setCalendar({});
    setError(null);
    async function refresh() {
      try {
        const response = await fetch(`/api/schedule-view?date=${selectedDate}`, { signal: controller.signal });
        const payload = await response.json() as {
          day: ScheduleDay;
          calendar: Record<string, CalendarCount>;
          teams: { team: string; team_id: number }[];
        };
        if (!response.ok || !payload.day.available) {
          throw new Error("Saved Schedule data is unavailable for this date.");
        }
        if (controller.signal.aborted) return;
        setScheduleDay(payload.day);
        setCalendar(payload.calendar);
        setTeams(payload.teams.map((team) => team.team).sort());
        setError(null);
        if (payload.day.games.some((game) => game.status.state !== "post")) {
          timer = window.setTimeout(refresh, 60_000);
        }
      } catch (reason) {
        if (!controller.signal.aborted) {
          setError((reason as Error).message);
          timer = window.setTimeout(refresh, 60_000);
        }
      }
    }
    void refresh();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [selectedDate]);
  const gamesByDate = useMemo(() => new Map(Object.entries(calendar).map(([day, value]) => [day, value.game_count])), [calendar]);
  const games = useMemo(() => (scheduleDay?.games ?? []).filter((game) => (
    teamFilter === "all" || game.home_team === teamFilter || game.away_team === teamFilter
  )).sort((left, right) => Date.parse(left.start_time) - Date.parse(right.start_time) || left.game_id.localeCompare(right.game_id)), [scheduleDay, teamFilter]);

  return (
    <section className="schedule-page">
      <div className="schedule-hero-row">
        <div className="schedule-hero">
          <p className="eyebrow">2026–27 season</p>
          <h1>The slate.</h1>
          <p>Every game, our pregame projection, and the market on one timeline. Move backward to audit results or forward to scout the week.</p>
        </div>
        <MarketPerformanceView report={marketReport} error={marketError} />
      </div>

      <div className="schedule-toolbar">
        <button type="button" onClick={() => selectDate(addDays(selectedDate, -7))} aria-label="Previous week"><ChevronLeft size={17} /></button>
        <label className="schedule-date-picker">
          <CalendarDays size={15} aria-hidden="true" />
          <input type="date" value={selectedDate} onChange={(event) => selectDate(event.target.value)} />
        </label>
        <button type="button" onClick={() => selectDate(addDays(selectedDate, 7))} aria-label="Next week"><ChevronRight size={17} /></button>
        <button className="schedule-today" type="button" onClick={() => selectDate(dateKey(new Date()))}>Today</button>
        <label className="schedule-team-filter">
          <span>Team</span>
          <select value={teamFilter} onChange={(event) => setTeamFilter(event.target.value)}>
            <option value="all">All teams</option>
            {teams.map((team) => <option key={team} value={team}>{team}</option>)}
          </select>
        </label>
      </div>

      <div className="schedule-date-rail" aria-label="Choose a date">
        {dateRail.map((date) => {
          const active = date === selectedDate;
          const today = date === dateKey(new Date());
          return <button className={active ? "active" : ""} type="button" key={date} onClick={() => selectDate(date)} aria-current={active ? "date" : undefined}>
            <span>{today ? "Today" : shortDay(date)}</span>
            <strong>{parseDate(date).getDate()}</strong>
            <small>{shortMonth(date)} · {gamesByDate.has(date) ? gameCountLabel(gamesByDate.get(date)!) : "—"}</small>
            <CalendarResult day={calendarSummary(calendar[date], marketReport, marketError)} />
          </button>;
        })}
      </div>

      <div className="schedule-day-heading">
        <div>
          <p className="eyebrow">Daily board</p>
          <h2>{longDate(selectedDate)}</h2>
        </div>
        <div className="schedule-source-note"><span>Schedule &amp; scores</span><strong>NBA official</strong><small>{archiveAge(scheduleDay?.live_score_collected_at ?? scheduleDay?.collected_at)}</small></div>
      </div>

      {scheduleDay && !error && (scheduleDay.coverage.missing_prediction_game_ids.length > 0 || scheduleDay.coverage.missing_observation_game_ids.length > 0) && <p className="schedule-report-warning" role="status">Some saved forecasts or scores are unavailable. All scheduled games remain visible.</p>}
      {error ? <p className="error schedule-error"><CircleAlert size={16} /> {error}</p> : !scheduleDay ? <p className="schedule-loading">Loading the schedule…</p> : games.length ? <div className="schedule-games">
        {games.map((game) => <ScheduleGameCard
          game={game}
          key={game.game_id}
        />)}
      </div> : <div className="schedule-empty">
        <CalendarDays size={28} aria-hidden="true" />
        <h3>{scheduleDay.games.length ? "No games for this team." : "No games on this date."}</h3>
        <p>Choose another date or team to view the schedule.</p>
      </div>}
    </section>
  );
}
