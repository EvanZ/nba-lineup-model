import { useEffect, useMemo, useState } from "react";
import { CalendarDays, ChevronLeft, ChevronRight, CircleAlert, Radio, TrendingUp } from "lucide-react";

import type { MinutesProjectionPayload, WinProjectionPayload, WinProjectionScheduledGame } from "./types";

type MarketLine = {
  provider: string;
  home_spread: number | null;
  away_spread: number | null;
  total: number | null;
  away_moneyline: number | null;
  home_moneyline: number | null;
};

type PredictionMarket = {
  provider: string;
  event_id: string;
  market_id: string;
  url: string;
  captured_at: string;
  pricing_method: string;
  away_probability: number;
  home_probability: number;
  away_probability_raw: number;
  home_probability_raw: number;
  raw_probability_sum: number;
  volume: number | null;
  liquidity: number | null;
  bid_ask_spread: number | null;
};

type InjuryAssumption = {
  player_id: number;
  player_name: string;
  team: string;
  status: string;
  reason: string | null;
  play_probability: number;
  projected_minutes: number;
};

type ProjectedMinute = {
  player_id: number;
  player_name: string;
  status: string;
  play_probability: number | null;
  conditional_minutes: number;
  projected_minutes: number;
  reason: string | null;
};

type ActualMinute = {
  player_id: number | null;
  player_name: string;
  actual_minutes: number;
  rating_matched: boolean;
};

type FinalAudit = {
  grading: {
    model_favorite_won: boolean;
    market: { model_ats_result: "win" | "loss" | "push" } | null;
  };
};

type ArchivedGame = {
  game_id: string | null;
  espn_game_id: string;
  start_time: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  broadcasts: string[];
  status: { state: string; completed: boolean; description: string; detail: string | null; display_clock?: string | null; period?: number | null };
  projection: { model: string; home_margin: number; home_win_probability: number };
  market: MarketLine | null;
  prediction_market?: PredictionMarket | null;
  injury_report: {
    source: string;
    source_url: string;
    published_at: string;
    status_probabilities: Record<string, number>;
    team_reports?: Record<string, { status: string }>;
    assumptions: InjuryAssumption[];
  };
  projected_minutes: { away: ProjectedMinute[]; home: ProjectedMinute[] };
  retrospective?: {
    computed_at: string;
    method: string;
    home_margin: number;
    home_win_probability: number;
    actual_minutes: { away: ActualMinute[]; home: ActualMinute[] };
  };
  final_audit?: FinalAudit;
  pregame?: ArchivedGame;
};

type ScheduleDay = {
  available: boolean;
  game_date: string;
  collected_at?: string;
  live_score_collected_at?: string;
  source: string;
  games: ArchivedGame[];
};

type DailyRecord = {
  wins: number;
  losses: number;
  atsWins: number;
  atsLosses: number;
  atsPushes: number;
};

const TEAM_LOGO_SLUGS: Record<string, string> = { NOP: "no", UTA: "utah" };

function teamLogoUrl(team: string) {
  return `https://a.espncdn.com/i/teamlogos/nba/500/${TEAM_LOGO_SLUGS[team] ?? team.toLowerCase()}.png`;
}

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

function sigmoid(value: number) {
  return 1 / (1 + Math.exp(-value));
}

function homeProjection(game: WinProjectionScheduledGame, forecast: WinProjectionPayload) {
  const strength = new Map(forecast.teams.map((team) => [team.team, team.team_strength]));
  const margin = (strength.get(game.home_team) ?? 0) - (strength.get(game.away_team) ?? 0)
    + forecast.home_court
    + forecast.back_to_back * (game.home_back_to_back - game.away_back_to_back);
  return {
    margin,
    winProbability: sigmoid(forecast.win_probability_scale * margin),
  };
}

function formatSpread(value: number) {
  if (Math.abs(value) < 0.05) return "PK";
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}`;
}

function formatMoneyline(value: number | null) {
  if (value === null) return "—";
  return `${value > 0 ? "+" : ""}${value}`;
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

function archiveAge(value?: string) {
  if (!value) return "Awaiting first archive";
  return `Updated ${new Intl.DateTimeFormat("en-US", { hour: "numeric", minute: "2-digit", timeZoneName: "short" }).format(new Date(value))}`;
}

function dailyRecord(scheduleDay: ScheduleDay | null | undefined): DailyRecord | null {
  const audits = (scheduleDay?.games ?? []).flatMap((game) => game.final_audit ? [game.final_audit] : []);
  if (!audits.length) return null;
  const record: DailyRecord = { wins: 0, losses: 0, atsWins: 0, atsLosses: 0, atsPushes: 0 };
  audits.forEach((audit) => {
    if (audit.grading.model_favorite_won) record.wins += 1;
    else record.losses += 1;
    const atsResult = audit.grading.market?.model_ats_result;
    if (atsResult === "win") record.atsWins += 1;
    if (atsResult === "loss") record.atsLosses += 1;
    if (atsResult === "push") record.atsPushes += 1;
  });
  return record;
}

function dailyRecordLabel(record: DailyRecord) {
  const atsGames = record.atsWins + record.atsLosses + record.atsPushes;
  const ats = atsGames
    ? `${record.atsWins}-${record.atsLosses}${record.atsPushes ? `-${record.atsPushes}` : ""}`
    : "—";
  return `W-L ${record.wins}-${record.losses} · ATS ${ats}`;
}

function reportTime(value?: string) {
  if (!value) return "Report time unavailable";
  return new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/New_York",
    timeZoneName: "short",
  }).format(new Date(value));
}

function liveStatus(game: ArchivedGame | undefined) {
  if (!game) return "Pregame";
  if (game.status.completed) return "Final";
  if (game.status.state !== "in") return game.status.description || "Pregame";
  if (game.status.description.toLowerCase().includes("halftime")) return "Halftime";
  const period = game.status.period ?? 0;
  const periodText = period <= 4 ? `Q${period}` : period === 5 ? "OT" : `${period - 4}OT`;
  return `${periodText} · ${game.status.display_clock ?? ""}`.replace(/ · $/, "");
}

function RotationTable({ team, players, actualPlayers, reportStatus }: { team: string; players: ProjectedMinute[]; actualPlayers?: ActualMinute[]; reportStatus?: string }) {
  const reportPending = reportStatus !== undefined && reportStatus !== "submitted";
  const actualByPlayer = new Map((actualPlayers ?? []).map((player) => [player.player_id ?? player.player_name, player]));
  const predictedIds = new Set(players.map((player) => player.player_id));
  const rows = [
    ...players,
    ...(actualPlayers ?? []).filter((player) => player.player_id === null || !predictedIds.has(player.player_id)).map((player) => ({
      player_id: player.player_id ?? -Math.abs(player.player_name.split("").reduce((total, character) => total + character.charCodeAt(0), 0)),
      player_name: player.player_name,
      status: "Available",
      play_probability: 1,
      conditional_minutes: 0,
      projected_minutes: 0,
      reason: null,
    })),
  ];
  return <div className="schedule-rotation-team">
    <h4><a href={teamHref(team)}>{team}</a> projected minutes</h4>
    {reportPending && <p className="schedule-report-pending">Availability unknown — team report pending. Minutes remain an unadjusted baseline.</p>}
    <div className={`schedule-rotation-table ${actualPlayers ? "has-actual" : ""}`}>
      {actualPlayers && <div className="schedule-rotation-header"><span>Player</span><small>Pred.</small><small>Actual</small><small>Δ</small></div>}
      {rows.map((player) => {
        const actual = actualByPlayer.get(player.player_id) ?? actualByPlayer.get(player.player_name);
        const difference = actual ? actual.actual_minutes - player.projected_minutes : null;
        const availability = reportPending
          ? "Status unknown — team report pending"
          : `${player.status}${player.reason ? ` · ${player.reason}` : ""}`;
        return <div key={player.player_id} className={player.status !== "Available" ? "injury-listed" : ""}>
        <span className="schedule-rotation-player">{player.player_id > 0 ? <a className="schedule-player-link" href={playerHref(player.player_id)}>{player.player_name}</a> : player.player_name}<small>{availability}</small></span>
        <strong>{player.projected_minutes.toFixed(1)}</strong>
        {actualPlayers && <><strong>{actual?.actual_minutes.toFixed(1) ?? "0.0"}</strong><strong className={difference !== null && difference < 0 ? "negative" : "positive"}>{difference === null ? "—" : `${difference > 0 ? "+" : ""}${difference.toFixed(1)}`}</strong></>}
      </div>;})}
    </div>
  </div>;
}

function gameCountLabel(count: number) {
  return `${count} ${count === 1 ? "game" : "games"}`;
}

function ScheduleGameCard({ game, forecast, archived }: { game: WinProjectionScheduledGame; forecast: WinProjectionPayload; archived?: ArchivedGame }) {
  const liveProjection = homeProjection(game, forecast);
  const pregame = archived?.pregame ?? archived;
  const projection = pregame?.projection
    ? { margin: pregame.projection.home_margin, winProbability: pregame.projection.home_win_probability }
    : liveProjection;
  const market = pregame?.market ?? archived?.market ?? null;
  const predictionMarket = pregame?.prediction_market ?? archived?.prediction_market ?? null;
  const injuryReport = pregame?.injury_report ?? archived?.injury_report;
  const projectedMinutes = pregame?.projected_minutes ?? archived?.projected_minutes;
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

  return (
    <article className="schedule-game-card">
      <div className="schedule-game-context">
        <span className="schedule-game-time">{gameTime(archived?.start_time)}</span>
        <span>{archived?.broadcasts.join(" · ") || "League Pass"}</span>
        <span className="schedule-game-status"><Radio size={11} aria-hidden="true" /> {statusLabel}</span>
        {archived?.status.completed && finalAudit && <div className="schedule-result-badges" aria-label="Model results">
          <span className={finalAudit.grading.model_favorite_won ? "win" : "loss"} title="Whether the model selected the winning team">
            Winner {finalAudit.grading.model_favorite_won ? "✓" : "✕"}
          </span>
          <span className={atsResult ?? "unavailable"} title="Whether the model's pregame selection covered the archived spread">
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
        <p>GESTALT projection</p>
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

      <div className="schedule-market-panel">
        <div className="schedule-market-heading">
          <p>{market?.provider ?? "Market line"}</p>
          <span>{pregame ? "Pregame" : "Live"}</span>
        </div>
        {market ? <>
          <div className="schedule-market-teams">
            <div><span>{game.away_team}</span><strong>{market.away_spread == null ? "—" : formatSpread(market.away_spread)}</strong><small>{formatMoneyline(market.away_moneyline)}</small></div>
            <div><span>{game.home_team}</span><strong>{market.home_spread == null ? "—" : formatSpread(market.home_spread)}</strong><small>{formatMoneyline(market.home_moneyline)}</small></div>
          </div>
        </> : <p className="schedule-market-empty">No market line yet</p>}
        <div className="schedule-polymarket">
          <div>
            {predictionMarket ? <a href={predictionMarket.url} target="_blank" rel="noreferrer">Polymarket</a> : <span>Polymarket</span>}
            <small title={predictionMarket?.pricing_method}>{predictionMarket ? "Midpoint" : "Unavailable"}</small>
          </div>
          {predictionMarket && <div className="schedule-polymarket-prices" aria-label={`Polymarket probabilities: ${game.away_team} ${(predictionMarket.away_probability * 100).toFixed(1)} percent, ${game.home_team} ${(predictionMarket.home_probability * 100).toFixed(1)} percent`}>
            <strong>{game.away_team} {(predictionMarket.away_probability * 100).toFixed(1)}%</strong>
            <strong>{game.home_team} {(predictionMarket.home_probability * 100).toFixed(1)}%</strong>
          </div>}
        </div>
      </div>

      <div className={`schedule-edge ${modelEdge === null || Math.abs(modelEdge) < 1 ? "muted" : ""}`}>
        <TrendingUp size={15} aria-hidden="true" />
        <span>{modelEdge === null ? "Awaiting line" : `${edgeTeam} edge`}</span>
        <strong>{modelEdge === null ? "—" : `${Math.abs(modelEdge).toFixed(1)} pts`}</strong>
      </div>

      {injuryReport && projectedMinutes && <details className="schedule-assumptions">
        <summary>
          <span>Rotation &amp; injury assumptions</span>
          <small>{pendingReportTeams.length ? `${pendingReportTeams.join(" & ")} pending` : `${injuryReport.assumptions.length} listed`} · NBA report {reportTime(injuryReport.published_at)}</small>
        </summary>
        <div className="schedule-assumptions-body">
          {pendingReportTeams.length > 0 && <p className="schedule-report-warning"><strong>{pendingReportTeams.join(" & ")} availability is unknown.</strong> The official team report has not been submitted or does not yet cover this game. Minutes remain unadjusted baseline estimates.</p>}
          <p className="schedule-assumption-policy">Submitted reports: Out 0% · Doubtful 25% · Questionable 50% · Probable 75% · Unlisted 100%. Expected minutes are renormalized to 240 per team.</p>
          {retrospective && <div className="schedule-retrospective">
            <span>Retrospective actual-minute mix</span>
            <strong>{game.home_team} {formatSpread(-retrospective.home_margin)}</strong>
            <small>{game.home_team} {(retrospective.home_win_probability * 100).toFixed(0)}% · uses final minutes, not a pregame prediction</small>
          </div>}
          <div className="schedule-rotations">
            <RotationTable team={game.away_team} players={projectedMinutes.away} actualPlayers={retrospective?.actual_minutes.away} reportStatus={injuryReport.team_reports?.[game.away_team]?.status} />
            <RotationTable team={game.home_team} players={projectedMinutes.home} actualPlayers={retrospective?.actual_minutes.home} reportStatus={injuryReport.team_reports?.[game.home_team]?.status} />
          </div>
          <a href={injuryReport.source_url} target="_blank" rel="noreferrer">Open official NBA injury report</a>
        </div>
      </details>}
    </article>
  );
}

export function SchedulePage({ initialDate }: { initialDate?: string }) {
  const [forecast, setForecast] = useState<WinProjectionPayload | null>(null);
  const [scheduleDay, setScheduleDay] = useState<ScheduleDay | null>(null);
  const [dateRecords, setDateRecords] = useState<Record<string, DailyRecord>>({});
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

  useEffect(() => {
    let cancelled = false;
    fetch("/api/win-projections")
      .then(async (response) => {
        const payload = await response.json() as MinutesProjectionPayload;
        if (!response.ok || !payload.win_projection) throw new Error("Schedule projections are unavailable.");
        if (!cancelled) setForecast(payload.win_projection);
      })
      .catch((reason: Error) => { if (!cancelled) setError(reason.message); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!forecast || initialDate) return;
    const dates = forecast.scheduled_games.map((game) => game.game_date).sort();
    const today = dateKey(new Date());
    if (dates.includes(today) || !dates.length) return;
    const fallbackDate = dates.find((date) => date > today) ?? dates[dates.length - 1];
    setSelectedDate(fallbackDate);
    window.history.replaceState(null, "", scheduleHref(fallbackDate));
  }, [forecast, initialDate]);

  useEffect(() => {
    let cancelled = false;
    setScheduleDay(null);
    fetch(`/api/schedule?date=${selectedDate}`)
      .then(async (response) => {
        const payload = await response.json() as ScheduleDay;
        if (!response.ok) throw new Error("Archived lines are temporarily unavailable.");
        if (!cancelled) setScheduleDay(payload);
      })
      .catch((reason: Error) => { if (!cancelled) setError(reason.message); });
    return () => { cancelled = true; };
  }, [selectedDate]);

  useEffect(() => {
    if (!scheduleDay?.games.some((game) => game.status.state === "in")) return;
    let cancelled = false;
    const refresh = () => {
      fetch(`/api/schedule?date=${selectedDate}`)
        .then(async (response) => {
          const payload = await response.json() as ScheduleDay;
          if (response.ok && !cancelled) setScheduleDay(payload);
        })
        .catch(() => undefined);
    };
    const timer = window.setInterval(refresh, 30_000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [scheduleDay, selectedDate]);

  const dateRail = useMemo(() => Array.from({ length: 7 }, (_, index) => addDays(selectedDate, index - 3)), [selectedDate]);
  useEffect(() => {
    let cancelled = false;
    Promise.all(dateRail.map(async (date) => {
      try {
        const response = await fetch(`/api/schedule?date=${date}`);
        if (!response.ok) return null;
        const payload = await response.json() as ScheduleDay;
        return [date, dailyRecord(payload)] as const;
      } catch {
        return null;
      }
    })).then((rows) => {
      if (cancelled) return;
      setDateRecords((current) => {
        const next = { ...current };
        rows.forEach((row) => {
          if (!row) return;
          const [date, record] = row;
          if (record) next[date] = record;
          else delete next[date];
        });
        return next;
      });
    });
    return () => { cancelled = true; };
  }, [dateRail]);
  const gamesByDate = useMemo(() => {
    const counts = new Map<string, number>();
    forecast?.scheduled_games.forEach((game) => counts.set(game.game_date, (counts.get(game.game_date) ?? 0) + 1));
    return counts;
  }, [forecast]);
  const archivedGames = useMemo(() => new Map((scheduleDay?.games ?? []).map((game) => [
    game.game_id ?? `${game.away_team}-${game.home_team}`,
    game,
  ])), [scheduleDay]);
  const games = useMemo(() => {
    const archivedGame = (game: WinProjectionScheduledGame) => (
      archivedGames.get(game.game_id) ?? archivedGames.get(`${game.away_team}-${game.home_team}`)
    );
    return (forecast?.scheduled_games ?? []).filter((game) => (
      game.game_date === selectedDate
      && (teamFilter === "all" || game.home_team === teamFilter || game.away_team === teamFilter)
    )).sort((left, right) => {
      const leftTime = Date.parse(archivedGame(left)?.start_time ?? "") || Number.POSITIVE_INFINITY;
      const rightTime = Date.parse(archivedGame(right)?.start_time ?? "") || Number.POSITIVE_INFINITY;
      if (leftTime !== rightTime) return leftTime - rightTime;
      return left.away_team.localeCompare(right.away_team) || left.home_team.localeCompare(right.home_team);
    });
  }, [archivedGames, forecast, selectedDate, teamFilter]);
  const teams = forecast?.teams.map((team) => team.team).sort() ?? [];

  return (
    <section className="schedule-page">
      <div className="schedule-hero">
        <p className="eyebrow">2026–27 season</p>
        <h1>The slate.</h1>
        <p>Every game, our pregame projection, and the market on one timeline. Move backward to audit results or forward to scout the week.</p>
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
          const record = active ? dailyRecord(scheduleDay) ?? dateRecords[date] : dateRecords[date];
          return <button className={active ? "active" : ""} type="button" key={date} onClick={() => selectDate(date)} aria-current={active ? "date" : undefined}>
            <span>{today ? "Today" : shortDay(date)}</span>
            <strong>{parseDate(date).getDate()}</strong>
            <small>{shortMonth(date)} · {gameCountLabel(gamesByDate.get(date) ?? 0)}</small>
            {record && <small className="schedule-date-record">{dailyRecordLabel(record)}</small>}
          </button>;
        })}
      </div>

      <div className="schedule-day-heading">
        <div>
          <p className="eyebrow">Daily board</p>
          <h2>{longDate(selectedDate)}</h2>
        </div>
        <div className="schedule-source-note"><span>Lines, scores &amp; crowd</span><strong>ESPN · DraftKings · Polymarket</strong><small>{archiveAge(scheduleDay?.live_score_collected_at ?? scheduleDay?.collected_at)}</small></div>
      </div>

      {error ? <p className="error schedule-error"><CircleAlert size={16} /> {error}</p> : !forecast ? <p className="schedule-loading">Loading the schedule…</p> : games.length ? <div className="schedule-games">
        {games.map((game) => <ScheduleGameCard
          game={game}
          forecast={forecast}
          archived={archivedGames.get(game.game_id) ?? archivedGames.get(`${game.away_team}-${game.home_team}`)}
          key={game.game_id}
        />)}
      </div> : <div className="schedule-empty">
        <CalendarDays size={28} aria-hidden="true" />
        <h3>No games on this date.</h3>
        <p>Past dates will keep the final score, the frozen pregame projection, and the closing line together for review.</p>
      </div>}
    </section>
  );
}
