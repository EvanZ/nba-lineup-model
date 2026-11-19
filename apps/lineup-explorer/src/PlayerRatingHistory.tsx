import { useEffect, useState } from "react";
import { releaseFetch as fetch } from "./releaseFetch";
import { teamLogoUrl } from "./teamLogos";

type Game = {
  game_id: string; team_id: number; team: string; opponent_team_id: number; opponent: string; home_away: string;
  participation: { status: "played" | "inactive" | "did_not_play" | "not_listed" | "unknown"; reason: string | null; box_score_row_present: boolean };
  actual_minutes: number; possessions: number; prediction_purpose: string;
  pregame: { nail: number; projected_minutes: number; injury_status: string | null; injury_reason: string | null } | null;
  team_score: number; opponent_score: number; result: string;
  stats: Record<string, number | null> | null;
};
type Point = {
  game_date_pacific: string; nail: number; cumulative_possessions: number;
  team_id: number | null; is_current: boolean; games: Game[];
};
export type RatingHistory = {
  kind: "saved_player_game_history"; player_id: number; player_name: string; season: string; complete: boolean;
  through_game_date_pacific: string;
  starting_point: { kind: "opening_prior" | "introduced_prior"; nail: number; initialized_on: string; first_state_date: string };
  current: { game_date_pacific: string; nail: number; cumulative_possessions: number; team_id: number | null };
  last_game: { game_date_pacific: string } | null;
  points: Point[];
};
const rating = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2, signDisplay: "exceptZero" });
const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const percent = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 });
const value = (n: number | null) => n === null ? "—" : number.format(n);
const shortDate = (day: string) => `${day.slice(5, 7)}/${day.slice(8, 10)}`;
const participationLabel = (g: Game) => ({ played: "Played", inactive: "Inactive", did_not_play: "DNP", not_listed: "Not listed", unknown: "Participation unknown" })[g.participation.status];
function tooltip(game: Game) {
  return [participationLabel(game), game.participation.reason,
    game.prediction_purpose === "historical_replay" ? "Historical replay" : "Frozen pregame projection",
    `Pregame NAIL: ${game.pregame ? rating.format(game.pregame.nail) : "unavailable"}`,
    `Pregame status: ${game.pregame?.injury_status ?? "unavailable"}`, game.pregame?.injury_reason].filter(Boolean).join(" · ");
}
const statColumns = [
  ["points", "PTS"], ["rebounds", "REB"], ["offensive_rebounds", "OREB"], ["defensive_rebounds", "DREB"],
  ["assists", "AST"], ["steals", "STL"], ["blocks", "BLK"], ["turnovers", "TOV"], ["personal_fouls", "PF"],
  ["fg", "FG M–A"], ["field_goal_percentage", "FG%"], ["three", "3P M–A"], ["three_point_percentage", "3P%"],
  ["ft", "FT M–A"], ["free_throw_percentage", "FT%"], ["plus_minus", "+/−"],
] as const;
function statValue(g: Game, key: string) {
  if (!g.stats) return "—";
  const shooting = ({ fg: ["field_goals_made", "field_goals_attempted"], three: ["three_pointers_made", "three_pointers_attempted"], ft: ["free_throws_made", "free_throws_attempted"] } as Record<string, string[]>)[key];
  if (shooting) return `${value(g.stats[shooting[0]])}–${value(g.stats[shooting[1]])}`;
  const n = g.stats[key];
  if (n === null) return "—";
  return key.endsWith("percentage") ? percent.format(n) : key === "plus_minus" ? new Intl.NumberFormat("en-US", { signDisplay: "exceptZero" }).format(n) : number.format(n);
}

export function RatingHistoryView({ history, summaryThrough }: { history: RatingHistory; summaryThrough: string }) {
  const prior = history.starting_point, points = history.points, current = history.current;
  const priorLabel = prior.kind === "opening_prior" ? "Opening prior" : "Introduced prior";
  const width = Math.max(850, (points.length + 1) * 48), height = 240;
  const values = [prior.nail, ...points.map((p) => p.nail)];
  const minimum = Math.floor((Math.min(...values) - .35) * 2) / 2;
  const maximum = Math.ceil((Math.max(...values) + .35) * 2) / 2;
  const x = (i: number) => 58 + i * (width - 82) / Math.max(1, points.length);
  const y = (n: number) => 28 + (maximum - n) / Math.max(.5, maximum - minimum) * 160;
  const ticks = Array.from({ length: Math.round((maximum - minimum) * 2) + 1 }, (_, i) => minimum + i / 2);
  const path = values.map((n, i) => `${i ? "L" : "M"}${x(i)},${y(n)}`).join(" ");
  return <>
    <p className="player-rating-path-note"><strong>{priorLabel}: {rating.format(prior.nail)}</strong> · Initialized {prior.initialized_on} · First saved state {prior.first_state_date}</p>
    <p className="player-rating-path-note" data-history-current-marker={current.game_date_pacific}><strong>Current saved NAIL {rating.format(current.nail)}</strong> · After slate {current.game_date_pacific}{history.last_game ? ` · Last game ${history.last_game.game_date_pacific}` : " · No saved games"}</p>
    <p className="player-rating-path-note">Ratings include the complete day’s slate. Changes reflect the joint daily model update, not an isolated game contribution.</p>
    {summaryThrough > history.through_game_date_pacific && <p className="schedule-report-warning" role="status">The current summary has newer evidence than this published history.</p>}
    {points.length > 0 && <figure className="current-rating-trajectory">
      <svg style={{ width: `max(100%, ${width}px)` }} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${history.player_name} ${history.season} game-day NAIL history`}>
        <title>{`${history.player_name} · NAIL after the complete daily slate`}</title>
        <g aria-label="NAIL axis: major ticks every 1 unit, minor ticks every 0.5 units">
          <text className="current-rating-axis-label" x={48} y={16} textAnchor="end">NAIL</text>
          {ticks.map((tick) => {
            const major = Number.isInteger(tick);
            return <g key={tick} data-nail-tick={tick} data-tick-kind={major ? "major" : "minor"}>
              <line className={major ? "current-rating-grid-major" : "current-rating-grid-minor"} x1={58} x2={width - 24} y1={y(tick)} y2={y(tick)} />
              <line className="current-rating-axis-tick" x1={major ? 51 : 54} x2={58} y1={y(tick)} y2={y(tick)} />
              {major && <text className="current-rating-axis-label" x={46} y={y(tick) + 3} textAnchor="end">{number.format(tick)}</text>}
            </g>;
          })}
          <line className="current-rating-axis-tick" x1={58} x2={58} y1={28} y2={188} />
        </g>
        <line className="current-rating-baseline" x1={58} x2={width - 24} y1={y(prior.nail)} y2={y(prior.nail)} />
        <path className="current-rating-line" d={path} />
        <g data-starting-prior={prior.kind}><title>{`${priorLabel}: ${rating.format(prior.nail)} · Initialized ${prior.initialized_on}`}</title>
          <circle className="current-rating-point" cx={x(0)} cy={y(prior.nail)} r={4} />
          <text className="current-rating-axis-label" x={x(0)} y={220} textAnchor="middle">Prior</text>
        </g>
        {points.map((p, i) => {
          const game = p.games.length === 1 ? p.games[0] : undefined;
          const skipped = game && ["inactive", "did_not_play", "not_listed"].includes(game.participation.status);
          const detail = p.games.map((g) => `${g.team} ${g.home_away === "home" ? "vs" : "at"} ${g.opponent} · ${tooltip(g)} · ${value(g.possessions)} game possessions`).join("; ");
          return <g key={p.game_date_pacific} data-history-date={p.game_date_pacific} data-history-current={p.is_current}>
            <title>{`${p.game_date_pacific}: ${rating.format(p.nail)} · ${detail}`}</title>
            <circle className={`current-rating-point${skipped ? " skipped" : ""}${p.is_current ? " latest" : ""}`} cx={x(i + 1)} cy={y(p.nail)} r={game ? 13 : p.is_current ? 5 : 3.5} />
            {game && <image href={teamLogoUrl(game.opponent)} x={x(i + 1) - 9} y={y(p.nail) - 9} width={18} height={18} />}
            {(i === 0 || i === points.length - 1 || i % 3 === 0) && <text className="current-rating-axis-label" x={x(i + 1)} y={220} textAnchor="middle">{shortDate(p.game_date_pacific)}</text>}
          </g>;
        })}
      </svg>
      <figcaption>Game days only, including inactive and DNP games. The current saved rating can change on an off day and is shown separately above.</figcaption>
    </figure>}
    <div className="current-season-game-log-table-wrap">
      <table className="current-season-game-log-table player-daily-history-table">
        <thead><tr><th>Date</th><th>Game / result</th><th>NAIL after slate</th><th>Projected min.</th><th>Actual min.</th><th>Game possessions</th>{statColumns.map(([key, label]) => <th key={key}>{label}</th>)}</tr></thead>
        <tbody>{[...points].reverse().flatMap((p) => p.games.map((g) => <tr key={`${p.game_date_pacific}-${g.game_id}`} className={["inactive", "did_not_play", "not_listed"].includes(g.participation.status) ? "current-season-game-log-dnp" : undefined} data-game-id={g.game_id} data-game-date={p.game_date_pacific}>
          <td className="current-season-game-log-date" data-label="Date"><time dateTime={p.game_date_pacific}>{shortDate(p.game_date_pacific)}</time></td>
          <td className="current-season-game-log-matchup" data-label="Game / result">
            <span className="current-season-game-log-opponent" title={tooltip(g)}>
              <img src={teamLogoUrl(g.opponent)} alt="" aria-hidden="true" loading="lazy" />
              <span><strong>{g.team} {g.home_away === "home" ? "vs." : "@"} {g.opponent}</strong><small className={g.result === "W" ? "win" : "loss"}>{g.result} {g.team_score}–{g.opponent_score}</small></span>
            </span>
            <small className="player-history-game-status" title={tooltip(g)}>{participationLabel(g)}{g.pregame?.injury_status && g.pregame.injury_status !== "Available" ? ` · ${g.pregame.injury_status}` : ""}</small>
          </td>
          <td className="current-season-game-log-rating" data-label="NAIL after slate"><span className={`rating-value ${p.nail < 0 ? "negative" : "positive"}`}>{rating.format(p.nail)}</span>{p.is_current && <small> Current</small>}</td>
          <td className="quantity-cell" data-label="Projected min." title={tooltip(g)}>{value(g.pregame?.projected_minutes ?? null)}</td>
          <td className="quantity-cell" data-label="Actual min.">{value(g.actual_minutes)}</td>
          <td className="quantity-cell" data-label="Game possessions">{value(g.possessions)}</td>
          {statColumns.map(([key, label]) => <td className="quantity-cell current-season-game-log-stat" data-label={label} key={key}>{statValue(g, key)}</td>)}
        </tr>))}</tbody>
      </table>
      {!points.length && <p className="player-rating-path-note">No saved game history is available for this player. The starting prior and current saved rating are shown above.</p>}
    </div>
  </>;
}

export function PlayerRatingHistory({ playerId, summaryThrough }: { playerId: number; summaryThrough: string }) {
  const [history, setHistory] = useState<RatingHistory | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    setHistory(null); setError(null);
    async function load() {
      try {
        const response = await fetch(`/api/players/${playerId}/rating-history`, { signal: controller.signal });
        if (!response.ok) throw new Error(response.status === 404 ? "No saved in-season game history is published for this player." : "Saved game history is temporarily unavailable.");
        const data = await response.json() as RatingHistory;
        if (data.kind !== "saved_player_game_history" || data.player_id !== playerId || !data.complete || !Array.isArray(data.points) || !data.current || !data.starting_point) throw new Error("Saved game history is incomplete or unavailable.");
        if (controller.signal.aborted) return;
        setHistory(data); setError(null);
        timer = setTimeout(load, 3600000 - Date.now() % 3600000 + 1000);
      } catch (cause) {
        if (controller.signal.aborted) return;
        setHistory(null); setError(cause instanceof Error ? cause.message : "Saved game history is temporarily unavailable.");
        timer = setTimeout(load, 60000);
      }
    }
    void load();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [playerId]);
  return <section className="player-rating-path"><h3>In-season NAIL history</h3>{error ? <p className="schedule-report-warning" role="status">{error}</p> : history ? <RatingHistoryView history={history} summaryThrough={summaryThrough} /> : <p className="player-rating-path-note">Loading saved game history…</p>}</section>;
}
