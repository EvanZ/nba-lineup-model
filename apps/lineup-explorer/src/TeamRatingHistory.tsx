import { useEffect, useState } from "react";
import { releaseFetch as fetch } from "./releaseFetch";
import { teamLogoUrl } from "./teamLogos";

type TeamRatingPoint = {
  game_id: string;
  game_date_pacific: string;
  opponent: string;
  home_away: "home" | "away";
  nail: number | null;
  availability: "available" | "missing";
  projection_state: "frozen_pregame" | "historical_replay" | "upcoming" | null;
  ratings_observed_through_game_date_pacific: string | null;
};
export type TeamRatingHistoryData = {
  kind: "saved_team_rating_history";
  team: string;
  season: string;
  complete: boolean;
  points: TeamRatingPoint[];
  coverage: { missing_prediction_game_ids: string[] };
};
const rating = new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1, signDisplay: "exceptZero" });
const shortDate = (date: string) => `${date.slice(5, 7)}/${date.slice(8, 10)}`;
const stateLabel = (point: TeamRatingPoint) => point.projection_state === "historical_replay"
  ? "Historical replay" : point.projection_state === "upcoming" ? "Upcoming game · projection may update" : "Frozen pregame projection";

export function TeamRatingHistoryView({ history }: { history: TeamRatingHistoryData }) {
  const available = history.points.filter(point => point.nail !== null);
  const width = Math.max(850, history.points.length * 48), height = 280;
  const values = available.map(point => point.nail!);
  const minimum = Math.floor((Math.min(...values) - .35) * 2) / 2;
  const maximum = Math.ceil((Math.max(...values) + .35) * 2) / 2;
  const x = (index: number) => 76 + index * (width - 108) / Math.max(1, history.points.length - 1);
  const y = (value: number) => 28 + (maximum - value) / Math.max(.5, maximum - minimum) * 168;
  const ticks = available.length ? Array.from({ length: Math.round((maximum - minimum) * 2) + 1 }, (_, i) => minimum + i / 2) : [];
  let connected = false;
  const path = history.points.map((point, index) => {
    if (point.nail === null) { connected = false; return ""; }
    const command = `${connected ? "L" : "M"}${x(index)},${y(point.nail)}`;
    connected = true;
    return command;
  }).join(" ");
  return <>
    <p className="player-rating-path-note">Expected lineup strength before each game, using that game’s player ratings and projected minutes.</p>
    {!history.complete && <p className="schedule-report-warning" role="status">{history.coverage.missing_prediction_game_ids.length} games have no saved prediction. Missing games are shown as gaps.</p>}
    {!history.points.length ? <p className="player-rating-path-note">No saved games are available for this season yet.</p>
      : <figure className="current-rating-trajectory team-rating-trajectory">
        <svg style={{ width: `max(100%, ${width}px)` }} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${history.team} ${history.season} pregame team NAIL history`}>
          <title>{history.team} · Game-by-game pregame team NAIL</title>
          <g aria-label="NAIL axis: major ticks every 1 unit, minor ticks every 0.5 units">
            <text className="current-rating-axis-label" x={48} y={16} textAnchor="end">NAIL</text>
            {ticks.map(tick => <g key={tick} data-nail-tick={tick} data-tick-kind={Number.isInteger(tick) ? "major" : "minor"}>
              <line className={Number.isInteger(tick) ? "current-rating-grid-major" : "current-rating-grid-minor"} x1={76} x2={width - 32} y1={y(tick)} y2={y(tick)} />
              {Number.isInteger(tick) && <text className="current-rating-axis-label" x={48} y={y(tick) + 3} textAnchor="end">{tick}</text>}
            </g>)}
          </g>
          <path className="current-rating-line" d={path} />
          {history.points.map((point, index) => {
            const label = `${point.game_date_pacific} · ${history.team} ${point.home_away === "home" ? "vs." : "@"} ${point.opponent}`;
            const description = point.nail === null ? `${label} · Saved prediction unavailable` : `${label} · NAIL ${rating.format(point.nail)} · ${stateLabel(point)} · ${point.ratings_observed_through_game_date_pacific ? `Ratings through ${point.ratings_observed_through_game_date_pacific}` : "Opening rating priors"}`;
            return <g key={point.game_id} data-game-id={point.game_id} data-nail={point.nail ?? "missing"} data-projection-state={point.projection_state ?? "missing"}>
              {point.nail === null ? <g className="team-rating-missing" role="img" aria-label={description}>
                <title>{description}</title><text className="current-rating-axis-label" x={x(index)} y={220} textAnchor="middle">×</text>
              </g> : <a href={`#schedule/${point.game_date_pacific}`} aria-label={description}>
                <title>{description}</title>
                <circle className={`current-rating-logo-halo team-rating-${point.projection_state}`} cx={x(index)} cy={y(point.nail)} r={10} />
                <image className="current-rating-opponent-logo" href={teamLogoUrl(point.opponent)} x={x(index) - 9} y={y(point.nail) - 9} width={18} height={18} />
                <text className="current-rating-value" x={x(index)} y={y(point.nail) + 21} textAnchor="middle">{rating.format(point.nail)}</text>
              </a>}
              {(index === 0 || index === history.points.length - 1 || index % 3 === 0) && <text className="current-rating-axis-label" x={x(index)} y={260} textAnchor="middle">{shortDate(point.game_date_pacific)}</text>}
            </g>;
          })}
        </svg>
        <figcaption>Opponent logos mark game days. Select a point to view the game.{history.points.some(point => point.projection_state === "historical_replay") ? " Dashed rings mark historical replays." : ""}{history.points.some(point => point.projection_state === "upcoming") ? " Dotted rings mark upcoming games." : ""}</figcaption>
      </figure>}
  </>;
}

export function TeamRatingHistory({ team, season }: { team: string; season: string }) {
  const [history, setHistory] = useState<TeamRatingHistoryData | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const response = await fetch(`/api/teams/${team}/rating-history?season=${encodeURIComponent(season)}`, { signal: controller.signal });
        if (!response.ok) throw new Error(response.status === 404 ? "No saved game-prediction history is published for this team and season." : "Team NAIL history is temporarily unavailable.");
        const data = await response.json() as TeamRatingHistoryData;
        if (data.kind !== "saved_team_rating_history" || data.team !== team || data.season !== season || !Array.isArray(data.points)) throw new Error("Team NAIL history is unavailable.");
        if (controller.signal.aborted) return;
        setHistory(data); setError(null);
        timer = setTimeout(load, 600000 - Date.now() % 600000 + 1000);
      } catch (cause) {
        if (controller.signal.aborted) return;
        setHistory(null); setError(cause instanceof Error ? cause.message : "Team NAIL history is temporarily unavailable.");
        timer = setTimeout(load, 60000);
      }
    }
    void load();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [team, season]);
  return <section className="player-rating-path team-rating-history" aria-labelledby="team-rating-history-title">
    <h3 id="team-rating-history-title">Game-by-game NAIL</h3>
    {error ? <p className="schedule-report-warning" role="status">{error}</p> : history ? <TeamRatingHistoryView history={history} /> : <p className="player-rating-path-note">Loading team NAIL history…</p>}
  </section>;
}
