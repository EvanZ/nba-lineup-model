import { useEffect, useRef, useState } from "react";
import { releaseFetch as fetch } from "./releaseFetch";
import { Download } from "lucide-react";
import { downloadTeamHistoryPng } from "./teamHistoryPng";

type TeamRatingPoint = {
  game_id: string;
  game_date_pacific: string;
  opponent: string;
  opponent_team_id: number;
  home_away: "home" | "away";
  nail: number | null;
  opponent_nail: number | null;
  adjustments: { hca: number; b2b: number; team_is_back_to_back: boolean; opponent_is_back_to_back: boolean } | null;
  predicted_margin: number | null;
  availability: "available" | "missing";
  projection_state: "frozen_pregame" | "historical_replay" | "upcoming" | null;
  ratings_observed_through_game_date_pacific: string | null;
};
export type TeamRatingHistoryData = {
  kind: "saved_team_rating_history";
  team: string;
  team_id: number;
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
  const [adjusted, setAdjusted] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const chartRef = useRef<SVGSVGElement>(null);
  const teamValue = (point: TeamRatingPoint) => point.nail === null ? null : point.nail + (adjusted ? point.adjustments!.hca + point.adjustments!.b2b : 0);
  async function downloadPng() {
    if (!chartRef.current) return;
    setExporting(true); setExportError(null);
    try { await downloadTeamHistoryPng(chartRef.current, history.team, history.season, adjusted); }
    catch (cause) { setExportError(cause instanceof Error ? cause.message : "Chart download failed."); }
    finally { setExporting(false); }
  }
  const available = history.points.filter(point => point.nail !== null);
  const width = Math.max(850, history.points.length * 48), height = 248;
  const values = available.flatMap(point => [teamValue(point)!, point.opponent_nail!]);
  const minimum = Math.floor((Math.min(...values) - .35) * 2) / 2;
  const maximum = Math.ceil((Math.max(...values) + .35) * 2) / 2;
  const x = (index: number) => 76 + index * (width - 108) / Math.max(1, history.points.length - 1);
  const y = (value: number) => 28 + (maximum - value) / Math.max(.5, maximum - minimum) * 168;
  const ticks = available.length ? Array.from({ length: Math.round((maximum - minimum) * 2) + 1 }, (_, i) => minimum + i / 2) : [];
  const labelStep = Math.max(1, Math.ceil((maximum - minimum) / 8));
  const offset = (point: TeamRatingPoint) => point.nail !== null && Math.abs(y(teamValue(point)!) - y(point.opponent_nail!)) < 22 ? 11 : 0;
  const linePath = (opponent: boolean) => {
    let connected = false;
    return history.points.map((point, index) => {
    if (point.nail === null) { connected = false; return ""; }
    const command = `${connected ? "L" : "M"}${x(index) + (opponent ? offset(point) : -offset(point))},${y(opponent ? point.opponent_nail! : teamValue(point)!)}`;
    connected = true;
    return command;
  }).join(" ");
  };
  return <>
    <div className="team-rating-controls">
      <div className="team-rating-mode" role="group" aria-label="Chart view">
        <button type="button" aria-pressed={!adjusted} disabled={exporting} onClick={() => setAdjusted(false)}>Raw NAIL</button>
        <button type="button" aria-pressed={adjusted} disabled={exporting} onClick={() => setAdjusted(true)}>Adjusted matchup</button>
      </div>
      <button className="win-loss-envelope-download" type="button" disabled={exporting || !available.length} onClick={() => void downloadPng()}><Download size={14} />{exporting ? "Preparing PNG…" : "Download PNG"}</button>
    </div>
    <div className="team-rating-legend" aria-label="Chart series"><span className="team-rating-team-key">{history.team}{adjusted ? " + HCA/B2B" : ""}</span><span className="team-rating-opponent-key">Opponent</span></div>
    <p className="player-rating-path-note">{adjusted ? `HCA and back-to-back adjustments are applied to ${history.team}. The gap to the opponent is the predicted winning margin.` : "Expected lineup strength before each game, using that game’s player ratings and projected minutes."}</p>
    {exportError && <p className="schedule-report-warning" role="status">{exportError}</p>}
    {!history.complete && <p className="schedule-report-warning" role="status">{history.coverage.missing_prediction_game_ids.length} games have no saved prediction. Missing games are shown as gaps.</p>}
    {!history.points.length ? <p className="player-rating-path-note">No saved games are available for this season yet.</p>
      : <figure className="current-rating-trajectory team-rating-trajectory">
        <svg ref={chartRef} data-chart-mode={adjusted ? "adjusted" : "raw"} style={{ width: `max(100%, ${width}px)` }} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${history.team} ${history.season} ${adjusted ? "adjusted matchup" : "pregame team NAIL"} history`}>
          <title>{history.team} · Game-by-game pregame team NAIL</title>
          <g aria-label="NAIL axis: major ticks every 1 unit, minor ticks every 0.5 units">
            <text className="current-rating-axis-label" x={48} y={16} textAnchor="end">NAIL</text>
            {ticks.map(tick => <g key={tick} data-nail-tick={tick} data-tick-kind={Number.isInteger(tick) ? "major" : "minor"}>
              <line className={Number.isInteger(tick) ? "current-rating-grid-major" : "current-rating-grid-minor"} x1={76} x2={width - 32} y1={y(tick)} y2={y(tick)} />
              {Number.isInteger(tick) && tick % labelStep === 0 && <text className="current-rating-axis-label" x={48} y={y(tick) + 3} textAnchor="end">{tick}</text>}
            </g>)}
          </g>
          {available.map(point => {
            const index = history.points.indexOf(point);
            return <line className="team-rating-gap" key={point.game_id} x1={x(index) - offset(point)} x2={x(index) + offset(point)} y1={y(teamValue(point)!)} y2={y(point.opponent_nail!)} />;
          })}
          <path className="current-rating-line" data-series="team" d={linePath(false)} />
          <path className="current-rating-line team-rating-opponent-line" data-series="opponent" d={linePath(true)} />
          {history.points.map((point, index) => {
            const label = `${point.game_date_pacific} · ${history.team} ${point.home_away === "home" ? "vs." : "@"} ${point.opponent}`;
            const description = point.nail === null ? `${label} · Saved prediction unavailable` : `${label} · ${history.team} NAIL ${rating.format(point.nail)} · ${point.opponent} NAIL ${rating.format(point.opponent_nail!)} · HCA ${rating.format(point.adjustments!.hca)} · B2B ${rating.format(point.adjustments!.b2b)} · Projected winning margin ${rating.format(point.predicted_margin!)} · ${stateLabel(point)} · ${point.ratings_observed_through_game_date_pacific ? `Ratings through ${point.ratings_observed_through_game_date_pacific}` : "Opening rating priors"}`;
            return <g key={point.game_id} data-game-id={point.game_id} data-nail={point.nail ?? "missing"} data-opponent-nail={point.opponent_nail ?? "missing"} data-predicted-margin={point.predicted_margin ?? "missing"} data-projection-state={point.projection_state ?? "missing"}>
              {point.nail === null ? <g className="team-rating-missing" role="img" aria-label={description}>
                <title>{description}</title><text className="current-rating-axis-label" x={x(index)} y={220} textAnchor="middle">×</text>
              </g> : [false, true].map(opponent => {
                const value = opponent ? point.opponent_nail! : teamValue(point)!;
                const otherValue = opponent ? teamValue(point)! : point.opponent_nail!;
                const pointX = x(index) + (opponent ? offset(point) : -offset(point));
                const team = opponent ? point.opponent : history.team;
                const labelY = y(value) + (value >= otherValue ? -15 : 21);
                return <a key={opponent ? "opponent" : "team"} data-series={opponent ? "opponent" : "team"} data-value={value} href={`#schedule/${point.game_date_pacific}`} aria-label={`${team}${!opponent && adjusted ? " adjusted" : ""} ${rating.format(value)} · ${description}`}>
                  <title>{description}</title>
                  <circle className={`current-rating-logo-halo team-rating-${point.projection_state}${opponent ? " team-rating-opponent-halo" : ""}`} cx={pointX} cy={y(value)} r={10} />
                  <image className="current-rating-opponent-logo" href={`/api/team-logos/${opponent ? point.opponent_team_id : history.team_id}.svg`} x={pointX - 9} y={y(value) - 9} width={18} height={18} />
                  <text className="current-rating-value" x={pointX} y={labelY} textAnchor="middle">{rating.format(value)}</text>
                </a>;
              })}
              {(index === 0 || index === history.points.length - 1 || index % 3 === 0) && <text className="current-rating-axis-label" x={x(index)} y={232} textAnchor="middle">{shortDate(point.game_date_pacific)}</text>}
            </g>;
          })}
        </svg>
        <figcaption>Each series uses its own team logo. Select a point to view the game.{history.points.some(point => point.projection_state === "historical_replay") ? " Dashed rings mark historical replays." : ""}{history.points.some(point => point.projection_state === "upcoming") ? " Dotted rings mark upcoming games." : ""}</figcaption>
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
        if (data.kind !== "saved_team_rating_history" || data.team !== team || data.season !== season || !Array.isArray(data.points) || data.points.some(point => point.availability === "available" && (!point.adjustments || ![point.nail, point.opponent_nail, point.adjustments.hca, point.adjustments.b2b, point.predicted_margin].every(Number.isFinite)))) throw new Error("Team NAIL history is unavailable.");
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
