import { useEffect, useState } from "react";
import { PlayerAgingChart } from "./PlayerHistoricalCharts";
import { teamLogoUrl } from "./teamLogos";

type Split = { team_id: number; team: string; possessions: number; games: number; is_latest_team: boolean };
type Season = {
  team: string; team_id: number | null; season_max_rating?: number; season_min_rating?: number; season_max_player_id?: number; season_max_player_name?: string;
  season: string; player_id: number; player_name: string; age: number | null;
  rating: number; prior_rating: number | null; season_update: number | null;
  additive_profile_adjustment: number | null; observed_context_exposure: number | null;
  offense_rating: number | null; defense_rating: number | null; nail_rank: number | null;
  games: number | null; games_started: number | null; possessions: number | null; team_splits: Split[];
};
export type SeasonHistory = { kind: "saved_player_season_history"; player_id: number; seasons: Season[]; through_season: string; source_id: string };
const decimal = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const whole = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const signed = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2, signDisplay: "exceptZero" });
function Rating({ value }: { value: number | null }) {
  return value === null ? <>—</> : <span className={`rating-value${value < 0 ? " negative" : ""}`}>{signed.format(value)}</span>;
}
function Teams({ row }: { row: Season }) {
  if (!row.team_splits.length) return <>—</>;
  return <div className="player-team-splits">{row.team_splits.map((team) => <span key={team.team_id} className={`player-team-split${team.is_latest_team ? " latest" : ""}`}>
    <a className="lineup-team-mark" href={`#team/${team.team}?season=${row.season}`} title={`${team.team} · ${row.season}`}><img src={teamLogoUrl(team.team)} alt="" aria-hidden="true" loading="lazy" /><small>{team.team}</small></a>
    {row.team_splits.length > 1 && <span>{whole.format(team.possessions)} poss. · {whole.format(team.games)} GP</span>}
  </span>)}</div>;
}
const count = (n: number | null) => n === null ? "—" : whole.format(n);
const metrics = [
  ["age", "Age"], ["games", "GP"], ["games_started", "GS"], ["possessions", "Possessions"],
  ["rating", "NAIL"], ["offense_rating", "Offense"], ["defense_rating", "Defense"], ["nail_rank", "NAIL rank"],
  ["prior_rating", "Prior"], ["season_update", "Season update"], ["additive_profile_adjustment", "Additive profile"],
  ["observed_context_exposure", "Non-Additive Lineup Edge"],
] as const;
type Metric = typeof metrics[number][0];
function metric(row: Season, key: Metric) {
  const n = row[key];
  if (n === null) return <>—</>;
  if (key === "age") return decimal.format(n);
  if (key === "nail_rank") return `#${whole.format(n)}`;
  if (["games", "games_started", "possessions"].includes(key)) return count(n);
  return <Rating value={n} />;
}

export function PlayerSeasonHistoryView({ history }: { history: SeasonHistory }) {
  const rows = [...history.seasons].reverse();
  if (!rows.length) return <p className="player-rating-path-note">No saved completed-season ratings are available for this player.</p>;
  return <>
    <PlayerAgingChart player={{ player_id: history.player_id, player_name: rows[0].player_name, rating_history: history.seasons }} />
    <div className="player-history-table-wrap player-season-history-desktop">
      <table className="player-history-table player-season-history-table">
        <thead><tr><th scope="col">Season</th><th scope="col">Team split</th>{metrics.map(([key, label]) => <th scope="col" key={key}>{label}</th>)}</tr></thead>
        <tbody>{rows.map((row) => <tr key={row.season} data-career-season={row.season}>
          <td>{row.season}</td><td><Teams row={row} /></td>
          {metrics.map(([key]) => <td key={key} className={key === "rating" ? "nail-history-cell" : key === "nail_rank" ? "nail-rank-cell" : "quantity-cell"}>{metric(row, key)}</td>)}
        </tr>)}</tbody>
      </table>
    </div>
    <ol className="player-history-mobile-list" aria-label="Season rating history">{rows.map((row) => <li className="player-history-mobile-card" key={row.season} data-career-card={row.season}>
      <header className="player-history-mobile-header"><div><strong>{row.season}</strong><small>Age {row.age === null ? "—" : decimal.format(row.age)}</small></div><span className="player-history-mobile-rating"><small>NAIL</small><Rating value={row.rating} /></span></header>
      <div className="player-history-mobile-team"><Teams row={row} /></div>
      <p className="player-history-mobile-meta">{count(row.games)} GP · {count(row.games_started)} GS · {count(row.possessions)} possessions · {row.nail_rank === null ? "Rank unavailable" : `#${whole.format(row.nail_rank)}`}</p>
      <dl className="player-history-mobile-metrics">{metrics.filter(([key]) => !["age", "games", "games_started", "possessions", "rating", "nail_rank"].includes(key)).map(([key, label]) => <div key={key}><dt>{label}</dt><dd>{metric(row, key)}</dd></div>)}</dl>
    </li>)}</ol>
  </>;
}

export function PlayerSeasonHistory({ playerId }: { playerId: number }) {
  const [history, setHistory] = useState<SeasonHistory | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let retry: ReturnType<typeof setTimeout>;
    setHistory(null); setError(null);
    async function load() {
      try {
        const response = await globalThis.fetch(`/api/players/${playerId}/season-history`, { signal: controller.signal, cache: "no-cache" });
        if (!response.ok) throw new Error("Saved season history is temporarily unavailable.");
        const data = await response.json() as SeasonHistory;
        if (data.kind !== "saved_player_season_history" || data.player_id !== playerId || !Array.isArray(data.seasons)) throw new Error("Saved season history is incomplete or unavailable.");
        if (!controller.signal.aborted) { setHistory(data); setError(null); }
      } catch (cause) {
        if (controller.signal.aborted) return;
        setError(cause instanceof Error ? cause.message : "Saved season history is unavailable.");
        retry = setTimeout(load, 60000);
      }
    }
    void load();
    return () => { controller.abort(); clearTimeout(retry); };
  }, [playerId]);
  return <section className="player-profile-section" aria-labelledby="player-season-history-title">
    <div className="player-profile-heading"><p className="section-kicker">Completed seasons</p><h2 id="player-season-history-title">Season ratings.</h2></div>
    {error ? <p className="schedule-report-warning" role="status">{error}</p> : history ? <PlayerSeasonHistoryView history={history} /> : <p className="player-rating-path-note">Loading season history…</p>}
  </section>;
}
