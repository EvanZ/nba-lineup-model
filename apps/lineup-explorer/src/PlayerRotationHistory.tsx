import { useEffect, useState } from "react";
import type { Player } from "./types";
import { PlayerRotationHistoryCards, PlayerRotationHistoryChart } from "./PlayerHistoricalCharts";

type Row = NonNullable<Player["rotation_history"]>[number];
export type RotationHistory = { kind: "saved_player_rotation_history"; player_id: number; player_name: string | null; rows: Row[] };
const decimal = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const whole = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const value = (n: number | null) => n === null ? "—" : decimal.format(n);

export function PlayerRotationHistoryView({ history }: { history: RotationHistory }) {
  if (!history.rows.length) return <p className="player-rating-path-note">No saved availability/minutes history is available for this player.</p>;
  return <>
    <PlayerRotationHistoryChart player={{ player_name: history.player_name ?? history.rows[0].player_name, rotation_history: history.rows }} />
    <p className="player-rating-path-note">Saved history begins in 2015–16. Solid marks show observed outcomes; dashed marks show saved preseason estimates on an 82-game scale. Any 2026–27 row is the frozen preseason forecast.</p>
    <div className="player-history-table-wrap player-rotation-history-desktop">
      <table className="player-history-table player-rotation-history-table">
        <thead><tr><th scope="col">Season</th><th scope="col">Available G</th><th scope="col">Availability estimate</th><th scope="col">Observed MPG when available</th><th scope="col">Minutes estimate</th><th scope="col">Total minutes</th></tr></thead>
        <tbody>{[...history.rows].reverse().map((row) => {
          const total = row.is_preseason_forecast ? row.projected_total_minutes : row.actual_total_minutes;
          return <tr key={row.season} data-rotation-season={row.season} className={row.is_preseason_forecast ? "rotation-history-forecast-row" : undefined}>
            <td>{row.season}{row.is_preseason_forecast && <small className="player-history-game-status">Preseason forecast</small>}</td>
            <td className="quantity-cell">{row.actual_available_games === null ? "—" : `${whole.format(row.actual_available_games)} / ${row.known_roster_games === null ? "—" : whole.format(row.known_roster_games)}`}</td>
            <td className="rotation-forecast-cell">{row.predicted_available_share === null ? "—" : `${decimal.format(row.predicted_available_share * 82)} G`}</td>
            <td className="quantity-cell">{value(row.actual_minutes_per_available_game)}</td>
            <td className="rotation-forecast-cell">{value(row.predicted_minutes_per_available_game)}</td>
            <td className="quantity-cell">{total === null ? "—" : whole.format(total)}{row.is_preseason_forecast && <small className="player-history-game-status">Projected</small>}</td>
          </tr>;
        })}</tbody>
      </table>
    </div>
    <PlayerRotationHistoryCards rows={history.rows} />
  </>;
}

export function PlayerRotationHistory({ playerId }: { playerId: number }) {
  const [history, setHistory] = useState<RotationHistory | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let retry: ReturnType<typeof setTimeout>;
    setHistory(null); setError(null);
    async function load() {
      try {
        const response = await globalThis.fetch(`/api/players/${playerId}/rotation-history`, { signal: controller.signal, cache: "no-cache" });
        if (!response.ok) throw new Error("Saved availability/minutes history is temporarily unavailable.");
        const data = await response.json() as RotationHistory;
        if (data.kind !== "saved_player_rotation_history" || data.player_id !== playerId || !Array.isArray(data.rows)) throw new Error("Saved availability/minutes history is incomplete or unavailable.");
        if (!controller.signal.aborted) { setHistory(data); setError(null); }
      } catch (cause) {
        if (controller.signal.aborted) return;
        setError(cause instanceof Error ? cause.message : "Saved availability/minutes history is unavailable.");
        retry = setTimeout(load, 60000);
      }
    }
    void load();
    return () => { controller.abort(); clearTimeout(retry); };
  }, [playerId]);
  return <section className="player-profile-section" aria-labelledby="player-rotation-history-title">
    <div className="player-profile-heading"><p className="section-kicker">Availability and rotation</p><h2 id="player-rotation-history-title">Availability and minutes.</h2></div>
    {error ? <p className="schedule-report-warning" role="status">{error}</p> : history ? <PlayerRotationHistoryView history={history} /> : <p className="player-rating-path-note">Loading availability/minutes history…</p>}
  </section>;
}
