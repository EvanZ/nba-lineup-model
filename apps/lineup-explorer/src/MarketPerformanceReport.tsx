import { useEffect, useState } from "react";

type Score = {
  games: number;
  brier: number | null;
  winner_accuracy: number | null;
  margin_mae: number | null;
  margin_rmse: number | null;
  ats: { wins: number; losses: number; pushes: number; no_picks: number; not_applicable: number; unavailable: number };
};
type Series = "model" | "bookmaker_consensus" | "polymarket" | "actual_minutes_counterfactual";
type EdgeBin = {
  key: string; label: string; n: number; share: number | null;
  wins: number; losses: number; pushes: number; no_picks: number; unavailable: number;
  cover_rate: number | null;
};
type Report = {
  as_of: string;
  windows: { key: string; game_count: number; comparison_game_count: number; series: Record<Series, Score>; edge_bins: { bins: EdgeBin[]; paired_games: number; missing_pairs: number } }[];
  coverage: { expected_game_count: number; dates_without_saved_reports: string[]; missing_market_fields: Record<string, unknown> };
};
const columns: [Series, string][] = [
  ["model", "Gestalt"], ["bookmaker_consensus", "Books"],
  ["polymarket", "Poly"], ["actual_minutes_counterfactual", "Actual mins"],
];
const number = (value: number | null, digits = 3) => value === null ? "—" : value.toFixed(digits);
const percent = (value: number | null) => value === null ? "—" : `${(value * 100).toFixed(1)}%`;

export function MarketPerformanceReport() {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    async function refresh() {
      try {
        // Pipeline selections are independent of the legacy site-release pin.
        const response = await globalThis.fetch("/api/market-performance?season=2026-27", {
          signal: controller.signal, cache: "default",
        });
        if (!response.ok) throw new Error("The saved market report is unavailable.");
        const next = await response.json() as Report;
        if (controller.signal.aborted) return;
        setReport(next);
        setError(null);
      } catch (reason) {
        if (!controller.signal.aborted) {
          setReport(null);
          setError((reason as Error).message);
        }
      } finally {
        if (!controller.signal.aborted) timer = window.setTimeout(refresh, Math.max(1_000, 3_600_000 - Date.now() % 3_600_000));
      }
    }
    void refresh();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, []);
  return <MarketPerformanceView report={report} error={error} />;
}

export function MarketPerformanceView({ report, error }: { report: Report | null; error: string | null }) {
  const season = report?.windows.find((window) => window.key === "season");
  return <aside className="schedule-performance-card" aria-labelledby="schedule-performance-title">
    <div className="schedule-performance-card-heading">
      <div><span>Model check · Season</span><strong id="schedule-performance-title">How we’re doing</strong></div>
    </div>
    {!season ? <p role={error ? "alert" : "status"}>{error ?? "Loading saved market report…"}</p> : <>
      <div className="schedule-performance-label">
        <span>{season.comparison_game_count} compared / {season.game_count} of {report?.coverage.expected_game_count} finals · through {report?.as_of}</span>
        <small>Same-game comparison cohort</small>
      </div>
      <div className="schedule-performance-columns schedule-performance-columns-header">
        <span></span>{columns.map(([key, label]) => <span key={key}>{label}<small>{key === "actual_minutes_counterfactual" ? "Postgame · " : ""}n={season.series[key].games}</small></span>)}
      </div>
      <div className="schedule-performance-columns"><span>Accuracy</span>{columns.map(([key]) => <strong key={key}>{percent(season.series[key].winner_accuracy)}</strong>)}</div>
      <div className="schedule-performance-columns"><span>Brier ↓</span>{columns.map(([key]) => <strong key={key}>{number(season.series[key].brier)}</strong>)}</div>
      <div className="schedule-performance-columns"><span>Margin MAE ↓</span>{columns.map(([key]) => <strong key={key}>{number(season.series[key].margin_mae, 1)}</strong>)}</div>
      <div className="schedule-performance-columns"><span>Margin RMSE ↓</span>{columns.map(([key]) => <strong key={key}>{number(season.series[key].margin_rmse, 1)}</strong>)}</div>
      <div className="schedule-performance-columns"><span>ATS W–L–P</span>{columns.map(([key]) => {
        const ats = season.series[key].ats;
        return <strong key={key}>{key === "bookmaker_consensus" ? "—" : `${ats.wins}–${ats.losses}–${ats.pushes}`}</strong>;
      })}</div>
      <div className="schedule-performance-section schedule-performance-edge-buckets">
        <span className="schedule-performance-edge-title">ATS by absolute model–market edge</span>
        <div className="schedule-performance-edge-grid">
          {season.edge_bins.bins.map((bin) => <div key={bin.key}>
            <span>{bin.label}</span>
            <strong>{bin.wins}–{bin.losses}–{bin.pushes}</strong>
            <small>{bin.n} games · {percent(bin.share)} of volume</small>
            <small>{percent(bin.cover_rate)} cover rate</small>
            {bin.no_picks > 0 && <small>{bin.no_picks} no-picks</small>}
            {bin.unavailable > 0 && <small>{bin.unavailable} ungraded</small>}
          </div>)}
        </div>
      </div>
    </>}
  </aside>;
}
