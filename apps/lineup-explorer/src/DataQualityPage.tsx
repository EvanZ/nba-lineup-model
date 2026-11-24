import { useEffect, useState } from "react";
import { CircleAlert, LoaderCircle } from "lucide-react";

import { releaseFetch as fetch } from "./releaseFetch";

type EventKind = "failure" | "warning" | "recovery";
type QualityEvent = {
  event_id: string;
  timestamp: string;
  game_date: string | null;
  pipeline: string;
  stage: string;
  kind: EventKind;
  message: string;
  recovered_at?: string | null;
  original_failure_id?: string | null;
};

type QualityReport = {
  season: string;
  checked_at: string;
  counts: Record<EventKind, number>;
  failures_today_without_recovery: number;
  events?: QualityEvent[];
};

const FILTERS = [
  { value: "all", label: "All events" },
  { value: "failure", label: "Failures" },
  { value: "warning", label: "Warnings" },
  { value: "recovery", label: "Recoveries" },
] as const;
const LABELS: Record<EventKind, string> = { failure: "Failure", warning: "Warning", recovery: "Recovery" };
const PAGE_SIZE = 30;
const pacificDateTime = new Intl.DateTimeFormat("en-US", {
  dateStyle: "medium", timeStyle: "short", timeZone: "America/Los_Angeles",
});
const localDateTime = (value: string) => pacificDateTime.format(new Date(value));
const label = (value: string) => value.replaceAll("_", " ");

export function useDataQuality(includeEvents: boolean) {
  const [report, setReport] = useState<QualityReport | null>(null);
  const [unavailable, setUnavailable] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let timeout = 0;
    const refresh = async () => {
      try {
        const endpoint = includeEvents ? "/api/reports/data-quality" : "/api/reports/data-quality?summary=true";
        const response = await fetch(endpoint, { signal: controller.signal });
        if (!response.ok) throw new Error("Saved data quality events are unavailable.");
        const data = await response.json() as QualityReport;
        if (!controller.signal.aborted) {
          setReport(data);
          setUnavailable(false);
        }
      } catch (error) {
        if (!controller.signal.aborted && (error as Error).name !== "AbortError") setUnavailable(true);
      } finally {
        if (!controller.signal.aborted) timeout = window.setTimeout(() => void refresh(), 60_000);
      }
    };
    void refresh();
    return () => {
      controller.abort();
      window.clearTimeout(timeout);
    };
  }, [includeEvents]);

  return { report, unavailable };
}

export function DataQualityPage({ report, unavailable }: {
  report: QualityReport | null;
  unavailable: boolean;
}) {
  const [kind, setKind] = useState<EventKind | "all">("all");
  const [pipeline, setPipeline] = useState("all");
  const [limit, setLimit] = useState(PAGE_SIZE);
  const events = report?.events;
  const pipelines = [...new Set(events?.map((event) => event.pipeline))].sort();
  const filtered = events?.filter((event) => (
    (kind === "all" || event.kind === kind) && (pipeline === "all" || event.pipeline === pipeline)
  ));

  return (
    <section className="data-quality-page" aria-labelledby="data-quality-title">
      <h1 id="data-quality-title">Data quality.</h1>
      <p>Saved pipeline failures, warnings, and recoveries for the 2026–27 season.</p>
      {unavailable && <p className="error" role="status"><CircleAlert size={16} aria-hidden="true" />
        {events ? "Refresh unavailable. Showing the last loaded events." : "Saved pipeline events are unavailable."}
      </p>}
      {!events && !unavailable && <p className="profile-loading"><LoaderCircle className="spin" size={16} aria-hidden="true" /> Loading saved events</p>}
      {events && report && filtered && <>
        <div className="data-quality-summary" aria-label="Archived event counts">
          <div><strong>{report.counts.failure}</strong><span>Failures</span></div>
          <div><strong>{report.counts.warning}</strong><span>Warnings</span></div>
          <div><strong>{report.counts.recovery}</strong><span>Recoveries</span></div>
          <p>Archive checked {localDateTime(report.checked_at)} PT.</p>
        </div>
        <div className="data-quality-controls">
          <label>Event type
            <select value={kind} onChange={(event) => {
              setKind(event.target.value as EventKind | "all");
              setLimit(PAGE_SIZE);
            }}>
              {FILTERS.map(({ value, label }) => <option value={value} key={value}>{label}</option>)}
            </select>
          </label>
          <label>Pipeline
            <select value={pipeline} onChange={(event) => {
              setPipeline(event.target.value);
              setLimit(PAGE_SIZE);
            }}>
              <option value="all">All pipelines</option>
              {pipelines.map((value) => <option key={value} value={value}>{label(value)}</option>)}
            </select>
          </label>
          <p>{Math.min(limit, filtered.length)} of {filtered.length} events</p>
        </div>
        {filtered.length === 0 && <p className="data-quality-empty">No saved events match these filters.</p>}
        <ol className="data-quality-events" aria-label="Saved pipeline events">
          {filtered.slice(0, limit).map((event) => (
            <li key={event.event_id} className="data-quality-event" data-event-kind={event.kind}>
              <div className="data-quality-event-context">
                <strong>{LABELS[event.kind]}</strong>
                <time dateTime={event.timestamp}>{localDateTime(event.timestamp)} PT</time>
                {event.game_date && <span>Game date {event.game_date}</span>}
              </div>
              <div className="data-quality-event-detail">
                <p className="data-quality-event-pipeline">{label(event.pipeline)} · {label(event.stage)}</p>
                <p>{event.message}</p>
                {event.kind === "failure" && <small>
                  {event.recovered_at ? `Recovery recorded ${localDateTime(event.recovered_at)} PT` : "No recovery recorded"}
                </small>}
                {event.kind === "recovery" && <small>
                  {event.original_failure_id ? "Linked to the original failure" : "Original failure could not be verified"}
                </small>}
              </div>
            </li>
          ))}
        </ol>
        {limit < filtered.length && <button className="data-quality-more" type="button" onClick={() => setLimit(limit + PAGE_SIZE)}>Show more events</button>}
      </>}
    </section>
  );
}
