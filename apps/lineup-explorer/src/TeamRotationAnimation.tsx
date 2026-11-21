import { useEffect, useRef, useState } from "react";
import { releaseFetch as fetch } from "./releaseFetch";
import { mountRotationAnimation } from "./teamRotationAnimationRenderer";
import type { RotationAnimationData } from "./teamRotationAnimationTypes";

export default function TeamRotationAnimation({ team, season, teamName }: {
  team: string;
  season: string;
  teamName: string;
}) {
  const [data, setData] = useState<RotationAnimationData>();
  const [error, setError] = useState<string>();
  const [progress, setProgress] = useState<number | null>(null);
  const [exportError, setExportError] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const encoding = useRef<AbortController>(null);
  const gif = useRef<Blob>(null);

  useEffect(() => {
    const request = new AbortController();
    async function loadHistory() {
      try {
        const query = new URLSearchParams({ season });
        const response = await fetch(`/api/teams/${encodeURIComponent(team)}/rotation-history?${query}`, {
          signal: request.signal,
        });
        if (response.status === 404) {
          throw new Error("No saved rotation history is published for this team-season.");
        }
        if (!response.ok) throw new Error("Saved rotation history is temporarily unavailable.");
        const result: RotationAnimationData = await response.json();
        if (result.kind !== "saved_team_rotation_history" || result.team !== team || result.season !== season ||
          !result.players.length || !result.frames.length) {
          throw new Error("Saved rotation history is unavailable.");
        }
        if (!request.signal.aborted) setData(result);
      } catch (failure) {
        if (!request.signal.aborted) setError((failure as Error).message);
      }
    }
    void loadHistory();
    return () => {
      request.abort();
      encoding.current?.abort();
    };
  }, [team, season]);

  useEffect(() => {
    if (root.current && data) return mountRotationAnimation(root.current, data);
  }, [data]);

  async function exportGif() {
    if (!data || encoding.current) return;
    const request = new AbortController();
    encoding.current = request;
    setProgress(0);
    setExportError(false);
    try {
      if (!gif.current) {
        const { makeRotationGif } = await import("./teamRotationGif");
        gif.current = await makeRotationGif(data, teamName, request.signal, setProgress);
      }
      if (request.signal.aborted) return;
      const link = document.createElement("a");
      link.href = URL.createObjectURL(gif.current);
      link.download = `${team.toLowerCase()}-${season}-clustered-rotation.gif`;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    } catch {
      if (!request.signal.aborted) setExportError(true);
    } finally {
      encoding.current = null;
      if (!request.signal.aborted) setProgress(null);
    }
  }

  if (error) return <p className="team-empty-state" role="status">{error}</p>;
  if (!data) return <p className="team-empty-state" role="status">Loading rotation history.</p>;

  return <div className="team-rotation-animation" ref={root}>
    <p className="team-rotation-note">Rotation clusters after each game. Minutes are season totals through the selected game.</p>
    <div className="team-rotation-animation-controls">
      <div className="team-rotation-toggle">
        <button type="button" data-previous>Previous</button>
        <button type="button" data-play>Play</button>
        <button type="button" data-next>Next</button>
      </div>
      <div className="team-rotation-toggle">
        <button type="button" onClick={exportGif} disabled={progress !== null}>
          {progress === null ? "Download GIF" : `Preparing GIF… ${Math.floor(progress * 100)}%`}
        </button>
      </div>
      <span className="team-rotation-animation-date" data-date />
      <span data-game aria-live="polite" />
    </div>
    {exportError && <p className="team-empty-state" role="alert">GIF export failed. Try again.</p>}
    <label className="team-rotation-animation-timeline">
      <span>Game <span data-position /></span>
      <input type="range" min="0" max={data.frames.length - 1} step="1" defaultValue="0" aria-label="Rotation game" />
    </label>
    <svg className="team-rotation-animation-chart" role="img" aria-label={`${teamName} game-by-game clustered rotation history`} />
    <div className="team-rotation-legend">
      <span>≤0</span><i aria-hidden="true" /><span>1</span><span>Midpoint 0.50</span>
      <span className="team-rotation-animation-missing">Not yet appeared</span>
    </div>
    <p className="team-rotation-note" data-stage />
  </div>;
}
