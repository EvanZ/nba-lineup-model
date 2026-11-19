import { useEffect, useState } from "react";
import { PlayerRatingHistory } from "./PlayerRatingHistory";
import { PlayerRotationHistory } from "./PlayerRotationHistory";
import { PlayerSeasonHistory } from "./PlayerSeasonHistory";

export type SavedPlayerSummary = {
  kind: "saved_player_summary";
  schema_version: number;
  season: string;
  player_id: number;
  player_name: string;
  team_id: number | null;
  team: string | null;
  position: string | null;
  biography_available: boolean;
  draft: { year: string | number | null; round: string | number | null; pick: string | number | null } | null;
  observed_through: string;
  effective_from: string;
  rating: { nail: number; starting_nail: number; delta: number; on_court_possessions: number; standard_error: number; initialization_source: string } | null;
  minutes: { conditional_estimate: number; starting_estimate: number; retained_observation_count: number } | null;
};

const decimal = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const whole = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const signed = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2, signDisplay: "exceptZero" });
const uncertainty = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

function Rating({ value }: { value: number }) {
  return <span className={`rating-value${value < 0 ? " negative" : ""}`}>{signed.format(value)}</span>;
}

function draftLabel(draft: SavedPlayerSummary["draft"]) {
  if (!draft) return "Draft information unavailable";
  if (String(draft.year).toLowerCase() === "undrafted") return "Undrafted";
  if (!draft.year || !Number.isFinite(Number(draft.year)) || Number(draft.year) <= 0) return "Draft information unavailable";
  const parts = [`${draft.year} draft`];
  if (Number(draft.round) > 0) parts.push(`Round ${draft.round}`);
  if (Number(draft.pick) > 0) parts.push(`Pick ${draft.pick}`);
  return parts.join(" · ");
}

function dateLabel(day: string) {
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }).format(new Date(`${day}T12:00:00Z`));
}

function PlayerPhoto({ player }: { player: SavedPlayerSummary }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [player.player_id]);
  if (failed) {
    const initials = player.player_name.split(/\s+/).map((part) => part[0]).join("").slice(0, 2);
    return <span className="player-headshot player-headshot-fallback" title="Player photo unavailable" aria-hidden="true">{initials}</span>;
  }
  return <img className="player-headshot" src={`/api/headshots/${player.player_id}.png`} alt={player.player_name} onError={() => setFailed(true)} />;
}

export function PlayerSummaryView({ player }: { player: SavedPlayerSummary }) {
  const rating = player.rating;
  const minutes = player.minutes;
  return <article className="player-profile-page" aria-labelledby="player-profile-title">
    <section className="player-profile-hero">
      <PlayerPhoto player={player} />
      <div>
        <p className="eyebrow">{player.team ? <a href={`#team/${player.team}`}>{player.team}</a> : "Team unavailable"} · {player.position || "Position unavailable"}</p>
        <h1 id="player-profile-title">{player.player_name}</h1>
        <p className="player-profile-meta">{draftLabel(player.draft)}</p>
      </div>
      <div className="profile-hero-rating">
        <span>{player.season} NAIL</span>
        {rating ? <Rating value={rating.nail} /> : <span className="rating-value">—</span>}
      </div>
    </section>
    {!player.biography_available && <p className="player-rating-path-note" role="status">Biography details are unavailable in the saved NBA player index. Name and current team come from the saved daily state.</p>}
    <section className="player-profile-section current-season-update" aria-labelledby="current-season-update-title">
      <div className="player-profile-heading">
        <p className="section-kicker">Active {player.season} state</p>
        <h2 id="current-season-update-title">In-season update.</h2>
      </div>
      <p className="player-rating-path-note">Evidence through {dateLabel(player.observed_through)} · Effective {dateLabel(player.effective_from)}</p>
      {rating && minutes ? <dl className="player-stat-grid player-summary-grid">
        <div><dt>Starting NAIL</dt><dd><Rating value={rating.starting_nail} /></dd></div>
        <div><dt>In-season delta</dt><dd><Rating value={rating.delta} /></dd></div>
        <div className="player-forecast-total"><dt>Current NAIL</dt><dd><Rating value={rating.nail} /></dd></div>
        <div><dt>On-court possessions</dt><dd>{whole.format(rating.on_court_possessions)}</dd></div>
        <div><dt>Rating uncertainty (1 SE)</dt><dd>±{uncertainty.format(rating.standard_error)}</dd></div>
        <div><dt>Starting conditional MPG</dt><dd>{decimal.format(minutes.starting_estimate)}</dd></div>
        <div className="player-forecast-total"><dt>Current conditional MPG</dt><dd>{decimal.format(minutes.conditional_estimate)}</dd></div>
        <div><dt>Observations in saved window</dt><dd>{minutes.retained_observation_count}</dd></div>
      </dl> : <p className="player-rating-path-note" role="status">No saved current-season rating or minutes state is available for this player.</p>}
      <p className="player-rating-path-note">Conditional MPG estimates minutes when available. It is separate from injury-adjusted game projections. Starting values are the saved initialization for this player.</p>
    </section>
    <PlayerRatingHistory playerId={player.player_id} summaryThrough={player.observed_through} />
    <PlayerSeasonHistory playerId={player.player_id} />
    <PlayerRotationHistory playerId={player.player_id} />
  </article>;
}
