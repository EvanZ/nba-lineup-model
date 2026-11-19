import { PlayerHeadshot } from "./PlayerHeadshot";

export type TeamRosterPlayer = {
  player_id: number;
  player_name: string;
  name_missing: boolean;
  position: string | null;
  rank: number | null;
  nail: number | null;
  offense_rating: number | null;
  defense_rating: number | null;
  possessions: number | null;
  games: number | null;
};

export type TeamRosterData = {
  status: "available";
  players: TeamRosterPlayer[];
  roster_as_of: string | null;
  identity_warning?: string | null;
  possessions_label: string;
  rating_state: {
    kind: "daily" | "completed_fit";
    season?: string;
    evidence_through_date?: string;
    effective_from_date?: string;
    rank_population_count?: number;
  };
} | { status: "unavailable"; message: string };

const whole = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const rating = (value: number | null) => value === null ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
const quantity = (value: number | null) => value === null ? "—" : whole.format(value);
const ratingClass = (value: number | null) => value === null ? "numeric rating-cell" : `${value < 0 ? "negative" : "positive"} numeric rating-cell`;
const playerHref = (id: number) => `#player/${id}`;

export function TeamRosterView({ roster, compact = false }: { roster?: TeamRosterData; compact?: boolean }) {
  if (!roster || roster.status === "unavailable") return <section className="team-roster-section" aria-label="Player ratings">
    <p className="error" role="alert">{roster?.message ?? "Saved roster data is unavailable. Please try again."}</p>
  </section>;
  const current = roster.rating_state.kind === "daily";
  const rankTitle = current ? `Rank across the accepted daily ratings population (${roster.rating_state.rank_population_count} players).` : "Saved league-wide NAIL rank.";
  const possessionsTitle = current ? "Cumulative rating possessions through the stated evidence date, across all teams." : "Player-season possessions across all teams, including before a trade.";
  return <section className="team-roster-section" aria-labelledby="team-roster-title">
    <div className="team-section-heading">
      <div><p className="section-kicker">Roster</p><h2 id="team-roster-title">Player ratings.</h2></div>
    </div>
    <p className="player-rating-path-note">{current
      ? `Roster: ${roster.roster_as_of} · NAIL after ${roster.rating_state.evidence_through_date} slate · effective ${roster.rating_state.effective_from_date}.`
      : `${roster.rating_state.season} season-end roster · completed-season NAIL.`}</p>
    {current && <p className="player-rating-path-note">Offense/defense splits and season game counts are not yet available. — indicates unavailable data.</p>}
    {roster.identity_warning && <p className="player-rating-path-note" role="alert">{roster.identity_warning}</p>}
    {!compact ? <div className="rankings-table-wrap">
      <table className="rankings-table team-roster-table">
        <thead><tr>
          <th scope="col" className="numeric" title={rankTitle}>Rank</th><th scope="col">Player</th><th scope="col">Pos</th>
          <th scope="col" className="numeric">NAIL</th><th scope="col" className="numeric">Off</th><th scope="col" className="numeric">Def</th>
          <th scope="col" className="numeric" title={possessionsTitle}>{roster.possessions_label}</th><th scope="col" className="numeric">Games</th>
        </tr></thead>
        <tbody>{roster.players.map(player => <tr key={player.player_id}>
          <td className="rank-number" title={rankTitle}>{quantity(player.rank)}</td>
          <th scope="row"><PlayerHeadshot player={player} /><a className="player-name-link" href={playerHref(player.player_id)} title={player.name_missing ? "Official name not yet available." : undefined}>{player.player_name}</a></th>
          <td>{player.position || "—"}</td>
          <td className={`${ratingClass(player.nail)} nail-rating-cell`}>{rating(player.nail)}</td>
          <td className={ratingClass(player.offense_rating)}>{rating(player.offense_rating)}</td>
          <td className={ratingClass(player.defense_rating)}>{rating(player.defense_rating)}</td>
          <td className="numeric quantity-cell">{quantity(player.possessions)}</td><td className="numeric quantity-cell">{quantity(player.games)}</td>
        </tr>)}</tbody>
      </table>
    </div> : <ol className="mobile-card-list team-roster-mobile-list" aria-label="Player ratings">
      {roster.players.map(player => <li className="mobile-player-card" key={player.player_id}>
        <div className="mobile-card-heading">
          <span className="mobile-card-rank" title={rankTitle}>{player.rank === null ? "—" : `#${player.rank}`}</span>
          <a className="mobile-player-identity" href={playerHref(player.player_id)}><PlayerHeadshot player={player} /><span><strong>{player.player_name}</strong><small>{player.position || "—"}</small></span></a>
          <span className="mobile-primary-rating"><small>NAIL</small><span className={`rating-value ${ratingClass(player.nail)}`}>{rating(player.nail)}</span></span>
        </div>
        <dl className="mobile-metric-grid team-roster-mobile-metrics">
          <div><dt>Offense</dt><dd>{rating(player.offense_rating)}</dd></div><div><dt>Defense</dt><dd>{rating(player.defense_rating)}</dd></div>
          <div><dt title={possessionsTitle}>{roster.possessions_label}</dt><dd>{quantity(player.possessions)}</dd></div><div><dt>Games</dt><dd>{quantity(player.games)}</dd></div>
        </dl>
      </li>)}
    </ol>}
  </section>;
}
