import { TeamRatingHistory } from "./TeamRatingHistory";
import { TeamRosterView, type TeamRosterData } from "./TeamRoster";
import { teamLogoUrl } from "./teamLogos";

export type TeamNavigationData = {
  kind: "saved_team_navigation";
  team: string;
  display_name: string;
  season: string;
  available_seasons: string[];
  teams: { team: string; display_name: string }[];
  roster_as_of: string | null;
  deferred_sections: string[];
  roster?: TeamRosterData;
};

const href = (team: string, season: string) =>
  `#team/${team}?season=${encodeURIComponent(season)}`;

export function TeamNavigationView({ navigation, compact = false }: { navigation: TeamNavigationData; compact?: boolean }) {
  const logo = teamLogoUrl(navigation.team);
  return (
    <article className="rankings-page team-page" aria-labelledby="team-navigation-title">
      <section className="rankings-hero team-page-hero">
        <div className="team-page-branding">
          <p className="eyebrow">{navigation.season} team season</p>
          <div className="team-page-title-row">
            {logo && <img className="team-page-watermark" src={logo} alt="" aria-hidden="true" />}
            <span aria-hidden="true">{navigation.team}</span>
            <h1 id="team-navigation-title">{navigation.display_name}.</h1>
          </div>
        </div>
      </section>
      <nav className="team-page-toolbar" aria-label="Team and season navigation">
        <div className="team-page-controls team-navigation-controls">
          <label className="rankings-season">
            <span>Team</span>
            <select aria-label="Team" value={navigation.team} onChange={event => {
              window.location.hash = href(event.target.value, navigation.season);
            }}>
              {navigation.teams.map(team => (
                <option key={team.team} value={team.team}>{team.display_name} · {team.team}</option>
              ))}
            </select>
          </label>
          <label className="rankings-season">
            <span>Season</span>
            <select aria-label="Season" value={navigation.season} onChange={event => {
              window.location.hash = href(navigation.team, event.target.value);
            }}>
              {navigation.available_seasons.map(season => (
                <option key={season} value={season}>{season}</option>
              ))}
            </select>
          </label>
        </div>
      </nav>
      <TeamRatingHistory key={`${navigation.team}-${navigation.season}`} team={navigation.team} season={navigation.season} />
      <TeamRosterView roster={navigation.roster} compact={compact} />
      <p className="player-rating-path-note" role="status">
        Wins, rotation, and lineup sections are not available yet.
      </p>
    </article>
  );
}
