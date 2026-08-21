import { useEffect, useMemo, useRef, useState } from "react";
import { BlockMath } from "react-katex";
import { trackPageView } from "./analytics";
import {
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  ChevronDown,
  ChevronUp,
  CircleAlert,
  Dices,
  Download,
  GitBranch,
  Info,
  LoaderCircle,
  Search,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";

import type { ContextFeature, FeatureResponseCurve, Matchup, Player, RankedLineup, RankedPlayer } from "./types";

type Side = "unit" | "opponent";
type AppView = "lab" | "rankings" | "lineups" | "about" | "player";
type AppRoute = { view: AppView; playerId?: number };
type Environment = "unit" | "neutral" | "opponent";
type AgingContributionKey = "prior" | "seasonUpdate" | "additiveProfile";

const SIDE_LABELS: Record<Side, string> = { unit: "Your unit", opponent: "Opponent" };
const MATERIAL_COMPONENT_CONTRIBUTION = 0.05;
const MODEL_LABEL = "NAIL-RAPM v1.2";
const FEATURE_DESCRIPTIONS: Record<string, string> = {
  home_minus_away_three_pa_per_100: "Sum of the five players' prior-season three-point attempts per 100 possessions.",
  home_minus_away_three_pm_per_100: "Sum of the five players' prior-season made three-pointers per 100 possessions.",
  home_minus_away_assists_per_100: "Sum of the five players' prior-season assists per 100 possessions.",
  home_minus_away_turnovers_per_100: "Sum of the five players' prior-season turnovers per 100 possessions.",
  home_minus_away_usage_per_100: "Sum of the five players' prior-season usage events per 100 possessions.",
  home_minus_away_offensive_rebounds_per_100: "Sum of the five players' prior-season offensive rebounds per 100 possessions.",
  home_minus_away_defensive_rebounds_per_100: "Sum of the five players' prior-season defensive rebounds per 100 possessions.",
  home_minus_away_steals_per_100: "Sum of the five players' prior-season steals per 100 possessions.",
  home_minus_away_blocks_per_100: "Sum of the five players' prior-season blocks per 100 possessions.",
  home_minus_away_bottom_two_three_pm: "Combined made three-pointers per 100 possessions for the two least prolific shooters in the unit.",
  home_minus_away_credible_shooter_count: "Number of players with at least two made three-pointers per 100 possessions in the prior season.",
  home_minus_away_top_two_assists: "Combined assists per 100 possessions for the two highest-assist players in the unit.",
  home_minus_away_usage_concentration: "Share of the unit's usage events supplied by its two highest-usage players.",
  home_minus_away_sqrt_offensive_rebounds: "Square root of the unit's total offensive rebounds per 100 possessions, encoding diminishing returns.",
  home_minus_away_sqrt_defensive_rebounds: "Square root of the unit's total defensive rebounds per 100 possessions, encoding diminishing returns.",
  home_minus_away_imputed_count: "Number of players whose profile is a cold-start or replacement estimate rather than a prior-season box-score profile.",
  home_minus_away_replacement_weight: "Sum of the players' exposure-gated replacement-profile weights.",
  home_minus_away_shooting_usage_interaction: "Bottom-two shooting multiplied by usage concentration; a shooting-depth-by-usage interaction.",
  home_minus_away_shooter_passing_interaction: "Credible-shooter count multiplied by top-two assists; a shooting-by-passing interaction.",
  home_minus_away_rebounding_usage_interaction: "Diminishing offensive rebounding multiplied by usage concentration.",
};

const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const wholeNumber = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

function formatRating(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
}

function formatChartRating(value: number) {
  return value.toFixed(1);
}

const TEAM_LOGO_SLUGS: Record<string, string> = {
  UTA: "utah",
};

function teamLogoUrl(team: string) {
  const slug = TEAM_LOGO_SLUGS[team] ?? team.toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/nba/500/${slug}.png`;
}

function playerHeadshotUrl(playerId: number) {
  return `/api/headshots/${playerId}.png`;
}

function Rating({ value, className }: { value: number; className?: string }) {
  return (
    <span className={["rating-value", value < 0 ? "negative" : "", className].filter(Boolean).join(" ")}>
      {formatRating(value)}
    </span>
  );
}

function useAppView(): AppRoute {
  const getView = (): AppRoute => {
    const playerMatch = window.location.hash.match(/^#player\/(\d+)$/);
    if (playerMatch) return { view: "player", playerId: Number(playerMatch[1]) };
    if (window.location.hash === "#about") return { view: "about" };
    if (window.location.hash === "#rankings") return { view: "rankings" };
    if (window.location.hash === "#lineups") return { view: "lineups" };
    return { view: "lab" };
  };
  const [view, setView] = useState<AppRoute>(getView);

  useEffect(() => {
    const updateView = () => setView(getView());
    window.addEventListener("hashchange", updateView);
    return () => window.removeEventListener("hashchange", updateView);
  }, []);

  useEffect(() => {
    trackPageView();
  }, [view]);

  return view;
}

function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);

  useEffect(() => {
    const mediaQuery = window.matchMedia(query);
    const updateMatches = () => setMatches(mediaQuery.matches);
    updateMatches();
    mediaQuery.addEventListener("change", updateMatches);
    return () => mediaQuery.removeEventListener("change", updateMatches);
  }, [query]);

  return matches;
}

function playerProfileHref(playerId: number) {
  return `#player/${playerId}`;
}

function PlayerProfilePage({ playerId }: { playerId: number }) {
  const [player, setPlayer] = useState<Player | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setPlayer(null);
    setError(null);
    void (async () => {
      try {
        const response = await fetch(`/api/players/${playerId}`, { signal: controller.signal });
        if (!response.ok) throw new Error("Player profile is unavailable.");
        setPlayer((await response.json()) as Player);
      } catch (profileError) {
        if ((profileError as Error).name !== "AbortError") setError((profileError as Error).message);
      }
    })();
    return () => controller.abort();
  }, [playerId]);

  if (error) return <p className="error profile-error"><CircleAlert size={16} /> {error}</p>;
  if (!player) return <div className="profile-loading"><LoaderCircle className="spin" size={20} /> Loading player profile</div>;
  const additiveProfileInputs = [
    player.three_pa_per_100,
    player.three_pm_per_100,
    player.assists_per_100,
    player.turnovers_per_100,
    player.usage_per_100,
    player.steals_per_100,
    player.blocks_per_100,
    player.offensive_rebound_pct,
  ];
  const hasAdditiveProfileInputs = additiveProfileInputs.every((value) => value !== null);
  const historyRows = completePlayerHistory(player);
  const isPreseasonPreview = player.rating_season === "2026-27";
  const ratingSeasonLabel = isPreseasonPreview
    ? "2026-27 preseason preview"
    : `${player.rating_season ?? "Latest"} ${MODEL_LABEL}`;
  const coldStartDescription = player.profile_source === "draft_cold_start_prior"
    ? "This preseason rating uses the forward draft cold-start prior."
    : "This preseason rating uses the published replacement-level cold-start prior.";
  const additiveProfileBreakdown = player.additive_profile_breakdown ?? [];
  const additiveProfileValueByFeature = new Map(
    additiveProfileBreakdown.map((component) => [component.feature, component.player_value]),
  );
  const additiveProfileEffectByFeature = new Map(
    additiveProfileBreakdown.map((component) => [component.feature, component.contribution]),
  );
  const additiveProfileValue = (feature: string, fallback: number | null) => (
    additiveProfileValueByFeature.get(feature) ?? fallback
  );
  const additiveProfileTiles = [
    ["three_pa_per_100", "3PA / 100", player.three_pa_per_100],
    ["three_pm_per_100", "3PM / 100", player.three_pm_per_100],
    ["assists_per_100", "Assists / 100", player.assists_per_100],
    ["turnovers_per_100", "Turnovers / 100", player.turnovers_per_100],
    ["usage_per_100", "Usage / 100", player.usage_per_100],
    ["steals_per_100", "Steals / 100", player.steals_per_100],
    ["blocks_per_100", "Blocks / 100", player.blocks_per_100],
    ["offensive_rebound_claim_total", "OREB claim %", player.offensive_rebound_pct],
  ] as const;

  return (
    <article className="player-profile-page" aria-labelledby="player-profile-title">
      <section className="player-profile-hero">
        <PlayerHeadshot player={player} />
        <div>
          <p className="eyebrow">{player.team} · {player.position} · Age {player.age === null ? "-" : number.format(player.age)}</p>
          <h1 id="player-profile-title">{player.player_name}</h1>
          <p className="player-profile-meta">{draftSummary(player)} · Rookie season {player.rookie_season ?? "-"}</p>
        </div>
        <div className="profile-hero-rating">
          <span>{ratingSeasonLabel}</span>
          <Rating value={player.rapm} />
        </div>
      </section>

      {isPreseasonPreview && <section className="player-profile-section preseason-forecast" aria-labelledby="preseason-forecast-title">
        <div className="player-profile-heading">
          <p className="section-kicker">Frozen before 2026-27 play</p>
          <h2 id="preseason-forecast-title">2026-27 forecast.</h2>
        </div>
        <dl className="player-stat-grid player-forecast-grid">
          <div>
            <dt>Forward prior</dt>
            <dd>{player.prior_rating === null || player.prior_rating === undefined ? "-" : <Rating value={player.prior_rating} />}</dd>
          </div>
          <div>
            <dt>Lagged additive profile</dt>
            <dd>{player.additive_profile_adjustment === null || player.additive_profile_adjustment === undefined ? "-" : <Rating value={player.additive_profile_adjustment} />}</dd>
          </div>
          <div className="player-forecast-total">
            <dt>{MODEL_LABEL} forecast</dt>
            <dd><Rating value={player.rapm} /></dd>
          </div>
        </dl>
        <p className="player-rating-path-note">The prior is a forward value-conditioned aging estimate. The additive profile uses lagged 2025-26 per-100 possession traits under frozen coefficients, not 2026-27 production.</p>
      </section>}

      {hasAdditiveProfileInputs && <section className="player-profile-section" aria-labelledby="profile-rates-title">
        <div className="player-profile-heading">
          <p className="section-kicker">Lagged player profile</p>
          <h2 id="profile-rates-title">Additive profile inputs.</h2>
        </div>
        <dl className="player-stat-grid player-additive-profile-grid">
          {additiveProfileTiles.map(([feature, label, fallback]) => {
            const effect = additiveProfileEffectByFeature.get(feature);
            return <div key={feature}>
              <dt>{label}</dt>
              <dd>
                <span>{number.format(additiveProfileValue(feature, fallback)!)}</span>
                {effect !== undefined && <Rating value={effect} className="additive-profile-effect" />}
              </dd>
            </div>;
          })}
        </dl>
        {additiveProfileBreakdown.length > 0 &&
          <p className="player-rating-path-note additive-profile-reference-note">
            Smaller signed values are each input&apos;s additive contribution relative to the shared possession-weighted 2025-26 forecast-pool reference.
          </p>
        }
      </section>}

      {historyRows.length === 0 ? (
        <section className="player-profile-section" aria-labelledby="rating-history-title">
          <div className="player-profile-heading">
            <p className="section-kicker">Preseason state</p>
            <h2 id="rating-history-title">No completed-fit history yet.</h2>
          </div>
          <p className="player-rating-path-note">{coldStartDescription}</p>
        </section>
      ) : <section className="player-profile-section" aria-labelledby="rating-history-title">
        <div className="player-profile-heading">
          <p className="section-kicker">Completed fits</p>
          <h2 id="rating-history-title">{MODEL_LABEL} history.</h2>
        </div>
        <PlayerAgingChart player={player} />
        <p className="player-rating-path-note">Non-Additive Lineup Edge is the possession-weighted residual non-additive edge of a player’s regular-season units. It is shared unit exposure, not individual causal credit.</p>
        <div className="player-history-table-wrap">
          <table className="player-history-table">
            <thead><tr><th>Season</th><th>Team split</th><th>Age</th><th>GP</th><th>GS</th><th>Possessions</th><th className="nail-history-column">{MODEL_LABEL}</th><th>NAIL rank</th><th>Prior</th><th>Season update</th><th>Additive profile</th><th>Non-Additive Lineup Edge</th></tr></thead>
            <tbody>{[...historyRows].reverse().map((row) => row.kind === "dnp" ? (
              <tr className="player-history-dnp" key={row.season}>
                <td>{row.season}</td><td><span className="dnp-label">DNP</span></td><td>{row.age === null ? "-" : number.format(row.age)}</td>
                <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
              </tr>
            ) : (
              <tr key={row.point.season}>
                <td>{row.point.season}</td><td><TeamSplits point={row.point} /></td><td>{row.point.age === null ? "-" : number.format(row.point.age)}</td>
                <td className="quantity-cell">{wholeNumber.format(row.point.games)}</td><td className="quantity-cell">{wholeNumber.format(row.point.games_started)}</td>
                <td className="quantity-cell">{wholeNumber.format(row.point.possessions)}</td>
                <td className="nail-history-cell"><Rating value={row.point.rating} /></td>
                <td className="nail-rank-cell">#{wholeNumber.format(row.point.nail_rank)}</td>
                <td>{row.point.prior_rating === null ? "-" : <Rating value={row.point.prior_rating} />}</td>
                <td>{row.point.season_update === null ? "-" : <Rating value={row.point.season_update} />}</td>
                <td>{row.point.additive_profile_adjustment === null ? "-" : <Rating value={row.point.additive_profile_adjustment} />}</td>
                <td>{row.point.observed_context_exposure === null ? "-" : <Rating value={row.point.observed_context_exposure} />}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </section>}
    </article>
  );
}

type PlayerHistoryRow =
  | { kind: "completed"; point: Player["rating_history"][number] }
  | { kind: "dnp"; season: string; age: number | null };

function draftSummary(player: Player): string {
  if (player.is_undrafted === true) {
    return player.draft_class_year === null
      ? "Undrafted"
      : `Undrafted · Entry class ${player.draft_class_year}`;
  }
  const details = [
    player.draft_year === null ? null : `Draft class ${player.draft_year}`,
    player.draft_number === null ? null : `No. ${player.draft_number} overall`,
  ].filter((value): value is string => value !== null);
  return details.length ? details.join(" · ") : "Draft information unavailable";
}

function completePlayerHistory(player: Player): PlayerHistoryRow[] {
  const observed = player.rating_history;
  if (!observed.length) return [];
  const observedBySeason = new Map(observed.map((point) => [point.season, point]));
  const firstObserved = observed[0];
  const firstSeasonYear = Number.parseInt(firstObserved.season.slice(0, 4), 10);
  const seasons = player.league_leader_history?.length
    ? player.league_leader_history.map((leader) => leader.season)
    : observed.map((point) => point.season);
  return seasons.map((season) => {
    const point = observedBySeason.get(season);
    if (point) return { kind: "completed", point };
    const seasonYear = Number.parseInt(season.slice(0, 4), 10);
    return {
      kind: "dnp",
      season,
      age: firstObserved.age === null || !Number.isFinite(seasonYear)
        ? null
        : firstObserved.age + seasonYear - firstSeasonYear,
    };
  });
}

function TeamSplits({ point }: { point: Player["rating_history"][number] }) {
  if (!point.team_splits || point.team_splits.length <= 1) return <>{point.team}</>;
  return (
    <div className="player-team-splits">
      {point.team_splits.map((split) => (
        <span key={split.team_id} className={split.is_latest_team ? "latest" : ""}>
          <strong>{split.team}</strong> {wholeNumber.format(split.possessions)} poss. · {wholeNumber.format(split.games)} GP
          {split.is_latest_team && <em>latest</em>}
        </span>
      ))}
    </div>
  );
}

function PlayerAgingChart({ player }: { player: Player }) {
  const chartRef = useRef<SVGSVGElement>(null);
  const [showLeagueLeaders, setShowLeagueLeaders] = useState(true);
  const [visibleContributionLayers, setVisibleContributionLayers] = useState<Record<AgingContributionKey, boolean>>({
    prior: true,
    seasonUpdate: true,
    additiveProfile: true,
  });
  const [hoveredLeader, setHoveredLeader] = useState<{
    name: string;
    rating: number;
    x: number;
    y: number;
  } | null>(null);
  const [hoveredPlayerPoint, setHoveredPlayerPoint] = useState<{
    season: string;
    rating: number;
    games: number;
    gamesStarted: number;
    possessions: number;
    x: number;
    y: number;
  } | null>(null);
  const points = player.rating_history.filter((point) => point.age !== null);
  if (points.length < 2) return null;
  const seasonStartYear = (season: string) => Number.parseInt(season.slice(0, 4), 10);
  const firstObserved = points[0];
  const firstSeasonYear = seasonStartYear(firstObserved.season);
  const leaderHistory = player.league_leader_history?.length
    ? player.league_leader_history
    : points.map((point) => ({
        season: point.season,
        rating: point.season_max_rating ?? point.rating,
        player_id: point.season_max_player_id,
        player_name: point.season_max_player_name ?? "League leader",
      }));
  const timeline = leaderHistory
    .filter((leader) => Number.isFinite(seasonStartYear(leader.season)))
    .map((leader) => ({
      ...leader,
      age: firstObserved.age! + seasonStartYear(leader.season) - firstSeasonYear,
    }));
  const contributionLayers: Array<{
    key: AgingContributionKey;
    label: string;
    color: string;
  }> = [
    { key: "prior", label: "Prior", color: "#3f77a8" },
    { key: "seasonUpdate", label: "Season update", color: "#d86732" },
    { key: "additiveProfile", label: "Additive profile", color: "#a64d67" },
  ];
  const contributionValues = points.map((point) => ({
    prior: point.prior_rating ?? 0,
    seasonUpdate: point.season_update ?? 0,
    additiveProfile: point.additive_profile_adjustment ?? 0,
  }));
  const positiveTotals = contributionValues.map((values) => contributionLayers.reduce(
    (total, layer) => total + Math.max(0, values[layer.key]), 0,
  ));
  const negativeTotals = contributionValues.map((values) => contributionLayers.reduce(
    (total, layer) => total + Math.min(0, values[layer.key]), 0,
  ));
  const width = 720;
  const height = 266;
  const margin = { top: 38, right: 34, bottom: 54, left: 48 };
  const ages = timeline.map((point) => point.age);
  const ratings = points.map((point) => point.rating);
  const seasonMaxes = timeline.map((point) => point.rating);
  const minAge = Math.min(...ages);
  const maxAge = Math.max(...ages);
  const lowerBound = Math.floor(Math.min(0, ...ratings, ...negativeTotals));
  const upperBound = Math.ceil(Math.max(0, ...ratings, ...seasonMaxes, ...positiveTotals));
  const ratingRange = Math.max(1, upperBound - lowerBound);
  const x = (age: number) => margin.left + ((age - minAge) / Math.max(1, maxAge - minAge)) * (width - margin.left - margin.right);
  const y = (rating: number) => margin.top + ((upperBound - rating) / ratingRange) * (height - margin.top - margin.bottom);
  const zeroY = y(0);
  const path = points.map((point, index) => {
    const previous = points[index - 1];
    const beginsNewSegment = index === 0
      || seasonStartYear(point.season) !== seasonStartYear(previous.season) + 1;
    return `${beginsNewSegment ? "M" : "L"}${x(point.age!)},${y(point.rating)}`;
  }).join(" ");
  function contributionAreaPath(layerIndex: number, polarity: "positive" | "negative") {
    const segments: Array<Array<{ point: Player["rating_history"][number]; lower: number; upper: number }>> = [];
    let segment: Array<{ point: Player["rating_history"][number]; lower: number; upper: number }> = [];
    for (let index = 0; index < points.length; index += 1) {
      const point = points[index];
      const previous = points[index - 1];
      if (index > 0 && seasonStartYear(point.season) !== seasonStartYear(previous.season) + 1) {
        if (segment.length) segments.push(segment);
        segment = [];
      }
      const values = contributionValues[index];
      const priorTotal = contributionLayers.slice(0, layerIndex).reduce(
        (total, layer) => total + (polarity === "positive" ? Math.max(0, values[layer.key]) : Math.min(0, values[layer.key])),
        0,
      );
      const layerValue = polarity === "positive"
        ? Math.max(0, values[contributionLayers[layerIndex].key])
        : Math.min(0, values[contributionLayers[layerIndex].key]);
      segment.push({ point, lower: priorTotal, upper: priorTotal + layerValue });
    }
    if (segment.length) segments.push(segment);
    return segments.map((entries) => {
      const upper = entries.map((entry, index) => `${index ? "L" : "M"}${x(entry.point.age!)},${y(entry.upper)}`).join(" ");
      const lower = [...entries].reverse().map((entry) => `L${x(entry.point.age!)},${y(entry.lower)}`).join(" ");
      return `${upper} ${lower} Z`;
    }).join(" ");
  }
  async function downloadPng() {
    const svg = chartRef.current;
    if (!svg) return;
    const copy = svg.cloneNode(true) as SVGSVGElement;
    copy.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    copy.querySelectorAll("[data-export-exclude]").forEach((element) => element.remove());
    await inlineSvgImages(copy);
    const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
    style.textContent = `
      .aging-chart-grid { stroke: #d9d6ce; stroke-width: 1; }
      .aging-chart-title { fill: #17201c; font-family: sans-serif; font-size: 13px; font-weight: 800; }
      .aging-chart-zero { stroke: #838a83; stroke-width: 1; stroke-dasharray: 5 4; }
      .aging-chart-leader-fallback { fill: #f6f3ec; stroke: #8e968f; stroke-width: 1; }
      .aging-chart-leader-headshot { overflow: visible; }
      .aging-chart-leader-value { fill: #7a827b; font-family: monospace; font-size: 8px; font-weight: 700; }
      .aging-chart-export-legend { display: block; }
      .aging-chart-export-legend-label { fill: #505952; font-family: sans-serif; font-size: 9px; font-weight: 700; }
      .aging-chart-line { fill: none; stroke: #174d3d; stroke-width: 3.5; stroke-linecap: round; stroke-linejoin: round; }
      .aging-chart-point { fill: #e8502f; stroke: #fffefa; stroke-width: 1.5; }
      .aging-chart-point.negative-point { fill: #b33b25; }
      .aging-chart-team-logo { overflow: visible; }
      .aging-chart-player-ring { fill: #fffefa; stroke: #174d3d; stroke-width: 1.8; }
      .aging-chart-y-label, .aging-chart-x-label, .aging-chart-value { fill: #68716a; font-family: monospace; font-size: 11px; }
      .aging-chart-value { fill: #187052; font-size: 10px; font-weight: 700; }
      .aging-chart-value.negative { fill: #b33b25; }
    `;
    copy.prepend(style);
    const image = new Image();
    const svgUrl = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(copy)], { type: "image/svg+xml" }));
    image.onload = () => {
      const canvas = document.createElement("canvas");
      const scale = 2;
      canvas.width = width * scale;
      canvas.height = height * scale;
      const context = canvas.getContext("2d");
      if (!context) return;
      context.fillStyle = "#f6f3ec";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(svgUrl);
      canvas.toBlob((blob) => {
        if (!blob) return;
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `${player.player_name.toLowerCase().replaceAll(/[^a-z0-9]+/g, "-").replaceAll(/^-|-$/g, "")}-nail-rapm-v11-history.png`;
        link.click();
        window.setTimeout(() => URL.revokeObjectURL(link.href), 0);
      }, "image/png");
    };
    image.src = svgUrl;
  }

  function toggleLeagueLeaders() {
    setShowLeagueLeaders((visible) => !visible);
    setHoveredLeader(null);
  }

  function toggleContributionLayer(layer: AgingContributionKey) {
    setVisibleContributionLayers((visible) => ({ ...visible, [layer]: !visible[layer] }));
  }

  return (
    <figure className="player-aging-chart">
      <div className="player-aging-chart-toolbar">
        <span>History</span>
        <div className="player-aging-chart-actions">
          <div className="player-aging-chart-layer-controls" role="group" aria-label="Visible player-rating components">
            {contributionLayers.map((layer) => (
              <button
                className="chart-component-toggle"
                key={layer.key}
                type="button"
                role="switch"
                aria-checked={visibleContributionLayers[layer.key]}
                onClick={() => toggleContributionLayer(layer.key)}
                title={`${visibleContributionLayers[layer.key] ? "Hide" : "Show"} ${layer.label}`}
              >
                <span className="chart-component-swatch" style={{ backgroundColor: layer.color }} aria-hidden="true" />
                <span>{layer.label}</span>
              </button>
            ))}
          </div>
          <button
            className="chart-component-toggle"
            type="button"
            role="switch"
            onClick={toggleLeagueLeaders}
            title={showLeagueLeaders ? "Hide league leaders" : "Show league leaders"}
            aria-label={showLeagueLeaders ? "Hide league leaders" : "Show league leaders"}
            aria-checked={showLeagueLeaders}
          >
            <span className="chart-component-swatch" style={{ backgroundColor: "#174d3d" }} aria-hidden="true" />
            <span>League maxima</span>
          </button>
          <button type="button" onClick={downloadPng} title="Download chart as PNG" aria-label={`Download ${player.player_name} rating history as PNG`}><Download size={14} /> Download PNG</button>
        </div>
      </div>
      <svg ref={chartRef} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${player.player_name} completed-fit ${MODEL_LABEL} history by age`}>
        <text className="aging-chart-title" x={margin.left} y="17">{player.player_name} · {MODEL_LABEL} history</text>
        <g className="aging-chart-export-legend" aria-label="Rating decomposition legend">
          {contributionLayers.filter((layer) => visibleContributionLayers[layer.key]).map((layer, index) => {
            const legendX = margin.left + index * 146;
            return <g key={layer.key} transform={`translate(${legendX}, 31)`}>
              <rect x="0" y="-8" width="10" height="10" rx="1" fill={layer.color} fillOpacity="0.8" />
              <text className="aging-chart-export-legend-label" x="15" y="0">{layer.label}</text>
            </g>;
          })}
        </g>
        <line className="aging-chart-grid" x1={margin.left} x2={width - margin.right} y1={margin.top} y2={margin.top} />
        <line className="aging-chart-zero" x1={margin.left} x2={width - margin.right} y1={zeroY} y2={zeroY} />
        <line className="aging-chart-grid" x1={margin.left} x2={width - margin.right} y1={height - margin.bottom} y2={height - margin.bottom} />
        <text className="aging-chart-y-label" x={margin.left - 10} y={margin.top + 4} textAnchor="end">{formatChartRating(upperBound)}</text>
        <text className="aging-chart-y-label" x={margin.left - 10} y={zeroY + 4} textAnchor="end">0.0</text>
        <text className="aging-chart-y-label" x={margin.left - 10} y={height - margin.bottom + 4} textAnchor="end">{formatChartRating(lowerBound)}</text>
        {contributionLayers.map((layer, index) => visibleContributionLayers[layer.key] && <g key={layer.key}>
          <path
            className="aging-chart-contribution-area"
            d={contributionAreaPath(index, "positive")}
            fill={layer.color}
            fillOpacity="0.38"
            stroke={layer.color}
            strokeOpacity="0.55"
            strokeWidth="0.65"
          />
          <path
            className="aging-chart-contribution-area"
            d={contributionAreaPath(index, "negative")}
            fill={layer.color}
            fillOpacity="0.38"
            stroke={layer.color}
            strokeOpacity="0.55"
            strokeWidth="0.65"
          />
        </g>)}
        {showLeagueLeaders && timeline.map((point) => {
          const leaderRating = point.rating;
          const leaderId = point.player_id;
          const leaderName = point.player_name;
          return <g key={`leader-${point.season}`}>
            {leaderId !== undefined && (
              <a
                href={playerProfileHref(leaderId)}
                aria-label={`${leaderName}, ${formatChartRating(leaderRating)} ${MODEL_LABEL}`}
                onMouseEnter={() => setHoveredLeader({ name: leaderName, rating: leaderRating, x: x(point.age), y: y(leaderRating) })}
                onMouseLeave={() => setHoveredLeader(null)}
                onFocus={() => setHoveredLeader({ name: leaderName, rating: leaderRating, x: x(point.age), y: y(leaderRating) })}
                onBlur={() => setHoveredLeader(null)}
              >
                <title>{leaderName} · {formatChartRating(leaderRating)} {MODEL_LABEL}</title>
                <circle className="aging-chart-leader-fallback" cx={x(point.age)} cy={y(leaderRating)} r="10" />
                <image
                  className="aging-chart-leader-headshot"
                  href={playerHeadshotUrl(leaderId)}
                  x={x(point.age) - 9}
                  y={y(leaderRating) - 9}
                  width="18"
                  height="18"
                  preserveAspectRatio="xMidYMid slice"
                  onError={(event) => { event.currentTarget.style.display = "none"; }}
                />
              </a>
            )}
            <text className="aging-chart-leader-value" x={x(point.age)} y={Math.max(margin.top - 8, y(leaderRating) - 14)} textAnchor="middle" pointerEvents="none">{formatChartRating(leaderRating)}</text>
          </g>;
        })}
        <path className="aging-chart-line" d={path} />
        {points.map((point) => {
          const pointX = x(point.age!);
          const pointY = y(point.rating);
          return <g
            key={point.season}
            className="aging-chart-player-point"
            onMouseEnter={() => setHoveredPlayerPoint({
              season: point.season,
              rating: point.rating,
              games: point.games,
              gamesStarted: point.games_started,
              possessions: point.possessions,
              x: pointX,
              y: pointY,
            })}
            onMouseLeave={() => setHoveredPlayerPoint(null)}
          >
            {point.team !== "-" && <circle className="aging-chart-player-ring" cx={pointX} cy={pointY} r="9" />}
            <circle className={point.rating < 0 ? "aging-chart-point negative-point" : "aging-chart-point"} cx={pointX} cy={pointY} r="4.5" />
            {point.team !== "-" && (
              <image
                className="aging-chart-team-logo"
                href={teamLogoUrl(point.team)}
                x={pointX - 8}
                y={pointY - 8}
                width="16"
                height="16"
                preserveAspectRatio="xMidYMid meet"
                crossOrigin="anonymous"
                onError={(event) => { event.currentTarget.style.display = "none"; }}
              >
                <title>{point.team}</title>
              </image>
            )}
            <text className={point.rating < 0 ? "aging-chart-value negative" : "aging-chart-value"} x={pointX} y={pointY + 17} textAnchor="middle">{formatChartRating(point.rating)}</text>
          </g>;
        })}
        {timeline.map((point) => <g key={`age-${point.season}`}>
          <line className="aging-chart-tick" x1={x(point.age)} x2={x(point.age)} y1={height - margin.bottom} y2={height - margin.bottom + 5} />
          <text className="aging-chart-x-label" x={x(point.age)} y={height - 27} textAnchor="middle">
            <tspan x={x(point.age)}>{point.season.slice(-2)}</tspan>
            <tspan x={x(point.age)} dy="11">{number.format(point.age)}</tspan>
          </text>
        </g>)}
        {showLeagueLeaders && hoveredLeader && (
          <g
            className="aging-chart-leader-tooltip"
            data-export-exclude="true"
            pointerEvents="none"
            transform={`translate(${Math.min(width - 146, hoveredLeader.x + 14)}, ${Math.max(margin.top + 4, hoveredLeader.y - 33)})`}
          >
            <rect width="132" height="29" rx="3" />
            <text x="7" y="12">{hoveredLeader.name}</text>
            <text x="7" y="23">{formatChartRating(hoveredLeader.rating)} {MODEL_LABEL}</text>
          </g>
        )}
        {hoveredPlayerPoint && (
          <g
            className="aging-chart-player-tooltip"
            data-export-exclude="true"
            pointerEvents="none"
            transform={`translate(${Math.min(width - 154, hoveredPlayerPoint.x + 11)}, ${Math.max(margin.top + 3, hoveredPlayerPoint.y - 49)})`}
          >
            <rect width="144" height="43" rx="3" />
            <text x="7" y="12">{hoveredPlayerPoint.season} · {formatChartRating(hoveredPlayerPoint.rating)} {MODEL_LABEL}</text>
            <text x="7" y="24">G {wholeNumber.format(hoveredPlayerPoint.games)} · GS {wholeNumber.format(hoveredPlayerPoint.gamesStarted)}</text>
            <text x="7" y="36">{wholeNumber.format(hoveredPlayerPoint.possessions)} possessions</text>
          </g>
        )}
      </svg>
      <figcaption>{showLeagueLeaders
        ? `Completed-fit ${MODEL_LABEL} by age. The colored areas are signed player-rating components stacked around zero; the green line is their sum. Headshots mark each season's league leader.`
        : `Completed-fit ${MODEL_LABEL} by age. The colored areas are signed player-rating components stacked around zero; the green line is their sum.`}
      </figcaption>
    </figure>
  );
}

async function inlineSvgImages(svg: SVGSVGElement) {
  const images = Array.from(svg.querySelectorAll("image[href]"));
  await Promise.all(images.map(async (image) => {
    const source = image.getAttribute("href");
    if (!source || (!source.startsWith("https://") && !source.startsWith("/"))) return;
    try {
      const response = await fetch(source, { mode: "cors" });
      if (!response.ok) throw new Error(`Logo request failed: ${response.status}`);
      const blob = await response.blob();
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);
      });
      image.setAttribute("href", dataUrl);
    } catch {
      image.remove();
    }
  }));
}

function App() {
  const route = useAppView();
  const view = route.view;
  const [unit, setUnit] = useState<Player[]>([]);
  const [opponent, setOpponent] = useState<Player[]>([]);
  const [unitSeason, setUnitSeason] = useState("2025-26");
  const [opponentSeason, setOpponentSeason] = useState("2025-26");
  const [unitTeam, setUnitTeam] = useState("all");
  const [opponentTeam, setOpponentTeam] = useState("all");
  const [availableLabSeasons, setAvailableLabSeasons] = useState<string[]>(["2025-26"]);
  const [environment, setEnvironment] = useState<Environment>("unit");
  const [result, setResult] = useState<Matchup | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [isLoadingUnit, setIsLoadingUnit] = useState(false);
  const [isLoadingOpponent, setIsLoadingOpponent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canEvaluate = unit.length === 5 && opponent.length === 5;
  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        const response = await fetch("/api/health", { signal: controller.signal });
        if (!response.ok) throw new Error("Historical seasons are unavailable.");
        const payload = (await response.json()) as { lab_seasons?: string[] };
        if (payload.lab_seasons?.length) setAvailableLabSeasons(payload.lab_seasons);
      } catch (healthError) {
        if ((healthError as Error).name !== "AbortError") setError((healthError as Error).message);
      }
    })();
    return () => controller.abort();
  }, []);

  async function loadRandomLineup(side: Side) {
    try {
      setError(null);
      const setLoading = side === "unit" ? setIsLoadingUnit : setIsLoadingOpponent;
      setLoading(true);
      const parameters = new URLSearchParams();
      const currentPlayers = side === "unit" ? unit : opponent;
      parameters.set("season", side === "unit" ? unitSeason : opponentSeason);
      const selectedTeam = side === "unit" ? unitTeam : opponentTeam;
      if (selectedTeam !== "all") parameters.set("team", selectedTeam);
      parameters.set("count", String(5 - currentPlayers.length));
      currentPlayers.forEach((player) => {
        parameters.append("exclude_player_id", String(player.player_id));
      });
      const response = await fetch(`/api/default-opponent?${parameters.toString()}`);
      if (!response.ok) throw new Error("Random lineup is unavailable.");
      const payload = (await response.json()) as { players: Player[] };
      if (side === "unit") {
        setUnit((current) => [...current, ...payload.players].slice(0, 5));
      } else {
        setOpponent((current) => [...current, ...payload.players].slice(0, 5));
      }
      setResult(null);
    } catch (opponentError) {
      setError((opponentError as Error).message);
    } finally {
      if (side === "unit") {
        setIsLoadingUnit(false);
      } else {
        setIsLoadingOpponent(false);
      }
    }
  }

  function removePlayer(side: Side, playerId: number) {
    const setter = side === "unit" ? setUnit : setOpponent;
    setter((current) => current.filter((player) => player.player_id !== playerId));
    setResult(null);
  }

  function clearLineup(side: Side) {
    const setter = side === "unit" ? setUnit : setOpponent;
    setter([]);
    setResult(null);
  }

  async function loadObservedLineup(side: Side, lineup: RankedLineup, season: string) {
    try {
      setError(null);
      const setLoading = side === "unit" ? setIsLoadingUnit : setIsLoadingOpponent;
      setLoading(true);
      const parameters = new URLSearchParams({ season });
      lineup.player_ids.forEach((playerId) => parameters.append("player_id", String(playerId)));
      const response = await fetch(`/api/players/by-id?${parameters.toString()}`);
      if (!response.ok) throw new Error("This lineup is unavailable in the Matchup Lab.");
      const payload = (await response.json()) as { players: Player[] };
      if (payload.players.length !== 5) throw new Error("This lineup is incomplete in the Matchup Lab.");
      if (side === "unit") {
        setUnitSeason(season);
        setUnitTeam(lineup.team);
        setUnit(payload.players);
      } else {
        setOpponentSeason(season);
        setOpponentTeam(lineup.team);
        setOpponent(payload.players);
      }
      setResult(null);
      window.location.hash = "#lab";
    } catch (lineupError) {
      setError((lineupError as Error).message);
    } finally {
      if (side === "unit") {
        setIsLoadingUnit(false);
      } else {
        setIsLoadingOpponent(false);
      }
    }
  }

  async function evaluate() {
    if (!canEvaluate) return;
    try {
      setError(null);
      setIsEvaluating(true);
      const response = await fetch("/api/matchups", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          unit_player_ids: unit.map((player) => player.player_id),
          opponent_player_ids: opponent.map((player) => player.player_id),
          unit_season: unitSeason,
          opponent_season: opponentSeason,
          environment,
          include_response_curves: true,
        }),
      });
      const payload = (await response.json()) as Matchup | { detail: string };
      if (!response.ok) throw new Error("detail" in payload ? payload.detail : "Lineup evaluation failed.");
      const matchup = payload as Matchup;
      setResult(matchup);
    } catch (evaluationError) {
      setError((evaluationError as Error).message);
    } finally {
      setIsEvaluating(false);
    }
  }

  return (
    <main>
      <header className="site-header">
        <a className="wordmark" href="#lab" aria-label="NBA GESTALT Matchup Lab">
          <span>NBA</span> GESTALT
        </a>
        <div className="header-meta">
          <nav className="header-navigation" aria-label="Primary navigation">
            <a className={view === "lab" ? "active" : ""} href="#lab">Lab</a>
            <a className={view === "rankings" ? "active" : ""} href="#rankings">Rankings</a>
            <a className={view === "lineups" ? "active" : ""} href="#lineups">Lineups</a>
            <a className={view === "about" ? "active" : ""} href="#about">About</a>
          </nav>
          <span className="header-links">
            <a
              className="header-link"
              href="https://github.com/EvanZ/nba-lineup-model"
              target="_blank"
              rel="noreferrer"
              aria-label="Open the NBA GESTALT GitHub repository"
            >
              <GitBranch size={15} aria-hidden="true" />
              <span>GitHub</span>
            </a>
            <a
              className="header-link"
              href="https://evanz.github.io/nba-lineup-model/"
              target="_blank"
              rel="noreferrer"
              aria-label="Open NBA GESTALT documentation"
            >
              <BookOpen size={15} aria-hidden="true" />
              <span>Docs</span>
            </a>
          </span>
          <span className="season-chip">2025-26 retrospective</span>
        </div>
      </header>

      {view === "about" ? <AboutPage /> : view === "rankings" ? <RankingsPage /> : view === "lineups" ? <LineupRankingsPage onLoadInLab={loadObservedLineup} /> : view === "player" && route.playerId ? <PlayerProfilePage playerId={route.playerId} /> : <>
        <section className="intro" aria-labelledby="page-title">
          <div className="gestalt-entry" aria-label="Definition of gestalt">
            <div className="gestalt-entry-heading">
              <span className="gestalt-word">gestalt</span>
              <span className="gestalt-part-of-speech">noun</span>
            </div>
            <p className="gestalt-pronunciation"><span>German</span> /gə-ˈshtält/</p>
            <p className="gestalt-definition">A whole whose properties emerge from the relationships among its parts.</p>
            <p className="gestalt-basketball-note">In basketball, five players can create an edge that no sum of individual ratings fully explains.</p>
          </div>
          <div className="intro-prompt">
            <p className="eyebrow">Matchup Lab</p>
            <h1 id="page-title">Build the five.</h1>
          </div>
        </section>

        <section className="lab-grid" aria-label="Lineup builder">
          <div className="builder-panel">
            <div className="lineup-columns">
              <LineupSelector
                side="unit"
                players={unit}
                season={unitSeason}
                availableSeasons={availableLabSeasons}
                team={unitTeam}
                isLoading={isLoadingUnit}
                onRefresh={() => void loadRandomLineup("unit")}
                onSeasonChange={(season) => {
                  setUnitSeason(season);
                  setUnitTeam("all");
                  setUnit([]);
                  setResult(null);
                }}
                onTeamChange={setUnitTeam}
                onAdd={(player) => {
                  setUnit((current) => [...current, player]);
                  setResult(null);
                }}
                onRemove={(playerId) => removePlayer("unit", playerId)}
                onClear={() => clearLineup("unit")}
                onError={setError}
              />
              <LineupSelector
                side="opponent"
                players={opponent}
                season={opponentSeason}
                availableSeasons={availableLabSeasons}
                team={opponentTeam}
                isLoading={isLoadingOpponent}
                onRefresh={() => void loadRandomLineup("opponent")}
                onSeasonChange={(season) => {
                  setOpponentSeason(season);
                  setOpponentTeam("all");
                  setOpponent([]);
                  setResult(null);
                }}
                onTeamChange={setOpponentTeam}
                onAdd={(player) => {
                  setOpponent((current) => [...current, player]);
                  setResult(null);
                }}
                onRemove={(playerId) => removePlayer("opponent", playerId)}
                onClear={() => clearLineup("opponent")}
                onError={setError}
              />
            </div>

            <section className="environment-control" aria-label="Evaluation era">
              <span>Evaluation era</span>
              <div className="environment-toggle" role="radiogroup" aria-label="Evaluation era">
                {([
                  ["unit", "Your unit"],
                  ["neutral", "Neutral"],
                  ["opponent", "Opponent unit"],
                ] as const).map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    role="radio"
                    aria-checked={environment === value}
                    className={environment === value ? "active" : ""}
                    disabled={unitSeason === opponentSeason}
                    title={unitSeason === opponentSeason ? "Evaluation era applies only to mixed-season matchups." : undefined}
                    onClick={() => { setEnvironment(value); setResult(null); }}
                  >{label}</button>
                ))}
              </div>
            </section>

            <button className="evaluate-button" onClick={() => void evaluate()} disabled={!canEvaluate || isEvaluating}>
              {isEvaluating ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}
              {isEvaluating ? "Evaluating" : "Evaluate matchup"}
              {!isEvaluating && <ArrowUpRight size={17} />}
            </button>
            {!canEvaluate && <p className="builder-note">Choose five players for each side to evaluate the matchup.</p>}
            {error && <p className="error"><CircleAlert size={16} /> {error}</p>}
          </div>

          <Results result={result} />
        </section>
      </>}
    </main>
  );
}

function AboutPage() {
  return (
    <article className="about-page" aria-labelledby="about-title">
      <section className="about-hero">
        <p className="eyebrow">NBA GESTALT methodology</p>
        <h1 id="about-title">A player model, then a lineup residual.</h1>
        <p className="about-lede">
          {MODEL_LABEL} estimates a neutral-court edge per 100 possessions. It separates what can be
          assigned to individual players from what belongs to the particular five-player combination.
        </p>
      </section>

      <section className="model-identity" aria-labelledby="model-name-title">
        <div>
          <p className="section-kicker">Current model</p>
          <h2 id="model-name-title">{MODEL_LABEL}</h2>
        </div>
        <p className="model-identity-name">Forward, age-informed, residual-lineup RAPM</p>
        <p>
          <b>N</b>on-<b>A</b>dditive <b>I</b>nteractions in <b>L</b>ineups,
          Regularized Adjusted Plus-Minus.
        </p>
      </section>

      <section className="about-equation" aria-labelledby="equation-title">
        <div>
          <p className="section-kicker">Stint model</p>
          <h2 id="equation-title">What one row means.</h2>
        </div>
        <div className="about-formula-stack">
          <div className="model-formula">
            <BlockMath math={String.raw`\hat{y}_s = b_{\mathrm{home}} + \sum_{i \in H_s} r_{i,t} - \sum_{j \in A_s} r_{j,t} + C_t(H_s, A_s) + \varepsilon_s`} />
          </div>
          <div className="model-subformula">
            <BlockMath math={String.raw`C_t(H,A) = C_{\mathrm{add},t}(H,A) + C_{\mathrm{nonadd},t}(H,A)`} />
          </div>
        </div>
        <p>
          Each row is a possession-weighted stint. <i>y_s</i> is its home-minus-away net-rating outcome,
          <i> b_home</i> is home court, and the two five-player sums are the raw player edge. <i>C</i> is
          the lineup-profile correction. The model estimates all reported ratings on a per-100-possession scale.
        </p>
      </section>

      <section className="about-section" aria-labelledby="forward-title">
        <div className="about-section-heading">
          <p className="section-kicker">Recursive fitting loop</p>
          <h2 id="forward-title">Each season starts before it happens.</h2>
          <p>
            For target season <i>t</i>, every starting quantity is constructed from seasons before <i>t</i>.
            Once season <i>t</i> is complete, its fitted state can inform season <i>t+1</i>, never itself.
          </p>
        </div>
        <div className="about-steps">
          <section>
            <span>01</span>
            <h3>Build the player prior</h3>
            <p>
              Returning players receive a centered, value-conditioned aging forecast from their completed
              history. The age model uses prior player state, age, experience, and prior exposure.
            </p>
          </section>
          <section>
            <span>02</span>
            <h3>Handle cold starts</h3>
            <p>
              A player without a usable prior season receives a draft-profile forecast blended with a
              replacement-level token. The exposure gate shifts the blend toward replacement when projected
              opportunity is low.
            </p>
          </section>
          <section>
            <span>03</span>
            <h3>Carry context forward</h3>
            <p>
              The prior season's lineup correction is subtracted from the current stint target before RAPM
              refits player coefficients. That prevents repeatable lineup composition from being absorbed
              wholesale into player ratings.
            </p>
          </section>
        </div>
      </section>

      <section className="about-section" aria-labelledby="rapm-title">
        <div className="about-section-heading">
          <p className="section-kicker">Player update</p>
          <h2 id="rapm-title">RAPM estimates the part that travels with a player.</h2>
          <p>
            The current season then updates the prior from possession outcomes, with the other nine players
            and home court in the same regression. Ridge regularization anchors uncertain estimates to the
            preseason prior rather than treating a small sample as a new truth.
          </p>
        </div>
        <div className="about-formula-callout">
          <div className="model-formula">
            <BlockMath math={String.raw`r^{\mathrm{raw}}_{i,t} = \mu_{i,t} + \Delta r_{i,t}`} />
          </div>
          <p>
            This is the raw player state before profile attribution. A player&apos;s update is identified by how
            the team performs across the possession-weighted stints in which that player appears, conditional
            on teammates and opponents. It is not a box-score total or a causal estimate of every action.
          </p>
        </div>
      </section>

      <section className="about-section" aria-labelledby="context-title">
        <div className="about-section-heading">
          <p className="section-kicker">Residual lineup model</p>
          <h2 id="context-title">Context is fit to what player RAPM leaves behind.</h2>
          <p>
            After the raw player edge is removed from each completed stint, a second standardized Ridge
            regression fits the residual against a fixed, strictly lagged five-player feature contract.
            The v1.2 model is linear in those features; it does not use splines or a black-box interaction network.
          </p>
        </div>
        <div className="feature-contract">
          <section>
            <p className="section-kicker">Player-compilable features</p>
            <h3>Eight additive profile coordinates.</h3>
            <p>
              These are sums of player profiles, so their fitted credit can be assigned exactly back to individuals
              without changing a lineup forecast.
            </p>
            <ul className="feature-list">
              <li>Three-point attempts and makes</li>
              <li>Assists, turnovers, and usage</li>
              <li>Steals and blocks</li>
              <li>Offensive-rebound claim total</li>
            </ul>
          </section>
          <section>
            <p className="section-kicker">Lineup-only features</p>
            <h3>Six non-additive shape coordinates.</h3>
            <p>
              These deliberately depend on the group rather than independent player totals, so they remain as a
              unit-level edge instead of being allocated to any one player.
            </p>
            <ul className="feature-list">
              <li>Bottom-two shooting and credible-shooter count</li>
              <li>Top-two assists and usage concentration</li>
              <li>Shooting-by-usage and shooter-by-passing</li>
            </ul>
          </section>
        </div>
      </section>

      <section className="about-section" aria-labelledby="attribution-title">
        <div className="about-section-heading">
          <p className="section-kicker">Attribution contract</p>
          <h2 id="attribution-title">The published player rating excludes residual lineup credit.</h2>
        </div>
        <div className="about-formula-callout">
          <div className="model-formula">
            <BlockMath math={String.raw`R^{\mathrm{NAIL}}_{i,t} = \mu_{i,t} + \Delta r_{i,t} + \delta^{\mathrm{profile}}_{i,t}`} />
          </div>
          <div className="model-subformula">
            <BlockMath math={String.raw`\operatorname{Edge}(U,O) = \operatorname{Player}(U) - \operatorname{Player}(O) + C_{\mathrm{nonadd},t}(U,O)`} />
          </div>
          <p>
            The eight additive features compile exactly into the player rating. The six non-additive features remain
            a separate unit-level term. In the Matchup Lab, the ledger shows both pieces. In player pages,
            <i> Non-Additive Lineup Edge</i> is a possession-weighted description of the units a player actually
            shared, not extra individual credit and not a causal allocation.
          </p>
        </div>
      </section>

      <section className="about-footer-note">
        <p className="section-kicker">Current publication</p>
        <p>
          The Matchup Lab serves the completed 2025-26 NAIL-RAPM v1.2 state with published statistic-specific
          profile shrinkage and gap-returner profile carry-forward. It is useful for retrospective
          lineup exploration and as the state carried forward into a forecast; it is not a live in-season update
          or a claim that the residual lineup effect belongs causally to a single player.
        </p>
      </section>
    </article>
  );
}

type RankingColumn = "rank" | "player_name" | "team" | "position" | "draft_number" | "rapm" | "prior_rating" | "season_update" | "additive_profile_adjustment" | "observed_context_exposure" | "possessions" | "games";

function RankingsPage() {
  const isCompact = useMediaQuery("(max-width: 720px)");
  const [players, setPlayers] = useState<RankedPlayer[]>([]);
  const [selectedSeason, setSelectedSeason] = useState("2025-26");
  const [availableSeasons, setAvailableSeasons] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [minimumPossessions, setMinimumPossessions] = useState(500);
  const [teamFilter, setTeamFilter] = useState("all");
  const [draftClassFilter, setDraftClassFilter] = useState("all");
  const [sortColumn, setSortColumn] = useState<RankingColumn>("rank");
  const [sortDirection, setSortDirection] = useState<"ascending" | "descending">("ascending");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        setIsLoading(true);
        const response = await fetch(`/api/rankings?season=${encodeURIComponent(selectedSeason)}`, { signal: controller.signal });
        if (!response.ok) throw new Error(`${MODEL_LABEL} rankings are unavailable.`);
        const payload = (await response.json()) as {
          available_seasons: string[];
          players: RankedPlayer[];
        };
        setPlayers(payload.players);
        setAvailableSeasons(payload.available_seasons);
      } catch (rankingError) {
        if ((rankingError as Error).name !== "AbortError") setError((rankingError as Error).message);
      } finally {
        if (!controller.signal.aborted) setIsLoading(false);
      }
    })();
    return () => controller.abort();
  }, [selectedSeason]);

  useEffect(() => {
    setTeamFilter("all");
    setDraftClassFilter("all");
    setMinimumPossessions(selectedSeason === "2026-27" ? 0 : 500);
  }, [selectedSeason]);

  const isPreseasonPreview = selectedSeason === "2026-27";

  const draftClasses = useMemo(() => [...new Set(
    players.flatMap((player) => player.draft_class_year === null ? [] : [player.draft_class_year]),
  )].sort((left, right) => right - left), [players]);
  const teams = useMemo(() => [...new Set(players.map((player) => player.team))].sort(), [players]);

  const visiblePlayers = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    const rows = (normalized
      ? players.filter((player) => player.player_name.toLocaleLowerCase().includes(normalized))
      : players)
      .filter((player) => player.possessions >= minimumPossessions)
      .filter((player) => teamFilter === "all" || player.team === teamFilter)
      .filter((player) => {
        if (draftClassFilter === "all") return true;
        if (draftClassFilter === "undrafted") return player.is_undrafted === true;
        return player.draft_class_year === Number(draftClassFilter);
      });
    const direction = sortDirection === "ascending" ? 1 : -1;
    return [...rows].sort((left, right) => {
      if (sortColumn === "draft_number") {
        const leftGroup = draftPickGroup(left);
        const rightGroup = draftPickGroup(right);
        if (leftGroup !== rightGroup) return leftGroup - rightGroup;
        if (leftGroup !== 0) return left.rank - right.rank;
        return direction * ((left.draft_number ?? 0) - (right.draft_number ?? 0)) || left.rank - right.rank;
      }
      const leftValue = left[sortColumn];
      const rightValue = right[sortColumn];
      if (leftValue === null || leftValue === undefined) return 1;
      if (rightValue === null || rightValue === undefined) return -1;
      if (typeof leftValue === "number" && typeof rightValue === "number") {
        return direction * (leftValue - rightValue) || left.rank - right.rank;
      }
      return direction * String(leftValue).localeCompare(String(rightValue)) || left.rank - right.rank;
    });
  }, [draftClassFilter, minimumPossessions, players, query, sortColumn, sortDirection, teamFilter]);

  function changeSort(column: RankingColumn) {
    if (column === sortColumn) {
      setSortDirection((direction) => direction === "ascending" ? "descending" : "ascending");
      return;
    }
    setSortColumn(column);
    setSortDirection(column === "player_name" || column === "team" || column === "position" ? "ascending" : "descending");
  }

  const columns: Array<{ key: RankingColumn; label: string; numeric?: boolean }> = [
    { key: "rank", label: "Rank", numeric: true },
    { key: "player_name", label: "Player" },
    { key: "team", label: "Team" },
    { key: "position", label: "Pos" },
    { key: "draft_number", label: "Pick", numeric: true },
    { key: "rapm", label: MODEL_LABEL, numeric: true },
    { key: "prior_rating", label: "Prior", numeric: true },
    { key: "season_update", label: "Season update", numeric: true },
    { key: "additive_profile_adjustment", label: "Additive profile", numeric: true },
    { key: "observed_context_exposure", label: "Non-Additive Lineup Edge", numeric: true },
    { key: "possessions", label: "Possessions", numeric: true },
    { key: "games", label: "Games", numeric: true },
  ];

  return (
    <article className="rankings-page" aria-labelledby="rankings-title">
      <section className="rankings-hero">
        <p className="eyebrow">{MODEL_LABEL} · {selectedSeason} {isPreseasonPreview ? "preseason preview" : "completed fit"}</p>
        <h1 id="rankings-title">Player rankings.</h1>
        {isPreseasonPreview ? (
          <p>
            Frozen before 2026-27 play. Returning players receive a forward value-conditioned aging prior plus the
            2025-26 additive profile adjustment; drafted and undrafted cold starts use their respective forward
            priors. Season updates and non-additive lineup edges are unavailable until the season is played.
          </p>
        ) : (
          <p>
            Completed-fit {MODEL_LABEL} coefficients per 100 possessions for the selected season. Each table is intended
            for within-season comparison, not an era-neutral all-time ranking.
          </p>
        )}
      </section>

      <section className="rankings-table-section" aria-label={`${MODEL_LABEL} player rankings`}>
        <div className="rankings-table-toolbar">
          <p>{isLoading ? "Loading rankings" : `Showing ${visiblePlayers.length} of ${players.length} players`}</p>
          <div className="rankings-table-controls">
            <label className="rankings-season">
              <span>Season</span>
              <select value={selectedSeason} onChange={(event) => setSelectedSeason(event.target.value)}>
                {(availableSeasons.length ? availableSeasons : [selectedSeason]).map((season) => (
                  <option key={season} value={season}>{season}</option>
                ))}
              </select>
            </label>
            <label className="rankings-minimum-possessions">
              <span>Min. possessions</span>
              <input
                type="number"
                min="0"
                step="100"
                value={minimumPossessions}
                onChange={(event) => setMinimumPossessions(Math.max(0, Number(event.target.value) || 0))}
              />
            </label>
            <label className="rankings-team">
              <span>Team</span>
              <select value={teamFilter} onChange={(event) => setTeamFilter(event.target.value)}>
                <option value="all">All teams</option>
                {teams.map((team) => <option key={team} value={team}>{team}</option>)}
              </select>
            </label>
            <label className="rankings-draft-class">
              <span>Draft class</span>
              <select value={draftClassFilter} onChange={(event) => setDraftClassFilter(event.target.value)}>
                <option value="all">All classes</option>
                {draftClasses.map((draftYear) => <option key={draftYear} value={draftYear}>{draftYear}</option>)}
                <option value="undrafted">Undrafted</option>
              </select>
            </label>
            <label className="rankings-search">
              <Search size={17} aria-hidden="true" />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search player" />
            </label>
            <div className="mobile-sort-controls" aria-label="Player ranking sort controls">
              <label>
                <span>Sort by</span>
                <select
                  value={sortColumn}
                  onChange={(event) => {
                    const column = event.target.value as RankingColumn;
                    setSortColumn(column);
                    setSortDirection(column === "rank" || column === "player_name" || column === "team" || column === "position" ? "ascending" : "descending");
                  }}
                >
                  {columns.map((column) => <option key={column.key} value={column.key}>{column.label}</option>)}
                </select>
              </label>
              <button
                type="button"
                onClick={() => setSortDirection((direction) => direction === "ascending" ? "descending" : "ascending")}
                aria-label={`Sort ${sortDirection === "ascending" ? "descending" : "ascending"}`}
                title={`Sort ${sortDirection === "ascending" ? "descending" : "ascending"}`}
              >
                {sortDirection === "ascending" ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
            </div>
          </div>
        </div>
        {error && <p className="error"><CircleAlert size={16} /> {error}</p>}
        {!error && !isCompact && <div className="rankings-table-wrap">
          <table className="rankings-table">
            <thead>
              <tr>
                {columns.map((column) => <th className={`${column.numeric ? "numeric " : ""}${column.key === "rapm" ? "nail-rating-column" : ""}`} scope="col" key={column.key}>
                  <button type="button" onClick={() => changeSort(column.key)}>
                    {column.label}
                    {sortColumn === column.key && (sortDirection === "ascending" ? <ChevronUp size={13} /> : <ChevronDown size={13} />)}
                  </button>
                </th>)}
              </tr>
            </thead>
            <tbody>
              {visiblePlayers.map((player) => <tr key={player.player_id}>
                <td className="rank-number">{player.rank}</td>
                <th scope="row"><PlayerHeadshot player={player} /><a className="player-name-link" href={playerProfileHref(player.player_id)}>{player.player_name}</a></th>
                <td>{player.team}</td>
                <td>{player.position}</td>
                <td className="numeric quantity-cell">{formatDraftPick(player)}</td>
                <td className={player.rapm < 0 ? "negative numeric rating-cell nail-rating-cell" : "positive numeric rating-cell nail-rating-cell"}>{formatRating(player.rapm)}</td>
                <td className={player.prior_rating === null ? "numeric" : player.prior_rating < 0 ? "negative numeric rating-cell" : "positive numeric rating-cell"}>{player.prior_rating === null ? "-" : formatRating(player.prior_rating)}</td>
                <td className={player.season_update === null ? "numeric" : player.season_update < 0 ? "negative numeric rating-cell" : "positive numeric rating-cell"}>{player.season_update === null ? "-" : formatRating(player.season_update)}</td>
                <td className={player.additive_profile_adjustment === null ? "numeric" : player.additive_profile_adjustment < 0 ? "negative numeric rating-cell" : "positive numeric rating-cell"}>{player.additive_profile_adjustment === null ? "-" : formatRating(player.additive_profile_adjustment)}</td>
                <td className={player.observed_context_exposure === null ? "numeric" : player.observed_context_exposure < 0 ? "negative numeric rating-cell" : "positive numeric rating-cell"}>{player.observed_context_exposure === null ? "-" : formatRating(player.observed_context_exposure)}</td>
                <td className="numeric quantity-cell">{wholeNumber.format(player.possessions)}</td>
                <td className="numeric quantity-cell">{player.games}</td>
              </tr>)}
            </tbody>
          </table>
        </div>}
        {!error && isCompact && <ol className="mobile-card-list mobile-player-rankings" aria-label={`${MODEL_LABEL} player rankings`}>
          {visiblePlayers.map((player) => <li className="mobile-player-card" key={player.player_id}>
            <div className="mobile-card-heading">
              <span className="mobile-card-rank">#{player.rank}</span>
              <a className="mobile-player-identity" href={playerProfileHref(player.player_id)}>
                <PlayerHeadshot player={player} />
                <span>
                  <strong>{player.player_name}</strong>
                  <small>{player.team} · {player.position} · Pick {formatDraftPick(player)}</small>
                </span>
              </a>
              <span className="mobile-primary-rating">
                <small>NAIL</small>
                <Rating value={player.rapm} />
              </span>
            </div>
            <dl className="mobile-metric-grid mobile-player-metrics">
              <div><dt>Prior</dt><dd>{player.prior_rating === null ? "-" : <Rating value={player.prior_rating} />}</dd></div>
              <div><dt>Season update</dt><dd>{player.season_update === null ? "-" : <Rating value={player.season_update} />}</dd></div>
              <div><dt>Additive profile</dt><dd>{player.additive_profile_adjustment === null ? "-" : <Rating value={player.additive_profile_adjustment} />}</dd></div>
              <div><dt>Non-additive edge</dt><dd>{player.observed_context_exposure === null ? "-" : <Rating value={player.observed_context_exposure} />}</dd></div>
            </dl>
            <p className="mobile-card-meta">{wholeNumber.format(player.possessions)} possessions · {player.games} games</p>
          </li>)}
        </ol>}
      </section>
    </article>
  );
}

function draftPickGroup(player: Player): number {
  if (player.draft_number !== null) return 0;
  return player.is_undrafted === true ? 1 : 2;
}

function formatDraftPick(player: Player): string {
  if (player.is_undrafted === true) return "U";
  return player.draft_number === null ? "-" : String(player.draft_number);
}

type LineupRankingColumn =
  | "rank"
  | "team"
  | "possessions"
  | "games"
  | "player_edge"
  | "context_edge"
  | "gestalt_score"
  | "actual_net_rating";

function LineupRankingsPage({
  onLoadInLab,
}: {
  onLoadInLab: (side: Side, lineup: RankedLineup, season: string) => void;
}) {
  const isCompact = useMediaQuery("(max-width: 720px)");
  const [lineups, setLineups] = useState<RankedLineup[]>([]);
  const [selectedSeason, setSelectedSeason] = useState("2025-26");
  const [availableSeasons, setAvailableSeasons] = useState<string[]>([]);
  const [minimumPossessions, setMinimumPossessions] = useState(500);
  const [query, setQuery] = useState("");
  const [teamFilter, setTeamFilter] = useState("all");
  const [sortColumn, setSortColumn] = useState<LineupRankingColumn>("context_edge");
  const [sortDirection, setSortDirection] = useState<"ascending" | "descending">("descending");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        setIsLoading(true);
        setError(null);
        const parameters = new URLSearchParams({
          season: selectedSeason,
          minimum_possessions: String(minimumPossessions),
        });
        const response = await fetch(`/api/lineups?${parameters.toString()}`, { signal: controller.signal });
        if (!response.ok) throw new Error("Observed lineup rankings are unavailable.");
        const payload = (await response.json()) as {
          available_seasons: string[];
          lineups: RankedLineup[];
        };
        setAvailableSeasons(payload.available_seasons);
        setLineups(payload.lineups);
      } catch (lineupError) {
        if ((lineupError as Error).name !== "AbortError") setError((lineupError as Error).message);
      } finally {
        if (!controller.signal.aborted) setIsLoading(false);
      }
    })();
    return () => controller.abort();
  }, [minimumPossessions, selectedSeason]);

  useEffect(() => {
    setTeamFilter("all");
  }, [selectedSeason]);

  const teams = useMemo(() => [...new Set(lineups.map((lineup) => lineup.team))].sort(), [lineups]);

  const visibleLineups = useMemo(() => {
    const tokens = query.split(",").map((token) => token.trim().toLocaleLowerCase()).filter(Boolean);
    const teamLineups = teamFilter === "all" ? lineups : lineups.filter((lineup) => lineup.team === teamFilter);
    const filtered = tokens.length
      ? teamLineups.filter((lineup) => tokens.every((token) =>
        lineup.player_names.some((name) => name.toLocaleLowerCase().includes(token)),
      ))
      : teamLineups;
    const direction = sortDirection === "ascending" ? 1 : -1;
    return [...filtered].sort((left, right) => {
      const leftValue = left[sortColumn];
      const rightValue = right[sortColumn];
      if (typeof leftValue === "number" && typeof rightValue === "number") {
        return direction * (leftValue - rightValue) || left.rank - right.rank;
      }
      return direction * String(leftValue).localeCompare(String(rightValue)) || left.rank - right.rank;
    });
  }, [lineups, query, sortColumn, sortDirection, teamFilter]);

  function changeSort(column: LineupRankingColumn) {
    if (column === sortColumn) {
      setSortDirection((direction) => direction === "ascending" ? "descending" : "ascending");
      return;
    }
    setSortColumn(column);
    setSortDirection(column === "rank" ? "ascending" : "descending");
  }

  const columns: Array<{
    key: LineupRankingColumn;
    label: string;
    tooltip: string;
    numeric?: boolean;
  }> = [
    { key: "rank", label: "Rank", tooltip: "Ranked by Context, then possessions. Sorting another column changes the display order, not this rank.", numeric: true },
    { key: "team", label: "Team", tooltip: "Team that used this five-man unit." },
    { key: "gestalt_score", label: "Edge", tooltip: "Expected edge per 100 possessions against the opponents this unit actually faced: NAIL plus Context.", numeric: true },
    { key: "player_edge", label: "NAIL", tooltip: "Difference in the summed NAIL-RAPM ratings of this unit and its actual opponents, per 100 possessions.", numeric: true },
    { key: "context_edge", label: "Context", tooltip: "Residual non-additive lineup effect beyond the five players' NAIL ratings, against actual opponents, per 100 possessions.", numeric: true },
    { key: "actual_net_rating", label: "Net Rating", tooltip: "Observed points scored minus points allowed per 100 possessions while this unit played.", numeric: true },
    { key: "possessions", label: "Poss", tooltip: "Regular-season possessions played by this exact five-man unit." , numeric: true },
    { key: "games", label: "Games", tooltip: "Regular-season games in which this exact five-man unit appeared.", numeric: true },
  ];

  return (
    <article className="rankings-page lineup-rankings-page" aria-labelledby="lineups-title">
      <section className="rankings-hero">
        <p className="eyebrow">{MODEL_LABEL} · {selectedSeason} completed fit</p>
        <h1 id="lineups-title">Lineup contexts.</h1>
        <p>
          Observed regular-season five-man units, scored against the opponents they actually faced. Edge combines
          NAIL player value with the residual lineup-context effect.
        </p>
      </section>

      <section className="rankings-table-section" aria-label="Observed five-man lineup rankings">
        <div className="rankings-table-toolbar">
          <p>{isLoading ? "Loading lineups" : `Showing ${visibleLineups.length} observed units`}</p>
          <div className="rankings-table-controls">
            <label className="rankings-season">
              <span>Season</span>
              <select value={selectedSeason} onChange={(event) => setSelectedSeason(event.target.value)}>
                {(availableSeasons.length ? availableSeasons : [selectedSeason]).map((season) => (
                  <option key={season} value={season}>{season}</option>
                ))}
              </select>
            </label>
            <label className="rankings-minimum-possessions">
              <span>Min. possessions</span>
              <input
                type="number"
                min="0"
                step="100"
                value={minimumPossessions}
                onChange={(event) => setMinimumPossessions(Math.max(0, Number(event.target.value) || 0))}
              />
            </label>
            <label className="rankings-team">
              <span>Team</span>
              <select value={teamFilter} onChange={(event) => setTeamFilter(event.target.value)}>
                <option value="all">All teams</option>
                {teams.map((team) => <option key={team} value={team}>{team}</option>)}
              </select>
            </label>
            <label className="rankings-search lineup-player-search">
              <Search size={17} aria-hidden="true" />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Contains players, comma-separated" />
            </label>
            <div className="mobile-sort-controls" aria-label="Lineup ranking sort controls">
              <label>
                <span>Sort by</span>
                <select
                  value={sortColumn}
                  onChange={(event) => {
                    const column = event.target.value as LineupRankingColumn;
                    setSortColumn(column);
                    setSortDirection(column === "rank" ? "ascending" : "descending");
                  }}
                >
                  {columns.map((column) => <option key={column.key} value={column.key}>{column.label}</option>)}
                </select>
              </label>
              <button
                type="button"
                onClick={() => setSortDirection((direction) => direction === "ascending" ? "descending" : "ascending")}
                aria-label={`Sort ${sortDirection === "ascending" ? "descending" : "ascending"}`}
                title={`Sort ${sortDirection === "ascending" ? "descending" : "ascending"}`}
              >
                {sortDirection === "ascending" ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
            </div>
          </div>
        </div>
        {error && <p className="error"><CircleAlert size={16} /> {error}</p>}
        {!error && !isCompact && <div className="rankings-table-wrap">
          <table className="rankings-table lineup-rankings-table">
            <thead>
              <tr>
                {columns.slice(0, 2).map((column) => <th className={column.numeric ? "numeric" : ""} scope="col" key={column.key}>
                  <button type="button" onClick={() => changeSort(column.key)}>
                    {column.label}
                    <span className="column-info" role="img" aria-label={column.tooltip} data-tooltip={column.tooltip}>
                      <Info size={12} aria-hidden="true" />
                    </span>
                    {sortColumn === column.key && (sortDirection === "ascending" ? <ChevronUp size={13} /> : <ChevronDown size={13} />)}
                  </button>
                </th>)}
                <th scope="col">
                  <span className="lineup-column-heading">
                    Five-man unit
                    <span className="column-info" role="img" aria-label="The exact five players grouped together in regular-season stints. Use the buttons below the names to load the unit into the Matchup Lab." data-tooltip="The exact five players grouped together in regular-season stints. Use the buttons below the names to load the unit into the Matchup Lab.">
                      <Info size={12} aria-hidden="true" />
                    </span>
                  </span>
                </th>
                {columns.slice(2).map((column) => <th className={[
                  column.numeric ? "numeric" : "",
                  column.key === "gestalt_score" ? "edge-column" : "",
                ].filter(Boolean).join(" ")} scope="col" key={column.key}>
                  <button type="button" onClick={() => changeSort(column.key)}>
                    {column.label}
                    <span className="column-info" role="img" aria-label={column.tooltip} data-tooltip={column.tooltip}>
                      <Info size={12} aria-hidden="true" />
                    </span>
                    {sortColumn === column.key && (sortDirection === "ascending" ? <ChevronUp size={13} /> : <ChevronDown size={13} />)}
                  </button>
                </th>)}
              </tr>
            </thead>
            <tbody>
              {visibleLineups.map((lineup) => <tr key={`${lineup.team_id}-${lineup.player_ids.join("-")}`}>
                <td className="rank-number">{lineup.rank}</td>
                <td>{lineup.team}</td>
                <th className="lineup-roster" scope="row">
                  <span className="lineup-roster-names">
                    {lineup.player_names.map((name, index) => <a key={lineup.player_ids[index]} className="player-name-link" href={playerProfileHref(lineup.player_ids[index])}>{name}</a>)}
                  </span>
                  <span className="lineup-lab-actions">
                    <button
                      type="button"
                      onClick={() => onLoadInLab("unit", lineup, selectedSeason)}
                      aria-label={`Load ${lineup.lineup_label} as your unit in the Matchup Lab`}
                      title="Load as your unit"
                    >
                      <ArrowLeft size={13} aria-hidden="true" /> Your
                    </button>
                    <button
                      type="button"
                      onClick={() => onLoadInLab("opponent", lineup, selectedSeason)}
                      aria-label={`Load ${lineup.lineup_label} as the opponent in the Matchup Lab`}
                      title="Load as opponent"
                    >
                      Opponent <ArrowRight size={13} aria-hidden="true" />
                    </button>
                  </span>
                </th>
                <td className={lineup.gestalt_score < 0 ? "negative numeric rating-cell edge-rating" : "positive numeric rating-cell edge-rating"}>{formatRating(lineup.gestalt_score)}</td>
                <td className={lineup.player_edge < 0 ? "negative numeric rating-cell" : "positive numeric rating-cell"}>{formatRating(lineup.player_edge)}</td>
                <td className={lineup.context_edge < 0 ? "negative numeric rating-cell" : "positive numeric rating-cell"}>{formatRating(lineup.context_edge)}</td>
                <td className={lineup.actual_net_rating < 0 ? "negative numeric rating-cell" : "positive numeric rating-cell"}>{formatRating(lineup.actual_net_rating)}</td>
                <td className="numeric quantity-cell">{wholeNumber.format(lineup.possessions)}</td>
                <td className="numeric quantity-cell">{lineup.games}</td>
              </tr>)}
            </tbody>
          </table>
        </div>}
        {!error && isCompact && <ol className="mobile-card-list mobile-lineup-rankings" aria-label="Observed five-man lineup rankings">
          {visibleLineups.map((lineup) => <li className="mobile-lineup-card" key={`${lineup.team_id}-${lineup.player_ids.join("-")}`}>
            <div className="mobile-card-heading lineup-card-heading">
              <span className="mobile-card-rank">#{lineup.rank}</span>
              <strong className="mobile-lineup-team">{lineup.team}</strong>
              <span className="mobile-primary-rating">
                <small>Edge</small>
                <Rating value={lineup.gestalt_score} />
              </span>
            </div>
            <div className="mobile-lineup-roster">
              {lineup.player_names.map((name, index) => <a key={lineup.player_ids[index]} className="player-name-link" href={playerProfileHref(lineup.player_ids[index])}>{name}</a>)}
            </div>
            <dl className="mobile-metric-grid mobile-lineup-metrics">
              <div><dt>NAIL</dt><dd><Rating value={lineup.player_edge} /></dd></div>
              <div><dt>Context</dt><dd><Rating value={lineup.context_edge} /></dd></div>
              <div><dt>Net rating</dt><dd><Rating value={lineup.actual_net_rating} /></dd></div>
            </dl>
            <div className="mobile-lineup-footer">
              <p className="mobile-card-meta">{wholeNumber.format(lineup.possessions)} possessions · {lineup.games} games</p>
              <span className="lineup-lab-actions">
                <button
                  type="button"
                  onClick={() => onLoadInLab("unit", lineup, selectedSeason)}
                  aria-label={`Load ${lineup.lineup_label} as your unit in the Matchup Lab`}
                  title="Load as your unit"
                >
                  <ArrowLeft size={13} aria-hidden="true" /> Your
                </button>
                <button
                  type="button"
                  onClick={() => onLoadInLab("opponent", lineup, selectedSeason)}
                  aria-label={`Load ${lineup.lineup_label} as the opponent in the Matchup Lab`}
                  title="Load as opponent"
                >
                  Opponent <ArrowRight size={13} aria-hidden="true" />
                </button>
              </span>
            </div>
          </li>)}
        </ol>}
      </section>
    </article>
  );
}

type LineupSelectorProps = {
  side: Side;
  players: Player[];
  season: string;
  team: string;
  availableSeasons: string[];
  onAdd: (player: Player) => void;
  onRemove: (playerId: number) => void;
  onClear: () => void;
  onSeasonChange: (season: string) => void;
  onTeamChange: (team: string) => void;
  onError: (message: string) => void;
  isLoading?: boolean;
  onRefresh?: () => void;
};

function LineupSelector({
  side,
  players,
  season,
  team,
  availableSeasons,
  onAdd,
  onRemove,
  onClear,
  onSeasonChange,
  onTeamChange,
  onError,
  isLoading = false,
  onRefresh,
}: LineupSelectorProps) {
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<Player[]>([]);
  const [teams, setTeams] = useState<string[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const searchInput = useRef<HTMLInputElement>(null);
  const label = SIDE_LABELS[side];
  const randomScope = team === "all" ? "randomly" : `from ${team}`;

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        const response = await fetch(`/api/teams?season=${encodeURIComponent(season)}`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error("Team filter is unavailable.");
        const payload = (await response.json()) as { teams: string[] };
        setTeams(payload.teams);
      } catch (teamError) {
        if ((teamError as Error).name !== "AbortError") onError((teamError as Error).message);
      }
    })();
    return () => controller.abort();
  }, [onError, season]);

  useEffect(() => {
    if (!query.trim()) {
      setMatches([]);
      return;
    }
    const controller = new AbortController();
    const delay = window.setTimeout(async () => {
      try {
        setIsSearching(true);
        const parameters = new URLSearchParams({ q: query, season, limit: "10" });
        if (team !== "all") parameters.set("team", team);
        const response = await fetch(`/api/players?${parameters.toString()}`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error("Player search is unavailable.");
        const payload = (await response.json()) as { players: Player[] };
        setMatches(payload.players.filter((player) => !players.some((selected) => selected.player_id === player.player_id)));
      } catch (searchError) {
        if ((searchError as Error).name !== "AbortError") onError((searchError as Error).message);
      } finally {
        setIsSearching(false);
      }
    }, 180);
    return () => {
      controller.abort();
      window.clearTimeout(delay);
    };
  }, [onError, players, query, season, team]);

  function addPlayer(player: Player) {
    if (players.length === 5) return;
    onAdd(player);
    setQuery("");
    setMatches([]);
    searchInput.current?.focus();
  }

  return (
    <section className="lineup-column" aria-label={`${label} selections`}>
      <div className="lineup-column-header">
        <h2>{label}</h2>
        <span className="lineup-controls">
          {players.length}/5
          {onRefresh && (
            <button
              className="refresh-button"
              onClick={onRefresh}
              disabled={isLoading || players.length === 5}
              aria-label={`Fill open ${label.toLowerCase()} slots ${randomScope}`}
              title={`Fill open ${label.toLowerCase()} slots ${randomScope}`}
            >
              <Dices className={isLoading ? "spin" : ""} size={15} />
            </button>
          )}
          <button
            className="clear-button"
            onClick={onClear}
            disabled={isLoading || players.length === 0}
            aria-label={`Clear ${label.toLowerCase()}`}
            title={`Clear ${label.toLowerCase()}`}
          >
            <Trash2 size={15} />
          </button>
        </span>
        <div className="lineup-filters">
          <label className="lineup-season">
            <span>Season</span>
            <select value={season} onChange={(event) => onSeasonChange(event.target.value)} disabled={isLoading}>
              {availableSeasons.map((availableSeason) => <option key={availableSeason} value={availableSeason}>{availableSeason}</option>)}
            </select>
          </label>
          <label className="lineup-team">
            <span>Team</span>
            <select
              value={team}
              onChange={(event) => onTeamChange(event.target.value)}
              disabled={isLoading}
            >
              <option value="all">All teams</option>
              {teams.map((availableTeam) => <option key={availableTeam} value={availableTeam}>{availableTeam}</option>)}
            </select>
          </label>
        </div>
      </div>
      <div className="slot-list">
        {[0, 1, 2, 3, 4].map((slot) => {
          const player = players[slot];
          return player ? (
            <div className="player-slot filled" key={player.player_id}>
              <span className="slot-number">{slot + 1}</span>
              <span className="slot-avatar">
                <PlayerHeadshot player={player} />
                <small>{player.team} · {player.position}</small>
                {player.age !== null && <small>Age {number.format(player.age)}</small>}
              </span>
              <span className="player-in-slot">
                <a className="player-name-link" href={playerProfileHref(player.player_id)}>{player.player_name}</a>
                <PlayerRatingSparkline player={player} />
              </span>
              <Rating className="slot-rating" value={player.rapm} />
              <button
                className="icon-button"
                onClick={() => onRemove(player.player_id)}
                aria-label={`Remove ${player.player_name}`}
                title={`Remove ${player.player_name}`}
              >
                <X size={16} />
              </button>
            </div>
          ) : (
            <div className="player-slot empty" key={slot}>
              <span className="slot-number">{slot + 1}</span>
              <span>Search to add a player</span>
            </div>
          );
        })}
      </div>
      <div className="search-area">
        <label className="search-box">
          <Search size={20} aria-hidden="true" />
          <input
            ref={searchInput}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={`Search ${label.toLowerCase()} players`}
            disabled={players.length === 5}
          />
          {isSearching && <LoaderCircle className="spin" size={18} aria-label="Searching" />}
        </label>
        {query && matches.length > 0 && (
          <div className="search-results" role="listbox">
            {matches.map((player) => (
              <button className="search-result" key={player.player_id} onClick={() => addPlayer(player)}>
                <PlayerHeadshot player={player} />
                <span className="search-result-details">
                  <strong>{player.player_name}</strong>
                  <PlayerRatingSparkline player={player} />
                  <small>{player.team} · {player.position} · {number.format(player.possessions)} poss.</small>
                </span>
                <Rating value={player.rapm} />
              </button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function PlayerRatingSparkline({ player }: { player: Player }) {
  const points = player.rating_history ?? [];
  if (points.length < 2) return null;
  const labelWidth = 18;
  const chartWidth = 64;
  const width = labelWidth + chartWidth;
  const height = 24;
  const padding = 2;
  const values = points.map((point) => point.rating);
  const extent = Math.max(1, ...values.map((value) => Math.abs(value)));
  const x = (index: number) => labelWidth + padding + (index * (chartWidth - padding * 2)) / (points.length - 1);
  const y = (value: number) => height / 2 - (value / extent) * (height / 2 - padding);
  const path = points.map((point, index) => `${index === 0 ? "M" : "L"}${x(index)},${y(point.rating)}`).join(" ");
  const latest = points.at(-1)!;
  const rookieYear = (player.rookie_season ?? points[0].season).slice(-2);
  return (
    <svg
      className="player-rating-sparkline"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${player.player_name} completed-fit ${MODEL_LABEL} ratings from ${points[0].season} through ${latest.season}`}
    >
      <title>Completed-fit {MODEL_LABEL} trajectory: {points[0].season} to {latest.season}</title>
      <text className="player-rating-rookie-label" x={labelWidth - 2} y={height / 2 + 3} textAnchor="end">{rookieYear}</text>
      <line className="player-rating-zero" x1={labelWidth} x2={width} y1={height / 2} y2={height / 2} />
      <path className="player-rating-line" d={path} />
      <circle className={latest.rating < 0 ? "player-rating-point negative-point" : "player-rating-point"} cx={x(points.length - 1)} cy={y(latest.rating)} r="2.5" />
    </svg>
  );
}

function PlayerHeadshot({ player }: { player: Player }) {
  const [failed, setFailed] = useState(false);

  useEffect(() => setFailed(false), [player.player_id]);

  if (failed) {
    return (
      <span className="player-headshot player-headshot-fallback" aria-hidden="true">
        {player.player_name.split(" ").map((part) => part[0]).join("").slice(0, 2)}
      </span>
    );
  }

  return (
    <img
      className="player-headshot"
      src={playerHeadshotUrl(player.player_id)}
      alt=""
      onError={() => setFailed(true)}
    />
  );
}

function Results({ result }: {
  result: Matchup | null;
}) {

  if (!result) {
    return (
      <aside className="results-panel empty-results">
        <div className="result-mark" aria-hidden="true">G</div>
        <p>Matchup output will appear here.</p>
      </aside>
    );
  }
  const compiledLinear = result.model_form === "compiled_linear_x3";
  return (
    <aside className="results-panel" aria-live="polite">
      <div className="result-heading">
        <span>Neutral-court estimate</span>
        <small>
          {result.environment === "neutral"
            ? `${MODEL_LABEL} · mean of ${result.unit_season} and ${result.opponent_season} eras`
            : `${MODEL_LABEL} · ${result.season} era`}
        </small>
      </div>
      <div className="gestalt-score">
        <strong className={result.predicted_net_rating < 0 ? "negative" : ""}>
          {formatRating(result.predicted_net_rating)}
        </strong>
        <span>GESTALT score</span>
        <small>Expected net rating per 100 possessions</small>
      </div>
      <table className="rating-ledger" aria-label="Lineup rating calculation">
        <thead>
          <tr>
            <th scope="col">Rating</th>
            <th scope="col">Your unit</th>
            <th scope="col">Opponent</th>
            <th scope="col">Edge</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Player rating</th>
            <td><Rating value={result.unit.additive_rating} /></td>
            <td><Rating value={result.opponent.additive_rating} /></td>
            <td><Rating value={result.additive_margin} /></td>
          </tr>
          {compiledLinear ? (
            <tr>
              <th scope="row">Non-additive lineup edge</th>
              <td><Rating value={result.unit_composition_rating} /></td>
              <td><Rating value={result.opponent_composition_rating} /></td>
              <td><Rating value={result.contextual_adjustment} /></td>
            </tr>
          ) : (
            <>
              <tr>
                <th scope="row">Composition rating</th>
                <td><Rating value={result.unit_composition_rating} /></td>
                <td><Rating value={result.opponent_composition_rating} /></td>
                <td><Rating value={result.portable_composition_margin} /></td>
              </tr>
              <tr className="ledger-subtotal">
                <th scope="row">Player + composition</th>
                <td />
                <td />
                <td><Rating value={result.additive_margin + result.portable_composition_margin} /></td>
              </tr>
              <tr>
                <th scope="row">Matchup bonus</th>
                <td />
                <td />
                <td><Rating value={result.matchup_adjustment} /></td>
              </tr>
            </>
          )}
          <tr className="ledger-total">
            <th scope="row">GESTALT score</th>
            <td />
            <td />
            <td><Rating value={result.predicted_net_rating} /></td>
          </tr>
        </tbody>
      </table>
      <ContextCurveExplorer result={result} />
    </aside>
  );
}

function ContextCurveExplorer({
  result,
}: {
  result: Matchup;
}) {
  const compiledLinear = result.model_form === "compiled_linear_x3";
  return (
    <section className="curve-explorer" aria-label="Context component curves">
      <h2>Context components</h2>
      <div className="curve-columns">
        {compiledLinear ? (
          <ContextCurveGroup
            title="Non-additive lineup edge"
            value={result.contextual_adjustment}
            signals={result.composition_feature_contributions}
            responseCurves={new Map()}
            kind="composition"
          />
        ) : (
          <>
            <ContextCurveGroup
              title="Lineup composition"
              value={result.portable_composition_margin}
              signals={result.composition_feature_contributions}
              responseCurves={new Map((result.composition_response_curves ?? []).map((curve) => [curve.id, curve]))}
              kind="composition"
            />
            <ContextCurveGroup
              title="Matchup bonus"
              value={result.matchup_adjustment}
              signals={result.matchup_feature_contributions}
              responseCurves={new Map((result.matchup_response_curves ?? []).map((curve) => [curve.id, curve]))}
              kind="matchup"
            />
          </>
        )}
      </div>
    </section>
  );
}

function ContextCurveGroup({
  title,
  value,
  signals,
  responseCurves,
  kind,
}: {
  title: string;
  value: number;
  signals: ContextFeature[];
  responseCurves: Map<string, FeatureResponseCurve>;
  kind: "composition" | "matchup";
}) {
  const rankedSignals = signals.filter(
    (signal) => Math.abs(signal.contribution) >= MATERIAL_COMPONENT_CONTRIBUTION,
  ).sort(
    (left, right) => Math.abs(right.contribution) - Math.abs(left.contribution),
  );
  return (
    <div className="curve-group">
      <div className="curve-group-heading">
        <h3>{title}</h3>
        <strong><Rating value={value} /></strong>
      </div>
      <div className="curve-grid">
      {rankedSignals.map((signal, index) => (
        <article className="curve-card" key={signal.id}>
          <div className="curve-card-heading">
            <span className="curve-card-title"><b>#{index + 1}</b><span className="curve-card-label">{signal.label}</span>
              <button
                className="feature-info"
                type="button"
                aria-label={`Explain ${signal.label}`}
                data-tooltip={FEATURE_DESCRIPTIONS[signal.id] ?? signal.label}
              ><Info size={13} aria-hidden="true" /></button>
            </span>
            <strong><Rating value={signal.contribution} /></strong>
          </div>
          {responseCurves.get(signal.id) && (
            <ResponseSparkline
              curve={responseCurves.get(signal.id)!}
              yAxisDecimals={title === "Matchup bonus" ? 2 : 1}
              showOpponent={kind === "composition"}
            />
          )}
        </article>
      ))}
      </div>
    </div>
  );
}

function ResponseSparkline({
  curve,
  yAxisDecimals,
  showOpponent,
}: {
  curve: FeatureResponseCurve;
  yAxisDecimals: number;
  showOpponent: boolean;
}) {
  const width = 238;
  const height = 72;
  const padding = 3;
  const axisHeight = 16;
  const axisWidth = 25;
  const values = curve.points.map((point) => point.value);
  const responses = curve.points.map((point) => point.contribution);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const minResponse = Math.min(0, curve.unit_contribution, curve.opponent_contribution, ...responses);
  const maxResponse = Math.max(0, curve.unit_contribution, curve.opponent_contribution, ...responses);
  const x = (value: number) => axisWidth + ((value - minValue) / (maxValue - minValue)) * (width - axisWidth - padding);
  const y = (value: number) => height - padding - axisHeight - ((value - minResponse) / (maxResponse - minResponse || 1)) * (height - padding * 2 - axisHeight);
  const path = curve.points.map((point) => `${x(point.value)},${y(point.contribution)}`).join(" ");
  const supportStart = x(curve.support_low);
  const supportEnd = x(curve.support_high);
  const ticks = [curve.support_low, (curve.support_low + curve.support_high) / 2, curve.support_high];
  const yTicks = Array.from(new Set([minResponse, 0, maxResponse])).sort((left, right) => left - right);

  return (
    <svg className="response-sparkline response-sparkline-expanded" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Context response curve">
        <rect className="sparkline-support" x={supportStart} y={padding} width={supportEnd - supportStart} height={height - padding * 2} />
        {yTicks.map((value) => (
          <text className="sparkline-tick" key={value} x={axisWidth - 3} y={y(value) + 3} textAnchor="end">
            <tspan className={value < 0 ? "negative" : ""}>{value.toFixed(yAxisDecimals)}</tspan>
          </text>
        ))}
        <line className="sparkline-zero" x1={axisWidth} x2={width - padding} y1={y(0)} y2={y(0)} />
        <polyline className="sparkline-line" points={path} />
        {showOpponent && <>
          <line className="sparkline-opponent-marker" x1={x(curve.opponent_value)} x2={x(curve.opponent_value)} y1={padding} y2={height - padding - axisHeight} />
          <circle className="sparkline-opponent-point" cx={x(curve.opponent_value)} cy={y(curve.opponent_contribution)} r="2.1" />
        </>}
        <line className="sparkline-marker" x1={x(curve.unit_value)} x2={x(curve.unit_value)} y1={padding} y2={height - padding - axisHeight} />
        <circle className="sparkline-point" cx={x(curve.unit_value)} cy={y(curve.unit_contribution)} r="2.3" />
        {ticks.map((value) => (
          <text className="sparkline-tick" key={value} x={x(value)} y={height - 2} textAnchor="middle">
            <tspan className={value < 0 ? "negative" : ""}>{value.toFixed(1)}</tspan>
          </text>
        ))}
    </svg>
  );
}

export default App;
