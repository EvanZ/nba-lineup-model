import { useEffect, useMemo, useRef, useState } from "react";
import {
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

import type { ContextFeature, FeatureResponseCurve, Matchup, Player, RankedPlayer } from "./types";

type Side = "unit" | "opponent";
type AppView = "lab" | "rankings" | "about" | "player";
type AppRoute = { view: AppView; playerId?: number };

const SIDE_LABELS: Record<Side, string> = { unit: "Your unit", opponent: "Opponent" };
const MATERIAL_COMPONENT_CONTRIBUTION = 0.05;
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
    return { view: "lab" };
  };
  const [view, setView] = useState<AppRoute>(getView);

  useEffect(() => {
    const updateView = () => setView(getView());
    window.addEventListener("hashchange", updateView);
    return () => window.removeEventListener("hashchange", updateView);
  }, []);

  return view;
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
  const contextInputs = [
    player.three_pm_per_100,
    player.assists_per_100,
    player.usage_per_100,
    player.offensive_rebounds_per_100,
    player.defensive_rebounds_per_100,
  ];
  const hasContextInputs = contextInputs.every((value) => value !== null);

  return (
    <article className="player-profile-page" aria-labelledby="player-profile-title">
      <section className="player-profile-hero">
        <PlayerHeadshot player={player} />
        <div>
          <p className="eyebrow">{player.team} · {player.position} · Age {player.age === null ? "-" : number.format(player.age)}</p>
          <h1 id="player-profile-title">{player.player_name}</h1>
          <p className="player-profile-meta">Rookie season {player.rookie_season ?? "-"} · {wholeNumber.format(player.games)} games · {wholeNumber.format(player.possessions)} possessions</p>
        </div>
        <div className="profile-hero-rating">
          <span>{player.rating_season ?? "Latest"} HIPSTER PM</span>
          <Rating value={player.rapm} />
        </div>
      </section>

      {hasContextInputs && <section className="player-profile-section" aria-labelledby="profile-rates-title">
        <div className="player-profile-heading">
          <p className="section-kicker">Prior-season profile</p>
          <h2 id="profile-rates-title">Context inputs.</h2>
        </div>
        <dl className="player-stat-grid">
          <div><dt>3PM / 100</dt><dd>{number.format(player.three_pm_per_100!)}</dd></div>
          <div><dt>Assists / 100</dt><dd>{number.format(player.assists_per_100!)}</dd></div>
          <div><dt>Usage / 100</dt><dd>{number.format(player.usage_per_100!)}</dd></div>
          <div><dt>OREB / 100</dt><dd>{number.format(player.offensive_rebounds_per_100!)}</dd></div>
          <div><dt>DREB / 100</dt><dd>{number.format(player.defensive_rebounds_per_100!)}</dd></div>
        </dl>
      </section>}

      <section className="player-profile-section" aria-labelledby="rating-history-title">
        <div className="player-profile-heading">
          <p className="section-kicker">Completed fits</p>
          <h2 id="rating-history-title">HIPSTER PM history.</h2>
        </div>
        <PlayerAgingChart player={player} />
        <p className="player-rating-path-note">Prior context edge is the possession-weighted lineup-versus-opponent context predicted from the previous season. It is shared unit exposure, not credit divided among teammates.</p>
        <div className="player-history-table-wrap">
          <table className="player-history-table">
            <thead><tr><th>Season</th><th>Team</th><th>Age</th><th>GP</th><th>GS</th><th>Possessions</th><th>Prior</th><th>Prior context edge</th><th>Season update</th><th>HIPSTER PM</th></tr></thead>
            <tbody>{[...player.rating_history].reverse().map((point) => (
              <tr key={point.season}>
                <td>{point.season}</td><td>{point.team}</td><td>{point.age === null ? "-" : number.format(point.age)}</td>
                <td>{wholeNumber.format(point.games)}</td><td>{wholeNumber.format(point.games_started)}</td>
                <td>{wholeNumber.format(point.possessions)}</td>
                <td>{point.prior_rating === null ? "-" : <Rating value={point.prior_rating} />}</td>
                <td>{point.prior_context_unit_edge === null ? "-" : <Rating value={point.prior_context_unit_edge} />}</td>
                <td>{point.season_update === null ? "-" : <Rating value={point.season_update} />}</td>
                <td><Rating value={point.rating} /></td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </section>
    </article>
  );
}

function PlayerAgingChart({ player }: { player: Player }) {
  const chartRef = useRef<SVGSVGElement>(null);
  const [showLeagueLeaders, setShowLeagueLeaders] = useState(true);
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
  const width = 720;
  const height = 266;
  const margin = { top: 38, right: 34, bottom: 54, left: 48 };
  const ages = points.map((point) => point.age!);
  const ratings = points.map((point) => point.rating);
  const seasonMaxes = points.map((point) => point.season_max_rating ?? point.rating);
  const minAge = Math.min(...ages);
  const maxAge = Math.max(...ages);
  const lowerBound = Math.floor(Math.min(0, ...ratings));
  const upperBound = Math.ceil(Math.max(0, ...ratings, ...seasonMaxes));
  const ratingRange = Math.max(1, upperBound - lowerBound);
  const x = (age: number) => margin.left + ((age - minAge) / Math.max(1, maxAge - minAge)) * (width - margin.left - margin.right);
  const y = (rating: number) => margin.top + ((upperBound - rating) / ratingRange) * (height - margin.top - margin.bottom);
  const zeroY = y(0);
  const path = points.map((point, index) => `${index === 0 ? "M" : "L"}${x(point.age!)},${y(point.rating)}`).join(" ");
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
        link.download = `${player.player_name.toLowerCase().replaceAll(/[^a-z0-9]+/g, "-").replaceAll(/^-|-$/g, "")}-hipster-pm-aging.png`;
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

  return (
    <figure className="player-aging-chart">
      <div className="player-aging-chart-toolbar">
        <span>Age trajectory</span>
        <div className="player-aging-chart-actions">
          <button
            className="chart-layer-toggle"
            type="button"
            role="switch"
            onClick={toggleLeagueLeaders}
            title={showLeagueLeaders ? "Hide league leaders" : "Show league leaders"}
            aria-label={showLeagueLeaders ? "Hide league leaders" : "Show league leaders"}
            aria-checked={showLeagueLeaders}
          >
            <span>League maxima</span>
            <span className="chart-layer-toggle-track" aria-hidden="true"><span /></span>
          </button>
          <button type="button" onClick={downloadPng} title="Download chart as PNG" aria-label={`Download ${player.player_name} age trajectory as PNG`}><Download size={14} /> Download PNG</button>
        </div>
      </div>
      <svg ref={chartRef} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${player.player_name} completed-fit HIPSTER PM by age`}>
        <text className="aging-chart-title" x={margin.left} y="17">{player.player_name} · HIPSTER PM age trajectory</text>
        <line className="aging-chart-grid" x1={margin.left} x2={width - margin.right} y1={margin.top} y2={margin.top} />
        <line className="aging-chart-zero" x1={margin.left} x2={width - margin.right} y1={zeroY} y2={zeroY} />
        <line className="aging-chart-grid" x1={margin.left} x2={width - margin.right} y1={height - margin.bottom} y2={height - margin.bottom} />
        <text className="aging-chart-y-label" x={margin.left - 10} y={margin.top + 4} textAnchor="end">{formatChartRating(upperBound)}</text>
        <text className="aging-chart-y-label" x={margin.left - 10} y={zeroY + 4} textAnchor="end">0.0</text>
        <text className="aging-chart-y-label" x={margin.left - 10} y={height - margin.bottom + 4} textAnchor="end">{formatChartRating(lowerBound)}</text>
        {showLeagueLeaders && points.map((point) => {
          const leaderRating = point.season_max_rating ?? point.rating;
          const leaderId = point.season_max_player_id;
          const leaderName = point.season_max_player_name ?? "League leader";
          return <g key={`leader-${point.season}`}>
            {leaderId !== undefined && (
              <a
                href={playerProfileHref(leaderId)}
                aria-label={`${leaderName}, ${formatChartRating(leaderRating)} HIPSTER PM`}
                onMouseEnter={() => setHoveredLeader({ name: leaderName, rating: leaderRating, x: x(point.age!), y: y(leaderRating) })}
                onMouseLeave={() => setHoveredLeader(null)}
                onFocus={() => setHoveredLeader({ name: leaderName, rating: leaderRating, x: x(point.age!), y: y(leaderRating) })}
                onBlur={() => setHoveredLeader(null)}
              >
                <title>{leaderName} · {formatChartRating(leaderRating)} HIPSTER PM</title>
                <circle className="aging-chart-leader-fallback" cx={x(point.age!)} cy={y(leaderRating)} r="10" />
                <image
                  className="aging-chart-leader-headshot"
                  href={playerHeadshotUrl(leaderId)}
                  x={x(point.age!) - 9}
                  y={y(leaderRating) - 9}
                  width="18"
                  height="18"
                  preserveAspectRatio="xMidYMid slice"
                  onError={(event) => { event.currentTarget.style.display = "none"; }}
                />
              </a>
            )}
            <text className="aging-chart-leader-value" x={x(point.age!)} y={Math.max(29, y(leaderRating) - 14)} textAnchor="middle" pointerEvents="none">{formatChartRating(leaderRating)}</text>
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
        {points.map((point) => <g key={`age-${point.season}`}>
          <line className="aging-chart-tick" x1={x(point.age!)} x2={x(point.age!)} y1={height - margin.bottom} y2={height - margin.bottom + 5} />
          <text className="aging-chart-x-label" x={x(point.age!)} y={height - 27} textAnchor="middle">
            <tspan x={x(point.age!)}>{point.season.slice(-2)}</tspan>
            <tspan x={x(point.age!)} dy="11">{number.format(point.age!)}</tspan>
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
            <text x="7" y="23">{formatChartRating(hoveredLeader.rating)} HIPSTER PM</text>
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
            <text x="7" y="12">{hoveredPlayerPoint.season} · {formatChartRating(hoveredPlayerPoint.rating)} HIPSTER PM</text>
            <text x="7" y="24">G {wholeNumber.format(hoveredPlayerPoint.games)} · GS {wholeNumber.format(hoveredPlayerPoint.gamesStarted)}</text>
            <text x="7" y="36">{wholeNumber.format(hoveredPlayerPoint.possessions)} possessions</text>
          </g>
        )}
      </svg>
      <figcaption>{showLeagueLeaders
        ? "Completed-fit HIPSTER PM by age. The green line is the player; headshots mark the league’s highest completed-fit player rating in each season."
        : "Completed-fit HIPSTER PM by age. The green line is the player."}
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
  const [result, setResult] = useState<Matchup | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [isLoadingUnit, setIsLoadingUnit] = useState(false);
  const [isLoadingOpponent, setIsLoadingOpponent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canEvaluate = unit.length === 5 && opponent.length === 5;
  const allSelectedIds = useMemo(
    () => new Set([...unit, ...opponent].map((player) => player.player_id)),
    [unit, opponent],
  );

  async function loadRandomLineup(side: Side) {
    try {
      setError(null);
      const setLoading = side === "unit" ? setIsLoadingUnit : setIsLoadingOpponent;
      setLoading(true);
      const parameters = new URLSearchParams();
      const currentPlayers = side === "unit" ? unit : opponent;
      const protectedPlayers = side === "unit" ? opponent : unit;
      [...currentPlayers, ...protectedPlayers].forEach((player) => {
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
        <a className="wordmark" href="#lab" aria-label="NBA GESTALT Lineup Lab">
          <span>NBA</span> GESTALT
        </a>
        <div className="header-meta">
          <nav className="header-navigation" aria-label="Primary navigation">
            <a className={view === "lab" ? "active" : ""} href="#lab">Lineup Lab</a>
            <a className={view === "rankings" ? "active" : ""} href="#rankings">Rankings</a>
            <a className={view === "about" ? "active" : ""} href="#about">About</a>
          </nav>
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
          <span className="season-chip">2025-26 retrospective</span>
        </div>
      </header>

      {view === "about" ? <AboutPage /> : view === "rankings" ? <RankingsPage /> : view === "player" && route.playerId ? <PlayerProfilePage playerId={route.playerId} /> : <>
        <section className="intro" aria-labelledby="page-title">
          <p className="eyebrow">A lineup is more than the sum of five player ratings.</p>
          <h1 id="page-title">Build the five.</h1>
        </section>

        <section className="lab-grid" aria-label="Lineup builder">
          <div className="builder-panel">
            <div className="lineup-columns">
              <LineupSelector
                side="unit"
                players={unit}
                unavailablePlayerIds={allSelectedIds}
                isLoading={isLoadingUnit}
                onRefresh={() => void loadRandomLineup("unit")}
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
                unavailablePlayerIds={allSelectedIds}
                isLoading={isLoadingOpponent}
                onRefresh={() => void loadRandomLineup("opponent")}
                onAdd={(player) => {
                  setOpponent((current) => [...current, player]);
                  setResult(null);
                }}
                onRemove={(playerId) => removePlayer("opponent", playerId)}
                onClear={() => clearLineup("opponent")}
                onError={setError}
              />
            </div>

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
        {result && <ContextCurveExplorer result={result} />}
      </>}
    </main>
  );
}

function AboutPage() {
  return (
    <article className="about-page" aria-labelledby="about-title">
      <section className="about-hero">
        <p className="eyebrow">About NBA GESTALT</p>
        <h1 id="about-title">A forecast built one season at a time.</h1>
        <p className="about-lede">
          HIPSTER PM estimates a neutral-court lineup edge per 100 possessions by combining
          player talent, attributable composition, and matchup-specific context.
        </p>
      </section>

      <section className="model-identity" aria-labelledby="model-name-title">
        <div>
          <p className="section-kicker">Current model</p>
          <h2 id="model-name-title">HIPSTER PM</h2>
        </div>
        <p className="model-identity-name">Forward age-informed HIPSTER PM</p>
        <p>
          <b>H</b>ierarchical <b>I</b>nterpretable <b>P</b>enalized-<b>S</b>pline,
          <b>T</b>eammate-adjusted, <b>E</b>xposure-gated, <b>R</b>egularized
          <b>P</b>lus-<b>M</b>inus.
        </p>
      </section>

      <section className="about-equation" aria-labelledby="equation-title">
        <div>
          <p className="section-kicker">The estimate</p>
          <h2 id="equation-title">Three terms, one score.</h2>
        </div>
        <p className="model-formula">
          HPM(A, B) = Player(A) - Player(B) + h(Profile(A)) - h(Profile(B)) + q(Profile(A), Profile(B))
        </p>
        <p>
          The player term comes from regularized adjusted plus-minus. The function <i>h</i> assigns
          an attributable composition rating to either unit. The function <i>q</i> captures the small
          residual created by this particular matchup.
        </p>
      </section>

      <section className="about-section" aria-labelledby="prior-title">
        <div className="about-section-heading">
          <p className="section-kicker">Player prior</p>
          <h2 id="prior-title">Forward RAPM.</h2>
          <p>Each player estimate begins with only information that was available before the target season.</p>
        </div>
        <div className="about-steps two-up">
          <section>
            <span>01</span>
            <h3>Forward</h3>
            <p>
              The model rolls from completed season to completed season. A target season never supplies its
              own possessions, outcomes, or box-score rates to its starting estimate.
            </p>
          </section>
          <section>
            <span>02</span>
            <h3>RAPM</h3>
            <p>
              Regularized Adjusted Plus-Minus estimates player coefficients from possession outcomes while
              accounting for the other nine players on the floor and home court. Its ridge penalty stabilizes
              correlated lineup data by pulling uncertain player estimates toward their preseason priors.
            </p>
          </section>
        </div>
      </section>

      <section className="about-section" aria-labelledby="aging-title">
        <div className="about-section-heading">
          <p className="section-kicker">Season-to-season state</p>
          <h2 id="aging-title">Aging and cold starts.</h2>
          <p>Returning players and players without an NBA-season estimate take different, leakage-safe paths.</p>
        </div>
        <div className="about-steps two-up">
          <section>
            <span>03</span>
            <h3>Aging</h3>
            <p>
              For returning players, an age-spline ridge model forecasts the next RAPM level from the last
              completed RAPM, on-court possession exposure, age, NBA experience, draft information, and physical profile.
            </p>
          </section>
          <section>
            <span>04</span>
            <h3>Exposure-gated cold start</h3>
            <p>
              Players without a usable prior-season estimate receive a blend of a draft-profile forecast and a
              historical replacement estimate. The blend shifts toward replacement when the forward exposure model
              expects little opportunity.
            </p>
          </section>
        </div>
      </section>

      <section className="about-section contextual-section" aria-labelledby="context-title">
        <div className="about-section-heading">
          <p className="section-kicker">Lineup context</p>
          <h2 id="context-title">Bounded hierarchical P-spline portable-matchup contextual prior.</h2>
          <p>These terms use prior-season unit profiles, not target-season results, to describe how five-player composition modifies the player edge.</p>
        </div>
        <dl className="model-lexicon">
          <div>
            <dt>Bounded</dt>
            <dd>Feature inputs are capped at their learned support so an unusual hypothetical lineup cannot extrapolate a curve into an unsupported tail.</dd>
          </div>
          <div>
            <dt>Hierarchical</dt>
            <dd>Each season’s contextual functions are softly anchored to the preceding season’s functions, preserving signal while allowing the league to change.</dd>
          </div>
          <div>
            <dt>P-spline</dt>
            <dd>Smooth penalized spline functions allow nonlinear effects, such as diminishing rebounding returns, without giving every local fluctuation a new parameter.</dd>
          </div>
          <div>
            <dt>Attributable composition</dt>
            <dd>The portable function <i>h</i> maps a unit’s own profile to its composition rating, so each side receives a separately visible context term.</dd>
          </div>
          <div>
            <dt>Matchup</dt>
            <dd>The residual function <i>q</i> measures what remains when those two unit profiles meet: a small, opponent-specific bonus or penalty.</dd>
          </div>
          <div>
            <dt>Contextual prior</dt>
            <dd>Those profile terms enter the RAPM estimation as prior context. They do not credit a player for the target season’s own rebounds, shooting, or turnovers.</dd>
          </div>
        </dl>
      </section>

      <section className="about-footer-note">
        <p className="section-kicker">Current publication</p>
        <p>
          The Lineup Lab serves the age-informed 2025-26 completed fit. It is a retrospective model state for
          exploring lineups, not a live forecast or a causal player-value claim.
        </p>
      </section>
    </article>
  );
}

type RankingColumn = "rank" | "player_name" | "team" | "position" | "rapm" | "possessions" | "games";

function RankingsPage() {
  const [players, setPlayers] = useState<RankedPlayer[]>([]);
  const [selectedSeason, setSelectedSeason] = useState("2025-26");
  const [availableSeasons, setAvailableSeasons] = useState<string[]>([]);
  const [query, setQuery] = useState("");
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
        if (!response.ok) throw new Error("HIPSTER PM rankings are unavailable.");
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

  const visiblePlayers = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    const rows = normalized
      ? players.filter((player) => [player.player_name, player.team, player.position]
        .some((value) => value.toLocaleLowerCase().includes(normalized)))
      : players;
    const direction = sortDirection === "ascending" ? 1 : -1;
    return [...rows].sort((left, right) => {
      const leftValue = left[sortColumn];
      const rightValue = right[sortColumn];
      if (typeof leftValue === "number" && typeof rightValue === "number") {
        return direction * (leftValue - rightValue) || left.rank - right.rank;
      }
      return direction * String(leftValue).localeCompare(String(rightValue)) || left.rank - right.rank;
    });
  }, [players, query, sortColumn, sortDirection]);

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
    { key: "rapm", label: "HIPSTER PM", numeric: true },
    { key: "possessions", label: "Possessions", numeric: true },
    { key: "games", label: "Games", numeric: true },
  ];

  return (
    <article className="rankings-page" aria-labelledby="rankings-title">
      <section className="rankings-hero">
        <p className="eyebrow">HIPSTER PM · {selectedSeason} completed fit</p>
        <h1 id="rankings-title">Player rankings.</h1>
        <p>
          Completed-fit HIPSTER PM coefficients per 100 possessions for the selected season. Each table is intended
          for within-season comparison, not an era-neutral all-time ranking.
        </p>
      </section>

      <section className="rankings-table-section" aria-label="HIPSTER PM player rankings">
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
            <label className="rankings-search">
              <Search size={17} aria-hidden="true" />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search player or team" />
            </label>
          </div>
        </div>
        {error && <p className="error"><CircleAlert size={16} /> {error}</p>}
        {!error && <div className="rankings-table-wrap">
          <table className="rankings-table">
            <thead>
              <tr>
                {columns.map((column) => <th className={column.numeric ? "numeric" : ""} scope="col" key={column.key}>
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
                <td className={player.rapm < 0 ? "negative numeric rating-cell" : "positive numeric rating-cell"}>{formatRating(player.rapm)}</td>
                <td className="numeric">{wholeNumber.format(player.possessions)}</td>
                <td className="numeric">{player.games}</td>
              </tr>)}
            </tbody>
          </table>
        </div>}
      </section>
    </article>
  );
}

type LineupSelectorProps = {
  side: Side;
  players: Player[];
  unavailablePlayerIds: Set<number>;
  onAdd: (player: Player) => void;
  onRemove: (playerId: number) => void;
  onClear: () => void;
  onError: (message: string) => void;
  isLoading?: boolean;
  onRefresh?: () => void;
};

function LineupSelector({
  side,
  players,
  unavailablePlayerIds,
  onAdd,
  onRemove,
  onClear,
  onError,
  isLoading = false,
  onRefresh,
}: LineupSelectorProps) {
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<Player[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const searchInput = useRef<HTMLInputElement>(null);
  const label = SIDE_LABELS[side];

  useEffect(() => {
    if (!query.trim()) {
      setMatches([]);
      return;
    }
    const controller = new AbortController();
    const delay = window.setTimeout(async () => {
      try {
        setIsSearching(true);
        const response = await fetch(`/api/players?q=${encodeURIComponent(query)}&limit=10`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error("Player search is unavailable.");
        const payload = (await response.json()) as { players: Player[] };
        setMatches(
          payload.players.filter((player) => !unavailablePlayerIds.has(player.player_id)),
        );
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
  }, [onError, query, unavailablePlayerIds]);

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
              aria-label={`Fill open ${label.toLowerCase()} slots randomly`}
              title={`Fill open ${label.toLowerCase()} slots randomly`}
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
      aria-label={`${player.player_name} completed-fit HPM ratings from ${points[0].season} through ${latest.season}`}
    >
      <title>Completed-fit HPM trajectory: {points[0].season} to {latest.season}</title>
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
  const combinedEdge = result.additive_margin + result.portable_composition_margin;
  return (
    <aside className="results-panel" aria-live="polite">
      <div className="result-heading">
        <span>Neutral-court estimate</span>
        <small>HIPSTER PM · 2025-26 completed fit</small>
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
            <td><Rating value={combinedEdge} /></td>
          </tr>
          <tr>
            <th scope="row">Matchup bonus</th>
            <td />
            <td />
            <td><Rating value={result.matchup_adjustment} /></td>
          </tr>
          <tr className="ledger-total">
            <th scope="row">GESTALT score</th>
            <td />
            <td />
            <td><Rating value={result.predicted_net_rating} /></td>
          </tr>
        </tbody>
      </table>
    </aside>
  );
}

function ContextCurveExplorer({
  result,
}: {
  result: Matchup;
}) {
  return (
    <section className="curve-explorer" aria-label="Context component curves">
      <h2>Context components</h2>
      <div className="curve-columns">
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
