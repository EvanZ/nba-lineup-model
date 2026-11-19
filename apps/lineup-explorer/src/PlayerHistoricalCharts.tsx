import { useRef, useState } from "react";
import { Download } from "lucide-react";
import type { Player } from "./types";
import { teamLogoUrl } from "./teamLogos";

type CareerPoint = Pick<Player["rating_history"][number], "season" | "rating" | "age" | "team" | "prior_rating" | "season_update" | "additive_profile_adjustment" | "season_min_rating" | "season_max_rating" | "season_max_player_id" | "season_max_player_name"> & { games: number | null; games_started: number | null; possessions: number | null };
type CareerChartPlayer = Pick<Player, "player_id" | "player_name" | "league_leader_history"> & { rating_history: CareerPoint[] };
type RotationChartPlayer = Pick<Player, "player_name" | "rotation_history">;
type RotationHistoryPoint = NonNullable<Player["rotation_history"]>[number];
type AgingContributionKey = "prior" | "seasonUpdate" | "additiveProfile";
const MODEL_LABEL = "NAIL-RAPM v1.2.1.3";
const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const wholeNumber = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const count = (n: number | null) => n === null ? "—" : wholeNumber.format(n);
const formatChartRating = (n: number) => n.toFixed(1);
const playerHeadshotUrl = (id: number) => `/api/headshots/${id}.png`;
const playerProfileHref = (id: number) => `#player/${id}`;

export function PlayerRotationHistoryCards({ rows }: { rows: RotationHistoryPoint[] }) {
  return (
    <ol className="player-history-mobile-list player-rotation-mobile-list" aria-label="Availability and rotation history">
      {[...rows].reverse().map((row) => {
        const availableGames = row.actual_available_games === null
          ? "-"
          : `${wholeNumber.format(row.actual_available_games)} / ${row.known_roster_games === null ? "—" : wholeNumber.format(row.known_roster_games)}`;
        const availabilityForecast = row.predicted_available_share === null
          ? "-"
          : `${number.format(row.predicted_available_share * 82)} G`;
        const observedMinutes = row.actual_minutes_per_available_game === null
          ? "-"
          : number.format(row.actual_minutes_per_available_game);
        const forecastMinutes = row.predicted_minutes_per_available_game === null
          ? "-"
          : number.format(row.predicted_minutes_per_available_game);
        const preseasonAvailabilityForecast = row.preseason_predicted_available_share === null || row.preseason_predicted_available_share === undefined
          ? null
          : `${number.format(row.preseason_predicted_available_share * 82)} G`;
        const preseasonMinutesForecast = row.preseason_predicted_minutes_per_available_game === null || row.preseason_predicted_minutes_per_available_game === undefined
          ? null
          : number.format(row.preseason_predicted_minutes_per_available_game);
        const totalMinutes = row.is_preseason_forecast ? row.projected_total_minutes : row.actual_total_minutes;
        return <li className={row.is_current_season_update ? "player-history-mobile-card current" : row.is_preseason_forecast ? "player-history-mobile-card forecast" : "player-history-mobile-card"} key={row.season}>
          <header className="player-history-mobile-header">
            <strong>{row.season}</strong>
            {row.is_preseason_forecast && <span className="player-history-mobile-forecast-label">Preseason forecast</span>}
            {row.is_current_season_update && <span className="player-history-mobile-forecast-label current">Current{row.evidence_through_date ? ` · ${row.evidence_through_date}` : ""}</span>}
          </header>
          <dl className="player-history-mobile-metrics">
            <div><dt>Available G</dt><dd>{availableGames}</dd></div>
            <div className="forecast"><dt>Availability estimate</dt><dd>{availabilityForecast}{row.is_current_season_update && preseasonAvailabilityForecast && <small>Preseason {preseasonAvailabilityForecast}</small>}</dd></div>
            <div><dt>MPG when available</dt><dd>{observedMinutes}</dd></div>
            <div className="forecast"><dt>Minutes estimate</dt><dd>{forecastMinutes}{row.is_current_season_update && preseasonMinutesForecast && <small>Preseason {preseasonMinutesForecast}</small>}</dd></div>
            <div className="player-history-mobile-total-minutes"><dt>Total minutes</dt><dd>{totalMinutes === null ? "-" : wholeNumber.format(totalMinutes)}</dd></div>
          </dl>
        </li>;
      })}
    </ol>
  );
}

export function PlayerRotationHistoryChart({ player }: { player: RotationChartPlayer }) {
  const points = (player.rotation_history ?? [])
    .filter((point) => Number.isFinite(point.season_start_year))
    .sort((left, right) => left.season_start_year - right.season_start_year);
  if (!points.length) return null;

  const width = 760;
  const height = 322;
  const left = 48;
  const right = 18;
  const availabilityTop = 30;
  const panelHeight = 92;
  const minutesTop = 183;
  const plotWidth = width - left - right;
  const minimumXGap = Math.max(46, plotWidth / Math.max(points.length - 1, 1));
  const chartWidth = Math.max(width, left + right + minimumXGap * Math.max(points.length - 1, 1));
  const x = (index: number) => left + (index * (chartWidth - left - right)) / Math.max(points.length - 1, 1);
  const maxMinutes = Math.max(
    20,
    ...points.flatMap((point) => [
      point.actual_minutes_per_available_game ?? 0,
      point.predicted_minutes_per_available_game ?? 0,
      point.preseason_predicted_minutes_per_available_game ?? 0,
    ]),
  );
  const minutesCeiling = Math.ceil(maxMinutes / 5) * 5;
  const availabilityY = (availableGames: number) => availabilityTop + panelHeight - (availableGames / 82) * panelHeight;
  const minutesY = (minutes: number) => minutesTop + panelHeight - (minutes / minutesCeiling) * panelHeight;
  const linePath = (selector: (point: RotationHistoryPoint) => number | null, y: (value: number) => number) => {
    const segments: string[] = [];
    let current: string[] = [];
    points.forEach((point, index) => {
      const value = selector(point);
      if (value === null || !Number.isFinite(value)) {
        if (current.length) segments.push(current.join(" "));
        current = [];
        return;
      }
      current.push(`${current.length ? "L" : "M"}${x(index)},${y(value)}`);
    });
    if (current.length) segments.push(current.join(" "));
    return segments.join(" ");
  };
  const availabilityForecast = linePath(
    (point) => point.predicted_available_share === null ? null : point.predicted_available_share * 82,
    availabilityY,
  );
  const fcmForecast = linePath(
    (point) => point.predicted_minutes_per_available_game,
    minutesY,
  );

  return (
    <figure className="player-rotation-history-chart">
      <div className="rotation-history-legend" aria-label="Rotation history legend">
        <span><i className="rotation-history-swatch observed" />Observed</span>
        <span><i className="rotation-history-swatch forecast" />Forecast / current estimate</span>
        <span><i className="rotation-history-swatch preseason-reference" />2026–27 preseason baseline</span>
      </div>
      <svg viewBox={`0 0 ${chartWidth} ${height}`} role="img" aria-label={`${player.player_name} availability and forward conditional minutes history`}>
        <title>{`${player.player_name} rotation history`}</title>
        {[availabilityTop, availabilityTop + panelHeight, minutesTop, minutesTop + panelHeight].map((value) => (
          <line key={value} className="rotation-history-grid" x1={left} x2={chartWidth - right} y1={value} y2={value} />
        ))}
        <text className="rotation-history-panel-title" x={left} y={16}>Available games</text>
        <text className="rotation-history-panel-title" x={left} y={169}>MPG when available</text>
        <text className="rotation-history-axis-label" x={left - 8} y={availabilityTop + 4} textAnchor="end">82</text>
        <text className="rotation-history-axis-label" x={left - 8} y={availabilityTop + panelHeight + 4} textAnchor="end">0</text>
        <text className="rotation-history-axis-label" x={left - 8} y={minutesTop + 4} textAnchor="end">{minutesCeiling}</text>
        <text className="rotation-history-axis-label" x={left - 8} y={minutesTop + panelHeight + 4} textAnchor="end">0</text>
        {points.map((point, index) => point.actual_available_games === null ? null : (
          <g key={`available-${point.season}`}>
            <rect
              className="rotation-history-availability-bar"
              x={x(index) - 7}
              y={availabilityY(point.actual_available_games)}
              width="14"
              height={availabilityTop + panelHeight - availabilityY(point.actual_available_games)}
              rx="1"
            >
              <title>{`${point.season}: ${wholeNumber.format(point.actual_available_games)} available games observed`}</title>
            </rect>
            <text
              className="rotation-history-value observed"
              x={x(index)}
              y={Math.max(availabilityTop + 10, availabilityY(point.actual_available_games) - 8)}
              textAnchor="middle"
            >
              {wholeNumber.format(point.actual_available_games)}
            </text>
          </g>
        ))}
        {availabilityForecast && <path className="rotation-history-forecast-line" d={availabilityForecast} />}
        {fcmForecast && <path className="rotation-history-forecast-line" d={fcmForecast} />}
        {points.map((point, index) => !point.is_current_season_update || point.preseason_predicted_available_share === null || point.preseason_predicted_available_share === undefined ? null : (
          <circle
            key={`available-preseason-${point.season}`}
            className="rotation-history-preseason-reference-point"
            cx={x(index)}
            cy={availabilityY(point.preseason_predicted_available_share * 82)}
            r="4"
          >
            <title>{`${point.season} preseason baseline: ${number.format(point.preseason_predicted_available_share * 82)} available games`}</title>
          </circle>
        ))}
        {points.map((point, index) => point.predicted_available_share === null ? null : (
          <text
            key={`available-forecast-${point.season}`}
            className="rotation-history-value forecast"
            x={x(index)}
            y={Math.min(
              availabilityTop + panelHeight - 5,
              availabilityY(point.predicted_available_share * 82) + 16,
            )}
            textAnchor="middle"
          >
            {number.format(point.predicted_available_share * 82)}
          </text>
        ))}
        {points.map((point, index) => point.actual_minutes_per_available_game === null ? null : (
          <g key={`minutes-${point.season}`}>
            <circle className="rotation-history-observed-point" cx={x(index)} cy={minutesY(point.actual_minutes_per_available_game)} r="4">
              <title>{`${point.season}: ${number.format(point.actual_minutes_per_available_game)} observed MPG when available`}</title>
            </circle>
            <text
              className="rotation-history-value observed"
              x={x(index)}
              y={Math.max(minutesTop + 12, minutesY(point.actual_minutes_per_available_game) - 10)}
              textAnchor="middle"
            >
              {number.format(point.actual_minutes_per_available_game)}
            </text>
          </g>
        ))}
        {points.map((point, index) => !point.is_current_season_update || point.preseason_predicted_minutes_per_available_game === null || point.preseason_predicted_minutes_per_available_game === undefined ? null : (
          <circle
            key={`minutes-preseason-${point.season}`}
            className="rotation-history-preseason-reference-point"
            cx={x(index)}
            cy={minutesY(point.preseason_predicted_minutes_per_available_game)}
            r="4"
          >
            <title>{`${point.season} preseason baseline: ${number.format(point.preseason_predicted_minutes_per_available_game)} MPG when available`}</title>
          </circle>
        ))}
        {points.map((point, index) => point.predicted_minutes_per_available_game === null ? null : (
          <text
            key={`minutes-forecast-${point.season}`}
            className="rotation-history-value forecast"
            x={x(index)}
            y={Math.min(
              minutesTop + panelHeight - 7,
              minutesY(point.predicted_minutes_per_available_game) + 16,
            )}
            textAnchor="middle"
          >
            {number.format(point.predicted_minutes_per_available_game)}
          </text>
        ))}
        {points.map((point, index) => (
          <g key={`labels-${point.season}`}>
            <line className="rotation-history-tick" x1={x(index)} x2={x(index)} y1={minutesTop + panelHeight} y2={minutesTop + panelHeight + 4} />
            <text className="rotation-history-x-label" x={x(index)} y={minutesTop + panelHeight + 21} textAnchor="middle">{point.season.slice(-2)}</text>
          </g>
        ))}
      </svg>
      <figcaption>Observed availability counts rostered games with a known status. The current-season availability and minutes estimates update after the nightly promoted slate; the hollow point preserves the frozen preseason baseline.</figcaption>
    </figure>
  );
}

export function PlayerAgingChart({ player }: { player: CareerChartPlayer }) {
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
    games: number | null;
    gamesStarted: number | null;
    possessions: number | null;
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
    prior: point.prior_rating,
    seasonUpdate: point.season_update,
    additiveProfile: point.additive_profile_adjustment,
  }));
  const positiveTotals = contributionValues.map((values) => contributionLayers.reduce(
    (total, layer) => total + Math.max(0, values[layer.key] ?? 0), 0,
  ));
  const negativeTotals = contributionValues.map((values) => contributionLayers.reduce(
    (total, layer) => total + Math.min(0, values[layer.key] ?? 0), 0,
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
    const segments: Array<Array<{ point: CareerPoint; lower: number; upper: number }>> = [];
    let segment: Array<{ point: CareerPoint; lower: number; upper: number }> = [];
    for (let index = 0; index < points.length; index += 1) {
      const point = points[index];
      // A band needs its own value and the lower layers that locate its baseline.
      // Missing later layers must not hide earlier, fully known components.
      const values = contributionValues[index];
      if (contributionLayers.slice(0, layerIndex + 1).some((layer) => !Number.isFinite(values[layer.key]))) {
        if (segment.length) segments.push(segment);
        segment = [];
        continue;
      }
      const previous = points[index - 1];
      if (index > 0 && seasonStartYear(point.season) !== seasonStartYear(previous.season) + 1) {
        if (segment.length) segments.push(segment);
        segment = [];
      }
      const priorTotal = contributionLayers.slice(0, layerIndex).reduce(
        (total, layer) => total + (polarity === "positive" ? Math.max(0, values[layer.key] ?? 0) : Math.min(0, values[layer.key] ?? 0)),
        0,
      );
      const layerValue = polarity === "positive"
        ? Math.max(0, values[contributionLayers[layerIndex].key] ?? 0)
        : Math.min(0, values[contributionLayers[layerIndex].key] ?? 0);
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
      if (!context) {
        URL.revokeObjectURL(svgUrl);
        return;
      }
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
                <title>{`${leaderName} · ${formatChartRating(leaderRating)} ${MODEL_LABEL}`}</title>
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
            <text x="7" y="24">G {count(hoveredPlayerPoint.games)} · GS {count(hoveredPlayerPoint.gamesStarted)}</text>
            <text x="7" y="36">{count(hoveredPlayerPoint.possessions)} possessions</text>
          </g>
        )}
      </svg>
      <figcaption>{showLeagueLeaders
        ? `Completed-fit ${MODEL_LABEL} by age. The green line shows saved season NAIL; colored areas show available saved rating components stacked around zero. Headshots mark each season's league leader.`
        : `Completed-fit ${MODEL_LABEL} by age. The green line shows saved season NAIL; colored areas show available saved rating components stacked around zero.`}
      </figcaption>
    </figure>
  );
}

export async function inlineSvgImages(svg: SVGSVGElement) {
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
