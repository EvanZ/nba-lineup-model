import { releaseFetch as fetch } from "./releaseFetch";
import { useEffect, useMemo, useRef, useState } from "react";
import { Check, Copy, Download, LoaderCircle, Search, X } from "lucide-react";

import type { Player } from "./types";
import type { RatingHistory } from "./PlayerRatingHistory";
import { teamLogoUrl } from "./teamLogos";

type SeasonView = "historical" | "current";
type Alignment = "age" | "season";
type RatingView = "total" | "offense" | "defense";

type ComparisonCandidate = {
  player_id: number;
  player_name: string | null;
  latest_season: string;
  latest_team: string | null;
  history_seasons: number;
};

type ChartPoint = Pick<Player["rating_history"][number],
  "season" | "age" | "rating" | "offense_rating" | "defense_rating" | "team" | "team_id" | "nail_rank">;
type ComparePlayer = Pick<Player, "player_id" | "player_name"> & { rating_history: ChartPoint[] };
type TrajectoryPoint = ChartPoint & { rating: number };
type HoveredTrajectoryPoint = { playerId: number; season: string };
type ChartRef = { current: SVGSVGElement | null };
type ExportLegendItem = { player: ComparePlayer; color: string };
type ComparisonTooltipEntry = { player: ComparePlayer; point: TrajectoryPoint; color: string };

const RATING_VIEWS: Record<RatingView, {
  title: string;
  exportTitle: string;
  emptyLabel: string;
  field: "rating" | "offense_rating" | "defense_rating";
  caption: string;
}> = {
  total: {
    title: "NAIL-RAPM history.",
    exportTitle: "NAIL-RAPM trajectory comparison",
    emptyLabel: "NAIL-RAPM",
    field: "rating",
    caption: "Each point is a completed-season NAIL-RAPM estimate. A break in a line denotes a season without a completed fit.",
  },
  offense: {
    title: "Offense.",
    exportTitle: "Offense NAIL-RAPM trajectory comparison",
    emptyLabel: "offensive split",
    field: "offense_rating",
    caption: "Each point is the descriptive constrained-split offensive allocation for a completed season.",
  },
  defense: {
    title: "Defense.",
    exportTitle: "Defense NAIL-RAPM trajectory comparison",
    emptyLabel: "defensive split",
    field: "defense_rating",
    caption: "Each point is the descriptive constrained-split defensive allocation for a completed season.",
  },
};

const DEFAULT_PLAYER_IDS = [1495, 708, 406, 203999];
const MAX_PLAYERS = 8;
const CHART_WIDTH = 1100;
const CHART_HEIGHT = 478;
const COMPACT_CHART_WIDTH = 530;
const COMPACT_CHART_HEIGHT = 240;
const CHART_MARGIN = { top: 30, right: 34, bottom: 54, left: 54 };
const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const SERIES_COLORS = [
  "#176b5a",
  "#3f77a8",
  "#d86732",
  "#a64d67",
  "#6c5f98",
  "#39786f",
  "#b6544a",
  "#55769a",
];

function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);

  useEffect(() => {
    const mediaQuery = window.matchMedia(query);
    const update = () => setMatches(mediaQuery.matches);
    update();
    mediaQuery.addEventListener("change", update);
    return () => mediaQuery.removeEventListener("change", update);
  }, [query]);

  return matches;
}

function readCompareRoute() {
  const queryIndex = window.location.hash.indexOf("?");
  const parameters = new URLSearchParams(queryIndex >= 0 ? window.location.hash.slice(queryIndex + 1) : "");
  const playerParameter = parameters.get("players");
  const playerIds = playerParameter === null
    ? DEFAULT_PLAYER_IDS
    : playerParameter.split(",")
      .map((value) => Number.parseInt(value, 10))
      .filter((value, index, values) => Number.isInteger(value) && value > 0 && values.indexOf(value) === index)
      .slice(0, MAX_PLAYERS);
  return {
    playerIds,
    seasonView: parameters.get("view") === "current" ? "current" as const : "historical" as const,
    alignment: parameters.get("align") === "season" ? "season" as const : "age" as const,
  };
}

function compareHash(playerIds: number[], alignment: Alignment, seasonView: SeasonView) {
  const parameters = new URLSearchParams({ align: alignment, players: playerIds.join(",") });
  if (seasonView === "current") parameters.set("view", "current");
  return `#compare?${parameters.toString()}`;
}

function seasonStartYear(season: string) {
  return Number.parseInt(season.slice(0, 4), 10);
}

function seasonEndYear(season: string) {
  return seasonStartYear(season) + 1;
}

function formatRating(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
}

function playerProfileHref(playerId: number) {
  return `#player/${playerId}`;
}

function headshotUrl(playerId: number) {
  return `/api/headshots/${playerId}.png`;
}

function trajectoryPoints(player: ComparePlayer, view: RatingView, alignment: Alignment): TrajectoryPoint[] {
  const field = RATING_VIEWS[view].field;
  return player.rating_history
    .flatMap((point) => {
      const rating = point[field];
      return (alignment === "season" || point.age !== null) && typeof rating === "number" && Number.isFinite(rating)
        ? [{ ...point, rating }]
        : [];
    })
    .sort((left, right) => seasonStartYear(left.season) - seasonStartYear(right.season));
}

function trajectoryPath(
  points: TrajectoryPoint[],
  x: (point: TrajectoryPoint) => number,
  y: (rating: number) => number,
) {
  return points.map((point, index) => {
    const previous = points[index - 1];
    const startsNewSegment = index === 0
      || seasonStartYear(point.season) !== seasonStartYear(previous.season) + 1;
    return `${startsNewSegment ? "M" : "L"}${x(point)},${y(point.rating)}`;
  }).join(" ");
}

function axisTicks(minimum: number, maximum: number, count = 7) {
  if (minimum === maximum) return [minimum];
  const rawStep = (maximum - minimum) / Math.max(count - 1, 1);
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const step = Math.ceil(rawStep / magnitude) * magnitude;
  const first = Math.ceil(minimum / step) * step;
  const ticks: number[] = [];
  for (let tick = first; tick <= maximum + step * 0.25; tick += step) ticks.push(tick);
  return ticks;
}

function appendExportStyle(svg: SVGSVGElement) {
  const style = document.createElementNS(SVG_NAMESPACE, "style");
  style.textContent = `
    .compare-chart-grid { stroke: #dedbd3; stroke-width: 1; }
    .compare-chart-zero { stroke: #6d766f; stroke-width: 1.15; stroke-dasharray: 6 5; }
    .compare-chart-tick { stroke: #737a73; stroke-width: 1; }
    .compare-chart-minor-tick { stroke: #a2a8a1; stroke-width: 0.8; }
    .compare-chart-linked-rule { stroke: #a9b1a9; stroke-width: 1; stroke-dasharray: 3 4; }
    .compare-chart-y-label, .compare-chart-x-label, .compare-chart-axis-title { fill: #68716a; font-family: monospace; font-size: 12px; font-weight: 700; }
    .compare-chart-axis-title { font-size: 11px; }
    .compare-chart-series path { fill: none; stroke-width: 3.4; stroke-linecap: round; stroke-linejoin: round; }
    .compare-chart-team-ring { fill: #fffefa; stroke: #174d3d; stroke-width: 1.7; }
    .compare-chart-team-fallback { stroke: #fffefa; stroke-width: 1.5; }
    .compare-export-title { fill: #17201c; font-family: sans-serif; font-size: 18px; font-weight: 800; }
    .compare-export-subtitle { fill: #68716a; font-family: sans-serif; font-size: 11px; }
    .compare-export-legend-label { fill: #505952; font-family: sans-serif; font-size: 10px; font-weight: 700; }
    .compare-export-section-title { fill: #17201c; font-family: sans-serif; font-size: 14px; font-weight: 800; }
  `;
  svg.prepend(style);
}

function appendExportLegend(svg: SVGSVGElement, items: ExportLegendItem[], top: number, width = CHART_WIDTH) {
  const definitions = document.createElementNS(SVG_NAMESPACE, "defs");
  svg.append(definitions);
  let legendX = CHART_MARGIN.left;
  let legendY = top;
  items.forEach(({ player, color }, index) => {
    const labelWidth = player.player_name.length * 6.1 + 52;
    if (legendX + labelWidth > width - CHART_MARGIN.right) {
      legendX = CHART_MARGIN.left;
      legendY += 22;
    }
    const swatch = document.createElementNS(SVG_NAMESPACE, "rect");
    swatch.setAttribute("x", String(legendX));
    swatch.setAttribute("y", String(legendY - 10));
    swatch.setAttribute("width", "9");
    swatch.setAttribute("height", "9");
    swatch.setAttribute("fill", color);
    svg.append(swatch);

    const clipId = `compare-export-avatar-${index}`;
    const clipPath = document.createElementNS(SVG_NAMESPACE, "clipPath");
    clipPath.setAttribute("id", clipId);
    const clipCircle = document.createElementNS(SVG_NAMESPACE, "circle");
    clipCircle.setAttribute("cx", String(legendX + 24));
    clipCircle.setAttribute("cy", String(legendY - 5));
    clipCircle.setAttribute("r", "8");
    clipPath.append(clipCircle);
    definitions.append(clipPath);

    const avatar = document.createElementNS(SVG_NAMESPACE, "image");
    avatar.setAttribute("href", headshotUrl(player.player_id));
    avatar.setAttribute("x", String(legendX + 16));
    avatar.setAttribute("y", String(legendY - 13));
    avatar.setAttribute("width", "16");
    avatar.setAttribute("height", "16");
    avatar.setAttribute("preserveAspectRatio", "xMidYMid slice");
    avatar.setAttribute("clip-path", `url(#${clipId})`);
    svg.append(avatar);

    const label = document.createElementNS(SVG_NAMESPACE, "text");
    label.setAttribute("class", "compare-export-legend-label");
    label.setAttribute("x", String(legendX + 36));
    label.setAttribute("y", String(legendY));
    label.textContent = player.player_name;
    svg.append(label);
    legendX += labelWidth;
  });
  return legendY + 22;
}

function appendExportText(svg: SVGSVGElement, className: string, x: number, y: number, value: string) {
  const text = document.createElementNS(SVG_NAMESPACE, "text");
  text.setAttribute("class", className);
  text.setAttribute("x", String(x));
  text.setAttribute("y", String(y));
  text.textContent = value;
  svg.append(text);
}

async function downloadSvgPng(svg: SVGSVGElement, width: number, height: number, filename: string) {
  const svgUrl = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], { type: "image/svg+xml" }));
  const image = new Image();
  await new Promise<void>((resolve) => {
    image.onload = () => {
      const canvas = document.createElement("canvas");
      const scale = 2;
      canvas.width = width * scale;
      canvas.height = height * scale;
      const context = canvas.getContext("2d");
      if (context) {
        context.fillStyle = "#f6f3ec";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
          if (blob) {
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            link.click();
            window.setTimeout(() => URL.revokeObjectURL(link.href), 0);
          }
          resolve();
        }, "image/png");
      } else {
        resolve();
      }
      URL.revokeObjectURL(svgUrl);
    };
    image.onerror = () => {
      URL.revokeObjectURL(svgUrl);
      resolve();
    };
    image.src = svgUrl;
  });
}

async function downloadCombinedComparisonPng({
  players,
  alignment,
  chartRefs,
}: {
  players: ComparePlayer[];
  alignment: Alignment;
  chartRefs: Record<RatingView, ChartRef>;
}) {
  const charts = (Object.keys(RATING_VIEWS) as RatingView[])
    .flatMap((view) => chartRefs[view].current ? [{ view, chart: chartRefs[view].current! }] : []);
  if (!charts.length) return;

  const svg = document.createElementNS(SVG_NAMESPACE, "svg");
  svg.setAttribute("xmlns", SVG_NAMESPACE);
  appendExportText(svg, "compare-export-title", CHART_MARGIN.left, 22, "NBA GESTALT · NAIL-RAPM trajectory comparison");
  appendExportText(svg, "compare-export-subtitle", CHART_MARGIN.left, 39, alignment === "age" ? "Aligned by age" : "Placed in actual seasons");
  let nextY = appendExportLegend(svg, players.map((player, index) => ({
    player,
    color: SERIES_COLORS[index % SERIES_COLORS.length],
  })), 58) + 24;

  const appendChart = (view: RatingView, chart: SVGSVGElement, x: number, y: number) => {
    const details = RATING_VIEWS[view];
    appendExportText(svg, "compare-export-section-title", x + CHART_MARGIN.left, y, details.title.replace(".", ""));
    const copy = chart.cloneNode(true) as SVGSVGElement;
    copy.querySelectorAll("[data-export-exclude]").forEach((element) => element.remove());
    const body = document.createElementNS(SVG_NAMESPACE, "g");
    body.setAttribute("transform", `translate(${x} ${y + 16})`);
    while (copy.firstChild) body.append(copy.firstChild);
    svg.append(body);
  };

  const totalChart = charts.find(({ view }) => view === "total");
  if (totalChart) {
    appendChart(totalChart.view, totalChart.chart, 0, nextY);
    nextY += totalChart.chart.viewBox.baseVal.height + 48;
  }

  const splitCharts = charts.filter(({ view }) => view !== "total");
  if (splitCharts.length) {
    const gap = CHART_WIDTH - splitCharts.reduce((width, { chart }) => width + chart.viewBox.baseVal.width, 0);
    let splitX = 0;
    let splitHeight = 0;
    splitCharts.forEach(({ view, chart }, index) => {
      appendChart(view, chart, splitX, nextY);
      splitX += chart.viewBox.baseVal.width + (index < splitCharts.length - 1 ? gap : 0);
      splitHeight = Math.max(splitHeight, chart.viewBox.baseVal.height);
    });
    nextY += splitHeight + 48;
  }

  svg.setAttribute("viewBox", `0 0 ${CHART_WIDTH} ${nextY - 18}`);
  appendExportStyle(svg);
  await inlineSvgImages(svg);
  await downloadSvgPng(svg, CHART_WIDTH, nextY - 18, `nba-gestalt-nail-rapm-${alignment}-comparison.png`);
}

function ComparisonTrajectoryChart({
  players,
  alignment,
  view,
  focusedPlayerId,
  setFocusedPlayerId,
  hoveredPoint,
  setHoveredPoint,
  svgRef,
  showLegend = false,
  compact = false,
  showTeamLogos = true,
}: {
  players: ComparePlayer[];
  alignment: Alignment;
  view: RatingView;
  focusedPlayerId: number | null;
  setFocusedPlayerId: (playerId: number | null) => void;
  hoveredPoint: HoveredTrajectoryPoint | null;
  setHoveredPoint: (point: HoveredTrajectoryPoint | null) => void;
  svgRef?: ChartRef;
  showLegend?: boolean;
  compact?: boolean;
  showTeamLogos?: boolean;
}) {
  const localChartRef = useRef<SVGSVGElement>(null);
  const chartRef = svgRef ?? localChartRef;
  const [hoveredComparisonX, setHoveredComparisonX] = useState<number | null>(null);
  const details = RATING_VIEWS[view];
  const series = useMemo(() => players.map((player, index) => ({
    player,
    color: SERIES_COLORS[index % SERIES_COLORS.length],
    points: trajectoryPoints(player, view, alignment),
  })).filter((entry) => entry.points.length > 0), [players, view, alignment]);

  if (!series.length) {
    return <p className="compare-empty-chart">Selected players need at least one completed-season {details.emptyLabel} estimate.</p>;
  }

  const xValue = (point: TrajectoryPoint) => alignment === "age" ? point.age! : seasonEndYear(point.season);
  const allPoints = series.flatMap((entry) => entry.points);
  const xValues = allPoints.map(xValue);
  const ratings = allPoints.map((point) => point.rating);
  const xMinimum = Math.min(...xValues);
  const xMaximum = Math.max(...xValues);
  const yMinimum = Math.floor(Math.min(0, ...ratings));
  const yMaximum = Math.ceil(Math.max(0, ...ratings));
  const yLower = yMinimum === yMaximum ? yMinimum - 1 : yMinimum;
  const yUpper = yMinimum === yMaximum ? yMaximum + 1 : yMaximum;
  const width = compact ? COMPACT_CHART_WIDTH : CHART_WIDTH;
  const height = compact ? COMPACT_CHART_HEIGHT : CHART_HEIGHT;
  const margin = CHART_MARGIN;
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const x = (point: ChartPoint) => margin.left + ((xValue(point) - xMinimum) / Math.max(1, xMaximum - xMinimum)) * plotWidth;
  const y = (rating: number) => margin.top + ((yUpper - rating) / Math.max(1, yUpper - yLower)) * plotHeight;
  const yTicks = axisTicks(yLower, yUpper, 6);
  const xTicks = axisTicks(xMinimum, xMaximum, 8)
    .filter((tick) => tick >= xMinimum && tick <= xMaximum);
  const minorXTicks = Array.from(
    { length: Math.max(0, Math.ceil(xMaximum) - Math.floor(xMinimum) + 1) },
    (_, index) => Math.floor(xMinimum) + index,
  );
  const hovered = hoveredPoint
    ? series.flatMap(({ player, points }) => points.map((point) => ({ player, point, x: x(point), y: y(point.rating) })))
      .find(({ player, point }) => player.player_id === hoveredPoint.playerId && point.season === hoveredPoint.season)
    : undefined;
  const tooltipX = hovered ? Math.min(width - 197, hovered.x + 14) : 0;
  const tooltipY = hovered ? Math.max(20, Math.min(height - 82, hovered.y - 66)) : 0;
  const comparisonEntries: ComparisonTooltipEntry[] = hoveredComparisonX === null
    ? []
    : series.flatMap(({ player, points, color }) => points
      .filter((point) => xValue(point) === hoveredComparisonX)
      .map((point) => ({ player, point, color })))
      .sort((left, right) => right.point.rating - left.point.rating || left.player.player_name.localeCompare(right.player.player_name));
  const comparisonTooltipWidth = 340;
  const comparisonTooltipHeight = 48 + Math.max(comparisonEntries.length, 1) * 15;
  const comparisonX = hoveredComparisonX === null
    ? 0
    : margin.left + ((hoveredComparisonX - xMinimum) / Math.max(1, xMaximum - xMinimum)) * plotWidth;
  const comparisonTooltipX = comparisonX > width / 2
    ? Math.max(margin.left, comparisonX - comparisonTooltipWidth - 12)
    : Math.min(width - margin.right - comparisonTooltipWidth, comparisonX + 12);
  const comparisonTooltipY = Math.max(8, Math.min(margin.top + 8, height - margin.bottom - comparisonTooltipHeight));
  const comparisonLabel = alignment === "age" ? `Age ${hoveredComparisonX}` : `Playoff year ${hoveredComparisonX}`;

  async function downloadPng() {
    const svg = chartRef.current;
    if (!svg) return;
    const copy = svg.cloneNode(true) as SVGSVGElement;
    const chartBody = document.createElementNS(SVG_NAMESPACE, "g");
    while (copy.firstChild) chartBody.append(copy.firstChild);
    copy.append(chartBody);
    appendExportText(copy, "compare-export-title", margin.left, 22, `NBA GESTALT · ${details.exportTitle}`);
    appendExportText(copy, "compare-export-subtitle", margin.left, 39, alignment === "age" ? "Aligned by age" : "Placed in actual seasons");
    const exportTop = appendExportLegend(copy, series.map(({ player, color }) => ({ player, color })), 58, width);
    chartBody.setAttribute("transform", `translate(0 ${exportTop})`);
    copy.setAttribute("xmlns", SVG_NAMESPACE);
    copy.setAttribute("viewBox", `0 0 ${width} ${height + exportTop}`);
    copy.querySelectorAll("[data-export-exclude]").forEach((element) => element.remove());
    await inlineSvgImages(copy);
    appendExportStyle(copy);
    await downloadSvgPng(copy, width, height + exportTop, `nba-gestalt-${view}-${alignment}-comparison.png`);
  }

  return (
    <figure className={compact ? "compare-trajectory-chart compare-trajectory-chart-compact" : "compare-trajectory-chart"}>
      <div className={showLegend ? "compare-chart-legend" : "compare-chart-secondary-actions"} aria-label={showLegend ? "Compared players" : undefined}>
        {showLegend && series.map(({ player, color }) => (
          <button
            key={player.player_id}
            className="compare-chart-legend-item"
            type="button"
            aria-pressed={focusedPlayerId === player.player_id}
            onClick={() => setFocusedPlayerId(focusedPlayerId === player.player_id ? null : player.player_id)}
          >
            <i style={{ backgroundColor: color }} />
            <img src={headshotUrl(player.player_id)} alt="" />
            {player.player_name}
          </button>
        ))}
        <button className="compare-chart-download" type="button" onClick={() => void downloadPng()} title={`Download ${details.title.replace(".", "")} chart as PNG`} aria-label={`Download ${details.title.replace(".", "")} chart as PNG`}>
          <Download size={16} aria-hidden="true" />
        </button>
      </div>
      <svg ref={chartRef} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${details.title.replace(".", "")} trajectories aligned by ${alignment}`}>
        <text className="compare-chart-axis-title" x={margin.left} y="14">{details.title.replace(".", "")} / 100 possessions</text>
        {yTicks.map((tick) => (
          <g key={`y-${tick}`}>
            <line className={tick === 0 ? "compare-chart-zero" : "compare-chart-grid"} x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} />
            <text className="compare-chart-y-label" x={margin.left - 12} y={y(tick) + 4} textAnchor="end">{formatRating(tick)}</text>
          </g>
        ))}
        {minorXTicks.map((tick) => {
          const tickX = margin.left + ((tick - xMinimum) / Math.max(1, xMaximum - xMinimum)) * plotWidth;
          const isMajor = xTicks.includes(tick);
          const isActive = hoveredComparisonX === tick;
          return <g
            key={`x-${tick}`}
            className={isActive ? "compare-chart-x-hover active" : "compare-chart-x-hover"}
            onPointerEnter={() => setHoveredComparisonX(tick)}
            onPointerLeave={() => setHoveredComparisonX(null)}
          >
            <line className={isMajor ? "compare-chart-tick compare-chart-major-tick" : "compare-chart-minor-tick"} x1={tickX} x2={tickX} y1={height - margin.bottom} y2={height - margin.bottom + (isMajor ? 5 : 3)} />
            {isMajor && <text className="compare-chart-x-label" x={tickX} y={height - margin.bottom + 25} textAnchor="middle">{String(tick)}</text>}
            <rect data-export-exclude="true" x={tickX - 10} y={height - margin.bottom - 3} width="20" height="37" fill="transparent" />
          </g>
        })}
        <text className="compare-chart-axis-title compare-chart-x-title" x={width - margin.right} y={height - 10} textAnchor="end">{alignment === "age" ? "Age" : "Playoff year"}</text>
        {hovered && <line className="compare-chart-linked-rule" x1={hovered.x} x2={hovered.x} y1={margin.top} y2={height - margin.bottom} />}
        {series.map(({ player, points, color }) => {
          const isDimmed = focusedPlayerId !== null && focusedPlayerId !== player.player_id;
          return (
            <g key={player.player_id} className={isDimmed ? "compare-chart-series dimmed" : "compare-chart-series"}>
              <path d={trajectoryPath(points, x, y)} stroke={color} />
              {points.map((point) => {
                const pointX = x(point);
                const pointY = y(point.rating);
                const logoSize = compact ? 11 : 16;
                const logoRingRadius = compact ? 7.25 : 9;
                const fallbackRadius = compact ? 3.25 : 4.5;
                const plainPointRadius = compact ? 4.25 : 5.5;
                const isLinked = hoveredPoint?.playerId === player.player_id && hoveredPoint.season === point.season;
                return <g
                  key={point.season}
                  className={isLinked ? "compare-chart-team-point linked" : "compare-chart-team-point"}
                  onPointerEnter={() => setHoveredPoint({ playerId: player.player_id, season: point.season })}
                  onPointerLeave={() => setHoveredPoint(null)}
                >
                  <title>{`${player.player_name}, ${point.season}: ${formatRating(point.rating)}`}</title>
                  {showTeamLogos ? <>
                    <circle className="compare-chart-team-ring" cx={pointX} cy={pointY} r={logoRingRadius} />
                    <circle className="compare-chart-team-fallback" cx={pointX} cy={pointY} r={fallbackRadius} fill={color} />
                    {point.team !== "-" && <image
                      className="compare-chart-team-logo"
                      href={teamLogoUrl(point.team)}
                      x={pointX - logoSize / 2}
                      y={pointY - logoSize / 2}
                      width={logoSize}
                      height={logoSize}
                      preserveAspectRatio="xMidYMid meet"
                      crossOrigin="anonymous"
                      onError={(event) => { event.currentTarget.style.display = "none"; }}
                    />}
                  </> : <circle className="compare-chart-plain-point" cx={pointX} cy={pointY} r={plainPointRadius} fill={color} />}
                </g>;
              })}
            </g>
          );
        })}
        {hovered && <g className="compare-chart-tooltip" data-export-exclude="true" transform={`translate(${tooltipX} ${tooltipY})`} pointerEvents="none">
          <rect width="188" height="70" rx="2" />
          <text x="10" y="18">{hovered.player.player_name}</text>
          <text x="10" y="37">{hovered.point.team} · {hovered.point.season}{hovered.point.age === null ? "" : ` · Age ${hovered.point.age}`}</text>
          <text x="10" y="56">{formatRating(hovered.point.rating)}{view === "total" ? ` · Rank #${hovered.point.nail_rank}` : ""}</text>
        </g>}
        {hoveredComparisonX !== null && <g className="compare-chart-comparison-tooltip" data-export-exclude="true" transform={`translate(${comparisonTooltipX} ${comparisonTooltipY})`} pointerEvents="none">
          <rect width={comparisonTooltipWidth} height={comparisonTooltipHeight} rx="2" />
          <text className="compare-chart-comparison-tooltip-title" x="10" y="16">{comparisonLabel} · {details.title.replace(".", "")}</text>
          <text className="compare-chart-comparison-tooltip-columns" x="10" y="31">ORDER  PLAYER  RATING  NAIL RK  TEAM  AGE  SEASON</text>
          {comparisonEntries.length > 0
            ? comparisonEntries.map(({ player, point, color }, index) => <text className="compare-chart-comparison-tooltip-row" key={player.player_id} x="10" y={47 + index * 15}>
              <tspan fill={color}>{`${index + 1}. ${player.player_name}`}</tspan>
              <tspan>{`  ${formatRating(point.rating)} · #${point.nail_rank} · ${point.team} · ${point.age ?? "—"} · ${point.season}`}</tspan>
            </text>)
            : <text className="compare-chart-comparison-tooltip-row" x="10" y="47">No completed fits at this point.</text>}
        </g>}
      </svg>
      <figcaption>{details.caption}</figcaption>
    </figure>
  );
}

async function inlineSvgImages(svg: SVGSVGElement) {
  const images = Array.from(svg.querySelectorAll<SVGImageElement>("image[href]"));
  const imagesBySource = new Map<string, SVGImageElement[]>();
  images.forEach((image) => {
    const source = image.getAttribute("href");
    if (!source || (!source.startsWith("https://") && !source.startsWith("/"))) return;
    imagesBySource.set(source, [...(imagesBySource.get(source) ?? []), image]);
  });
  await Promise.all([...imagesBySource.entries()].map(async ([source, sourceImages]) => {
    try {
      const response = await fetch(source, { mode: "cors" });
      if (!response.ok) throw new Error(`Image request failed: ${response.status}`);
      const blob = await response.blob();
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);
      });
      sourceImages.forEach((image) => image.setAttribute("href", dataUrl));
    } catch {
      sourceImages.forEach((image) => image.remove());
    }
  }));
}

type CurrentRankings = {
  players: { player_id: number; player_name: string; rank: number; rapm: number }[];
  rating_state: { evidence_through_date: string; effective_from_date: string };
};

type CurrentHistory =
  | { data: RatingHistory }
  | { message: string };

function CurrentComparison({ playerIds, players }: {
  playerIds: number[]; players: ComparePlayer[];
}) {
  const [rankings, setRankings] = useState<CurrentRankings | null>(null);
  const [rankingError, setRankingError] = useState(false);
  const [histories, setHistories] = useState<Record<number, CurrentHistory>>({});
  const hasPlayers = playerIds.length > 0;

  useEffect(() => {
    if (!hasPlayers) return;
    const controller = new AbortController();
    const signal = controller.signal;
    void (async () => {
      try {
        const response = await fetch("/api/rankings?season=2026-27", { signal });
        if (!response.ok) throw new Error("No current rankings");
        const data = await response.json() as CurrentRankings;
        if (!signal.aborted) {
          setRankings(data);
          setRankingError(false);
        }
      } catch {
        if (!signal.aborted) setRankingError(true);
      }
    })();
    return () => controller.abort();
  }, [hasPlayers]);

  useEffect(() => {
    const controller = new AbortController();
    const signal = controller.signal;
    for (const id of playerIds) {
      void (async () => {
        let result: CurrentHistory;
        try {
          const response = await fetch(`/api/players/${id}/rating-history`, { signal });
          if (response.status === 404) {
            result = { message: "No saved 2026–27 history" };
          } else {
            if (!response.ok) throw new Error("Current history unavailable");
            const data = await response.json() as RatingHistory;
            if (data.kind !== "saved_player_game_history" || !data.complete || data.player_id !== id) {
              throw new Error("Invalid current history");
            }
            result = { data };
          }
        } catch {
          result = { message: "Saved history is temporarily unavailable" };
        }
        if (!signal.aborted) setHistories((previous) => ({ ...previous, [id]: result }));
      })();
    }
    return () => controller.abort();
  }, [playerIds]);

  const rankedPlayers = new Map(rankings?.players.map((player) => [player.player_id, player]));
  const series = players.flatMap((player, index) => {
    const result = histories[player.player_id];
    if (!result || !("data" in result)) return [];
    const history = result.data;
    return [{ history, color: SERIES_COLORS[index], name: history.player_name }];
  });

  return (
    <section className="compare-chart-section" aria-labelledby="compare-current-title">
      <div className="compare-chart-heading">
        <div>
          <p className="section-kicker">2026–27 season</p>
          <h2 id="compare-current-title">In-season NAIL.</h2>
        </div>
        {rankings && hasPlayers && <p className="compare-search-message">
          Rankings through {rankings.rating_state.evidence_through_date} · Effective {rankings.rating_state.effective_from_date}
        </p>}
      </div>
      {!hasPlayers && <p className="compare-empty-chart">Add a player to compare current-season NAIL.</p>}
      {rankingError && hasPlayers && <p role="status" className="error">Current NAIL rankings are temporarily unavailable.</p>}
      {hasPlayers && <ul className="compare-current-players" aria-label="Latest NAIL and league rank">
        {players.map((player, index) => {
          const id = player.player_id;
          const ranked = rankedPlayers.get(id);
          const result = histories[id];
          const history = result && "data" in result ? result.data : null;
          const historyStatus = result && "message" in result ? result.message :
            history ? `History through ${history.through_game_date_pacific} · ${history.points.length} game dates` : "Loading game history";
          return <li key={id} style={{ borderColor: SERIES_COLORS[index] }}>
            <a href={playerProfileHref(id)}>{ranked?.player_name ?? history?.player_name ?? player.player_name}</a>
            <strong>{ranked ? `${formatRating(ranked.rapm)} · Rank #${ranked.rank}` :
              rankings ? "No current rating" : rankingError ? "Ranking unavailable" : "Loading ranking"}</strong>
            <small>{historyStatus}</small>
          </li>;
        })}
      </ul>}
      {series.length > 0 && <CurrentComparisonChart series={series} />}
    </section>
  );
}

const DAY_MS = 86_400_000;

function CurrentComparisonChart({ series }: {
  series: { history: RatingHistory; color: string; name: string }[];
}) {
  const lines = series.map(({ history, color, name }) => {
    const prior = history.starting_point;
    return {
      id: history.player_id, color, name,
      points: [
        {
          day: Date.parse(prior.first_state_date) / DAY_MS - 0.4,
          date: prior.first_state_date, nail: prior.nail, prior: true,
          label: prior.kind === "opening_prior" ? "Opening prior" : "Introduced prior",
        },
        ...history.points.map((point) => ({
          day: Date.parse(point.game_date_pacific) / DAY_MS,
          date: point.game_date_pacific, nail: point.nail, prior: false, label: "After slate",
        })),
      ],
    };
  });
  const points = lines.flatMap((line) => line.points);
  const days = points.map((point) => point.day);
  const ratings = points.map((point) => point.nail);
  const firstDay = Math.floor(Math.min(...days));
  const lastDay = Math.max(firstDay + 1, Math.ceil(Math.max(...days)));
  const low = Math.floor(Math.min(0, ...ratings));
  const high = Math.max(low + 1, Math.ceil(Math.max(0, ...ratings)));
  const width = 780;
  const height = 310;
  const { left, right, top, bottom } = CHART_MARGIN;
  const x = (day: number) => left + (day - firstDay) * (width - left - right) / (lastDay - firstDay);
  const y = (nail: number) => top + (high - nail) * (height - top - bottom) / (high - low);
  const dateStep = Math.ceil((lastDay - firstDay) / 6);
  const dateTicks = Array.from(
    { length: Math.floor((lastDay - firstDay) / dateStep) + 1 },
    (_, index) => firstDay + index * dateStep,
  );
  const formatDate = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: "UTC" });

  return (
    <figure className="compare-trajectory-chart compare-current-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Current-season NAIL comparison by game date">
        <text className="compare-chart-axis-title" x={left} y={16}>NAIL / 100 possessions</text>
        {axisTicks(low, high, 6).filter((tick) => tick <= high).map((tick) => <g key={tick}>
          <line className={tick === 0 ? "compare-chart-zero" : "compare-chart-grid"} x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} />
          <text className="compare-chart-y-label" x={left - 10} y={y(tick) + 4} textAnchor="end">{formatRating(tick)}</text>
        </g>)}
        {dateTicks.map((day) => <g key={day}>
          <line className="compare-chart-tick" x1={x(day)} x2={x(day)} y1={height - bottom} y2={height - bottom + 5} />
          <text className="compare-chart-x-label" x={x(day)} y={height - bottom + 22} textAnchor="middle">{formatDate.format(day * DAY_MS)}</text>
        </g>)}
        {lines.map((line) => <g className="compare-chart-series" key={line.id}>
          <path stroke={line.color} d={line.points.map((point, index) =>
            `${index ? "L" : "M"}${x(point.day)},${y(point.nail)}`,
          ).join(" ")} />
          {line.points.map((point) => <circle key={point.day} cx={x(point.day)} cy={y(point.nail)}
            r={point.prior ? 5 : 4} fill={point.prior ? "#fffefa" : line.color} stroke={line.color} strokeWidth={2}>
            <title>{`${line.name} · ${point.label} · ${point.date} · ${formatRating(point.nail)}`}</title>
          </circle>)}
        </g>)}
      </svg>
      <figcaption>Open circles mark starting priors. Each filled point is NAIL after the complete slate on a player’s game date.</figcaption>
    </figure>
  );
}

export function ComparePage() {
  const isMobile = useMediaQuery("(max-width: 580px)");
  const [selectedPlayerIds, setSelectedPlayerIds] = useState(() => readCompareRoute().playerIds);
  const [players, setPlayers] = useState<ComparePlayer[]>([]);
  const [seasonView, setSeasonView] = useState<SeasonView>(() => readCompareRoute().seasonView);
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<ComparisonCandidate[]>([]);
  const [searchMessage, setSearchMessage] = useState<string | null>(null);
  const orderedPlayers = selectedPlayerIds.map((id) => players.find((player) => player.player_id === id) ?? {
    player_id: id, player_name: `Player #${id}`, rating_history: [],
  });
  const [alignment, setAlignment] = useState<Alignment>(() => readCompareRoute().alignment);
  const [focusedPlayerId, setFocusedPlayerId] = useState<number | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<HoveredTrajectoryPoint | null>(null);
  const [linkCopied, setLinkCopied] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isSearching, setIsSearching] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const totalChartRef = useRef<SVGSVGElement>(null);
  const offenseChartRef = useRef<SVGSVGElement>(null);
  const defenseChartRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const syncFromHash = () => {
      if (!window.location.hash.startsWith("#compare")) return;
      const next = readCompareRoute();
      setSelectedPlayerIds(next.playerIds);
      setAlignment(next.alignment);
      setSeasonView(next.seasonView);
      setFocusedPlayerId(null);
      setHoveredPoint(null);
    };
    window.addEventListener("hashchange", syncFromHash);
    return () => window.removeEventListener("hashchange", syncFromHash);
  }, []);

  useEffect(() => {
    const nextHash = compareHash(selectedPlayerIds, alignment, seasonView);
    if (window.location.hash === nextHash) return;
    window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}${nextHash}`);
  }, [alignment, selectedPlayerIds, seasonView]);

  useEffect(() => {
    const controller = new AbortController();
    const signal = controller.signal;
    setHistoryError(null);
    setIsLoading(selectedPlayerIds.length > 0);
    void Promise.allSettled(selectedPlayerIds.map(async (playerId): Promise<ComparePlayer> => {
      const response = await fetch(`/api/players/${playerId}/season-history`, { signal });
      if (!response.ok) throw new Error("Season history unavailable.");
      const data = await response.json() as {
        kind: string; player_id: number; player_name: string | null; seasons: ChartPoint[];
      };
      if (data.kind !== "saved_player_season_history" || data.player_id !== playerId || !Array.isArray(data.seasons)) {
        throw new Error("Invalid season history.");
      }
      let name = data.player_name;
      if (!name) {
        const summary = await fetch(`/api/players/${playerId}`, { signal });
        if (summary.ok) name = (await summary.json()).player_name;
      }
      return { player_id: playerId, player_name: name ?? `Player #${playerId}`, rating_history: data.seasons };
    })).then((results) => {
      if (signal.aborted) return;
      setPlayers((previous) => results.flatMap((result, index) =>
        result.status === "fulfilled" ? [result.value] : previous.filter((p) => p.player_id === selectedPlayerIds[index]),
      ));
      const unavailable = results.flatMap((result, index) =>
        result.status === "rejected" ? [selectedPlayerIds[index]] : [],
      );
      if (unavailable.length) setHistoryError(`Saved season history unavailable for player IDs ${unavailable.join(", ")}.`);
      setIsLoading(false);
    });
    return () => controller.abort();
  }, [selectedPlayerIds]);

  useEffect(() => {
    const normalized = query.trim();
    setSearchMessage(null);
    setSuggestions([]);
    if (normalized.length < 2) {
      setIsSearching(false);
      return;
    }
    const controller = new AbortController();
    const timeout = window.setTimeout(() => {
      void (async () => {
        try {
          setIsSearching(true);
          const response = await fetch(`/api/compare/players?q=${encodeURIComponent(normalized)}`, { signal: controller.signal });
          if (!response.ok) throw new Error("Player search is unavailable.");
          const payload = (await response.json()) as { players: ComparisonCandidate[]; unavailable: string[] };
          if (controller.signal.aborted) return;
          const candidates = payload.players.filter((candidate) => !selectedPlayerIds.includes(candidate.player_id));
          setSuggestions(candidates);
          if (payload.unavailable.length) setSearchMessage(`Search is missing ${payload.unavailable.join(" and ")} players.`);
          else if (!candidates.length) setSearchMessage("No matching players to add.");
        } catch (searchError) {
          if (!controller.signal.aborted && (searchError as Error).name !== "AbortError") setSearchMessage((searchError as Error).message);
        } finally {
          if (!controller.signal.aborted) setIsSearching(false);
        }
      })();
    }, 180);
    return () => {
      controller.abort();
      window.clearTimeout(timeout);
    };
  }, [query, selectedPlayerIds]);

  function addPlayer(candidate: ComparisonCandidate) {
    if (selectedPlayerIds.length >= MAX_PLAYERS || selectedPlayerIds.includes(candidate.player_id)) return;
    setSelectedPlayerIds((current) => [...current, candidate.player_id]);
    setPlayers((current) => [...current, {
      player_id: candidate.player_id, player_name: candidate.player_name ?? `Player #${candidate.player_id}`, rating_history: [],
    }]);
    setQuery("");
    setSuggestions([]);
  }

  function removePlayer(playerId: number) {
    setSelectedPlayerIds((current) => current.filter((id) => id !== playerId));
    if (focusedPlayerId === playerId) setFocusedPlayerId(null);
    if (hoveredPoint?.playerId === playerId) setHoveredPoint(null);
  }

  async function copyShareLink() {
    const url = new URL(window.location.href);
    url.hash = compareHash(selectedPlayerIds, alignment, seasonView);
    try {
      await navigator.clipboard.writeText(url.toString());
      setLinkCopied(true);
      window.setTimeout(() => setLinkCopied(false), 1800);
    } catch {
      setError("Could not copy the comparison link.");
    }
  }

  return (
    <article className="compare-page" aria-labelledby="compare-title">
      <section className="compare-hero">
        <p className="eyebrow">Player comparison</p>
        <h1 id="compare-title">NAIL through time.</h1>
        <p>Compare completed seasons or follow current-season NAIL on each player’s game dates.</p>
      </section>

      <section className="compare-controls" aria-label="Comparison controls">
        <div className="compare-player-search">
          <label htmlFor="compare-player-search">Players <span>{selectedPlayerIds.length} / {MAX_PLAYERS}</span></label>
          <div className="compare-search-input">
            <Search size={18} aria-hidden="true" />
            <input
              id="compare-player-search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Add a player"
              disabled={selectedPlayerIds.length >= MAX_PLAYERS}
              autoComplete="off"
              maxLength={128}
            />
            {isSearching && <LoaderCircle className="spin" size={16} aria-label="Searching players" />}
          </div>
          {suggestions.length > 0 && <div className="compare-player-suggestions" role="listbox" aria-label="Player suggestions">
            {suggestions.map((candidate) => (
              <button key={candidate.player_id} type="button" onClick={() => void addPlayer(candidate)}>
                <strong>{candidate.player_name ?? `Player #${candidate.player_id}`}</strong>
                <span>{[candidate.latest_team === "-" ? null : candidate.latest_team, candidate.latest_season,
                  candidate.history_seasons ? `${candidate.history_seasons} completed seasons` : "Current season only",
                ].filter(Boolean).join(" · ")}</span>
              </button>
            ))}
          </div>}
          {searchMessage && <p className="compare-search-message" role="status">{searchMessage}</p>}
        </div>
        <div className="compare-alignment-control">
          <span>View</span>
          <div className="compare-alignment-actions">
            <div role="group" aria-label="Comparison period">
              <button type="button" aria-pressed={seasonView === "historical"} onClick={() => setSeasonView("historical")}>Career</button>
              <button type="button" aria-pressed={seasonView === "current"} onClick={() => setSeasonView("current")}>2026–27</button>
            </div>
            {seasonView === "historical" && <div role="group" aria-label="Trajectory alignment">
              <button type="button" aria-pressed={alignment === "age"} onClick={() => setAlignment("age")}>Age</button>
              <button type="button" aria-pressed={alignment === "season"} onClick={() => setAlignment("season")}>Season</button>
            </div>}
            <button className="compare-copy-link" type="button" onClick={() => void copyShareLink()} title={linkCopied ? "Comparison link copied" : "Copy comparison link"} aria-label={linkCopied ? "Comparison link copied" : "Copy comparison link"}>
              {linkCopied ? <Check size={16} aria-hidden="true" /> : <Copy size={16} aria-hidden="true" />}
            </button>
          </div>
        </div>
      </section>

      {selectedPlayerIds.length > 0 && <section className="compare-selected-players" aria-label="Selected players">
        <div className="compare-selected-player-list">
          {orderedPlayers.map((player, index) => (
            <div className="compare-player-chip" key={player.player_id}>
              <i style={{ backgroundColor: SERIES_COLORS[index % SERIES_COLORS.length] }} />
              <img src={headshotUrl(player.player_id)} alt="" />
              <a href={playerProfileHref(player.player_id)}>{player.player_name}</a>
              <button type="button" onClick={() => removePlayer(player.player_id)} aria-label={`Remove ${player.player_name}`}><X size={15} /></button>
            </div>
          ))}
        </div>
        <button className="compare-clear-all" type="button" onClick={() => { setSelectedPlayerIds([]); setFocusedPlayerId(null); setHoveredPoint(null); }}>
          <X size={14} aria-hidden="true" />
          Clear all
        </button>
      </section>}

      {error && <p className="error compare-error">{error}</p>}
      {seasonView === "current" && <CurrentComparison playerIds={selectedPlayerIds} players={orderedPlayers} />}
      {seasonView === "historical" && <>
        {historyError && <p className="error compare-error" role="status">{historyError}</p>}
        {isLoading ? <div className="compare-loading"><LoaderCircle className="spin" size={20} /> Loading trajectories</div> : <section className="compare-chart-section" aria-labelledby="compare-chart-title">
          <div className="compare-chart-heading">
            <div>
              <p className="section-kicker">Completed fits</p>
              <h2 id="compare-chart-title">NAIL-RAPM history.</h2>
            </div>
            <div className="compare-chart-heading-actions">
              <p>{alignment === "age" ? "Careers aligned at the same age." : "Careers placed in their actual seasons."}</p>
              <button
                className="compare-combined-download"
                type="button"
                onClick={() => void downloadCombinedComparisonPng({
                  players: orderedPlayers,
                  alignment,
                  chartRefs: { total: totalChartRef, offense: offenseChartRef, defense: defenseChartRef },
                })}
                disabled={orderedPlayers.every((player) => trajectoryPoints(player, "total", alignment).length === 0)}
                title="Download total, offense, and defense charts as PNG"
              >
                <Download size={16} aria-hidden="true" />
                Download comparison
              </button>
            </div>
          </div>
          {orderedPlayers.some((player) => player.rating_history.length === 0) && (
            <p className="compare-search-message">No completed-season points shown for {orderedPlayers.filter((player) =>
              player.rating_history.length === 0,
            ).map((player) => player.player_name).join(", ")}.</p>
          )}
          <div className="compare-trajectory-layout">
            <div className="compare-total-chart">
              <ComparisonTrajectoryChart
                players={orderedPlayers}
                alignment={alignment}
                view="total"
                focusedPlayerId={focusedPlayerId}
                setFocusedPlayerId={setFocusedPlayerId}
                hoveredPoint={hoveredPoint}
                setHoveredPoint={setHoveredPoint}
                svgRef={totalChartRef}
                showLegend
                compact={isMobile}
                showTeamLogos={!isMobile}
              />
            </div>
            <div className="compare-split-charts">
              {(["offense", "defense"] as const).map((view) => (
                <section className="compare-split-chart" key={view} aria-labelledby={`compare-${view}-chart-title`}>
                  <div className="compare-split-chart-heading">
                    <p className="section-kicker">Constrained split</p>
                    <h3 id={`compare-${view}-chart-title`}>{RATING_VIEWS[view].title}</h3>
                  </div>
                  <ComparisonTrajectoryChart
                    players={orderedPlayers}
                    alignment={alignment}
                    view={view}
                    focusedPlayerId={focusedPlayerId}
                    setFocusedPlayerId={setFocusedPlayerId}
                    hoveredPoint={hoveredPoint}
                    setHoveredPoint={setHoveredPoint}
                    svgRef={view === "offense" ? offenseChartRef : defenseChartRef}
                    compact
                    showTeamLogos={!isMobile}
                  />
                </section>
              ))}
            </div>
          </div>
        </section>}
      </>}
    </article>
  );
}
