import { useEffect, useMemo, useRef, useState } from "react";
import { Check, Copy, Download, LoaderCircle, Search, X } from "lucide-react";

import type { Player } from "./types";

type Alignment = "age" | "season";
type RatingView = "total" | "offense" | "defense";

type ComparisonCandidate = {
  player_id: number;
  player_name: string;
  latest_season: string;
  latest_team: string;
  history_seasons: number;
};

type ChartPoint = Player["rating_history"][number];
type TrajectoryPoint = ChartPoint & { rating: number };
type HoveredTrajectoryPoint = { playerId: number; season: string };
type ChartRef = { current: SVGSVGElement | null };
type ExportLegendItem = { player: Player; color: string };
type ComparisonTooltipEntry = { player: Player; point: TrajectoryPoint; color: string };

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
const TEAM_LOGO_SLUGS: Record<string, string> = {
  NOP: "no",
  UTA: "utah",
};

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
    alignment: parameters.get("align") === "season" ? "season" as const : "age" as const,
  };
}

function compareHash(playerIds: number[], alignment: Alignment) {
  const parameters = new URLSearchParams({ align: alignment });
  if (playerIds.length) parameters.set("players", playerIds.join(","));
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

function teamLogoUrl(team: string) {
  const slug = TEAM_LOGO_SLUGS[team] ?? team.toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/nba/500/${slug}.png`;
}

function trajectoryPoints(player: Player, view: RatingView): TrajectoryPoint[] {
  const field = RATING_VIEWS[view].field;
  return player.rating_history
    .flatMap((point) => {
      const rating = point[field];
      return point.age !== null && typeof rating === "number" && Number.isFinite(rating)
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
  players: Player[];
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
}: {
  players: Player[];
  alignment: Alignment;
  view: RatingView;
  focusedPlayerId: number | null;
  setFocusedPlayerId: (playerId: number | null) => void;
  hoveredPoint: HoveredTrajectoryPoint | null;
  setHoveredPoint: (point: HoveredTrajectoryPoint | null) => void;
  svgRef?: ChartRef;
  showLegend?: boolean;
  compact?: boolean;
}) {
  const localChartRef = useRef<SVGSVGElement>(null);
  const chartRef = svgRef ?? localChartRef;
  const [hoveredComparisonX, setHoveredComparisonX] = useState<number | null>(null);
  const details = RATING_VIEWS[view];
  const series = useMemo(() => players.map((player, index) => ({
    player,
    color: SERIES_COLORS[index % SERIES_COLORS.length],
    points: trajectoryPoints(player, view),
  })).filter((entry) => entry.points.length > 0), [players, view]);

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
                const isLinked = hoveredPoint?.playerId === player.player_id && hoveredPoint.season === point.season;
                return <g
                  key={point.season}
                  className={isLinked ? "compare-chart-team-point linked" : "compare-chart-team-point"}
                  onPointerEnter={() => setHoveredPoint({ playerId: player.player_id, season: point.season })}
                  onPointerLeave={() => setHoveredPoint(null)}
                >
                  <title>{`${player.player_name}, ${point.season}: ${formatRating(point.rating)}`}</title>
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
                </g>;
              })}
            </g>
          );
        })}
        {hovered && <g className="compare-chart-tooltip" data-export-exclude="true" transform={`translate(${tooltipX} ${tooltipY})`} pointerEvents="none">
          <rect width="188" height="70" rx="2" />
          <text x="10" y="18">{hovered.player.player_name}</text>
          <text x="10" y="37">{hovered.point.team} · {hovered.point.season} · Age {hovered.point.age}</text>
          <text x="10" y="56">{formatRating(hovered.point.rating)}{view === "total" ? ` · Rank #${hovered.point.nail_rank}` : ""}</text>
        </g>}
        {hoveredComparisonX !== null && <g className="compare-chart-comparison-tooltip" data-export-exclude="true" transform={`translate(${comparisonTooltipX} ${comparisonTooltipY})`} pointerEvents="none">
          <rect width={comparisonTooltipWidth} height={comparisonTooltipHeight} rx="2" />
          <text className="compare-chart-comparison-tooltip-title" x="10" y="16">{comparisonLabel} · {details.title.replace(".", "")}</text>
          <text className="compare-chart-comparison-tooltip-columns" x="10" y="31">ORDER  PLAYER  RATING  NAIL RK  TEAM  AGE  SEASON</text>
          {comparisonEntries.length > 0
            ? comparisonEntries.map(({ player, point, color }, index) => <text className="compare-chart-comparison-tooltip-row" key={player.player_id} x="10" y={47 + index * 15}>
              <tspan fill={color}>{`${index + 1}. ${player.player_name}`}</tspan>
              <tspan>{`  ${formatRating(point.rating)} · #${point.nail_rank} · ${point.team} · ${point.age} · ${point.season}`}</tspan>
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

export function ComparePage() {
  const [selectedPlayerIds, setSelectedPlayerIds] = useState(() => readCompareRoute().playerIds);
  const [players, setPlayers] = useState<Player[]>([]);
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<ComparisonCandidate[]>([]);
  const [alignment, setAlignment] = useState<Alignment>(() => readCompareRoute().alignment);
  const [focusedPlayerId, setFocusedPlayerId] = useState<number | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<HoveredTrajectoryPoint | null>(null);
  const [linkCopied, setLinkCopied] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isSearching, setIsSearching] = useState(false);
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
      setFocusedPlayerId(null);
      setHoveredPoint(null);
    };
    window.addEventListener("hashchange", syncFromHash);
    return () => window.removeEventListener("hashchange", syncFromHash);
  }, []);

  useEffect(() => {
    const nextHash = compareHash(selectedPlayerIds, alignment);
    if (window.location.hash === nextHash) return;
    window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}${nextHash}`);
  }, [alignment, selectedPlayerIds]);

  useEffect(() => {
    const controller = new AbortController();
    if (!selectedPlayerIds.length) {
      setError(null);
      setPlayers([]);
      setIsLoading(false);
      return () => controller.abort();
    }
    void (async () => {
      try {
        setError(null);
        setIsLoading(true);
        const responses = await Promise.all(selectedPlayerIds.map(async (playerId) => {
          const response = await fetch(`/api/players/${playerId}`, { signal: controller.signal });
          if (!response.ok) throw new Error("A selected player comparison is unavailable.");
          return (await response.json()) as Player;
        }));
        setPlayers(responses);
      } catch (loadError) {
        if ((loadError as Error).name !== "AbortError") setError((loadError as Error).message);
      } finally {
        if (!controller.signal.aborted) setIsLoading(false);
      }
    })();
    return () => controller.abort();
  }, [selectedPlayerIds]);

  useEffect(() => {
    const normalized = query.trim();
    if (normalized.length < 2) {
      setSuggestions([]);
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
          const payload = (await response.json()) as { players: ComparisonCandidate[] };
          setSuggestions(payload.players.filter((candidate) => !selectedPlayerIds.includes(candidate.player_id)));
        } catch (searchError) {
          if ((searchError as Error).name !== "AbortError") setError((searchError as Error).message);
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
    url.hash = compareHash(selectedPlayerIds, alignment);
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
        <p>Compare career trajectories on a shared NAIL-RAPM scale.</p>
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
            />
            {isSearching && <LoaderCircle className="spin" size={16} aria-label="Searching players" />}
          </div>
          {suggestions.length > 0 && <div className="compare-player-suggestions" role="listbox" aria-label="Player suggestions">
            {suggestions.map((candidate) => (
              <button key={candidate.player_id} type="button" onClick={() => void addPlayer(candidate)}>
                <strong>{candidate.player_name}</strong>
                <span>{candidate.latest_team} · through {candidate.latest_season} · {candidate.history_seasons} seasons</span>
              </button>
            ))}
          </div>}
        </div>
        <div className="compare-alignment-control">
          <span>Align by</span>
          <div className="compare-alignment-actions">
            <div role="group" aria-label="Trajectory alignment">
              <button type="button" aria-pressed={alignment === "age"} onClick={() => setAlignment("age")}>Age</button>
              <button type="button" aria-pressed={alignment === "season"} onClick={() => setAlignment("season")}>Season</button>
            </div>
            <button className="compare-copy-link" type="button" onClick={() => void copyShareLink()} title={linkCopied ? "Comparison link copied" : "Copy comparison link"} aria-label={linkCopied ? "Comparison link copied" : "Copy comparison link"}>
              {linkCopied ? <Check size={16} aria-hidden="true" /> : <Copy size={16} aria-hidden="true" />}
            </button>
          </div>
        </div>
      </section>

      {selectedPlayerIds.length > 0 && <section className="compare-selected-players" aria-label="Selected players">
        <div className="compare-selected-player-list">
          {players.map((player, index) => (
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
                players,
                alignment,
                chartRefs: { total: totalChartRef, offense: offenseChartRef, defense: defenseChartRef },
              })}
              title="Download total, offense, and defense charts as PNG"
            >
              <Download size={16} aria-hidden="true" />
              Download comparison
            </button>
          </div>
        </div>
        <div className="compare-trajectory-layout">
          <div className="compare-total-chart">
            <ComparisonTrajectoryChart
              players={players}
              alignment={alignment}
              view="total"
              focusedPlayerId={focusedPlayerId}
              setFocusedPlayerId={setFocusedPlayerId}
              hoveredPoint={hoveredPoint}
              setHoveredPoint={setHoveredPoint}
              svgRef={totalChartRef}
              showLegend
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
                  players={players}
                  alignment={alignment}
                  view={view}
                  focusedPlayerId={focusedPlayerId}
                  setFocusedPlayerId={setFocusedPlayerId}
                  hoveredPoint={hoveredPoint}
                  setHoveredPoint={setHoveredPoint}
                  svgRef={view === "offense" ? offenseChartRef : defenseChartRef}
                  compact
                />
              </section>
            ))}
          </div>
        </div>
      </section>}
    </article>
  );
}
