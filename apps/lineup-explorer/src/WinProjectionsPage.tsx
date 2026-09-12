import { type PointerEvent, useEffect, useMemo, useRef, useState } from "react";
import { ArrowUpRight, Calculator, ChevronDown, ChevronUp, CircleAlert, Download, LoaderCircle, RotateCcw } from "lucide-react";

import type {
  MinutesProjectionPayload,
  MinutesProjectionPlayer,
  WinProjectionPayload,
  WinProjectionScheduledGame,
  WinProjectionTeam,
} from "./types";

type PlayerInputOverride = {
  availabilityProbability?: number;
  conditionalMinutesPerGame?: number;
  nailRating?: number;
};

type Overrides = Record<string, PlayerInputOverride>;
type ForecastSortColumn = "team" | "team_strength" | "betmgm_win_total" | "projected_wins" | "delta";
type SortDirection = "ascending" | "descending";

type MinuteAllocation = {
  minutes: Map<number, number>;
  inputs: Map<number, PlayerInputs>;
  rotationPlayerIds: Set<number>;
  totalRawMinutes: number;
};

type PlayerInputs = {
  availabilityProbability: number;
  conditionalMinutesPerGame: number;
  nailRating: number;
};

// The active-15 baseline changes the meaning of every residual allocation.
// Do not carry manual judgments made against the earlier all-roster baseline.
const STORAGE_KEY = "nba-gestalt:win-projection-input-overrides:v3";
const SEASON_GAMES = 82;
const ENVELOPE_TRIALS = 10_000;
const WIN_PROJECTIONS_DOCUMENTATION_URL = "https://evanz.github.io/nba-lineup-model/rotation-models/win-projections/";

type WinLossEnvelopePoint = {
  game: number;
  month: string | null;
  opponent: string;
  opponentTeam: string | null;
  backToBack: boolean;
  winProbability: number;
  p1: number;
  p5: number;
  median: number;
  p95: number;
  p99: number;
};

type WinLossEnvelope = {
  points: WinLossEnvelopePoint[];
  gameCount: number;
};

type WinLossEnvelopeGame = {
  gameDate: string;
  gameId: string;
  opponent: string;
  opponentTeam: string | null;
  backToBack: boolean;
  winProbability: number;
};

const MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const TEAM_LOGO_SLUGS: Record<string, string> = {
  NOP: "no",
  UTA: "utah",
};

function monthLabel(date: string) {
  const month = Number(date.slice(5, 7));
  return MONTH_LABELS[month - 1] ?? "";
}

function teamLogoUrl(team: string) {
  const slug = TEAM_LOGO_SLUGS[team] ?? team.toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/nba/500/${slug}.png`;
}

function escapeSvg(value: string) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function blobAsDataUrl(blob: Blob) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("Could not read team logo."));
    reader.readAsDataURL(blob);
  });
}

async function loadTeamLogoDataUrls(teams: string[]) {
  const entries = await Promise.all([...new Set(teams)].map(async (team) => {
    try {
      const response = await fetch(teamLogoUrl(team));
      if (!response.ok) return [team, ""] as const;
      return [team, await blobAsDataUrl(await response.blob())] as const;
    } catch {
      return [team, ""] as const;
    }
  }));
  return new Map(entries);
}

async function downloadSvgAsPng(svg: string, filename: string) {
  const sourceUrl = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
  try {
    const image = new Image();
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error("Could not render the Win-Loss Envelope."));
      image.src = sourceUrl;
    });
    const canvas = document.createElement("canvas");
    canvas.width = 2400;
    canvas.height = 1600;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("PNG export is unavailable in this browser.");
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const png = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Could not encode the PNG.")), "image/png");
    });
    const downloadUrl = URL.createObjectURL(png);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(downloadUrl);
  } finally {
    URL.revokeObjectURL(sourceUrl);
  }
}

const WIN_LOSS_ENVELOPE_EXPORT_STYLE = `
  text { fill: #69716b; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 9px; font-weight: 700; }
  .export-brand { fill: #69716b; font-size: 17px; font-weight: 800; }
  .export-title { fill: #17201c; font-family: Inter, ui-sans-serif, system-ui, sans-serif; font-size: 32px; font-weight: 800; }
  .export-subtitle { fill: #69716b; font-size: 15px; }
  .export-legend text { fill: #4f5851; font-size: 12px; }
  .export-endpoint-label { fill: #69716b; font-size: 12px; font-weight: 800; }
  .export-endpoint-value { fill: #245a47; font-size: 25px; font-weight: 900; }
  line { stroke: #d9d6ce; stroke-width: 1; }
  .win-loss-envelope-axis-label { fill: #4f5851; font-size: 10px; font-weight: 800; text-transform: uppercase; }
  .win-loss-envelope-month-tick { stroke: #8d948e; stroke-width: 1; }
  .win-loss-envelope-month-label { fill: #4f5851; font-size: 9px; font-weight: 800; }
  .win-loss-envelope-watermark { opacity: 0.08; }
  .win-loss-envelope-outer-band { fill: #c9e0d4; opacity: 0.32; }
  .win-loss-envelope-band { fill: #c9e0d4; opacity: 0.78; }
  .win-loss-envelope-outer-boundary { fill: none; stroke: #9aafa3; stroke-dasharray: 2 4; stroke-width: 1; }
  .win-loss-envelope-boundary { fill: none; stroke: #6e8d7d; stroke-dasharray: 4 4; stroke-width: 1.25; }
  .win-loss-envelope-median { fill: none; stroke: #245a47; stroke-linecap: round; stroke-linejoin: round; stroke-width: 2.5; }
  .win-loss-envelope-tough-opponent line { stroke: #e8502f; stroke-dasharray: 2 3; stroke-width: 1; opacity: 0.62; }
  .win-loss-envelope-tough-game { fill: #e8502f; stroke: #fffefa; stroke-width: 1.5; }
  .win-loss-envelope-hit-area { fill: transparent; }
`;

function overrideKey(team: string, playerId: number) {
  return `${team}:${playerId}`;
}

function formatRating(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
}

function marketDirection(projectedWins: number, total: number | undefined) {
  if (total === undefined) return "";
  if (Math.abs(projectedWins - total) < 0.05) return "P";
  return projectedWins > total ? "O" : "U";
}

function marketDelta(team: WinProjectionTeam) {
  return team.betmgm_win_total === undefined
    ? null
    : team.projected_wins - team.betmgm_win_total;
}

function marketIntervalPosition(team: WinProjectionTeam) {
  if (team.betmgm_win_total === undefined) return "inside";
  if (team.betmgm_win_total < team.win_total_p5) return "below";
  if (team.betmgm_win_total > team.win_total_p95) return "above";
  return "inside";
}

function marketIntervalDescription(team: WinProjectionTeam) {
  const position = marketIntervalPosition(team);
  if (position === "below") return "BetMGM total is below the model's 5th percentile.";
  if (position === "above") return "BetMGM total is above the model's 95th percentile.";
  return "BetMGM total is within the model's 5th to 95th percentile range.";
}

function sigmoid(value: number) {
  return 1 / (1 + Math.exp(-Math.max(-35, Math.min(35, value))));
}

function scheduledHomeWinProbability(
  game: WinProjectionScheduledGame,
  teamStrength: Map<string, number>,
  winProbabilityScale: number,
  homeCourt: number,
  backToBack: number,
) {
  const edge = (teamStrength.get(game.home_team) ?? 0) - (teamStrength.get(game.away_team) ?? 0)
    + homeCourt
    + backToBack * (game.home_back_to_back - game.away_back_to_back);
  return sigmoid(winProbabilityScale * edge);
}

function hashSeed(value: string) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash = Math.imul(hash ^ value.charCodeAt(index), 16777619);
  }
  return hash >>> 0;
}

function seededRandom(seed: number) {
  let state = seed;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}

function percentileFromHistogram(
  histogram: Uint32Array,
  offset: number,
  binCount: number,
  trials: number,
  percentile: number,
) {
  const threshold = percentile * (trials - 1);
  let cumulative = 0;
  for (let value = 0; value < binCount; value += 1) {
    cumulative += histogram[offset + value];
    if (cumulative > threshold) return value;
  }
  return binCount - 1;
}

function simulateWinLossEnvelope(forecast: WinProjectionPayload, team: string): WinLossEnvelope | null {
  const teamStrength = new Map(forecast.teams.map((item) => [item.team, item.team_strength]));
  const scheduledGames: WinLossEnvelopeGame[] = forecast.scheduled_games
    .filter((game) => game.home_team === team || game.away_team === team)
    .map((game) => {
      const homeProbability = scheduledHomeWinProbability(
        game,
        teamStrength,
        forecast.win_probability_scale,
        forecast.home_court,
        forecast.back_to_back,
      );
      const home = team === game.home_team;
      return {
        gameDate: game.game_date,
        gameId: game.game_id,
        opponent: home ? `vs ${game.away_team}` : `@ ${game.home_team}`,
        opponentTeam: home ? game.away_team : game.home_team,
        backToBack: home ? game.home_back_to_back === 1 : game.away_back_to_back === 1,
        winProbability: home ? homeProbability : 1 - homeProbability,
      };
    });
  const cupGames: WinLossEnvelopeGame[] = Array.from(
    { length: forecast.unassigned_regular_games_per_team },
    (_, index) => {
      const homeCourt = index % 2 === 0 ? forecast.home_court : -forecast.home_court;
      return {
        gameDate: index === 0 ? "2026-12-04" : "2026-12-10",
        gameId: `NBA-CUP-${team}-${index + 1}`,
        opponent: "NBA CUP",
        opponentTeam: null,
        backToBack: false,
        winProbability: sigmoid(forecast.win_probability_scale * ((teamStrength.get(team) ?? 0) + homeCourt)),
      };
    },
  );
  const games = [...scheduledGames, ...cupGames].sort((left, right) => (
    left.gameDate.localeCompare(right.gameDate) || left.gameId.localeCompare(right.gameId)
  ));
  const probabilities = games.map((game) => game.winProbability);
  if (!probabilities.length) return null;

  const binCount = probabilities.length + 1;
  const histogram = new Uint32Array(probabilities.length * binCount);
  const random = seededRandom(hashSeed(`${team}:${probabilities.map((value) => value.toFixed(6)).join(",")}`));
  for (let trial = 0; trial < ENVELOPE_TRIALS; trial += 1) {
    let wins = 0;
    for (let game = 0; game < probabilities.length; game += 1) {
      if (random() < probabilities[game]) wins += 1;
      histogram[game * binCount + wins] += 1;
    }
  }

  const points: WinLossEnvelopePoint[] = [{
    game: 0, month: null, opponent: "", opponentTeam: null, backToBack: false, winProbability: 0, p1: 0, p5: 0, median: 0, p95: 0, p99: 0,
  }];
  for (let game = 0; game < probabilities.length; game += 1) {
    const offset = game * binCount;
    const currentMonth = games[game].gameDate.slice(0, 7);
    const previousMonth = games[game - 1]?.gameDate.slice(0, 7);
    points.push({
      game: game + 1,
      month: currentMonth === previousMonth ? null : monthLabel(games[game].gameDate),
      opponent: games[game].opponent,
      opponentTeam: games[game].opponentTeam,
      backToBack: games[game].backToBack,
      winProbability: games[game].winProbability,
      p1: percentileFromHistogram(histogram, offset, binCount, ENVELOPE_TRIALS, 0.01),
      p5: percentileFromHistogram(histogram, offset, binCount, ENVELOPE_TRIALS, 0.05),
      median: percentileFromHistogram(histogram, offset, binCount, ENVELOPE_TRIALS, 0.5),
      p95: percentileFromHistogram(histogram, offset, binCount, ENVELOPE_TRIALS, 0.95),
      p99: percentileFromHistogram(histogram, offset, binCount, ENVELOPE_TRIALS, 0.99),
    });
  }
  return { points, gameCount: probabilities.length };
}

function attachBrowserWinTotalIntervals(forecast: WinProjectionPayload): WinProjectionPayload {
  const intervals = new Map(forecast.teams.flatMap((team) => {
    const endpoint = simulateWinLossEnvelope(forecast, team.team)?.points.at(-1);
    return endpoint ? [[team.team, endpoint] as const] : [];
  }));
  return {
    ...forecast,
    win_total_interval_trials: ENVELOPE_TRIALS,
    teams: forecast.teams.map((team) => {
      const endpoint = intervals.get(team.team);
      if (!endpoint) return team;
      return {
        ...team,
        win_total_p1: endpoint.p1,
        win_total_p5: endpoint.p5,
        win_total_p50: endpoint.median,
        win_total_p95: endpoint.p95,
        win_total_p99: endpoint.p99,
      };
    }),
  };
}

function WinLossEnvelopeChart({ forecast, team }: { forecast: WinProjectionPayload; team: string }) {
  const envelope = useMemo(() => simulateWinLossEnvelope(forecast, team), [forecast, team]);
  const chartRef = useRef<SVGSVGElement>(null);
  const [hoveredGame, setHoveredGame] = useState<number | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  if (!envelope) return null;

  const width = 560;
  const height = 300;
  const margin = { top: 22, right: 20, bottom: 52, left: 38 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  const x = (game: number) => margin.left + chartWidth * game / envelope.gameCount;
  const y = (wins: number) => margin.top + chartHeight * (1 - wins / envelope.gameCount);
  const line = (key: "p1" | "p5" | "median" | "p95" | "p99") => envelope.points.map(
    (point) => `${x(point.game)},${y(point[key])}`,
  ).join(" ");
  const band = (low: "p1" | "p5", high: "p95" | "p99") => [
    ...envelope.points.map((point) => `${x(point.game)},${y(point[high])}`),
    ...[...envelope.points].reverse().map((point) => `${x(point.game)},${y(point[low])}`),
  ].join(" ");
  const yTicks = [0, 20, 40, 60, 80].filter((tick) => tick <= envelope.gameCount);
  const xTicks = [0, 20, 40, 60, 80].filter((tick) => tick <= envelope.gameCount);
  const monthTicks = envelope.points.filter((point) => point.month);
  const toughestGames = new Set(
    envelope.points.slice(1)
      .sort((left, right) => left.winProbability - right.winProbability || left.game - right.game)
      .slice(0, 5)
      .map((point) => point.game),
  );
  const final = envelope.points.at(-1)!;
  const hovered = hoveredGame === null ? null : envelope.points[hoveredGame];
  const tooltipWidth = 174;
  const tooltipHeight = 136;
  const tooltipX = hovered ? Math.min(x(hovered.game) + 10, width - margin.right - tooltipWidth) : 0;
  const tooltipY = hovered ? Math.max(margin.top + 4, y(hovered.median) - tooltipHeight - 8) : 0;
  const onPointerMove = (event: PointerEvent<SVGRectElement>) => {
    const svg = event.currentTarget.ownerSVGElement;
    if (!svg) return;
    const bounds = svg.getBoundingClientRect();
    const position = (event.clientX - bounds.left) * width / bounds.width;
    const game = Math.max(1, Math.min(
      envelope.gameCount,
      Math.round((position - margin.left) * envelope.gameCount / chartWidth),
    ));
    setHoveredGame(game);
  };
  const record = (wins: number, game: number) => `${wins} W · ${game - wins} L`;
  const downloadPng = async () => {
    if (!chartRef.current || isExporting) return;
    setIsExporting(true);
    setExportError(null);
    try {
      const logos = await loadTeamLogoDataUrls([
        team,
        ...envelope.points.flatMap((point) => point.opponentTeam ? [point.opponentTeam] : []),
      ]);
      const inlinedByUrl = new Map(
        Array.from(logos, ([logoTeam, dataUrl]) => [teamLogoUrl(logoTeam), dataUrl]),
      );
      const chart = chartRef.current.cloneNode(true) as SVGSVGElement;
      chart.querySelectorAll("image").forEach((image) => {
        const source = image.getAttribute("href");
        const dataUrl = source ? inlinedByUrl.get(source) : undefined;
        if (dataUrl) image.setAttribute("href", dataUrl);
        else image.remove();
      });
      const exportSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 800">
        <style>${WIN_LOSS_ENVELOPE_EXPORT_STYLE}</style>
        <rect width="1200" height="800" fill="#f8f6ef"/>
        <text class="export-brand" x="44" y="50">NBA GESTALT</text>
        <text class="export-title" x="44" y="87">${escapeSvg(team)} Win-Loss Envelope</text>
        <text class="export-subtitle" x="44" y="114">${escapeSvg(forecast.season)} preseason outlook · ${envelope.gameCount} modeled games · ${ENVELOPE_TRIALS.toLocaleString()} trials</text>
        <g class="export-legend" transform="translate(620 43)">
          <rect x="0" y="0" width="24" height="13" fill="#c9e0d4" opacity="0.32"/><text x="31" y="11" font-size="12">P1–P99</text>
          <rect x="132" y="0" width="24" height="13" fill="#c9e0d4" opacity="0.78"/><text x="163" y="11" font-size="12">P5–P95</text>
          <line x1="0" x2="24" y1="31" y2="31" stroke="#245a47" stroke-width="3"/><text x="31" y="35" font-size="12">Median</text>
          <circle cx="144" cy="31" r="5" fill="#e8502f" stroke="#fffefa" stroke-width="1.5"/><text x="157" y="35" font-size="12">Five toughest games</text>
        </g>
        <g transform="translate(40 145) scale(2)">${chart.innerHTML}</g>
        <line x1="40" x2="1160" y1="754" y2="754" stroke="#d9d6ce" stroke-width="1"/>
        <g text-anchor="middle">
          <text class="export-endpoint-label" x="152" y="772">P1</text><text class="export-endpoint-value" x="152" y="792">${final.p1}</text>
          <text class="export-endpoint-label" x="376" y="772">P5</text><text class="export-endpoint-value" x="376" y="792">${final.p5}</text>
          <text class="export-endpoint-label" x="600" y="772">Median</text><text class="export-endpoint-value" x="600" y="792">${final.median}</text>
          <text class="export-endpoint-label" x="824" y="772">P95</text><text class="export-endpoint-value" x="824" y="792">${final.p95}</text>
          <text class="export-endpoint-label" x="1048" y="772">P99</text><text class="export-endpoint-value" x="1048" y="792">${final.p99}</text>
        </g>
      </svg>`;
      await downloadSvgAsPng(
        exportSvg,
        `${team.toLowerCase()}-${forecast.season.replaceAll(/[^0-9]+/g, "-")}-win-loss-envelope.png`,
      );
    } catch (error) {
      setExportError((error as Error).message);
    } finally {
      setIsExporting(false);
    }
  };

  return <section className="win-loss-envelope" aria-labelledby="win-loss-envelope-title">
    <div className="win-loss-envelope-heading">
      <div><p className="eyebrow">{team} outlook</p><h2 id="win-loss-envelope-title">Win-Loss Envelope.</h2></div>
      <div className="win-loss-envelope-meta">
        <p>{envelope.gameCount} modeled games · {ENVELOPE_TRIALS.toLocaleString()} trials</p>
        <button type="button" className="win-loss-envelope-download" disabled={isExporting} onClick={() => void downloadPng()}>
          {isExporting ? <LoaderCircle className="spin" size={13} /> : <Download size={13} />}
          <span>{isExporting ? "Building PNG" : "Download PNG"}</span>
        </button>
      </div>
    </div>
    {exportError && <p className="win-loss-envelope-export-error"><CircleAlert size={14} /> {exportError}</p>}
    <svg ref={chartRef} className="win-loss-envelope-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${team} cumulative wins through ${envelope.gameCount} known games, with 1st to 99th and 5th to 95th percentile prediction intervals`}>
      {yTicks.map((tick) => <g key={`y-${tick}`}>
        <line x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} />
        <text x={margin.left - 7} y={y(tick) + 4} textAnchor="end">{tick}</text>
      </g>)}
      {xTicks.map((tick) => <g key={`x-${tick}`}>
        <text x={x(tick)} y={height - margin.bottom + 20} textAnchor="middle">{tick}</text>
      </g>)}
      {monthTicks.map((point) => <g key={`month-${point.game}`}>
        <line className="win-loss-envelope-month-tick" x1={x(point.game)} x2={x(point.game)} y1={height - margin.bottom} y2={height - margin.bottom + 5} />
        <text className="win-loss-envelope-month-label" x={x(point.game)} y={height - margin.bottom + 37} textAnchor="middle">{point.month}</text>
      </g>)}
      <image
        aria-hidden="true"
        className="win-loss-envelope-watermark"
        height="220"
        href={teamLogoUrl(team)}
        preserveAspectRatio="xMidYMid meet"
        width="220"
        x={margin.left + (chartWidth - 220) / 2}
        y={margin.top + (chartHeight - 220) / 2}
      />
      <polygon className="win-loss-envelope-outer-band" points={band("p1", "p99")} />
      <polygon className="win-loss-envelope-band" points={band("p5", "p95")} />
      <polyline className="win-loss-envelope-outer-boundary" points={line("p1")} />
      <polyline className="win-loss-envelope-outer-boundary" points={line("p99")} />
      <polyline className="win-loss-envelope-boundary" points={line("p5")} />
      <polyline className="win-loss-envelope-boundary" points={line("p95")} />
      <polyline className="win-loss-envelope-median" points={line("median")} />
      {envelope.points.slice(1).filter((point) => toughestGames.has(point.game) && point.opponentTeam).map((point) => (
        <g key={`tough-opponent-${point.game}`} className="win-loss-envelope-tough-opponent">
          <line x1={x(point.game)} x2={x(point.game)} y1={margin.top + 30} y2={y(point.median) - 5} />
          <image
            aria-hidden="true"
            height="24"
            href={teamLogoUrl(point.opponentTeam!)}
            preserveAspectRatio="xMidYMid meet"
            width="24"
            x={x(point.game) - 12}
            y={margin.top + 2}
          />
        </g>
      ))}
      {envelope.points.slice(1).filter((point) => toughestGames.has(point.game)).map((point) => (
        <circle key={`tough-${point.game}`} className="win-loss-envelope-tough-game" cx={x(point.game)} cy={y(point.median)} r="4">
          <title>One of the five lowest win-probability games</title>
        </circle>
      ))}
      {hovered && <>
        <line className="win-loss-envelope-hover-line" x1={x(hovered.game)} x2={x(hovered.game)} y1={margin.top} y2={height - margin.bottom} />
        <circle className="win-loss-envelope-hover-point outer" cx={x(hovered.game)} cy={y(hovered.p1)} r="2.5" />
        <circle className="win-loss-envelope-hover-point" cx={x(hovered.game)} cy={y(hovered.p5)} r="3" />
        <circle className="win-loss-envelope-hover-point" cx={x(hovered.game)} cy={y(hovered.median)} r="3.5" />
        <circle className="win-loss-envelope-hover-point" cx={x(hovered.game)} cy={y(hovered.p95)} r="3" />
        <circle className="win-loss-envelope-hover-point outer" cx={x(hovered.game)} cy={y(hovered.p99)} r="2.5" />
        <g className="win-loss-envelope-tooltip" transform={`translate(${tooltipX} ${tooltipY})`}>
          <rect width={tooltipWidth} height={tooltipHeight} />
          <text x="8" y="15">Game {hovered.game} · {hovered.opponent}{hovered.backToBack ? " · B2B" : ""}</text>
          <text x="8" y="34">P1: {record(hovered.p1, hovered.game)}</text>
          <text x="8" y="52">P5: {record(hovered.p5, hovered.game)}</text>
          <text className="win-loss-envelope-tooltip-median" x="8" y="70">Median: {record(hovered.median, hovered.game)}</text>
          <text x="8" y="88">P95: {record(hovered.p95, hovered.game)}</text>
          <text x="8" y="106">P99: {record(hovered.p99, hovered.game)}</text>
          <text x="8" y="124">Win probability: {(hovered.winProbability * 100).toFixed(1)}%</text>
        </g>
      </>}
      <rect className="win-loss-envelope-hit-area" x={margin.left} y={margin.top} width={chartWidth} height={chartHeight} onPointerMove={onPointerMove} onPointerLeave={() => setHoveredGame(null)} />
      <text className="win-loss-envelope-axis-label" x={margin.left} y={12}>Wins</text>
    </svg>
    <div className="win-loss-envelope-endpoints" aria-label="Final modeled-schedule win range">
      <span><small>P1</small>{final.p1}</span>
      <span><small>P5</small>{final.p5}</span>
      <span><small>Median</small>{final.median}</span>
      <span><small>P95</small>{final.p95}</span>
      <span><small>P99</small>{final.p99}</span>
    </div>
  </section>;
}

function readOverrides(): Overrides {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (!saved) return {};
    const parsed = JSON.parse(saved) as Record<string, unknown>;
    return Object.fromEntries(
      Object.entries(parsed).flatMap(([key, value]) => {
        if (!value || typeof value !== "object") return [];
        const candidate = value as Record<string, unknown>;
        const availabilityProbability = candidate.availabilityProbability;
        const conditionalMinutesPerGame = candidate.conditionalMinutesPerGame;
        const nailRating = candidate.nailRating;
        const output: PlayerInputOverride = {};
        if (typeof availabilityProbability === "number" && Number.isFinite(availabilityProbability)) {
          output.availabilityProbability = availabilityProbability;
        }
        if (typeof conditionalMinutesPerGame === "number" && Number.isFinite(conditionalMinutesPerGame)) {
          output.conditionalMinutesPerGame = conditionalMinutesPerGame;
        }
        if (typeof nailRating === "number" && Number.isFinite(nailRating)) {
          output.nailRating = nailRating;
        }
        return Object.keys(output).length ? [[key, output]] : [];
      }),
    ) as Overrides;
  } catch {
    return {};
  }
}

function directPlayerInputs(player: MinutesProjectionPlayer, overrides: Overrides): PlayerInputs {
  const override = overrides[overrideKey(player.team, player.player_id)];
  return {
    availabilityProbability: override?.availabilityProbability ?? player.availability_probability,
    conditionalMinutesPerGame: override?.conditionalMinutesPerGame ?? player.conditional_minutes_per_game,
    nailRating: override?.nailRating ?? player.projected_nail,
  };
}

function incumbentTeamStrength(
  players: MinutesProjectionPlayer[],
  overrides: Overrides,
  fallback: number,
) {
  const incumbents = players.filter((player) => player.is_incumbent_team_strength_incumbent);
  const weight = incumbents.reduce((total, player) => {
    const inputs = directPlayerInputs(player, overrides);
    return total + inputs.availabilityProbability * inputs.conditionalMinutesPerGame;
  }, 0);
  if (weight <= 0) return fallback;
  return incumbents.reduce((total, player) => {
    const inputs = directPlayerInputs(player, overrides);
    return total + inputs.availabilityProbability * inputs.conditionalMinutesPerGame * inputs.nailRating;
  }, 0) / weight;
}

function adjustedPlayerInputs(
  player: MinutesProjectionPlayer,
  overrides: Overrides,
  payload: MinutesProjectionPayload,
  teamStrength: number,
): PlayerInputs {
  const inputs = directPlayerInputs(player, overrides);
  const contract = payload.incumbent_team_strength;
  const hasConditionalOverride = overrides[overrideKey(player.team, player.player_id)]?.conditionalMinutesPerGame !== undefined;
  if (!contract || !player.uses_incumbent_team_strength || hasConditionalOverride) return inputs;
  const reference = contract.teams.find((item) => item.team === player.team)?.incumbent_team_nail
    ?? contract.league_strength_fallback;
  const adjustedLogMinutes = Math.log1p(inputs.conditionalMinutesPerGame)
    + contract.team_strength_log_minutes_coefficient * (teamStrength - reference);
  return {
    ...inputs,
    conditionalMinutesPerGame: Math.max(0, Math.expm1(Math.max(-20, Math.min(20, adjustedLogMinutes)))),
  };
}

function minuteAllocation(
  players: MinutesProjectionPlayer[],
  overrides: Overrides,
  regulationMinutes: number,
  rotationSize: number,
  payload: MinutesProjectionPayload,
): MinuteAllocation {
  const strength = incumbentTeamStrength(
    players,
    overrides,
    payload.incumbent_team_strength?.league_strength_fallback ?? 0,
  );
  const inputs = new Map(players.map((player) => [
    player.player_id,
    adjustedPlayerInputs(player, overrides, payload, strength),
  ]));
  const rawTotals = new Map(players.map((player) => {
    const input = inputs.get(player.player_id)!;
    const rawMinutes = SEASON_GAMES * input.availabilityProbability * input.conditionalMinutesPerGame;
    return [player.player_id, rawMinutes];
  }));
  const rotationPlayers = [...players].sort((left, right) => (
    (rawTotals.get(right.player_id) ?? 0) - (rawTotals.get(left.player_id) ?? 0)
    || left.player_name.localeCompare(right.player_name)
    || left.player_id - right.player_id
  )).slice(0, rotationSize);
  const rotationPlayerIds = new Set(rotationPlayers.map((player) => player.player_id));
  const totalRawMinutes = rotationPlayers.reduce(
    (total, player) => total + (rawTotals.get(player.player_id) ?? 0),
    0,
  );
  const minutes = totalRawMinutes <= 0
    ? new Map(players.map((player) => [player.player_id, 0]))
    : new Map(players.map((player) => [
    player.player_id,
    rotationPlayerIds.has(player.player_id)
      ? regulationMinutes * (rawTotals.get(player.player_id) ?? 0) / totalRawMinutes
      : 0,
  ]));
  return { minutes, inputs, rotationPlayerIds, totalRawMinutes };
}

function calculateProjection(
  payload: MinutesProjectionPayload,
  baseline: WinProjectionPayload,
  overrides: Overrides,
): WinProjectionPayload {
  const rawStrength = new Map<string, number>();
  const marketTotals = new Map(
    baseline.teams.map((team) => [team.team, team.betmgm_win_total]),
  );
  const baselineTeams = new Map(baseline.teams.map((team) => [team.team, team]));
  for (const team of payload.teams) {
    const players = payload.players.filter((player) => player.team === team);
    const allocation = minuteAllocation(
      players,
      overrides,
      payload.regulation_team_minutes,
      payload.initial_rotation_size,
      payload,
    );
    rawStrength.set(team, players.reduce((total, player) => (
      total + (allocation.minutes.get(player.player_id) ?? 0)
        * (allocation.inputs.get(player.player_id)?.nailRating ?? 0) / 48
    ), 0));
  }
  const meanStrength = Array.from(rawStrength.values()).reduce((total, value) => total + value, 0)
    / rawStrength.size;
  const strength = new Map(Array.from(rawStrength, ([team, value]) => [team, value - meanStrength]));
  const scheduledWins = new Map(payload.teams.map((team) => [team, 0]));

  for (const game of baseline.scheduled_games) {
    const homeWinProbability = scheduledHomeWinProbability(
      game,
      strength,
      baseline.win_probability_scale,
      baseline.home_court,
      baseline.back_to_back,
    );
    scheduledWins.set(
      game.home_team,
      (scheduledWins.get(game.home_team) ?? 0) + homeWinProbability,
    );
    scheduledWins.set(
      game.away_team,
      (scheduledWins.get(game.away_team) ?? 0) + 1 - homeWinProbability,
    );
  }
  const rawUnassignedWins = new Map(payload.teams.map((team) => {
    const teamStrength = strength.get(team) ?? 0;
    const neutralHome = sigmoid(baseline.win_probability_scale * (teamStrength + baseline.home_court));
    const neutralAway = sigmoid(baseline.win_probability_scale * (teamStrength - baseline.home_court));
    return [
      team,
      baseline.unassigned_regular_games_per_team * (neutralHome + neutralAway) / 2,
    ] as const;
  }));
  const rawUnassignedMean = Array.from(rawUnassignedWins.values()).reduce(
    (total, value) => total + value,
    0,
  ) / rawUnassignedWins.size;
  const neutralUnassignedWins = baseline.unassigned_regular_games_per_team / 2;
  const teams: WinProjectionTeam[] = payload.teams.map((team) => {
    const baselineTeam = baselineTeams.get(team);
    if (!baselineTeam) throw new Error(`Missing baseline interval for ${team}.`);
    const teamStrength = strength.get(team) ?? 0;
    const scheduled = scheduledWins.get(team) ?? 0;
    const unassigned = (rawUnassignedWins.get(team) ?? 0) - rawUnassignedMean + neutralUnassignedWins;
    const wins = scheduled + unassigned;
    return {
      team,
      team_strength: teamStrength,
      betmgm_win_total: marketTotals.get(team),
      win_total_p1: baselineTeam.win_total_p1,
      win_total_p5: baselineTeam.win_total_p5,
      win_total_p50: baselineTeam.win_total_p50,
      win_total_p95: baselineTeam.win_total_p95,
      win_total_p99: baselineTeam.win_total_p99,
      scheduled_wins: scheduled,
      scheduled_games: baseline.scheduled_games_per_team,
      unassigned_wins: unassigned,
      projected_wins: wins,
      projected_losses: 82 - wins,
    };
  });
  teams.sort((left, right) => right.projected_wins - left.projected_wins || left.team.localeCompare(right.team));
  return attachBrowserWinTotalIntervals({ ...baseline, teams });
}

export function WinProjectionsPage() {
  const [payload, setPayload] = useState<MinutesProjectionPayload | null>(null);
  const [team, setTeam] = useState("");
  const [overrides, setOverrides] = useState<Overrides>(readOverrides);
  const [calculatedForecast, setCalculatedForecast] = useState<WinProjectionPayload | null>(null);
  const [forecastNeedsUpdate, setForecastNeedsUpdate] = useState(false);
  const [forecastSortColumn, setForecastSortColumn] = useState<ForecastSortColumn>("projected_wins");
  const [forecastSortDirection, setForecastSortDirection] = useState<SortDirection>("descending");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        const response = await fetch("/api/win-projections", { signal: controller.signal });
        if (!response.ok) throw new Error("Preseason minute projections are unavailable.");
        const next = (await response.json()) as MinutesProjectionPayload;
        setPayload(next);
        setTeam((current) => current || next.teams[0] || "");
      } catch (requestError) {
        if ((requestError as Error).name !== "AbortError") setError((requestError as Error).message);
      }
    })();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
  }, [overrides]);

  const players = useMemo(() => (
    payload?.players.filter((player) => player.team === team) ?? []
  ), [payload, team]);
  const allocation = useMemo(() => (
    payload
      ? minuteAllocation(
        players,
        overrides,
        payload.regulation_team_minutes,
        payload.initial_rotation_size,
        payload,
      )
      : { minutes: new Map<number, number>(), inputs: new Map<number, PlayerInputs>(), rotationPlayerIds: new Set<number>(), totalRawMinutes: 0 }
  ), [overrides, payload, players]);
  const hasInvalidOverrides = payload !== null && allocation.totalRawMinutes <= 0;
  const minutes = allocation.minutes;
  const orderedPlayers = useMemo(() => (
    [...players].sort((left, right) => (
      (minutes.get(right.player_id) ?? 0) - (minutes.get(left.player_id) ?? 0)
      || left.player_name.localeCompare(right.player_name)
      || left.player_id - right.player_id
    ))
  ), [minutes, players]);
  const forecast = calculatedForecast ?? payload?.win_projection ?? null;
  const sortedForecastTeams = useMemo(() => {
    if (!forecast) return [];
    const direction = forecastSortDirection === "ascending" ? 1 : -1;
    return [...forecast.teams].sort((left, right) => {
      const leftValue = forecastSortColumn === "delta" ? marketDelta(left) : left[forecastSortColumn];
      const rightValue = forecastSortColumn === "delta" ? marketDelta(right) : right[forecastSortColumn];
      if (leftValue === null || leftValue === undefined) return 1;
      if (rightValue === null || rightValue === undefined) return -1;
      if (typeof leftValue === "number" && typeof rightValue === "number") {
        return direction * (leftValue - rightValue) || left.team.localeCompare(right.team);
      }
      return direction * String(leftValue).localeCompare(String(rightValue)) || left.team.localeCompare(right.team);
    });
  }, [forecast, forecastSortColumn, forecastSortDirection]);
  const activeOverrides = players.filter((player) => (
    overrides[overrideKey(player.team, player.player_id)] !== undefined
  )).length;
  const updateOverride = (
    player: MinutesProjectionPlayer,
    field: keyof PlayerInputOverride,
    rawValue: string,
  ) => {
    const key = overrideKey(player.team, player.player_id);
    if (!rawValue.trim()) {
      setForecastNeedsUpdate(true);
      setOverrides((current) => {
        const next = { ...current };
        const currentOverride = { ...(next[key] ?? {}) };
        delete currentOverride[field];
        if (Object.keys(currentOverride).length) next[key] = currentOverride;
        else delete next[key];
        return next;
      });
      return;
    }
    const value = Number(rawValue);
    if (!Number.isFinite(value)) return;
    setForecastNeedsUpdate(true);
    setOverrides((current) => ({
      ...current,
      [key]: {
        ...(current[key] ?? {}),
        [field]: field === "availabilityProbability"
          ? Math.max(0, Math.min(1, value / SEASON_GAMES))
          : field === "conditionalMinutesPerGame"
            ? Math.max(0, Math.min(48, value))
            : value,
      },
    }));
  };

  const selectForecastTeam = (nextTeam: string) => {
    setTeam(nextTeam);
  };

  const changeForecastSort = (column: ForecastSortColumn) => {
    if (column === forecastSortColumn) {
      setForecastSortDirection((direction) => direction === "ascending" ? "descending" : "ascending");
      return;
    }
    setForecastSortColumn(column);
    setForecastSortDirection(column === "team" ? "ascending" : "descending");
  };

  if (error) return <p className="error win-projections-error"><CircleAlert size={16} /> {error}</p>;
  if (!payload) return <div className="profile-loading"><LoaderCircle className="spin" size={20} /> Loading preseason minutes</div>;
  const baselineForecast = payload.win_projection;
  const calculateForecast = () => {
    if (!baselineForecast || hasInvalidOverrides) return;
    setCalculatedForecast(calculateProjection(payload, baselineForecast, overrides));
    setForecastNeedsUpdate(false);
  };
  const resetTeam = () => {
    const nextOverrides = Object.fromEntries(
      Object.entries(overrides).filter(([key]) => !key.startsWith(`${team}:`)),
    );
    setOverrides(nextOverrides);
    if (baselineForecast) {
      setCalculatedForecast(calculateProjection(payload, baselineForecast, nextOverrides));
    }
    setForecastNeedsUpdate(false);
  };

  return (
    <article className="win-projections-page" aria-labelledby="win-projections-title">
      <section className="win-projections-hero">
        <p className="eyebrow">{payload.season} preseason planning</p>
        <h1 id="win-projections-title">Win projections.</h1>
        <p>Adjust medical availability and conditional playing time, then recalculate. The top {payload.initial_rotation_size} override-adjusted raw projections form the rotation.</p>
        <a
          className="win-projections-doc-link"
          href={WIN_PROJECTIONS_DOCUMENTATION_URL}
          target="_blank"
          rel="noreferrer"
        >
          Read the methodology <ArrowUpRight size={15} aria-hidden="true" />
        </a>
      </section>

      <section className="win-projections-workspace" aria-label="Team minute projections">
        {forecast && <section className="win-forecast" aria-labelledby="win-forecast-title">
          <div className="win-forecast-heading">
            <div><p className="eyebrow">{calculatedForecast && !forecastNeedsUpdate ? "Custom forecast" : "Baseline forecast"}</p><h2 id="win-forecast-title">Projected wins.</h2></div>
            <p>{forecast.scheduled_games_per_team} scheduled games + {forecast.unassigned_regular_games_per_team} modeled NBA Cup games.</p>
          </div>
          <div className="win-forecast-table-wrap"><table className="win-forecast-table">
            <thead><tr>
              {([
                ["team", "Team"],
                ["team_strength", "Strength"],
                ["betmgm_win_total", "BetMGM"],
                ["projected_wins", "Projected Wins"],
                ["delta", "Delta"],
              ] as Array<[ForecastSortColumn, string]>).map(([column, label]) => <th scope="col" key={column} aria-sort={forecastSortColumn === column ? forecastSortDirection : "none"}>
                <button className="win-forecast-sort" type="button" onClick={() => changeForecastSort(column)}>
                  {label}
                  {forecastSortColumn === column && (forecastSortDirection === "ascending" ? <ChevronUp size={12} /> : <ChevronDown size={12} />)}
                </button>
              </th>)}<th scope="col">P5-P95</th>
            </tr></thead>
            <tbody>{sortedForecastTeams.map((item) => {
              const delta = marketDelta(item);
              const direction = marketDirection(item.projected_wins, item.betmgm_win_total);
              const directionClass = delta !== null && delta < 0 ? "negative" : "positive";
              const intervalPosition = marketIntervalPosition(item);
              const rowClassName = [
                item.team === team ? "selected-team" : "",
                intervalPosition === "below" ? "market-below-range" : "",
                intervalPosition === "above" ? "market-above-range" : "",
              ].filter(Boolean).join(" ");
              return <tr key={item.team} className={rowClassName}>
                <th><button className="win-forecast-team-link" type="button" aria-controls="team-minutes" aria-pressed={item.team === team} onClick={() => selectForecastTeam(item.team)}>{item.team}</button></th>
                <td className={item.team_strength < 0 ? "negative" : "positive"}>{formatRating(item.team_strength)}</td>
                <td title={marketIntervalDescription(item)}>{item.betmgm_win_total?.toFixed(1) ?? "-"}</td>
                <td className="win-forecast-wins">{item.projected_wins.toFixed(1)}</td>
                <td className={directionClass}>{delta === null ? "-" : `${direction} ${formatRating(delta)}`}</td>
                <td className="win-forecast-interval">{item.win_total_p5}&ndash;{item.win_total_p95}</td>
              </tr>;
            })}</tbody>
          </table></div>
          <p className="win-projections-note">{forecast.contract} {forecast.market_win_totals && <><a href={forecast.market_win_totals.source_url} target="_blank" rel="noreferrer">{forecast.market_win_totals.provider} win totals</a> captured {forecast.market_win_totals.as_of}; O/U compares projected wins to that line. </>}Calibration: {forecast.calibration.calibration_season} only; {forecast.calibration.holdout_season} Brier {forecast.calibration.holdout_brier.toFixed(3)}, accuracy {(forecast.calibration.holdout_accuracy * 100).toFixed(1)}%.</p>
        </section>}
        <section className="win-team-panel" id="team-minutes" aria-label={`${team} minute projection`}>
        {forecast && <WinLossEnvelopeChart forecast={forecast} team={team} />}
        <div className="win-projections-toolbar">
          <label className="win-projections-team-filter">
            <span>Team</span>
            <select value={team} onChange={(event) => setTeam(event.target.value)}>
              {payload.teams.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <button className="win-projections-reset" type="button" onClick={resetTeam} disabled={activeOverrides === 0} title="Reset this team to its baseline availability, conditional minutes, and rating inputs">
            <RotateCcw size={15} aria-hidden="true" />
            <span>Reset team</span>
          </button>
          <button className="win-projections-calculate" type="button" onClick={calculateForecast} disabled={hasInvalidOverrides}>
            <Calculator size={15} aria-hidden="true" />
            <span>Calculate projection</span>
          </button>
        </div>
        {hasInvalidOverrides && <p className="error win-projections-error"><CircleAlert size={16} /> At least one player must have positive availability and conditional minutes.</p>}
        <div className="win-projections-table-wrap">
          <table className="win-projections-table">
            <thead><tr><th>Player</th><th>+/-</th><th><i>G</i><sub>available</sub></th><th>E[MPG | available]</th><th>Squashed MPG</th></tr></thead>
            <tbody>
              {orderedPlayers.map((player) => {
                const key = overrideKey(player.team, player.player_id);
                const override = overrides[key];
                const inputs = allocation.inputs.get(player.player_id) ?? directPlayerInputs(player, overrides);
                const projected = minutes.get(player.player_id) ?? player.baseline_minutes_per_game;
                return <tr key={player.player_id} className={override === undefined ? "" : "has-minute-override"}>
                  <th scope="row"><a href={`#player/${player.player_id}`}>{player.player_name}</a><small>{player.position} · Age {player.age?.toFixed(0) ?? "-"}{allocation.rotationPlayerIds.has(player.player_id) ? "" : " · Outside rotation"}{player.is_rating_fallback ? " · NAIL fallback" : ""}</small></th>
                  <td className={inputs.nailRating < 0 ? "negative" : "positive"}><input className={inputs.nailRating < 0 ? "negative" : "positive"} aria-label={`Plus minus rating for ${player.player_name}`} type="number" step="0.1" value={inputs.nailRating.toFixed(1)} onChange={(event) => updateOverride(player, "nailRating", event.target.value)} /></td>
                  <td><input aria-label={`Projected available games for ${player.player_name}`} type="number" min="0" max={SEASON_GAMES} step="1" value={Math.round((override?.availabilityProbability ?? player.availability_probability) * SEASON_GAMES)} onChange={(event) => updateOverride(player, "availabilityProbability", event.target.value)} /></td>
                  <td><input className="conditional-minutes-input" aria-label={`Conditional minutes for ${player.player_name}`} type="number" min="0" max="48" step="0.5" value={inputs.conditionalMinutesPerGame.toFixed(1)} onChange={(event) => updateOverride(player, "conditionalMinutesPerGame", event.target.value)} /></td>
                  <td className="projected-minutes">{projected.toFixed(1)}</td>
                </tr>;
              })}
            </tbody>
          </table>
        </div>
        <ul className="win-projections-mobile-list">
          {orderedPlayers.map((player) => {
            const key = overrideKey(player.team, player.player_id);
            const override = overrides[key];
            const inputs = allocation.inputs.get(player.player_id) ?? directPlayerInputs(player, overrides);
            const projected = minutes.get(player.player_id) ?? player.baseline_minutes_per_game;
            return <li key={player.player_id} className={override === undefined ? "" : "has-minute-override"}>
              <div className="win-projections-mobile-heading">
                <a href={`#player/${player.player_id}`}>{player.player_name}</a>
                <span className={inputs.nailRating < 0 ? "negative" : "positive"}>{formatRating(inputs.nailRating)}</span>
              </div>
              <p>{player.position} · Age {player.age?.toFixed(0) ?? "-"}{allocation.rotationPlayerIds.has(player.player_id) ? "" : " · Outside rotation"}{player.is_rating_fallback ? " · NAIL fallback" : ""}</p>
              <dl>
                <div><dt>+/-</dt><dd><input className={inputs.nailRating < 0 ? "negative" : "positive"} aria-label={`Plus minus rating for ${player.player_name}`} type="number" step="0.1" value={inputs.nailRating.toFixed(1)} onChange={(event) => updateOverride(player, "nailRating", event.target.value)} /></dd></div>
                <div><dt><i>G</i><sub>available</sub></dt><dd><input aria-label={`Projected available games for ${player.player_name}`} type="number" min="0" max={SEASON_GAMES} step="1" value={Math.round((override?.availabilityProbability ?? player.availability_probability) * SEASON_GAMES)} onChange={(event) => updateOverride(player, "availabilityProbability", event.target.value)} /></dd></div>
                <div><dt>E[MPG | available]</dt><dd><input className="conditional-minutes-input" aria-label={`Conditional minutes for ${player.player_name}`} type="number" min="0" max="48" step="0.5" value={inputs.conditionalMinutesPerGame.toFixed(1)} onChange={(event) => updateOverride(player, "conditionalMinutesPerGame", event.target.value)} /></dd></div>
                <div><dt>Squashed MPG</dt><dd className="projected-minutes">{projected.toFixed(1)}</dd></div>
              </dl>
            </li>;
          })}
        </ul>
        <p className="win-projections-note">Baseline: {payload.model}. {payload.contract}</p>
        </section>
      </section>
    </article>
  );
}
