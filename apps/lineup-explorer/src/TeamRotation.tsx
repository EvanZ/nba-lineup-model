import { useEffect, useRef, useState } from "react";
import { Download } from "lucide-react";
import { releaseFetch as fetch } from "./releaseFetch";
import { inlineSvgImages } from "./PlayerHistoricalCharts";
import type { TeamRotationPayload } from "./types";

const wholeNumber = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const playerProfileHref = (id: number) => `#player/${id}`;
const playerHeadshotUrl = (id: number) => `/api/headshots/${id}.png`;

type RotationOrdering = "aoe" | "fpc" | "hclust";
type RotationCellArea = "full" | "shared";

function rotationPlayerLabel(playerName: string) {
  const parts = playerName.trim().split(/\s+/).filter(Boolean);
  if (parts.length < 2) return playerName;
  const suffix = parts.at(-1) === "Jr." || parts.at(-1) === "II" || parts.at(-1) === "III"
    ? ` ${parts.slice(-2).join(" ")}`
    : ` ${parts.at(-1)}`;
  return `${parts[0].slice(0, 1)}.${suffix}`;
}

function blendedColor(
  start: [number, number, number],
  end: [number, number, number],
  weight: number,
) {
  const bounded = Math.max(0, Math.min(1, weight));
  return `rgb(${start.map((value, index) => Math.round(value + (end[index] - value) * bounded)).join(", ")})`;
}

function rotationHeatColor(value: number, midpoint: number, isDiagonal: boolean) {
  if (isDiagonal) return "#17201c";
  const boundedValue = Math.max(0, Math.min(1, value));
  if (midpoint <= 0) return blendedColor([255, 254, 250], [230, 159, 0], boundedValue);
  if (midpoint >= 1) return blendedColor([0, 114, 178], [255, 254, 250], boundedValue);
  if (boundedValue <= midpoint) {
    return blendedColor([0, 114, 178], [255, 254, 250], boundedValue / midpoint);
  }
  return blendedColor([255, 254, 250], [230, 159, 0], (boundedValue - midpoint) / (1 - midpoint));
}

function rotationCellLabelColor(value: number, midpoint: number) {
  const boundedValue = Math.max(0, Math.min(1, value));
  if (midpoint <= 0) return "#17201c";
  const blueIntensity = boundedValue <= midpoint
    ? 1 - boundedValue / midpoint
    : 0;
  return blueIntensity > 0.55 ? "#fffefa" : "#17201c";
}

export function TeamRotationMap({
  team,
  teamName,
  season,
  isCompact,
}: {
  team: string;
  teamName: string;
  season: string;
  isCompact: boolean;
}) {
  // Preserve the complete observed team-season rotation on every viewport.
  // The SVG can scale vertically on narrow screens; omitting low-exposure
  // players changes the clustering problem itself.
  const maxPlayers = 30;
  const [payload, setPayload] = useState<TeamRotationPayload | null>(null);
  const [ordering, setOrdering] = useState<RotationOrdering>("hclust");
  const [colorMidpoint, setColorMidpoint] = useState(0.5);
  const [cellArea, setCellArea] = useState<RotationCellArea>("full");
  const [error, setError] = useState<string | null>(null);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; column: number } | null>(null);
  const chartRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    setPayload(null);
    setHoveredCell(null);
    setError(null);
    let timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const parameters = new URLSearchParams({
          season,
          max_players: String(maxPlayers),
          ordering,
          order_metric: "floor",
        });
        const response = await fetch(
          `/api/teams/${encodeURIComponent(team)}/rotation?${parameters.toString()}`,
          { signal: controller.signal },
        );
        if (!response.ok) throw new Error(response.status === 404 ? "No saved rotation chart is published for this team-season." : "Saved rotation data is temporarily unavailable.");
        const data = await response.json() as TeamRotationPayload;
        if (data.kind !== "saved_team_rotation" || data.team !== team || data.season !== season || !Array.isArray(data.players)) throw new Error("Saved rotation data is unavailable.");
        if (controller.signal.aborted) return;
        setPayload(data); setError(null);
        timer = setTimeout(load, 600000 - Date.now() % 600000 + 1000);
      } catch (rotationError) {
        if (!controller.signal.aborted) { setPayload(null); setError((rotationError as Error).message); timer = setTimeout(load, 60000); }
      }
    }
    void load();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [maxPlayers, ordering, season, team]);

  const matrix = payload?.floor_correlation ?? [];
  const players = payload?.players ?? [];
  const cell = isCompact ? 25 : 34;
  const treeWidth = isCompact ? 38 : 70;
  const avatarSize = isCompact ? 13 : 18;
  const labelWidth = isCompact ? 76 : 132;
  const left = treeWidth + 12;
  const top = isCompact ? 102 : 152;
  const gridWidth = players.length * cell;
  const rowLabelX = left + gridWidth + 8;
  const topTreeHeight = isCompact ? 31 : 58;
  const topTreeBottom = top - (isCompact ? 49 : 74);
  const width = left + gridWidth + labelWidth + 10;
  const height = top + players.length * cell + 10;
  const tooltip = hoveredCell && players[hoveredCell.row] && players[hoveredCell.column]
    ? {
      left: players[hoveredCell.row],
      right: players[hoveredCell.column],
      correlation: payload?.floor_correlation[hoveredCell.row]?.[hoveredCell.column] ?? 0,
      sharedPossessions: payload?.shared_possessions[hoveredCell.row]?.[hoveredCell.column] ?? 0,
      x: left + (hoveredCell.column + 1) * cell + 7,
      y: top + hoveredCell.row * cell - 5,
    }
    : null;
  const tooltipWidth = isCompact ? 182 : 218;
  const tooltipHeight = 51;
  const tooltipX = tooltip ? Math.max(4, Math.min(width - tooltipWidth - 4, tooltip.x)) : 0;
  const tooltipY = tooltip ? Math.max(4, Math.min(height - tooltipHeight - 4, tooltip.y)) : 0;
  const selectedLabel = "Floor-time correlation";
  const orderingLabel = ordering === "aoe" ? "AOE" : ordering === "fpc" ? "FPC" : "complete linkage";

  async function downloadPng() {
    const svg = chartRef.current;
    if (!svg) return;
    const copy = svg.cloneNode(true) as SVGSVGElement;
    const exportTitleHeight = 48;
    const exportHeight = height + exportTitleHeight;
    copy.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    copy.setAttribute("viewBox", `0 -${exportTitleHeight} ${width} ${exportHeight}`);
    copy.setAttribute("width", String(width));
    copy.setAttribute("height", String(exportHeight));
    copy.querySelectorAll("[data-export-exclude]").forEach((element) => element.remove());
    await inlineSvgImages(copy);

    const exportOrderingLabel = ordering === "hclust"
      ? "Complete-linkage clustering"
      : ordering === "aoe"
        ? "AOE ordering"
        : "FPC ordering";
    const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
    title.setAttribute("class", "team-rotation-export-title");
    title.setAttribute("x", "0");
    title.setAttribute("y", "-27");
    title.textContent = `${teamName} · ${season}`;
    copy.append(title);
    const subtitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
    subtitle.setAttribute("x", "0"); subtitle.setAttribute("y", "-9");
    subtitle.setAttribute("fill", "#58635b"); subtitle.setAttribute("font-family", "sans-serif"); subtitle.setAttribute("font-size", "11");
    subtitle.textContent = `${exportOrderingLabel} · Team-only · Through ${payload?.game_date_pacific}`;
    copy.append(subtitle);

    const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
    style.textContent = `
      .team-rotation-export-title { fill: #17201c; font-family: Georgia, serif; font-size: 16px; font-weight: 700; }
      .team-rotation-column-label, .team-rotation-row-label { fill: #58635b; font-family: monospace; font-size: 9px; font-weight: 700; }
      .team-rotation-dendrogram { fill: none; stroke: #69716b; stroke-linecap: square; stroke-linejoin: miter; stroke-width: 1; }
      .team-rotation-cell-fill { stroke: #fffefa; stroke-width: 1; }
      .team-rotation-cell-value { font-family: monospace; font-size: 8px; font-weight: 800; }
      .team-rotation-row-headshot-outline { fill: none; stroke: #245a47; stroke-width: 1.2; }
    `;
    copy.prepend(style);

    const image = new Image();
    const svgUrl = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(copy)], { type: "image/svg+xml" }));
    image.onload = () => {
      const canvas = document.createElement("canvas");
      const scale = 2;
      canvas.width = width * scale;
      canvas.height = exportHeight * scale;
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
        link.download = `${team.toLowerCase()}-${season}-rotation-correlation.png`;
        link.click();
        window.setTimeout(() => URL.revokeObjectURL(link.href), 0);
      }, "image/png");
    };
    image.onerror = () => URL.revokeObjectURL(svgUrl);
    image.src = svgUrl;
  }

  return <section className="team-rotation-section" aria-labelledby="team-rotation-title">
    <div className="team-section-heading team-rotation-heading">
      <div>
        <p className="section-kicker">Rotation patterns</p>
        <h2 id="team-rotation-title">Who played together.</h2>
      </div>
      <div className="team-rotation-actions">
        <div className="team-rotation-legend" aria-label={`${selectedLabel} color scale`}>
          <span>Less together</span>
          <i
            aria-hidden="true"
            style={{ background: `linear-gradient(90deg, #0072b2 0%, #fffefa ${colorMidpoint * 100}%, #e69f00 100%)` }}
          />
          <span>More together</span>
        </div>
        <label className="team-rotation-scale">
          <span>Color midpoint <b>{colorMidpoint.toFixed(2)}</b></span>
          <input
            aria-label="Correlation color midpoint"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={colorMidpoint}
            onChange={(event) => setColorMidpoint(Number(event.target.value))}
          />
        </label>
        <div className="team-rotation-fill-control">
          <span>Cell area</span>
          <div className="team-rotation-toggle" aria-label="Rotation map cell area">
            {(["full", "shared"] as RotationCellArea[]).map((option) => <button
              type="button"
              className={cellArea === option ? "active" : ""}
              aria-pressed={cellArea === option}
              key={option}
              onClick={() => setCellArea(option)}
            >{option === "full" ? "Full" : "Shared"}</button>)}
          </div>
        </div>
        <div className="team-rotation-toggle" aria-label="Rotation map ordering">
          {(["hclust", "aoe", "fpc"] as RotationOrdering[]).map((option) => <button
            type="button"
            className={ordering === option ? "active" : ""}
            aria-pressed={ordering === option}
            key={option}
            onClick={() => setOrdering(option)}
          >{option === "aoe" ? "AOE" : option === "fpc" ? "FPC" : "Cluster"}</button>)}
        </div>
        <button
          className="team-rotation-download"
          type="button"
          onClick={downloadPng}
          disabled={!payload}
          title="Download rotation map as PNG"
          aria-label={`Download ${team} ${season} rotation map as PNG`}
        ><Download size={15} /></button>
      </div>
    </div>
    {payload && <p className="team-rotation-note">
      {payload.coverage.scope === "saved_historical_regular_season_dataset" ? "Saved historical rotation. " : ""}{selectedLabel} across this team’s positive-possession stints through {payload.game_date_pacific}. {players.length} players across {payload.game_count} {payload.coverage.scope === "saved_historical_regular_season_dataset" ? "saved games" : "games"}. Ordered by corrplot-style {orderingLabel}{ordering === "hclust" ? "; the paired dendrograms show that hierarchy." : "."}
    </p>}
    {error && <p className="team-empty-state">{error}</p>}
    {!payload && !error && <p className="team-empty-state">Loading observed rotation structure.</p>}
    {payload && <div className="team-rotation-content">
      <div className="team-rotation-chart-wrap">
        <svg
          ref={chartRef}
          className="team-rotation-chart"
          data-ordering={payload.ordering}
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={`${team} ${season} ${selectedLabel.toLowerCase()} heatmap`}
        >
        {ordering === "hclust" && payload.dendrogram.branches.map((branch, index) => <polyline
          className="team-rotation-dendrogram"
          key={`top-branch-${index}`}
          points={branch.x.map((coordinate, branchIndex) => {
            const x = left + cell / 2 + coordinate * Math.max(0, gridWidth - cell);
            const y = topTreeBottom - branch.height[branchIndex] * topTreeHeight;
            return `${x},${y}`;
          }).join(" ")}
        />)}
        {ordering === "hclust" && payload.dendrogram.branches.map((branch, index) => <polyline
          className="team-rotation-dendrogram"
          key={`left-branch-${index}`}
          points={branch.x.map((coordinate, branchIndex) => {
            const x = left - 10 - branch.height[branchIndex] * treeWidth;
            const y = top + cell / 2 + coordinate * Math.max(0, gridWidth - cell);
            return `${x},${y}`;
          }).join(" ")}
        />)}
        {players.map((player, index) => {
          const position = left + index * cell + cell / 2;
          const rowY = top + index * cell + cell / 2;
          const clipId = `rotation-headshot-${team}-${season}-${player.player_id}`;
          return <g key={`column-${player.player_id}`}>
            <a
              className="team-rotation-player-link"
              href={playerProfileHref(player.player_id)}
              aria-label={`Open ${player.player_name}'s player bio`}
            >
              <title>Open {player.player_name}'s player bio</title>
              <text
                x={position}
                y={top - 9}
                textAnchor="start"
                className="team-rotation-column-label"
                transform={`rotate(-55 ${position} ${top - 9})`}
              >{rotationPlayerLabel(player.player_name)}</text>
            </a>
            <defs>
              <clipPath id={clipId}>
                <circle cx={rowLabelX + avatarSize / 2} cy={rowY} r={avatarSize / 2} />
              </clipPath>
            </defs>
            <a
              className="team-rotation-player-link"
              href={playerProfileHref(player.player_id)}
              aria-label={`Open ${player.player_name}'s player bio`}
            >
              <title>Open {player.player_name}'s player bio</title>
              <image
                className="team-rotation-row-headshot"
                href={playerHeadshotUrl(player.player_id)}
                x={rowLabelX}
                y={rowY - avatarSize / 2}
                width={avatarSize}
                height={avatarSize}
                preserveAspectRatio="xMidYMid slice"
                clipPath={`url(#${clipId})`}
              />
              <circle
                className="team-rotation-row-headshot-outline"
                cx={rowLabelX + avatarSize / 2}
                cy={rowY}
                r={avatarSize / 2}
              />
              <text
                x={rowLabelX + avatarSize + 5}
                y={rowY + 3}
                textAnchor="start"
                className="team-rotation-row-label"
              >{rotationPlayerLabel(player.player_name)}</text>
            </a>
          </g>;
        })}
        {players.map((rowPlayer, row) => players.map((columnPlayer, column) => {
          const value = matrix[row]?.[column] ?? 0;
          const correlation = payload.floor_correlation[row]?.[column] ?? 0;
          const sharedPossessions = payload.shared_possessions[row]?.[column] ?? 0;
          const combinedExposure = players[row].on_court_possessions + players[column].on_court_possessions;
          const sharedExposure = combinedExposure > 0
            ? Math.max(0, Math.min(1, 2 * sharedPossessions / combinedExposure))
            : 0;
          const fillScale = cellArea === "shared" && row < column ? Math.sqrt(sharedExposure) : 1;
          const fillSize = (cell - 1) * fillScale;
          const fillOffset = ((cell - 1) - fillSize) / 2;
          const description = `${rowPlayer.player_name} and ${columnPlayer.player_name}: ${wholeNumber.format(sharedPossessions)} shared possessions; ${correlation >= 0 ? "+" : ""}${correlation.toFixed(2)} floor-time correlation.`;
          const showTooltip = () => setHoveredCell({ row, column });
          return <g
            key={`${rowPlayer.player_id}-${columnPlayer.player_id}`}
            className="team-rotation-cell"
            data-row-player={rowPlayer.player_id}
            data-column-player={columnPlayer.player_id}
            data-correlation={correlation}
            data-shared-possessions={sharedPossessions}
            tabIndex={0}
            role="img"
            aria-label={description}
            onMouseEnter={showTooltip}
            onMouseLeave={() => setHoveredCell(null)}
            onFocus={showTooltip}
            onBlur={() => setHoveredCell(null)}
          >
            <title>{description}</title>
            <rect
              className="team-rotation-cell-background"
              x={left + column * cell}
              y={top + row * cell}
              width={cell - 1}
              height={cell - 1}
              fill="#fffefa"
            />
            <rect
              className="team-rotation-cell-fill"
              x={left + column * cell + fillOffset}
              y={top + row * cell + fillOffset}
              width={fillSize}
              height={fillSize}
              fill={rotationHeatColor(value, colorMidpoint, row === column)}
            />
            {!isCompact && row > column && <text
              className="team-rotation-cell-value"
              x={left + column * cell + cell / 2}
              y={top + row * cell + cell / 2 + 2.5}
              textAnchor="middle"
              fill={cellArea === "shared" && fillScale < 0.8 ? "#17201c" : rotationCellLabelColor(correlation, colorMidpoint)}
              pointerEvents="none"
              aria-hidden="true"
            >{correlation.toFixed(2)}</text>}
          </g>;
        }))}
        {tooltip && <g
          className="team-rotation-tooltip"
          data-export-exclude="true"
          pointerEvents="none"
          transform={`translate(${tooltipX}, ${tooltipY})`}
        >
          <rect width={tooltipWidth} height={tooltipHeight} rx="3" />
          <text x="7" y="12">{rotationPlayerLabel(tooltip.left.player_name)} + {rotationPlayerLabel(tooltip.right.player_name)}</text>
          <text x="7" y="25">Shared floor {wholeNumber.format(tooltip.sharedPossessions)} possessions</text>
          <text x="7" y="38">Correlation {tooltip.correlation >= 0 ? "+" : ""}{tooltip.correlation.toFixed(2)}</text>
        </g>}
        </svg>
      </div>
      <aside className="team-rotation-methods" aria-label="Rotation map methods">
        <p className="section-kicker">How it is calculated</p>
        <p className="team-rotation-method-copy">For each positive-possession regular-season stint <i>s</i> for this team, a player’s floor-time signal is:</p>
        <p className="team-rotation-formula"><i>x</i><sub>s,i</sub> = <i>p</i><sub>s</sub> · 1(player <i>i</i> is on court)</p>
        <p className="team-rotation-method-copy"><i>p</i><sub>s</sub> is the stint’s possession count; inactive players receive zero.</p>
        <p className="team-rotation-formula"><i>r</i><sub>i,j</sub> = cor<sub>s</sub>(<i>x</i><sub>s,i</sub>, <i>x</i><sub>s,j</sub>)</p>
        <p className="team-rotation-method-copy">Each cell is the Pearson correlation of those team-season vectors. Games played for other clubs are excluded.</p>
        <p className="team-rotation-method-copy">With Shared cell area, the upper-triangle square’s area is <i>a</i><sub>i,j</sub> = 2<i>S</i><sub>i,j</sub> / (<i>P</i><sub>i</sub> + <i>P</i><sub>j</sub>), where <i>S</i> is shared possessions and <i>P</i> is a player’s on-court possessions.</p>

        <dl className="team-rotation-ordering-definitions">
          <div>
            <dt>AOE</dt>
            <dd>Sorts the angle <i>atan2</i>(<i>e</i><sub>2,i</sub>, <i>e</i><sub>1,i</sub>) from the first two eigenvectors of <i>R</i>.</dd>
          </div>
          <div>
            <dt>FPC</dt>
            <dd>Sorts each player’s score <i>e</i><sub>1,i</sub> on the first principal component of <i>R</i>.</dd>
          </div>
          <div>
            <dt>Cluster</dt>
            <dd>Uses complete-linkage hierarchical clustering with distance <i>d</i><sub>i,j</sub> = 1 − <i>r</i><sub>i,j</sub>.</dd>
          </div>
        </dl>
      </aside>
    </div>}
  </section>;
}
