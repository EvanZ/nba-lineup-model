import { useCallback, useEffect, useMemo, useState } from "react";
import { geoAlbersUsa } from "d3-geo";
import { GIFEncoder, applyPalette, quantize } from "gifenc";
import usaMap from "@svg-maps/usa";
import {
  BaseEdge,
  Controls,
  EdgeLabelRenderer,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  applyNodeChanges,
  type Edge,
  type EdgeMouseHandler,
  type EdgeProps,
  type Node,
  type NodeChange,
  type NodeMouseHandler,
  type NodeProps,
  type ReactFlowInstance,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { CircleAlert, Copy, Download, Filter, Hand, LoaderCircle, MousePointer2, X } from "lucide-react";

import type { ExternalRosterArrival, Player, RosterMove, RosterMovesPayload } from "./types";

type MoveFilter = "all" | "external" | "rookies" | RosterMove["move_type"];
type TeamNodeData = { team: string; conference: "East" | "West" };
type MapNodeData = Record<string, never>;
type OutsideNodeData = { label: string; isCollege: boolean };
type GraphNode = Node<TeamNodeData, "team"> | Node<MapNodeData, "usaMap"> | Node<OutsideNodeData, "outside">;
type GraphMove = RosterMove & {
  isExternalArrival: boolean;
  externalOriginLabel?: string;
  externalOriginKind?: "college" | "country" | "fallback";
  isRookie?: boolean;
};
type MoveEdgeData = GraphMove & { showLabel: boolean; animateTeamSelection: boolean };
type CanvasMode = "move" | "pan";
type MoveRoute = { filter: MoveFilter; team: string; autoplay: boolean };
type UsaMapLocation = { id: string; name: string; path: string };
type GraphPositions = Record<string, { x: number; y: number }>;
type CurveGeometry = {
  path: string;
  pointAt: (t: number) => { x: number; y: number };
  angleAt: (t: number) => number;
};

const MAP_WIDTH = 1320;
const MAP_HEIGHT = 730;
const TEAM_NODE_WIDTH = 82;
const TEAM_NODE_HEIGHT = 34;
const GIF_WIDTH = 1200;
const GIF_HEIGHT = 663;
const GIF_FRAME_COUNT = 48;
const GIF_FRAME_DELAY = 100;
const TRAVELER_HEADSHOT_MIN_SIZE = 22;
const TRAVELER_HEADSHOT_MAX_SCALE = 3;
const TRAVELER_HEADSHOT_RATING_AT_MAX_SCALE = 6;
const CONTINENTAL_PROJECTION_BOUNDS = { left: 80, top: 30, width: 820, height: 450 };
const continentalProjection = geoAlbersUsa();
const TEAM_GEOGRAPHY: Record<string, Omit<TeamNodeData, "team"> & { latitude: number; longitude: number; offsetX?: number; offsetY?: number }> = {
  ATL: { conference: "East", latitude: 33.75, longitude: -84.39 },
  BOS: { conference: "East", latitude: 42.36, longitude: -71.06 },
  BKN: { conference: "East", latitude: 40.68, longitude: -73.94, offsetX: -10, offsetY: 14 },
  CHA: { conference: "East", latitude: 35.23, longitude: -80.84 },
  CHI: { conference: "East", latitude: 41.88, longitude: -87.63 },
  CLE: { conference: "East", latitude: 41.5, longitude: -81.69 },
  DET: { conference: "East", latitude: 42.33, longitude: -83.05 },
  IND: { conference: "East", latitude: 39.77, longitude: -86.16 },
  MIA: { conference: "East", latitude: 25.76, longitude: -80.19 },
  MIL: { conference: "East", latitude: 43.04, longitude: -87.91 },
  NYK: { conference: "East", latitude: 40.71, longitude: -74.01, offsetX: 10, offsetY: -14 },
  ORL: { conference: "East", latitude: 28.54, longitude: -81.38 },
  PHI: { conference: "East", latitude: 39.95, longitude: -75.17 },
  TOR: { conference: "East", latitude: 43.65, longitude: -79.38 },
  WAS: { conference: "East", latitude: 38.91, longitude: -77.04 },
  DAL: { conference: "West", latitude: 32.78, longitude: -96.8 },
  DEN: { conference: "West", latitude: 39.74, longitude: -104.99 },
  GSW: { conference: "West", latitude: 37.77, longitude: -122.42 },
  HOU: { conference: "West", latitude: 29.76, longitude: -95.37 },
  LAC: { conference: "West", latitude: 34.05, longitude: -118.24, offsetX: -10, offsetY: -14 },
  LAL: { conference: "West", latitude: 34.05, longitude: -118.24, offsetX: 10, offsetY: 14 },
  MEM: { conference: "West", latitude: 35.15, longitude: -90.05 },
  MIN: { conference: "West", latitude: 44.98, longitude: -93.27 },
  NOP: { conference: "West", latitude: 29.95, longitude: -90.07 },
  OKC: { conference: "West", latitude: 35.47, longitude: -97.52 },
  PHX: { conference: "West", latitude: 33.45, longitude: -112.07 },
  POR: { conference: "West", latitude: 45.52, longitude: -122.68 },
  SAC: { conference: "West", latitude: 38.58, longitude: -121.49 },
  SAS: { conference: "West", latitude: 29.42, longitude: -98.49 },
  UTA: { conference: "West", latitude: 40.76, longitude: -111.89 },
};
const COLLEGE_GEOGRAPHY: Record<string, { latitude: number; longitude: number }> = {
  Alabama: { latitude: 33.209, longitude: -87.569 },
  "Alabama-Birmingham": { latitude: 33.502, longitude: -86.809 },
  Arizona: { latitude: 32.231, longitude: -110.951 },
  Arkansas: { latitude: 36.068, longitude: -94.176 },
  Baylor: { latitude: 31.548, longitude: -97.113 },
  "Brigham Young": { latitude: 40.251, longitude: -111.65 },
  Butler: { latitude: 39.84, longitude: -86.167 },
  Cincinnati: { latitude: 39.132, longitude: -84.516 },
  Connecticut: { latitude: 41.808, longitude: -72.249 },
  Duke: { latitude: 36.001, longitude: -78.938 },
  Houston: { latitude: 29.72, longitude: -95.342 },
  Illinois: { latitude: 40.102, longitude: -88.228 },
  Iowa: { latitude: 41.662, longitude: -91.535 },
  "Iowa State": { latitude: 42.026, longitude: -93.646 },
  Kansas: { latitude: 38.956, longitude: -95.255 },
  Kentucky: { latitude: 38.031, longitude: -84.504 },
  Louisville: { latitude: 38.215, longitude: -85.758 },
  Miami: { latitude: 25.721, longitude: -80.279 },
  Michigan: { latitude: 42.278, longitude: -83.738 },
  "North Carolina": { latitude: 35.904, longitude: -79.046 },
  "North Carolina State": { latitude: 35.785, longitude: -78.663 },
  Northwestern: { latitude: 42.056, longitude: -87.675 },
  "Ohio State": { latitude: 40, longitude: -83.015 },
  Oregon: { latitude: 44.045, longitude: -123.07 },
  Purdue: { latitude: 40.424, longitude: -86.918 },
  "Santa Clara": { latitude: 37.349, longitude: -121.938 },
  "South Florida": { latitude: 28.063, longitude: -82.413 },
  "Southern Methodist": { latitude: 32.842, longitude: -96.785 },
  Stanford: { latitude: 37.427, longitude: -122.17 },
  "St. John's": { latitude: 40.722, longitude: -73.795 },
  Tennessee: { latitude: 35.955, longitude: -83.925 },
  Texas: { latitude: 30.285, longitude: -97.735 },
  "Texas Tech": { latitude: 33.584, longitude: -101.878 },
  UCLA: { latitude: 34.068, longitude: -118.445 },
  Vanderbilt: { latitude: 36.144, longitude: -86.802 },
  Virginia: { latitude: 38.034, longitude: -78.51 },
  "Virginia Tech": { latitude: 37.228, longitude: -80.423 },
  Washington: { latitude: 47.655, longitude: -122.304 },
};
const MOVE_FILTERS: Array<{ value: MoveFilter; label: string }> = [
  { value: "all", label: "All moves" },
  { value: "external", label: "External arrivals" },
  { value: "rookies", label: "Rookies" },
  { value: "trade", label: "Trades" },
  { value: "signing", label: "Signings" },
  { value: "waiver", label: "Waivers" },
];
const MOVE_COLORS: Record<RosterMove["move_type"], string> = {
  trade: "#e8502f",
  signing: "#2563a8",
  waiver: "#9a5f12",
  other: "#69716b",
};

function parseMoveRoute(): MoveRoute {
  const queryIndex = window.location.hash.indexOf("?");
  const parameters = new URLSearchParams(queryIndex >= 0 ? window.location.hash.slice(queryIndex + 1) : "");
  const requestedFilter = parameters.get("filter") as MoveFilter | null;
  const filter = MOVE_FILTERS.some((candidate) => candidate.value === requestedFilter) ? requestedFilter! : "all";
  return {
    filter,
    team: parameters.get("team")?.toUpperCase() ?? "all",
    autoplay: parameters.get("autoplay") === "1",
  };
}

function animationLink(team: string, filter: MoveFilter) {
  const parameters = new URLSearchParams({ team, autoplay: "1" });
  if (filter !== "all") parameters.set("filter", filter);
  return `${window.location.origin}${window.location.pathname}#moves?${parameters.toString()}`;
}
const TEAM_LOGO_SLUGS: Record<string, string> = {
  NOP: "no",
  UTA: "utah",
};

function teamLogoUrl(team: string) {
  const slug = TEAM_LOGO_SLUGS[team] ?? team.toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/nba/500/${slug}.png`;
}

function curveGeometry(sourceX: number, sourceY: number, targetX: number, targetY: number, playerId: number): CurveGeometry {
  const deltaX = targetX - sourceX;
  const deltaY = targetY - sourceY;
  const distance = Math.max(1, Math.hypot(deltaX, deltaY));
  const bend = 22 + (playerId % 5) * 9;
  const direction = playerId % 2 === 0 ? 1 : -1;
  const controlX = (sourceX + targetX) / 2 + (-deltaY / distance) * bend * direction;
  const controlY = (sourceY + targetY) / 2 + (deltaX / distance) * bend * direction;
  return {
    path: `M ${sourceX},${sourceY} Q ${controlX},${controlY} ${targetX},${targetY}`,
    pointAt: (t: number) => ({
      x: (1 - t) ** 2 * sourceX + 2 * (1 - t) * t * controlX + t ** 2 * targetX,
      y: (1 - t) ** 2 * sourceY + 2 * (1 - t) * t * controlY + t ** 2 * targetY,
    }),
    angleAt: (t: number) => Math.atan2(
      2 * (1 - t) * (controlY - sourceY) + 2 * t * (targetY - controlY),
      2 * (1 - t) * (controlX - sourceX) + 2 * t * (targetX - controlX),
    ) * (180 / Math.PI),
  };
}

function escapeSvg(value: string) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function graphPositionsFromNodes(nodes: GraphNode[]): GraphPositions {
  return Object.fromEntries(nodes.flatMap((node) => (
    node.type === "team" || node.type === "outside" ? [[node.id, node.position] as const] : []
  )));
}

function externalNodeId(playerId: number) {
  return `outside-${playerId}`;
}

function externalArrivalMove(arrival: ExternalRosterArrival): GraphMove {
  const school = arrival.school ?? undefined;
  const hasMappedSchool = school !== undefined && COLLEGE_GEOGRAPHY[school] !== undefined;
  const country = arrival.country ?? undefined;
  return {
    ...arrival,
    source_team: externalNodeId(arrival.player_id),
    prior_season_minutes: 0,
    isExternalArrival: true,
    externalOriginLabel: hasMappedSchool ? school : country ?? "Outside prior roster",
    externalOriginKind: hasMappedSchool ? "college" : country ? "country" : "fallback",
    isRookie: arrival.is_rookie,
  };
}

function graphMovesFromPayload(payload: RosterMovesPayload): GraphMove[] {
  return [
    ...payload.moves.map((move) => ({ ...move, isExternalArrival: false })),
    ...payload.external_arrivals.map(externalArrivalMove),
  ];
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
  const results = await Promise.all(teams.map(async (team) => {
    try {
      const response = await fetch(teamLogoUrl(team));
      if (!response.ok) return [team, ""] as const;
      return [team, await blobAsDataUrl(await response.blob())] as const;
    } catch {
      return [team, ""] as const;
    }
  }));
  return Object.fromEntries(results);
}

function playerHeadshotUrl(playerId: number) {
  return `/api/headshots/${playerId}.png`;
}

function travelerHeadshotGeometry(projectedRating: number | null | undefined) {
  const positiveRating = Math.max(0, projectedRating ?? 0);
  const scale = Math.min(
    TRAVELER_HEADSHOT_MAX_SCALE,
    1 + ((TRAVELER_HEADSHOT_MAX_SCALE - 1) * positiveRating) / TRAVELER_HEADSHOT_RATING_AT_MAX_SCALE,
  );
  const size = TRAVELER_HEADSHOT_MIN_SIZE * scale;
  const radius = size / 2;
  const rightEdge = -5;
  const centerX = rightEdge - radius;
  return {
    centerX,
    labelY: radius + 12,
    left: rightEdge - size,
    radius,
    size,
    strokeWidth: Math.max(1.25, Math.min(2.2, size * 0.055)),
  };
}

async function loadPlayerHeadshotDataUrls(moves: GraphMove[]) {
  const playerIds = [...new Set(moves.map((move) => move.player_id))];
  const results = await Promise.all(playerIds.map(async (playerId) => {
    try {
      const response = await fetch(playerHeadshotUrl(playerId));
      if (!response.ok) return [playerId, ""] as const;
      return [playerId, await blobAsDataUrl(await response.blob())] as const;
    } catch {
      return [playerId, ""] as const;
    }
  }));
  return Object.fromEntries(results);
}

function makeGifFrame(
  teams: string[],
  moves: GraphMove[],
  positions: GraphPositions,
  logoDataUrls: Record<string, string>,
  headshotDataUrls: Record<number, string>,
  title: string,
  progress: number,
) {
  const statePaths = (usaMap.locations as UsaMapLocation[])
    .filter((location) => location.id !== "ak" && location.id !== "hi")
    .map((location) => `<path d="${location.path}" />`)
    .join("");
  const routeMarkup = moves.map((move) => {
    const source = positions[move.source_team];
    const target = positions[move.target_team];
    if (!source || !target) return "";
    const sourceX = move.isExternalArrival ? source.x : source.x + TEAM_NODE_WIDTH / 2;
    const sourceY = move.isExternalArrival ? source.y : source.y + TEAM_NODE_HEIGHT / 2;
    const route = curveGeometry(
      sourceX,
      sourceY,
      target.x + TEAM_NODE_WIDTH / 2,
      target.y + TEAM_NODE_HEIGHT / 2,
      move.player_id,
    );
    const phase = (move.player_id % 7) / 7;
    const point = route.pointAt((progress + phase) % 1);
    const angle = route.angleAt((progress + phase) % 1);
    const color = MOVE_COLORS[move.move_type];
    const width = move.isExternalArrival ? 1.8 : Math.min(4.6, 1.15 + Math.sqrt(move.prior_season_minutes) / 18);
    const dashArray = move.isExternalArrival ? " stroke-dasharray=\"7 5\"" : "";
    const headshot = headshotDataUrls[move.player_id];
    const headshotId = `headshot-${move.player_id}`;
    const headshotGeometry = travelerHeadshotGeometry(move.projected_rating);
    return `<path d="${route.path}" fill="none" marker-end="url(#move-arrow)" opacity="0.72" stroke="${color}" stroke-width="${width}"${dashArray} />
      ${headshot ? `<image clip-path="url(#${headshotId})" height="${headshotGeometry.size}" href="${escapeSvg(headshot)}" preserveAspectRatio="xMidYMin slice" width="${headshotGeometry.size}" x="${point.x + headshotGeometry.left}" y="${point.y - headshotGeometry.radius}" /><circle class="traveler-headshot-ring" cx="${point.x + headshotGeometry.centerX}" cy="${point.y}" r="${headshotGeometry.radius}" stroke-width="${headshotGeometry.strokeWidth}" />` : ""}
      <g class="plane" transform="translate(${point.x - 12} ${point.y - 12}) rotate(${angle} 12 12)"><path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z" /></g>
      <text class="traveler" text-anchor="middle" x="${point.x + headshotGeometry.centerX}" y="${point.y + headshotGeometry.labelY}">${escapeSvg(move.player_name)}</text>`;
  }).join("");
  const teamMarkup = teams.map((team) => {
    const position = positions[team];
    const geography = TEAM_GEOGRAPHY[team];
    if (!position || !geography) return "";
    const logo = logoDataUrls[team];
    const labelX = logo ? 31 : 8;
    return `<g class="team" transform="translate(${position.x} ${position.y})"><rect height="${TEAM_NODE_HEIGHT}" rx="3" width="${TEAM_NODE_WIDTH}" />${logo ? `<image height="21" href="${escapeSvg(logo)}" preserveAspectRatio="xMidYMid meet" width="21" x="5" y="6" />` : ""}<text x="${labelX}" y="21">${team}</text></g>`;
  }).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" height="${GIF_HEIGHT}" viewBox="0 0 ${MAP_WIDTH} ${MAP_HEIGHT}" width="${GIF_WIDTH}">
    <style>
      text { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
      .map path { fill: #edf2e9; stroke: #cbd4c7; stroke-width: .7; }
      .team rect { fill: none; stroke: none; }
      .team text { fill: #17201c; font-size: 11px; font-weight: 800; }
      .traveler { fill: #17201c; font-size: 13px; font-weight: 800; paint-order: stroke; stroke: #fffefa; stroke-linejoin: round; stroke-width: 4px; }
      .traveler-headshot-ring { fill: none; stroke: #17201c; stroke-width: 1.25px; }
      .plane path { fill: #fffefa; stroke: #17201c; stroke-linecap: round; stroke-linejoin: round; stroke-width: 1.6px; }
      .heading { fill: #17201c; font-family: Georgia, serif; font-size: 29px; }
      .subtitle { fill: #69716b; font-size: 12px; font-weight: 800; }
    </style>
    <rect fill="#fffefa" height="${MAP_HEIGHT}" width="${MAP_WIDTH}" />
    <svg class="map" height="${MAP_HEIGHT}" viewBox="270 20 950 525" width="${MAP_WIDTH}">${statePaths}</svg>
    <text class="heading" x="36" y="52">NBA GESTALT · ${escapeSvg(title)}</text>
    <text class="subtitle" x="37" y="75">2025-26 to 2026-27 offseason movement</text>
    <defs><marker id="move-arrow" markerHeight="7" markerWidth="7" orient="auto" refX="6" refY="3.5"><path d="M0,0 L7,3.5 L0,7 z" fill="#17201c" /></marker>${moves.map((move) => `<clipPath clipPathUnits="objectBoundingBox" id="headshot-${move.player_id}"><circle cx=".5" cy=".5" r=".5" /></clipPath>`).join("")}</defs>
    <g>${routeMarkup}</g>
    <g>${teamMarkup}</g>
  </svg>`;
}

async function drawGifFrame(canvas: HTMLCanvasElement, svg: string) {
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas export is unavailable in this browser.");
  const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
  try {
    const image = new Image();
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error("Could not render the movement map."));
      image.src = url;
    });
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function CurvedMoveEdge({
  data,
  id,
  markerEnd,
  selected,
  sourceX,
  sourceY,
  style,
  targetX,
  targetY,
}: EdgeProps) {
  const move = data as MoveEdgeData | undefined;
  const playerId = move?.player_id ?? 0;
  const animateTeamSelection = move?.animateTeamSelection ?? false;
  const deltaX = targetX - sourceX;
  const deltaY = targetY - sourceY;
  const distance = Math.max(1, Math.hypot(deltaX, deltaY));
  const bend = 22 + (playerId % 5) * 9;
  const direction = playerId % 2 === 0 ? 1 : -1;
  const controlX = (sourceX + targetX) / 2 + (-deltaY / distance) * bend * direction;
  const controlY = (sourceY + targetY) / 2 + (deltaX / distance) * bend * direction;
  const labelT = 0.76;
  const labelStartWeight = (1 - labelT) ** 2;
  const labelControlWeight = 2 * (1 - labelT) * labelT;
  const labelTargetWeight = labelT ** 2;
  const labelX = labelStartWeight * sourceX + labelControlWeight * controlX + labelTargetWeight * targetX;
  const labelY = labelStartWeight * sourceY + labelControlWeight * controlY + labelTargetWeight * targetY;
  const path = `M ${sourceX},${sourceY} Q ${controlX},${controlY} ${targetX},${targetY}`;
  const headshotClipId = `roster-move-headshot-${id}`;
  const headshotGeometry = travelerHeadshotGeometry(move?.projected_rating);

  return (
    <>
      <BaseEdge id={id} interactionWidth={18} markerEnd={markerEnd} path={path} style={style} />
      {(selected || animateTeamSelection) && (
        <>
          <defs>
            <clipPath clipPathUnits="objectBoundingBox" id={headshotClipId}>
              <circle cx=".5" cy=".5" r=".5" />
            </clipPath>
          </defs>
          <g aria-hidden="true" className={`roster-edge-traveler-headshot ${selected ? "selected" : "team-selected"}`}>
            <image
              clipPath={`url(#${headshotClipId})`}
              height={headshotGeometry.size}
              href={playerHeadshotUrl(playerId)}
              preserveAspectRatio="xMidYMin slice"
              width={headshotGeometry.size}
              x={headshotGeometry.left}
              y={-headshotGeometry.radius}
            />
            <circle
              cx={headshotGeometry.centerX}
              cy="0"
              r={headshotGeometry.radius}
              strokeWidth={headshotGeometry.strokeWidth}
            />
            <animateMotion begin={`-${(playerId % 7) * 0.55}s`} dur={selected ? "4s" : "5s"} path={path} repeatCount="indefinite" />
          </g>
          <g aria-hidden="true" className={`roster-edge-traveler-plane ${selected ? "selected" : "team-selected"}`}>
            <g transform="translate(-12 -12)">
              <g transform="rotate(45 12 12)">
                <path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z" />
              </g>
            </g>
            <animateMotion begin={`-${(playerId % 7) * 0.55}s`} dur={selected ? "4s" : "5s"} path={path} repeatCount="indefinite" rotate="auto" />
          </g>
          {(selected || animateTeamSelection) && (
            <text aria-hidden="true" className="roster-edge-traveler-label" textAnchor="middle" x={headshotGeometry.centerX} y={headshotGeometry.labelY}>{move?.player_name}
              <animateMotion begin={`-${(playerId % 7) * 0.55}s`} dur={selected ? "4s" : "5s"} path={path} repeatCount="indefinite" />
            </text>
          )}
        </>
      )}
      {move && !selected && !animateTeamSelection && move.showLabel && (
        <EdgeLabelRenderer>
          <span
            className={`roster-edge-label ${move.move_type}`}
            style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
          >
            {move.player_name}
          </span>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

function TeamNode({ data }: NodeProps<Node<TeamNodeData>>) {
  return (
    <div className="roster-team-node">
      <Handle type="target" position={Position.Left} />
      <img alt="" src={teamLogoUrl(data.team)} />
      <span className="roster-team-node-name"><strong>{data.team}</strong></span>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

function OutsideNode({ data }: NodeProps<Node<OutsideNodeData>>) {
  return (
    <div className={`roster-outside-node ${data.isCollege ? "college" : "external"}`} title={data.label}>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

function UsaMapNode() {
  const contiguousStates = usaMap.locations as UsaMapLocation[];
  return (
    <div className="roster-us-map" aria-hidden="true">
      <svg viewBox="270 20 950 525" xmlns="http://www.w3.org/2000/svg">
        {contiguousStates.filter((location) => location.id !== "ak" && location.id !== "hi").map((location) => (
          <path d={location.path} data-state={location.id} key={location.id} />
        ))}
      </svg>
    </div>
  );
}

const nodeTypes = { team: TeamNode, usaMap: UsaMapNode, outside: OutsideNode };
const edgeTypes = { move: CurvedMoveEdge };

function geographicMapPoint(geography: { latitude: number; longitude: number }) {
  const projected = continentalProjection([geography.longitude, geography.latitude]);
  if (!projected) return { x: 0, y: 0 };
  const [projectedX, projectedY] = projected;
  return {
    x: ((projectedX - CONTINENTAL_PROJECTION_BOUNDS.left) / CONTINENTAL_PROJECTION_BOUNDS.width) * MAP_WIDTH,
    y: ((projectedY - CONTINENTAL_PROJECTION_BOUNDS.top) / CONTINENTAL_PROJECTION_BOUNDS.height) * MAP_HEIGHT,
  };
}

function teamMapPosition(geography: typeof TEAM_GEOGRAPHY[string]) {
  const point = geographicMapPoint(geography);
  return {
    x: point.x - TEAM_NODE_WIDTH / 2 + (geography.offsetX ?? 0),
    y: point.y - TEAM_NODE_HEIGHT / 2 + (geography.offsetY ?? 0),
  };
}

function externalOriginPosition(target: { x: number; y: number }, playerId: number) {
  const targetX = target.x + TEAM_NODE_WIDTH / 2;
  const targetY = target.y + TEAM_NODE_HEIGHT / 2;
  const horizontal = targetX - MAP_WIDTH / 2;
  const vertical = targetY - MAP_HEIGHT / 2;
  const jitter = ((playerId % 9) - 4) * 14;
  if (Math.abs(horizontal) >= Math.abs(vertical)) {
    return {
      x: horizontal < 0 ? 28 : MAP_WIDTH - 28,
      y: Math.max(28, Math.min(MAP_HEIGHT - 28, targetY + jitter)),
    };
  }
  return {
    x: Math.max(28, Math.min(MAP_WIDTH - 28, targetX + jitter)),
    y: vertical < 0 ? 28 : MAP_HEIGHT - 28,
  };
}

function countryOriginPosition(target: { x: number; y: number }, playerId: number, country: string) {
  const targetX = target.x + TEAM_NODE_WIDTH / 2;
  const targetY = target.y + TEAM_NODE_HEIGHT / 2;
  const jitter = ((playerId % 9) - 4) * 14;
  if (country === "Canada") {
    return { x: Math.max(28, Math.min(MAP_WIDTH - 28, targetX + jitter)), y: 28 };
  }
  if (country === "Mexico" || country === "Trinidad and Tobago") {
    return { x: Math.max(28, Math.min(MAP_WIDTH - 28, targetX + jitter)), y: MAP_HEIGHT - 28 };
  }
  return { x: MAP_WIDTH - 28, y: Math.max(28, Math.min(MAP_HEIGHT - 28, targetY + jitter)) };
}

function buildGraphNodes(teams: string[], moves: GraphMove[]): GraphNode[] {
  const mapNode: Node<MapNodeData, "usaMap"> = {
    id: "usa-map",
    type: "usaMap",
    position: { x: 0, y: 0 },
    data: {},
    draggable: false,
    selectable: false,
    connectable: false,
    focusable: false,
    zIndex: -1,
  };
  const teamNodes: Node<TeamNodeData, "team">[] = teams.flatMap((team) => {
    const geography = TEAM_GEOGRAPHY[team];
    if (!geography) return [];
    return [{
      id: team,
      type: "team",
      position: teamMapPosition(geography),
      data: { team, conference: geography.conference },
      draggable: true,
      connectable: false,
      selectable: false,
    }];
  });
  const teamPositions = Object.fromEntries(teamNodes.map((node) => [node.id, node.position]));
  const outsideNodes: Node<OutsideNodeData, "outside">[] = moves.flatMap((move) => {
    if (!move.isExternalArrival) return [];
    const target = teamPositions[move.target_team];
    if (!target) return [];
    const school = move.externalOriginKind === "college" ? COLLEGE_GEOGRAPHY[move.externalOriginLabel ?? ""] : undefined;
    const position = school
      ? geographicMapPoint(school)
      : move.externalOriginKind === "country"
        ? countryOriginPosition(target, move.player_id, move.externalOriginLabel ?? "")
        : externalOriginPosition(target, move.player_id);
    return [{
      id: move.source_team,
      type: "outside",
      position,
      data: { label: move.externalOriginLabel ?? "Outside prior roster", isCollege: school !== undefined },
      draggable: false,
      selectable: false,
      connectable: false,
      focusable: false,
    }];
  });
  return [mapNode, ...teamNodes, ...outsideNodes];
}

function moveEdge(
  move: GraphMove,
  selectedMoveId: number | null,
  showLabel: boolean,
  animateTeamSelection: boolean,
): Edge<MoveEdgeData> {
  const isSelected = selectedMoveId === move.player_id;
  const width = move.isExternalArrival ? 1.8 : Math.min(4.6, 1.15 + Math.sqrt(move.prior_season_minutes) / 18);
  return {
    id: `move-${move.player_id}`,
    source: move.source_team,
    target: move.target_team,
    type: "move",
    data: { ...move, showLabel, animateTeamSelection },
    selected: isSelected,
    markerEnd: { type: MarkerType.ArrowClosed, color: MOVE_COLORS[move.move_type] },
    style: {
      stroke: isSelected ? "#17201c" : MOVE_COLORS[move.move_type],
      strokeWidth: isSelected ? width + 1.8 : width,
      strokeDasharray: move.isExternalArrival ? "7 5" : undefined,
      opacity: isSelected ? 1 : 0.6,
    },
  };
}

function formatMinutes(minutes: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(minutes);
}

function formatRating(value: number | null | undefined) {
  if (value === null || value === undefined) return "--";
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
}

export function RosterMovesPage() {
  const initialRoute = parseMoveRoute();
  const [payload, setPayload] = useState<RosterMovesPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [moveFilter, setMoveFilter] = useState<MoveFilter>(initialRoute.filter);
  const [teamFilter, setTeamFilter] = useState(initialRoute.team);
  const [autoplay, setAutoplay] = useState(initialRoute.autoplay);
  const [copiedAnimationLink, setCopiedAnimationLink] = useState(false);
  const [selectedMoveId, setSelectedMoveId] = useState<number | null>(null);
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [flowInstance, setFlowInstance] = useState<ReactFlowInstance<GraphNode, Edge<MoveEdgeData>> | null>(null);
  const [canvasMode, setCanvasMode] = useState<CanvasMode>("move");
  const [isExportingGif, setIsExportingGif] = useState(false);
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        setError(null);
        const response = await fetch("/api/roster-moves", { signal: controller.signal });
        if (!response.ok) throw new Error("Roster movement data is unavailable.");
        const nextPayload = (await response.json()) as RosterMovesPayload;
        setPayload(nextPayload);
        setNodes(buildGraphNodes(nextPayload.teams, graphMovesFromPayload(nextPayload)));
      } catch (fetchError) {
        if ((fetchError as Error).name !== "AbortError") setError((fetchError as Error).message);
      }
    })();
    return () => controller.abort();
  }, []);

  const allMoves = useMemo(
    () => (payload ? graphMovesFromPayload(payload) : []),
    [payload],
  );
  const visibleMoves = useMemo(() => allMoves.filter((move) => (
    (
      moveFilter === "all"
      || (moveFilter === "external" && move.isExternalArrival)
      || (moveFilter === "rookies" && move.isRookie)
      || (moveFilter !== "external" && moveFilter !== "rookies" && move.move_type === moveFilter)
    )
    && (teamFilter === "all" || move.source_team === teamFilter || move.target_team === teamFilter)
  )), [allMoves, moveFilter, teamFilter]);
  const showEdgeLabels = visibleMoves.length <= 24;
  const animateTeamSelection = teamFilter !== "all" || autoplay;
  const canExportGif = visibleMoves.length > 0 && visibleMoves.length <= 24;
  const canShareAnimation = teamFilter !== "all" && visibleMoves.length > 0 && visibleMoves.length <= 24;
  const edges = useMemo(
    () => visibleMoves.map((move) => moveEdge(move, selectedMoveId, showEdgeLabels, animateTeamSelection)),
    [animateTeamSelection, selectedMoveId, showEdgeLabels, visibleMoves],
  );
  const selectedMove = useMemo(
    () => visibleMoves.find((move) => move.player_id === selectedMoveId) ?? null,
    [selectedMoveId, visibleMoves],
  );
  const priorSeasonRating = selectedPlayer?.rating_history.find(
    (point) => point.season === payload?.prior_season,
  ) ?? null;
  const onNodesChange = useCallback(
    (changes: NodeChange<GraphNode>[]) => setNodes((current) => applyNodeChanges(changes, current)),
    [],
  );
  const onEdgeClick = useCallback<EdgeMouseHandler>((_, edge) => {
    setSelectedMoveId((edge.data as GraphMove | undefined)?.player_id ?? null);
  }, []);
  const selectTeam = useCallback((team: string) => {
    setTeamFilter((current) => (current === team ? "all" : team));
    setAutoplay(false);
    setSelectedMoveId(null);
  }, []);
  const clearGraphSelection = useCallback(() => {
    setTeamFilter("all");
    setAutoplay(false);
    setSelectedMoveId(null);
  }, []);
  const copyAnimationLink = useCallback(async () => {
    if (!canShareAnimation) return;
    await navigator.clipboard.writeText(animationLink(teamFilter, moveFilter));
    setCopiedAnimationLink(true);
    window.setTimeout(() => setCopiedAnimationLink(false), 1800);
  }, [canShareAnimation, moveFilter, teamFilter]);
  const exportGif = useCallback(async () => {
    if (!payload || !canExportGif || isExportingGif) return;
    setIsExportingGif(true);
    try {
      const canvas = document.createElement("canvas");
      canvas.width = GIF_WIDTH;
      canvas.height = GIF_HEIGHT;
      const encoder = GIFEncoder();
      const positions = graphPositionsFromNodes(nodes);
      const title = teamFilter === "all" ? "Roster moves" : `${teamFilter} roster moves`;
      const [logoDataUrls, headshotDataUrls] = await Promise.all([
        loadTeamLogoDataUrls(payload.teams),
        loadPlayerHeadshotDataUrls(visibleMoves),
      ]);
      for (let frame = 0; frame < GIF_FRAME_COUNT; frame += 1) {
        const svg = makeGifFrame(
          payload.teams,
          visibleMoves,
          positions,
          logoDataUrls,
          headshotDataUrls,
          title,
          frame / GIF_FRAME_COUNT,
        );
        await drawGifFrame(canvas, svg);
        const imageData = canvas.getContext("2d")?.getImageData(0, 0, canvas.width, canvas.height);
        if (!imageData) throw new Error("Could not encode the movement map.");
        const rgba = new Uint8Array(imageData.data.buffer, imageData.data.byteOffset, imageData.data.byteLength);
        const palette = quantize(rgba, 256);
        encoder.writeFrame(applyPalette(rgba, palette), canvas.width, canvas.height, { delay: GIF_FRAME_DELAY, palette });
      }
      encoder.finish();
      const fileName = `${title.toLowerCase().replaceAll(/[^a-z0-9]+/g, "-").replaceAll(/^-|-$/g, "")}.gif`;
      const encodedBytes = encoder.bytesView();
      const encodedBuffer = encodedBytes.buffer.slice(encodedBytes.byteOffset, encodedBytes.byteOffset + encodedBytes.byteLength) as ArrayBuffer;
      const url = URL.createObjectURL(new Blob([encodedBuffer], { type: "image/gif" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = fileName;
      link.click();
      URL.revokeObjectURL(url);
    } finally {
      setIsExportingGif(false);
    }
  }, [canExportGif, isExportingGif, nodes, payload, teamFilter, visibleMoves]);
  const onNodeClick = useCallback<NodeMouseHandler>((_, node) => {
    if (node.type === "team") {
      selectTeam(node.id);
      return;
    }
    clearGraphSelection();
  }, [clearGraphSelection, selectTeam]);

  useEffect(() => {
    if (selectedMoveId !== null && !visibleMoves.some((move) => move.player_id === selectedMoveId)) {
      setSelectedMoveId(null);
    }
  }, [selectedMoveId, visibleMoves]);

  useEffect(() => {
    if (!selectedMove) {
      setSelectedPlayer(null);
      return;
    }
    const controller = new AbortController();
    setSelectedPlayer(null);
    void (async () => {
      try {
        const response = await fetch(`/api/players/${selectedMove.player_id}`, { signal: controller.signal });
        if (!response.ok) return;
        setSelectedPlayer((await response.json()) as Player);
      } catch (fetchError) {
        if ((fetchError as Error).name !== "AbortError") setSelectedPlayer(null);
      }
    })();
    return () => controller.abort();
  }, [selectedMove]);

  useEffect(() => {
    if (payload && teamFilter !== "all" && !payload.teams.includes(teamFilter)) {
      setTeamFilter("all");
      setAutoplay(false);
    }
  }, [payload, teamFilter]);

  useEffect(() => {
    if (!flowInstance || nodes.length === 0) return;
    const frame = window.requestAnimationFrame(() => flowInstance.fitView({ padding: 0.08, duration: 0 }));
    return () => window.cancelAnimationFrame(frame);
  }, [flowInstance, nodes.length]);

  if (error) return <p className="error roster-moves-error"><CircleAlert size={16} /> {error}</p>;
  if (!payload) return <div className="profile-loading"><LoaderCircle className="spin" size={20} /> Loading roster movement graph</div>;

  return (
    <article className="roster-moves-page" aria-labelledby="roster-moves-title">
      <section className="roster-moves-hero">
        <p className="eyebrow">{payload.prior_season} to {payload.current_season} offseason</p>
        <h1 id="roster-moves-title">Roster moves.</h1>
        <p>
          Each solid arrow follows a returning player from their {payload.source_definition} to their listed 2026-27 club.
          Dashed arrows mark players entering from outside the prior NBA roster record.
          Animated headshots scale with the 2026-27 preseason NAIL projection: non-positive ratings use the base size, while a +6.0 rating reaches the 3x cap.
          Drag team nodes to inspect a path; filter the graph to reveal player labels, then select a path or a player below for transaction detail.
        </p>
      </section>

      <section className="roster-moves-summary" aria-label="Roster movement summary">
        <div><strong>{payload.returning_mover_count}</strong><span>returning movers</span></div>
        <div><strong>{payload.current_roster_count}</strong><span>current rostered players</span></div>
        <div><strong>{payload.new_or_unmatched_current_player_count}</strong><span>external arrivals</span></div>
        <p>External arrivals begin just beyond the map boundary and end at their current team. They include rookies and players without a prior-season NBA roster record.</p>
      </section>

      <section className="roster-moves-workspace" aria-label="Interactive roster movement graph">
        <div className="roster-moves-controls">
          <div className="roster-move-filter-group" aria-label="Movement type filter">
            {MOVE_FILTERS.map((filter) => (
              <button
                className={moveFilter === filter.value ? "active" : ""}
                key={filter.value}
                onClick={() => setMoveFilter(filter.value)}
                type="button"
              >
                {filter.label}
              </button>
            ))}
          </div>
          <label className="roster-moves-team-filter">
            <Filter size={14} aria-hidden="true" />
            <span>Team</span>
            <select value={teamFilter} onChange={(event) => selectTeam(event.target.value)}>
              <option value="all">All teams</option>
              {payload.teams.map((team) => <option key={team} value={team}>{team}</option>)}
            </select>
          </label>
          <div className="roster-graph-mode" aria-label="Canvas interaction mode">
            <button
              aria-label="Move team nodes"
              className={canvasMode === "move" ? "active" : ""}
              onClick={() => setCanvasMode("move")}
              title="Move team nodes"
              type="button"
            ><MousePointer2 size={15} /><span>Move teams</span></button>
            <button
              aria-label="Pan canvas"
              className={canvasMode === "pan" ? "active" : ""}
              onClick={() => setCanvasMode("pan")}
              title="Pan canvas"
              type="button"
            ><Hand size={15} /><span>Pan canvas</span></button>
          </div>
          <button
            aria-label="Download animated roster movement GIF"
            className="roster-gif-export"
            disabled={!canExportGif || isExportingGif}
            onClick={() => void exportGif()}
            title={canExportGif ? "Download the visible player paths as an animated GIF" : "Filter to 24 or fewer player paths to export a readable GIF"}
            type="button"
          >
            {isExportingGif ? <LoaderCircle className="spin" size={14} /> : <Download size={14} />}
            <span>{isExportingGif ? "Building GIF" : "Download GIF"}</span>
          </button>
          <button
            aria-label="Copy a link that opens this team animation"
            className="roster-gif-export"
            disabled={!canShareAnimation}
            onClick={() => void copyAnimationLink()}
            title={canShareAnimation ? "Copy a link that opens and animates these player paths" : "Select one team and filter to 24 or fewer player paths to create an animation link"}
            type="button"
          >
            <Copy size={14} />
            <span>{copiedAnimationLink ? "Link copied" : "Copy animation link"}</span>
          </button>
          <p>{visibleMoves.length} player paths shown</p>
        </div>

        <div className="roster-moves-canvas">
          <ReactFlow<GraphNode, Edge<MoveEdgeData>>
            edges={edges}
            edgeTypes={edgeTypes}
            maxZoom={1.55}
            minZoom={0.28}
            nodesDraggable={canvasMode === "move"}
            nodeTypes={nodeTypes}
            nodes={nodes}
            onEdgeClick={onEdgeClick}
            onInit={setFlowInstance}
            onNodeClick={onNodeClick}
            onNodesChange={onNodesChange}
            onPaneClick={clearGraphSelection}
            panOnDrag={canvasMode === "pan" ? true : [1, 2]}
            proOptions={{ hideAttribution: true }}
          >
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>

        <aside className="roster-moves-detail" aria-live="polite">
          {selectedMove ? (
            <>
              <button
                className="roster-moves-dismiss"
                type="button"
                onClick={() => setSelectedMoveId(null)}
                aria-label="Clear selected movement"
                title="Clear selected movement"
              ><X size={16} /></button>
              <p className={`roster-move-kind ${selectedMove.move_type}`}>{selectedMove.isExternalArrival ? "external arrival" : selectedMove.move_type}</p>
              <a className="roster-move-player-link" href={`#player/${selectedMove.player_id}`}>
                <h2>{selectedMove.player_name}</h2>
              </a>
              <p className="roster-move-route"><strong>{selectedMove.isExternalArrival ? selectedMove.externalOriginLabel ?? "Outside prior roster" : selectedMove.source_team}</strong><span>to</span><strong>{selectedMove.target_team}</strong></p>
              <dl className="roster-move-player-bio">
                <div>
                  <dt>{payload.prior_season} NAIL</dt>
                  <dd className={priorSeasonRating && priorSeasonRating.rating < 0 ? "negative" : ""}>{priorSeasonRating ? formatRating(priorSeasonRating.rating) : "No prior-season fit"}</dd>
                </div>
                <div>
                  <dt>{payload.current_season} projection</dt>
                  <dd className={selectedMove.projected_rating !== null && selectedMove.projected_rating < 0 ? "negative" : ""}>{formatRating(selectedMove.projected_rating)}</dd>
                </div>
              </dl>
              <p>{selectedMove.how_acquired ?? "Current roster acquisition detail unavailable."}</p>
              <small>{selectedMove.isExternalArrival ? `No ${payload.prior_season} NBA roster record` : `${formatMinutes(selectedMove.prior_season_minutes)} regular-season minutes in ${payload.prior_season}`}</small>
            </>
          ) : (
            <>
              <p className="section-kicker">Selected path</p>
              <h2>Choose a move.</h2>
              <p>Each colored arrow represents one player. Select an arrow in the network or a row in the list.</p>
            </>
          )}
        </aside>
      </section>

      <section className="roster-moves-list-section" aria-labelledby="roster-moves-list-title">
        <div className="roster-moves-list-heading">
          <div>
            <p className="section-kicker">All displayed movement</p>
            <h2 id="roster-moves-list-title">Player paths.</h2>
          </div>
          <span>{visibleMoves.length} of {allMoves.length}</span>
        </div>
        <ol className="roster-moves-list">
          {visibleMoves.map((move) => (
            <li key={move.player_id}>
              <button
                className={selectedMoveId === move.player_id ? "active" : ""}
                type="button"
                onClick={() => setSelectedMoveId(move.player_id)}
              >
                <span className={`roster-move-dot ${move.move_type}`} />
                <strong>{move.player_name}</strong>
                <span>{move.isExternalArrival ? move.externalOriginLabel ?? "Outside" : move.source_team}</span><span className="roster-move-arrow">to</span><span>{move.target_team}</span>
                <small>{move.isExternalArrival ? "No prior roster record" : `${formatMinutes(move.prior_season_minutes)} min.`}</small>
              </button>
            </li>
          ))}
        </ol>
      </section>
    </article>
  );
}
