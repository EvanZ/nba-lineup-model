import { GIFEncoder, applyPalette } from "gifenc";
import { interpolateNumber, interpolateRgb } from "d3-interpolate";
import { rotationHeatColor, rotationPlayerLabel } from "./teamRotationColors";
import { cubicInOut, type RotationAnimationData, type RotationAnimationFrame } from "./teamRotationAnimationTypes";

const FILM_STEPS = 12;
const PAPER = "#f6f3ec";
const INK = "#17201c";
const MISSING = "#d9d6ce";
const MUTED = "#58635b";

function makePalette() {
  const heat = Array.from({ length: 201 }, (_, i) => rotationHeatColor(i / 200, 0.5, false));
  const stops = [PAPER, INK, MISSING, MUTED, ...heat];
  return stops.map((color) => color.startsWith("#")
    ? [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16))
    : color.match(/\d+/g)!.map(Number));
}

export async function makeRotationGif(
  data: RotationAnimationData,
  teamName: string,
  signal: AbortSignal,
  progress: (value: number) => void,
): Promise<Blob> {
  const { players, frames } = data;
  const width = 624;
  const left = 184;
  const top = 195;
  const size = (width - left - 42) / players.length;
  const height = Math.ceil(top + size * players.length + 68);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("GIF export is unavailable in this browser.");
  const ctx = context;
  const encoder = GIFEncoder();
  const palette = makePalette();
  const ranks = (frame: RotationAnimationFrame) => {
    const positions = Array<number>(players.length);
    frame.order.forEach((player, rank) => { positions[player] = rank; });
    return positions;
  };
  const heat = (frame: RotationAnimationFrame, i: number, j: number) => {
    const value = frame.matrix[i][j];
    return value === null ? MISSING : rotationHeatColor(value, 0.5, i === j);
  };
  function text(value: string, x: number, y: number, font: string, color = INK, align: CanvasTextAlign = "left") {
    ctx.font = font;
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.fillText(value, x, y);
  }

  const total = 1 + (frames.length - 1) * FILM_STEPS;
  let rendered = 0;
  async function transition(from: RotationAnimationFrame, to: RotationAnimationFrame, steps: number) {
    const start = ranks(from);
    const end = ranks(to);
    const position = start.map((value, i) => interpolateNumber(value, end[i]));
    const minutes = from.minutes.map((value, i) => interpolateNumber(value ?? 0, to.minutes[i] ?? 0));
    const colors = players.map((_, i) => players.map((_, j) => interpolateRgb(heat(from, i, j), heat(to, i, j))));
    for (let step = 1; step <= steps; step += 1) {
      signal.throwIfAborted();
      const t = cubicInOut(step / steps);
      const settled = step === steps;
      const slots = position.map((interpolate) => interpolate(t));
      ctx.fillStyle = PAPER;
      ctx.fillRect(0, 0, width, height);
      text(teamName, 20, 35, "25px Georgia");
      text(data.season, width - 20, 34, "14px sans-serif", INK, "right");
      text("Rotation patterns · complete linkage", 20, 60, "14px sans-serif", MUTED);
      text(settled ? to.label : `${from.label} → ${to.label}`, 20, 86, "13px sans-serif");
      text(`Game ${to.game} of ${frames.length}`, width - 20, 86, "13px sans-serif", INK, "right");
      text("MIN", left - 10, top - 10, "11px sans-serif", MUTED, "right");
      players.forEach((_, i) => players.forEach((__, j) => {
        ctx.fillStyle = colors[i][j](t);
        ctx.fillRect(left + slots[j] * size, top + slots[i] * size, size - 1, size - 1);
      }));
      players.forEach((player, i) => {
        const y = top + (slots[i] + 0.5) * size + 4;
        const absent = to.minutes[i] === null && (settled || from.minutes[i] === null);
        text(rotationPlayerLabel(player.name), left - 56, y, "12px monospace", absent ? MUTED : INK, "right");
        text(absent ? "—" : minutes[i](t).toFixed(1), left - 10, y, "12px monospace", INK, "right");
        ctx.save();
        ctx.translate(left + (slots[i] + 0.5) * size, top - 10);
        ctx.rotate(-Math.PI / 3);
        text(rotationPlayerLabel(player.name), 0, 0, "12px monospace", absent ? MUTED : INK);
        ctx.restore();
      });
      const legendY = top + size * players.length + 18;
      const gradient = ctx.createLinearGradient(left, 0, left + 200, 0);
      gradient.addColorStop(0, "#0072b2");
      gradient.addColorStop(0.5, "#fffefa");
      gradient.addColorStop(1, "#e69f00");
      ctx.fillStyle = gradient;
      ctx.fillRect(left, legendY, 200, 8);
      text("≤0", left, legendY + 24, "11px sans-serif", MUTED);
      text("0.50", left + 100, legendY + 24, "11px sans-serif", MUTED, "center");
      text("1", left + 200, legendY + 24, "11px sans-serif", MUTED, "right");
      if (to.minutes.includes(null)) {
        ctx.fillStyle = MISSING;
        ctx.fillRect(left + 220, legendY, 9, 9);
        text("Not yet appeared", left + 236, legendY + 8, "11px sans-serif", MUTED);
      }
      const rgba = new Uint8Array(ctx.getImageData(0, 0, width, height).data.buffer);
      encoder.writeFrame(applyPalette(rgba, palette), width, height, {
        ...(rendered === 0 ? { palette } : {}),
        delay: settled ? (to.game === frames.length ? 2200 : 1000) : 150,
      });
      rendered += 1;
      if (rendered % 3 === 0 || rendered === total) {
        progress(rendered / total);
        await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
      }
    }
  }

  await transition(frames[0], frames[0], 1);
  for (let i = 1; i < frames.length; i += 1) {
    await transition(frames[i - 1], frames[i], FILM_STEPS);
  }
  signal.throwIfAborted();
  encoder.finish();
  return new Blob([new Uint8Array(encoder.bytesView())], { type: "image/gif" });
}
