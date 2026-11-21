import { select } from "d3-selection";
import "d3-transition";
import { interpolateNumber } from "d3-interpolate";
import { rotationHeatColor, rotationPlayerLabel } from "./teamRotationColors";
import { cubicInOut } from "./teamRotationAnimationTypes";

const INK = "#17201c";
const MUTED = "#58635b";
const MISSING = "#d9d6ce";

export function mountRotationAnimation(root, { players, frames }) {
  const svg = select(root.querySelector("svg"));
  const slider = root.querySelector("input");
  const play = root.querySelector("[data-play]");
  const previous = root.querySelector("[data-previous]");
  const next = root.querySelector("[data-next]");
  const date = root.querySelector("[data-date]");
  const game = root.querySelector("[data-game]");
  const position = root.querySelector("[data-position]");
  const stage = root.querySelector("[data-stage]");
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const events = new AbortController();
  const playerData = players.map((player, index) => ({ ...player, index }));
  const displayedMinutes = new WeakMap();
  let index = 0;
  let playing = false;
  let phase = "idle";
  let timer, layout;
  let revision = 0;

  const rows = svg.append("g").selectAll("g").data(playerData, (player) => player.id)
    .join("g").attr("class", "gs-row").attr("data-player", (player) => player.id);
  const cells = rows.selectAll("rect").data(
    ({ index: row, id }) => players.map((player, column) => ({ row, column, key: `${id}:${player.id}` })),
    (cell) => cell.key,
  ).join("rect").attr("class", "gs-cell").attr("data-pair", (cell) => cell.key);
  const tooltips = cells.append("title");
  const names = rows.append("text").attr("class", "gs-name").attr("text-anchor", "end").attr("dy", ".35em");
  const minutes = rows.append("text").attr("class", "gs-minutes").attr("text-anchor", "end").attr("dy", ".35em");
  const columns = svg.append("g").selectAll("g").data(playerData, (player) => player.id)
    .join("g").attr("class", "gs-column").attr("data-player", (player) => player.id);
  const columnNames = columns.append("text").text((player) => rotationPlayerLabel(player.name));
  const minuteTitle = svg.append("text").attr("text-anchor", "end").text("MIN");

  function annotations() {
    const frame = frames[index];
    date.textContent = frame.label;
    game.textContent = `Game ${frame.game} of ${frames.length}`;
    position.textContent = `${frame.game} of ${frames.length}`;
    slider.value = String(index);
    slider.setAttribute("aria-valuetext", `Game ${frame.game}, ${frame.label}`);
    previous.disabled = index === 0;
    next.disabled = index === frames.length - 1;
    play.disabled = frames.length < 2;
    slider.disabled = frames.length < 2;
    play.textContent = playing ? "Pause" : phase === "paused" ? "Resume" : next.disabled ? "Replay" : "Play";
    stage.textContent = phase === "moving" ? "Transitioning between saved games"
      : phase === "paused" ? "Paused between saved games" : "Saved postgame snapshot";
    root.dataset.game = String(frame.game);
    root.dataset.phase = phase;
  }

  function stopTransitions() {
    revision += 1;
    svg.interrupt("game");
    svg.selectAll("*").interrupt("game");
  }

  function targets(frame) {
    const slots = Array(players.length);
    frame.order.forEach((player, slot) => { slots[player] = slot; });
    const { left, top, cell, row } = layout;
    return {
      rowPosition: (player) => `translate(0,${top + slots[player.index] * row})`,
      cellPosition: ({ column }) => left + slots[column] * cell + 0.5,
      columnPosition: (player) => `translate(${left + (slots[player.index] + 0.5) * cell},${top - 10})`,
      fill: ({ row, column }) => {
        const value = frame.matrix[row][column];
        return value === null ? MISSING : rotationHeatColor(value, 0.5, row === column);
      },
      textColor: (player) => frame.minutes[player.index] === null ? MUTED : INK,
    };
  }

  function settle() {
    const frame = frames[index];
    const target = targets(frame);
    const { left, top, cell, row, narrow } = layout;
    rows.attr("transform", target.rowPosition);
    cells.attr("x", target.cellPosition).attr("y", 0.5)
      .attr("width", Math.max(1, cell - 1)).attr("height", row - 1).attr("fill", target.fill);
    names.attr("x", left - (narrow ? 46 : 56)).attr("y", row / 2).style("fill", target.textColor)
      .text((player) => narrow ? rotationPlayerLabel(player.name) : player.name);
    minutes.attr("x", left - 10).attr("y", row / 2).style("fill", target.textColor)
      .each(function (player) {
        const value = frame.minutes[player.index];
        displayedMinutes.set(this, value);
        this.textContent = value === null ? "—" : value.toFixed(1);
      });
    columns.attr("transform", target.columnPosition);
    columnNames.attr("transform", narrow ? "rotate(-90)" : "rotate(-60)").style("fill", target.textColor);
    minuteTitle.attr("x", left - 10).attr("y", top - 10);
    tooltips.text(({ row, column }) => {
      const value = frame.matrix[row][column];
      return `${players[row].name} × ${players[column].name}: ${value === null ? "not yet appeared" : value.toFixed(3) + " correlation"}`;
    });
    phase = "idle";
    annotations();
  }

  function schedule() {
    clearTimeout(timer);
    if (!playing) return;
    if (index === frames.length - 1) {
      playing = false;
      annotations();
    } else {
      timer = setTimeout(() => move(index + 1), 1000);
    }
  }

  function move(newIndex, animate = true) {
    clearTimeout(timer);
    stopTransitions();
    const currentRevision = revision;
    index = Math.max(0, Math.min(frames.length - 1, newIndex));
    if (!animate || reducedMotion.matches) {
      settle();
      schedule();
      return;
    }
    const frame = frames[index];
    const target = targets(frame);
    phase = "moving";
    annotations();
    tooltips.text("");
    const transition = svg.transition("game").duration(1800).ease(cubicInOut);
    rows.transition(transition).attr("transform", target.rowPosition);
    cells.transition(transition).attr("x", target.cellPosition).attr("fill", target.fill);
    columns.transition(transition).attr("transform", target.columnPosition);
    names.transition(transition).style("fill", target.textColor);
    columnNames.transition(transition).style("fill", target.textColor);
    minutes.transition(transition).style("fill", target.textColor).tween("minutes", function (player) {
      const from = displayedMinutes.get(this);
      const to = frame.minutes[player.index];
      const interpolate = interpolateNumber(from ?? 0, to ?? 0);
      return (t) => {
        const value = from === null && to === null ? null : interpolate(t);
        displayedMinutes.set(this, value);
        this.textContent = value === null ? "—" : value.toFixed(1);
      };
    });
    transition.end().then(() => {
      if (revision !== currentRevision) return;
      settle();
      schedule();
    }).catch(() => { /* A pause, resize, or new selection interrupts this transition. */ });
  }

  function pause() {
    playing = false;
    clearTimeout(timer);
    if (phase === "moving") {
      stopTransitions();
      phase = "paused";
    }
    annotations();
  }

  function resize() {
    const width = root.clientWidth;
    if (width <= 0 || width === layout?.width) return;
    stopTransitions();
    const narrow = width < 560;
    const left = narrow ? Math.min(150, width * 0.48) : Math.min(230, width * 0.36);
    const cell = (width - left - (narrow ? 12 : 48)) / players.length;
    const row = Math.max(23, cell);
    const top = narrow ? 114 : 106;
    const height = Math.ceil(top + players.length * row + 4);
    layout = { width, left, cell, row, top, narrow };
    svg.attr("width", width).attr("height", height).attr("viewBox", `0 0 ${width} ${height}`);
    settle();
    schedule();
  }

  const options = { signal: events.signal };
  previous.addEventListener("click", () => { pause(); move(index - 1); }, options);
  next.addEventListener("click", () => { pause(); move(index + 1); }, options);
  slider.addEventListener("input", () => {
    const target = Number(slider.value);
    pause();
    move(target);
  }, options);
  play.addEventListener("click", () => {
    if (playing) {
      pause();
      return;
    }
    const resume = phase === "paused";
    if (!resume && index === frames.length - 1) move(0, false);
    playing = true;
    annotations();
    if (resume) move(index);
    else if (phase !== "moving") schedule();
  }, options);
  const observer = new ResizeObserver(resize);
  observer.observe(root);
  resize();
  return () => {
    clearTimeout(timer);
    stopTransitions();
    observer.disconnect();
    events.abort();
    svg.selectAll("*").remove();
  };
}
