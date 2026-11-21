export function rotationPlayerLabel(playerName: string) {
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

export function rotationHeatColor(value: number, midpoint: number, isDiagonal: boolean) {
  if (isDiagonal) return "#17201c";
  const boundedValue = Math.max(0, Math.min(1, value));
  if (midpoint <= 0) return blendedColor([255, 254, 250], [230, 159, 0], boundedValue);
  if (midpoint >= 1) return blendedColor([0, 114, 178], [255, 254, 250], boundedValue);
  if (boundedValue <= midpoint) {
    return blendedColor([0, 114, 178], [255, 254, 250], boundedValue / midpoint);
  }
  return blendedColor([255, 254, 250], [230, 159, 0], (boundedValue - midpoint) / (1 - midpoint));
}

export function rotationCellLabelColor(value: number, midpoint: number) {
  const boundedValue = Math.max(0, Math.min(1, value));
  if (midpoint <= 0) return "#17201c";
  const blueIntensity = boundedValue <= midpoint
    ? 1 - boundedValue / midpoint
    : 0;
  return blueIntensity > 0.55 ? "#fffefa" : "#17201c";
}
