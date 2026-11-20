import { inlineSvgImages } from "./PlayerHistoricalCharts";

/** Export the displayed chart with embedded logos, independently of scroll position. */
export async function downloadTeamHistoryPng(chart: SVGSVGElement, team: string, season: string, adjusted: boolean) {
  const svgNS = "http://www.w3.org/2000/svg";
  const copy = chart.cloneNode(true) as SVGSVGElement;
  const originals = [...chart.querySelectorAll("*")];
  const copies = [...copy.querySelectorAll("*")];
  const properties = ["fill", "stroke", "stroke-width", "stroke-dasharray", "stroke-linecap", "stroke-linejoin", "font-family", "font-size", "font-weight", "opacity", "text-anchor"];
  originals.forEach((node, index) => {
    const style = getComputedStyle(node);
    properties.forEach(property => copies[index].setAttribute(property, style.getPropertyValue(property)));
  });
  const imageCount = copy.querySelectorAll("image").length;
  await inlineSvgImages(copy);
  if (copy.querySelectorAll("image").length !== imageCount) throw new Error("Some team logos could not be loaded. Please try again.");
  const width = chart.viewBox.baseVal.width, height = chart.viewBox.baseVal.height;
  const output = document.createElementNS(svgNS, "svg");
  output.setAttribute("xmlns", svgNS);
  output.setAttribute("viewBox", `0 0 ${width} ${height + 70}`);
  output.setAttribute("width", String(width)); output.setAttribute("height", String(height + 70));
  function text(content: string, y: number, size: number) {
    const node = document.createElementNS(svgNS, "text");
    node.textContent = content;
    node.setAttribute("x", "48"); node.setAttribute("y", String(y));
    node.setAttribute("fill", "#17201c"); node.setAttribute("font-family", "sans-serif");
    node.setAttribute("font-size", String(size)); node.setAttribute("font-weight", "700");
    return node;
  }
  output.append(text(`${team} · ${season} game-by-game NAIL`, 26, 18));
  output.append(text(`${adjusted ? "Adjusted matchup" : "Raw NAIL"} · Green: ${team}${adjusted ? " + HCA/B2B" : ""} · Blue: opponent`, 49, 12));
  const notes = [copy.querySelector(".team-rating-historical_replay") ? "Dashed rings: historical replays" : "", copy.querySelector(".team-rating-upcoming") ? "Dotted rings: upcoming games" : ""].filter(Boolean);
  if (notes.length) output.append(text(notes.join(" · "), 65, 10));
  copy.removeAttribute("style");
  copy.setAttribute("x", "0"); copy.setAttribute("y", "70");
  copy.setAttribute("width", String(width)); copy.setAttribute("height", String(height));
  output.append(copy);
  const sourceUrl = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(output)], { type: "image/svg+xml" }));
  try {
    const image = new Image();
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve(); image.onerror = () => reject(new Error("Could not render the chart PNG."));
      image.src = sourceUrl;
    });
    const canvas = document.createElement("canvas");
    canvas.width = width * 2; canvas.height = (height + 70) * 2;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("PNG export is unavailable in this browser.");
    context.fillStyle = "#f6f3ec"; context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const png = await new Promise<Blob>((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error("Could not encode the chart PNG.")), "image/png"));
    const url = URL.createObjectURL(png);
    const link = document.createElement("a");
    link.href = url; link.download = `${team.toLowerCase()}-${season}-${adjusted ? "adjusted-matchup" : "raw-nail"}-history.png`;
    link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  } finally { URL.revokeObjectURL(sourceUrl); }
}
