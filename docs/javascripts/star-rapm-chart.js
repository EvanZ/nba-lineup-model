(() => {
  const svgNamespace = "http://www.w3.org/2000/svg";
  const colors = ["#1e628f", "#a24c25", "#4f8c78", "#614f9b", "#bd7a2a", "#aa4c65", "#2e7d91"];

  function element(name, attributes = {}) {
    const node = document.createElementNS(svgNamespace, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  }

  function render(container, rows) {
    const width = 860;
    const height = 480;
    const margin = { top: 30, right: 205, bottom: 48, left: 60 };
    const ages = rows.map((row) => Number(row.age));
    const ratings = rows.map((row) => Number(row.rapm));
    const xMin = Math.floor(Math.min(...ages));
    const xMax = Math.ceil(Math.max(...ages));
    const yPad = 0.7;
    const yMin = Math.floor(Math.min(...ratings) - yPad);
    const yMax = Math.ceil(Math.max(...ratings) + yPad);
    const x = (value) => margin.left + ((value - xMin) / (xMax - xMin)) * (width - margin.left - margin.right);
    const y = (value) => height - margin.bottom - ((value - yMin) / (yMax - yMin)) * (height - margin.top - margin.bottom);
    const grouped = [...rows.reduce((map, row) => {
      const values = map.get(row.player_name) || [];
      values.push(row);
      map.set(row.player_name, values);
      return map;
    }, new Map()).entries()];
    const svg = element("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Observed one-season RAPM by age" });
    const zero = element("line", { x1: margin.left, x2: width - margin.right, y1: y(0), y2: y(0), class: "star-rapm-zero" });
    svg.append(zero);
    for (let age = xMin; age <= xMax; age += 2) {
      const label = element("text", { x: x(age), y: height - 20, class: "star-rapm-axis", "text-anchor": "middle" });
      label.textContent = age;
      svg.append(label);
    }
    for (let rating = yMin; rating <= yMax; rating += 2) {
      const label = element("text", { x: margin.left - 10, y: y(rating) + 3, class: "star-rapm-axis", "text-anchor": "end" });
      label.textContent = rating;
      svg.append(label);
    }
    const paths = new Map();
    const legends = new Map();
    function focus(name) {
      paths.forEach((group, player) => group.classList.toggle("is-muted", player !== name));
      legends.forEach((legend, player) => legend.classList.toggle("is-muted", player !== name));
    }
    function reset() {
      paths.forEach((group) => group.classList.remove("is-muted"));
      legends.forEach((legend) => legend.classList.remove("is-muted"));
    }
    grouped.forEach(([name, values], index) => {
      values.sort((a, b) => Number(a.age) - Number(b.age));
      const color = colors[index % colors.length];
      const group = element("g", { class: "star-rapm-series", tabindex: "0" });
      const path = element("path", { d: values.map((row, point) => `${point ? "L" : "M"}${x(Number(row.age))},${y(Number(row.rapm))}`).join(" "), stroke: color });
      group.append(path);
      values.forEach((row) => group.append(element("circle", { cx: x(Number(row.age)), cy: y(Number(row.rapm)), r: 3.4, fill: color })));
      group.addEventListener("mouseenter", () => focus(name));
      group.addEventListener("mouseleave", reset);
      group.addEventListener("focus", () => focus(name));
      group.addEventListener("blur", reset);
      svg.append(group);
      paths.set(name, group);
      const legend = element("text", { x: width - margin.right + 18, y: margin.top + index * 25, fill: color, class: "star-rapm-legend", tabindex: "0" });
      legend.textContent = name;
      legend.addEventListener("mouseenter", () => focus(name));
      legend.addEventListener("mouseleave", reset);
      legend.addEventListener("focus", () => focus(name));
      legend.addEventListener("blur", reset);
      svg.append(legend);
      legends.set(name, legend);
    });
    container.replaceChildren(svg);
  }

  function initialize() {
    document.querySelectorAll(".star-rapm-chart[data-source]").forEach(async (container) => {
      const response = await fetch(container.dataset.source);
      if (response.ok) render(container, await response.json());
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize);
  } else {
    initialize();
  }
})();
