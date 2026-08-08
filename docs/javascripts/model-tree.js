(() => {
  const selector = "[data-model-tree]";
  const nodeWidth = 208;
  const nodeHeight = 62;
  const nodeRadius = 4;
  const leftMargin = 88;
  const rightMargin = 28;
  const topMargin = 56;
  const rowGap = 72;
  const unrankedGap = 86;

  function metricValue(model, metric) {
    const value = model.metrics?.[metric.id];
    return Number.isFinite(value) ? value : null;
  }

  function formatValue(value, metric) {
    return value.toFixed(metric.decimals);
  }

  function rankModels(models, metric) {
    const eligible = models
      .filter((model) => metricValue(model, metric) !== null)
      .sort((left, right) => {
        const difference = metricValue(left, metric) - metricValue(right, metric);
        return metric.direction === "lower" ? difference : -difference;
      });
    const ranks = new Map();
    eligible.forEach((model, index) => ranks.set(model.id, index + 1));
    return { eligible, ranks };
  }

  function depthFor(model, byId, memo) {
    if (memo.has(model.id)) return memo.get(model.id);
    const depth = model.parent ? depthFor(byId.get(model.parent), byId, memo) + 1 : 0;
    memo.set(model.id, depth);
    return depth;
  }

  function tooltipContent(model, metric, rank, byId) {
    const value = metricValue(model, metric);
    const parent = model.parent ? byId.get(model.parent) : null;
    const parentValue = parent ? metricValue(parent, metric) : null;
    const delta = value !== null && parentValue !== null ? value - parentValue : null;
    const deltaText = delta === null
      ? "No directly comparable parent result"
      : `${delta > 0 ? "+" : ""}${formatValue(delta, metric)} vs parent`;
    return `
      <strong>${model.name}</strong>
      <span>${model.change}</span>
      <dl>
        <div><dt>${metric.label}</dt><dd>${value === null ? "Not evaluated" : formatValue(value, metric)}</dd></div>
        <div><dt>Rank</dt><dd>${rank ? `#${rank}` : "Not evaluated"}</dd></div>
        <div><dt>Parent delta</dt><dd>${deltaText}</dd></div>
      </dl>
      <em>Open model documentation</em>
    `;
  }

  function makeTooltip(container) {
    const tooltip = document.createElement("div");
    tooltip.className = "model-tree__tooltip";
    tooltip.hidden = true;
    container.append(tooltip);
    return tooltip;
  }

  function positionTooltip(tooltip, event) {
    const gap = 14;
    const maxLeft = window.innerWidth - tooltip.offsetWidth - gap;
    const maxTop = window.innerHeight - tooltip.offsetHeight - gap;
    tooltip.style.left = `${Math.max(gap, Math.min(event.clientX + gap, maxLeft))}px`;
    tooltip.style.top = `${Math.max(gap, Math.min(event.clientY + gap, maxTop))}px`;
  }

  function render(container, registry, metricId) {
    const metric = registry.metrics.find((candidate) => candidate.id === metricId);
    const models = registry.models;
    const byId = new Map(models.map((model) => [model.id, model]));
    const memo = new Map();
    models.forEach((model) => {
      model.depth = depthFor(model, byId, memo);
    });
    const { eligible, ranks } = rankModels(models, metric);
    const unranked = models.filter((model) => !ranks.has(model.id));
    const unrankedPositions = new Map(unranked.map((model, index) => [model.id, index]));
    const maxDepth = Math.max(...models.map((model) => model.depth));
    const containerWidth = Math.max(container.clientWidth, 680);
    const chartWidth = Math.max(containerWidth, leftMargin + rightMargin + nodeWidth + maxDepth * 238);
    const chartHeight = topMargin
      + Math.max(1, eligible.length) * rowGap
      + unrankedGap
      + Math.max(1, unranked.length) * rowGap
      + nodeHeight;
    const x = d3.scaleLinear().domain([0, Math.max(1, maxDepth)]).range([leftMargin, chartWidth - rightMargin - nodeWidth]);
    const unrankedY = topMargin + Math.max(1, eligible.length) * rowGap + 22;

    models.forEach((model) => {
      const rank = ranks.get(model.id);
      model.x = x(model.depth);
      model.y = rank
        ? topMargin + (rank - 1) * rowGap
        : unrankedY + unrankedPositions.get(model.id) * rowGap;
    });

    const chart = container.querySelector(".model-tree__chart");
    const svg = d3.select(chart)
      .attr("viewBox", `0 0 ${chartWidth} ${chartHeight}`)
      .attr("width", chartWidth)
      .attr("height", chartHeight);
    const nodeById = new Map(models.map((model) => [model.id, model]));
    const transition = svg.transition().duration(500).ease(d3.easeCubicOut);

    const axis = svg.selectAll(".model-tree__axis-row")
      .data(eligible, (model) => model.id)
      .join(
        (enter) => enter.append("g").attr("class", "model-tree__axis-row")
          .attr("transform", (model) => `translate(0,${model.y + nodeHeight / 2})`)
          .call((group) => group.append("line").attr("x1", leftMargin - 18).attr("x2", chartWidth - rightMargin))
          .call((group) => group.append("text").attr("x", 0).attr("y", 4).text((model) => `#${ranks.get(model.id)}`)),
        (update) => update,
        (exit) => exit.remove(),
      );
    axis.transition(transition).attr("transform", (model) => `translate(0,${model.y + nodeHeight / 2})`);
    axis.select("line").attr("x2", chartWidth - rightMargin);
    axis.select("text").text((model) => `#${ranks.get(model.id)}`);

    svg.selectAll(".model-tree__unranked-rule")
      .data([unrankedY])
      .join("line")
      .attr("class", "model-tree__unranked-rule")
      .attr("x1", leftMargin - 18)
      .attr("x2", chartWidth - rightMargin)
      .attr("y1", (value) => value + nodeHeight / 2)
      .attr("y2", (value) => value + nodeHeight / 2);
    svg.selectAll(".model-tree__unranked-label")
      .data([unrankedY])
      .join("text")
      .attr("class", "model-tree__unranked-label")
      .attr("x", 0)
      .attr("y", (value) => value + nodeHeight / 2 + 4)
      .text("N/A");

    const links = models.filter((model) => model.parent).map((model) => ({
      source: nodeById.get(model.parent),
      target: model,
    }));
    svg.selectAll(".model-tree__link")
      .data(links, (link) => link.target.id)
      .join("path")
      .attr("class", "model-tree__link")
      .transition(transition)
      .attr("d", (link) => {
        const sourceX = link.source.x + nodeWidth;
        const sourceY = link.source.y + nodeHeight / 2;
        const targetX = link.target.x;
        const targetY = link.target.y + nodeHeight / 2;
        const middle = sourceX + (targetX - sourceX) / 2;
        return `M${sourceX},${sourceY}C${middle},${sourceY} ${middle},${targetY} ${targetX},${targetY}`;
      });

    const tooltip = container.querySelector(".model-tree__tooltip") || makeTooltip(container);
    const nodes = svg.selectAll(".model-tree__node")
      .data(models, (model) => model.id)
      .join((enter) => {
        const group = enter.append("g")
          .attr("class", "model-tree__node")
          .attr("role", "link")
          .attr("tabindex", 0);
        group.append("rect").attr("rx", nodeRadius).attr("ry", nodeRadius)
          .attr("width", nodeWidth).attr("height", nodeHeight);
        group.append("text").attr("class", "model-tree__node-name").attr("x", 12).attr("y", 22);
        group.append("text").attr("class", "model-tree__node-value").attr("x", 12).attr("y", 44);
        return group;
      });

    nodes
      .attr("aria-label", (model) => `${model.name}. ${metric.label}: ${metricValue(model, metric) === null ? "not evaluated" : formatValue(metricValue(model, metric), metric)}.`)
      .on("mouseenter", (event, model) => {
        tooltip.innerHTML = tooltipContent(model, metric, ranks.get(model.id), byId);
        tooltip.hidden = false;
        positionTooltip(tooltip, event);
      })
      .on("mousemove", (event) => positionTooltip(tooltip, event))
      .on("mouseleave", () => { tooltip.hidden = true; })
      .on("click", (_, model) => { window.location.href = model.docs; })
      .on("keydown", (event, model) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          window.location.href = model.docs;
        }
      })
      .classed("is-unranked", (model) => !ranks.has(model.id));
    nodes.transition(transition).attr("transform", (model) => `translate(${model.x},${model.y})`);
    nodes.select(".model-tree__node-name").text((model) => model.short_name);
    nodes.select(".model-tree__node-value").text((model) => {
      const value = metricValue(model, metric);
      const rank = ranks.get(model.id);
      return value === null ? "Not evaluated" : `#${rank}  ${formatValue(value, metric)}`;
    });

    const eligibleCount = container.querySelector(".model-tree__eligible-count");
    eligibleCount.textContent = `${eligible.length} eligible models; ${metric.direction} is better.`;
  }

  async function initialize(container) {
    if (!window.d3) {
      container.innerHTML = "<p>The model evolution visualization could not load D3.</p>";
      return;
    }
    const response = await fetch(container.dataset.source);
    if (!response.ok) throw new Error(`Could not load model registry (${response.status}).`);
    const registry = await response.json();
    const controls = document.createElement("div");
    controls.className = "model-tree__controls";
    controls.innerHTML = `
      <label>Rank models by <select aria-label="Rank models by metric"></select></label>
      <span class="model-tree__eligible-count" aria-live="polite"></span>
    `;
    const chart = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    chart.classList.add("model-tree__chart");
    chart.setAttribute("role", "img");
    chart.setAttribute("aria-label", "Interactive model evolution tree");
    container.replaceChildren(controls, chart);
    const select = controls.querySelector("select");
    registry.metrics.forEach((metric) => {
      const option = document.createElement("option");
      option.value = metric.id;
      option.textContent = metric.label;
      option.selected = metric.id === registry.default_metric;
      select.append(option);
    });
    const draw = () => render(container, registry, select.value);
    select.addEventListener("change", draw);
    window.addEventListener("resize", draw, { passive: true });
    draw();
  }

  function boot(root = document) {
    root.querySelectorAll(selector).forEach((container) => {
      if (container.dataset.modelTreeInitialized) return;
      container.dataset.modelTreeInitialized = "true";
      initialize(container).catch((error) => {
        container.innerHTML = `<p>Unable to render the model evolution tree: ${error.message}</p>`;
      });
    });
  }

  if (typeof document$ !== "undefined") document$.subscribe(boot);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => boot());
  } else {
    boot();
  }
})();
