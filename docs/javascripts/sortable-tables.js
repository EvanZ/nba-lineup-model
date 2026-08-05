(() => {
  function value(cell) {
    const text = cell.textContent.trim().replaceAll(",", "");
    const numeric = Number(text);
    return Number.isFinite(numeric) ? numeric : text.toLocaleLowerCase();
  }

  function sortTable(table, column, direction) {
    const body = table.tBodies[0];
    const rows = [...body.rows].map((row, index) => ({ row, index }));
    rows.sort((left, right) => {
      const leftValue = value(left.row.cells[column]);
      const rightValue = value(right.row.cells[column]);
      if (leftValue === rightValue) return left.index - right.index;
      if (typeof leftValue === "number" && typeof rightValue === "number") {
        return direction * (leftValue - rightValue);
      }
      return direction * String(leftValue).localeCompare(String(rightValue));
    });
    rows.forEach(({ row }) => body.append(row));
  }

  function enhance(table) {
    if (table.dataset.sortableInitialized) return;
    table.dataset.sortableInitialized = "true";
    table.classList.add("sortable-ranking-table");
    [...table.tHead.rows[0].cells].forEach((header, column) => {
      const label = header.textContent.trim();
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.setAttribute("aria-label", `Sort by ${label}`);
      button.setAttribute("aria-sort", "none");
      button.addEventListener("click", () => {
        const ascending = button.getAttribute("aria-sort") !== "ascending";
        [...table.tHead.querySelectorAll("button")].forEach((other) => {
          other.setAttribute("aria-sort", "none");
        });
        button.setAttribute("aria-sort", ascending ? "ascending" : "descending");
        sortTable(table, column, ascending ? 1 : -1);
      });
      header.replaceChildren(button);
    });
  }

  function initialize(root = document) {
    root.querySelectorAll("h3").forEach((heading) => {
      if (!heading.textContent.trim().startsWith("Top 25 ")) return;
      const table = heading.nextElementSibling;
      if (table?.tagName === "TABLE") enhance(table);
    });
  }

  if (typeof document$ !== "undefined") document$.subscribe(initialize);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initialize());
  } else {
    initialize();
  }
})();
