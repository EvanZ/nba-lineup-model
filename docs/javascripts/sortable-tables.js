(() => {
  function value(cell) {
    const text = cell.textContent.trim().replaceAll(",", "");
    const numeric = Number(text);
    return Number.isFinite(numeric) ? numeric : text.toLocaleLowerCase();
  }

  function sortTable(table, column, direction) {
    const body = table.tBodies[0];
    const header = table.tHead.rows[0].cells[column].textContent.trim();
    const isPickColumn = header.toLocaleLowerCase() === "pick";
    const rows = [...body.rows].map((row, index) => ({ row, index }));
    rows.sort((left, right) => {
      const leftText = left.row.cells[column].textContent.trim();
      const rightText = right.row.cells[column].textContent.trim();
      const leftIsUndrafted = isPickColumn && leftText === "-";
      const rightIsUndrafted = isPickColumn && rightText === "-";

      if (leftIsUndrafted !== rightIsUndrafted) {
        return leftIsUndrafted ? 1 : -1;
      }

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

  function nearestHeading(table) {
    const content = table.closest(".md-typeset") ?? document;
    let latest;
    content.querySelectorAll("h1, h2, h3, h4").forEach((heading) => {
      if (heading.compareDocumentPosition(table) & Node.DOCUMENT_POSITION_FOLLOWING) {
        latest = heading;
      }
    });
    return latest?.textContent.trim() ?? "";
  }

  function isRankingTable(table) {
    const headerRow = table.tHead?.rows[0];
    if (!headerRow || !table.tBodies[0]) return false;

    const headers = [...headerRow.cells].map((header) => header.textContent.trim());
    return headers.some((header) => /\brank\b/i.test(header))
      || /\brankings?\b/i.test(nearestHeading(table));
  }

  function enhance(table) {
    const headerRow = table.tHead?.rows[0];
    if (!headerRow || !table.tBodies[0] || table.dataset.sortableInitialized) return;

    table.dataset.sortableInitialized = "true";
    table.classList.add("sortable-ranking-table");
    [...headerRow.cells].forEach((header, column) => {
      const label = header.textContent.trim();
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.title = `Sort by ${label}`;
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
    root.querySelectorAll("table").forEach((table) => {
      if (isRankingTable(table)) enhance(table);
    });
  }

  if (typeof document$ !== "undefined") document$.subscribe(initialize);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initialize());
  } else {
    initialize();
  }
})();
