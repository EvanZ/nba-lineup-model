(() => {
  function moveLastUpdated(root = document) {
    const updated = root.querySelector(".page-last-updated");
    const title = root.querySelector(".md-content__inner > h1");
    if (!updated || !title || updated.dataset.positioned) return;

    updated.dataset.positioned = "true";
    title.insertAdjacentElement("afterend", updated);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => moveLastUpdated());
  } else {
    moveLastUpdated();
  }
  document.addEventListener("zensical:content-replaced", (event) => {
    moveLastUpdated(event.target);
  });
})();
