// EC2-only browser QA against the exact deployed files and API.
// The loopback preview does not alter nginx authentication or expose a new port.
import { createServer } from "node:http";
import { readFile, mkdir, writeFile, realpath } from "node:fs/promises";
import { resolve, extname, join } from "node:path";
import { pathToFileURL } from "node:url";
import assert from "node:assert/strict";

const [frontendArg, api, output, playwrightPath] = process.argv.slice(2);
const frontend = await realpath(frontendArg);
const { chromium } = await import(pathToFileURL(playwrightPath).href);
await mkdir(output, { recursive: true });
const mime = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png", ".woff": "font/woff", ".woff2": "font/woff2" };
const server = createServer(async (req, res) => {
  try {
    if (req.url.startsWith("/api/")) {
      const response = await fetch(api + req.url, { headers: { "X-Gestalt-Release": req.headers["x-gestalt-release"] || "" } });
      res.writeHead(response.status, Object.fromEntries([...response.headers].filter(([key]) => !["content-encoding", "transfer-encoding", "content-length"].includes(key))));
      res.end(Buffer.from(await response.arrayBuffer()));
    } else {
      const relative = new URL(req.url, "http://localhost").pathname;
      const path = resolve(frontend, "." + (relative === "/" ? "/index.html" : relative));
      if (!path.startsWith(frontend + "/")) { res.writeHead(403); res.end(); return; }
      res.writeHead(200, { "Content-Type": mime[extname(path)] || "application/octet-stream" });
      res.end(await readFile(path));
    }
  } catch (error) { res.writeHead(500); res.end(String(error)); }
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const address = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (error) => errors.push(String(error)));
const failedRequests = [];
page.on("response", (response) => {
  if (response.status() >= 400 && response.url().includes("/api/")) {
    failedRequests.push({ url: response.url(), status: response.status() });
  }
});
const report = { frontend, api, views: [], errors, failedRequests };
try {
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    for (const [name, route, selector] of [
      ["schedule", "#schedule/2026-11-02", ".schedule-result-badges"],
      ["player", "#player/201939", ".current-season-game-log-table tbody tr, .current-season-game-log-card"],
      ["rankings", "#rankings", ".rankings-table tbody tr, .mobile-player-rankings li"],
      ["team", "#team/GSW?season=2026-27", ".team-roster-table tbody tr, .team-roster-mobile-list li"],
      ["lineups", "#lineups", ".rankings-table tbody tr, .mobile-lineup-card"],
      ["wins", "#wins", ".win-forecast-table tbody tr"],
    ]) {
      await page.goto(address + "/" + route);
      try {
        await page.locator(selector).first().waitFor({ state: "visible", timeout: 60000 });
      } catch (error) {
        await page.screenshot({ path: join(output, `failed-${name}-${viewport.width}.png`) });
        await writeFile(join(output, `failed-${name}-${viewport.width}.txt`), await page.locator("body").innerText());
        throw error;
      }
      if (name === "schedule") {
        await page.locator("details.schedule-assumptions").first().evaluate((el) => { el.open = true; });
      }
      await page.evaluate(() => document.fonts.ready);
      const layout = await page.evaluate(() => ({
        viewport: innerWidth, document: document.documentElement.scrollWidth,
        title: document.querySelector("h1")?.textContent,
        games: document.querySelectorAll(".schedule-game-card").length,
        badges: document.querySelectorAll(".schedule-result-badges").length,
        gameLogRows: document.querySelectorAll(".current-season-game-log-table tbody tr").length,
        errors: [...document.querySelectorAll(".error")].map((el) => el.textContent).filter(Boolean),
      }));
      report.views.push({ name, viewport, ...layout });
      await page.screenshot({ path: join(output, `${name}-${viewport.width}.png`), fullPage: false });
      assert.ok(layout.document <= viewport.width + 1, `${name}: page overflows at ${viewport.width}`);
      assert.deepEqual(layout.errors, [], `${name}: visible API error`);
    }
  }
  assert.deepEqual(errors, []);
  assert.deepEqual(failedRequests, []);
} finally {
  await writeFile(join(output, "browser-evidence.json"), JSON.stringify(report, null, 2));
  await browser.close();
  server.close();
}
console.log(JSON.stringify(report, null, 2));
