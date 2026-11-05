// EC2-only browser QA against the exact deployed files and API.
// The loopback preview does not alter nginx authentication or expose a new port.
import { createServer } from "node:http";
import { readFile, mkdir, writeFile, realpath } from "node:fs/promises";
import { resolve, extname, join } from "node:path";
import { pathToFileURL } from "node:url";
import assert from "node:assert/strict";

const [frontendArg, api, output, playwrightPath, baselinePath] = process.argv.slice(2);
assert.equal(process.env.GESTALT_QA_RESOURCE_LIMITED, "1", "Run this check in the documented resource-limited systemd unit.");
const scheduleDate = process.env.GESTALT_QA_GAME_DATE || "2026-11-02";
assert.match(scheduleDate, /^\d{4}-\d{2}-\d{2}$/);
const baseline = baselinePath ? JSON.parse(await readFile(baselinePath, "utf8")) : null;
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
let page;
const errors = [];
const failedRequests = [];
const layoutFailures = [];
const report = { frontend, api, baselinePath, views: [], errors, failedRequests, layoutFailures, layoutRegressions: [] };
try {
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    for (const [name, route, selector] of [
      ["schedule", `#schedule/${scheduleDate}`, ".schedule-game-card"],
      ["player", "#player/201939", ".current-season-game-log-table tbody tr, .current-season-game-log-card"],
      ["rankings", "#rankings", ".rankings-table tbody tr, .mobile-player-rankings li"],
      ["team", "#team/GSW?season=2026-27", ".team-roster-table tbody tr, .team-roster-mobile-list li"],
      ["lineups", "#lineups", ".rankings-table tbody tr, .mobile-lineup-card"],
      ["wins", "#wins", ".win-forecast-table tbody tr"],
    ]) {
      console.log(`Checking ${name} at ${viewport.width}px`);
      page = await browser.newPage({ viewport });
      page.on("pageerror", (error) => errors.push(String(error)));
      page.on("crash", () => errors.push(`Browser page crashed: ${name} ${viewport.width}`));
      page.on("response", (response) => {
        if (response.status() >= 400 && response.url().includes("/api/"))
          failedRequests.push({ url: response.url(), status: response.status() });
      });
      await page.goto(address + "/" + route, { waitUntil: "domcontentloaded" });
      try {
        await page.locator(selector).first().waitFor({ state: "visible", timeout: 60000 });
      } catch (error) {
        await writeFile(join(output, `failed-${name}-${viewport.width}.txt`), await page.locator("body").innerText({ timeout: 5000 }).catch(String));
        await page.screenshot({ path: join(output, `failed-${name}-${viewport.width}.png`), timeout: 10000 }).catch((error) => errors.push(String(error)));
        throw error;
      }
      if (name === "schedule") {
        await page.locator("details.schedule-assumptions").first().evaluate((el) => { el.open = true; });
        const links = await page.locator('a[href*="/referee/injury/"]').evaluateAll(nodes => nodes.map(n => n.href));
        report.injuryReportLinks = links;
        if (process.env.GESTALT_QA_EXPECT_REPORT_DATE) {
          assert.ok(links.length > 0, "No rendered official injury report links");
          assert.ok(links.every(link => link.includes(`Injury-Report_${process.env.GESTALT_QA_EXPECT_REPORT_DATE}_`)), "Rendered injury links use another report date");
        }
      }
      await page.evaluate(() => Promise.race([document.fonts.ready, new Promise((resolve) => setTimeout(resolve, 10000))]));
      await page.addStyleTag({ content: "html, body, * { scroll-behavior: auto !important; }" });
      const layout = await page.evaluate(() => ({
        viewport: innerWidth, document: document.documentElement.scrollWidth,
        title: document.querySelector("h1")?.textContent,
        games: document.querySelectorAll(".schedule-game-card").length,
        badges: document.querySelectorAll(".schedule-result-badges").length,
        gameLogRows: document.querySelectorAll(".current-season-game-log-table tbody tr").length,
        gameLogDates: [...document.querySelectorAll(".current-season-game-log-date time")].map((el) => el.getAttribute("datetime")),
        trajectoryDates: [...document.querySelectorAll(".current-rating-date")].map((el) => el.textContent),
        errors: [...document.querySelectorAll(".error")].map((el) => el.textContent).filter(Boolean),
      }));
      report.views.push({ name, viewport, ...layout });
      await page.screenshot({ path: join(output, `${name}-${viewport.width}.png`), fullPage: false });
      if (layout.document > viewport.width + 1) layoutFailures.push(`${name}: page overflows at ${viewport.width} (${layout.document}px)`);
      assert.deepEqual(layout.errors, [], `${name}: visible API error`);
      const detailSelectors = name === "player"
        ? [".current-rating-trajectory", ".current-season-game-log-table-wrap"]
        : name === "schedule" ? [".schedule-spread-ledger"] : [];
      for (const [index, detail] of detailSelectors.entries()) {
        const section = page.locator(detail).first();
        await section.evaluate((el) => window.scrollTo({ top: window.scrollY + el.getBoundingClientRect().top - 20, behavior: "instant" }));
        await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await page.screenshot({ path: join(output, `${name}-detail-${index}-${viewport.width}.png`) });
      }
      await page.close();
    }
  }
  assert.deepEqual(errors, []);
  assert.deepEqual(failedRequests, []);
  if (baseline) {
    // Keep existing defects visible; never exempt a page or permit worse overflow.
    for (const view of report.views) {
      const previous = baseline.views.find((entry) => entry.name === view.name && entry.viewport === view.viewport);
      assert.ok(previous, `Missing baseline view: ${view.name} ${view.viewport}`);
      if (view.document > Math.max(view.viewport, previous.document) + 1)
        report.layoutRegressions.push(`${view.name}: ${previous.document}px → ${view.document}px`);
      assert.deepEqual(view.gameLogDates, previous.gameLogDates, `${view.name}: changed game-log dates`);
      assert.deepEqual(view.trajectoryDates, previous.trajectoryDates, `${view.name}: changed trajectory points`);
      assert.equal(view.games, previous.games, `${view.name}: changed game coverage`);
      assert.equal(view.badges, previous.badges, `${view.name}: changed result-badge coverage`);
    }
    assert.deepEqual(report.layoutRegressions, []);
  } else {
    assert.deepEqual(layoutFailures, []);
  }
} finally {
  await writeFile(join(output, "browser-evidence.json"), JSON.stringify(report, null, 2));
  await browser.close();
  server.close();
}
console.log(JSON.stringify(report, null, 2));
