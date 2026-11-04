import { createReadStream, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, request as httpRequest } from "node:http";
import { extname, resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";

function required(name) {
  const value = process.env[name];
  if (!value) throw new Error("Missing " + name);
  return value;
}

const webRoot = resolve(required("GESTALT_BROWSER_WEB"));
const outputRoot = resolve(required("GESTALT_BROWSER_OUTPUT"));
const oldApi = new URL(required("GESTALT_BROWSER_OLD_API"));
const newApi = new URL(required("GESTALT_BROWSER_NEW_API"));
const oldRelease = required("GESTALT_BROWSER_OLD_RELEASE");
const newRelease = required("GESTALT_BROWSER_NEW_RELEASE");
const playwrightModule = required("GESTALT_PLAYWRIGHT_MODULE");
mkdirSync(outputRoot, { recursive: true });

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

let mode = "steady";
let backend = newApi;
let traffic = [];

function proxy(request, response) {
  const selected = backend;
  const headers = { ...request.headers, host: selected.host };
  const upstream = httpRequest({
    hostname: selected.hostname,
    port: selected.port,
    method: request.method,
    path: request.url,
    headers,
  }, (incoming) => {
    response.writeHead(incoming.statusCode ?? 502, incoming.headers);
    incoming.pipe(response);
    incoming.on("end", () => {
      const entry = {
        path: request.url,
        backend: selected.origin,
        requestRelease: request.headers["x-gestalt-release"] ?? null,
        status: incoming.statusCode ?? null,
        responseRelease: incoming.headers["x-gestalt-release"] ?? null,
      };
      traffic.push(entry);
      if (
        mode === "transition"
        && selected.origin === oldApi.origin
        && request.url.startsWith("/api/serving-view")
      ) {
        backend = newApi;
      }
    });
  });
  upstream.on("error", (error) => {
    response.writeHead(502, { "content-type": "application/json" });
    response.end(JSON.stringify({ error: error.message }));
  });
  request.pipe(upstream);
}

const server = createServer((request, response) => {
  const url = new URL(request.url ?? "/", "http://127.0.0.1");
  if (url.pathname.startsWith("/api/")) {
    proxy(request, response);
    return;
  }
  const relative = decodeURIComponent(url.pathname).replace(/^\/+/, "") || "index.html";
  const path = resolve(webRoot, relative);
  if (path !== webRoot && !path.startsWith(webRoot + sep)) {
    response.writeHead(403);
    response.end("forbidden");
    return;
  }
  const selected = existsSync(path) ? path : resolve(webRoot, "index.html");
  response.writeHead(200, {
    "cache-control": "no-store",
    "content-type": mimeTypes[extname(selected)] ?? "application/octet-stream",
  });
  createReadStream(selected).pipe(response);
});
await new Promise((resolveReady) => server.listen(0, "127.0.0.1", resolveReady));
const address = server.address();
const baseUrl = "http://127.0.0.1:" + address.port;

const { chromium } = await import(pathToFileURL(playwrightModule).href);
const browser = await chromium.launch({ headless: true, args: ["--no-sandbox"] });
const page = await browser.newPage();
page.setDefaultTimeout(90000);
const consoleErrors = [];
const pageErrors = [];
const failedRequests = [];
const responseReleases = [];
const requestedUrls = [];
page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
});
page.on("pageerror", (error) => pageErrors.push(error.message));
page.on("request", (request) => requestedUrls.push(request.url()));
page.on("requestfailed", (request) => {
  failedRequests.push({ url: request.url(), error: request.failure()?.errorText ?? "unknown" });
});
page.on("response", (response) => {
  if (response.url().includes("/api/")) {
    responseReleases.push({
      url: response.url(),
      status: response.status(),
      release: response.headers()["x-gestalt-release"] ?? null,
    });
  }
});

const pages = [
  { name: "schedule", hash: "#schedule/2026-11-02", title: "The slate." },
  { name: "player", hash: "#player/201939", title: "Stephen Curry" },
  { name: "rankings", hash: "#rankings", title: "Player rankings." },
  { name: "team", hash: "#team/GSW?season=2026-27", title: "Golden State Warriors." },
  { name: "lineups", hash: "#lineups", title: "Lineup contexts." },
  { name: "wins", hash: "#wins", title: "Win projections." },
];
const viewports = [1440, 768, 390];
const views = [];
const functionalFailures = [];
const layoutFindings = [];

for (const width of viewports) {
  await page.setViewportSize({ width, height: 900 });
  for (const item of pages) {
    await page.goto(baseUrl + "/" + item.hash, { waitUntil: "domcontentloaded" });
    await page.waitForFunction((expected) => {
      const heading = document.querySelector("h1")?.textContent?.trim();
      return heading === expected;
    }, item.title);
    if (item.name === "schedule") {
      await page.waitForFunction(() => document.querySelectorAll(".schedule-game-card").length === 15);
    }
    if (item.name === "player") {
      await page.waitForFunction(() => Boolean(document.querySelector('time[datetime="2026-11-02"]')));
    }
    await page.waitForTimeout(600);
    const view = await page.evaluate(({ name, expectedTitle, viewport }) => {
      const images = Array.from(document.querySelectorAll('img[src*="cdn.nba.com/logos/nba/"]'));
      return {
        name,
        viewport,
        documentWidth: document.documentElement.scrollWidth,
        title: document.querySelector("h1")?.textContent?.trim() ?? "",
        expectedTitle,
        games: document.querySelectorAll(".schedule-game-card").length,
        badges: document.querySelectorAll(".schedule-result-badges").length,
        ledgers: document.querySelectorAll(".schedule-spread-ledger").length,
        actualLedgers: document.querySelectorAll(".schedule-spread-ledger.has-actual").length,
        gameLogRows: document.querySelectorAll(".current-season-game-log-table tbody tr").length,
        nov2Rows: document.querySelectorAll('time[datetime="2026-11-02"]').length,
        trajectoryLogos: document.querySelectorAll(".current-rating-opponent-logo").length,
        nbaLogoImages: images.length,
        loadedNbaLogoImages: images.filter((image) => image.complete && image.naturalWidth > 0).length,
        alerts: Array.from(document.querySelectorAll('[role="alert"]')).map((node) => node.textContent?.trim()),
      };
    }, { name: item.name, expectedTitle: item.title, viewport: width });
    views.push(view);
    await page.screenshot({
      path: outputRoot + "/" + item.name + "-" + width + ".png",
      fullPage: true,
    });
    if (view.title !== item.title) functionalFailures.push(item.name + " title mismatch at " + width);
    if (item.name === "schedule") {
      if (view.games !== 15) functionalFailures.push("schedule game count " + view.games + " at " + width);
      if (view.badges !== 15) functionalFailures.push("schedule badge count " + view.badges + " at " + width);
      if (view.ledgers !== 15 || view.actualLedgers !== 15) {
        functionalFailures.push("schedule ledger coverage " + view.ledgers + "/" + view.actualLedgers + " at " + width);
      }
    }
    if (item.name === "player") {
      if (view.nov2Rows !== 1) functionalFailures.push("Curry Nov 2 row missing at " + width);
      if (view.gameLogRows < 8) functionalFailures.push("Curry game-log rows " + view.gameLogRows + " at " + width);
      if (view.trajectoryLogos < 8) functionalFailures.push("Curry trajectory logos " + view.trajectoryLogos + " at " + width);
    }
    if (view.nbaLogoImages > 0 && view.loadedNbaLogoImages === 0) {
      functionalFailures.push(item.name + " NBA logos failed to render at " + width);
    }
    if (view.alerts.filter(Boolean).length) {
      functionalFailures.push(item.name + " rendered an alert at " + width + ": " + view.alerts.join("; "));
    }
    if (view.documentWidth > width) {
      layoutFindings.push(item.name + " overflows at " + width + " (" + view.documentWidth + "px)");
    }
  }
}

mode = "transition";
backend = oldApi;
traffic = [];
responseReleases.length = 0;
await page.setViewportSize({ width: 1440, height: 900 });
await page.goto(baseUrl + "/#schedule/2026-11-02", { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => document.querySelectorAll(".schedule-game-card").length === 15);
await page.waitForTimeout(1000);
await page.screenshot({ path: outputRoot + "/activation-transition.png", fullPage: true });

const transition = {
  traffic,
  oldRelease,
  newRelease,
  sawConflict: traffic.some((item) => item.status === 409 && item.requestRelease === oldRelease && item.responseRelease === newRelease),
  finalReleaseRequests: traffic.filter((item) => item.status === 200 && item.requestRelease === newRelease).length,
  mismatchedSuccesses: traffic.filter((item) => (
    item.status === 200
    && item.requestRelease
    && item.responseRelease
    && item.requestRelease !== item.responseRelease
  )),
};
if (!transition.sawConflict) functionalFailures.push("browser transition did not observe the fenced 409");
if (!transition.finalReleaseRequests) functionalFailures.push("browser transition did not recover onto the new release");
if (transition.mismatchedSuccesses.length) functionalFailures.push("browser accepted a mixed-release 200 response");

const espnRequests = requestedUrls.filter((url) => url.toLowerCase().includes("espn"));
if (espnRequests.length) functionalFailures.push("ESPN request observed");
if (consoleErrors.length) functionalFailures.push("browser console errors: " + consoleErrors.join("; "));
if (pageErrors.length) functionalFailures.push("browser page errors: " + pageErrors.join("; "));

const evidence = {
  webRoot,
  oldApi: oldApi.origin,
  newApi: newApi.origin,
  oldRelease,
  newRelease,
  views,
  transition,
  responseReleases,
  consoleErrors,
  pageErrors,
  failedRequests,
  espnRequests,
  layoutFindings: Array.from(new Set(layoutFindings)),
  functionalFailures,
};
writeFileSync(outputRoot + "/browser-evidence.json", JSON.stringify(evidence, null, 2) + "\n");
await browser.close();
await new Promise((resolveClosed) => server.close(resolveClosed));
console.log(JSON.stringify({
  browserEvidence: outputRoot + "/browser-evidence.json",
  views: views.length,
  functionalFailures: functionalFailures.length,
  layoutFindings: evidence.layoutFindings.length,
}));
if (functionalFailures.length) process.exitCode = 1;
