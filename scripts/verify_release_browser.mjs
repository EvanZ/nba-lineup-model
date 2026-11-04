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
const chromiumArgs = ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"];
let browser = await chromium.launch({ headless: true, args: chromiumArgs });
let page = await browser.newPage();
page.setDefaultTimeout(90000);

async function capturePage(path, fullPage) {
  const session = await page.context().newCDPSession(page);
  try {
    const metrics = await session.send("Page.getLayoutMetrics");
    const viewport = page.viewportSize();
    const size = fullPage ? metrics.cssContentSize : viewport;
    if (!size) throw new Error("Chromium did not report screenshot dimensions");
    const result = await session.send("Page.captureScreenshot", {
      format: "png",
      fromSurface: true,
      captureBeyondViewport: fullPage,
      clip: {
        x: 0,
        y: 0,
        width: Math.ceil(size.width),
        height: Math.ceil(size.height),
        scale: 1,
      },
    });
    writeFileSync(path, Buffer.from(result.data, "base64"));
    return "chromium-cdp";
  } finally {
    await session.detach();
  }
}

const officialLogoCache = new Map();
const logoProxyErrors = [];
await page.route("https://cdn.nba.com/logos/nba/**", async (route) => {
  const url = route.request().url();
  try {
    let body = officialLogoCache.get(url);
    if (!body) {
      const response = await fetch(url);
      if (!response.ok) throw new Error("NBA logo HTTP " + response.status);
      body = Buffer.from(await response.arrayBuffer());
      officialLogoCache.set(url, body);
    }
    await route.fulfill({ status: 200, contentType: "image/svg+xml", body });
  } catch (error) {
    logoProxyErrors.push({ url, error: String(error) });
    await route.abort("failed");
  }
});
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

async function replacePage() {
  await browser.close();
  browser = await chromium.launch({ headless: true, args: chromiumArgs });
  page = await browser.newPage();
  page.setDefaultTimeout(90000);
  await page.route("https://cdn.nba.com/logos/nba/**", async (route) => {
    const url = route.request().url();
    try {
      let body = officialLogoCache.get(url);
      if (!body) {
        const response = await fetch(url);
        if (!response.ok) throw new Error("NBA logo HTTP " + response.status);
        body = Buffer.from(await response.arrayBuffer());
        officialLogoCache.set(url, body);
      }
      await route.fulfill({ status: 200, contentType: "image/svg+xml", body });
    } catch (error) {
      logoProxyErrors.push({ url, error: String(error) });
      await route.abort("failed");
    }
  });
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
}

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
    const logoHandles = await page.locator('img[src*="cdn.nba.com/logos/nba/"]').elementHandles();
    for (const logo of logoHandles) {
      if (await logo.evaluate((image) => image.getClientRects().length > 0)) {
        await logo.scrollIntoViewIfNeeded();
      }
    }
    await page.waitForFunction(() => {
      const images = Array.from(document.querySelectorAll('img[src*="cdn.nba.com/logos/nba/"]'));
      const visible = images.filter((image) => image.getClientRects().length > 0);
      return visible.every((image) => image.complete && image.naturalWidth > 0);
    });
    await page.evaluate(() => window.scrollTo(0, 0));
    const view = await page.evaluate(({ name, expectedTitle, viewport }) => {
      const images = Array.from(document.querySelectorAll('img[src*="cdn.nba.com/logos/nba/"]'));
      const visibleImages = images.filter((image) => image.getClientRects().length > 0);
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
        visibleNbaLogoImages: visibleImages.length,
        loadedVisibleNbaLogoImages: visibleImages.filter((image) => image.complete && image.naturalWidth > 0).length,
        alerts: Array.from(document.querySelectorAll('[role="alert"]')).map((node) => node.textContent?.trim()),
      };
    }, { name: item.name, expectedTitle: item.title, viewport: width });
    view.captureMethod = await capturePage(
      outputRoot + "/" + item.name + "-" + width + ".png",
      item.name === "schedule" || item.name === "player",
    );
    views.push(view);
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
    if (view.visibleNbaLogoImages > 0 && view.loadedVisibleNbaLogoImages !== view.visibleNbaLogoImages) {
      functionalFailures.push(item.name + " visible NBA logos failed to render at " + width);
    }
    if (view.alerts.filter(Boolean).length) {
      functionalFailures.push(item.name + " rendered an alert at " + width + ": " + view.alerts.join("; "));
    }
    if (view.documentWidth > width) {
      layoutFindings.push(item.name + " overflows at " + width + " (" + view.documentWidth + "px)");
    }
    await replacePage();
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
const transitionCaptureMethod = await capturePage(outputRoot + "/activation-transition.png", true);

const transition = {
  traffic,
  oldRelease,
  newRelease,
  sawConflict: traffic.some((item) => item.status === 409 && item.requestRelease === oldRelease && item.responseRelease === newRelease),
  finalReleaseRequests: traffic.filter((item) => item.status === 200 && item.requestRelease === newRelease).length,
  captureMethod: transitionCaptureMethod,
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
const expectedConflictConsoleErrors = consoleErrors.filter((message) => message.includes("status of 409 (Conflict)"));
const unexpectedConsoleErrors = consoleErrors.filter((message) => !message.includes("status of 409 (Conflict)"));
if (unexpectedConsoleErrors.length) functionalFailures.push("browser console errors: " + unexpectedConsoleErrors.join("; "));
if (logoProxyErrors.length) functionalFailures.push("official NBA logo proxy errors");
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
  expectedConflictConsoleErrors,
  unexpectedConsoleErrors,
  logoProxyErrors,
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
