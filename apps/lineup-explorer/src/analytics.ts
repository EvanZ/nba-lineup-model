type GoatCounterRequest = {
  path: string;
  title: string;
};

type GoatCounterClient = {
  count: (request: GoatCounterRequest) => void;
};

declare global {
  interface Window {
    goatcounter?: Partial<GoatCounterClient>;
  }
}

let lastTrackedPath: string | null = null;
let pendingPath: string | null = null;
let waitingForClient = false;

function analyticsPath(): string {
  if (window.location.hash.startsWith("#player/")) return "/player";
  if (window.location.hash === "#about") return "/about";
  if (window.location.hash === "#rankings") return "/rankings";
  if (window.location.hash === "#lineups") return "/lineups";
  return "/lab";
}

function pageTitle(path: string): string {
  const label = path.slice(1).replace(/\b\w/g, (letter) => letter.toUpperCase());
  return `NBA GESTALT | ${label}`;
}

function flushPendingPageView() {
  if (!pendingPath) return;
  if (typeof window.goatcounter?.count !== "function") {
    const script = document.getElementById("goatcounter-script");
    if (!waitingForClient && script) {
      waitingForClient = true;
      script.addEventListener("load", () => {
        waitingForClient = false;
        flushPendingPageView();
      }, { once: true });
    }
    return;
  }

  const path = pendingPath;
  pendingPath = null;
  window.goatcounter.count({ path, title: pageTitle(path) });
}

export function trackPageView() {
  const path = analyticsPath();
  if (path === lastTrackedPath) return;

  lastTrackedPath = path;
  pendingPath = path;
  flushPendingPageView();
}
