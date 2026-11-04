/** A pin lasts for one mounted page/surface, never for the browser session. */
const rawFetch = globalThis.fetch.bind(globalThis);
let epoch = 0;
let page = "";
let release: Promise<string | null> | null = null;

export function beginSurface(key: string) {
  if (key === page && release) return;
  page = key;
  epoch += 1;
  release = null;
}

export function resetSurface() {
  epoch += 1;
  release = null;
}

function changed() {
  resetSurface();
  window.dispatchEvent(new Event("gestalt-release-changed"));
  return new DOMException("The release changed; refreshing this page together.", "AbortError");
}

export async function releaseFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  if (!url.startsWith("/api/")) return rawFetch(input, init);
  const requestEpoch = epoch;
  release ??= rawFetch("/api/serving-view", { cache: "no-store" }).then(async (response) => {
    if (!response.ok) throw new Error("Cannot verify the selected release.");
    return (await response.json()).release_id as string | null;
  });
  const expected = await release;
  if (requestEpoch !== epoch) throw new DOMException("Obsolete page request", "AbortError");
  const headers = new Headers(init?.headers);
  if (expected) headers.set("X-Gestalt-Release", expected);
  const response = await rawFetch(input, { ...init, headers, cache: "no-store" });
  if (requestEpoch !== epoch) throw new DOMException("Obsolete page response", "AbortError");
  if (response.status === 409 || (expected && response.headers.get("X-Gestalt-Release") !== expected)) {
    throw changed();
  }
  // Parsing a large response can finish after a page or edition switch too.
  const readJSON = response.json.bind(response);
  response.json = async () => {
    const value = await readJSON();
    if (requestEpoch !== epoch) throw new DOMException("Obsolete page data", "AbortError");
    return value;
  };
  return response;
}
