/** Keep obsolete page responses out of the UI without pinning a data release. */
const rawFetch = globalThis.fetch.bind(globalThis);
let epoch = 0;
let page = "";

export function beginSurface(key: string) {
  if (key === page) return;
  page = key;
  epoch += 1;
}

export function resetSurface() {
  epoch += 1;
}

export async function releaseFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  if (!url.startsWith("/api/")) return rawFetch(input, init);
  const requestEpoch = epoch;
  const response = await rawFetch(input, { ...init, cache: "no-store" });
  if (requestEpoch !== epoch) throw new DOMException("Obsolete page response", "AbortError");
  const readJSON = response.json.bind(response);
  response.json = async () => {
    const value = await readJSON();
    if (requestEpoch !== epoch) throw new DOMException("Obsolete page data", "AbortError");
    return value;
  };
  return response;
}
