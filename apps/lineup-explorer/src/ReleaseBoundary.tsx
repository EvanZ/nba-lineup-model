import { useEffect, useState } from "react";
import App from "./App";
import { beginSurface } from "./releaseFetch";

export function ReleaseBoundary() {
  const [surface, setSurface] = useState(() => ({ route: window.location.hash, revision: 0 }));
  beginSurface(`${surface.route}:${surface.revision}`);
  useEffect(() => {
    const navigate = () => setSurface((value) => ({ route: window.location.hash, revision: value.revision + 1 }));
    window.addEventListener("hashchange", navigate);
    window.addEventListener("gestalt-release-changed", navigate);
    return () => {
      window.removeEventListener("hashchange", navigate);
      window.removeEventListener("gestalt-release-changed", navigate);
    };
  }, []);
  return <App key={`${surface.route}:${surface.revision}`} />;
}
