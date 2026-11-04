import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "katex/dist/katex.min.css";

import { ReleaseBoundary } from "./ReleaseBoundary";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ReleaseBoundary />
  </StrictMode>,
);
