---
last_updated: "2026-08-08"
---

# Build Context Function Audit

The Context Function Audit evaluates saved Forward Portable-Matchup Contextual
RAPM spline states. It does not ingest data or retrain a RAPM model.

```bash
uv run --group docs nba-build-context-function-audit
```

The command reads the latest 2025-26 portable-matchup artifact, writes
response-curve and feature-summary parquet files beneath
`artifacts/analysis/context_function_audit/`, renders two SVG response atlases,
and refreshes [Context Function Audit](../models/context-function-audit.md).

Use it before proposing a new contextual function family. It is a screening
report: final choices still require a frozen predictive refit and evaluation.
