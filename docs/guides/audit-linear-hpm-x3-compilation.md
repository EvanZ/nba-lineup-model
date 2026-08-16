---
last_updated: "2026-08-15"
---

# Audit NAIL-RAPM Compilation

For NAIL-RAPM v1.0, verify that additive context coordinates compile exactly
into player-level adjustments while the remaining non-additive lineup
coordinates stay in the context term:

```bash
uv run nba-audit-linear-hpm-x3-compilation
```

The audit checks every frozen 2023-24 through 2025-26 regular-season and
playoff possession. It writes an identity summary and target-season compiled
player adjustments under `artifacts/models/analysis/linear_hpm_x3_compilation_audit/`.
