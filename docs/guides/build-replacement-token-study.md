# Build the Replacement-Token Study

This retrospective diagnostic replaces every cataloged low-exposure player in
a completed season with one shared lineup token, then fits that season's RAPM
with its canonical selected lambda.

```bash
uv run nba-build-replacement-token-study --through-season 2025-26
```

The default definition is a realized regular-season team-possession share below
5%. It is intentionally retrospective and must not be used as a preseason
player classification.

Run an explicit cutoff sensitivity analysis with a different definition:

```bash
uv run nba-build-replacement-token-study \
  --through-season 2025-26 \
  --replacement-share-cutoff 0.02
```

## Outputs

Each run is immutable:

```text
artifacts/models/replacement_token/<through-season>/<run_id>/
```

| File | Purpose |
| --- | --- |
| `season_replacement_token_coefficients.parquet` | Per-season pooled replacement effect and separately ridged comparison |
| `replacement_token_summary.json` | Season-balanced estimate and uncertainty interval |
| `replacement-token-by-season.svg` | Documentation chart |
| `metadata.json` / `manifest.json` | Reproducibility and integrity records |

The chart is copied to
`docs/assets/images/replacement-token/replacement-token-by-season.svg`. See
[Pooled Replacement-Token RAPM](../models/replacement-token.md) for the model
contract and interpretation.
