---
last_updated: "2026-07-27"
---

# RAPM Modeling Stints

The first analytical modeling contract combines curated lineup stints with
curated possession segments. It is deliberately restricted to regular-season
games and retains only stints with positive possession exposure.

## Possession exposure

A possession contained within one lineup segment contributes one possession to
that segment. When a substitution divides a possession into multiple segments,
each segment receives an equal share:

\[
w_{segment} = \frac{1}{\text{segments in possession}}
\]

This policy conserves exactly one possession while assigning event points to
the lineup actually on the court. Home and away offensive shares are summed
separately for every stint. The model weight is their average:

\[
w_{stint} =
\frac{\text{home offensive possessions} +
      \text{away offensive possessions}}{2}
\]

The target is home-team point differential per 100 average team possessions:

\[
y_{stint} =
100 \times
\frac{\text{home points} - \text{away points}}{w_{stint}}
\]

Zero-exposure stints are counted in the manifest and excluded from fitting.
They are not silently deleted from canonical lineup data.

## Columns

Each row contains:

- season, game, period, and stint identifiers;
- schedule time and home/away team identities;
- exactly five home and five away player IDs;
- elapsed-time boundaries and duration;
- home, away, and average possession exposure;
- points, home margin, and the net-rating target;
- game-quality, processing-code, build, and source-hash provenance.

## Storage

```text
data/analytical/rapm_stints/{season}/regular/
  _manifest.json
  part-00000.parquet
```

The manifest records exact curated input manifests, source counts, included
and excluded stints, possession totals, player count, modeling code
fingerprint, and output file integrity.
