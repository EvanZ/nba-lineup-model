# Build the Exposure-Gated Cold-Start Prior

Build the immutable first-year prior after the draft-rate study, exposure gate,
and a replacement-token study through the immediately preceding season exist:

```bash
uv run nba-build-exposure-gated-cold-start-prior --season 2025-26
```

The command resolves the latest compatible artifacts by default, validates that
all training stops at 2024-25, and writes the continuous blend plus revised
rookie rankings under:

```text
artifacts/models/exposure_gated_cold_start/<season>/<run_id>/
```

Evaluate the frozen vector on regular season and playoffs separately:

```bash
uv run nba-evaluate-frozen-exposure-gated-cold-start-prior --season 2025-26
```

Pass `--draft-run-id`, `--exposure-run-id`, or `--replacement-run-id` to pin
the blend inputs. Pass `--exposure-gated-run-id` or `--reference-prior-run-id`
to pin the frozen evaluation inputs.

See [Exposure-Gated Cold-Start Prior](../models/exposure-gated-cold-start.md)
for the formula, rankings, and evaluation result.
