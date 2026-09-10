#!/usr/bin/env bash
set -euo pipefail

# Historical Stats V3 provides the listed player rows; Summary V2 supplies the
# explicitly inactive roster members needed for the binary availability target.
uv run nba-fetch-stats-history \
  --season 2015-16 \
  --season 2016-17 \
  --season 2017-18 \
  --season 2018-19 \
  --season 2019-20 \
  --season 2020-21 \
  --season 2021-22 \
  --season 2022-23 \
  --season 2023-24 \
  --season 2024-25 \
  --endpoint boxscoresummaryv2 \
  --max-workers 2 \
  --min-request-interval 0.75 \
  --request-interval-jitter 0.10 \
  --max-retries 3 \
  --run-id availability-summary-v1

uv run nba-build-player-availability \
  2015-16 2016-17 2017-18 2018-19 2019-20 \
  2020-21 2021-22 2022-23 2023-24 2024-25 2025-26
