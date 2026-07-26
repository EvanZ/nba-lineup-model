from __future__ import annotations

import argparse
from pathlib import Path

from nba_lineup_model.ingest.nba_cdn import NbaCdnClient, RawJsonCache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch one NBA game from direct CDN endpoints.")
    parser.add_argument("game_id", help="10-digit NBA game ID, e.g. 0022000180")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw JSON cache directory")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached responses")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    client = NbaCdnClient(cache=RawJsonCache(Path(args.raw_dir)))
    use_cache = not args.refresh

    pbp = client.fetch_play_by_play(args.game_id, use_cache=use_cache)
    box = client.fetch_boxscore(args.game_id, use_cache=use_cache)

    print(f"play_by_play: {len(pbp.payload.get('game', {}).get('actions', []))} actions")
    print(f"boxscore: {box.payload.get('game', {}).get('gameStatusText', 'unknown status')}")


if __name__ == "__main__":
    main()
