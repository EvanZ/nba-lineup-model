import { useEffect, useState } from "react";
import type { Player } from "./types";

type PlayerIdentity = Pick<Player, "player_id" | "player_name">;

export function PlayerHeadshot({ player }: { player: PlayerIdentity }) {
  const [failed, setFailed] = useState(false);

  useEffect(() => setFailed(false), [player.player_id]);

  if (failed) {
    return (
      <span className="player-headshot player-headshot-fallback" aria-hidden="true">
        {player.player_name.split(" ").map((part) => part[0]).join("").slice(0, 2)}
      </span>
    );
  }

  return (
    <img
      className="player-headshot"
      src={`/api/headshots/${player.player_id}.png`}
      alt=""
      onError={() => setFailed(true)}
    />
  );
}

