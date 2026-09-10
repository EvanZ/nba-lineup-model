import { useEffect, useMemo, useState } from "react";
import { Calculator, CircleAlert, LoaderCircle, RotateCcw } from "lucide-react";

import type {
  MinutesProjectionPayload,
  MinutesProjectionPlayer,
  WinProjectionPayload,
  WinProjectionTeam,
} from "./types";

type PlayerInputOverride = {
  availabilityProbability?: number;
  conditionalMinutesPerGame?: number;
};

type Overrides = Record<string, PlayerInputOverride>;

type MinuteAllocation = {
  minutes: Map<number, number>;
  rotationPlayerIds: Set<number>;
  totalRawMinutes: number;
};

// The active-15 baseline changes the meaning of every residual allocation.
// Do not carry manual judgments made against the earlier all-roster baseline.
const STORAGE_KEY = "nba-gestalt:win-projection-input-overrides:v3";

function overrideKey(team: string, playerId: number) {
  return `${team}:${playerId}`;
}

function formatRating(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
}

function marketDirection(projectedWins: number, total: number | undefined) {
  if (total === undefined) return "";
  if (Math.abs(projectedWins - total) < 0.05) return "P";
  return projectedWins > total ? "O" : "U";
}

function formatMarketDelta(projectedWins: number, total: number | undefined) {
  if (total === undefined) return "";
  const delta = projectedWins - total;
  return `(${delta >= 0 ? "+" : ""}${delta.toFixed(1)})`;
}

function readOverrides(): Overrides {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (!saved) return {};
    const parsed = JSON.parse(saved) as Record<string, unknown>;
    return Object.fromEntries(
      Object.entries(parsed).flatMap(([key, value]) => {
        if (!value || typeof value !== "object") return [];
        const candidate = value as Record<string, unknown>;
        const availabilityProbability = candidate.availabilityProbability;
        const conditionalMinutesPerGame = candidate.conditionalMinutesPerGame;
        const output: PlayerInputOverride = {};
        if (typeof availabilityProbability === "number" && Number.isFinite(availabilityProbability)) {
          output.availabilityProbability = availabilityProbability;
        }
        if (typeof conditionalMinutesPerGame === "number" && Number.isFinite(conditionalMinutesPerGame)) {
          output.conditionalMinutesPerGame = conditionalMinutesPerGame;
        }
        return Object.keys(output).length ? [[key, output]] : [];
      }),
    ) as Overrides;
  } catch {
    return {};
  }
}

function playerInputs(player: MinutesProjectionPlayer, overrides: Overrides) {
  const override = overrides[overrideKey(player.team, player.player_id)];
  return {
    availabilityProbability: override?.availabilityProbability ?? player.availability_probability,
    conditionalMinutesPerGame: override?.conditionalMinutesPerGame ?? player.conditional_minutes_per_game,
  };
}

function minuteAllocation(
  players: MinutesProjectionPlayer[],
  overrides: Overrides,
  regulationMinutes: number,
  rotationSize: number,
): MinuteAllocation {
  const rawTotals = new Map(players.map((player) => {
    const inputs = playerInputs(player, overrides);
    const rawMinutes = 82 * inputs.availabilityProbability * inputs.conditionalMinutesPerGame;
    return [player.player_id, rawMinutes];
  }));
  const rotationPlayers = [...players].sort((left, right) => (
    (rawTotals.get(right.player_id) ?? 0) - (rawTotals.get(left.player_id) ?? 0)
    || left.player_name.localeCompare(right.player_name)
    || left.player_id - right.player_id
  )).slice(0, rotationSize);
  const rotationPlayerIds = new Set(rotationPlayers.map((player) => player.player_id));
  const totalRawMinutes = rotationPlayers.reduce(
    (total, player) => total + (rawTotals.get(player.player_id) ?? 0),
    0,
  );
  const minutes = totalRawMinutes <= 0
    ? new Map(players.map((player) => [player.player_id, 0]))
    : new Map(players.map((player) => [
    player.player_id,
    rotationPlayerIds.has(player.player_id)
      ? regulationMinutes * (rawTotals.get(player.player_id) ?? 0) / totalRawMinutes
      : 0,
  ]));
  return { minutes, rotationPlayerIds, totalRawMinutes };
}

function calculateProjection(
  payload: MinutesProjectionPayload,
  baseline: WinProjectionPayload,
  overrides: Overrides,
): WinProjectionPayload {
  const rawStrength = new Map<string, number>();
  const marketTotals = new Map(
    baseline.teams.map((team) => [team.team, team.betmgm_win_total]),
  );
  for (const team of payload.teams) {
    const players = payload.players.filter((player) => player.team === team);
    const { minutes } = minuteAllocation(
      players,
      overrides,
      payload.regulation_team_minutes,
      payload.initial_rotation_size,
    );
    rawStrength.set(team, players.reduce((total, player) => (
      total + (minutes.get(player.player_id) ?? 0) * player.projected_nail / 48
    ), 0));
  }
  const meanStrength = Array.from(rawStrength.values()).reduce((total, value) => total + value, 0)
    / rawStrength.size;
  const strength = new Map(Array.from(rawStrength, ([team, value]) => [team, value - meanStrength]));
  const scheduledWins = new Map(payload.teams.map((team) => [team, 0]));
  const sigmoid = (value: number) => 1 / (1 + Math.exp(-Math.max(-35, Math.min(35, value))));

  for (const game of baseline.scheduled_games) {
    const edge = (strength.get(game.home_team) ?? 0) - (strength.get(game.away_team) ?? 0)
      + baseline.home_court
      + baseline.back_to_back * (game.home_back_to_back - game.away_back_to_back);
    const homeWinProbability = sigmoid(baseline.win_probability_scale * edge);
    scheduledWins.set(
      game.home_team,
      (scheduledWins.get(game.home_team) ?? 0) + homeWinProbability,
    );
    scheduledWins.set(
      game.away_team,
      (scheduledWins.get(game.away_team) ?? 0) + 1 - homeWinProbability,
    );
  }
  const teams: WinProjectionTeam[] = payload.teams.map((team) => {
    const teamStrength = strength.get(team) ?? 0;
    const scheduled = scheduledWins.get(team) ?? 0;
    const neutralHome = sigmoid(baseline.win_probability_scale * (teamStrength + baseline.home_court));
    const neutralAway = sigmoid(baseline.win_probability_scale * (teamStrength - baseline.home_court));
    const unassigned = baseline.unassigned_regular_games_per_team * (neutralHome + neutralAway) / 2;
    const wins = scheduled + unassigned;
    return {
      team,
      team_strength: teamStrength,
      betmgm_win_total: marketTotals.get(team),
      scheduled_wins: scheduled,
      scheduled_games: baseline.scheduled_games_per_team,
      unassigned_wins: unassigned,
      projected_wins: wins,
      projected_losses: 82 - wins,
    };
  });
  teams.sort((left, right) => right.projected_wins - left.projected_wins || left.team.localeCompare(right.team));
  return { ...baseline, teams };
}

export function WinProjectionsPage() {
  const [payload, setPayload] = useState<MinutesProjectionPayload | null>(null);
  const [team, setTeam] = useState("");
  const [overrides, setOverrides] = useState<Overrides>(readOverrides);
  const [calculatedForecast, setCalculatedForecast] = useState<WinProjectionPayload | null>(null);
  const [forecastNeedsUpdate, setForecastNeedsUpdate] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        const response = await fetch("/api/win-projections", { signal: controller.signal });
        if (!response.ok) throw new Error("Preseason minute projections are unavailable.");
        const next = (await response.json()) as MinutesProjectionPayload;
        setPayload(next);
        setTeam((current) => current || next.teams[0] || "");
      } catch (requestError) {
        if ((requestError as Error).name !== "AbortError") setError((requestError as Error).message);
      }
    })();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
  }, [overrides]);

  const players = useMemo(() => (
    payload?.players.filter((player) => player.team === team) ?? []
  ), [payload, team]);
  const allocation = useMemo(() => (
    payload
      ? minuteAllocation(
        players,
        overrides,
        payload.regulation_team_minutes,
        payload.initial_rotation_size,
      )
      : { minutes: new Map<number, number>(), rotationPlayerIds: new Set<number>(), totalRawMinutes: 0 }
  ), [overrides, payload, players]);
  const hasInvalidOverrides = payload !== null && allocation.totalRawMinutes <= 0;
  const minutes = allocation.minutes;
  const orderedPlayers = useMemo(() => (
    [...players].sort((left, right) => (
      (minutes.get(right.player_id) ?? 0) - (minutes.get(left.player_id) ?? 0)
      || left.player_name.localeCompare(right.player_name)
      || left.player_id - right.player_id
    ))
  ), [minutes, players]);
  const activeOverrides = players.filter((player) => (
    overrides[overrideKey(player.team, player.player_id)] !== undefined
  )).length;
  const updateOverride = (
    player: MinutesProjectionPlayer,
    field: keyof PlayerInputOverride,
    rawValue: string,
  ) => {
    const key = overrideKey(player.team, player.player_id);
    if (!rawValue.trim()) {
      setForecastNeedsUpdate(true);
      setOverrides((current) => {
        const next = { ...current };
        const currentOverride = { ...(next[key] ?? {}) };
        delete currentOverride[field];
        if (Object.keys(currentOverride).length) next[key] = currentOverride;
        else delete next[key];
        return next;
      });
      return;
    }
    const value = Number(rawValue);
    if (!Number.isFinite(value)) return;
    setForecastNeedsUpdate(true);
    setOverrides((current) => ({
      ...current,
      [key]: {
        ...(current[key] ?? {}),
        [field]: field === "availabilityProbability"
          ? Math.max(0, Math.min(1, value / 100))
          : Math.max(0, Math.min(48, value)),
      },
    }));
  };

  const resetTeam = () => {
    setForecastNeedsUpdate(true);
    setOverrides((current) => Object.fromEntries(
      Object.entries(current).filter(([key]) => !key.startsWith(`${team}:`)),
    ));
  };

  const selectForecastTeam = (nextTeam: string) => {
    setTeam(nextTeam);
  };

  if (error) return <p className="error win-projections-error"><CircleAlert size={16} /> {error}</p>;
  if (!payload) return <div className="profile-loading"><LoaderCircle className="spin" size={20} /> Loading preseason minutes</div>;
  const baselineForecast = payload.win_projection;
  const forecast = calculatedForecast ?? baselineForecast;
  const calculateForecast = () => {
    if (!baselineForecast || hasInvalidOverrides) return;
    setCalculatedForecast(calculateProjection(payload, baselineForecast, overrides));
    setForecastNeedsUpdate(false);
  };

  return (
    <article className="win-projections-page" aria-labelledby="win-projections-title">
      <section className="win-projections-hero">
        <p className="eyebrow">{payload.season} preseason planning</p>
        <h1 id="win-projections-title">Win projections.</h1>
        <p>Adjust medical availability and conditional playing time, then recalculate. The top {payload.initial_rotation_size} override-adjusted raw projections form the rotation.</p>
      </section>

      <section className="win-projections-workspace" aria-label="Team minute projections">
        {forecast && <section className="win-forecast" aria-labelledby="win-forecast-title">
          <div className="win-forecast-heading">
            <div><p className="eyebrow">{calculatedForecast && !forecastNeedsUpdate ? "Custom forecast" : "Baseline forecast"}</p><h2 id="win-forecast-title">Projected wins.</h2></div>
            <p>{forecast.scheduled_games_per_team} scheduled games + {forecast.unassigned_regular_games_per_team} unassigned NBA Cup-related games.</p>
          </div>
          <div className="win-forecast-table-wrap"><table className="win-forecast-table">
            <thead><tr><th>Team</th><th>Strength</th><th>BetMGM</th><th>GESTALT</th><th>Losses</th></tr></thead>
            <tbody>{forecast.teams.map((item) => {
              const direction = marketDirection(item.projected_wins, item.betmgm_win_total);
              const directionClass = direction === "U" ? "negative" : "positive";
              return <tr key={item.team} className={item.team === team ? "selected-team" : ""}>
                <th><button className="win-forecast-team-link" type="button" aria-controls="team-minutes" aria-pressed={item.team === team} onClick={() => selectForecastTeam(item.team)}>{item.team}</button></th>
                <td className={item.team_strength < 0 ? "negative" : "positive"}>{formatRating(item.team_strength)}</td>
                <td>{item.betmgm_win_total?.toFixed(1) ?? "-"}</td>
                <td className="win-forecast-wins">{item.projected_wins.toFixed(1)} <small className={directionClass}>{direction} {formatMarketDelta(item.projected_wins, item.betmgm_win_total)}</small></td>
                <td>{item.projected_losses.toFixed(1)}</td>
              </tr>;
            })}</tbody>
          </table></div>
          <p className="win-projections-note">{forecast.contract} {forecast.market_win_totals && <><a href={forecast.market_win_totals.source_url} target="_blank" rel="noreferrer">{forecast.market_win_totals.provider} win totals</a> captured {forecast.market_win_totals.as_of}; O/U compares GESTALT wins to that line. </>}Calibration: {forecast.calibration.calibration_season} only; {forecast.calibration.holdout_season} Brier {forecast.calibration.holdout_brier.toFixed(3)}, accuracy {(forecast.calibration.holdout_accuracy * 100).toFixed(1)}%.</p>
        </section>}
        <section className="win-team-panel" id="team-minutes" aria-label={`${team} minute projection`}>
        <div className="win-projections-toolbar">
          <label className="win-projections-team-filter">
            <span>Team</span>
            <select value={team} onChange={(event) => setTeam(event.target.value)}>
              {payload.teams.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <button className="win-projections-reset" type="button" onClick={resetTeam} disabled={activeOverrides === 0} title="Reset this team to its baseline availability and conditional-minute inputs">
            <RotateCcw size={15} aria-hidden="true" />
            <span>Reset team</span>
          </button>
          <button className="win-projections-calculate" type="button" onClick={calculateForecast} disabled={hasInvalidOverrides}>
            <Calculator size={15} aria-hidden="true" />
            <span>{forecastNeedsUpdate ? "Recalculate" : "Calculate projection"}</span>
          </button>
        </div>
        {hasInvalidOverrides && <p className="error win-projections-error"><CircleAlert size={16} /> At least one player must have positive availability and conditional minutes.</p>}
        <div className="win-projections-table-wrap">
          <table className="win-projections-table">
            <thead><tr><th>Player</th><th>NAIL</th><th>P(available)</th><th>E[MPG | available]</th><th>Squashed MPG</th></tr></thead>
            <tbody>
              {orderedPlayers.map((player) => {
                const key = overrideKey(player.team, player.player_id);
                const override = overrides[key];
                const projected = minutes.get(player.player_id) ?? player.baseline_minutes_per_game;
                return <tr key={player.player_id} className={override === undefined ? "" : "has-minute-override"}>
                  <th scope="row"><a href={`#player/${player.player_id}`}>{player.player_name}</a><small>{player.position} · Age {player.age?.toFixed(0) ?? "-"}{allocation.rotationPlayerIds.has(player.player_id) ? "" : " · Outside rotation"}{player.is_rating_fallback ? " · NAIL fallback" : ""}</small></th>
                  <td className={player.projected_nail < 0 ? "negative" : "positive"}>{formatRating(player.projected_nail)}</td>
                  <td><input aria-label={`Availability probability for ${player.player_name}`} type="number" min="0" max="100" step="1" value={((override?.availabilityProbability ?? player.availability_probability) * 100).toFixed(1)} onChange={(event) => updateOverride(player, "availabilityProbability", event.target.value)} /><small>%</small></td>
                  <td><input aria-label={`Conditional minutes for ${player.player_name}`} type="number" min="0" max="48" step="0.5" value={(override?.conditionalMinutesPerGame ?? player.conditional_minutes_per_game).toFixed(1)} onChange={(event) => updateOverride(player, "conditionalMinutesPerGame", event.target.value)} /></td>
                  <td className="projected-minutes">{projected.toFixed(1)}</td>
                </tr>;
              })}
            </tbody>
          </table>
        </div>
        <ul className="win-projections-mobile-list">
          {orderedPlayers.map((player) => {
            const key = overrideKey(player.team, player.player_id);
            const override = overrides[key];
            const projected = minutes.get(player.player_id) ?? player.baseline_minutes_per_game;
            return <li key={player.player_id} className={override === undefined ? "" : "has-minute-override"}>
              <div className="win-projections-mobile-heading">
                <a href={`#player/${player.player_id}`}>{player.player_name}</a>
                <span className={player.projected_nail < 0 ? "negative" : "positive"}>{formatRating(player.projected_nail)}</span>
              </div>
              <p>{player.position} · Age {player.age?.toFixed(0) ?? "-"}{allocation.rotationPlayerIds.has(player.player_id) ? "" : " · Outside rotation"}{player.is_rating_fallback ? " · NAIL fallback" : ""}</p>
              <dl>
                <div><dt>P(available)</dt><dd><input aria-label={`Availability probability for ${player.player_name}`} type="number" min="0" max="100" step="1" value={((override?.availabilityProbability ?? player.availability_probability) * 100).toFixed(1)} onChange={(event) => updateOverride(player, "availabilityProbability", event.target.value)} />%</dd></div>
                <div><dt>E[MPG | available]</dt><dd><input aria-label={`Conditional minutes for ${player.player_name}`} type="number" min="0" max="48" step="0.5" value={(override?.conditionalMinutesPerGame ?? player.conditional_minutes_per_game).toFixed(1)} onChange={(event) => updateOverride(player, "conditionalMinutesPerGame", event.target.value)} /></dd></div>
                <div><dt>Squashed MPG</dt><dd className="projected-minutes">{projected.toFixed(1)}</dd></div>
              </dl>
            </li>;
          })}
        </ul>
        <p className="win-projections-note">Baseline: {payload.model}. {payload.contract}</p>
        </section>
      </section>
    </article>
  );
}
