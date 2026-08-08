import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUpRight,
  CircleAlert,
  Dices,
  LoaderCircle,
  Search,
  Sparkles,
  X,
} from "lucide-react";

import type { Matchup, Player } from "./types";

type Side = "unit" | "opponent";

const SIDE_LABELS: Record<Side, string> = { unit: "Your unit", opponent: "Opponent" };

const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

function formatRating(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
}

function App() {
  const [unit, setUnit] = useState<Player[]>([]);
  const [opponent, setOpponent] = useState<Player[]>([]);
  const [result, setResult] = useState<Matchup | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [isLoadingOpponent, setIsLoadingOpponent] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const canEvaluate = unit.length === 5 && opponent.length === 5;
  const allSelectedIds = useMemo(
    () => new Set([...unit, ...opponent].map((player) => player.player_id)),
    [unit, opponent],
  );

  useEffect(() => {
    void loadDefaultOpponent();
  }, []);

  async function loadDefaultOpponent() {
    try {
      setError(null);
      setIsLoadingOpponent(true);
      const parameters = new URLSearchParams();
      unit.forEach((player) => parameters.append("exclude_player_id", String(player.player_id)));
      const response = await fetch(`/api/default-opponent?${parameters.toString()}`);
      if (!response.ok) throw new Error("Default opponent is unavailable.");
      const payload = (await response.json()) as { players: Player[] };
      setOpponent(payload.players);
      setResult(null);
    } catch (opponentError) {
      setError((opponentError as Error).message);
    } finally {
      setIsLoadingOpponent(false);
    }
  }

  function removePlayer(side: Side, playerId: number) {
    const setter = side === "unit" ? setUnit : setOpponent;
    setter((current) => current.filter((player) => player.player_id !== playerId));
    setResult(null);
  }

  async function evaluate() {
    if (!canEvaluate) return;
    try {
      setError(null);
      setIsEvaluating(true);
      const response = await fetch("/api/matchups", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          unit_player_ids: unit.map((player) => player.player_id),
          opponent_player_ids: opponent.map((player) => player.player_id),
        }),
      });
      const payload = (await response.json()) as Matchup | { detail: string };
      if (!response.ok) throw new Error("detail" in payload ? payload.detail : "Lineup evaluation failed.");
      setResult(payload as Matchup);
    } catch (evaluationError) {
      setError((evaluationError as Error).message);
    } finally {
      setIsEvaluating(false);
    }
  }

  return (
    <main>
      <header className="site-header">
        <a className="wordmark" href="/" aria-label="NBA GESTALT home">
          <span>NBA</span> GESTALT
        </a>
        <div className="header-meta">
          <span>Lineup Lab</span>
          <span className="season-chip">2025-26 retrospective</span>
        </div>
      </header>

      <section className="intro" aria-labelledby="page-title">
        <p className="eyebrow">Gestalt Estimates Situational Teammate-Adjusted Lineup Terms</p>
        <h1 id="page-title">Build the five.</h1>
      </section>

      <section className="lab-grid" aria-label="Lineup builder">
        <div className="builder-panel">
          <div className="lineup-columns">
            <LineupSelector
              side="unit"
              players={unit}
              unavailablePlayerIds={allSelectedIds}
              onAdd={(player) => {
                setUnit((current) => [...current, player]);
                setResult(null);
              }}
              onRemove={(playerId) => removePlayer("unit", playerId)}
              onError={setError}
            />
            <LineupSelector
              side="opponent"
              players={opponent}
              unavailablePlayerIds={allSelectedIds}
              isLoading={isLoadingOpponent}
              onRefresh={() => void loadDefaultOpponent()}
              onAdd={(player) => {
                setOpponent((current) => [...current, player]);
                setResult(null);
              }}
              onRemove={(playerId) => removePlayer("opponent", playerId)}
              onError={setError}
            />
          </div>

          <button className="evaluate-button" onClick={evaluate} disabled={!canEvaluate || isEvaluating}>
            {isEvaluating ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}
            {isEvaluating ? "Evaluating" : "Evaluate matchup"}
            {!isEvaluating && <ArrowUpRight size={17} />}
          </button>
          {!canEvaluate && <p className="builder-note">Choose five players for each side to evaluate the matchup.</p>}
          {error && <p className="error"><CircleAlert size={16} /> {error}</p>}
        </div>

        <Results result={result} />
      </section>
    </main>
  );
}

type LineupSelectorProps = {
  side: Side;
  players: Player[];
  unavailablePlayerIds: Set<number>;
  onAdd: (player: Player) => void;
  onRemove: (playerId: number) => void;
  onError: (message: string) => void;
  isLoading?: boolean;
  onRefresh?: () => void;
};

function LineupSelector({
  side,
  players,
  unavailablePlayerIds,
  onAdd,
  onRemove,
  onError,
  isLoading = false,
  onRefresh,
}: LineupSelectorProps) {
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<Player[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const searchInput = useRef<HTMLInputElement>(null);
  const label = SIDE_LABELS[side];

  useEffect(() => {
    if (!query.trim()) {
      setMatches([]);
      return;
    }
    const controller = new AbortController();
    const delay = window.setTimeout(async () => {
      try {
        setIsSearching(true);
        const response = await fetch(`/api/players?q=${encodeURIComponent(query)}&limit=10`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error("Player search is unavailable.");
        const payload = (await response.json()) as { players: Player[] };
        setMatches(
          payload.players.filter((player) => !unavailablePlayerIds.has(player.player_id)),
        );
      } catch (searchError) {
        if ((searchError as Error).name !== "AbortError") onError((searchError as Error).message);
      } finally {
        setIsSearching(false);
      }
    }, 180);
    return () => {
      controller.abort();
      window.clearTimeout(delay);
    };
  }, [onError, query, unavailablePlayerIds]);

  function addPlayer(player: Player) {
    if (players.length === 5) return;
    onAdd(player);
    setQuery("");
    setMatches([]);
    searchInput.current?.focus();
  }

  return (
    <section className="lineup-column" aria-label={`${label} selections`}>
      <div className="lineup-column-header">
        <h2>{label}</h2>
        <span className="lineup-controls">
          {players.length}/5
          {onRefresh && (
            <button
              className="refresh-button"
              onClick={onRefresh}
              disabled={isLoading}
              aria-label="Choose another high-possession opponent"
              title="Choose another high-possession opponent"
            >
              <Dices className={isLoading ? "spin" : ""} size={15} />
            </button>
          )}
        </span>
      </div>
      <div className="slot-list">
        {[0, 1, 2, 3, 4].map((slot) => {
          const player = players[slot];
          return player ? (
            <div className="player-slot filled" key={player.player_id}>
              <span className="slot-number">{slot + 1}</span>
              <span className="player-in-slot">
                <strong>{player.player_name}</strong>
                <small>{player.team} · {player.position}</small>
              </span>
              <span className="slot-rating">{formatRating(player.rapm)}</span>
              <button
                className="icon-button"
                onClick={() => onRemove(player.player_id)}
                aria-label={`Remove ${player.player_name}`}
                title={`Remove ${player.player_name}`}
              >
                <X size={16} />
              </button>
            </div>
          ) : (
            <div className="player-slot empty" key={slot}>
              <span className="slot-number">{slot + 1}</span>
              <span>Search to add a player</span>
            </div>
          );
        })}
      </div>
      <div className="search-area">
        <label className="search-box">
          <Search size={20} aria-hidden="true" />
          <input
            ref={searchInput}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={`Search ${label.toLowerCase()} players`}
            disabled={players.length === 5}
          />
          {isSearching && <LoaderCircle className="spin" size={18} aria-label="Searching" />}
        </label>
        {query && matches.length > 0 && (
          <div className="search-results" role="listbox">
            {matches.map((player) => (
              <button className="search-result" key={player.player_id} onClick={() => addPlayer(player)}>
                <span>
                  <strong>{player.player_name}</strong>
                  <small>{player.team} · {player.position} · {number.format(player.possessions)} poss.</small>
                </span>
                <span>{formatRating(player.rapm)}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function Results({ result }: { result: Matchup | null }) {
  if (!result) {
    return (
      <aside className="results-panel empty-results">
        <div className="result-mark" aria-hidden="true">G</div>
        <p>Matchup output will appear here.</p>
      </aside>
    );
  }
  const decisiveSignals = result.feature_contributions.slice(0, 6);
  return (
    <aside className="results-panel" aria-live="polite">
      <div className="result-heading">
        <span>Neutral-court estimate</span>
        <small>2025-26 completed fit</small>
      </div>
      <div className="primary-rating">
        <strong>{formatRating(result.predicted_net_rating)}</strong>
        <span>net rating · your unit vs opponent</span>
      </div>
      <dl className="metric-row">
        <div><dt>Player value</dt><dd>{formatRating(result.additive_margin)}</dd></div>
        <div><dt>Context</dt><dd>{formatRating(result.contextual_adjustment)}</dd></div>
        <div><dt>Your five</dt><dd>{formatRating(result.unit.additive_rating)}</dd></div>
      </dl>

      <div className="signal-section">
        <h2>Context signals</h2>
        {decisiveSignals.map((signal) => (
          <div className="signal-row" key={signal.id}>
            <span>{signal.label}</span>
            <strong className={signal.contribution >= 0 ? "positive" : "negative"}>
              {formatRating(signal.contribution)}
            </strong>
          </div>
        ))}
      </div>

      <div className="roster-summary">
        <h2>Unit profile</h2>
        <div className="profile-grid">
          <ProfileMetric label="3PM / 100" value={average(result.unit.players, "three_pm_per_100")} />
          <ProfileMetric label="AST / 100" value={average(result.unit.players, "assists_per_100")} />
          <ProfileMetric label="Usage / 100" value={average(result.unit.players, "usage_per_100")} />
          <ProfileMetric label="DREB / 100" value={average(result.unit.players, "defensive_rebounds_per_100")} />
        </div>
      </div>
    </aside>
  );
}

function ProfileMetric({ label, value }: { label: string; value: number }) {
  return <div><span>{label}</span><strong>{value.toFixed(1)}</strong></div>;
}

function average(players: Player[], column: keyof Player) {
  return players.reduce((total, player) => total + Number(player[column]), 0) / players.length;
}

export default App;
