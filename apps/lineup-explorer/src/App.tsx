import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUpRight,
  BookOpen,
  CircleAlert,
  Dices,
  GitBranch,
  Info,
  LoaderCircle,
  Search,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";

import type { ContextFeature, FeatureResponseCurve, Matchup, Player } from "./types";

type Side = "unit" | "opponent";

const SIDE_LABELS: Record<Side, string> = { unit: "Your unit", opponent: "Opponent" };
const MATERIAL_COMPONENT_CONTRIBUTION = 0.05;
const FEATURE_DESCRIPTIONS: Record<string, string> = {
  home_minus_away_three_pa_per_100: "Sum of the five players' prior-season three-point attempts per 100 possessions.",
  home_minus_away_three_pm_per_100: "Sum of the five players' prior-season made three-pointers per 100 possessions.",
  home_minus_away_assists_per_100: "Sum of the five players' prior-season assists per 100 possessions.",
  home_minus_away_turnovers_per_100: "Sum of the five players' prior-season turnovers per 100 possessions.",
  home_minus_away_usage_per_100: "Sum of the five players' prior-season usage events per 100 possessions.",
  home_minus_away_offensive_rebounds_per_100: "Sum of the five players' prior-season offensive rebounds per 100 possessions.",
  home_minus_away_defensive_rebounds_per_100: "Sum of the five players' prior-season defensive rebounds per 100 possessions.",
  home_minus_away_steals_per_100: "Sum of the five players' prior-season steals per 100 possessions.",
  home_minus_away_blocks_per_100: "Sum of the five players' prior-season blocks per 100 possessions.",
  home_minus_away_bottom_two_three_pm: "Combined made three-pointers per 100 possessions for the two least prolific shooters in the unit.",
  home_minus_away_credible_shooter_count: "Number of players with at least two made three-pointers per 100 possessions in the prior season.",
  home_minus_away_top_two_assists: "Combined assists per 100 possessions for the two highest-assist players in the unit.",
  home_minus_away_usage_concentration: "Share of the unit's usage events supplied by its two highest-usage players.",
  home_minus_away_sqrt_offensive_rebounds: "Square root of the unit's total offensive rebounds per 100 possessions, encoding diminishing returns.",
  home_minus_away_sqrt_defensive_rebounds: "Square root of the unit's total defensive rebounds per 100 possessions, encoding diminishing returns.",
  home_minus_away_imputed_count: "Number of players whose profile is a cold-start or replacement estimate rather than a prior-season box-score profile.",
  home_minus_away_replacement_weight: "Sum of the players' exposure-gated replacement-profile weights.",
  home_minus_away_shooting_usage_interaction: "Bottom-two shooting multiplied by usage concentration; a shooting-depth-by-usage interaction.",
  home_minus_away_shooter_passing_interaction: "Credible-shooter count multiplied by top-two assists; a shooting-by-passing interaction.",
  home_minus_away_rebounding_usage_interaction: "Diminishing offensive rebounding multiplied by usage concentration.",
};

const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

function formatRating(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
}

function Rating({ value, className }: { value: number; className?: string }) {
  return (
    <span className={["rating-value", value < 0 ? "negative" : "", className].filter(Boolean).join(" ")}>
      {formatRating(value)}
    </span>
  );
}

function App() {
  const [unit, setUnit] = useState<Player[]>([]);
  const [opponent, setOpponent] = useState<Player[]>([]);
  const [result, setResult] = useState<Matchup | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [isLoadingUnit, setIsLoadingUnit] = useState(false);
  const [isLoadingOpponent, setIsLoadingOpponent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canEvaluate = unit.length === 5 && opponent.length === 5;
  const allSelectedIds = useMemo(
    () => new Set([...unit, ...opponent].map((player) => player.player_id)),
    [unit, opponent],
  );

  async function loadRandomLineup(side: Side) {
    try {
      setError(null);
      const setLoading = side === "unit" ? setIsLoadingUnit : setIsLoadingOpponent;
      setLoading(true);
      const parameters = new URLSearchParams();
      const currentPlayers = side === "unit" ? unit : opponent;
      const protectedPlayers = side === "unit" ? opponent : unit;
      [...currentPlayers, ...protectedPlayers].forEach((player) => {
        parameters.append("exclude_player_id", String(player.player_id));
      });
      const response = await fetch(`/api/default-opponent?${parameters.toString()}`);
      if (!response.ok) throw new Error("Random lineup is unavailable.");
      const payload = (await response.json()) as { players: Player[] };
      if (side === "unit") {
        setUnit((current) => [...current, ...payload.players].slice(0, 5));
      } else {
        setOpponent((current) => [...current, ...payload.players].slice(0, 5));
      }
      setResult(null);
    } catch (opponentError) {
      setError((opponentError as Error).message);
    } finally {
      if (side === "unit") {
        setIsLoadingUnit(false);
      } else {
        setIsLoadingOpponent(false);
      }
    }
  }

  function removePlayer(side: Side, playerId: number) {
    const setter = side === "unit" ? setUnit : setOpponent;
    setter((current) => current.filter((player) => player.player_id !== playerId));
    setResult(null);
  }

  function clearLineup(side: Side) {
    const setter = side === "unit" ? setUnit : setOpponent;
    setter([]);
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
          include_response_curves: true,
        }),
      });
      const payload = (await response.json()) as Matchup | { detail: string };
      if (!response.ok) throw new Error("detail" in payload ? payload.detail : "Lineup evaluation failed.");
      const matchup = payload as Matchup;
      setResult(matchup);
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
          <a
            className="header-link"
            href="https://github.com/EvanZ/nba-lineup-model"
            target="_blank"
            rel="noreferrer"
            aria-label="Open the NBA GESTALT GitHub repository"
          >
            <GitBranch size={15} aria-hidden="true" />
            <span>GitHub</span>
          </a>
          <a
            className="header-link"
            href="https://evanz.github.io/nba-lineup-model/"
            target="_blank"
            rel="noreferrer"
            aria-label="Open NBA GESTALT documentation"
          >
            <BookOpen size={15} aria-hidden="true" />
            <span>Docs</span>
          </a>
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
              isLoading={isLoadingUnit}
              onRefresh={() => void loadRandomLineup("unit")}
              onAdd={(player) => {
                setUnit((current) => [...current, player]);
                setResult(null);
              }}
              onRemove={(playerId) => removePlayer("unit", playerId)}
              onClear={() => clearLineup("unit")}
              onError={setError}
            />
            <LineupSelector
              side="opponent"
              players={opponent}
              unavailablePlayerIds={allSelectedIds}
              isLoading={isLoadingOpponent}
              onRefresh={() => void loadRandomLineup("opponent")}
              onAdd={(player) => {
                setOpponent((current) => [...current, player]);
                setResult(null);
              }}
              onRemove={(playerId) => removePlayer("opponent", playerId)}
              onClear={() => clearLineup("opponent")}
              onError={setError}
            />
          </div>

          <button className="evaluate-button" onClick={() => void evaluate()} disabled={!canEvaluate || isEvaluating}>
            {isEvaluating ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}
            {isEvaluating ? "Evaluating" : "Evaluate matchup"}
            {!isEvaluating && <ArrowUpRight size={17} />}
          </button>
          {!canEvaluate && <p className="builder-note">Choose five players for each side to evaluate the matchup.</p>}
          {error && <p className="error"><CircleAlert size={16} /> {error}</p>}
        </div>

        <Results result={result} />
      </section>
      {result && <ContextCurveExplorer result={result} />}
    </main>
  );
}

type LineupSelectorProps = {
  side: Side;
  players: Player[];
  unavailablePlayerIds: Set<number>;
  onAdd: (player: Player) => void;
  onRemove: (playerId: number) => void;
  onClear: () => void;
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
  onClear,
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
              disabled={isLoading || players.length === 5}
              aria-label={`Fill open ${label.toLowerCase()} slots randomly`}
              title={`Fill open ${label.toLowerCase()} slots randomly`}
            >
              <Dices className={isLoading ? "spin" : ""} size={15} />
            </button>
          )}
          <button
            className="clear-button"
            onClick={onClear}
            disabled={isLoading || players.length === 0}
            aria-label={`Clear ${label.toLowerCase()}`}
            title={`Clear ${label.toLowerCase()}`}
          >
            <Trash2 size={15} />
          </button>
        </span>
      </div>
      <div className="slot-list">
        {[0, 1, 2, 3, 4].map((slot) => {
          const player = players[slot];
          return player ? (
            <div className="player-slot filled" key={player.player_id}>
              <span className="slot-number">{slot + 1}</span>
              <PlayerHeadshot player={player} />
              <span className="player-in-slot">
                <strong>{player.player_name}</strong>
                <small>{player.team} · {player.position}</small>
              </span>
              <Rating className="slot-rating" value={player.rapm} />
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
                <PlayerHeadshot player={player} />
                <span className="search-result-details">
                  <strong>{player.player_name}</strong>
                  <small>{player.team} · {player.position} · {number.format(player.possessions)} poss.</small>
                </span>
                <Rating value={player.rapm} />
              </button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function PlayerHeadshot({ player }: { player: Player }) {
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
      src={`https://cdn.nba.com/headshots/nba/latest/1040x760/${player.player_id}.png`}
      alt=""
      onError={() => setFailed(true)}
    />
  );
}

function Results({ result }: {
  result: Matchup | null;
}) {

  if (!result) {
    return (
      <aside className="results-panel empty-results">
        <div className="result-mark" aria-hidden="true">G</div>
        <p>Matchup output will appear here.</p>
      </aside>
    );
  }
  const combinedEdge = result.additive_margin + result.portable_composition_margin;
  return (
    <aside className="results-panel" aria-live="polite">
      <div className="result-heading">
        <span>Neutral-court estimate</span>
        <small>2025-26 completed fit</small>
      </div>
      <div className="gestalt-score">
        <strong className={result.predicted_net_rating < 0 ? "negative" : ""}>
          {formatRating(result.predicted_net_rating)}
        </strong>
        <span>GESTALT score</span>
        <small>Expected net rating per 100 possessions</small>
      </div>
      <table className="rating-ledger" aria-label="Lineup rating calculation">
        <thead>
          <tr>
            <th scope="col">Rating</th>
            <th scope="col">Your unit</th>
            <th scope="col">Opponent</th>
            <th scope="col">Edge</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Player rating</th>
            <td><Rating value={result.unit.additive_rating} /></td>
            <td><Rating value={result.opponent.additive_rating} /></td>
            <td><Rating value={result.additive_margin} /></td>
          </tr>
          <tr>
            <th scope="row">Composition rating</th>
            <td><Rating value={result.unit_composition_rating} /></td>
            <td><Rating value={result.opponent_composition_rating} /></td>
            <td><Rating value={result.portable_composition_margin} /></td>
          </tr>
          <tr className="ledger-subtotal">
            <th scope="row">Player + composition</th>
            <td />
            <td />
            <td><Rating value={combinedEdge} /></td>
          </tr>
          <tr>
            <th scope="row">Matchup bonus</th>
            <td />
            <td />
            <td><Rating value={result.matchup_adjustment} /></td>
          </tr>
          <tr className="ledger-total">
            <th scope="row">GESTALT score</th>
            <td />
            <td />
            <td><Rating value={result.predicted_net_rating} /></td>
          </tr>
        </tbody>
      </table>
    </aside>
  );
}

function ContextCurveExplorer({
  result,
}: {
  result: Matchup;
}) {
  return (
    <section className="curve-explorer" aria-label="Context component curves">
      <h2>Context components</h2>
      <div className="curve-columns">
        <ContextCurveGroup
          title="Lineup composition"
          value={result.portable_composition_margin}
          signals={result.composition_feature_contributions}
          responseCurves={new Map((result.composition_response_curves ?? []).map((curve) => [curve.id, curve]))}
          kind="composition"
        />
        <ContextCurveGroup
          title="Matchup bonus"
          value={result.matchup_adjustment}
          signals={result.matchup_feature_contributions}
          responseCurves={new Map((result.matchup_response_curves ?? []).map((curve) => [curve.id, curve]))}
          kind="matchup"
        />
      </div>
    </section>
  );
}

function ContextCurveGroup({
  title,
  value,
  signals,
  responseCurves,
  kind,
}: {
  title: string;
  value: number;
  signals: ContextFeature[];
  responseCurves: Map<string, FeatureResponseCurve>;
  kind: "composition" | "matchup";
}) {
  const rankedSignals = signals.filter(
    (signal) => Math.abs(signal.contribution) >= MATERIAL_COMPONENT_CONTRIBUTION,
  ).sort(
    (left, right) => Math.abs(right.contribution) - Math.abs(left.contribution),
  );
  return (
    <div className="curve-group">
      <div className="curve-group-heading">
        <h3>{title}</h3>
        <strong><Rating value={value} /></strong>
      </div>
      <div className="curve-grid">
      {rankedSignals.map((signal, index) => (
        <article className="curve-card" key={signal.id}>
          <div className="curve-card-heading">
            <span className="curve-card-title"><b>#{index + 1}</b><span className="curve-card-label">{signal.label}</span>
              <button
                className="feature-info"
                type="button"
                aria-label={`Explain ${signal.label}`}
                data-tooltip={FEATURE_DESCRIPTIONS[signal.id] ?? signal.label}
              ><Info size={13} aria-hidden="true" /></button>
            </span>
            <strong><Rating value={signal.contribution} /></strong>
          </div>
          {responseCurves.get(signal.id) && (
            <ResponseSparkline
              curve={responseCurves.get(signal.id)!}
              yAxisDecimals={title === "Matchup bonus" ? 2 : 1}
              showOpponent={kind === "composition"}
            />
          )}
        </article>
      ))}
      </div>
    </div>
  );
}

function ResponseSparkline({
  curve,
  yAxisDecimals,
  showOpponent,
}: {
  curve: FeatureResponseCurve;
  yAxisDecimals: number;
  showOpponent: boolean;
}) {
  const width = 238;
  const height = 72;
  const padding = 3;
  const axisHeight = 16;
  const axisWidth = 25;
  const values = curve.points.map((point) => point.value);
  const responses = curve.points.map((point) => point.contribution);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const minResponse = Math.min(0, curve.unit_contribution, curve.opponent_contribution, ...responses);
  const maxResponse = Math.max(0, curve.unit_contribution, curve.opponent_contribution, ...responses);
  const x = (value: number) => axisWidth + ((value - minValue) / (maxValue - minValue)) * (width - axisWidth - padding);
  const y = (value: number) => height - padding - axisHeight - ((value - minResponse) / (maxResponse - minResponse || 1)) * (height - padding * 2 - axisHeight);
  const path = curve.points.map((point) => `${x(point.value)},${y(point.contribution)}`).join(" ");
  const supportStart = x(curve.support_low);
  const supportEnd = x(curve.support_high);
  const ticks = [curve.support_low, (curve.support_low + curve.support_high) / 2, curve.support_high];
  const yTicks = Array.from(new Set([minResponse, 0, maxResponse])).sort((left, right) => left - right);

  return (
    <svg className="response-sparkline response-sparkline-expanded" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Context response curve">
        <rect className="sparkline-support" x={supportStart} y={padding} width={supportEnd - supportStart} height={height - padding * 2} />
        {yTicks.map((value) => (
          <text className="sparkline-tick" key={value} x={axisWidth - 3} y={y(value) + 3} textAnchor="end">
            <tspan className={value < 0 ? "negative" : ""}>{value.toFixed(yAxisDecimals)}</tspan>
          </text>
        ))}
        <line className="sparkline-zero" x1={axisWidth} x2={width - padding} y1={y(0)} y2={y(0)} />
        <polyline className="sparkline-line" points={path} />
        {showOpponent && <>
          <line className="sparkline-opponent-marker" x1={x(curve.opponent_value)} x2={x(curve.opponent_value)} y1={padding} y2={height - padding - axisHeight} />
          <circle className="sparkline-opponent-point" cx={x(curve.opponent_value)} cy={y(curve.opponent_contribution)} r="2.1" />
        </>}
        <line className="sparkline-marker" x1={x(curve.unit_value)} x2={x(curve.unit_value)} y1={padding} y2={height - padding - axisHeight} />
        <circle className="sparkline-point" cx={x(curve.unit_value)} cy={y(curve.unit_contribution)} r="2.3" />
        {ticks.map((value) => (
          <text className="sparkline-tick" key={value} x={x(value)} y={height - 2} textAnchor="middle">
            <tspan className={value < 0 ? "negative" : ""}>{value.toFixed(1)}</tspan>
          </text>
        ))}
    </svg>
  );
}

export default App;
