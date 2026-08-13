export type Player = {
  player_id: number;
  player_name: string;
  team: string;
  position: string;
  age: number | null;
  rapm: number;
  prior_context_unit_edge: number | null;
  rating_season?: string;
  possessions: number;
  games: number;
  profile_source: string;
  profile_imputed: number | null;
  profile_replacement_weight: number | null;
  three_pm_per_100: number | null;
  assists_per_100: number | null;
  usage_per_100: number | null;
  offensive_rebounds_per_100: number | null;
  defensive_rebounds_per_100: number | null;
  rookie_season: string | null;
  rating_history: Array<{
    season: string;
    rating: number;
    prior_rating: number | null;
    season_update: number | null;
    prior_context_unit_edge: number | null;
    age: number | null;
    team_id: number | null;
    team: string;
    possessions: number;
    games: number;
    games_started: number;
    season_min_rating?: number;
    season_max_rating?: number;
    season_max_player_id?: number;
    season_max_player_name?: string;
    team_splits?: Array<{
      team_id: number;
      team: string;
      possessions: number;
      games: number;
    }>;
  }>;
  league_leader_history?: Array<{
    season: string;
    rating: number;
    player_id: number;
    player_name: string;
  }>;
};

export type RankedPlayer = Player & {
  rank: number;
};

export type RankedLineup = {
  rank: number;
  team_id: number;
  team: string;
  player_ids: number[];
  player_names: string[];
  lineup_label: string;
  possessions: number;
  games: number;
  player_rating: number;
  player_edge: number;
  composition_rating: number;
  composition_edge: number;
  matchup_bonus: number;
  context_edge: number;
  gestalt_score: number;
  actual_net_rating: number;
};

export type ContextFeature = {
  id: string;
  label: string;
  value: number;
  contribution: number;
};

export type FeatureResponseCurve = {
  id: string;
  support_low: number;
  support_high: number;
  unit_value: number;
  unit_contribution: number;
  opponent_value: number;
  opponent_contribution: number;
  points: Array<{ value: number; contribution: number }>;
};

export type Matchup = {
  season: string;
  run_id: string;
  retrospective: boolean;
  unit_season: string;
  opponent_season: string;
  environment: "unit" | "neutral" | "opponent";
  environment_seasons: string[];
  unit: { additive_rating: number; players: Player[] };
  opponent: { additive_rating: number; players: Player[] };
  additive_margin: number;
  contextual_adjustment: number;
  unit_composition_rating: number;
  opponent_composition_rating: number;
  portable_composition_margin: number;
  matchup_adjustment: number;
  predicted_net_rating: number;
  feature_contributions: ContextFeature[];
  composition_feature_contributions: ContextFeature[];
  matchup_feature_contributions: ContextFeature[];
  composition_response_curves: FeatureResponseCurve[];
  matchup_response_curves: FeatureResponseCurve[];
};
