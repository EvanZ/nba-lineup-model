export type Player = {
  player_id: number;
  player_name: string;
  team: string;
  position: string;
  draft_year: number | null;
  draft_round: number | null;
  draft_number: number | null;
  is_undrafted: boolean | null;
  draft_class_year: number | null;
  age: number | null;
  rapm: number;
  offense_rating?: number | null;
  defense_rating?: number | null;
  prior_rating?: number | null;
  season_update?: number | null;
  additive_profile_adjustment?: number | null;
  additive_profile_breakdown?: Array<{
    feature: string;
    player_value: number;
    reference_value: number;
    contribution: number;
  }>;
  observed_context_exposure: number | null;
  rating_season?: string;
  possessions: number;
  games: number;
  profile_source: string;
  profile_imputed: number | null;
  profile_replacement_weight: number | null;
  three_pa_per_100: number | null;
  three_pm_per_100: number | null;
  assists_per_100: number | null;
  turnovers_per_100: number | null;
  usage_per_100: number | null;
  steals_per_100: number | null;
  blocks_per_100: number | null;
  offensive_rebound_pct: number | null;
  rookie_season: string | null;
  rating_history: Array<{
    season: string;
    rating: number;
    offense_rating?: number | null;
    defense_rating?: number | null;
    nail_rank: number;
    prior_rating: number | null;
    season_update: number | null;
    additive_profile_adjustment: number | null;
    observed_context_exposure: number | null;
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
      is_primary_team: boolean;
      is_latest_team: boolean;
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
  prior_rating: number | null;
  season_update: number | null;
  additive_profile_adjustment: number | null;
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
  offensive_edge: number | null;
  defensive_edge: number | null;
  composition_rating: number;
  composition_edge: number;
  matchup_bonus: number;
  context_edge: number;
  gestalt_score: number;
  actual_net_rating: number;
};

export type RosterMove = {
  player_id: number;
  player_name: string;
  source_team: string;
  target_team: string;
  move_type: "trade" | "signing" | "waiver" | "other";
  how_acquired: string | null;
  prior_season_minutes: number;
  projected_rating: number | null;
};

export type ExternalRosterArrival = {
  player_id: number;
  player_name: string;
  target_team: string;
  move_type: RosterMove["move_type"];
  how_acquired: string | null;
  school: string | null;
  country: string | null;
  is_rookie: boolean;
  projected_rating: number | null;
};

export type RosterMovesPayload = {
  current_season: string;
  prior_season: string;
  source_definition: string;
  teams: string[];
  moves: RosterMove[];
  external_arrivals: ExternalRosterArrival[];
  current_roster_count: number;
  returning_mover_count: number;
  new_or_unmatched_current_player_count: number;
};

export type MinutesProjectionPlayer = {
  player_id: number;
  player_name: string;
  position: string;
  age: number | null;
  team: string;
  prior_season_minutes: number;
  is_rating_fallback: boolean;
  is_baseline_rotation_candidate: boolean;
  has_prior_availability_state: boolean;
  has_prior_minutes_state: boolean;
  availability_probability: number;
  conditional_minutes_per_game: number;
  raw_expected_total_minutes: number;
  baseline_minute_share: number;
  baseline_minutes_per_game: number;
  projected_nail: number;
  rating_source: string;
  uses_incumbent_team_strength?: boolean;
  is_incumbent_team_strength_incumbent?: boolean;
};

export type IncumbentTeamStrength = {
  league_strength_fallback: number;
  team_strength_log_minutes_coefficient: number;
  teams: Array<{
    team: string;
    incumbent_weight: number;
    weighted_nail: number;
    incumbent_team_nail: number;
  }>;
};

export type MinutesProjectionPayload = {
  season: string;
  completed_season: string;
  model: string;
  initial_rotation_size: number;
  regulation_team_minutes: number;
  contract: string;
  teams: string[];
  players: MinutesProjectionPlayer[];
  incumbent_team_strength?: IncumbentTeamStrength;
  win_projection?: WinProjectionPayload;
};

export type WinProjectionTeam = {
  team: string;
  team_strength: number;
  betmgm_win_total?: number;
  win_total_p1: number;
  win_total_p5: number;
  win_total_p50: number;
  win_total_p95: number;
  win_total_p99: number;
  scheduled_wins: number;
  scheduled_games: number;
  unassigned_wins: number;
  projected_wins: number;
  projected_losses: number;
};

export type WinProjectionScheduledGame = {
  game_id: string;
  game_date: string;
  home_team: string;
  away_team: string;
  home_back_to_back: number;
  away_back_to_back: number;
};

export type WinProjectionPayload = {
  model: string;
  season: string;
  scheduled_game_count: number;
  scheduled_games_per_team: number;
  unassigned_regular_games_per_team: number;
  win_probability_scale: number;
  home_court: number;
  back_to_back: number;
  win_total_interval_trials: number;
  calibration: {
    calibration_season: string;
    holdout_season: string;
    calibration_beta: number;
    holdout_brier: number;
    holdout_log_loss: number;
    holdout_accuracy: number;
    production_beta: number;
  };
  market_win_totals?: {
    provider: string;
    as_of: string;
    source_url: string;
  };
  teams: WinProjectionTeam[];
  scheduled_games: WinProjectionScheduledGame[];
  contract: string;
};

export type ContextFeatureDetail = {
  kind: "generic" | "usage_concentration" | "top_two_assists";
  unit_value: number;
  opponent_value: number;
  difference: number;
  standard_deviation: number;
  standardized_difference: number;
  standardized_coefficient: number;
  raw_coefficient: number;
  unit_total?: number;
  opponent_total?: number;
  unit_top_players?: Array<{ player_name: string; value: number }>;
  opponent_top_players?: Array<{ player_name: string; value: number }>;
};

export type ContextFeature = {
  id: string;
  label: string;
  value: number;
  contribution: number;
  detail?: ContextFeatureDetail;
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
  model_form?: "compiled_linear_x3";
  unit_season: string;
  opponent_season: string;
  environment: "unit" | "neutral" | "opponent";
  environment_seasons: string[];
  context_source_seasons?: string[];
  unit: {
    additive_rating: number;
    offense_rating?: number | null;
    defense_rating?: number | null;
    players: Player[];
  };
  opponent: {
    additive_rating: number;
    offense_rating?: number | null;
    defense_rating?: number | null;
    players: Player[];
  };
  additive_margin: number;
  od_split_available?: boolean;
  offensive_player_edge?: number | null;
  defensive_player_edge?: number | null;
  contextual_adjustment: number;
  unit_composition_rating: number;
  opponent_composition_rating: number;
  portable_composition_margin: number;
  matchup_adjustment: number;
  base_predicted_net_rating: number;
  court: "neutral" | "unit_home" | "opponent_home";
  unit_back_to_back: boolean;
  opponent_back_to_back: boolean;
  home_court_adjustment: number;
  back_to_back_adjustment: number;
  schedule_adjustment: number;
  home_court_reference: number;
  back_to_back_reference: number;
  schedule_control_source_season_count: number;
  predicted_net_rating: number;
  predicted_win_pct: number;
  feature_contributions: ContextFeature[];
  composition_feature_contributions: ContextFeature[];
  matchup_feature_contributions: ContextFeature[];
  composition_response_curves: FeatureResponseCurve[];
  matchup_response_curves: FeatureResponseCurve[];
};
