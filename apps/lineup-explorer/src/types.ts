export type Player = {
  player_id: number;
  player_name: string;
  team: string;
  position: string;
  rapm: number;
  possessions: number;
  games: number;
  profile_source: string;
  profile_imputed: number;
  profile_replacement_weight: number;
  three_pm_per_100: number;
  assists_per_100: number;
  usage_per_100: number;
  offensive_rebounds_per_100: number;
  defensive_rebounds_per_100: number;
};

export type ContextFeature = {
  id: string;
  label: string;
  value: number;
  contribution: number;
};

export type Matchup = {
  season: string;
  run_id: string;
  retrospective: boolean;
  unit: { additive_rating: number; players: Player[] };
  opponent: { additive_rating: number; players: Player[] };
  additive_margin: number;
  contextual_adjustment: number;
  contextual_intercept: number;
  predicted_net_rating: number;
  feature_contributions: ContextFeature[];
};
