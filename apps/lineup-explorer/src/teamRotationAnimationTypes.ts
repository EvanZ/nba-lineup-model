export type RotationAnimationPlayer = { id: number; name: string };
export type RotationAnimationFrame = {
  date: string;
  label: string;
  game: number;
  order: number[];
  matrix: (number | null)[][];
  minutes: (number | null)[];
  source: { uri: string; sha256: string };
};
export type RotationAnimationData = {
  kind: "saved_team_rotation_history";
  team: string;
  season: string;
  players: RotationAnimationPlayer[];
  frames: RotationAnimationFrame[];
  through_game_date_pacific: string;
};
export const cubicInOut = (t: number) => t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
