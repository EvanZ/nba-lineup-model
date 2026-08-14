"""Forward-safe profile tensors for the profile-aware Deep Sets model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from nba_lineup_model.modeling.neural_data import _encode_lineups
from nba_lineup_model.modeling.profile_token_mart import (
    DEFAULT_OUTPUT_DIR,
    TOKEN_FEATURE_COLUMNS,
    validate_profile_token_mart,
)


@dataclass(frozen=True)
class ProfileFeatureScaler:
    """Training-window standardization parameters for profile-token fields."""

    feature_columns: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    player_count: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Standardize a two-dimensional profile matrix."""

        if values.ndim != 2 or values.shape[1] != len(self.feature_columns):
            raise ValueError("Profile values do not match the scaler feature contract")
        return (values - self.means) / self.scales


@dataclass(frozen=True)
class SeasonProfileTable:
    """One standardized token table, stored once for one target season."""

    player_columns: dict[int, int]
    values: torch.Tensor


class ProfilePossessionTensorDataset(Dataset[dict[str, torch.Tensor]]):
    """Possession tensors with fixed player IDs and forward profile tokens."""

    def __init__(
        self,
        possessions: pd.DataFrame,
        player_columns: Mapping[int, int],
        profile_lookups: Mapping[str, Mapping[int, np.ndarray]],
    ) -> None:
        if possessions.empty:
            raise ValueError("Tensor possession dataset cannot be empty")
        if "season" not in possessions:
            raise ValueError("Profile Deep Sets possessions require a season column")
        mapping = {int(player_id): int(column) for player_id, column in player_columns.items()}
        self.offense_player_indices = torch.as_tensor(
            _encode_lineups(possessions["offense_player_ids"], mapping),
            dtype=torch.long,
        )
        self.defense_player_indices = torch.as_tensor(
            _encode_lineups(possessions["defense_player_ids"], mapping),
            dtype=torch.long,
        )
        if self.offense_player_indices.eq(0).any() or self.defense_player_indices.eq(0).any():
            raise ValueError("Profile Deep Sets cannot use an unknown player token")
        self.offense_profiles, self.defense_profiles = _lineup_profiles(
            possessions,
            profile_lookups,
        )
        self.home_offense_sign = torch.as_tensor(
            possessions["home_offense_sign"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        self.target = torch.as_tensor(
            possessions["target_offense_margin"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        self.row_index = torch.as_tensor(
            possessions.index.to_numpy(dtype=np.int64, copy=True),
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        offense = self.offense_player_indices[index]
        defense = self.defense_player_indices[index]
        return {
            "offense_player_indices": offense,
            "defense_player_indices": defense,
            "offense_profiles": self.offense_profiles[index],
            "defense_profiles": self.defense_profiles[index],
            "home_offense_sign": self.home_offense_sign[index],
            "target": self.target[index],
            "row_index": self.row_index[index],
        }


class LazyProfilePossessionTensorDataset(Dataset[dict[str, torch.Tensor]]):
    """Keep profile tokens compact and gather them once per minibatch.

    ``__getitem__`` remains useful for inspection and small tests. Production
    loaders must use :func:`profile_possession_loader`, which calls ``batch``
    on a vector of possession rows and avoids Python-level profile gathers for
    every player in every possession.
    """

    def __init__(
        self,
        possessions: pd.DataFrame,
        identity_player_columns: Mapping[int, int],
        profile_tables: Mapping[str, SeasonProfileTable],
        *,
        allow_missing_profiles: bool = False,
    ) -> None:
        if possessions.empty:
            raise ValueError("Tensor possession dataset cannot be empty")
        if "season" not in possessions:
            raise ValueError("Profile Deep Sets possessions require a season column")
        identity_mapping = {
            int(player_id): int(column)
            for player_id, column in identity_player_columns.items()
        }
        self.offense_player_indices = torch.as_tensor(
            _encode_lineups(possessions["offense_player_ids"], identity_mapping),
            dtype=torch.long,
        )
        self.defense_player_indices = torch.as_tensor(
            _encode_lineups(possessions["defense_player_ids"], identity_mapping),
            dtype=torch.long,
        )
        self.profile_tables = tuple(profile_tables.values())
        season_to_index = {season: index for index, season in enumerate(profile_tables)}
        season_values = possessions["season"].astype(str).to_numpy()
        if missing := sorted(set(season_values) - set(season_to_index)):
            raise ValueError(f"Profile tables are missing seasons: {missing}")
        self.season_indices = torch.as_tensor(
            [season_to_index[season] for season in season_values],
            dtype=torch.long,
        )
        self.offense_profile_indices = torch.empty_like(self.offense_player_indices)
        self.defense_profile_indices = torch.empty_like(self.defense_player_indices)
        for season, table_index in season_to_index.items():
            rows = np.flatnonzero(season_values == season)
            table = self.profile_tables[table_index]
            try:
                self.offense_profile_indices[rows] = torch.as_tensor(
                    _encode_lineups(
                        possessions.iloc[rows]["offense_player_ids"],
                        table.player_columns,
                    ),
                    dtype=torch.long,
                )
                self.defense_profile_indices[rows] = torch.as_tensor(
                    _encode_lineups(
                        possessions.iloc[rows]["defense_player_ids"],
                        table.player_columns,
                    ),
                    dtype=torch.long,
                )
            except KeyError as error:
                raise ValueError(
                    f"Profile token mart is missing player {error.args[0]} for {season}"
                ) from error
        if (
            not allow_missing_profiles
            and (
                self.offense_profile_indices.eq(0).any()
                or self.defense_profile_indices.eq(0).any()
            )
        ):
            raise ValueError("Profile token tables must reserve no lineup player as zero")
        self.missing_profile_slots = int(
            self.offense_profile_indices.eq(0).sum()
            + self.defense_profile_indices.eq(0).sum()
        )
        max_rows = max(table.values.shape[0] for table in self.profile_tables)
        feature_count = len(TOKEN_FEATURE_COLUMNS)
        self.profile_values = torch.zeros(
            (len(self.profile_tables), max_rows, feature_count),
            dtype=torch.float32,
        )
        for table_index, table in enumerate(self.profile_tables):
            self.profile_values[table_index, : table.values.shape[0]] = table.values
        self.home_offense_sign = torch.as_tensor(
            possessions["home_offense_sign"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        self.target = torch.as_tensor(
            possessions["target_offense_margin"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        self.row_index = torch.as_tensor(
            possessions.index.to_numpy(dtype=np.int64, copy=True),
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        table = self.profile_tables[int(self.season_indices[index])]
        offense = self.offense_profile_indices[index]
        defense = self.defense_profile_indices[index]
        return {
            "offense_player_indices": self.offense_player_indices[index],
            "defense_player_indices": self.defense_player_indices[index],
            "offense_profiles": table.values[offense],
            "defense_profiles": table.values[defense],
            "home_offense_sign": self.home_offense_sign[index],
            "target": self.target[index],
            "row_index": self.row_index[index],
        }

    def batch(self, rows: list[int] | torch.Tensor) -> dict[str, torch.Tensor]:
        """Vectorize all profile lookups for one minibatch of possession rows."""

        row_indices = torch.as_tensor(rows, dtype=torch.long)
        season_indices = self.season_indices[row_indices]
        offense_profile_indices = self.offense_profile_indices[row_indices]
        defense_profile_indices = self.defense_profile_indices[row_indices]
        return {
            "offense_player_indices": self.offense_player_indices[row_indices],
            "defense_player_indices": self.defense_player_indices[row_indices],
            "offense_profiles": self.profile_values[
                season_indices.unsqueeze(1), offense_profile_indices
            ],
            "defense_profiles": self.profile_values[
                season_indices.unsqueeze(1), defense_profile_indices
            ],
            "home_offense_sign": self.home_offense_sign[row_indices],
            "target": self.target[row_indices],
            "row_index": self.row_index[row_indices],
        }


def profile_possession_loader(
    dataset: LazyProfilePossessionTensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    generator: torch.Generator | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Return a loader that materializes each profile batch in one tensor gather."""

    return DataLoader(
        range(len(dataset)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        collate_fn=dataset.batch,
        persistent_workers=num_workers > 0,
    )


class ProfilePossessionDataModule(L.LightningDataModule):
    """Build profile tensors using moments fit only on the train-game window."""

    def __init__(
        self,
        possessions: pd.DataFrame,
        player_columns: Mapping[int, int],
        tokens: pd.DataFrame,
        *,
        train_game_ids: tuple[str, ...],
        validation_game_ids: tuple[str, ...] = (),
        test_game_ids: tuple[str, ...] = (),
        batch_size: int = 2_048,
        num_workers: int = 0,
        random_seed: int = 17,
    ) -> None:
        super().__init__()
        if batch_size < 1 or num_workers < 0 or random_seed < 0:
            raise ValueError("Invalid profile data-module runtime configuration")
        self.possessions = possessions
        self.player_columns = dict(player_columns)
        self.tokens = tokens
        self.train_game_ids = train_game_ids
        self.validation_game_ids = validation_game_ids
        self.test_game_ids = test_game_ids
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.scaler: ProfileFeatureScaler | None = None
        self.profile_tables: dict[str, SeasonProfileTable] | None = None
        self.train_dataset: LazyProfilePossessionTensorDataset | None = None
        self.validation_dataset: LazyProfilePossessionTensorDataset | None = None
        self.test_dataset: LazyProfilePossessionTensorDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        train_keys = player_season_keys_in_games(self.possessions, self.train_game_ids)
        self.scaler = fit_profile_feature_scaler_for_keys(self.tokens, train_keys)
        self.profile_tables = profile_token_tables(
            self.tokens,
            self.scaler,
            tuple(self.possessions["season"].astype(str).unique()),
        )
        game_ids = self.possessions["game_id"].astype(str)
        self.train_dataset = self._dataset(game_ids.isin(self.train_game_ids))
        self.validation_dataset = self._optional_dataset(
            game_ids.isin(self.validation_game_ids)
        )
        self.test_dataset = self._optional_dataset(game_ids.isin(self.test_game_ids))

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        if self.train_dataset is None:
            raise RuntimeError("Data module has not been set up")
        generator = torch.Generator().manual_seed(self.random_seed)
        return profile_possession_loader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=generator,
        )

    def val_dataloader(
        self,
    ) -> DataLoader[dict[str, torch.Tensor]] | list[DataLoader[dict[str, torch.Tensor]]]:
        if self.validation_dataset is None:
            return []
        return self._evaluation_loader(self.validation_dataset)

    def test_dataloader(
        self,
    ) -> DataLoader[dict[str, torch.Tensor]] | list[DataLoader[dict[str, torch.Tensor]]]:
        if self.test_dataset is None:
            return []
        return self._evaluation_loader(self.test_dataset)

    def _dataset(self, mask: pd.Series) -> LazyProfilePossessionTensorDataset:
        if self.profile_tables is None:
            raise RuntimeError("Profile tables are unavailable before setup")
        return LazyProfilePossessionTensorDataset(
            self.possessions.loc[mask],
            self.player_columns,
            self.profile_tables,
        )

    def _optional_dataset(self, mask: pd.Series) -> LazyProfilePossessionTensorDataset | None:
        if not mask.any():
            return None
        return self._dataset(mask)

    def _evaluation_loader(
        self,
        dataset: LazyProfilePossessionTensorDataset,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        return profile_possession_loader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def read_profile_tokens(
    season: str,
    *,
    profile_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """Read one target-season token population after validating the mart."""

    root = Path(profile_dir)
    validate_profile_token_mart(root)
    tokens = pd.read_parquet(root / "player_profile_tokens.parquet")
    output = tokens.loc[tokens["target_season"].astype(str).eq(season)].copy()
    if output.empty:
        raise ValueError(f"Profile token mart has no target-season rows for {season}")
    if output["player_id"].duplicated().any():
        raise ValueError(f"Profile token mart has duplicate player rows for {season}")
    missing = set(TOKEN_FEATURE_COLUMNS) - set(output)
    if missing:
        raise ValueError(f"Profile token mart is missing columns: {sorted(missing)}")
    values = output.loc[:, TOKEN_FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Profile token mart has non-finite features for {season}")
    return output


def player_ids_in_games(
    possessions: pd.DataFrame,
    game_ids: tuple[str, ...],
) -> tuple[int, ...]:
    """Return the unique players in a training-game window."""

    frame = possessions.loc[possessions["game_id"].astype(str).isin(game_ids)]
    if frame.empty:
        raise ValueError("Cannot fit a profile scaler without training possessions")
    return tuple(
        sorted(
            set().union(
                *frame["offense_player_ids"],
                *frame["defense_player_ids"],
            )
        )
    )


def player_season_keys_in_games(
    possessions: pd.DataFrame,
    game_ids: tuple[str, ...],
) -> tuple[tuple[str, int], ...]:
    """Return distinct season/player token keys in a training-game window."""

    frame = possessions.loc[possessions["game_id"].astype(str).isin(game_ids)]
    if frame.empty:
        raise ValueError("Cannot fit a profile scaler without training possessions")
    if "season" not in frame:
        raise ValueError("Profile Deep Sets possessions require a season column")
    return tuple(
        sorted(
            {
                (str(season), int(player_id))
                for season, offense, defense in zip(
                    frame["season"],
                    frame["offense_player_ids"],
                    frame["defense_player_ids"],
                    strict=True,
                )
                for player_id in [*offense, *defense]
            }
        )
    )


def fit_profile_feature_scaler(
    tokens: pd.DataFrame,
    player_ids: tuple[int, ...],
) -> ProfileFeatureScaler:
    """Fit per-feature moments only from players in a training-game window."""

    if not player_ids:
        raise ValueError("Profile scaler requires at least one training player")
    indexed = tokens.set_index("player_id", verify_integrity=True)
    missing = sorted(set(player_ids) - set(indexed.index.astype(int)))
    if missing:
        raise ValueError(f"Profile token mart is missing training players: {missing[:10]}")
    values = indexed.loc[list(player_ids), TOKEN_FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Profile scaler cannot fit non-finite values")
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales == 0.0] = 1.0
    return ProfileFeatureScaler(
        feature_columns=TOKEN_FEATURE_COLUMNS,
        means=means,
        scales=scales,
        player_count=len(player_ids),
    )


def fit_profile_feature_scaler_for_keys(
    tokens: pd.DataFrame,
    player_season_keys: tuple[tuple[str, int], ...],
) -> ProfileFeatureScaler:
    """Fit moments from only player-season tokens active in training games."""

    if not player_season_keys:
        raise ValueError("Profile scaler requires at least one training token")
    if "target_season" not in tokens:
        raise ValueError("Profile tokens require target_season for a multi-season fit")
    indexed = tokens.set_index(["target_season", "player_id"], verify_integrity=True)
    missing = [key for key in player_season_keys if key not in indexed.index]
    if missing:
        raise ValueError(f"Profile token mart is missing training tokens: {missing[:10]}")
    values = indexed.loc[list(player_season_keys), TOKEN_FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Profile scaler cannot fit non-finite values")
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales == 0.0] = 1.0
    return ProfileFeatureScaler(
        feature_columns=TOKEN_FEATURE_COLUMNS,
        means=means,
        scales=scales,
        player_count=len(player_season_keys),
    )


def profile_token_matrix(
    tokens: pd.DataFrame,
    player_columns: Mapping[int, int],
    scaler: ProfileFeatureScaler,
) -> np.ndarray:
    """Align standardized target-season profiles to a player vocabulary."""

    if scaler.feature_columns != TOKEN_FEATURE_COLUMNS:
        raise ValueError("Profile scaler feature columns do not match the mart contract")
    indexed = tokens.set_index("player_id", verify_integrity=True)
    player_ids = tuple(int(player_id) for player_id in player_columns)
    missing = sorted(set(player_ids) - set(indexed.index.astype(int)))
    if missing:
        raise ValueError(f"Profile token mart is missing lineup players: {missing[:10]}")
    raw = indexed.loc[list(player_ids), TOKEN_FEATURE_COLUMNS].to_numpy(dtype=float)
    standardized = scaler.transform(raw).astype(np.float32)
    matrix = np.zeros((len(player_columns) + 1, len(TOKEN_FEATURE_COLUMNS)), dtype=np.float32)
    for row, player_id in enumerate(player_ids):
        matrix[int(player_columns[player_id])] = standardized[row]
    return matrix


def profile_token_matrices(
    tokens: pd.DataFrame,
    player_columns: Mapping[int, int],
    scaler: ProfileFeatureScaler,
    seasons: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Align each target season's profiles to the shared player vocabulary."""

    output: dict[str, np.ndarray] = {}
    for season in seasons:
        rows = tokens.loc[tokens["target_season"].astype(str).eq(season)]
        if rows.empty:
            raise ValueError(f"Profile token mart has no target tokens for {season}")
        indexed = rows.set_index("player_id", verify_integrity=True)
        matrix = np.zeros(
            (len(player_columns) + 1, len(TOKEN_FEATURE_COLUMNS)),
            dtype=np.float32,
        )
        available_player_ids = [
            int(player_id) for player_id in player_columns if int(player_id) in indexed.index
        ]
        raw = indexed.loc[available_player_ids, TOKEN_FEATURE_COLUMNS].to_numpy(dtype=float)
        standardized = scaler.transform(raw).astype(np.float32)
        for row, player_id in enumerate(available_player_ids):
            matrix[int(player_columns[player_id])] = standardized[row]
        output[season] = matrix
    return output


def profile_token_lookups(
    tokens: pd.DataFrame,
    scaler: ProfileFeatureScaler,
    seasons: tuple[str, ...],
) -> dict[str, dict[int, np.ndarray]]:
    """Standardize profile tokens by season while preserving unseen-player profiles."""

    output: dict[str, dict[int, np.ndarray]] = {}
    for season in seasons:
        rows = tokens.loc[tokens["target_season"].astype(str).eq(season)]
        if rows.empty:
            raise ValueError(f"Profile token mart has no target tokens for {season}")
        if rows["player_id"].duplicated().any():
            raise ValueError(f"Profile token mart has duplicate player rows for {season}")
        values = scaler.transform(
            rows.loc[:, TOKEN_FEATURE_COLUMNS].to_numpy(dtype=float)
        ).astype(np.float32)
        output[season] = {
            int(player_id): values[index]
            for index, player_id in enumerate(rows["player_id"], start=0)
        }
    return output


def profile_token_tables(
    tokens: pd.DataFrame,
    scaler: ProfileFeatureScaler,
    seasons: tuple[str, ...],
) -> dict[str, SeasonProfileTable]:
    """Build one compact standardized table per season for lazy batch lookup."""

    output: dict[str, SeasonProfileTable] = {}
    for season in seasons:
        rows = tokens.loc[tokens["target_season"].astype(str).eq(season)]
        if rows.empty or rows["player_id"].duplicated().any():
            raise ValueError(f"Invalid profile token rows for {season}")
        values = scaler.transform(
            rows.loc[:, TOKEN_FEATURE_COLUMNS].to_numpy(dtype=float)
        ).astype(np.float32)
        table_values = np.zeros(
            (len(rows) + 1, len(TOKEN_FEATURE_COLUMNS)),
            dtype=np.float32,
        )
        table_values[1:] = values
        output[season] = SeasonProfileTable(
            player_columns={
                int(player_id): index
                for index, player_id in enumerate(rows["player_id"], start=1)
            },
            values=torch.as_tensor(table_values),
        )
    return output


def profile_data_module_factory(
    possessions: pd.DataFrame,
    player_columns: Mapping[int, int],
    tokens: pd.DataFrame,
    *,
    batch_size: int,
    num_workers: int,
):
    """Return a fold-aware Lightning data-module factory for neural training."""

    def build(
        train_game_ids: tuple[str, ...],
        validation_game_ids: tuple[str, ...],
        test_game_ids: tuple[str, ...],
        random_seed: int,
    ) -> ProfilePossessionDataModule:
        return ProfilePossessionDataModule(
            possessions,
            player_columns,
            tokens,
            train_game_ids=train_game_ids,
            validation_game_ids=validation_game_ids,
            test_game_ids=test_game_ids,
            batch_size=batch_size,
            num_workers=num_workers,
            random_seed=random_seed,
        )

    return build


def _lineup_profiles(
    possessions: pd.DataFrame,
    profile_lookups: Mapping[str, Mapping[int, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor]:
    offense = np.empty((len(possessions), 5, len(TOKEN_FEATURE_COLUMNS)), dtype=np.float32)
    defense = np.empty_like(offense)
    season_values = possessions["season"].astype(str).to_numpy()
    for season in np.unique(season_values):
        lookup = profile_lookups.get(str(season))
        if lookup is None:
            raise ValueError(f"Profile lookup is missing season {season}")
        rows = np.flatnonzero(season_values == season)
        try:
            offense[rows] = np.asarray(
                [
                    [lookup[int(player_id)] for player_id in lineup]
                    for lineup in possessions.iloc[rows]["offense_player_ids"]
                ],
                dtype=np.float32,
            )
            defense[rows] = np.asarray(
                [
                    [lookup[int(player_id)] for player_id in lineup]
                    for lineup in possessions.iloc[rows]["defense_player_ids"]
                ],
                dtype=np.float32,
            )
        except KeyError as error:
            raise ValueError(
                f"Profile token mart is missing lineup-player token {error.args[0]} for {season}"
            ) from error
    return torch.as_tensor(offense), torch.as_tensor(defense)
