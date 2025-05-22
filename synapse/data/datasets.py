"""Dataset classes for loading and processing tabular data.

This module provides a CSVDataset class that handles both numerical and categorical
features with proper scaling and encoding. The dataset supports lazy loading and
multi-worker processing.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

from typing import (
    Dict, List, Optional, Tuple, Union, Iterator,
    Sequence, Any, Iterable
)

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from torch.utils.data import IterableDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from scipy.stats import special_ortho_group

from synapse.types.torch import (
    FloatTensor,
    LongTensor,
    Tensor1D,
    Tensor2D,
    Device
)


class CSVDataset(IterableDataset):
    """Lazy-loading CSV dataset with numerical and categorical feature support.

    Features:
        - Lazy loading with chunked processing
        - Multi-worker support
        - Automatic scaling of numerical features
        - Ordinal encoding of categorical features
        - Train/val/test split preparation

    Args:
        file_path: Path to CSV file
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        max_workers: Maximum number of workers for parallel loading
    """

    def __init__(
        self,
        file_path: str,
        numerical_cols: List[str],
        categorical_cols: List[str],
        max_workers: int = 4
    ) -> None:
        """Initialize the dataset."""
        self.csv_path = file_path  # Store path instead of loading immediately
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.max_workers = max(max_workers, 1)

        # Initialize empty transforms
        self.scaler = MinMaxScaler()
        self.encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        self._fitted = False
        self._categorical_cardinalities: Dict[str, int] = {}
        self.label_col: str | None = None
        # Calculate length without loading full CSV
        with open(self.csv_path) as f:
            self.num_samples = sum(1 for _ in f) - 1  # Subtract header
        self.row_indices: List[int] = np.arange(self.num_samples).tolist()

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean a DataFrame chunk by removing NA values and selecting columns.

        Args:
            df: Input DataFrame chunk

        Returns:
            Cleaned DataFrame with only selected columns and no NA values
        """
        if self.label_col is None:
            cols = set(self.numerical_cols + list(self.categorical_cols))
        else:
            cols = set(self.numerical_cols + list(self.categorical_cols) + [self.label_col,])

        return df[list(cols)].dropna()

    def _fit_transforms(self, train_df: pd.DataFrame) -> None:
        """Fit scaler and encoder on training data.

        Args:
            train_df: Training DataFrame used for fitting transforms
        """
        if len(self.numerical_cols) > 0:
            self.scaler.fit(train_df[self.numerical_cols].values)

        if len(self.categorical_cols) > 0:
            # Calculate cardinalities before fitting encoder
            self._categorical_cardinalities = {
                col: train_df[col].nunique()
                for col in self.categorical_cols
            }
            self.encoder.fit(train_df[self.categorical_cols].values)

        self._fitted = True

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def prepare_splits(
        self,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: Optional[int] = 42
    ) -> Tuple['CSVDataset', 'CSVDataset', 'CSVDataset']:
        """Prepare train/validation/test splits.

        Args:
            test_size: Fraction of data to use for test set
            val_size: Fraction of data to use for validation set
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Read and clean training data only for fitting
        main_df = pd.read_csv(self.csv_path)
        main_df = self._clean_data(main_df)

        # Split data
        train_df, test_val_df = train_test_split(
            main_df,
            test_size=(test_size + val_size),
            random_state=random_state
        )
        test_df, val_df = train_test_split(
            test_val_df,
            test_size=val_size/(test_size + val_size),
            random_state=random_state
        )

        # Fit transforms
        self._fit_transforms(main_df)

        # Create datasets with pre-computed row indices
        datasets: List[CSVDataset] = []
        for df in [train_df, val_df, test_df]:
            dataset = CSVDataset.__new__(CSVDataset)
            dataset.csv_path = self.csv_path
            dataset.numerical_cols = self.numerical_cols
            dataset.categorical_cols = self.categorical_cols
            dataset.max_workers = self.max_workers
            dataset.scaler = copy.deepcopy(self.scaler)
            dataset.encoder = copy.deepcopy(self.encoder)
            dataset._fitted = True
            dataset._categorical_cardinalities = self._categorical_cardinalities
            dataset.row_indices = df.index.tolist()  # Store row indices
            dataset.num_samples = len(df)
            dataset.label_col = self.label_col
            datasets.append(dataset)

        return tuple(datasets)  # type: ignore

    def __iter__(self) -> Iterator[
        Tuple[
            Tuple[Tensor1D, LongTensor],
            Tuple[Tensor1D, LongTensor]
        ]
    ]:
        """Iterate through the dataset with support for multi-processing.

        Yields:
            Tuple containing:
                - Tuple of (numerical_features, numerical_mask_indices)
                - Tuple of (categorical_features, categorical_mask_indices)
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single process
            indices = self.row_indices
        else:  # Multi-process
            per_worker = len(self.row_indices) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = (worker_id + 1) * per_worker if worker_id < worker_info.num_workers - 1 else len(self.row_indices)
            indices = self.row_indices[start:end]

        # Read CSV in chunks
        for chunk in pd.read_csv(self.csv_path, chunksize=1024):
            chunk = self._clean_data(chunk)
            chunk = chunk[chunk.index.isin(indices)]

            if len(chunk) == 0:
                continue

            # Batch transform
            numerical = torch.FloatTensor(
                self.scaler.transform(
                    chunk[self.numerical_cols].values)
            )
            categorical = torch.LongTensor(
                self.encoder.transform(chunk[self.categorical_cols].values)
            )

            idx = list(range(len(chunk)))
            for i in idx:
                yield numerical[i], categorical[i]

    @property
    def categorical_dims(self) -> List[int]:
        """Returns cardinalities of categorical features.

        Returns:
            List of cardinalities for each categorical feature

        Raises:
            RuntimeError: If called before prepare_splits()
        """
        if not self._fitted:
            raise RuntimeError("Cardinalities are only available after prepare_splits()")
        return list(self._categorical_cardinalities.values())

    @property
    def cardinalities(self) -> List[int]:
        """Alias for categorical_dims."""
        return self.categorical_dims


class CSVLabeledDataset(CSVDataset):
    """
    Same as CSVDataset but yields **(x_num, x_cat, label)**.

    Parameters
    ----------
    file_path        : str | Path
    numerical_cols   : list[str]
    categorical_cols : list[str]
    label_col        : str                 – column name with ground-truth (0/1)
    max_workers      : int                 – passed to parent constructor
    label_dtype      : torch.dtype         – default torch.long
    """

    def __init__(
        self,
        file_path: str | Path,
        numerical_cols: List[str],
        categorical_cols: List[str],
        label_col: str,
        max_workers: int = 1,
        label_dtype: torch.dtype = torch.long,
    ) -> None:
        # build everything the parent needs
        super().__init__(
            file_path=file_path,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            max_workers=max_workers,
        )

        self.label_col = label_col
        # read the single label column once (cheap compared to full preprocessing)
        cols = self.numerical_cols + self.categorical_cols + [self.label_col,]
        self.df: pd.DataFrame = pd.read_csv(file_path, usecols=cols)
        if label_col not in self.df.columns:
            raise ValueError(f"Label column '{label_col}' not found in CSV.")

    def prepare_splits(
        self,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: Optional[int] = 42
    ) -> Tuple['CSVDataset', 'CSVDataset', 'CSVDataset']:
        """Prepare train/validation/test splits.

        Args:
            test_size: Fraction of data to use for test set
            val_size: Fraction of data to use for validation set
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Read and clean training data only for fitting
        main_df = self._clean_data(self.df)

        # Split data
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=(test_size + val_size),
            random_state=0
        )
        (train_index, test_val_index) = next(sss.split(main_df[self.numerical_cols + self.categorical_cols], main_df[[self.label_col]]))
        train_df = main_df.loc[train_index, :].reset_index(drop=True)
        test_val_df = main_df.loc[test_val_index, :].reset_index(drop=True)
        # train_df, test_val_df = train_test_split(
        #     main_df,
        #     test_size=(test_size + val_size),
        #     random_state=random_state,
        #     stratify=[self.label_col,]
        # )
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_size/(test_size + val_size),
            random_state=0
        )
        (test_index, val_index) = next(sss.split(test_val_df[self.numerical_cols + self.categorical_cols], test_val_df[[self.label_col]]))
        test_df = test_val_df.loc[test_index, :].reset_index(drop=True)
        val_df = test_val_df.loc[val_index, :].reset_index(drop=True)
        # train_test_split(
        #     test_val_df,
        #     test_size=val_size/(test_size + val_size),
        #     random_state=random_state,
        #     stratify=[self.label_col,]
        # )

        # Fit transforms
        self._fit_transforms(main_df)

        # Create datasets with pre-computed row indices
        datasets: List[CSVLabeledDataset] = []
        for df in [train_df, val_df, test_df]:
            dataset = CSVLabeledDataset.__new__(CSVLabeledDataset)
            dataset.csv_path = self.csv_path
            dataset.df = df.copy()
            dataset.numerical_cols = self.numerical_cols
            dataset.categorical_cols = self.categorical_cols
            dataset.max_workers = self.max_workers
            dataset.scaler = copy.deepcopy(self.scaler)
            dataset.encoder = copy.deepcopy(self.encoder)
            dataset._fitted = True
            dataset._categorical_cardinalities = self._categorical_cardinalities
            dataset.row_indices = df.index.tolist()  # Store row indices
            dataset.num_samples = len(df)
            dataset.label_col = self.label_col
            datasets.append(dataset)

        return tuple(datasets)  # type: ignore

    # -----------------------------------------------------------------
    def __iter__(self):

        # Batch transform
        numerical = torch.FloatTensor(
            self.scaler.transform(
                self.df[self.numerical_cols].values)
        )
        categorical = torch.LongTensor(
            self.encoder.transform(
                self.df[self.categorical_cols].values)
        )

        labels = torch.FloatTensor(
            self.df[[self.label_col,]].values
        )

        for x_num, x_cat, y in zip(numerical, categorical, labels):
         yield x_num, x_cat, y   # parent gives the features


class SequenceWindowDataset(CSVLabeledDataset):
    """
    Dataset yielding random-length windows over a CSV file of raw data,
    applying preprocessing and using a Transformer to compute embeddings on the fly.

    Args:
        csv_path: path to CSV file containing raw data
        transformer: model with an `encode(x_num, x_cat) -> embeddings` method
        numerical_cols: list of column names for numerical features
        categorical_cols: list of column names for categorical features
        label_col: name of the label column (0/1)
        len_min: minimum window length
        len_max: maximum window length
    """
    def __init__(
        self,
        transformer: nn.Module,
        file_path: str | Path,
        numerical_cols: List[str],
        categorical_cols: List[str],
        label_col: str,
        max_workers: int = 1,
        label_dtype: torch.dtype = torch.long,
        len_min: int = 64,
        len_max: int = 256
    ) -> None:
        # build everything the parent needs
        super().__init__(
            file_path=file_path,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            max_workers=max_workers,
            label_col=label_col
        )
        # Transformer for on-the-fly embedding
        self.transformer: nn.Module = transformer.eval()

        # Columns configuration
        self.len_min = len_min
        self.len_max = len_max

        self.N = len(self.df)


    def prepare_splits(
        self,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: Optional[int] = 42
    ) -> Tuple['CSVDataset', 'CSVDataset', 'CSVDataset']:
        """Prepare train/validation/test splits.

        Args:
            test_size: Fraction of data to use for test set
            val_size: Fraction of data to use for validation set
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Read and clean training data only for fitting
        main_df = self._clean_data(self.df)

        # Split data
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=(test_size + val_size),
            random_state=0
        )
        (train_index, test_val_index) = next(sss.split(main_df[self.numerical_cols + self.categorical_cols], main_df[[self.label_col]]))
        train_df = main_df.loc[train_index, :].reset_index(drop=True)
        test_val_df = main_df.loc[test_val_index, :].reset_index(drop=True)
        # train_df, test_val_df = train_test_split(
        #     main_df,
        #     test_size=(test_size + val_size),
        #     random_state=random_state,
        #     stratify=[self.label_col,]
        # )
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_size/(test_size + val_size),
            random_state=0
        )
        (test_index, val_index) = next(sss.split(test_val_df[self.numerical_cols + self.categorical_cols], test_val_df[[self.label_col]]))
        test_df = test_val_df.loc[test_index, :].reset_index(drop=True)
        val_df = test_val_df.loc[val_index, :].reset_index(drop=True)
        # train_test_split(
        #     test_val_df,
        #     test_size=val_size/(test_size + val_size),
        #     random_state=random_state,
        #     stratify=[self.label_col,]
        # )

        # Fit transforms
        self._fit_transforms(main_df)

        # Create datasets with pre-computed row indices
        datasets: List[SequenceWindowDataset] = []
        for df in [train_df, val_df, test_df]:
            dataset = SequenceWindowDataset.__new__(SequenceWindowDataset)
            dataset.transformer = self.transformer
            dataset.csv_path = self.csv_path
            dataset.df = df.copy()
            dataset.numerical_cols = self.numerical_cols
            dataset.categorical_cols = self.categorical_cols
            dataset.max_workers = self.max_workers
            dataset.scaler = copy.deepcopy(self.scaler)
            dataset.encoder = copy.deepcopy(self.encoder)
            dataset._fitted = True
            dataset._categorical_cardinalities = self._categorical_cardinalities
            dataset.row_indices = df.index.tolist()  # Store row indices
            dataset.num_samples = len(df)
            dataset.label_col = self.label_col
            dataset.len_min = self.len_min
            dataset.len_max = self.len_max
            dataset.N = len(df)
            datasets.append(dataset)

        return tuple(datasets)  # type: ignore

    def __iter__(self):
        has_anomaly = False
        while not has_anomaly:
            # Sample a random window length
            L = np.random.randint(self.len_min, self.len_max)
            start = np.random.randint(0, self.N - L)
            end = start + L

            # Batch transform
            numerical = torch.FloatTensor(
                self.scaler.transform(
                    self.df[self.numerical_cols].values[start:end, :])
            ).to(next(self.transformer.parameters()).device)
            categorical = torch.LongTensor(
                self.encoder.transform(
                    self.df[self.categorical_cols].values[start:end, :])
            ).to(next(self.transformer.parameters()).device)

            labels = torch.FloatTensor(
                self.df[[self.label_col,]].values[start:end, :]
            ).to(next(self.transformer.parameters()).device)

            # ensure there is at least one anomaly in the sequence
            has_anomaly = (labels == 1).sum().item() > 0

        for x_num, x_cat, y_lbl in zip(numerical, categorical, labels):
            with torch.no_grad():
                emb = self.transformer.encode(x_num.unsqueeze(0), x_cat.unsqueeze(0))
            emb = emb.squeeze(0)

            yield emb, y_lbl


class RotationPairDataset(IterableDataset):
    """
    IterableDataset that *streams* (R, P) matrices where

        R : true rotation      shape [D, D]
        P : perturbed rotation shape [D, D]

    One item  ==  one **pair** (not a batch). Let DataLoader’s
    `batch_size` collate them into [B, D, D].

    Parameters
    ----------
    dim            : int         – matrix dimension (D)
    n_samples      : int | None  – total samples (None = infinite)
    epsilon        : float       – perturb noise factor
    min_singular   : float
    max_singular   : float
    device         : 'cpu' | 'cuda' | torch.device
    """
    def __init__(
        self,
        dim: int,
        n_samples: int | None = None,
        epsilon: float = 0.05,
        min_singular: float = 0.01,
        max_singular: float = 10.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.dim = dim
        self.n_samples = n_samples
        self.epsilon = epsilon
        self.min_sv = min_singular
        self.max_sv = max_singular
        self.device = torch.device(device)

    def _random_rotation(self) -> torch.Tensor:
        R = torch.from_numpy(special_ortho_group.rvs(self.dim)).float()
        return R

    def _perturb(self, R: torch.Tensor) -> torch.Tensor:
        U, S, Vh = torch.linalg.svd(R)
        S = torch.clamp(S * (1 + torch.randn_like(S) * self.epsilon),
                        self.min_sv, self.max_sv)
        return (U @ torch.diag(S) @ Vh)

    # ---------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Each worker gets its own iterator with a disjoint slice.
        Random seeds are offset by the worker id for independence.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:                      # single-process
            start, step = 0, 1
        else:                                        # multi-worker
            start = worker_info.id
            step = worker_info.num_workers
            # modify RNG seed for NumPy & Torch
            np.random.seed(np.random.get_state()[1][0] + worker_info.id)
            torch.manual_seed(torch.initial_seed() + worker_info.id)

        idx = start
        produced = 0

        while self.n_samples is None or produced < self.n_samples:
            R = self._random_rotation()
            P = self._perturb(R)
            yield R, P

            produced += 1
            idx += step
            if self.n_samples is not None and idx >= self.n_samples:
                break
