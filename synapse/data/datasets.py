"""Dataset classes for loading and processing tabular data.

This module provides a CSVDataset class that handles both numerical and categorical
features with proper scaling and encoding. The dataset supports lazy loading and
multi-worker processing.
"""

import copy
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

from synapse.types import (
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

        # Calculate length without loading full CSV
        with open(self.csv_path) as f:
            self.num_samples = sum(1 for _ in f) - 1  # Subtract header
        self.row_indices = np.arange(self.num_samples)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean a DataFrame chunk by removing NA values and selecting columns.

        Args:
            df: Input DataFrame chunk

        Returns:
            Cleaned DataFrame with only selected columns and no NA values
        """
        cols = set(self.numerical_cols + list(self.categorical_cols))
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
        self._fit_transforms(train_df)

        # Create datasets with pre-computed row indices
        datasets = []
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

            for i in range(len(chunk)):
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
