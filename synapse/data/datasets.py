import copy

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from torch.utils.data import IterableDataset, Dataset, random_split
from sklearn.model_selection import train_test_split


class SyntheticDataset(Dataset):
    """Generates synthetic training data"""
    def __init__(self, num_samples, numerical_cols, categorical_cols):
        self.num_samples = num_samples
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self._categorical_cardinalities = np.random.randint(2, 17, size=(len(categorical_cols))).tolist()

        self.numerical_data = torch.randn(num_samples, len(self.numerical_cols))
        self.categorical_data = torch.stack([
            torch.randint(0, dim, (num_samples,)) for dim in self._categorical_cardinalities
        ], dim=1)

    def __len__(self):
        return len(self.numerical_data)

    def __getitem__(self, idx):
        return self.numerical_data[idx], self.categorical_data[idx]

    @property
    def categorical_dims(self):
        """Returns list of categorical features"""
        return self.categorical_cols

    @property
    def cardinalities(self):
        """Returns cardinalities of categorical features"""
        return self._categorical_cardinalities


class CSVDataset(IterableDataset):

    def __init__(self, csv_path, numerical_cols, categorical_cols, target_col=None, max_workers=4):
        self.csv_path = csv_path  # Store path instead of loading immediately
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.max_workers = max(max_workers, 1)

        # Initialize empty transforms
        self.scaler = MinMaxScaler()
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self._fitted = False
        self._categorical_cardinalities = None

        # Calculate length without loading full CSV
        with open(csv_path) as f:
            self.num_samples = sum(1 for _ in f) - 1  # Subtract header

    def _clean_data(self, df):
        """Clean a DataFrame chunk"""
        cols = set(self.numerical_cols + list(self.categorical_cols))
        if self.target_col:
            cols.add(self.target_col)
        return df[list(cols)].dropna()

    def _fit_transforms(self, train_df):
        """Fit scaler and encoder on training data"""
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

    def __len__(self):
           return self.num_samples

    def prepare_splits(self, test_size=0.2, val_size=0.1, random_state=42):
        # Read and clean training data only for fitting
        train_df = pd.read_csv(self.csv_path)
        train_df = self._clean_data(train_df)

        # Split data
        train_df, test_df = train_test_split(train_df, test_size=test_size, random_state=random_state)
        # train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), random_state=random_state)
        train_df, val_df = train_test_split(train_df,
                                           test_size=val_size,  # Directly 0.1
                                           random_state=random_state)
        # Fit transforms
        self._fit_transforms(train_df)

        # Create datasets with pre-computed row indices
        datasets = []
        for df in [train_df, val_df, test_df]:
            dataset = CSVDataset.__new__(CSVDataset)
            dataset.csv_path = self.csv_path
            dataset.numerical_cols = self.numerical_cols
            dataset.categorical_cols = self.categorical_cols
            dataset.target_col = self.target_col
            dataset.max_workers = self.max_workers
            dataset.scaler = copy.deepcopy(self.scaler)
            dataset.encoder = copy.deepcopy(self.encoder)
            dataset._fitted = True
            dataset._categorical_cardinalities = self._categorical_cardinalities
            dataset.row_indices = df.index.tolist()  # Store row indices instead of DataFrame
            dataset.num_samples = len(df)
            datasets.append(dataset)

        return tuple(datasets)

    def __iter__(self):
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
            numerical = torch.FloatTensor(self.scaler.transform(chunk[self.numerical_cols].values)) # if self.numerical_cols else None
            categorical = torch.LongTensor(self.encoder.transform(chunk[self.categorical_cols].values)) # if self.categorical_cols else None

            # targets = torch.FloatTensor(chunk[self.target_col].values) if self.target_col else None

            for i in range(len(chunk)):
                yield (numerical[i], categorical[i])
                #     (numerical[i], categorical[i]) if categorical is not None else numerical[i],
                #     targets[i] if targets is not None else None
                # )

    @property
    def categorical_dims(self):
        """Returns cardinalities of categorical features"""
        return list(self.categorical_cols.values())

    @property
    def cardinalities(self):
        """Returns cardinalities of categorical features"""
        if not self._fitted:
            raise RuntimeError("Cardinalities are only available after prepare_splits()")
        return list(self._categorical_cardinalities.values())


class ContrastiveSphereDataset(Dataset):
    """Dataset for contrastive rotation-based classification"""
    def __init__(self, codecs, rotation_augmenter, pairs_per_codec=5):
        """
        Args:
            codecs: Tensor of shape [num_samples, dim] from trained model
            rotation_augmenter: RotationAugmentation instance
            pairs_per_codec: Number of rotation/perturbed pairs per codec
        """
        self.codecs = codecs
        self.augmenter = rotation_augmenter
        self.pairs_per_codec = pairs_per_codec
        self.dim = codecs.size(1)

    def __len__(self):
        return len(self.codecs) * self.pairs_per_codec * 2  # *2 for rotated/perturbed

    def __getitem__(self, idx):
        # Determine if this sample should be rotated or perturbed
        is_rotation = idx % 2 == 0
        codec_idx = (idx // (self.pairs_per_codec * 2)) % len(self.codecs)
        codec = self.codecs[codec_idx]

        # Generate new matrix pair every pairs_per_codec samples
        if idx % (self.pairs_per_codec * 2) == 0:
            self.current_R, self.current_P = self.augmenter.generate_pair()

        # Apply transformation
        if is_rotation:
            transformed = codec @ self.current_R
            label = 1.0  # Positive sample (valid rotation)
        else:
            transformed = codec @ self.current_P
            label = 0.0  # Negative sample (invalid transformation)

        return transformed, label
