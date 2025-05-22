# config/config_parser.py
"""
Lightweight dataclass-based configuration parser
===============================================

• Loads a YAML into a `RunConfiguration` instance.
• Generates a unique run-ID if none provided.
• Provides `attach_dataset()` to enrich the config once
  the CSV splits are ready.

NOTE: keep this module import-light so that loading the YAML
doesn't trigger heavy dependencies.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple

import yaml

from synapse.data.datasets import CSVDataset, CSVLabeledDataset, SequenceWindowDataset

# --------------------------------------------------------------------- #
#  helper factories
# --------------------------------------------------------------------- #
def _generate_run_id(length: int = 8) -> str:
    """Random uppercase + digit string, e.g. '4F7C9K2M'."""
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))


def _default_ranges() -> List[Tuple[float, float]]:
    """Dummy (0-1) numerical min/max for when dataset not attached yet."""
    return []


# --------------------------------------------------------------------- #
#  main dataclass
# --------------------------------------------------------------------- #
@dataclass
class RunConfiguration:
    # ————————————————————————————————— DATA ————————————————————————————
    csv_data_path: str
    csv_clf_data_path: str
    test_fraction: float
    val_fraction: float

    numerical_cols: List[str]
    categorical_cols: List[str]
    label_col: str
    # ———————————————————————————— TRAINING ————————————————————————————
    num_workers: int
    batch_size: int
    num_epochs: int
    sph_num_epochs: int
    learning_rate: float
    weight_decay: float
    mask_prob: float

    # —————————————————————————— ARCHITECTURE ——————————————————————————
    embedding_dim: int
    num_heads: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    codec_dim: int
    hidden_dim: int
    hidden_dim: int # embedding size inside SphereClassifier
    lambda_bce: float # weight for BCE term
    patience: int
    bottleneck_arch: str

    # —————————————————————————— MISC / LOGGING ——————————————————————————
    viz_dir: str = "./snapshots"
    run_id: str = field(default_factory=_generate_run_id)

    # ————————————————— INTERNAL (filled by attach_dataset) ——————————
    training_dataset: CSVDataset | CSVLabeledDataset | SequenceWindowDataset = None
    validation_dataset: CSVDataset | CSVLabeledDataset | SequenceWindowDataset = None
    testing_dataset: CSVDataset | CSVLabeledDataset | SequenceWindowDataset= None

    num_numerical: int = 0
    numerical_ranges: List[Tuple[float, float]] = field(default_factory=_default_ranges)
    numerical_depths: List[int] = field(default_factory=list[int])

    num_categorical: int = 0
    categorical_dims: List[int] = field(default_factory=list[int])

    num_features: int = 0
    num_samples: int = 0

    # -----------------------------------------------------------------
    # factory
    # -----------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfiguration":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    # -----------------------------------------------------------------
    # to be called *after* you build the dataset splits
    # -----------------------------------------------------------------
    def attach_dataset(
        self,
        train_ds: CSVDataset | CSVLabeledDataset | SequenceWindowDataset,
        val_ds: CSVDataset | CSVLabeledDataset | SequenceWindowDataset,
        test_ds: CSVDataset | CSVLabeledDataset | SequenceWindowDataset,
        default_depth: int = 8,
    ) -> None:
        """Populate fields that depend on the concrete datasets."""
        self.training_dataset = train_ds
        self.validation_dataset = val_ds
        self.testing_dataset = test_ds

        # numerical
        self.num_numerical = len(train_ds.numerical_cols)
        self.numerical_ranges = [(0.0, 1.0)] * self.num_numerical
        self.numerical_depths = [default_depth] * self.num_numerical

        # categorical
        self.categorical_dims = list(train_ds.cardinalities)
        self.num_categorical = len(self.categorical_dims)

        # totals
        self.num_features = self.num_numerical + self.num_categorical
        self.num_samples = len(train_ds)
