# config/config_parser.py
import random
import string
from dataclasses import dataclass, field
from typing import Any, List
import yaml

from synapse.data.datasets import CSVDataset

def generate_run_id(length: int = 8) -> str:
    """
    Generates a random alphanumeric (A-Z, 0-9) string of given length.
    Default length is 8 characters.
    """
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=length))

@dataclass
class RunConfiguration:
    # Data-related
    csv_data_path: str = ''
    train_fraction: float = 0.7
    test_fraction: float = 0.2
    val_fraction: float = 0.1
    training_dataset: Any = None
    testing_dataset: Any = None
    validation_dataset: Any = None

    num_numerical: int = 0
    numerical_cols: List[str] = field(default_factory=list[str])

    categorical_cols: List[str] = field(default_factory=list[str])
    categorical_dims: List[int] = field(default_factory=list[int])
    num_categorical: int = 0

    num_features: int = 0
    num_samples: int = 0

    # Embedding
    embedding_dim: int = 32

    # Transformer
    num_heads: int = 4
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.15

    # Encoder
    codec_dim: int = 3
    bottleneck_arch: str = "mlp"

    # Training
    num_workers: int = 4
    batch_size: int = 128
    num_epochs: int = 1000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-10
    mask_prob: float = 0.125
    run_id: str = field(default_factory=generate_run_id)
    viz_dir: str = "./snapshots"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RunConfiguration":
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def attach_dataset(self,
        train_ds: CSVDataset,
        test_ds: CSVDataset,
        val_ds: CSVDataset):

        self.training_dataset = train_ds
        self.testing_dataset = test_ds
        self.validation_dataset = val_ds

        self.num_numerical = len(train_ds.numerical_cols)
        self.categorical_dims = [c for c in train_ds.cardinalities]
        self.num_categorical = len(self.categorical_dims)
        self.seq_len = self.num_numerical + self.num_categorical
        self.num_samples = len(train_ds)
