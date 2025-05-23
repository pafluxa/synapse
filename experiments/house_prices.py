import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import multiprocessing as mp

from synapse.data.datasets import CSVDataset
from synapse.training.trainers import MaskedEmbeddingTrainer

# Extended config
class RunConfiguration:
    """Configuration class for the model and training"""
    def __init__(self, train_ds: CSVDataset): #, test_ds: CSVDataset, val_ds: CSVDataset):

        # Data configuration
        self.training_dataset = train_ds
        # self.testing_dataset = test_ds
        # self.validation_dataset = val_ds

        self.num_numerical = len(train_ds.numerical_cols)
        self.categorical_dims = [c for c in train_ds.cardinalities]
        self.num_categorical = len(self.categorical_dims)
        self.seq_len = self.num_numerical + self.num_categorical
        self.num_samples = len(train_ds)

        # Embedding configuration
        self.embedding_dim = 64

        # Transformer configuration
        self.num_heads= 8
        self.num_layers = 6
        self.dim_feedforward = 1024
        self.dropout = 0.1
        self.mask_ratio = 0.1

        # Encoder configuration
        self.codec_dim = 12
        self.width_mul = 1.0
        self.depth_mul = 1.0

        # MoE configuration
        self.num_experts = 8
        self.experts_hidden = [128, 64, 16]
        self.top_k = 2
        self.capacity_factor = 3.0

        # Training configuration
        self.num_workers = 4
        self.batch_size = 512
        self.num_epochs = 2000
        self.learning_rate = 1e-4
        self.weight_decay = 1e-8
        self.viz_dir = "./vis"

if __name__ == '__main__':

    mp.set_start_method('spawn')  # Critical for CUDA + multiprocessing

    numerical_cols = [
       'price',          # Continuous numerical
        'bedrooms',       # Discrete numerical (could also be treated as categorical if few unique values)
        'bathrooms',      # Continuous numerical (e.g., 2.25 bathrooms)
        'sqft_living',    # Continuous numerical
        'sqft_lot',       # Continuous numerical
        'floors',         # Discrete numerical (could be categorical if few unique values)
        'sqft_above',     # Continuous numerical
        'sqft_basement',  # Continuous numerical
        'yr_built',       # Discrete numerical (year)
    #    'yr_renovated',   # Discrete numerical (year, sparse)
    #    'lat',            # Continuous numerical (latitude)
    #    'long',           # Continuous numerical (longitude)
        'sqft_living15',  # Continuous numerical
        'sqft_lot15'      # Continuous numerical
    ]

    categorical_cols = [
    #    'id',             # Unique identifier (could drop for modeling)
    #    'date',           # Datetime (needs parsing, could be treated as categorical or numerical after feature engineering)
       'waterfront',     # Binary ('N' or 'Y', or NaN)
        'view',           # Likely ordinal (0, 1, 2, ...)
        'condition',      # Ordinal (e.g., 'Average', 'Very Good')
       'grade',          # Ordinal (numerical but may represent categories)
    #    'zipcode'         # Nominal (geographic category)
    ]

    # Initialize dataset (assuming CSV has these columns)
    dataset = CSVDataset(
        file_path="./data/house_pricing/house_prices.csv",
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        max_workers=4
    )

    # Create splits with consistent transforms
    train_set, val_set, test_set = dataset.prepare_splits(
        test_size=0.1,
        val_size=0.1
    )

    config = RunConfiguration(train_set)
    trainer = MaskedEmbeddingTrainer(config)
    trainer.train()
