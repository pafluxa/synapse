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
        self.embedding_dim = 16

        # Transformer configuration
        self.n_heads= 1
        self.num_layers = 1
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.mask_ratio = 0.125

        # Bottleneck configuration
        self.bottleneck_dim = 3
        self.bottleneck_hidden = [32, 64]

        # MoE configuration
        self.num_experts = 16
        self.experts_hidden = [32, 64]
        self.moe_k = 2
        self.capacity_factor = 1.5

        # Training configuration
        self.num_workers = 4
        self.batch_size = 256
        self.num_epochs = 1000
        self.learning_rate = 1e-4
        self.weight_decay = 1e-6
        # self.mask_prob = 0.125
        self.viz_dir = "./viz"

if __name__ == '__main__':
    base_columns = [
        # "Name",
        # "Location",
        "Year",
        "Kilometers_Driven",
        "Fuel_Type",
        "Transmission",
        "Owner_Type",
        "Mileage",
        "Engine",
        "Power",
        "Seats",
        # "New_Price",
        # "Price"
    ]

    categorical_cols = [
        # "Name",
        # "Location",
        "Year",
        "Fuel_Type",
        "Transmission",
        "Owner_Type",
        "Engine",
        "Power",
        "Seats",
    ]

    numerical_cols = [col for col in base_columns if col not in categorical_cols]

    # Initialize dataset (assuming CSV has these columns)
    dataset = CSVDataset(
        csv_path="../../data/used_car_prices/train-data.csv",
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
