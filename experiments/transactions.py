import multiprocessing as mp

from synapse.data.datasets import CSVDataset
from synapse.training.trainers import MaskedEmbeddingTrainer

# Extended config
class RunConfiguration:
    """Configuration class for the model and training"""
    def __init__(self, train_ds: CSVDataset, test_ds: CSVDataset, val_ds: CSVDataset):

        # Data configuration
        self.training_dataset = train_ds
        self.testing_dataset = test_ds
        self.validation_dataset = val_ds

        self.num_numerical = len(train_ds.numerical_cols)
        self.categorical_dims = [c for c in train_ds.cardinalities]
        self.num_categorical = len(self.categorical_dims)
        self.seq_len = self.num_numerical + self.num_categorical
        self.num_samples = len(train_ds)

        # Embedding configuration
        self.embedding_dim = 32

        # Transformer configuration
        self.num_heads= 8
        self.num_layers = 8
        self.dim_feedforward = 1024
        self.dropout = 0.15

        # Bottleneck configuration
        self.codec_dim = 3

        # Training configuration
        self.num_workers = 4
        self.batch_size = 1024
        self.num_epochs = 1000
        self.learning_rate = 1e-4
        self.weight_decay = 1e-9
        self.mask_prob = 0.25
        self.viz_dir = "./viz"

if __name__ == "__main__":

    mp.set_start_method('spawn')  # Critical for CUDA + multiprocessing

    column_types = {
        # "accountNumber": "numerical",
        # "customerId": "numerical",
        "creditLimit": "categorical",
        "availableMoney": "numerical",
        "transactionAmount": "numerical",
        "merchantName": "categorical",
        "acqCountry": "categorical",
        "merchantCountryCode": "categorical",
        # "posEntryMode": "categorical",
        # "posConditionCode": "categorical",
        "merchantCategoryCode": "categorical",
        "cardCVV": "numerical",
        "enteredCVV": "numerical",
        "cardLast4Digits": "numerical",
        "transactionType": "categorical",
        "currentBalance": "numerical",
        "cardPresent": "categorical",
        "expirationDateKeyInMatch": "categorical",
        "transactionDateTime_year": "categorical",
        "transactionDateTime_month": "categorical",
        "transactionDateTime_day": "categorical",
        "transactionDateTime_hour": "categorical",
        "transactionDateTime_minute": "categorical",
        "transactionDateTime_second": "categorical",
        "currentExpDate_year": "categorical",
        "currentExpDate_month": "categorical",
        "currentExpDate_day": "categorical",
        # "accountOpenDate_year": "categorical",
        # "accountOpenDate_month": "categorical",
        # "accountOpenDate_day": "categorical",
        # "dateOfLastAddressChange_year": "categorical",
        # "dateOfLastAddressChange_month": "categorical",
        # "dateOfLastAddressChange_day": "categorical"
    }
    categorical_cols = [colname for colname,coltype in column_types.items() if coltype == 'categorical']
    numerical_cols = [colname for colname,coltype in column_types.items() if coltype == 'numerical']

    # Initialize dataset (assuming CSV has these columns)
    dataset = CSVDataset(
        file_path="./data/transactions/small_nofraud.csv",
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        max_workers=8
    )

    # Create splits with consistent transforms
    train_set, val_set, test_set = dataset.prepare_splits(
        val_size=0.2,
        test_size=0.1,
    )

    # train_set = SyntheticDataset(1000000, numerical_cols[0:4], categorical_cols[0:6])
    config = RunConfiguration(train_set, val_ds=val_set, test_ds=test_set)
    trainer = MaskedEmbeddingTrainer(config)
    trainer.train()
