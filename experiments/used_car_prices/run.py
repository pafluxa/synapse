import multiprocessing as mp

from synapse.data.datasets import CSVDataset
from synapse.utils.config_parser import RunConfiguration
from synapse.training.trainers import MaskedEmbeddingTrainer


if __name__ == '__main__':

    mp.set_start_method('spawn')

    cfg = RunConfiguration.from_yaml('./experiments/used_car_prices/config.yaml')

    # Initialize dataset (assuming CSV has these columns)
    dataset = CSVDataset(
        file_path=cfg.csv_data_path,
        numerical_cols=cfg.numerical_cols,
        categorical_cols=cfg.categorical_cols,
        max_workers=4
    )

    # Create splits with consistent transforms
    train_set, val_set, test_set = dataset.prepare_splits(
        test_size=cfg.test_fraction,
        val_size=cfg.val_fraction
    )

    cfg.attach_dataset(train_set, val_set, test_set)
    trainer = MaskedEmbeddingTrainer(cfg)
    trainer.train()
