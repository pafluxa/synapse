# run.py  – end-to-end:  tabular BERT  ➜  sphere classifier
import sys

from typing import List, Tuple

import argparse
import multiprocessing as mp
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

# ──────────────────────────────────────────────────────────────────────
#  Local imports – adjust package paths if needed
# ──────────────────────────────────────────────────────────────────────
from synapse.data.datasets import CSVDataset
from synapse.data.datasets import SequenceWindowDataset
from synapse.utils.config_parser import RunConfiguration
from synapse.models.transformers import MaskedTransformerAutoencoder
from synapse.training.trainers import MaskedEmbeddingTrainer
from synapse.training.trainers import TransformerAnomalyTrainer

# ----------------------------------------------------------------------
def main(argv=None):

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--ae_checkpoint", required=False)
    args = p.parse_args(argv)

    cfg = RunConfiguration.from_yaml(args.config)

    # === 1. CSV → Datasets -------------------------------------------
    dataset = CSVDataset(
        file_path=cfg.csv_data_path,
        numerical_cols=cfg.numerical_cols,
        categorical_cols=cfg.categorical_cols,
        max_workers=cfg.num_workers,
    )
    train_set, val_set, test_set = dataset.prepare_splits(
        test_size=cfg.test_fraction,
        val_size=cfg.val_fraction,
    )
    cfg.attach_dataset(train_set, val_set, test_set)

    best_ckpt = ''
    if args.ae_checkpoint is None:
        # === 2. Train MaskedEmbedding model ------------------------------
        trafo_trainer = MaskedEmbeddingTrainer(cfg)
        trafo_trainer.train()

        # Path to best auto-encoder checkpoint (created by the trainer)
        best_ckpt = Path(trafo_trainer.ckpt_dir) / "best.pt"

    else:
        best_ckpt = args.ae_checkpoint

    # === 3. Load encoder & extract codecs ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trafo = MaskedTransformerAutoencoder.load(
        best_ckpt,
        cfg,
        device=device
    )

    dataset = SequenceWindowDataset(
        transformer=trafo,
        file_path=cfg.csv_clf_data_path,
        numerical_cols=cfg.numerical_cols,
        categorical_cols=cfg.categorical_cols,
        label_col=cfg.label_col,
        max_workers=cfg.num_workers,
    )
    train_set, val_set, test_set = dataset.prepare_splits(
        test_size=cfg.test_fraction,
        val_size=cfg.val_fraction,
    )
    cfg.attach_dataset(train_set, val_set, test_set)

    # # === 4. Train Classifier -----------------------------------
    clf_trainer = TransformerAnomalyTrainer(cfg)
    clf_trainer.train()

if __name__ == '__main__':
    mp.set_start_method("spawn")
    sys.exit(main())
