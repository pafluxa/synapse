# run.py  – end-to-end:  tabular BERT  ➜  sphere classifier
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# ──────────────────────────────────────────────────────────────────────
#  Local imports – adjust package paths if needed
# ──────────────────────────────────────────────────────────────────────
from synapse.data.datasets import CSVDataset
from synapse.utils.config_parser import RunConfiguration
from synapse.training.trainers import MaskedEmbeddingTrainer
from synapse.training.trainers import RotationTripletTrainer
from synapse.models.auto_encoders import TabularBERT

# ----------------------------------------------------------------------
def codecs_from_dataset(
    dataset,
    model: TabularBERT,
    batch_size: int = 1024,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Pass the full dataset through the encoder and return a single [N,D] tensor.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_z = []
    with torch.no_grad():
        for x_num, x_cat in loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            z = model.encode(x_num, x_cat)           # [B, codec_dim]
            all_z.append(z.cpu())
    return torch.cat(all_z, dim=0)                   # [N, D]

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

    if args.ae_checkpoint is None:
        # === 2. Train MaskedEmbedding model ------------------------------
        bert_trainer = MaskedEmbeddingTrainer(cfg)
        bert_trainer.train()

        # Path to best auto-encoder checkpoint (created by the trainer)
        best_ckpt = Path(bert_trainer.ckpt_dir) / "best.pt"

    else:
        best_ckpt = args.ae_checkpoint

    # === 3. Load encoder & extract codecs ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = TabularBERT.load(best_ckpt, cfg, device=device)
    train_codecs = codecs_from_dataset(train_set, encoder, cfg.batch_size, device)
    val_codecs   = codecs_from_dataset(val_set,   encoder, cfg.batch_size, device)

    # === 4. Train SphereClassifier -----------------------------------
    sphere_trainer = RotationTripletTrainer(
        train_codecs=train_codecs,
        val_codecs=val_codecs,
        cfg=cfg,
    )
    sphere_trainer.fit()

if __name__ == '__main__':
    mp.set_start_method("spawn")
    sys.exit(main())
