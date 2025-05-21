#!/usr/bin/env python
# infer.py  -------------------------------------------------------------
"""
CSV  ➜  TabularBERT.encode() ➜ SphereClassifier ➜  metrics + predictions.
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# sklearn for detailed metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# project imports (adjust PYTHONPATH as needed)
from synapse.utils.config_parser import RunConfiguration
from synapse.data.datasets import CSVDataset, CSVInferenceDataset
from synapse.models.auto_encoders import TabularBERT
from synapse.models.sphere_classifier import SphereClassifier


# ------------------------------------------------------------------ helpers
@torch.inference_mode()
def run_inference(ae, sph, loader, device):
    preds, labels = [], []
    for batch in loader:
        x_num, x_cat, y = batch
        labels.append(y.numpy())

        x_num, x_cat = x_num.to(device), x_cat.to(device)
        z = ae.encode(x_num, x_cat)
        p = sph.predict(z)               # 0 / 1
        preds.append(p.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels) if labels else None
    return preds, labels


# ------------------------------------------------------------------ CLI
def main(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--ae_ckpt", required=True)
    p.add_argument("--sphere_ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)

    # 1) config + dataset
    cfg = RunConfiguration.from_yaml(args.config)
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

    inference_dataset = CSVInferenceDataset(
        file_path=args.csv,
        numerical_cols=cfg.numerical_cols,
        categorical_cols=cfg.categorical_cols,
        label_col=cfg.label_col,
        max_workers=cfg.num_workers,
    )
    loader = DataLoader(inference_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 2) models
    ae = TabularBERT.load(args.ae_ckpt, cfg, device=device)
    sph = SphereClassifier.load(
        args.sphere_ckpt,
        codec_dim=cfg.codec_dim,
        hidden_dim=cfg.hidden_dim,
        device=device,
    )

    # 3) inference
    preds, labels = run_inference(ae, sph, loader, device)

    # 4) results
    if labels is not None:
        print("\n----------------  Classification report  ----------------")
        print(classification_report(labels, preds, digits=4))

        cm = confusion_matrix(labels, preds)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        print("Confusion matrix (rows=true, cols=pred):\n", cm)
        print(
            f"Accuracy={acc:.4f}  Precision={prec:.4f}  "
            f"Recall={rec:.4f}  F1={f1:.4f}"
        )
    else:
        print("No ground-truth label column found — skipping metrics.")

    # 5) write predictions
    out_path = Path(args.csv).with_suffix(".preds.csv")
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["prediction"] if labels is None else ["label", "prediction"]
        w.writerow(header)
        for i, p in enumerate(preds):
            print(p, labels[i])
            row = [int(p)] if labels is None else [int(labels[i]), int(p)]
            w.writerow(row)
    print("Predictions saved to", out_path)


if __name__ == "__main__":
    sys.exit(main())
