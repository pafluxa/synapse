#!/usr/bin/env python
# infer.py  -------------------------------------------------------------
"""
CSV  ➜  TabularBERT.encode() ➜ SphereClassifier ➜  metrics + predictions.
"""
from typing import Tuple

import argparse
import csv
from pathlib import Path
import sys
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.cluster import HDBSCAN
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
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
from synapse.models.sphere_classifier import DirectionRadiusGatedMLP as SphereClassifier


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
        preds.append(z.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels) if labels else None
    return preds, labels.ravel()

def kmeans_label_alignment(
    X: np.ndarray,
    y: np.ndarray,
    visualize: bool = True,
    random_state: int = 0,
) -> Tuple[float, np.ndarray]:
    """
    Cluster a batch of D-dimensional vectors into **2 clusters** and
    measure how well *cluster index* (0/1) aligns with *binary labels*
    (0/1).

    Parameters
    ----------
    X : (N, D) numpy array of vectors
    y : (N,)   numpy int array with values {0,1}
    visualize : bool  – if True, shows a PCA-2D scatter with colours=cluster,
                        edge-colours=labels
    random_state : int  – passed to sklearn.KMeans for reproducibility

    Returns
    -------
    score : float  in [0, 1]
            (corr + 1) / 2, where corr = Pearson(cluster, label)
    clusters : (N,) int array of assigned cluster indices
    """
    if X.shape[0] != len(y):
        raise ValueError("X and y must have the same length")

    # --- k-means in original D ---------------------------------------
    kmeans = HDBSCAN()  #(n_clusters=2, n_init="auto", random_state=random_state)
    clusters = kmeans.fit_predict(X)               # values {0,1}
    print(np.unique(clusters))
    # --- correlation --------------------------------------------------
    corr, _ = pearsonr(clusters, y.ravel())                # --> [-1, 1]
    score = (corr + 1) / 2                         # --> [0, 1]

    # --- optional plot -----------------------------------------------
    if visualize:
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            X2 = pca.fit_transform(X)

            fig, ax = plt.subplots(figsize=(6, 5))
            scatter = ax.scatter(
                X2[:, 0],
                X2[:, 1],
                c=clusters,
                cmap="coolwarm",
                alpha=0.7,
                edgecolors=["k" if l == 0 else "w" for l in y],
                linewidths=0.7,
            )
            ax.set_title(f"k-means vs labels,  score = {score:.3f}")
            ax.set_xlabel("PCA-1")
            ax.set_ylabel("PCA-2")
            plt.legend(
                handles=scatter.legend_elements()[0],
                labels=["cluster 0", "cluster 1"],
                loc="best",
            )
            plt.tight_layout()
            plt.savefig('clusters.png')
        except ImportError:
            print("matplotlib or sklearn.decomposition missing – skipping plot.")

    return score, clusters


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
        in_dim=cfg.codec_dim,
        device=device,
    )

    # 3) inference
    embs, labels = run_inference(ae, sph, loader, device)
    print("mean norm for label = 0", np.average(np.linalg.norm(embs[labels == 0], axis=1)))
    print("mean norm for label = 1", np.average(np.linalg.norm(embs[labels == 1], axis=1)))
    clf = OneClassSVM(gamma='auto').fit(embs)
    preds = clf.predict(embs)
    preds[preds == -1] = 1
    preds[preds == 1] = 0
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
            row = [int(p)] if labels is None else [int(labels[i]), int(p)]
            w.writerow(row)
    print("Predictions saved to", out_path)


if __name__ == "__main__":
    sys.exit(main())
