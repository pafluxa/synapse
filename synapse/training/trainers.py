# -------------------------------------------------------------------------
# MaskedEmbeddingTrainer  ✧  v2
# – Adds: early stopping  +  checkpointing with best-model flag
# -------------------------------------------------------------------------
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from synapse.utils.config_parser import RunConfiguration
from synapse.models.transformers import MaskedTransformerAutoencoder
from synapse.models.transformers import TransformerAnomalyDetector
from synapse.utils.visuals import SnapshotGenerator


def focal_binary_cross_entropy(
    probs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Focal loss for binary classification when the network already outputs
    probabilities (after Sigmoid).

    Parameters
    ----------
    probs : Tensor [B] or [B,1]  – model outputs ∈ (0,1)
    targets : Tensor [B]         – {0,1} ground-truth labels
    alpha : float                – class-balancing weight (0.25 for positives)
    gamma : float                – focusing parameter (>0)
    reduction : "mean"|"sum"|"none"
    eps : float                  – numeric stability

    Returns
    -------
    loss : Tensor (scalar unless reduction="none")
    """
    probs = probs.clamp(eps, 1.0 - eps)           # avoid log(0)
    targets = targets.float()

    p_t = probs * targets + (1 - probs) * (1 - targets)  # prob of true class
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    loss = -alpha_t * (1 - p_t) ** gamma * torch.log(p_t)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss  # "none"

def binary_metrics_from_logits(logits: torch.Tensor,
                               labels: torch.Tensor,
                               threshold: float = 0.5):
    """
    logits : shape [N] in [0, 1]
    labels : {0,1} ground-truth
    """
    preds = (logits >= threshold).long()
    correct = (preds == labels.long()).sum().item()
    acc     = correct / len(labels)

    tp = ((logits > threshold).float() * (labels > threshold).float()).sum().item()
    fp = ((logits < threshold).float() * (labels < threshold).float()).sum().item()
    fn = ((logits < threshold).float() * (labels > threshold).float()).sum().item()

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


class MaskedEmbeddingTrainer:
    """Trainer for the MaskedTransformerAutoencoder.

    Fixes compared to the original version
    --------------------------------------
    1. **Metric keys** match the model's output (`ce`, `mse`).
    2. Added a **cosine LR scheduler**; LR now decays every epoch.
    3. Validation uses a **deterministic mask** so the loss is stable.
    4. `shuffle=True` is avoided automatically if the dataset is an
       ``IterableDataset`` (PyTorch forbids it).
    5. Removed unused *w1/w2* annealing logic; the model already blends CE
       and MSE internally.
    6. Cleaner variance‑of‑norms computation.
    """

    _metric_template = {
        "loss": 0.0,
        "ce": 0.0,
        "mse": 0.0,
        "var_norm": 0.0,
        "knn_entropy": 0.0,  # placeholder for future use
    }

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def __init__(
        self,
        cfg: RunConfiguration,
        log_root: str = "runs",
        ckpt_root: str = "checkpoints",
        early_stop_patience: int = 10,
    ):
        self.cfg = cfg

        # -------- model & optimiser --------
        self.model = MaskedTransformerAutoencoder(cfg).cuda()
        self.opt = optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=float(cfg.weight_decay),
        )
        self.sched = optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg.num_epochs
        )

        # -------- dataloaders --------
        dl_args = dict(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

        def _make_loader(ds):
            if isinstance(ds, IterableDataset):
                # IterableDataset cannot be shuffled
                return DataLoader(ds, **dl_args)
            return DataLoader(ds, shuffle=True, **dl_args)

        self.train_loader = _make_loader(cfg.training_dataset)
        self.val_loader = _make_loader(cfg.validation_dataset)
        self.test_loader = (
            _make_loader(cfg.testing_dataset)
            if getattr(cfg, "testing_dataset", None) is not None
            else None
        )

        # -------- visualiser --------
        self.visualiser = SnapshotGenerator(
            num_epochs=cfg.num_epochs,
            codec_dim=cfg.codec_dim,
            metrics=self._metric_template.copy(),
            path_to_viz=cfg.viz_dir,
        )
        self.q = self.visualiser.get_queue()
        self.viz_procs = self.visualiser.listen()

        # -------- TensorBoard --------
        run_id = getattr(cfg, "run_id", datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(Path(log_root) / f"transformer_{run_id}")

        # -------- checkpoints / early‑stop --------
        self.ckpt_dir = Path(ckpt_root) / f"transformer_{run_id}"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.patience = cfg.patience
        self.best_val_loss = float("inf")
        self.bad_epochs = 0

        # -------- histories --------
        self.train_hist, self.val_hist = [], []

    # ------------------------------------------------------------------
    # visualiser shutdown helper
    # ------------------------------------------------------------------
    def _shutdown_visualiser(self):
        try:
            self.q.close(); self.q.join_thread()
        except (AttributeError, ValueError):
            pass
        for p in self.viz_procs:
            p.join(timeout=0.1)
            if p.is_alive():
                p.terminate()

    # ------------------------------------------------------------------
    # epoch runner
    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[Dict[str, float], np.ndarray]:
        self.model.train(train)
        split = "train" if train else "eval"

        metrics = {k: 0.0 for k in self._metric_template}
        all_codecs: List[np.ndarray] = []

        pbar = tqdm(loader, leave=False, desc=split)
        for x_num, x_cat in pbar:
            x_num, x_cat = x_num.cuda(non_blocking=True), x_cat.cuda(non_blocking=True)

            with torch.set_grad_enabled(train):
                decoded, loss, batch_m = self.model(
                    x_num,
                    x_cat,
                    mask_prob=self.cfg.mask_prob if train else 0.0
                )
                if train:
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    self.opt.step()

            if train:
                self.sched.step()

            # ----- extra metrics (no‑grad) -----
            with torch.no_grad():
                z_unit = F.normalize(decoded.flatten(1), dim=-1)
                metrics["var_norm"] += torch.var(torch.norm(decoded, dim=-1)).item()
                metrics["knn_entropy"] += 0.0  # placeholder

            for k, v in batch_m.items():
                metrics[k] += v

            all_codecs.append(z_unit.cpu().numpy())

        n_batches = len(loader)
        for k in metrics:
            metrics[k] /= n_batches

        return metrics, np.concatenate(all_codecs, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # save / load checkpoints
    # ------------------------------------------------------------------
    def _save_ckpt(self, epoch: int, is_best: bool = False):
        ckpt_path = self.ckpt_dir / f"epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "opt_state": self.opt.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            ckpt_path,
        )
        if is_best:
            best_path = self.ckpt_dir / "best.pt"
            try:
                if best_path.exists() or best_path.is_symlink():
                    best_path.unlink()
                best_path.symlink_to(ckpt_path.name)
            except OSError:
                shutil.copy2(ckpt_path, best_path)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def train(self):
        for epoch in range(self.cfg.num_epochs):
            train_m, train_codecs = self._run_epoch(self.train_loader, train=True)
            val_m, val_codecs = self._run_epoch(self.val_loader, train=False)

            # ----- record histories -----
            self.train_hist.append(train_m); self.val_hist.append(val_m)
            for split, m in [("Train", train_m), ("Val", val_m)]:
                for k, v in m.items():
                    self.writer.add_scalar(f"{split}/{k}", v, epoch)

            # ----- console log -----
            print(
                f"\nEpoch {epoch:03d}: train_loss={train_m['loss']:.4f}  val_loss={val_m['loss']:.4f}"
            )

            # ----- visualiser queue -----
            self.q.put(
                {
                    "epoch": epoch,
                    "codecs": val_codecs,
                    "history": self.val_hist,
                    "metrics": val_m,
                    "stop": False,
                }
            )

            # ----- checkpoints & early‑stopping -----
            self._save_ckpt(epoch, is_best=val_m["loss"] < self.best_val_loss)

            if val_m["loss"] < self.best_val_loss:
                self.best_val_loss = val_m["loss"]; self.bad_epochs = 0
            else:
                self.bad_epochs += 1
                print(f"  ↳ no improvement ({self.bad_epochs}/{self.patience})")

            if self.bad_epochs >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break

        # ---------------- optional test ----------------
        if self.test_loader is not None:
            test_m, test_codecs = self._run_epoch(self.test_loader, train=False)
            for k, v in test_m.items():
                self.writer.add_scalar(f"Test/{k}", v, epoch + 1)
            print(
                "\n=== Test === "
                f"loss={test_m['loss']:.4f}  ce={test_m['ce']:.4f}  mse={test_m['mse']:.4f}  "
                f"var_norm={test_m['var_norm']:.4f}  knn_ent={test_m['knn_entropy']:.4f}"
            )
            self.q.put(
                {
                    "epoch": epoch + 1,
                    "codecs": test_codecs,
                    "history": self.train_hist,
                    "metrics": test_m,
                    "stop": True,
                }
            )
        else:
            self.q.put({"stop": True})

        self.writer.close()
        self._shutdown_visualiser()


class RotationTripletTrainer:
    """
    Learns to assign **similar scalar scores** to unitary rotations and
    **distant scores** (≥ margin) to perturbed (non-unitary) rotations.

    Loss:   TripletMarginLoss(anchor, positive, negative, margin=1.0, p=2)
    Output: 1-D embedding (= scalar score)
    """

    def __init__(
        self,
        train_codecs: torch.Tensor,          # [N, D]
        val_codecs: torch.Tensor,            # [M, D]
        cfg: RunConfiguration,                                 # RunConfiguration
        margin: float = 1.0,
    ):
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ---------------- data loaders -------------------------------
        self.train_loader = DataLoader(
            TensorDataset(train_codecs.to(self.device)),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            persistent_workers=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            TensorDataset(val_codecs.to(self.device)),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            persistent_workers=True,
            drop_last=True
        )
        rp_mats = RotationPairDataset(cfg.codec_dim, n_samples=2 * train_codecs.shape[0])
        self.aug_loader = DataLoader(rp_mats,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            persistent_workers=True,
        )
        # ---------------- model + loss + opt -------------------------
        self.model = SphereClassifier(cfg.codec_dim).to(self.device)

        self.triplet = nn.TripletMarginWithDistanceLoss(
                        distance_function=lambda x,y: 1 - F.cosine_similarity(x,y),
                        margin=0.3)
        self.opt      = optim.Adam(
            self.model.parameters(), lr=cfg.learning_rate,
        )

        # ---------------- logging & checkpoints ----------------------
        run_id = getattr(cfg, "run_id", datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(os.path.join("runs", f"rot_triplet_{run_id}"))
        self.ckpt_dir = os.path.join("checkpoints", f"rot_triplet_{run_id}")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.best_val = float("inf")
        self.bad_epochs = 0

    # ----------------------------------------------------------------
    def _epoch(self, loader, aug_loader, train: bool):
        self.model.train(train)
        losses = 0.0
        for data, (R, P) in tqdm(zip(loader, aug_loader), leave=False):
            c = data[0]
            R = R.to(self.device)
            P = P.to(self.device)
            z = c.unsqueeze(-1).to(self.device)
            # print("matrices:", R.shape, P.shape, z.shape)
            pos = torch.bmm(R, z).squeeze()
            neg = torch.bmm(P, z).squeeze()
            anc = z.clone().squeeze()

            s_a = self.model(anc).squeeze()
            s_p = self.model(pos).squeeze()
            s_n = self.model(neg).squeeze()

            loss = self.triplet(s_a, s_p, s_n)

            if train:
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()

            losses += loss.item()

        return losses / len(loader)

    # ----------------------------------------------------------------
    def fit(self):
        for epoch in range(self.cfg.num_epochs):
            train_loss = self._epoch(self.train_loader, self.aug_loader, train=True)
            val_loss   = self._epoch(self.val_loader, self.aug_loader, train=False)

            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val",   val_loss,   epoch)
            print(f"Epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}")

            ckpt = os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt")
            torch.save({"model": self.model.state_dict(),
                        "val_loss": val_loss,
                        "epoch": epoch}, ckpt)

            if val_loss < self.best_val:
                self.best_val = val_loss
                self.bad_epochs = 0
                best = os.path.join(self.ckpt_dir, "best.pt")
                try:
                    if os.path.islink(best) or os.path.exists(best):
                        os.unlink(best)
                    os.symlink(os.path.basename(ckpt), best)
                except OSError:
                    shutil.copy2(ckpt, best)
            else:
                self.bad_epochs += 1

            if self.bad_epochs >= self.cfg.patience:
                print("Early stopping.")
                break

        self.writer.close()


class TransformerAnomalyTrainer:
    """
    Train TransformerAnomalyTrainer using.

    Args
    cfg          : RunConfiguration  (reads lr, batch_size, num_epochs, patience, etc.)
    """

    def __init__(
        self,
        cfg: RunConfiguration,
    ):
        self.cfg = cfg

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # ─── data -----------------------------------------------------
        self.train_loader = DataLoader(
            self.cfg.training_dataset,
            batch_size=cfg.batch_size,
            collate_fn=collate_fn,
            drop_last=False
        )
        self.val_loader = DataLoader(
            cfg.validation_dataset,
            batch_size=cfg.batch_size,
            collate_fn=collate_fn,
            drop_last=False
        )

        # ─── model / opt ---------------------------------------------
        self.model = TransformerAnomalyDetector(cfg).to(self.device)

        self.opt = optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate
        )
        # ─── logs / checkpoints --------------------------------------
        run_id = getattr(cfg, "run_id", datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(os.path.join("runs", f"anomaly_detector_{run_id}"))
        self.ckpt_dir = os.path.join("checkpoints", f"anomaly_detector_{run_id}")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.best_val = float("inf")
        self.bad_epochs = 0

    # ------------ one optimizer/eval step ----------------------------
    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], train: bool):

        x_batch, y_batch, _ = batch
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        # Forward
        logits, padding_mask = self.model(x_batch)  # (B, L)
        # Compute loss per element (no reduction)
        loss_all = self.model.compute_loss(logits, y_batch.float(), padding_mask)  # (B, L)
        # Zero out padded positions
        loss = (loss_all * padding_mask).sum() / padding_mask.sum()

        # Backward
        if train:
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

        with torch.no_grad():
            metrics = binary_metrics_from_logits(logits, y_batch)

        return {"loss": loss.item(),
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "f1-score": metrics['f1']}

    def train(self):
        for epoch in range(self.cfg.sph_num_epochs):
            # --- train ------------------------------------------------
            self.model.train()
            train_m = {"loss": 0.0, "precision": 0.0, "recall": 0.0, "f1-score": 0.0}
            for batch in tqdm(self.train_loader, leave=False, desc=f"train {epoch}"):
                m = self._step(batch, train=True)
                for k in train_m: train_m[k] += m[k]
            for k in train_m: train_m[k] /= len(self.train_loader)

            # --- val --------------------------------------------------
            self.model.eval()
            val_m = {"loss": 0.0, "precision": 0.0, "recall": 0.0, "f1-score": 0.0}
            with torch.no_grad():
                for batch in self.val_loader:
                    m = self._step(batch, train=False)
                    for k in val_m: val_m[k] += m[k]
            for k in val_m: val_m[k] /= len(self.val_loader)

            # --- TensorBoard & print ---------------------------------
            for split, mm in [("Train", train_m), ("Val", val_m)]:
                for k,v in mm.items():
                    self.writer.add_scalar(f"{split}/{k}", v, epoch)

            print(f"Epoch {epoch:03d}  "
                  f"train_loss={train_m['loss']:.4f}  "
                  f"val_loss={val_m['loss']:.4f}  "
                  f"val_precision ={val_m['precision']:.2%}  "
                  f"val_recall ={val_m['recall']:.2%}  "
                  f"val_f1 ={val_m['f1-score']:.2%}  "
            )

            # --- checkpoint & early-stop -----------------------------
            ckpt = os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt")
            torch.save({"model": self.model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_m["loss"]}, ckpt)

            if val_m["loss"] < self.best_val:
                self.best_val = val_m["loss"]
                self.bad_epochs = 0
                best = os.path.join(self.ckpt_dir, "best.pt")
                try:
                    if os.path.islink(best) or os.path.exists(best):
                        os.unlink(best)
                    os.symlink(os.path.basename(ckpt), best)
                except OSError:
                    shutil.copy2(ckpt, best)
            else:
                self.bad_epochs += 1

            if self.bad_epochs >= self.cfg.patience:
                print("Early stopping.")
                break

        self.writer.close()


def collate_fn(batch):
    """
    Collate function to pad variable-length sequences in a batch.
    Returns padded embeddings, labels, and a boolean mask.
    """
    batch_size = len(batch)
    dims = batch[0][0].size(1)
    max_len = max(x.size(0) for x, _ in batch)

    x_padded = torch.zeros(batch_size, max_len, dims)
    y_padded = torch.zeros(batch_size, max_len)
    pad_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (x, y) in enumerate(batch):
        L = x.size(0)
        x_padded[i, :L] = x
        y_padded[i, :L] = y
        pad_mask[i, :L] = True

    return x_padded, y_padded, pad_mask
