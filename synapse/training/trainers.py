# -------------------------------------------------------------------------
# MaskedEmbeddingTrainer  ✧  v2
# – Adds: early stopping  +  checkpointing with best-model flag
# -------------------------------------------------------------------------
import os, time, shutil
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from synapse.models.sphere_classifier import RotationConvNet as SphereClassifier
from synapse.models.sphere_classifier import SVDRotationAugmenter
from synapse.utils.config_parser import RunConfiguration
from synapse.models.auto_encoders import TabularBERT, knn_entropy
from synapse.utils.visuals import SnapshotGenerator


def contrastive_loss_fn(x1, x2, label, margin: float = 1.0):
    """
    Computes Contrastive Loss
    """

    dist = torch.nn.functional.pairwise_distance(x1, x2)

    loss = (1 - label) * torch.pow(dist, 2) \
        + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)

    return loss

# ------------------------------------------------------------------ metrics
def binary_metrics_from_logits(logits_pred: torch.Tensor,
                               logits_anch: torch.Tensor,
                               threshold: float = 0.0):
    """
    logits : shape [N] in [-1, 1]
    labels : {0,1} ground-truth
    """
    preds = (logits_pred >= threshold).long()
    labels = (logits_anch >= threshold).long()

    correct = (preds == labels).sum().item()
    acc     = correct / len(labels)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ------------------------------------------------------------------ timers
@contextmanager
def timer(name: str, store: Optional[Dict[str, float]] = None):
    t0 = time.time()
    yield
    if store is not None:
        store[name] = store.get(name, 0.0) + (time.time() - t0)

# -------------------------------------------------------- main trainer ---
class MaskedEmbeddingTrainer:
    """
    Trainer for masked-reconstruction TabularBERT.

    Features
    --------
    • Train / per-epoch validation / final test
    • TensorBoard logging
    • SnapshotGenerator queue intact
    • Early-stopping on val-loss (patience k)
    • Checkpoint every epoch  +  best.pt symlink / copy
    """

    _metric_template = {
        "loss": 0.0,
        "mse_loss": 0.0,
        "var_norm": 0.0,
        "knn_entropy": 0.0,
        "sph_rad": 0.0,
        "sph_uni": 0.0,
        "mean_norm": 0.0,
    }

    # ------------------------- init -------------------------------------
    def __init__(
        self,
        config: RunConfiguration,
        log_root: str = "runs",
        ckpt_root: str = "checkpoints",
        early_stop_patience: int = 10,
    ):
        self.cfg = config
        self.cat_dims: List[int] = config.training_dataset.cardinalities

        # model & optimiser ------------------------------------------------
        self.model = TabularBERT(config).cuda()
        self.opt = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=float(config.weight_decay),
        )

        # data -------------------------------------------------------------
        dl_args = dict(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            persistent_workers=True,
        )
        self.train_loader = DataLoader(
            config.training_dataset, shuffle=False, **dl_args
        )
        self.val_loader = DataLoader(
            config.validation_dataset, shuffle=False, **dl_args
        )
        self.test_loader = (
            DataLoader(
                config.testing_dataset, shuffle=False, **dl_args
            )
            if getattr(config, "testing_dataset", None) is not None
            else None
        )

        # visualiser / queue ----------------------------------------------
        self.visualiser = SnapshotGenerator(
            num_epochs=config.num_epochs,
            codec_dim=config.codec_dim,
            metrics=self._metric_template.copy(),
            path_to_viz=config.viz_dir
        )
        self.q = self.visualiser.get_queue()
        self.viz_procs = self.visualiser.listen()

        # TensorBoard ------------------------------------------------------
        run_id = getattr(config, "run_id", datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(os.path.join(log_root, f"masked_bert_{run_id}"))

        # checkpoints ------------------------------------------------------
        self.ckpt_dir = os.path.join(ckpt_root, f"masked_bert_{run_id}")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # early stop -------------------------------------------------------
        self.patience = early_stop_patience
        self.best_val_loss = float("inf")
        self.bad_epochs = 0

        # histories --------------------------------------------------------
        self.train_hist, self.val_hist = [], []

    # --------------------------------------------------------------
    def _shutdown_visualiser(self):
        """Close queue, join/kill child processes so Python can exit."""
        try:
            self.q.close()
            self.q.join_thread()             # flush / close writer thread
        except (AttributeError, ValueError):
            pass                             # queue already closed

        for p in self.viz_procs:
            p.join(timeout=0.1)
            if p.is_alive():
                p.terminate()

    # ------------------ masking helper ----------------------------------
    def mask_data(
        self, x_num: torch.Tensor, x_cat: torch.Tensor, prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n_num = x_num.shape
        n_cat = x_cat.shape[1]
        mask = torch.rand(b, n_num + n_cat, device=x_num.device) < prob

        num_mask, cat_mask = mask[:, :n_num], mask[:, n_num:]

        x_num_m = x_num.clone()
        x_num_m[num_mask] = torch.randn_like(x_num)[num_mask]

        x_cat_m = x_cat.clone()
        for i in range(n_cat):
            m = cat_mask[:, i]
            if m.any():
                randv = torch.randint(
                    0, self.cat_dims[i], (int(m.sum()),), device=x_cat.device
                )
                x_cat_m[m, i] = randv
        return x_num_m, x_cat_m, mask

    # ------------------ epoch runner ------------------------------------
    def _run_epoch(
        self, loader: DataLoader, w1: torch.Tensor, w2: torch.Tensor, train: bool
    ) -> Tuple[Dict[str, float], np.ndarray]:
        self.model.train(train)
        split = "train" if train else "eval"

        metrics = {k: 0.0 for k in self._metric_template}
        codecs_all: List[np.ndarray] = []

        for x_num, x_cat in tqdm(loader, leave=False, desc=f"{split}"):
            x_num, x_cat = x_num.cuda(non_blocking=True), x_cat.cuda(non_blocking=True)
            x_num_m, x_cat_m, mask = self.mask_data(x_num, x_cat, self.cfg.mask_prob)

            with torch.set_grad_enabled(train):
                z, decoded = self.model(x_num_m, x_cat_m)
                loss, batch_m = self.model.loss(
                    (z, decoded), (x_num, x_cat), mask, w1, w2
                )

                if train:
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    self.opt.step()

            with torch.no_grad():
                z_unit = F.normalize(z, dim=1, eps=1e-8)
                metrics["knn_entropy"] += knn_entropy(z_unit, k=4).mean().item()
                metrics["var_norm"] += torch.var(torch.norm(z, dim=-1)).item()

            for k in batch_m:
                if k in metrics:
                    metrics[k] += batch_m[k]

            codecs_all.append(z.detach().cpu().numpy())

        n_batches = len(loader)
        for k in metrics:
            metrics[k] /= n_batches

        return metrics, np.concatenate(codecs_all, axis=0).astype(np.float32)

    # ------------------ save / load checkpoints -------------------------
    def _save_ckpt(self, epoch: int, is_best: bool = False):
        ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt")
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
            best_path = os.path.join(self.ckpt_dir, "best.pt")
            # overwrite / symlink
            try:
                if os.path.islink(best_path) or os.path.exists(best_path):
                    os.unlink(best_path)
                os.symlink(os.path.basename(ckpt_path), best_path)
            except OSError:
                shutil.copy2(ckpt_path, best_path)

    # ------------------ main loop ---------------------------------------
    def train(self):
        n1, n2 = torch.tensor(-20.0, device="cuda"), torch.tensor(-20.0, device="cuda")
        w1, w2 = torch.tensor(0.0, device="cuda"), torch.tensor(0.0, device="cuda")

        for epoch in range(self.cfg.num_epochs):
            train_m, train_codecs = self._run_epoch(
                self.train_loader, w1, w2, train=True
            )
            val_m, _ = self._run_epoch(self.val_loader, w1, w2, train=False)

            # schedule (unchanged)
            n1 += float(train_m["mse_loss"] < 1.0)
            n2 -= float(train_m["mse_loss"] >= 1.0)
            n1.clamp_(-20, 20)
            n2.clamp_(-20, 20)
            w1 = torch.sigmoid(n1 / 4.0)
            w2 = torch.tensor(0.001, device="cuda")

            # hist + tb
            self.train_hist.append(train_m)
            self.val_hist.append(val_m)
            for split, m in [("Train", train_m), ("Val", val_m)]:
                for k in self._metric_template:
                    self.writer.add_scalar(f"{split}/{k}", m[k], epoch)

            # console
            print(
                f"\nEpoch {epoch:03d}: "
                f"train_loss={train_m['loss']:.4f}  val_loss={val_m['loss']:.4f}"
            )

            # visual queue
            self.q.put(
                {
                    "epoch": epoch,
                    "codecs": train_codecs,
                    "history": self.train_hist,
                    "metrics": train_m,
                    "stop": False,
                }
            )

            # -------------- early-stop & checkpoints ----------------------
            self._save_ckpt(epoch, is_best=val_m["loss"] < self.best_val_loss)

            if val_m["loss"] < self.best_val_loss:
                self.best_val_loss = val_m["loss"]
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
                print(f"  ↳ no improvement ({self.bad_epochs}/{self.patience})")

            if self.bad_epochs >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

        # ---------------- test ------------------------------------------
        if self.test_loader is not None:
            test_m, test_codecs = self._run_epoch(
                self.test_loader, w1, w2, train=False
            )
            for k in self._metric_template:
                self.writer.add_scalar(f"Test/{k}", test_m[k], epoch + 1)

            print(
                "\n=== Test === "
                f"loss={test_m['loss']:.4f}  mse={test_m['mse_loss']:.4f}  "
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
# -------------------------------------------------------------------------


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
        margin: float = 0.25,
    ):
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ---------------- data loaders -------------------------------
        self.train_loader = DataLoader(
            TensorDataset(train_codecs.to(self.device)),
            batch_size=cfg.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(val_codecs.to(self.device)),
            batch_size=cfg.batch_size, shuffle=False
        )

        # ---------------- model + loss + opt -------------------------
        self.model = SphereClassifier(cfg.codec_dim).to(self.device)

        self.triplet = nn.TripletMarginLoss(margin=margin, p=2)
        self.opt      = optim.Adam(
            self.model.parameters(), lr=cfg.learning_rate,
        )

        self.aug = SVDRotationAugmenter(train_codecs.shape[1], epsilon=0.1)

        # ---------------- logging & checkpoints ----------------------
        run_id = getattr(cfg, "run_id", datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(os.path.join("runs", f"rot_triplet_{run_id}"))
        self.ckpt_dir = os.path.join("checkpoints", f"rot_triplet_{run_id}")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.best_val = float("inf")
        self.bad_epochs = 0

    # ----------------------------------------------------------------
    def _make_triplet(self, z: torch.Tensor):
        R, P = self.aug.generate_pair(self.device)
        pos = (R @ z.T).T
        neg = (P @ z.T).T
        return z, pos, neg

    # ----------------------------------------------------------------
    def _epoch(self, loader, train: bool):
        self.model.train(train)
        losses = 0.0
        for data in tqdm(loader, leave=False):
            anchor = data[0].to(self.device)
            a, p, n = self._make_triplet(anchor)

            s_a = self.model(a).unsqueeze(1)   # shape [B,1]
            s_p = self.model(p).unsqueeze(1)
            s_n = self.model(n).unsqueeze(1)

            loss = contrastive_loss_fn(s_a, s_p, torch.tensor(0.0), torch.tensor(0.5))
            loss += contrastive_loss_fn(s_a, s_n, torch.tensor(1.0), torch.tensor(0.5))

            if train:
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()

            losses += loss.item()

        return losses / len(loader)

    # ----------------------------------------------------------------
    def fit(self):
        for epoch in range(self.cfg.num_epochs):
            train_loss = self._epoch(self.train_loader, train=True)
            val_loss   = self._epoch(self.val_loader,   train=False)

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


class SphereContrastiveTrainer:
    """
    Train SphereClassifier using triplet contrastive + BCE.

    Args
    ----
    train_codecs : Tensor[N,D]
    val_codecs   : Tensor[M,D]
    test_codecs  : Tensor[K,D] | None
    cfg          : RunConfiguration  (reads lr, batch_size, num_epochs, patience, etc.)
    """

    def __init__(
        self,
        train_codecs: torch.Tensor,
        val_codecs: torch.Tensor,
        test_codecs: Optional[torch.Tensor],
        cfg,
    ):
        self.cfg = cfg
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ─── data -----------------------------------------------------
        self.train_loader = DataLoader(
            TensorDataset(train_codecs.to(self.device)),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(val_codecs.to(self.device)),
            batch_size=cfg.batch_size,
            shuffle=False,
        )
        self.test_loader = (
            DataLoader(
                TensorDataset(test_codecs.to(self.device)),
                batch_size=cfg.batch_size,
                shuffle=False,
            )
            if test_codecs is not None else None
        )

        # ─── model / opt ---------------------------------------------
        self.model = SphereClassifier(
            codec_dim=train_codecs.shape[1],
            hidden_dim=cfg.hidden_dim,          # add this field to YAML or keep default
        ).to(self.device)

        self.triplet = nn.TripletMarginLoss(margin=1.0, p=2)
        self.opt     = optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate
        )

        self.aug = SVDRotationAugmenter(train_codecs.shape[1], epsilon=0.1)

        # ─── logs / checkpoints --------------------------------------
        run_id = getattr(cfg, "run_id", datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(os.path.join("runs", f"sphere_{run_id}"))
        self.ckpt_dir = os.path.join("checkpoints", f"sphere_{run_id}")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.best_val = float("inf")
        self.bad_epochs = 0

    # ------------ helper to build triplet ----------------------------
    def _make_triplet(self, z):
        R, P = self.aug.generate_pair(self.device)
        pos = (R @ z.T).T
        neg = (P @ z.T).T
        return z, pos, neg

    # ------------ one optimizer/eval step ----------------------------
    def _step(self, batch, train: bool):
        (anchor,) = batch
        anchor = anchor.to(self.device)

        a, p, n = self._make_triplet(anchor)

        logits_a, h_a = self.model(a)
        logits_p, h_p = self.model(p)
        logits_n, h_n = self.model(n)

        loss_trip = self.triplet(logits_a, logits_p, logits_n)

        loss = loss_trip

        if train:
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

        with torch.no_grad():
            metrics_p = binary_metrics_from_logits(logits_a, logits_p)
            metrics_n = binary_metrics_from_logits(logits_a, logits_n)

        return {"loss": loss.item(),
                "triplet": loss_trip.item(),
                "bce": 0.0,
                "acc_p": metrics_p['accuracy'],
                "acc_n": metrics_n['accuracy']}

    # ------------ main loop ------------------------------------------
    def fit(self):

        for epoch in range(self.cfg.sph_num_epochs):
            # --- train ------------------------------------------------
            self.model.train()
            train_m = {"loss":0,"triplet":0,"acc_p":0,"acc_n":0}
            for batch in tqdm(self.train_loader, leave=False, desc=f"train {epoch}"):
                m = self._step(batch, train=True)
                for k in train_m: train_m[k] += m[k]
            for k in train_m: train_m[k] /= len(self.train_loader)

            # --- val --------------------------------------------------
            self.model.eval()
            val_m = {"loss":0,"triplet":0,"acc_p":0,"acc_n":0}
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
                  f"val_acc (pos)={val_m['acc_p']:.2%}  "
                  f"val_acc (neg)={val_m['acc_n']:.2%}  "
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
