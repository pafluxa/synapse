
import time
from contextlib import contextmanager
import os
from tqdm import tqdm

import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

from synapse.models.auto_encoders import TabularBERT
from synapse.utils.visuals import SnapshotGenerator


@contextmanager
def timer(name, metrics_dict=None):
    start = time.time()
    yield
    elapsed = time.time() - start
    if metrics_dict is not None:
        metrics_dict[name] = metrics_dict.get(name, 0.0) + elapsed
    # print(f"{name} took: {elapsed:.4f} seconds")

class MaskedEmbeddingTrainer:

    # Initialize metrics with timing fields
    metrics = {
        'loss': 0.0,
        'mse_loss': 0.0,
        'kld_loss': 0.0,
        'sph_ent': 0.0,
        'sph_rad': 0.0,
        'sph_rep': 0.0,
        'sph_uni': 0.0,
        'mean_norm': 0.0,
        # Timing metrics
        'data_load': 0.0,
        'forward_pass': 0.0,
        'loss_compute': 0.0,
        'backward_pass': 0.0,
        'optimizer_step': 0.0,
        'metrics_update': 0.0
    }

    def __init__(self, config):
        self.config = config
        self.cat_dims = self.config.training_dataset.cardinalities

        self.model = TabularBERT(config).cuda()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Create output directory for visualizations
        self.viz_dir = "vis/training"
        os.makedirs(self.viz_dir, exist_ok=True)

        self.n_train_samples = config.training_dataset.num_samples
        self.data_loader = DataLoader(
            config.training_dataset,
            batch_size=config.batch_size,
            num_workers=4,
            persistent_workers=True
        )
        self.val_loader = DataLoader(
            config.validation_dataset,
            batch_size=config.batch_size,
            num_workers=2,
            persistent_workers=True
        )

        self.visualizer = SnapshotGenerator(
            num_epochs=config.num_epochs,
            codec_dim=config.codec_dim,
            metrics=self.metrics,
            path_to_viz='vis/training'
        )

        self.q = self.visualizer.get_queue()
        self.viz_procs = self.visualizer.listen()
        self.train_history = []
        self.val_history = []

    def mask_data(self, x_num, x_cat, mask_prob=0.5):

        batch_size, num_num = x_num.shape
        num_cat = x_cat.shape[1]

        mask = torch.rand(batch_size, num_num + num_cat) < mask_prob
        num_mask = mask[:, :num_num]
        cat_mask = mask[:, num_num:]

        # Mask numerical with Gaussian noise
        masked_num = x_num.clone()
        masked_num[num_mask] = torch.randn_like(masked_num)[num_mask]

        # Mask categorical with random categories
        masked_cat = x_cat.clone()
        for i in range(num_cat):
            col_mask = cat_mask[:, i]
            random_vals = torch.randint(0, self.cat_dims[i], (col_mask.sum().item(),))
            masked_cat[:, i][col_mask] = random_vals.to(x_cat.device)

        return masked_num, masked_cat, mask

    def train_epoch(self, epoch: int):
        self.model.train()

        metrics = self.metrics
        metrics = {
            'loss': 0.0,
            'mse_loss': 0.0,
            'sph_rad': 0.0,
            'sph_rep': 0.0,
            'sph_ent': 0.0,
            'sph_uni': 0.0,
            'mean_norm': 0.0,
            # Timing metrics
            'data_load': 0.0,
            'forward_pass': 0.0,
            'loss_compute': 0.0,
            'backward_pass': 0.0,
            'optimizer_step': 0.0,
            'metrics_update': 0.0
        }

        all_codecs = []
        progress_bar = tqdm(self.data_loader, desc=f"Epoch {epoch}")

        for batch_idx, (x_num, x_cat) in enumerate(progress_bar):

            # Data loading and transfer
            with timer('data_load', metrics):
                masked_num, masked_cat, mask = self.mask_data(x_num, x_cat)
                masked_num = masked_num.cuda()
                masked_cat = masked_cat.cuda()
                mask = mask.cuda()
                x_num = x_num.cuda(non_blocking=True)
                x_cat = x_cat.cuda(non_blocking=True)
            # Zero gradients
            with timer('optimizer_zero_grad', metrics):
                self.optimizer.zero_grad()  # More efficient in newer PyTorch

            # Forward pass
            with timer('forward_pass', metrics):
                codecs, decoded = self.model(masked_num, masked_cat)
                outputs = (codecs, decoded)
            # Loss computation
            with timer('loss_compute', metrics):
                loss, metrics_dict = self.model.loss(
                    outputs, (x_num, x_cat), mask, epoch
                )

            # Backward pass
            with timer('backward_pass', metrics):
                loss.backward()

            # Optimizer step
            with timer('optimizer_step', metrics):
                self.optimizer.step()

            # Metrics update
            with timer('metrics_update', metrics):
                for k in metrics_dict:
                    if k in metrics:
                        metrics[k] += metrics_dict[k]

                progress_bar.set_postfix({
                    'loss': metrics_dict['loss'],
                    'data_ms': metrics['data_load']*1000/(batch_idx+1),
                    'fwd_ms': metrics['forward_pass']*1000/(batch_idx+1),
                    'bwd_ms': metrics['backward_pass']*1000/(batch_idx+1)
                })
                with torch.no_grad():
                    codecs = codecs.clone().detach().cpu()
                    all_codecs.append(codecs)


        return metrics, np.concat(all_codecs, axis=0).astype(np.float32)

    def evaluate_epoch(self, epoch: int):
        if self.val_loader is None:
            return

        self.model.eval()
        val_metrics = self.metrics

        all_codecs = []
        with torch.no_grad():
            for x_num, x_cat in tqdm(self.val_loader, desc=f"Validation {epoch}"):
                masked_num, masked_cat, mask = self.mask_data(x_num, x_cat)

                masked_num = masked_num.cuda()
                masked_cat = masked_cat.cuda()
                mask = mask.cuda()
                x_num = x_num.cuda(non_blocking=True)
                x_cat = x_cat.cuda(non_blocking=True)

                with torch.no_grad():
                    codecs, decoded = self.model(masked_num, masked_cat)
                    outputs = (codecs, decoded)
                    np_codecs = codecs.clone().detach().cpu().numpy()
                    all_codecs.append(np_codecs)
                    metrics_dict = self.model.loss(outputs, (x_num, x_cat), mask, epoch)

                for k in val_metrics:
                    val_metrics[k] += metrics_dict.get(k, 0.0)

        # Average metrics
        num_batches = len(self.val_loader)
        for k in val_metrics:
            val_metrics[k] /= num_batches

        return val_metrics, np.hstack(all_codecs)

    def train(self):
        # torch.backends#.cuda.enable_flash_sdp(True)

        # print(f"\nNumber of parameters: {self.model.num_parameters}")
        print(f"\nNumber of training samples: {self.n_train_samples}")
        for epoch in range(self.config.num_epochs):
            train_metrics, t_codecs = self.train_epoch(epoch)
            self.train_history.append(train_metrics)
            print(f"\nEpoch {epoch} metrics:")
            print(f"  loss: {train_metrics['loss']:.4f}")
            print(f"  MSE: {train_metrics['mse_loss']:.4f}")
            print(f"  Radial: {train_metrics['sph_rad']:.4f}")
            print(f"  Uniformity: {train_metrics['sph_uni']:.4f}")
            # val_metrics, v_codecs = self.evaluate_epoch(epoch)
            # self.val_history.append(val_metrics)

            payload = {
                'epoch': epoch,
                'codecs': t_codecs,
                'history': self.train_history,
                'metrics': train_metrics,
                'stop': False
            }
            self.q.put(payload)

        # stop visualization engine
        self.visualizer.stop()
