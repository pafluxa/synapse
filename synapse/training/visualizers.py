from os import read
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing
from multiprocessing import Process
from multiprocessing import Queue

import time
import sys

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def generate_static_basis(input_dim, output_dim=3, seed=42):
    """
    Generates a fixed random orthonormal basis using NumPy.

    Args:
        input_dim (int): Dimensionality of input vectors.
        output_dim (int): Target projection dimension (default 3).
        seed (int): Random seed for reproducibility.

    Returns:
        Q (np.ndarray): Orthonormal basis matrix of shape (input_dim, output_dim).
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(input_dim, output_dim))
    # QR decomposition to get orthonormal columns
    Q, _ = np.linalg.qr(A)
    return Q


class SnapshotGenerator:

    def __init__(self, num_epochs: int, codec_dim: int, metrics: Dict, path_to_viz: str):
        self.viz_dir = path_to_viz
        self.num_epochs = num_epochs
        self.q = Queue(maxsize=200)
        self.mu = 1.0
        self.metrics = metrics
        self.codec_dim = codec_dim
        self.proj = generate_static_basis(self.codec_dim)
        self.reader_procs = []

    def get_queue(self):
        return self.q

    def listen(self):
        """ Spawn process to read from the queue """
        self.metrics = []
        for _ in range(8):
            proc = Process(
                target=self.generator,
                args=(self.q, self.viz_dir, self.proj, self.num_epochs,)
            )
            self.reader_procs.append(proc)

        for proc in self.reader_procs:
            proc.daemon = True
            proc.start()

    def stop(self):

        # Wait for the visualization engine to finish
        while not self.q.empty():
            time.sleep(0.1)

        for proc in self.reader_procs:
            self.q.put({'stop': True})
            proc.join()


    @staticmethod
    def generator(q: Queue, viz_dir, proj, num_epochs):
        """Read from the queue; this spawns as a separate Process"""
        while True:
            data = q.get()  # Read from the queue and do nothing
            if data['stop']:
                break
            SnapshotGenerator.visualize_epoch(
                viz_dir=viz_dir,
                proj=proj,
                num_epochs=num_epochs,
                epoch=data['epoch'],
                codecs=data['codecs'],
                history=data['history'],
                metrics=data['metrics'],
                )


    @staticmethod
    def visualize_epoch(
        viz_dir: str = '',
        proj: np.ndarray = None,
        epoch: int = 0,
        num_epochs: int = 0,
        history: List[Dict[str, Any]] = [],
        metrics: Dict[str, float] = {},
        codecs: np.ndarray = None
    ):
        """Generate visualization plots for the current epoch"""

        points = np.dot(codecs, proj)

        plt.figure(figsize=(18, 12))

        # 1. Plot codec norms distribution
        plt.subplot(2, 2, 1)
        norms = np.linalg.norm(points, axis=1)
        plt.hist(norms, bins=50, alpha=0.7, color='blue', label='Codec Norms', histtype='barstacked')
        plt.title(f'Epoch {epoch}: Codec Norms Distribution\n'
                 f'Mean: {norms.mean():.4f}, Var: {norms.var():.6f}')
        plt.xlabel('L2 Norm')
        plt.ylabel('Frequency')
        plt.xlim(0.01, 6.0)
        plt.ylim(0, points.shape[0]//5)
        # plt.yscale('log')
        plt.legend()

        # 2. Plot first 3 dimensions of codecs
        plt.subplot(2, 2, 2, projection='3d')
        if codecs.shape[1] >= 3:

            # points = points / self.mu
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.1, s=0.1)

            # Create + 2 sigma sphere for reference
            norms = np.linalg.norm(points, axis=1)

            r1 = 1.5
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = r1 * np.cos(u)*np.sin(v)
            y = r1 * np.sin(u)*np.sin(v)
            z = r1 * np.cos(v)
            ax.plot_wireframe(x, y, z, color="r", alpha=0.1, label="reference")

            r = r1
            ax.quiver(-r, 0, 0, 2 * r, 0, 0, color='k', arrow_length_ratio=0.05) # x-axis
            ax.quiver(0, -r, 0, 0, 2 * r, 0, color='k', arrow_length_ratio=0.05) # y-axis
            ax.quiver(0, 0, -r, 0, 0, 2 * r, color='k', arrow_length_ratio=0.05) # z-axis

            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-2.2, 2.2)
            ax.set_zlim(-2.2, 2.2)

            plt.title(f'First 3 Dimensions of Codecs\n(Total dim: {codecs.shape[1]})')

            ax.set_axis_off()

            elev = -5
            azim = ((epoch + 1) * 5 + 180) % 360
            roll = 0

            # Update the axis view and title
            ax.view_init(elev, azim, roll)
            plt.legend()
        else:
            plt.title('Not enough dimensions for 3D plot')

        # 3. Plot loss components
        plt.subplot(2, 2, 3)
        ax = plt.gca()
        ax.set_ylim(1E-3, 2.5E3)
        ax.set_yscale('log')
        loss_components = {
            'MSE': metrics['mse_loss'],
            'VMF': metrics['sph_vmf'],
            'Rad. reg': metrics['sph_rad'],
            'Rep. reg': metrics['sph_rep'],
        }
        plt.bar(loss_components.keys(), loss_components.values(), color=['blue', 'red', 'green'])
        plt.title('Loss Components Breakdown')
        plt.ylabel('Loss Value')

        # 4. Plot metrics over time (placeholder - will be updated in subsequent epochs)
        plt.subplot(2, 2, 4)
        ax = plt.gca()

        epochs = list(range(len(history)))
        plt.plot(epochs, [h['loss'] for h in history], label='total loss')
        plt.plot(epochs, [h['mse_loss'] for h in history], label='mse loss')
        plt.plot(epochs, [h['sph_vmf'] for h in history], label='vmf loss')
        plt.plot(epochs, [h['sph_rad'] for h in history], label='rad. reg')
        plt.plot(epochs, [h['sph_rep'] for h in history], label='rep. reg')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.legend()
        ax.set_ylim(1E-3, 2.5E3)
        ax.set_xlim(0, num_epochs)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(f'{viz_dir}/epoch_{epoch:03d}.png')
        plt.close()
