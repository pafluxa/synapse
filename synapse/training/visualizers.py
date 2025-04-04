
from os import read
from typing import Dict, List, Tuple, Optional, Any
from multiprocessing import Process
from multiprocessing import Queue

import time
import sys

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class AutoMotVisualizer:

    viz_dir: str
    history: List[Dict[str, float]]
    num_epochs: int
    q: Queue
    pca: PCA = PCA(n_components=3)
    mu: float

    def __init__(self, num_epochs: int, path_to_viz: str):
        self.history = []
        self.viz_dir = path_to_viz
        self.num_epochs = num_epochs
        self.q = Queue(maxsize=0)
        self.mu = 1.0

    def get_queue(self):
        return self.q

    def generator(self, q: Queue):
        """Read from the queue; this spawns as a separate Process"""
        while True:
            data = q.get()  # Read from the queue and do nothing
            if data['stop']:
                break
            self.visualize_epoch(
                epoch=data['epoch'],
                codecs=data['codecs'],
                history=data['history'],
                metrics=data['metrics'],
            )

    def listen(self):
        """ Spawn process to read from the queue """
        reader_proc = Process(target=self.generator, args=((self.q),))
        reader_proc.daemon = True
        reader_proc.start()

        return reader_proc

    def visualize_epoch(self,
        epoch: int = 0,
        history: List[Dict[str, Any]] = [], metrics: Dict[str, float] = {}, codecs: np.ndarray = None):
        """Generate visualization plots for the current epoch"""

        points = self.pca.fit_transform(codecs)
        # points = codecs

        plt.figure(figsize=(18, 12))

        # 1. Plot codec norms distribution
        plt.subplot(2, 2, 1)
        norms = np.linalg.norm(points, axis=1)
        plt.hist(norms, bins=50, alpha=0.7, color='blue', density=True, label='Codec Norms', histtype='barstacked')
        plt.title(f'Epoch {epoch}: Codec Norms Distribution\n'
                 f'Mean: {norms.mean():.4f}, Var: {norms.var():.6f}')
        plt.xlabel('L2 Norm')
        plt.ylabel('Frequency')
        plt.xlim(1E-2, 1E1)
        plt.ylim(1E-2, 1E5)
        plt.yscale('log')
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

            r1 = 8.0
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = r1 * np.cos(u)*np.sin(v)
            y = r1 * np.sin(u)*np.sin(v)
            z = r1 * np.cos(v)
            ax.plot_wireframe(x, y, z, color="r", alpha=0.1, label="reference")

            r = r1
            ax.quiver(-r, 0, 0, 2 * r, 0, 0, color='k', arrow_length_ratio=0.05) # x-axis
            ax.quiver(0, -r, 0, 0, 2 * r, 0, color='k', arrow_length_ratio=0.05) # y-axis
            ax.quiver(0, 0, -r, 0, 0, 2 * r, color='k', arrow_length_ratio=0.05) # z-axis

            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.set_zlim(-10, 10)

            plt.title(f'First 3 Dimensions of Codecs\n(Total dim: {codecs.shape[1]})')

            ax.set_axis_off()

            elev = -10
            azim = 30 # ((epoch + 1) * 5 + 180) % 360
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
            'Numerical': metrics['num_loss'],
            'Categorical': metrics['cat_loss'],
            'SEC': metrics['sphere_loss'],
        }
        plt.bar(loss_components.keys(), loss_components.values(), color=['blue', 'orange', 'green'])
        plt.title('Loss Components Breakdown')
        plt.ylabel('Loss Value')

        # 4. Plot metrics over time (placeholder - will be updated in subsequent epochs)
        plt.subplot(2, 2, 4)
        ax = plt.gca()

        epochs = list(range(len(history)))
        plt.plot(epochs, [h['loss'] for h in history], label='total loss')
        plt.plot(epochs, [h['num_loss'] for h in history], label='num loss')
        plt.plot(epochs, [h['cat_loss'] for h in history], label='cat loss')
        plt.plot(epochs, [h['sphere_loss'] for h in history], label='sec')
        plt.plot(epochs, [h['moe_loss'] for h in history], label='moe loss')
        #plt.axhline(y=0.0, color='r', linestyle='--', label='Zero variance')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.legend()
        ax.set_ylim(1E-3, 2.5E3)
        ax.set_xlim(0, self.num_epochs)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/epoch_{epoch:03d}.png')
        plt.close()
