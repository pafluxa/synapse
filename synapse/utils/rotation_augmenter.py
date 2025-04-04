import torch
import numpy as np
from scipy.stats import special_ortho_group


class RotationAugmenter:
    def __init__(self, dim, epsilon=0.1):
        self.dim = dim
        self.epsilon = epsilon  # Perturbation magnitude

    def generate_pair(self, device='cuda'):
        """Returns (rotation_matrix, perturbed_matrix)"""
        # Generate true rotation matrix
        R = torch.from_numpy(
            special_ortho_group.rvs(self.dim)
        ).float().to(device)

        # Generate perturbed version
        P = R + torch.randn_like(R) * self.epsilon
        P = P @ torch.randn(self.dim, self.dim, device=device) * 0.05

        return R, P

    def validate_rotation(self, matrix):
        """Check how close matrix is to being a true rotation"""
        identity = torch.eye(self.dim, device=matrix.device)
        return torch.norm(matrix @ matrix.T - identity).item()

class SVDRotationAugmenter:
    def __init__(self, dim, epsilon=0.1, min_singular=0.5, max_singular=1.5):
        """
        Args:
            dim (int): Dimension of rotation matrices
            epsilon (float): Magnitude of perturbation (0-1 scale)
            min_singular (float): Minimum singular value for perturbed matrices
            max_singular (float): Maximum singular value for perturbed matrices
        """
        self.dim = dim
        self.epsilon = epsilon
        self.min_singular = min_singular
        self.max_singular = max_singular

    def generate_rotation(self, device='cuda'):
        """Generate true rotation matrix using QR decomposition"""
        q, _ = torch.linalg.qr(torch.randn(self.dim, self.dim, device=device))
        return q

    def perturb_with_svd(self, R):
        """Apply controlled SVD-based perturbation"""
        U, S, Vh = torch.linalg.svd(R)

        # Perturb singular values
        perturbation = torch.rand(S.shape, device=R.device) * self.epsilon
        S_perturbed = S * (1 + perturbation)

        # Enforce singular value bounds
        S_perturbed = torch.clamp(S_perturbed, self.min_singular, self.max_singular)

        # Reconstruct matrix
        return U @ torch.diag(S_perturbed) @ Vh

    def generate_pair(self, device='cuda'):
        """Generate (rotation, perturbed) matrix pair"""
        R = self.generate_rotation(device)
        P = self.perturb_with_svd(R)

        # Ensure P is not orthogonal
        P = P @ torch.randn(self.dim, self.dim, device=device) * 0.05

        return R, P

    def orthogonality_measure(self, matrix):
        """Compute deviation from orthogonality (0=perfect rotation)"""
        return torch.norm(matrix @ matrix.T - torch.eye(self.dim, device=matrix.device)).item()
