import torch
import torch.nn as nn
import torch.nn.functional as F

def hypsph_surf_const(codecs, epsilon=1e-6):
    """
    Penalizes codecs to lie on a hypersphere of *any* radius (batch-adaptive).

    Args:
        codecs: Tensor of shape [B, C], where C is the codec dimension.
        epsilon: Small value to avoid division by zero.

    Returns:
        Loss term encouraging codecs to share a common radius (computed per batch).
    """
    norms = torch.norm(codecs, p=2, dim=1, keepdim=False)  # Shape [B]
    norm_avg = norms.clone().detach().mean()  # Detach to avoid second-order gradients
    loss = torch.mean((norms - norm_avg) ** 2)

    return loss

def uniformity_loss(codecs, temperature=0.1):
    """Encourages codecs to be evenly distributed on their hyperspheres."""
    # Normalize to unit sphere (direction only, ignores radius)
    z_dir = torch.nn.functional.normalize(codecs, p=2, dim=-1)  # [B, C]

    # Pairwise cosine similarities
    sim = torch.mm(z_dir, z_dir.T)  # [B, B]

    # Penalize non-diagonal similarities (pushes samples apart)
    mask = ~torch.eye(z_dir.size(0), dtype=torch.bool, device=z_dir.device)
    loss = torch.exp(sim[mask] / temperature).mean()
    return loss


class MoEBalancingLoss(nn.Module):
    def __init__(self, num_experts, num_features, eps=1e-6):
        super().__init__()
        self.num_experts = num_experts
        self.num_features = num_features
        self.eps = eps

        # Track (feature, expert) pairs
        self.register_buffer('feature_expert_counts',
                           torch.zeros(num_features, num_experts))

    def forward(self, gates, expert_indices, feature_indices):
        """
        Args:
            gates: [B, S, K] - gate probabilities
            expert_indices: [B, S, K] - selected expert IDs
            feature_indices: [B, S] - feature positions (0 to num_features-1)
        """
        batch_size = gates.size(0)

        # 1. Convert to linear indices for 2D counting
        flat_features = feature_indices.unsqueeze(-1).expand_as(expert_indices).flatten()  # [B*S*K]
        flat_experts = expert_indices.flatten()  # [B*S*K]
        linear_indices = flat_features * self.num_experts + flat_experts  # [B*S*K]

        # 2. Update counts (detached for stability)
        with torch.no_grad():
            counts = torch.zeros_like(self.feature_expert_counts).flatten()
            counts.scatter_add_(0, linear_indices, torch.ones_like(linear_indices, dtype=torch.float))
            self.feature_expert_counts += counts.view_as(self.feature_expert_counts)

        # 3. Calculate specialization metric
        feature_probs = self.feature_expert_counts / (self.feature_expert_counts.sum(1, keepdim=True) + self.eps)
        specialization = -(feature_probs * torch.log(feature_probs + self.eps)).mean()

        # 4. Traditional balance loss
        importance = torch.zeros(self.num_experts, device=gates.device)
        importance.scatter_add_(0, flat_experts, gates.flatten())
        balance = importance.std() / (importance.mean() + self.eps)

        return balance + (1 - specialization), {
            'balance': balance.item(),
            'specialization': specialization.item()
        }
