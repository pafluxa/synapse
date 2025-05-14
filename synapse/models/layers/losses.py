import torch
import torch.nn as nn
import torch.nn.functional as F

class SphericalLoss(nn.Module):
    def __init__(self, 
                 target_radius=1.0,
                 momentum=0.99,
                 repulsion_strength=0.001,
                 variance_weight=0.5,
                 stability_weight=0.5,
                 epsilon=1e-6):
        super().__init__()
        self.momentum = momentum
        self.target_radius = target_radius
        self.repulsion_strength = repulsion_strength
        self.variance_weight = variance_weight
        self.stability_weight = stability_weight
        self.epsilon = epsilon
        
        # Register buffers for EMA tracking
        # self.register_buffer('ema_radius', torch.tensor(target_radius))
        # self.register_buffer('epoch_radius_sum', torch.tensor(0.0))
        # self.register_buffer('epoch_samples', torch.tensor(0))
        
    def forward(self, vectors):
        B, D = vectors.shape
        if B < 2:
            return torch.tensor(0.0, device=vectors.device), {}
            
        # Current batch statistics
        norms = torch.norm(vectors, dim=1)
        current_mean = torch.mean(norms)
        current_var = torch.var(norms)
        
        # Accumulate for epoch-level EMA update (detached from graph)
        # with torch.no_grad():
        #     self.epoch_radius_sum += torch.sum(norms)
        #     self.epoch_samples += B
        
        # ----- Variance Constraint -----
        norm_variance = current_var / (current_mean**2 + self.epsilon)
        
        # ----- Radius Stability Loss -----
        radius_stability = torch.mean((norms - current_mean.detach())**2)
        
        # ----- Spherical Repulsion -----
        unit_vectors = vectors / (norms.unsqueeze(1) + self.epsilon)
        cos_sim = torch.mm(unit_vectors, unit_vectors.T)
        angles = torch.acos(torch.clamp(cos_sim, -1+self.epsilon, 1-self.epsilon))
        
        mask = torch.eye(B, device=vectors.device, dtype=torch.bool)
        valid_angles = angles[~mask].view(B, B-1)
        repulsion = torch.mean(1.0 / (valid_angles + self.epsilon)**2)
        
        # Combine losses
        total_loss = (
            self.variance_weight * norm_variance +
            self.repulsion_strength * repulsion +
            self.stability_weight * radius_stability
        )
        
        return total_loss, {
            'variance': norm_variance.detach()
        }
    

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

    return loss, norm_avg

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

def sphere_uniform_loss(vectors, radius=1.0, alpha=1.0, beta=1.0, gamma=1.0, epsilon=1e-6):
    """
    Penalizes vectors not evenly distributed inside a sphere.
    
    Args:
        vectors (torch.Tensor): Batch of vectors, shape (B, D).
        radius (float): Radius of the sphere.
        alpha (float): Weight for the outside penalty.
        beta (float): Weight for the direction uniformity loss.
        gamma (float): Weight for the radial distribution loss.
        epsilon (float): Small value to avoid division by zero.
    
    Returns:
        torch.Tensor: Scalar loss value.
    """
    B, D = vectors.shape
    if B == 0:
        return torch.tensor(0.0, device=vectors.device)
    
    # Compute norms of each vector
    norms = torch.norm(vectors, dim=1)  # (B,)
    
    # Penalty for vectors not at the surface of the sphere
    outside_penalty = torch.sum(torch.relu(norms - radius))
    
    # Avoid division by zero when normalizing to unit vectors
    unit_vectors = vectors / (norms.unsqueeze(1) + epsilon)  # (B, D)
    
    # Direction uniformity loss (covariance should be (1/D) * I)
    mean_uv = torch.mean(unit_vectors, dim=0)  # (D,)
    cov = (unit_vectors.T @ unit_vectors) / B - torch.outer(mean_uv, mean_uv)
    target_cov = torch.eye(D, device=vectors.device) / D
    direction_loss = torch.norm(cov - target_cov, p='fro')
    
    # Radial distribution loss
    mean_r = torch.mean(norms).detach()
    expected_mean_r = ((D * mean_r) / (D + 1))
    mean_r2 = torch.mean(norms**2).detach()
    expected_mean_r2 = ((D * mean_r2) / (D + 2))
    radial_loss = (norms - expected_mean_r)**2 + (norms**2 - expected_mean_r2)**2
    radial_loss = torch.mean(radial_loss)
    # Combine loss components
    total_loss = (alpha * outside_penalty + 
                  beta * direction_loss + 
                  gamma * radial_loss)
    
    return total_loss

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
