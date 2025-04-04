import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class BatchMoEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.codec_dim = config.codec_dim
        self.num_numerical = config.num_numerical
        self.num_categorical = config.num_categorical
        self.seq_len = config.seq_len
        self.embedding_dim = config.embedding_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.hidden_dims = config.experts_hidden

        # 1. Embedding layers
        self.pos_emb = nn.Embedding(self.seq_len, self.codec_dim)
        self.type_emb = nn.Embedding(2, self.codec_dim)  # 0=num, 1=cat

        # 2. Gate network (3*codec_dim + 1 for mask)
        self.gate = nn.Sequential(
            nn.Linear(3 * self.codec_dim + 1, 256),
            nn.GELU(),
            nn.Linear(256, self.num_experts)
        )

        # 3. Expert networks
        self.experts = nn.ModuleList([
            self._build_expert(self.codec_dim, self.embedding_dim, self.hidden_dims, i)
            for i in range(self.num_experts)
        ])

        # 4. Reconstruction heads
        self.num_recons = nn.ModuleList([
            nn.Linear(self.embedding_dim, 1)
            for _ in range(self.num_numerical)
        ])
        self.cat_recons = nn.ModuleList([
            nn.Linear(self.embedding_dim, dim)
            for dim in config.categorical_dims
        ])

    def _build_expert(self, input_dim, output_dim, hidden_dims, expert_idx):
        """Build expert with scaled initialization based on index"""
        layers = []
        current_dim = input_dim
        scale = 0.1 * (expert_idx % 3 + 1)  # Vary initialization

        for hidden_dim in hidden_dims:
            linear = nn.Linear(current_dim, hidden_dim)
            nn.init.normal_(linear.weight, mean=0, std=scale*0.02)
            layers.extend([linear, nn.LeakyReLU()])
            current_dim = hidden_dim

        final_linear = nn.Linear(current_dim, output_dim)
        nn.init.normal_(final_linear.weight, mean=0, std=scale*0.02)
        layers.append(final_linear)

        return nn.Sequential(*layers)

    def _compute_gates(self, codec, mask_pos):
        batch_size, seq_len = mask_pos.shape
        device = codec.device

        # 1. Position and type embeddings
        positions = torch.arange(seq_len, device=device)
        is_categorical = (positions >= self.num_numerical).long()

        pos_embed = self.pos_emb(positions)  # [S, C]
        type_embed = self.type_emb(is_categorical)  # [S, C]

        # 2. Expand codec and mask
        codec_expanded = codec.unsqueeze(1).expand(-1, seq_len, -1)  # [B, S, C]
        mask_expanded = mask_pos.unsqueeze(-1).float()  # [B, S, 1]

        # 3. Concatenate inputs
        gate_input = torch.cat([
            codec_expanded,
            pos_embed.unsqueeze(0).expand(batch_size, -1, -1),
            type_embed.unsqueeze(0).expand(batch_size, -1, -1),
            mask_expanded
        ], dim=-1)  # [B, S, 3C + 1]

        # 4. Compute gate logits
        gate_logits = self.gate(gate_input)  # [B, S, E]

        # 5. Mask out non-masked positions
        gate_logits = gate_logits * mask_pos.unsqueeze(-1)

        # 6. Select top-k experts per position
        topk_gates, topk_experts = torch.topk(gate_logits, self.top_k, dim=-1)

        return F.softmax(topk_gates, dim=-1), topk_experts

    def forward(self, codec, mask_pos):
        # Get feature indices (0 to seq_len-1)
        feature_indices = torch.arange(self.seq_len, device=codec.device)
        feature_indices = feature_indices.unsqueeze(0).expand(mask_pos.size(0), -1)

        batch_size = codec.size(0)
        seq_len = mask_pos.size(1)
        D = self.embedding_dim

        # 1. Compute gates and expert indices
        gate_probs, expert_indices = self._compute_gates(codec, mask_pos)  # [B, S, K]

        # 2. Collect unique experts and remap indices
        unique_experts, mapped_indices = torch.unique(expert_indices, return_inverse=True)

        # 3. Compute only required expert outputs
        expert_outputs = torch.stack([
            self.experts[idx](codec) for idx in unique_experts
        ], dim=0)  # [U, B, D]

        # 4. Reshape and align dimensions for gathering
        expert_outputs = expert_outputs[:, :, None, :].expand(-1, -1, seq_len, -1)  # [U, B, S, D]
        expert_outputs = expert_outputs.permute(1, 2, 0, 3)  # [B, S, U, D]

        # 5. Gather using remapped indices
        expert_outputs = torch.gather(
            expert_outputs,
            dim=2,
            index=mapped_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        )  # [B, S, K, D]

        # 6. Combine using gate probabilities
        combined = torch.einsum('bsk,bskd->bsd', gate_probs, expert_outputs)

        # 4. Split and reconstruct features
        num_features = combined[:, :self.num_numerical, :]
        cat_features = combined[:, self.num_numerical:, :]

        return {
            'num_recon': torch.cat([head(num_features[:, i])
                                  for i, head in enumerate(self.num_recons)], dim=1),
            'cat_recon': [head(cat_features[:, i])
                        for i, head in enumerate(self.cat_recons)],
            'gates': gate_probs,
            'expert_indices': expert_indices,
            'mask_pos': mask_pos,
            'expert_indices': expert_indices,
            'feature_indices': feature_indices
        }


class _B_BatchMoEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.seq_len = config.seq_len
        self.num_numerical = config.num_numerical
        self.num_categorical = config.num_categorical
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor
        self.codec_dim = config.codec_dim
        self.embedding_dim = config.embedding_dim
        self.expert_hidden = config.expert_hidden

        # Feature position and type embeddings
        self.pos_emb = nn.Embedding(self.seq_len, self.codec_dim)
        # Feature types: 0=num, 1=cat
        self.type_emb = nn.Embedding(2, self.codec_dim)
        # Feature sub-types (for categorical features)
        self.cat_feat_emb = nn.Embedding(self.num_categorical, self.codec_dim)

        # Categorical feature embeddings (if any)
        if self.num_categorical > 0:
            self.cat_idx_emb = nn.Embedding(self.num_categorical, self.codec_dim)

        # Calculate gate input dimension
        gate_input_dim = (3 * self.codec_dim + 1)
        print("gate_input_dim:", gate_input_dim)
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.num_experts)
        )

        self.gate_input_dim = gate_input_dim
        # Experts - each outputs one feature's embedding
        self.experts = nn.ModuleList([
            self._build_expert(
                self.codec_dim,
                self.embedding_dim,
                self.expert_hidden,
                i
            ) for i in range(self.num_experts)
        ])

        # Reconstruction heads
        self.num_recons = nn.ModuleList([
            nn.Linear(self.embedding_dim, 1) for _ in range(self.num_numerical)
        ])
        self.cat_recons = nn.ModuleList([
            InverseEmbedding(config.embedding_dim, dim)
            for dim in config.categorical_dims
        ])
        self.register_buffer('expert_counts', torch.zeros(self.num_experts))

    def _build_expert(self, input_dim, output_dim, hidden_dims, expert_idx):
        """Build expert with scaled initialization based on index"""
        layers = []
        current_dim = input_dim
        scale = 0.1 * (expert_idx % 3 + 1)  # Vary initialization

        for hidden_dim in hidden_dims:
            linear = nn.Linear(current_dim, hidden_dim)
            nn.init.normal_(linear.weight, mean=0, std=scale*0.02)
            layers.extend([linear, nn.LeakyReLU()])
            current_dim = hidden_dim

        final_linear = nn.Linear(current_dim, output_dim)
        nn.init.normal_(final_linear.weight, mean=0, std=scale*0.02)
        layers.append(final_linear)

        return nn.Sequential(*layers)

    def forward(self, codec, mask_pos):
        """
        Args:
            codec: [B, C] - compressed representation
            mask_pos: [B, S] - boolean mask of masked positions
        """
        batch_size, seq_len = mask_pos.shape

        # 1. Generate positional and type embeddings
        positions = torch.arange(seq_len, device=codec.device)
        is_categorical = (positions >= self.num_numerical).long()

        pos_embed = self.pos_emb(positions)  # [S, C]
        type_embed = self.type_emb(is_categorical)  # [S, C]

        # 2. Expand codec and mask
        codec_expanded = codec.unsqueeze(1)  # [B, 1, C]
        mask_expanded = mask_pos.unsqueeze(-1).float()  # [B, S, 1]

        # 3. Concatenate codec + position + type + mask
        gate_input = torch.cat([
            codec_expanded.expand(-1, seq_len, -1),  # [B, S, C]
            pos_embed.unsqueeze(0).expand(batch_size, -1, -1),  # [B, S, C]
            type_embed.unsqueeze(0).expand(batch_size, -1, -1),  # [B, S, C]
            mask_expanded  # [B, S, 1]
        ], dim=-1)  # [B, S, 3C + 1]

        # 4. Compute gates (now mask-aware)
        print(gate_input.shape)
        gate_logits = self.gate(gate_input)  # [B, S, E]

        # 5. Apply mask to gates (zero out unmasked positions)
        gate_logits = gate_logits * mask_pos.unsqueeze(-1)

        # 6. Select top-k experts PER MASKED POSITION
        topk_gates, topk_experts = torch.topk(gate_logits, self.top_k, dim=-1)

        return {
            'gates': F.softmax(topk_gates, dim=-1),
            'expert_indices': topk_experts,
            'mask_pos': mask_pos
        }

    def _forward(self, codec):
        # Exploit batch-level mask counsistency
        batch_size, _ = codec.size()
        device = codec.device

        # Precompute all positions once
        positions = torch.arange(self.seq_len, device=device)
        is_categorical = (positions >= self.num_numerical).long()

        # Vectorized expert computation
        expert_outputs = torch.stack([expert(codec) for expert in self.experts], 1)

        pos_embed = self.pos_emb(positions)  # [S, D]
        type_embed = self.type_emb(is_categorical)  # [S, D]

        # 2. Expand codec and concatenate ALL components
        codec_expanded = codec.unsqueeze(1)  # [B, 1, D]
        gate_input = torch.cat([
            codec_expanded.expand(-1, self.seq_len, -1),  # [B, S, D]
            pos_embed.unsqueeze(0).expand(batch_size, -1, -1),  # [B, S, D]
            type_embed.unsqueeze(0).expand(batch_size, -1, -1)  # [B, S, D]
        ], dim=-1)  # [B, S, 3D]

        gate_logits = self.gate(gate_input.flatten(start_dim=1))
        gate_logits = gate_logits.view(batch_size, self.seq_len, self.num_experts)
        topk_gates, topk_experts = torch.topk(gate_logits, self.top_k, dim=-1)
        # Clamp indices to valid range
        # topk_experts = topk_experts.clamp(max=self.num_experts-1)
        # gates = self.gate(gate_input)  # [B, S, E]
        # topk_gates, topk_experts = torch.topk(gates, self.top_k, dim=-1)

        # Vectorized expert combination
        expert_mask = torch.zeros_like(gate_logits)
        expert_mask.scatter_(-1, topk_experts, F.softmax(topk_gates, dim=-1))
        combined = torch.einsum('bse,bed->bsd', expert_mask, expert_outputs)

        num_recon, cat_recon = self._reconstruct_features(combined).values()

        return {
            'output': combined,
            'gates': topk_gates, #gate_logits,  # Average over sequence
            'num_recon': num_recon,
            'cat_recon': cat_recon,
            'expert_indices': topk_experts
        }


    # def forward(self, codec):
    #     """
    #     Args:
    #         codec: [B, codec_dim]
    #     Returns:
    #         dict containing:
    #         - output: [B, seq_len, embedding_dim]
    #         - gates: [B, seq_len, num_experts]
    #         - topk_idx: [B, seq_len, top_k]
    #         - expert_diversity: scalar loss
    #     """
    #     batch_size = codec.size(0)
    #     device = codec.device
    #     outputs = []
    #     gates = []
    #     topk_indices = []

    #     # Pre-compute positions and types
    #     positions = torch.arange(self.seq_len, device=device)
    #     is_categorical = positions >= self.num_numerical
    #     cat_indices = torch.clamp(positions - self.num_numerical, 0, self.num_categorical-1)

    #     for pos in range(self.seq_len):
    #         # Always build full input (some components will be masked)
    #         components = [
    #             codec,  # [B,D]
    #             self.pos_emb(positions[pos]).expand(batch_size, -1),
    #             self.type_emb(is_categorical[pos].long()).expand(batch_size, -1),
    #             self.cat_idx_emb(cat_indices[pos]).expand(batch_size, -1) if self.cat_idx_emb else torch.zeros(batch_size, self.codec_dim, device=device)
    #         ]

    #         # Create mask (0 for unused components)
    #         mask = torch.ones(4, device=device)  # 4 components
    #         if not is_categorical[pos] or self.cat_idx_emb is None:
    #             mask[3] = 0  # Mask out categorical index if not needed

    #         # Apply mask and concatenate
    #         gate_input = torch.cat([
    #             c * m for c, m in zip(components, mask.unsqueeze(1))
    #         ], dim=1)

    #         # Debug assertion
    #         assert gate_input.size(1) == self.gate_input_dim, \
    #             f"Gate input dim mismatch: got {gate_input.shape}, expected {self.gate_input_dim}"

    #         # Compute gates
    #         gate_logits = self.gate(gate_input)  # [B, num_experts]
    #         topk_gates, topk_experts = torch.topk(gate_logits, self.top_k, dim=1)
    #         # Expert diversity loss
    #         self.expert_counts.scatter_add_(0,
    #             topk_experts.flatten(),
    #             torch.ones_like(topk_experts.flatten(), dtype=torch.float32))
    #         topk_gates = F.softmax(topk_gates, dim=1)

    #         # 3. Process through experts
    #         expert_out = torch.stack([expert(codec) for expert in self.experts], dim=1)  # [B, E, D]

    #         # 4. Combine top-k experts
    #         expert_mask = torch.zeros_like(gate_logits)  # [B, E]
    #         expert_mask.scatter_(1, topk_experts, topk_gates)
    #         combined = torch.einsum('be,bed->bd', expert_mask, expert_out)  # [B, D]

    #         outputs.append(combined)
    #         gates.append(gate_logits)
    #         topk_indices.append(topk_experts)

    #     # Stack outputs across sequence
    #     outputs = torch.stack(outputs, dim=1)  # [B, seq_len, D]
    #     gate_logits = torch.stack(gates, dim=1)  # [B, seq_len, E]
    #     topk_indices = torch.stack(topk_indices, dim=1)  # [B, seq_len, top_k]

    #     expert_diversity = (self.expert_counts.std() / (self.expert_counts.mean() + 1e-6)) ** 2

    #     recon_features = self._reconstruct_features(outputs)

    #     return {
    #         'output': outputs,
    #         'gates': gate_logits,
    #         'topk_idx': topk_indices,
    #         'expert_diversity': expert_diversity,
    #         'numerical': recon_features['numerical'],
    #         'categorical': recon_features['categorical']
    #     }

    def reset_expert_counts(self):
        self.expert_counts.zero_()

    def _reconstruct_features(self, embeddings):
        """Convert embeddings back to original features"""
        print(embeddings.shape)

        num_recons = []
        cat_recons = []

        # Numerical reconstructions
        for i in range(self.num_numerical):
            recon = self.num_recons[i](embeddings[:, i, :])  # [B, 1]
            num_recons.append(recon)

        # Categorical reconstructions
        for i in range(self.num_categorical):
            pos = self.num_numerical + i
            logits = self.cat_recons[i](embeddings[:, pos, :])  # [B, num_classes]
            cat_recons.append(logits)

        return {
            'numerical': torch.cat(num_recons, dim=-1) if num_recons else None,
            'categorical': cat_recons if cat_recons else None
        }

class InverseEmbedding(nn.Module):
    """Enhanced inverse embedding with residual connections"""
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        # Learnable class embeddings
        self.class_emb = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.normal_(self.class_emb, mean=0, std=0.02)

    def forward(self, x):
        x = self.proj(x)  # [..., D]
        return x @ self.class_emb.T  # [..., num_classes]
