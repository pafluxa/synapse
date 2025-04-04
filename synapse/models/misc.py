import torch
import torch.nn as nn


class BottleneckNetwork(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, hidden_dims=[512, 256]):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = input_dim

        # Encoder part
        for hidden_dim in hidden_dims:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU()
                )
            )
            current_dim = hidden_dim

        # Bottleneck
        self.bottleneck = nn.Linear(current_dim, bottleneck_dim)
        self.bottleneck_activation = nn.Tanh()  # Constrain codec values

    def forward(self, x):
        # Flatten if needed (assuming x comes from transformer)
        if x.dim() == 3:
            x = x.flatten(1)  # [batch_size, seq_len * d_model]

        for layer in self.layers:
            x = layer(x)
        return self.bottleneck_activation(self.bottleneck(x))
