import torch
from torch import nn
import numpy as np


class MLP(nn.Module):
    def __init__(
        self,
        input_dim:   int,
        hidden_dims: list[int],
        output_dim:  int,
        dropout:     float = 0.5 # initial with 50% 
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(hidden_dims):
                layers.append(nn.BatchNorm1d(dims[i+1])) # Reduce overfitting and smooths to training.
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout)) # Norm L2 regulrization to reduce the model overfitting issues
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)