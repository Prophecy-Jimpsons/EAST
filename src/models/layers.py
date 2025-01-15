# src/models/layers.py
import torch
import torch.nn as nn

class DyReLUB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.coefficient_generator = nn.Conv2d(channels, 2*channels, 1)
        
    def forward(self, x):
        coeffs = self.coefficient_generator(x)
        a, b = torch.chunk(coeffs, 2, dim=1)
        return torch.max(a * x + b, x)

class WeightSharing(nn.Module):
    def __init__(self, base_block):
        super().__init__()
        self.base_block = base_block
        self.scale_factor = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        return self.base_block(x) * self.scale_factor

