"""
Stream 3: Geometric Landmark Analysis (GLA) module for GLA-FatigueNet.

Processes physiological geometric features (EAR, MAR, head pose, etc.)
through a learned MLP to produce rich fatigue-indicative representations.

Key Design:
- Batch normalization on raw geometric features for scale normalization
- 2-layer MLP with residual connection
- Dropout for regularization
- Produces 128-dimensional fatigue-aware embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricStream(nn.Module):
    """
    Stream 3: Geometric Landmark Analysis.
    
    Takes hand-crafted geometric features (EAR, MAR, head pose, etc.)
    and learns a non-linear embedding that captures fatigue-indicative
    patterns beyond simple thresholding.
    
    Architecture:
        Input (15-d) → BN → FC(64) → ReLU → Dropout → FC(128) → ReLU
        With residual projection from input to output.
    """

    def __init__(self, config=None):
        super().__init__()

        if config:
            cfg = config['model']['gla']
            input_dim = cfg['geometric_features']
            hidden_dim = cfg['hidden_dim']
            output_dim = cfg['output_dim']
            dropout = cfg['dropout']
        else:
            input_dim = 15
            hidden_dim = 64
            output_dim = 128
            dropout = 0.3

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)

        # Main pathway
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.act2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        # Residual projection
        self.residual_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

        # Feature importance attention
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: geometric features tensor (B, input_dim)
        Returns:
            features: (B, output_dim)
        """
        # Input normalization
        x_norm = self.input_norm(x)

        # Feature importance attention
        attn_weights = self.feature_attention(x_norm)
        x_attended = x_norm * attn_weights

        # Main pathway
        h = self.fc1(x_attended)
        h = self.bn1(h)
        h = self.act1(h)
        h = self.drop1(h)

        h = self.fc2(h)
        h = self.bn2(h)

        # Residual connection
        residual = self.residual_proj(x_norm)
        h = h + residual

        h = self.act2(h)
        h = self.drop2(h)

        return h
