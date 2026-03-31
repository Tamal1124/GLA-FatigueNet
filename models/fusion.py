"""
Adaptive Gated Fusion Module (AGFM) for GLA-FatigueNet.

Dynamically learns to weight the contributions of three feature streams
(GhostCNN, CAT, GLA) based on input characteristics.

Key Design:
- Learned gating mechanism for each stream
- Sigmoid gates allow soft selection of stream importance
- Context-dependent: same model adapts gate values per sample
- Superior to static concatenation or fixed-weight fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGatedFusion(nn.Module):
    """
    Adaptive Gated Fusion Module.
    
    For each input sample, learns gate values that dynamically weight
    the three stream outputs. This allows the model to emphasize
    different streams depending on input characteristics:
    - GhostCNN: spatial texture features
    - CAT: global context features
    - GLA: physiological fatigue indicators
    
    gate_i = σ(W_i · [f_ghost, f_cat, f_gla] + b_i)
    F_fused = [gate_1 ⊙ f_ghost, gate_2 ⊙ f_cat, gate_3 ⊙ f_gla]
    output = FC(F_fused)
    """

    def __init__(self, ghost_dim=256, cat_dim=256, gla_dim=128,
                 fused_dim=512, dropout=0.3):
        super().__init__()

        total_dim = ghost_dim + cat_dim + gla_dim  # 640

        # Gate networks for each stream
        self.gate_ghost = nn.Sequential(
            nn.Linear(total_dim, ghost_dim),
            nn.Sigmoid(),
        )
        self.gate_cat = nn.Sequential(
            nn.Linear(total_dim, cat_dim),
            nn.Sigmoid(),
        )
        self.gate_gla = nn.Sequential(
            nn.Linear(total_dim, gla_dim),
            nn.Sigmoid(),
        )

        # Stream-specific projection layers for alignment
        self.proj_ghost = nn.Sequential(
            nn.Linear(ghost_dim, ghost_dim),
            nn.LayerNorm(ghost_dim),
        )
        self.proj_cat = nn.Sequential(
            nn.Linear(cat_dim, cat_dim),
            nn.LayerNorm(cat_dim),
        )
        self.proj_gla = nn.Sequential(
            nn.Linear(gla_dim, gla_dim),
            nn.LayerNorm(gla_dim),
        )

        # Cross-stream interaction attention
        self.cross_attention = nn.Sequential(
            nn.Linear(total_dim, total_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(total_dim // 4, total_dim),
            nn.Sigmoid(),
        )

        # Fusion output projection
        self.output_proj = nn.Sequential(
            nn.Linear(total_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
        )

        # Store dimensions for ablation studies
        self.ghost_dim = ghost_dim
        self.cat_dim = cat_dim
        self.gla_dim = gla_dim
        self.total_dim = total_dim

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize gate biases to 0.5 (equal weighting initially)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, f_ghost, f_cat, f_gla):
        """
        Args:
            f_ghost: GhostCNN features (B, ghost_dim)
            f_cat:   CAT features (B, cat_dim)
            f_gla:   GLA features (B, gla_dim)
        Returns:
            fused: (B, fused_dim)
            gate_values: dict of gate tensors for analysis
        """
        # Project each stream
        f_ghost_proj = self.proj_ghost(f_ghost)
        f_cat_proj = self.proj_cat(f_cat)
        f_gla_proj = self.proj_gla(f_gla)

        # Concatenate for context
        concat = torch.cat([f_ghost_proj, f_cat_proj, f_gla_proj], dim=1)

        # Compute gate values
        g_ghost = self.gate_ghost(concat)
        g_cat = self.gate_cat(concat)
        g_gla = self.gate_gla(concat)

        # Apply gates
        gated_ghost = g_ghost * f_ghost_proj
        gated_cat = g_cat * f_cat_proj
        gated_gla = g_gla * f_gla_proj

        # Cross-stream interaction
        gated_concat = torch.cat([gated_ghost, gated_cat, gated_gla], dim=1)
        cross_attn = self.cross_attention(gated_concat)
        enhanced_concat = gated_concat * cross_attn

        # Final projection
        fused = self.output_proj(enhanced_concat)

        gate_values = {
            'ghost': g_ghost.mean().item(),
            'cat': g_cat.mean().item(),
            'gla': g_gla.mean().item(),
        }

        return fused, gate_values


class ConcatFusion(nn.Module):
    """Simple concatenation fusion (baseline comparison)."""

    def __init__(self, ghost_dim=256, cat_dim=256, gla_dim=128,
                 fused_dim=512, dropout=0.3):
        super().__init__()
        total_dim = ghost_dim + cat_dim + gla_dim

        self.proj = nn.Sequential(
            nn.Linear(total_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, f_ghost, f_cat, f_gla):
        concat = torch.cat([f_ghost, f_cat, f_gla], dim=1)
        fused = self.proj(concat)
        gate_values = {'ghost': 1.0, 'cat': 1.0, 'gla': 1.0}
        return fused, gate_values


class AttentionFusion(nn.Module):
    """Attention-based fusion (baseline comparison)."""

    def __init__(self, ghost_dim=256, cat_dim=256, gla_dim=128,
                 fused_dim=512, dropout=0.3):
        super().__init__()
        
        # Project all to same dim
        self.proj_ghost = nn.Linear(ghost_dim, fused_dim)
        self.proj_cat = nn.Linear(cat_dim, fused_dim)
        self.proj_gla = nn.Linear(gla_dim, fused_dim)

        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(fused_dim * 3, 3),
            nn.Softmax(dim=1),
        )

        self.output = nn.Sequential(
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, f_ghost, f_cat, f_gla):
        p_ghost = self.proj_ghost(f_ghost)
        p_cat = self.proj_cat(f_cat)
        p_gla = self.proj_gla(f_gla)

        # Stack and compute attention weights
        stacked = torch.stack([p_ghost, p_cat, p_gla], dim=1)  # (B, 3, D)
        concat = torch.cat([p_ghost, p_cat, p_gla], dim=1)
        weights = self.attention(concat).unsqueeze(2)            # (B, 3, 1)

        # Weighted sum
        fused = (stacked * weights).sum(dim=1)                   # (B, D)
        fused = self.output(fused)

        gate_values = {
            'ghost': weights[:, 0].mean().item(),
            'cat': weights[:, 1].mean().item(),
            'gla': weights[:, 2].mean().item(),
        }

        return fused, gate_values


def get_fusion_module(method='adaptive_gated', **kwargs):
    """Factory function to create fusion module by name."""
    fusion_map = {
        'adaptive_gated': AdaptiveGatedFusion,
        'concat': ConcatFusion,
        'attention': AttentionFusion,
    }
    if method not in fusion_map:
        raise ValueError(f"Unknown fusion method: {method}. "
                         f"Choose from {list(fusion_map.keys())}")
    return fusion_map[method](**kwargs)
