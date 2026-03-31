"""
Stream 2: Coordinate Attention Transformer (CAT) for GLA-FatigueNet.

Novel integration of Coordinate Attention into Transformer encoder blocks.
Unlike standard ViT's position embeddings, Coordinate Attention captures
spatial relationships along both width and height directions, providing
position-aware global feature learning.

Key Design:
- Patch embedding (16×16 patches from 224×224)
- 4 CAT blocks with Multi-Head Self-Attention
- Coordinate Attention replaces standard position encoding
- Lightweight design suitable for CPU inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module.
    
    Captures long-range dependencies along both spatial directions (H and W)
    while embedding positional information into channel attention.
    
    Unlike SE attention (global pooling → lose spatial info),
    Coordinate Attention preserves spatial structure by decomposing
    global pooling into two 1D operations along H and W axes.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Pool along W
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # Pool along H

        self.conv_reduce = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        self.conv_h = nn.Conv2d(mid, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.size()

        # Encode spatial information along each axis
        x_h = self.pool_h(x)                      # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)

        # Concatenate along spatial dim
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.conv_reduce(y)             # (B, mid, H+W, 1)

        # Split back
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate attention maps
        a_h = torch.sigmoid(self.conv_h(x_h))  # (B, C, H, 1)
        a_w = torch.sigmoid(self.conv_w(x_w))  # (B, C, 1, W)

        return x * a_h * a_w


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings using convolution."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Sequential(
            # Two-stage patch embedding for better features
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=0.1)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patch projection
        x = self.proj(x)                    # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)    # (B, num_patches, embed_dim)

        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add position embedding (interpolate if needed)
        if x.shape[1] != self.pos_embed.shape[1]:
            pos = self._interpolate_pos_embed(x.shape[1])
            x = x + pos
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)
        return x

    def _interpolate_pos_embed(self, target_len):
        """Interpolate position embeddings to match sequence length."""
        pos = self.pos_embed[:, 1:, :]  # Exclude class token
        cls_pos = self.pos_embed[:, :1, :]

        N = pos.shape[1]
        if target_len - 1 == N:
            return self.pos_embed

        dim = pos.shape[-1]
        sqrt_N = int(math.sqrt(N))
        sqrt_target = int(math.sqrt(target_len - 1))

        pos = pos.reshape(1, sqrt_N, sqrt_N, dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(sqrt_target, sqrt_target), mode='bilinear')
        pos = pos.permute(0, 2, 3, 1).reshape(1, -1, dim)

        return torch.cat([cls_pos, pos], dim=1)


class CATBlock(nn.Module):
    """
    Coordinate Attention Transformer Block.
    
    Combines:
    - Multi-Head Self-Attention (global token interaction)
    - Coordinate Attention (spatial-aware channel recalibration)
    - Feed-Forward Network (non-linear feature transformation)
    
    The Coordinate Attention is applied after reshaping tokens back to 2D,
    providing spatial awareness that standard position embeddings lack.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0,
                 dropout=0.1, attn_dropout=0.1, num_patches_side=14):
        super().__init__()

        self.num_patches_side = num_patches_side

        # Layer Norm 1
        self.norm1 = nn.LayerNorm(embed_dim)

        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=attn_dropout, batch_first=True
        )

        # Coordinate Attention (applied on 2D feature map)
        self.coord_attn = CoordinateAttention(embed_dim, reduction=4)

        # Layer Norm 2
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-Forward Network (MLP)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer Norm for coordinate attention
        self.norm_ca = nn.LayerNorm(embed_dim)

        # Dropout
        self.proj_drop = nn.Dropout(dropout)

        # Learnable scale for coordinate attention contribution
        self.ca_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        """
        Args:
            x: (B, N+1, embed_dim) where N is num_patches, +1 for cls token
        """
        B, N_total, C = x.shape

        # --- Multi-Head Self-Attention ---
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.proj_drop(attn_out)

        # --- Coordinate Attention (on patch tokens only) ---
        cls_token = x[:, :1, :]           # (B, 1, C)
        patch_tokens = x[:, 1:, :]         # (B, N, C)
        N = patch_tokens.shape[1]
        
        # Reshape to 2D for coordinate attention
        h = w = int(math.sqrt(N))
        if h * w == N:
            patch_2d = patch_tokens.transpose(1, 2).reshape(B, C, h, w)
            ca_out = self.coord_attn(patch_2d)
            ca_out = ca_out.flatten(2).transpose(1, 2)  # Back to (B, N, C)
            
            # Residual with learnable scale
            patch_tokens = patch_tokens + self.ca_scale * self.norm_ca(ca_out)
        
        x = torch.cat([cls_token, patch_tokens], dim=1)

        # --- Feed-Forward Network ---
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x


class CoordAttnTransformer(nn.Module):
    """
    Stream 2: Coordinate Attention Transformer (CAT).
    
    Architecture:
        PatchEmbedding → CATBlock × 4 → LayerNorm → cls token → FC
    
    Produces a 256-dimensional feature vector with global context
    and spatial awareness via coordinate attention.
    """

    def __init__(self, config=None):
        super().__init__()

        if config:
            cfg = config['model']['cat_transformer']
            img_size = config['data']['image_size']
            patch_size = cfg['patch_size']
            embed_dim = cfg['embed_dim']
            num_heads = cfg['num_heads']
            num_layers = cfg['num_layers']
            mlp_ratio = cfg['mlp_ratio']
            dropout = cfg['dropout']
            attn_dropout = cfg['attn_dropout']
            output_dim = cfg['output_dim']
        else:
            img_size = 224
            patch_size = 16
            embed_dim = 192
            num_heads = 4
            num_layers = 4
            mlp_ratio = 4.0
            dropout = 0.1
            attn_dropout = 0.1
            output_dim = 256

        num_patches_side = img_size // patch_size

        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )

        # CAT Blocks
        self.blocks = nn.ModuleList([
            CATBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                num_patches_side=num_patches_side,
            )
            for _ in range(num_layers)
        ])

        # Final LayerNorm
        self.norm = nn.LayerNorm(embed_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        """
        Args:
            x: input tensor (B, 3, 224, 224)
        Returns:
            features: (B, output_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches+1, embed_dim)

        # Process through CAT blocks
        for block in self.blocks:
            x = block(x)

        # Extract class token
        x = self.norm(x)
        cls_output = x[:, 0]     # (B, embed_dim)

        # Project to output dimension
        out = self.output_proj(cls_output)  # (B, output_dim)

        return out
