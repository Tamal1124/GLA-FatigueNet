"""
Stream 1: Ghost-Enhanced Lightweight CNN for GLA-FatigueNet.

Implements Ghost Bottleneck blocks with SE attention and
a lightweight Feature Pyramid Network (FPN) for multi-scale fusion.

Key Design:
- Ghost modules generate feature maps cheaply via linear transforms
- SE (Squeeze-and-Excitation) attention for channel recalibration
- Multi-scale FPN fuses features from stages S2-S4
- ~50% fewer parameters than standard CNNs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEModule(nn.Module):
    """Squeeze-and-Excitation attention block."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GhostModule(nn.Module):
    """
    Ghost Module: generates more features from cheap operations.
    
    Instead of applying a full convolution to produce `out_channels` features,
    it first produces `out_channels // ratio` "intrinsic" features, then
    generates the remaining "ghost" features via depthwise convolutions.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1,
                 ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # Primary convolution (intrinsic features)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride,
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

        # Cheap operation (ghost features via depthwise conv)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1,
                      dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    """
    Ghost Bottleneck block with optional SE attention.
    
    Structure:
        GhostModule (expand) → DW Conv (if stride=2) → SE → GhostModule (squeeze)
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 dw_kernel_size=3, stride=1, use_se=True, se_reduction=4):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        # Point-wise expansion (Ghost Module 1)
        self.ghost1 = GhostModule(in_channels, mid_channels, relu=True)

        # Depthwise convolution (only when stride > 1)
        if stride > 1:
            self.conv_dw = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, dw_kernel_size,
                          stride, dw_kernel_size // 2,
                          groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
            )
        else:
            self.conv_dw = nn.Identity()

        # Squeeze-and-Excitation
        self.se = SEModule(mid_channels, se_reduction) if use_se else nn.Identity()

        # Point-wise linear projection (Ghost Module 2, no ReLU)
        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)

        # Shortcut connection
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, dw_kernel_size,
                          stride, dw_kernel_size // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x

        out = self.ghost1(x)
        out = self.conv_dw(out)
        out = self.se(out)
        out = self.ghost2(out)

        if self.use_residual:
            out = out + residual
        else:
            out = out + self.shortcut(residual)

        return out


class LightweightFPN(nn.Module):
    """
    Lightweight Feature Pyramid Network for multi-scale feature fusion.
    Fuses features from multiple stages into a single representation.
    """

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.smooth_convs = nn.ModuleList()

        for in_ch in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, 1, 1, 0, bias=False)
            )
            self.smooth_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1,
                              groups=out_channels, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels,
                      1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        """
        Args:
            features: list of feature maps from different stages (low to high level)
        Returns:
            fused: single fused feature map
        """
        laterals = []
        for i, feat in enumerate(features):
            laterals.append(self.lateral_convs[i](feat))

        # Top-down pathway with upsampling
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='nearest'
            )

        # Smooth
        smoothed = []
        target_size = laterals[0].shape[2:]
        for i, lat in enumerate(laterals):
            s = self.smooth_convs[i](lat)
            s = F.adaptive_avg_pool2d(s, target_size)
            smoothed.append(s)

        # Concatenate and fuse
        concat = torch.cat(smoothed, dim=1)
        fused = self.fusion_conv(concat)

        return fused


class GhostCNN(nn.Module):
    """
    Stream 1: Ghost-Enhanced Lightweight CNN.
    
    Architecture:
        Stem → Stage1 → Stage2 → Stage3 → Stage4 → FPN → Global Pool → FC
    
    Produces a 256-dimensional feature vector.
    """

    def __init__(self, config=None):
        super().__init__()

        if config:
            channels = config['model']['ghost_cnn']['channels']
            ghost_ratio = config['model']['ghost_cnn']['ghost_ratio']
            use_se = config['model']['ghost_cnn']['use_se']
            se_reduction = config['model']['ghost_cnn']['se_reduction']
            output_dim = config['model']['ghost_cnn']['output_dim']
        else:
            channels = [16, 24, 48, 96, 192]
            ghost_ratio = 2
            use_se = True
            se_reduction = 4
            output_dim = 256

        # Stem: standard conv for initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Build stages
        self.stage1 = self._make_stage(channels[0], channels[1], 2, stride=2,
                                        use_se=use_se, se_reduction=se_reduction)
        self.stage2 = self._make_stage(channels[1], channels[2], 2, stride=2,
                                        use_se=use_se, se_reduction=se_reduction)
        self.stage3 = self._make_stage(channels[2], channels[3], 3, stride=2,
                                        use_se=use_se, se_reduction=se_reduction)
        self.stage4 = self._make_stage(channels[3], channels[4], 3, stride=1,
                                        use_se=use_se, se_reduction=se_reduction)

        # Lightweight FPN for multi-scale fusion
        self.fpn = LightweightFPN(
            in_channels_list=[channels[2], channels[3], channels[4]],
            out_channels=output_dim
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

        self._initialize_weights()

    def _make_stage(self, in_ch, out_ch, num_blocks, stride=1,
                    use_se=True, se_reduction=4):
        """Build a stage with multiple Ghost Bottleneck blocks."""
        layers = []
        mid_ch = out_ch * 2  # Expansion

        # First block with stride
        layers.append(GhostBottleneck(
            in_ch, mid_ch, out_ch,
            stride=stride, use_se=use_se, se_reduction=se_reduction
        ))

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(GhostBottleneck(
                out_ch, mid_ch, out_ch,
                stride=1, use_se=use_se, se_reduction=se_reduction
            ))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: input tensor (B, 3, 224, 224)
        Returns:
            features: (B, output_dim)
        """
        x = self.stem(x)        # (B, 16, 112, 112)
        x = self.stage1(x)      # (B, 24, 56, 56)
        s2 = self.stage2(x)     # (B, 48, 28, 28)
        s3 = self.stage3(s2)    # (B, 96, 14, 14)
        s4 = self.stage4(s3)    # (B, 192, 14, 14)

        # Multi-scale fusion via FPN
        fused = self.fpn([s2, s3, s4])  # (B, 256, 28, 28)

        # Global pooling → vector
        out = self.global_pool(fused).flatten(1)  # (B, 256)
        out = self.output_proj(out)                # (B, 256)

        return out
