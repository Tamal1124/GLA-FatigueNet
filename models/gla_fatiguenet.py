"""
GLA-FatigueNet: Main model combining all three streams and task heads.
"""

import torch
import torch.nn as nn
from models.ghost_cnn import GhostCNN
from models.coord_attn_transformer import CoordAttnTransformer
from models.geometric_stream import GeometricStream
from models.fusion import get_fusion_module


class TaskHead(nn.Module):
    """Classification head for a single task."""
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.4):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.extend([nn.Linear(in_d, h_d), nn.BatchNorm1d(h_d), nn.ReLU(True), nn.Dropout(dropout)])
            in_d = h_d
        layers.append(nn.Linear(in_d, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


class GLAFatigueNet(nn.Module):
    """
    GLA-FatigueNet: Triple-Stream Adaptive Fusion Network.
    
    Streams:
        1. GhostCNN - spatial features via Ghost bottlenecks + FPN
        2. CoordAttnTransformer - global features via CAT blocks
        3. GeometricStream - physiological features (EAR, MAR, head pose)
    
    Fusion: Adaptive Gated Fusion Module
    Heads: Fatigue (3-class) + Emotion (7-class)
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config

        # Three streams
        self.ghost_cnn = GhostCNN(config)
        self.cat_transformer = CoordAttnTransformer(config)
        self.geometric_stream = GeometricStream(config)

        # Dimensions
        if config:
            ghost_dim = config['model']['ghost_cnn']['output_dim']
            cat_dim = config['model']['cat_transformer']['output_dim']
            gla_dim = config['model']['gla']['output_dim']
            fused_dim = config['model']['fusion']['fused_dim']
            fusion_method = config['model']['fusion']['method']
            fusion_dropout = config['model']['fusion']['dropout']
            fat_cfg = config['model']['fatigue_head']
            emo_cfg = config['model']['emotion_head']
        else:
            ghost_dim, cat_dim, gla_dim, fused_dim = 256, 256, 128, 512
            fusion_method, fusion_dropout = 'adaptive_gated', 0.3
            fat_cfg = {'hidden_dims': [256, 128], 'num_classes': 3, 'dropout': 0.4}
            emo_cfg = {'hidden_dims': [256, 128], 'num_classes': 7, 'dropout': 0.4}

        # Fusion module
        self.fusion = get_fusion_module(
            method=fusion_method, ghost_dim=ghost_dim, cat_dim=cat_dim,
            gla_dim=gla_dim, fused_dim=fused_dim, dropout=fusion_dropout
        )

        # Task heads
        self.fatigue_head = TaskHead(fused_dim, fat_cfg['hidden_dims'], fat_cfg['num_classes'], fat_cfg['dropout'])
        self.emotion_head = TaskHead(fused_dim, emo_cfg['hidden_dims'], emo_cfg['num_classes'], emo_cfg['dropout'])

        # Flags for ablation
        self.use_ghost = True
        self.use_cat = True
        self.use_gla = True

    def disable_stream(self, stream_name):
        """Disable a stream for ablation study."""
        if stream_name == 'ghost': self.use_ghost = False
        elif stream_name == 'cat': self.use_cat = False
        elif stream_name == 'gla': self.use_gla = False

    def enable_all_streams(self):
        self.use_ghost = True
        self.use_cat = True
        self.use_gla = True

    def forward(self, image, geometric_features):
        device = image.device
        B = image.size(0)
        ghost_dim = self.ghost_cnn.output_proj[0].out_features
        cat_dim = self.cat_transformer.output_proj[0].out_features
        gla_dim = self.geometric_stream.fc2.out_features

        f_ghost = self.ghost_cnn(image) if self.use_ghost else torch.zeros(B, ghost_dim, device=device)
        f_cat = self.cat_transformer(image) if self.use_cat else torch.zeros(B, cat_dim, device=device)
        f_gla = self.geometric_stream(geometric_features) if self.use_gla else torch.zeros(B, gla_dim, device=device)

        fused, gate_values = self.fusion(f_ghost, f_cat, f_gla)
        fatigue_logits = self.fatigue_head(fused)
        emotion_logits = self.emotion_head(fused)

        return {
            'fatigue_logits': fatigue_logits,
            'emotion_logits': emotion_logits,
            'gate_values': gate_values,
            'features': {'ghost': f_ghost, 'cat': f_cat, 'gla': f_gla, 'fused': fused},
        }
