# GLA-FatigueNet

## A Novel Triple-Stream Adaptive Fusion Network for Driver Fatigue and Emotion Detection

### Architecture Overview

GLA-FatigueNet combines three complementary feature streams:

1. **Ghost-Enhanced CNN** — Spatial features via Ghost Bottleneck blocks + SE attention + Lightweight FPN
2. **Coordinate Attention Transformer (CAT)** — Global features via Transformer blocks with Coordinate Attention
3. **Geometric Landmark Analysis (GLA)** — Physiological features (EAR, MAR, head pose) via learned MLP

These are fused via an **Adaptive Gated Fusion Module** and fed to dual task heads (Fatigue + Emotion).

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Smoke test (3 epochs, synthetic data)
python train.py --smoke-test

# Full training
python train.py

# Benchmark model
python evaluation/benchmark.py

# Ablation study
python evaluation/ablation.py

# Real-time webcam demo
python inference/realtime_demo.py
```

### Dataset Setup

Place your dataset in `./datasets/`:
```
datasets/
├── fer2013_images/       # FER2013 in folder format
│   ├── train/
│   │   ├── angry/
│   │   ├── happy/
│   │   └── ...
│   ├── val/
│   └── test/
└── fatigue/              # Fatigue-specific dataset
    ├── train/
    │   ├── alert/
    │   ├── drowsy/
    │   └── fatigued/
    ├── val/
    └── test/
```

### Project Structure

```
gla_fatiguenet/
├── config/config.yaml         # All hyperparameters
├── data/                      # Data pipeline
├── models/                    # Model architecture
│   ├── ghost_cnn.py           # Stream 1
│   ├── coord_attn_transformer.py  # Stream 2
│   ├── geometric_stream.py    # Stream 3
│   ├── fusion.py              # Adaptive Gated Fusion
│   └── gla_fatiguenet.py      # Main model
├── training/                  # Training pipeline
├── evaluation/                # Eval, ablation, benchmark
├── inference/                 # Prediction & real-time demo
├── train.py                   # Entry point
└── results/                   # Output directory
```

### Key Novelties

1. **Triple-Stream Architecture** — First to combine Ghost CNN + Transformer + Geometric streams for fatigue detection
2. **Ghost Module Backbone** — ~50% fewer parameters than standard CNNs
3. **Coordinate Attention in Transformer** — Spatial-aware global features without explicit position encoding
4. **Adaptive Gated Fusion** — Learned dynamic weighting of stream contributions
5. **Multi-Task Learning** — Joint fatigue + emotion detection with consistency loss
