"""Visualization utilities for GLA-FatigueNet."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(history, save_dir='./results/plots'):
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(history, str):
        with open(history) as f:
            history = json.load(f)
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GLA-FatigueNet Training Curves', fontsize=16, fontweight='bold')
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_title('Total Loss'); axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(epochs, history['train_fatigue_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_fatigue_acc'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_title('Fatigue Detection Accuracy'); axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(epochs, history['train_emotion_acc'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_emotion_acc'], 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_title('Emotion Recognition Accuracy'); axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1, 1].set_title('Learning Rate Schedule'); axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('LR')
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Training curves saved to {path}")


def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{title} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")


def plot_classification_report_heatmap(report_str, title, save_path):
    """Parse classification report string and plot as heatmap."""
    lines = [l for l in report_str.strip().split('\n') if l.strip()]
    data, labels = [], []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 5 and parts[0] not in ('accuracy', 'macro', 'weighted'):
            labels.append(parts[0])
            data.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not data:
        return
    arr = np.array(data)
    plt.figure(figsize=(8, max(4, len(labels) * 0.5 + 2)))
    sns.heatmap(arr, annot=True, fmt='.3f', cmap='YlGnBu', xticklabels=['Precision', 'Recall', 'F1-Score'], yticklabels=labels)
    plt.title(f'{title} - Classification Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_gate_analysis(gate_history, save_path):
    """Plot gate value evolution during training."""
    if not gate_history:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    keys = list(gate_history[0].keys())
    for key in keys:
        vals = [g[key] for g in gate_history]
        ax.plot(vals, label=f'{key.upper()} Stream', linewidth=2)
    ax.set_title('Adaptive Gate Values During Training', fontsize=14, fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Gate Value')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results_dict, save_path):
    """Plot comparison bar chart of multiple models."""
    models = list(results_dict.keys())
    metrics_names = ['fatigue_acc', 'emotion_acc', 'fatigue_f1', 'emotion_f1']
    display_names = ['Fatigue Acc', 'Emotion Acc', 'Fatigue F1', 'Emotion F1']
    x = np.arange(len(display_names))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model_name in enumerate(models):
        vals = [results_dict[model_name].get(m, 0) for m in metrics_names]
        ax.bar(x + i * width, vals, width, label=model_name, alpha=0.85)
    ax.set_xlabel('Metric'); ax.set_ylabel('Score')
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(display_names)
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
