"""Full evaluation pipeline for GLA-FatigueNet."""
import os, torch, json
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_config, load_checkpoint, get_device, set_seed
from models.gla_fatiguenet import GLAFatigueNet
from models.losses import MultiTaskLoss
from training.metrics import MetricsCalculator
from data.dataset import create_dataloaders
from evaluation.visualize import plot_confusion_matrix, plot_classification_report_heatmap


def evaluate(config_path="config/config.yaml", checkpoint_path=None):
    config = load_config(config_path)
    set_seed(config['project']['seed'])
    device = get_device(config)
    _, _, test_loader = create_dataloaders(config)
    model = GLAFatigueNet(config).to(device)
    cp = checkpoint_path or config['inference']['checkpoint_path']
    if os.path.exists(cp):
        load_checkpoint(model, cp)
    else:
        print(f"[WARNING] No checkpoint at {cp}, evaluating untrained model")
    criterion = MultiTaskLoss(config)
    metrics_calc = MetricsCalculator(config['data']['fatigue_classes'], config['data']['emotion_classes'])
    model.eval()
    total_loss = 0
    n_batches = 0
    all_gate_values = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            geo = batch['geometric_features'].to(device)
            fat_labels = batch['fatigue_label'].to(device)
            emo_labels = batch['emotion_label'].to(device)
            outputs = model(images, geo)
            loss, _ = criterion(outputs['fatigue_logits'], outputs['emotion_logits'], fat_labels, emo_labels)
            total_loss += loss.item()
            n_batches += 1
            metrics_calc.update(outputs['fatigue_logits'], outputs['emotion_logits'], fat_labels, emo_labels)
            all_gate_values.append(outputs['gate_values'])
    metrics = metrics_calc.compute()
    plot_dir = config['training']['logging']['plot_dir']
    os.makedirs(plot_dir, exist_ok=True)
    print("\n" + "="*60)
    print("FATIGUE DETECTION RESULTS")
    print("="*60)
    print(metrics['fatigue_report'])
    print("\nEMOTION RECOGNITION RESULTS")
    print("="*60)
    print(metrics['emotion_report'])
    avg_loss = total_loss / max(n_batches, 1)
    print(f"\nTest Loss: {avg_loss:.4f}")
    print(f"Fatigue Accuracy: {metrics['fatigue_acc']:.4f} | F1: {metrics['fatigue_f1']:.4f}")
    print(f"Emotion Accuracy: {metrics['emotion_acc']:.4f} | F1: {metrics['emotion_f1']:.4f}")
    if all_gate_values:
        avg_gates = {k: np.mean([g[k] for g in all_gate_values]) for k in all_gate_values[0]}
        print(f"\nAverage Gate Values: Ghost={avg_gates['ghost']:.3f}, CAT={avg_gates['cat']:.3f}, GLA={avg_gates['gla']:.3f}")
    try:
        plot_confusion_matrix(metrics['fatigue_cm'], config['data']['fatigue_classes'], 'Fatigue Detection', os.path.join(plot_dir, 'fatigue_confusion_matrix.png'))
        plot_confusion_matrix(metrics['emotion_cm'], config['data']['emotion_classes'], 'Emotion Recognition', os.path.join(plot_dir, 'emotion_confusion_matrix.png'))
    except Exception as e:
        print(f"[WARNING] Could not generate plots: {e}")
    results = {'test_loss': avg_loss, 'fatigue_accuracy': metrics['fatigue_acc'], 'fatigue_f1': metrics['fatigue_f1'],
               'emotion_accuracy': metrics['emotion_acc'], 'emotion_f1': metrics['emotion_f1']}
    with open(os.path.join(plot_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return metrics


if __name__ == '__main__':
    evaluate()
