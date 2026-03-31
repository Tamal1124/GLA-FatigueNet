"""Ablation study runner for GLA-FatigueNet."""
import os, sys, json, torch, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_config, set_seed, get_device
from models.gla_fatiguenet import GLAFatigueNet
from models.losses import MultiTaskLoss
from models.fusion import get_fusion_module
from data.dataset import create_dataloaders
from training.trainer import Trainer
from training.optimizer import get_optimizer, get_scheduler
from utils.logger import TrainingLogger
from evaluation.visualize import plot_model_comparison


def run_ablation(config_path="config/config.yaml", quick_epochs=10):
    """Run ablation study: disable each stream and each fusion method."""
    config = load_config(config_path)
    set_seed(config['project']['seed'])
    device = get_device(config)
    train_loader, val_loader, _ = create_dataloaders(config)
    results = {}

    # --- Stream ablation ---
    ablation_configs = [
        ("Full Model (Ours)", None),
        ("w/o GhostCNN", "ghost"),
        ("w/o CAT", "cat"),
        ("w/o GLA", "gla"),
    ]

    for name, disabled_stream in ablation_configs:
        print(f"\n{'='*50}\nAblation: {name}\n{'='*50}")
        model = GLAFatigueNet(config).to(device)
        if disabled_stream:
            model.disable_stream(disabled_stream)
        criterion = MultiTaskLoss(config)
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)
        logger = TrainingLogger(experiment_name=f"ablation_{name.replace(' ', '_').replace('/', '_')}")
        trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, config, logger, device)
        # Quick training
        orig_epochs = config['training']['epochs']
        history = trainer.train(quick_epochs)
        config['training']['epochs'] = orig_epochs
        val_metrics = trainer.validate(quick_epochs)
        results[name] = {
            'fatigue_acc': val_metrics['fatigue_acc'],
            'emotion_acc': val_metrics['emotion_acc'],
            'fatigue_f1': val_metrics.get('fatigue_f1', 0),
            'emotion_f1': val_metrics.get('emotion_f1', 0),
        }
        logger.close()

    # --- Fusion method ablation ---
    for fusion_method in ['concat', 'attention', 'adaptive_gated']:
        name = f"Fusion: {fusion_method}"
        print(f"\n{'='*50}\n{name}\n{'='*50}")
        config_copy = copy.deepcopy(config)
        config_copy['model']['fusion']['method'] = fusion_method
        model = GLAFatigueNet(config_copy).to(device)
        criterion = MultiTaskLoss(config_copy)
        optimizer = get_optimizer(model, config_copy)
        scheduler = get_scheduler(optimizer, config_copy)
        logger = TrainingLogger(experiment_name=f"ablation_fusion_{fusion_method}")
        trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, config_copy, logger, device)
        history = trainer.train(quick_epochs)
        val_metrics = trainer.validate(quick_epochs)
        results[name] = {
            'fatigue_acc': val_metrics['fatigue_acc'],
            'emotion_acc': val_metrics['emotion_acc'],
            'fatigue_f1': val_metrics.get('fatigue_f1', 0),
            'emotion_f1': val_metrics.get('emotion_f1', 0),
        }
        logger.close()

    # Save and plot
    plot_dir = config['training']['logging']['plot_dir']
    os.makedirs(plot_dir, exist_ok=True)
    with open(os.path.join(plot_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    try:
        plot_model_comparison(results, os.path.join(plot_dir, 'ablation_comparison.png'))
    except Exception as e:
        print(f"Plot error: {e}")

    print("\n" + "="*60 + "\nABLATION STUDY RESULTS\n" + "="*60)
    print(f"{'Model':<35} {'Fat.Acc':>8} {'Emo.Acc':>8} {'Fat.F1':>8} {'Emo.F1':>8}")
    print("-" * 75)
    for name, m in results.items():
        print(f"{name:<35} {m['fatigue_acc']:>8.4f} {m['emotion_acc']:>8.4f} {m['fatigue_f1']:>8.4f} {m['emotion_f1']:>8.4f}")
    return results

if __name__ == '__main__':
    run_ablation()
