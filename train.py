"""
GLA-FatigueNet: Main Training Entry Point
==========================================
Usage:
    python train.py                    # Train with default config
    python train.py --config path.yaml # Train with custom config
    python train.py --smoke-test       # Quick 3-epoch smoke test
"""

import os
import sys
import argparse
import torch

from utils.helpers import load_config, set_seed, get_device, ensure_dirs, count_parameters, format_params
from utils.logger import TrainingLogger
from models.gla_fatiguenet import GLAFatigueNet
from models.losses import MultiTaskLoss
from data.dataset import create_dataloaders
from training.trainer import Trainer
from training.optimizer import get_optimizer, get_scheduler
from evaluation.visualize import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description='Train GLA-FatigueNet')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--smoke-test', action='store_true', help='Quick 3-epoch test')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    set_seed(config['project']['seed'])
    device = get_device(config)

    # Create directories
    ensure_dirs(
        config['training']['checkpoint']['save_dir'],
        config['training']['logging']['log_dir'],
        config['training']['logging']['plot_dir'],
    )

    # Logger
    logger = TrainingLogger(
        log_dir=config['training']['logging']['log_dir'],
        experiment_name=config['project']['name']
    )
    if config['training']['logging'].get('use_tensorboard', False):
        logger.setup_tensorboard()

    logger.info(f"{'='*60}")
    logger.info(f"  GLA-FatigueNet Training")
    logger.info(f"  Device: {device}")
    logger.info(f"{'='*60}")

    # Data
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Model
    logger.info("Building model...")
    model = GLAFatigueNet(config).to(device)
    total, trainable = count_parameters(model)
    logger.log_model_summary(total, trainable)
    logger.info(f"Model size: {format_params(total)} total, {format_params(trainable)} trainable")

    # Loss, optimizer, scheduler
    criterion = MultiTaskLoss(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        from utils.helpers import load_checkpoint
        start_epoch, _ = load_checkpoint(model, args.resume, optimizer, scheduler)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Epochs
    num_epochs = 3 if args.smoke_test else config['training']['epochs']
    if args.smoke_test:
        logger.info("*** SMOKE TEST MODE: 3 epochs ***")

    # Train
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, config, logger, device)
    history = trainer.train(num_epochs)

    # Plot training curves
    try:
        plot_training_curves(history, config['training']['logging']['plot_dir'])
    except Exception as e:
        logger.warning(f"Could not generate training plots: {e}")

    # Final evaluation on test set
    logger.info("\nRunning final evaluation on test set...")
    from training.metrics import MetricsCalculator
    metrics_calc = MetricsCalculator(config['data']['fatigue_classes'], config['data']['emotion_classes'])
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            geo = batch['geometric_features'].to(device)
            outputs = model(images, geo)
            metrics_calc.update(outputs['fatigue_logits'], outputs['emotion_logits'],
                                batch['fatigue_label'], batch['emotion_label'])
    test_metrics = metrics_calc.compute()

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Fatigue Accuracy: {test_metrics['fatigue_acc']:.4f}")
    logger.info(f"Fatigue F1:       {test_metrics['fatigue_f1']:.4f}")
    logger.info(f"Emotion Accuracy: {test_metrics['emotion_acc']:.4f}")
    logger.info(f"Emotion F1:       {test_metrics['emotion_f1']:.4f}")
    logger.info(f"\nFatigue Report:\n{test_metrics['fatigue_report']}")
    logger.info(f"Emotion Report:\n{test_metrics['emotion_report']}")

    logger.close()
    print("\n[DONE] Training complete! Check results/ directory for outputs.")


if __name__ == '__main__':
    # Windows multiprocessing fix
    if sys.platform == 'win32':
        torch.multiprocessing.set_start_method('spawn', force=True)
    main()
