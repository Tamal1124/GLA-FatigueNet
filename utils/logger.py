"""
Training logger for GLA-FatigueNet.
Supports console logging and TensorBoard.
"""

import os
import json
import logging
from datetime import datetime


class TrainingLogger:
    """Comprehensive training logger with file and console output."""

    def __init__(self, log_dir="./results/logs", experiment_name=None):
        os.makedirs(log_dir, exist_ok=True)

        if experiment_name is None:
            experiment_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}.log")
        self.metrics_file = os.path.join(log_dir, f"{experiment_name}_metrics.json")
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_fatigue_acc': [], 'val_fatigue_acc': [],
            'train_emotion_acc': [], 'val_emotion_acc': [],
            'lr': [],
        }

        # Setup Python logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # TensorBoard writer (optional)
        self.tb_writer = None

    def setup_tensorboard(self, tb_log_dir=None):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            if tb_log_dir is None:
                tb_log_dir = os.path.join(self.log_dir, "tensorboard", self.experiment_name)
            self.tb_writer = SummaryWriter(tb_log_dir)
            self.info(f"TensorBoard logging to: {tb_log_dir}")
        except ImportError:
            self.warning("TensorBoard not available. Skipping.")

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def log_epoch(self, epoch, train_metrics, val_metrics, lr):
        """Log epoch-level metrics."""
        self.history['train_loss'].append(train_metrics.get('loss', 0))
        self.history['val_loss'].append(val_metrics.get('loss', 0))
        self.history['train_fatigue_acc'].append(train_metrics.get('fatigue_acc', 0))
        self.history['val_fatigue_acc'].append(val_metrics.get('fatigue_acc', 0))
        self.history['train_emotion_acc'].append(train_metrics.get('emotion_acc', 0))
        self.history['val_emotion_acc'].append(val_metrics.get('emotion_acc', 0))
        self.history['lr'].append(lr)

        msg = (
            f"Epoch [{epoch}] | "
            f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
            f"Val Loss: {val_metrics.get('loss', 0):.4f} | "
            f"Train Fatigue Acc: {train_metrics.get('fatigue_acc', 0):.4f} | "
            f"Val Fatigue Acc: {val_metrics.get('fatigue_acc', 0):.4f} | "
            f"Train Emotion Acc: {train_metrics.get('emotion_acc', 0):.4f} | "
            f"Val Emotion Acc: {val_metrics.get('emotion_acc', 0):.4f} | "
            f"LR: {lr:.6f}"
        )
        self.info(msg)

        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar('Loss/train', train_metrics.get('loss', 0), epoch)
            self.tb_writer.add_scalar('Loss/val', val_metrics.get('loss', 0), epoch)
            self.tb_writer.add_scalar('Accuracy/train_fatigue', train_metrics.get('fatigue_acc', 0), epoch)
            self.tb_writer.add_scalar('Accuracy/val_fatigue', val_metrics.get('fatigue_acc', 0), epoch)
            self.tb_writer.add_scalar('Accuracy/train_emotion', train_metrics.get('emotion_acc', 0), epoch)
            self.tb_writer.add_scalar('Accuracy/val_emotion', val_metrics.get('emotion_acc', 0), epoch)
            self.tb_writer.add_scalar('LR', lr, epoch)

    def log_model_summary(self, total_params, trainable_params):
        """Log model parameter summary."""
        self.info(f"Total Parameters: {total_params:,}")
        self.info(f"Trainable Parameters: {trainable_params:,}")
        self.info(f"Non-trainable Parameters: {total_params - trainable_params:,}")

    def save_history(self):
        """Save training history to JSON."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        self.info(f"Training history saved to {self.metrics_file}")

    def get_history(self):
        """Return training history."""
        return self.history

    def close(self):
        """Close all handlers."""
        if self.tb_writer:
            self.tb_writer.close()
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
