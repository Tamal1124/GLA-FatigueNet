"""Evaluation metrics for GLA-FatigueNet."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import torch


class MetricsCalculator:
    """Compute classification metrics for both tasks."""
    def __init__(self, fatigue_classes=None, emotion_classes=None):
        self.fatigue_classes = fatigue_classes or ['alert', 'drowsy', 'fatigued']
        self.emotion_classes = emotion_classes or ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.reset()

    def reset(self):
        self.fatigue_preds, self.fatigue_targets = [], []
        self.emotion_preds, self.emotion_targets = [], []

    def update(self, fatigue_logits, emotion_logits, fatigue_targets, emotion_targets):
        fp = torch.argmax(fatigue_logits, dim=1).cpu().numpy()
        ep = torch.argmax(emotion_logits, dim=1).cpu().numpy()
        self.fatigue_preds.extend(fp)
        self.fatigue_targets.extend(fatigue_targets.cpu().numpy())
        self.emotion_preds.extend(ep)
        self.emotion_targets.extend(emotion_targets.cpu().numpy())

    def compute(self):
        fp, ft = np.array(self.fatigue_preds), np.array(self.fatigue_targets)
        ep, et = np.array(self.emotion_preds), np.array(self.emotion_targets)
        f_acc = accuracy_score(ft, fp)
        e_acc = accuracy_score(et, ep)
        f_p, f_r, f_f1, _ = precision_recall_fscore_support(ft, fp, average='weighted', zero_division=0)
        e_p, e_r, e_f1, _ = precision_recall_fscore_support(et, ep, average='weighted', zero_division=0)
        return {
            'fatigue_acc': f_acc, 'fatigue_precision': f_p, 'fatigue_recall': f_r, 'fatigue_f1': f_f1,
            'emotion_acc': e_acc, 'emotion_precision': e_p, 'emotion_recall': e_r, 'emotion_f1': e_f1,
            'fatigue_cm': confusion_matrix(ft, fp), 'emotion_cm': confusion_matrix(et, ep),
            'fatigue_report': classification_report(ft, fp, target_names=self.fatigue_classes, zero_division=0),
            'emotion_report': classification_report(et, ep, target_names=self.emotion_classes, zero_division=0),
        }
