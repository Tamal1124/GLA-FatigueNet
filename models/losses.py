"""
Custom loss functions for GLA-FatigueNet multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in fatigue detection."""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
                focal_loss = alpha_t * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss


class LabelSmoothingCE(nn.Module):
    """Cross-Entropy with Label Smoothing for emotion classification."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return (-true_dist * log_probs).sum(dim=-1).mean()


class ConsistencyLoss(nn.Module):
    """KL-divergence consistency loss between fatigue and emotion predictions."""
    def __init__(self):
        super().__init__()
        # Mapping: fatigue_class -> expected emotion distribution
        self.fatigue_to_emotion = {
            0: [0.05, 0.02, 0.03, 0.40, 0.05, 0.40, 0.05],  # alert -> happy/surprise
            1: [0.05, 0.05, 0.05, 0.05, 0.30, 0.05, 0.45],  # drowsy -> neutral/sad
            2: [0.10, 0.15, 0.20, 0.02, 0.35, 0.03, 0.15],  # fatigued -> sad/fear/disgust
        }

    def forward(self, fatigue_logits, emotion_logits, fatigue_targets):
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        batch_size = fatigue_targets.size(0)
        device = emotion_logits.device
        target_dist = torch.zeros(batch_size, 7, device=device)
        for i in range(batch_size):
            ft = fatigue_targets[i].item()
            if ft in self.fatigue_to_emotion:
                target_dist[i] = torch.tensor(self.fatigue_to_emotion[ft], device=device)
            else:
                target_dist[i] = torch.ones(7, device=device) / 7.0
        target_dist = target_dist + 1e-8
        target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True)
        kl_loss = F.kl_div(
            torch.log(emotion_probs + 1e-8), target_dist,
            reduction='batchmean'
        )
        return kl_loss


class MultiTaskLoss(nn.Module):
    """Combined multi-task loss for GLA-FatigueNet."""
    def __init__(self, config=None):
        super().__init__()
        if config:
            loss_cfg = config['training']['loss']
            self.alpha = loss_cfg['fatigue_weight']
            self.beta = loss_cfg['emotion_weight']
            self.gamma_consistency = loss_cfg['consistency_weight']
            focal_gamma = loss_cfg['focal_gamma']
            smoothing = loss_cfg['label_smoothing']
        else:
            self.alpha, self.beta, self.gamma_consistency = 1.0, 0.8, 0.2
            focal_gamma, smoothing = 2.0, 0.1

        self.fatigue_loss = FocalLoss(gamma=focal_gamma)
        self.emotion_loss = LabelSmoothingCE(smoothing=smoothing)
        self.consistency_loss = ConsistencyLoss()

    def forward(self, fatigue_logits, emotion_logits, fatigue_targets, emotion_targets):
        l_fatigue = self.fatigue_loss(fatigue_logits, fatigue_targets)
        l_emotion = self.emotion_loss(emotion_logits, emotion_targets)
        l_consistency = self.consistency_loss(fatigue_logits, emotion_logits, fatigue_targets)
        total = self.alpha * l_fatigue + self.beta * l_emotion + self.gamma_consistency * l_consistency
        return total, {
            'fatigue_loss': l_fatigue.item(),
            'emotion_loss': l_emotion.item(),
            'consistency_loss': l_consistency.item(),
            'total_loss': total.item(),
        }
