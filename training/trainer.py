"""Training loop for GLA-FatigueNet."""
import time, os, torch
from tqdm import tqdm
from utils.helpers import AverageMeter, EarlyStopping, save_checkpoint
from training.metrics import MetricsCalculator


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config, logger, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.logger = logger
        self.device = device
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        es_cfg = config['training']['early_stopping']
        self.early_stopping = EarlyStopping(patience=es_cfg['patience'], min_delta=es_cfg['min_delta'])
        self.save_dir = config['training']['checkpoint']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics_calc = MetricsCalculator(config['data']['fatigue_classes'], config['data']['emotion_classes'])

    def train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter('Loss')
        self.metrics_calc.reset()
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}', leave=False)
        for batch in pbar:
            images = batch['image'].to(self.device)
            geo = batch['geometric_features'].to(self.device)
            fat_labels = batch['fatigue_label'].to(self.device)
            emo_labels = batch['emotion_label'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images, geo)
            loss, loss_dict = self.criterion(outputs['fatigue_logits'], outputs['emotion_logits'], fat_labels, emo_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            loss_meter.update(loss.item(), images.size(0))
            self.metrics_calc.update(outputs['fatigue_logits'].detach(), outputs['emotion_logits'].detach(), fat_labels, emo_labels)
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        metrics = self.metrics_calc.compute()
        return {'loss': loss_meter.avg, 'fatigue_acc': metrics['fatigue_acc'], 'emotion_acc': metrics['emotion_acc']}

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        loss_meter = AverageMeter('Loss')
        self.metrics_calc.reset()
        for batch in tqdm(self.val_loader, desc=f'Val Epoch {epoch}', leave=False):
            images = batch['image'].to(self.device)
            geo = batch['geometric_features'].to(self.device)
            fat_labels = batch['fatigue_label'].to(self.device)
            emo_labels = batch['emotion_label'].to(self.device)
            outputs = self.model(images, geo)
            loss, _ = self.criterion(outputs['fatigue_logits'], outputs['emotion_logits'], fat_labels, emo_labels)
            loss_meter.update(loss.item(), images.size(0))
            self.metrics_calc.update(outputs['fatigue_logits'], outputs['emotion_logits'], fat_labels, emo_labels)
        metrics = self.metrics_calc.compute()
        return {'loss': loss_meter.avg, 'fatigue_acc': metrics['fatigue_acc'], 'emotion_acc': metrics['emotion_acc'],
                'fatigue_f1': metrics['fatigue_f1'], 'emotion_f1': metrics['emotion_f1']}

    def train(self, num_epochs):
        self.logger.info(f"Starting training for {num_epochs} epochs on {self.device}")
        start = time.time()
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            self.logger.log_epoch(epoch, train_metrics, val_metrics, lr)
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_acc = val_metrics['fatigue_acc']
                save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, val_metrics,
                                os.path.join(self.save_dir, 'best_model.pth'))
            if epoch % self.config['training']['checkpoint'].get('save_every', 10) == 0:
                save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, val_metrics,
                                os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth'))
            if self.early_stopping(val_metrics['loss']):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        elapsed = time.time() - start
        self.logger.info(f"Training complete in {elapsed/60:.1f} min. Best val loss: {self.best_val_loss:.4f}, Best val acc: {self.best_val_acc:.4f}")
        self.logger.save_history()
        return self.logger.get_history()
