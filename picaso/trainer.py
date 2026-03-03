"""PICASO trainer."""

import time
from collections import defaultdict

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm

from .evaluator import Evaluator
from .loss import PICASOLoss


class PICASOTrainer:
    """End-to-end trainer for PICASO models."""

    def __init__(self, model, dataset, config):
        self.model = model.to(config.device)
        self.dataset = dataset
        self.config = config
        self.device = config.device
        self.loss_fn = PICASOLoss(config)
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scaler = GradScaler()
        self.evaluator = Evaluator(model, dataset, config.device)
        self.history = defaultdict(list)
        self.best_mrr = 0
        self.patience_counter = 0

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_losses = defaultdict(float)
        self.optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(pbar):
            h = batch['head'].to(self.device)
            r = batch['relation'].to(self.device)
            t = batch['tail'].to(self.device)
            neg_t = batch['neg_tails'].to(self.device)
            neg_h = batch['neg_heads'].to(self.device)
            h_types = batch['head_types'].to(self.device) if 'head_types' in batch else None
            t_types = batch['tail_types'].to(self.device) if 'tail_types' in batch else None
            with autocast():
                out = self.model(h, r, t, neg_t, neg_h, h_types, t_types)
                losses = self.loss_fn(out, self.model)
                loss = losses['total'] / self.config.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            for k, v in losses.items():
                total_losses[k] += v.item()
            pbar.set_postfix({'loss': f'{losses["total"].item():.4f}'})
        return {k: v / len(loader) for k, v in total_losses.items()}

    def train(self, loader, epochs=None, eval_every=None, save_path='picaso_best.pt'):
        epochs = epochs or self.config.epochs
        eval_every = eval_every or self.config.eval_every
        print(f"\nTraining PICASO on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.cosine_T0,
            T_mult=self.config.cosine_T_mult,
            eta_min=1e-6,
        )
        start = time.time()
        for epoch in range(1, epochs + 1):
            losses = self.train_epoch(loader, epoch)
            for k, v in losses.items():
                self.history[k].append(v)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            scheduler.step()
            print(f"Epoch {epoch}: Loss={losses['total']:.4f}, LR={self.optimizer.param_groups[0]['lr']:.2e}")
            if epoch % eval_every == 0:
                metrics = self.evaluator.link_prediction(
                    self.dataset.valid_triples[:1000], desc="Valid",
                )
                self.history['mrr'].append(metrics['MRR'])
                self.history['hits10'].append(metrics['Hits@10'])
                print(f"  Valid: MRR={metrics['MRR']:.4f}, H@10={metrics['Hits@10']:.4f}")
                if metrics['MRR'] > self.best_mrr:
                    self.best_mrr = metrics['MRR']
                    self.patience_counter = 0
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  Saved best model (MRR={self.best_mrr:.4f})")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.patience // eval_every:
                        print(f"\nEarly stopping at epoch {epoch}")
                        break
        print(f"\nTraining completed in {(time.time() - start) / 60:.1f} min")
        return dict(self.history)
