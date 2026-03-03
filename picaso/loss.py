"""PICASO loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PICASOLoss(nn.Module):
    """PICASO loss with Spearman correlation-based calibration."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.adv_temp = config.adversarial_temperature
        self.smoothing = config.label_smoothing
        self.adv_weight = config.adversarial_loss_weight
        self.ce_weight = config.ce_loss_weight
        self.cal_weight = config.calibration_loss_weight
        self.type_weight = config.type_loss_weight
        self.spread_weight = config.spread_loss_weight

    def adversarial_loss(self, pos_score, neg_score):
        with torch.no_grad():
            neg_weights = F.softmax(neg_score * self.adv_temp, dim=-1)
        return -(F.logsigmoid(pos_score).mean() + (neg_weights * F.logsigmoid(-neg_score)).sum(dim=-1).mean())

    def ce_loss(self, pos_score, neg_score):
        B, N = neg_score.shape
        all_scores = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
        log_probs = F.log_softmax(all_scores, dim=-1)
        targets = torch.zeros_like(log_probs)
        targets[:, 0] = 1.0 - self.smoothing
        targets[:, 1:] = self.smoothing / N
        return -(targets * log_probs).sum(dim=-1).mean()

    def _soft_rank(self, x):
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        ranks = torch.sigmoid(diff / 0.1).sum(dim=1)
        return ranks

    def spearman_calibration_loss(self, uncertainty, prediction_error):
        n = min(len(uncertainty), 1024)
        if n < 10:
            return torch.tensor(0.0, device=uncertainty.device)
        idx = torch.randperm(len(uncertainty), device=uncertainty.device)[:n]
        u = uncertainty[idx]
        e = prediction_error[idx]
        u_ranks = self._soft_rank(u)
        e_ranks = self._soft_rank(e)
        u_centered = u_ranks - u_ranks.mean()
        e_centered = e_ranks - e_ranks.mean()
        numerator = (u_centered * e_centered).sum()
        denominator = torch.sqrt((u_centered ** 2).sum() * (e_centered ** 2).sum() + 1e-8)
        spearman_corr = numerator / denominator
        cv = u.std() / (u.mean().abs() + 1e-8)
        spread_penalty = F.relu(0.3 - cv)
        return (1.0 - spearman_corr) + self.spread_weight * spread_penalty

    def forward(self, model_output, model):
        losses = {}
        pos_score = model_output['pos_score']
        if 'neg_tail_score' in model_output:
            losses['loss_tail'] = (
                self.adv_weight * self.adversarial_loss(pos_score, model_output['neg_tail_score'])
                + self.ce_weight * self.ce_loss(pos_score, model_output['neg_tail_score'])
            )
        if 'neg_head_score' in model_output and 'pos_recip_score' in model_output:
            losses['loss_head'] = (
                self.adv_weight * self.adversarial_loss(model_output['pos_recip_score'], model_output['neg_head_score'])
                + self.ce_weight * self.ce_loss(model_output['pos_recip_score'], model_output['neg_head_score'])
            )
        if 'triple_uncertainty' in model_output and 'pred_error' in model_output:
            losses['loss_calibration'] = self.cal_weight * self.spearman_calibration_loss(
                model_output['triple_uncertainty'], model_output['pred_error'],
            )
        if 'h_type_score' in model_output and 't_type_score' in model_output:
            losses['loss_type'] = self.type_weight * (
                -torch.log(model_output['h_type_score'].clamp(min=1e-8)).mean()
                - torch.log(model_output['t_type_score'].clamp(min=1e-8)).mean()
            ) / 2
        losses['total'] = sum(losses.values())
        return losses
