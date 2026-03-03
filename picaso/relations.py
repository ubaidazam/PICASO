"""PICASO relation transform and distribution similarity modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationTransform(nn.Module):
    """Relations as probabilistic transformations."""

    def __init__(self, num_relations, dim, init_var=0.3, use_rotation=True, use_scaling=True, min_var=0.02, max_var=3.0):
        super().__init__()
        self.num_relations, self.dim = num_relations, dim
        self.use_rotation, self.use_scaling = use_rotation, use_scaling
        self.min_var, self.max_var = min_var, max_var
        self.translation = nn.Embedding(num_relations, dim)
        self.log_var = nn.Embedding(num_relations, dim)
        if use_rotation:
            self.phase = nn.Embedding(num_relations, dim // 2)
        if use_scaling:
            self.log_scale = nn.Embedding(num_relations, dim)
        nn.init.uniform_(self.translation.weight, -0.1, 0.1)
        nn.init.constant_(self.log_var.weight, math.log(init_var))
        if use_rotation:
            nn.init.uniform_(self.phase.weight, -math.pi, math.pi)
        if use_scaling:
            nn.init.zeros_(self.log_scale.weight)

    def get_var(self, ids):
        return torch.clamp(F.softplus(self.log_var(ids)) + self.min_var, max=self.max_var)

    def transform(self, mu, var, rel_ids):
        if self.use_rotation:
            phase = self.phase(rel_ids)
            re, im = mu.chunk(2, dim=-1)
            cos_p, sin_p = torch.cos(phase), torch.sin(phase)
            mu = torch.cat([re * cos_p - im * sin_p, re * sin_p + im * cos_p], dim=-1)
        if self.use_scaling:
            scale = torch.exp(torch.clamp(self.log_scale(rel_ids), -2, 2))
            mu, var = mu * scale, var * (scale ** 2)
        return mu + self.translation(rel_ids), var + self.get_var(rel_ids)

    def forward(self, ids):
        return self.translation(ids), self.get_var(ids)


class DistributionSimilarity(nn.Module):
    """Distribution similarity measures: kl_divergence, hellinger, wasserstein, l2."""

    def __init__(self, method="kl_divergence"):
        super().__init__()
        self.method = method

    def forward(self, mu1, var1, mu2, var2):
        if self.method == "kl_divergence":
            return -(self._kl(mu1, var1, mu2, var2) + self._kl(mu2, var2, mu1, var1)) / 2
        elif self.method == "hellinger":
            return -self._hellinger(mu1, var1, mu2, var2)
        elif self.method == "wasserstein":
            return -self._wasserstein(mu1, var1, mu2, var2)
        return -((mu1 - mu2) ** 2 / (var1 + var2 + 1e-8)).sum(dim=-1)

    def _kl(self, mu1, var1, mu2, var2):
        d = mu1.shape[-1]
        return 0.5 * (
            (var1 / (var2 + 1e-8)).sum(dim=-1)
            + ((mu2 - mu1) ** 2 / (var2 + 1e-8)).sum(dim=-1)
            - d
            + (torch.log(var2 + 1e-8) - torch.log(var1 + 1e-8)).sum(dim=-1)
        )

    def _hellinger(self, mu1, var1, mu2, var2):
        var_mean = (var1 + var2) / 2
        log_bc = (
            0.25 * (torch.log(var1 + 1e-8) + torch.log(var2 + 1e-8) - 2 * torch.log(var_mean + 1e-8)).sum(dim=-1)
            - 0.25 * ((mu1 - mu2) ** 2 / (var_mean + 1e-8)).sum(dim=-1)
        )
        return torch.sqrt((1 - torch.exp(log_bc.clamp(max=0))).clamp(min=0))

    def _wasserstein(self, mu1, var1, mu2, var2):
        return torch.sqrt(
            ((mu1 - mu2) ** 2).sum(dim=-1)
            + ((torch.sqrt(var1) - torch.sqrt(var2)) ** 2).sum(dim=-1)
            + 1e-8
        )
