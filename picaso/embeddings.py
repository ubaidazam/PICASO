"""PICASO embedding modules."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianEmbedding(nn.Module):
    """Entity embeddings as Gaussian distributions N(mu, Sigma)."""

    def __init__(self, num_entities, dim, init_std=1e-3, init_var=0.5, min_var=0.02, max_var=3.0):
        super().__init__()
        self.num_entities, self.dim = num_entities, dim
        self.min_var, self.max_var, self.init_var = min_var, max_var, init_var
        self.mu = nn.Embedding(num_entities, dim)
        self.log_var = nn.Embedding(num_entities, dim)
        nn.init.uniform_(self.mu.weight, -init_std, init_std)
        nn.init.constant_(self.log_var.weight, math.log(init_var))

    def initialize_with_frequencies(self, entity_frequencies, alpha=1.0):
        freq = entity_frequencies.astype(np.float32)
        freq_normalized = (freq - freq.min()) / (freq.max() - freq.min() + 1e-8)
        init_vars = self.init_var * (1 + alpha * (1 - freq_normalized))
        init_vars = np.clip(init_vars, self.min_var, self.max_var)
        log_var_tensor = torch.tensor(np.log(init_vars), dtype=torch.float32).unsqueeze(1).expand(-1, self.dim)
        with torch.no_grad():
            self.log_var.weight.copy_(log_var_tensor)
        print(f"  Frequency-aware variance: min={init_vars.min():.4f}, max={init_vars.max():.4f}")

    def forward(self, ids):
        mu = self.mu(ids)
        var = torch.clamp(F.softplus(self.log_var(ids)) + self.min_var, max=self.max_var)
        return mu, var

    def get_uncertainty(self, ids):
        return self.forward(ids)[1].sum(dim=-1)

    def get_all_variances(self):
        all_ids = torch.arange(self.num_entities, device=self.mu.weight.device)
        _, var = self.forward(all_ids)
        return var


class SemanticTypeEmbedding(nn.Module):
    """Semantic Types as Probabilistic Regions."""

    def __init__(self, num_types, dim, init_var=1.0, min_var=0.1, max_var=5.0):
        super().__init__()
        self.num_types, self.dim, self.min_var, self.max_var = num_types, dim, min_var, max_var
        if num_types > 0:
            self.mu = nn.Embedding(num_types, dim)
            self.log_var = nn.Embedding(num_types, dim)
            nn.init.xavier_uniform_(self.mu.weight)
            nn.init.constant_(self.log_var.weight, math.log(init_var))
        else:
            self.mu, self.log_var = None, None

    def compute_membership(self, entity_mu, entity_var, type_ids, kernel="gaussian"):
        if self.mu is None:
            return (
                torch.zeros(entity_mu.shape[0], 1, device=entity_mu.device),
                torch.ones(entity_mu.shape[0], 1, device=entity_mu.device),
            )
        valid_mask = (type_ids >= 0).float()
        type_ids_safe = type_ids.clamp(min=0)
        type_mu = self.mu(type_ids_safe)
        type_var = torch.clamp(F.softplus(self.log_var(type_ids_safe)) + self.min_var, max=self.max_var)
        if kernel == "gaussian":
            combined_var = entity_var.unsqueeze(1) + type_var
            diff = entity_mu.unsqueeze(1) - type_mu
            mahal_dist = (diff ** 2 / (combined_var + 1e-8)).sum(dim=-1)
            membership = torch.exp(-0.5 * mahal_dist)
        else:
            membership = torch.ones(entity_mu.shape[0], type_ids.shape[1], device=entity_mu.device)
        return membership * valid_mask, valid_mask
