"""PICASO model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import GaussianEmbedding, SemanticTypeEmbedding
from .relations import RelationTransform, DistributionSimilarity


class PICASO(nn.Module):
    """Probabilistic Conceptual Spaces Sensemaking under Uncertainty."""

    def __init__(self, num_entities, num_relations, num_types, config):
        super().__init__()
        self.num_entities, self.num_relations, self.num_types = num_entities, num_relations, num_types
        self.config, self.dim = config, config.embedding_dim

        self.entities = GaussianEmbedding(
            num_entities, config.embedding_dim, config.entity_init_std,
            config.entity_init_var, config.min_var, config.max_var,
        )
        self.types = (
            SemanticTypeEmbedding(num_types, config.type_embedding_dim)
            if config.use_semantic_types and num_types > 0
            else None
        )
        self.rel_transform = RelationTransform(
            num_relations, config.embedding_dim, config.relation_init_var,
            config.use_rotation, config.use_scaling, config.min_var, config.max_var,
        )
        if config.use_reciprocal:
            self.rel_transform_inv = RelationTransform(
                num_relations, config.embedding_dim, config.relation_init_var,
                config.use_rotation, config.use_scaling, config.min_var, config.max_var,
            )
        self.dist_similarity = DistributionSimilarity(config.scoring_method)
        self.rel_diag = nn.Embedding(num_relations, config.embedding_dim)
        nn.init.ones_(self.rel_diag.weight)
        self.rel_re = nn.Embedding(num_relations, config.embedding_dim // 2)
        self.rel_im = nn.Embedding(num_relations, config.embedding_dim // 2)
        nn.init.xavier_uniform_(self.rel_re.weight)
        nn.init.xavier_uniform_(self.rel_im.weight)
        self.gamma = nn.Parameter(torch.tensor([config.margin]))
        if config.use_ensemble:
            self.ensemble_weights = nn.Parameter(torch.tensor([
                config.geometric_weight, config.translational_weight,
                config.kl_weight, config.bilinear_weight, config.complex_weight,
            ]))

    def _geometric_score(self, h_mu, h_var, t_mu, t_var, rel_ids):
        pred_mu, pred_var = self.rel_transform.transform(h_mu, h_var, rel_ids)
        return self.gamma - ((pred_mu - t_mu) ** 2 / (pred_var + t_var + 1e-8)).sum(dim=-1)

    def _translational_score(self, h_mu, h_var, t_mu, t_var, rel_ids):
        rel_t, rel_v = self.rel_transform(rel_ids)
        return self.gamma - ((h_mu + rel_t - t_mu) ** 2 / (h_var + rel_v + t_var + 1e-8)).sum(dim=-1)

    def _kl_score(self, h_mu, h_var, t_mu, t_var, rel_ids):
        pred_mu, pred_var = self.rel_transform.transform(h_mu, h_var, rel_ids)
        return self.dist_similarity(pred_mu, pred_var, t_mu, t_var)

    def _bilinear_score(self, h_mu, t_mu, rel_ids):
        return (h_mu * self.rel_diag(rel_ids) * t_mu).sum(dim=-1)

    def _complex_score(self, h_mu, t_mu, rel_ids):
        h_re, h_im = h_mu.chunk(2, dim=-1)
        t_re, t_im = t_mu.chunk(2, dim=-1)
        r_re, r_im = self.rel_re(rel_ids), self.rel_im(rel_ids)
        return (
            h_re * r_re * t_re + h_im * r_re * t_im
            + h_re * r_im * t_im - h_im * r_im * t_re
        ).sum(dim=-1)

    def score(self, h_ids, r_ids, t_ids, mode='tail'):
        h_mu, h_var = self.entities(h_ids)
        t_mu, t_var = self.entities(t_ids)
        if self.config.use_ensemble:
            w = F.softmax(self.ensemble_weights, dim=0)
            return (
                w[0] * self._geometric_score(h_mu, h_var, t_mu, t_var, r_ids)
                + w[1] * self._translational_score(h_mu, h_var, t_mu, t_var, r_ids)
                + w[2] * self._kl_score(h_mu, h_var, t_mu, t_var, r_ids)
                + w[3] * self._bilinear_score(h_mu, t_mu, r_ids)
                + w[4] * self._complex_score(h_mu, t_mu, r_ids)
            )
        return self._kl_score(h_mu, h_var, t_mu, t_var, r_ids)

    def score_reciprocal(self, h_ids, r_ids, t_ids):
        if not self.config.use_reciprocal:
            return self.score(h_ids, r_ids, t_ids)
        h_mu, h_var = self.entities(h_ids)
        t_mu, t_var = self.entities(t_ids)
        pred_mu, pred_var = self.rel_transform_inv.transform(t_mu, t_var, r_ids)
        return self.dist_similarity(pred_mu, pred_var, h_mu, h_var)

    def compute_type_score(self, entity_ids, type_ids):
        if self.types is None:
            return torch.zeros(entity_ids.shape[0], device=entity_ids.device)
        e_mu, e_var = self.entities(entity_ids)
        membership, valid_mask = self.types.compute_membership(e_mu, e_var, type_ids)
        return membership.sum(dim=-1) / valid_mask.sum(dim=-1).clamp(min=1)

    def compute_prediction_error(self, h_ids, r_ids, t_ids):
        h_mu, h_var = self.entities(h_ids)
        t_mu, _ = self.entities(t_ids)
        pred_mu, _ = self.rel_transform.transform(h_mu, h_var, r_ids)
        return (pred_mu - t_mu).pow(2).sum(dim=-1)

    def get_triple_uncertainty(self, h_ids, r_ids, t_ids):
        """Compute calibrated uncertainty for a triple."""
        h_mu, h_var = self.entities(h_ids)
        t_mu, t_var = self.entities(t_ids)
        pred_mu, pred_var = self.rel_transform.transform(h_mu, h_var, r_ids)
        epistemic = pred_var.mean(dim=-1) + t_var.mean(dim=-1)
        diff_sq = (pred_mu - t_mu) ** 2
        combined_var = pred_var + t_var + 1e-8
        aleatoric = (diff_sq / combined_var).mean(dim=-1)
        raw_error = diff_sq.mean(dim=-1)
        aleatoric_weight = getattr(self.config, 'aleatoric_weight', 5.0)
        combined = epistemic + aleatoric_weight * (aleatoric + 0.1 * raw_error)
        return torch.log1p(combined)

    def get_entity_uncertainty(self, entity_ids):
        _, var = self.entities(entity_ids)
        return var.mean(dim=-1)

    def forward(self, h, r, t, neg_t=None, neg_h=None, h_types=None, t_types=None):
        results = {
            'pos_score': self.score(h, r, t),
            'pred_error': self.compute_prediction_error(h, r, t).detach(),
            'triple_uncertainty': self.get_triple_uncertainty(h, r, t),
        }
        if self.types is not None and h_types is not None and t_types is not None:
            results['h_type_score'] = self.compute_type_score(h, h_types)
            results['t_type_score'] = self.compute_type_score(t, t_types)
        if neg_t is not None:
            B, N = neg_t.shape
            results['neg_tail_score'] = self.score(
                h.unsqueeze(1).expand(-1, N).reshape(-1),
                r.unsqueeze(1).expand(-1, N).reshape(-1),
                neg_t.reshape(-1),
            ).reshape(B, N)
        if neg_h is not None:
            B, N = neg_h.shape
            results['neg_head_score'] = self.score_reciprocal(
                neg_h.reshape(-1),
                r.unsqueeze(1).expand(-1, N).reshape(-1),
                t.unsqueeze(1).expand(-1, N).reshape(-1),
            ).reshape(B, N)
            results['pos_recip_score'] = self.score_reciprocal(h, r, t)
        return results
