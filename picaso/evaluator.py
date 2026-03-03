"""PICASO evaluator for knowledge graph link prediction and uncertainty."""

import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm.auto import tqdm


class Evaluator:
    """Comprehensive evaluator for knowledge graph embeddings."""

    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device

    # ==================== LINK PREDICTION ====================
    @torch.no_grad()
    def link_prediction(self, triples, batch_size=64, desc="LP Eval"):
        """Standard link prediction: MR, MRR, Hits@1/3/10."""
        self.model.eval()
        ranks_h, ranks_t = [], []
        all_ents = torch.arange(self.dataset.num_entities, device=self.device)

        for i in tqdm(range(0, len(triples), batch_size), desc=desc):
            batch = triples[i:i+batch_size]
            for h, r, t in batch:
                h_t = torch.tensor([h], device=self.device)
                r_t = torch.tensor([r], device=self.device)
                scores_t = self.model.score(
                    h_t.expand(self.dataset.num_entities),
                    r_t.expand(self.dataset.num_entities),
                    all_ents,
                )
                for other_t in self.dataset.hr_to_t.get((h, r), set()):
                    if other_t != t:
                        scores_t[other_t] = float('-inf')
                ranks_t.append((scores_t > scores_t[t]).sum().item() + 1)
                t_t = torch.tensor([t], device=self.device)
                scores_h = self.model.score_reciprocal(
                    all_ents,
                    r_t.expand(self.dataset.num_entities),
                    t_t.expand(self.dataset.num_entities),
                )
                for other_h in self.dataset.tr_to_h.get((t, r), set()):
                    if other_h != h:
                        scores_h[other_h] = float('-inf')
                ranks_h.append((scores_h > scores_h[h]).sum().item() + 1)

        all_ranks = np.array(ranks_h + ranks_t)
        return {
            'MR': float(np.mean(all_ranks)),
            'MRR': float(np.mean(1.0 / all_ranks)),
            'Hits@1': float(np.mean(all_ranks <= 1)),
            'Hits@3': float(np.mean(all_ranks <= 3)),
            'Hits@10': float(np.mean(all_ranks <= 10)),
            'ranks': all_ranks,
        }

    # ==================== TRIPLE CLASSIFICATION ====================
    @torch.no_grad()
    def triple_classification(self, test_triples, valid_triples=None):
        """Triple classification: Accuracy, Precision, Recall, F1, ROC-AUC."""
        self.model.eval()

        def generate_negatives(triples):
            negatives = []
            for h, r, t in triples:
                if np.random.random() < 0.5:
                    new_t = np.random.randint(0, self.dataset.num_entities)
                    while (h, r, new_t) in self.dataset.all_true_triples:
                        new_t = np.random.randint(0, self.dataset.num_entities)
                    negatives.append((h, r, new_t))
                else:
                    new_h = np.random.randint(0, self.dataset.num_entities)
                    while (new_h, r, t) in self.dataset.all_true_triples:
                        new_h = np.random.randint(0, self.dataset.num_entities)
                    negatives.append((new_h, r, t))
            return negatives

        def get_scores(triples):
            return np.array([
                self.model.score(
                    torch.tensor([h], device=self.device),
                    torch.tensor([r], device=self.device),
                    torch.tensor([t], device=self.device),
                ).item()
                for h, r, t in triples
            ])

        test_neg = generate_negatives(test_triples)

        if valid_triples:
            valid_neg = generate_negatives(valid_triples)
            v_pos, v_neg = get_scores(valid_triples), get_scores(valid_neg)
            all_v = np.concatenate([v_pos, v_neg])
            labels_v = np.array([1] * len(v_pos) + [0] * len(v_neg))
            best_acc, best_thresh = 0, 0
            for thresh in np.percentile(all_v, np.linspace(0, 100, 500)):
                acc = accuracy_score(labels_v, (all_v >= thresh).astype(int))
                if acc > best_acc:
                    best_acc, best_thresh = acc, thresh
        else:
            best_thresh = 0

        t_pos, t_neg = get_scores(test_triples), get_scores(test_neg)
        all_t = np.concatenate([t_pos, t_neg])
        labels_t = np.array([1] * len(t_pos) + [0] * len(t_neg))
        preds = (all_t >= best_thresh).astype(int)

        prec, rec, f1, _ = precision_recall_fscore_support(labels_t, preds, average='binary')
        try:
            auc = roc_auc_score(labels_t, all_t)
        except Exception:
            auc = 0.0

        return {
            'accuracy': accuracy_score(labels_t, preds),
            'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc,
            'threshold': best_thresh, 'scores': all_t, 'labels': labels_t,
        }

    # ==================== QUERY-BASED RANKING ====================
    @torch.no_grad()
    def query_based_ranking(self, sample_size=1000):
        """Query-based ranking evaluation: NDCG and MRR."""
        self.model.eval()
        ndcg_scores, mrr_scores = [], []

        test_sample = (
            self.dataset.test_triples[:sample_size]
            if len(self.dataset.test_triples) > sample_size
            else self.dataset.test_triples
        )

        for h, r, t in tqdm(test_sample, desc="Query Ranking"):
            h_t = torch.tensor([h], device=self.device)
            r_t = torch.tensor([r], device=self.device)

            all_ents = torch.arange(self.dataset.num_entities, device=self.device)
            scores = self.model.score(
                h_t.expand(self.dataset.num_entities),
                r_t.expand(self.dataset.num_entities),
                all_ents,
            ).cpu().numpy()

            relevance = np.zeros(self.dataset.num_entities)
            for true_t in self.dataset.hr_to_t.get((h, r), set()):
                relevance[true_t] = 1

            top_k = 10
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            dcg = sum(relevance[idx] / np.log2(i + 2) for i, idx in enumerate(ranked_indices))
            ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(int(relevance.sum()), top_k)))
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
            ndcg_scores.append(ndcg)

            ranked_all = np.argsort(scores)[::-1]
            for rank, idx in enumerate(ranked_all):
                if relevance[idx] == 1:
                    mrr_scores.append(1.0 / (rank + 1))
                    break
            else:
                mrr_scores.append(0)

        return {
            'NDCG@10': float(np.mean(ndcg_scores)),
            'MRR': float(np.mean(mrr_scores)),
            'NDCG_std': float(np.std(ndcg_scores)),
            'MRR_std': float(np.std(mrr_scores)),
        }

    # ==================== UNCERTAINTY QUANTIFICATION ====================
    @torch.no_grad()
    def uncertainty_evaluation(self, triples, sample_size=3000):
        """Comprehensive uncertainty evaluation: Brier, RMSE, Spearman, Calibration."""
        self.model.eval()
        if len(triples) > sample_size:
            indices = np.random.choice(len(triples), sample_size, replace=False)
            triples = [triples[i] for i in indices]

        uncertainties, rank_errors, is_correct, prediction_errors, model_probs = [], [], [], [], []
        all_ents = torch.arange(self.dataset.num_entities, device=self.device)

        for h, r, t in tqdm(triples, desc="Uncertainty Eval"):
            h_t = torch.tensor([h], device=self.device)
            r_t = torch.tensor([r], device=self.device)
            t_t = torch.tensor([t], device=self.device)

            uncertainty = self.model.get_triple_uncertainty(h_t, r_t, t_t).item()
            uncertainties.append(uncertainty)
            pred_error = self.model.compute_prediction_error(h_t, r_t, t_t).item()
            prediction_errors.append(pred_error)
            scores = self.model.score(
                h_t.expand(self.dataset.num_entities),
                r_t.expand(self.dataset.num_entities),
                all_ents,
            )
            probs = F.softmax(scores, dim=0)
            model_prob = probs[t].item()
            model_probs.append(model_prob)
            for other_t in self.dataset.hr_to_t.get((h, r), set()):
                if other_t != t:
                    scores[other_t] = float('-inf')
            rank = (scores > scores[t]).sum().item() + 1
            is_correct.append(1 if rank == 1 else 0)
            rank_errors.append(rank / self.dataset.num_entities)

        uncertainties = np.array(uncertainties)
        rank_errors = np.array(rank_errors)
        is_correct = np.array(is_correct)
        prediction_errors = np.array(prediction_errors)
        model_probs = np.array(model_probs)

        brier_score = np.mean((model_probs - is_correct) ** 2)
        rmse = np.sqrt(np.mean(prediction_errors))
        correlation, p_value = spearmanr(uncertainties, rank_errors)
        pearson_corr, pearson_p = pearsonr(uncertainties, rank_errors)

        n_bins = 10
        bins = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
        calibration_data = []
        for i in range(n_bins):
            mask = (uncertainties >= bins[i]) & (uncertainties < bins[i + 1])
            if mask.sum() > 0:
                calibration_data.append({
                    'bin': i + 1,
                    'mean_uncertainty': float(uncertainties[mask].mean()),
                    'mean_error': float(rank_errors[mask].mean()),
                    'accuracy': float(is_correct[mask].mean()),
                    'mean_prob': float(model_probs[mask].mean()),
                    'count': int(mask.sum()),
                })

        return {
            'brier_score': float(brier_score),
            'rmse': float(rmse),
            'spearman_correlation': float(correlation),
            'spearman_p_value': float(p_value),
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p),
            'mean_uncertainty': float(uncertainties.mean()),
            'std_uncertainty': float(uncertainties.std()),
            'uncertainty_spread': float(uncertainties.max() - uncertainties.min()),
            'calibration': calibration_data,
            'uncertainties': uncertainties,
            'rank_errors': rank_errors,
            'is_correct': is_correct,
            'prediction_errors': prediction_errors,
            'model_probs': model_probs,
        }

    # ==================== COMPLEXITY ANALYSIS ====================
    def complexity_analysis(self, num_samples=1000):
        """Complexity analysis: Time and space complexity."""
        self.model.eval()
        results = {}

        test_sample = self.dataset.test_triples[:num_samples]

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()

        with torch.no_grad():
            for h, r, t in test_sample:
                h_t = torch.tensor([h], device=self.device)
                r_t = torch.tensor([r], device=self.device)
                t_t = torch.tensor([t], device=self.device)
                _ = self.model.score(h_t, r_t, t_t)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_time = time.perf_counter() - start_time

        results['inference_time_total'] = inference_time
        results['inference_time_per_triple'] = inference_time / num_samples
        results['throughput_triples_per_second'] = num_samples / inference_time

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        param_memory_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)

        results['total_parameters'] = total_params
        results['trainable_parameters'] = trainable_params
        results['model_memory_mb'] = param_memory_mb

        if torch.cuda.is_available():
            results['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
            results['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)

        process = psutil.Process()
        results['cpu_memory_mb'] = process.memory_info().rss / (1024 ** 2)

        n_entities = self.dataset.num_entities
        n_relations = self.dataset.num_relations
        dim = self.model.dim

        results['big_o_analysis'] = {
            'scoring_single_triple': f'O({dim})',
            'scoring_all_tails': f'O({n_entities} * {dim})',
            'entity_embedding_space': f'O({n_entities} * {dim} * 2)',
            'relation_embedding_space': f'O({n_relations} * {dim} * 4)',
            'total_space': f'O(({n_entities} + {n_relations}) * {dim})',
        }

        return results
