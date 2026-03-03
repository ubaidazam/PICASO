"""Evaluate a trained PICASO model on the defense-domain KG."""

import json
import os

import numpy as np
import torch

from picaso import (
    PICASO,
    Evaluator,
    KnowledgeGraph,
    PICASOConfig,
    load_model_safe,
    set_seed,
)

# ---------- paths ----------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Defense Dataset", "Wikidata_output")
JSON_PATH = os.path.join(DATA_DIR, "entity_relations.json")
MODEL_PATH = "picaso_defense_best.pt"

# ---------- config ----------
config = PICASOConfig(dataset_name="Defense-Wikidata")
set_seed(config.seed)

# ---------- data ----------
dataset = KnowledgeGraph.from_json(JSON_PATH)

# ---------- model ----------
model = PICASO(dataset.num_entities, dataset.num_relations, dataset.num_types, config)
model = load_model_safe(model, MODEL_PATH, config.device)
model.to(config.device)
model.eval()

evaluator = Evaluator(model, dataset, config.device)

# ---------- link prediction ----------
print("\n" + "=" * 60)
print("LINK PREDICTION (Test Set)")
print("=" * 60)
lp = evaluator.link_prediction(dataset.test_triples, batch_size=config.eval_batch_size, desc="Test LP")
for k in ("MR", "MRR", "Hits@1", "Hits@3", "Hits@10"):
    print(f"  {k:8s}: {lp[k]:.4f}")

# ---------- uncertainty ----------
print("\n" + "=" * 60)
print("UNCERTAINTY QUANTIFICATION")
print("=" * 60)
unc = evaluator.uncertainty_evaluation(dataset.test_triples, sample_size=3000)
print(f"  Brier Score:          {unc['brier_score']:.4f}")
print(f"  RMSE:                 {unc['rmse']:.4f}")
print(f"  Spearman Correlation: {unc['spearman_correlation']:.4f} (p={unc['spearman_p_value']:.2e})")

# ---------- selective prediction ----------
uncertainties = unc['uncertainties']
is_correct = unc['is_correct']
sorted_idx = np.argsort(uncertainties)
overall_acc = is_correct.mean()
top25 = sorted_idx[:int(0.25 * len(sorted_idx))]
top50 = sorted_idx[:int(0.50 * len(sorted_idx))]
print(f"\nSelective Prediction:")
print(f"  Full Coverage Accuracy: {overall_acc:.4f}")
print(f"  50% Coverage Accuracy:  {is_correct[top50].mean():.4f}")
print(f"  25% Coverage Accuracy:  {is_correct[top25].mean():.4f}")

# ---------- save results ----------
results = {
    "link_prediction": {k: lp[k] for k in ("MR", "MRR", "Hits@1", "Hits@3", "Hits@10")},
    "uncertainty": {
        "brier_score": unc["brier_score"],
        "spearman_correlation": unc["spearman_correlation"],
        "rmse": unc["rmse"],
    },
}
with open("defense_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to defense_results.json")
