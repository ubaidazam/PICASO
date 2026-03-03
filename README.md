# PICASO

**Probabilistic Conceptual Spaces Sensemaking under Uncertainty**

A pip-installable Python package for knowledge graph embedding with built-in uncertainty quantification. PICASO represents entities as Gaussian distributions rather than point vectors, enabling calibrated confidence estimates for every prediction.

## Installation

```bash
pip install git+https://github.com/ubaidazam/PICASO.git
```

## Quick Start

```python
from torch.utils.data import DataLoader
from picaso import *

# 1. Configure
config = PICASOConfig(embedding_dim=100, epochs=10, eval_every=5)
set_seed(config.seed)

# 2. Load data (from triples, JSON, or TSV)
triples = [(0, 0, 1), (1, 0, 2), (2, 1, 3), (3, 1, 0), (0, 2, 3)]
dataset = KnowledgeGraph.from_triples(triples, num_entities=4, num_relations=3)

# 3. Create model and train
model = PICASO(dataset.num_entities, dataset.num_relations, dataset.num_types, config)
freqs = compute_entity_frequencies(dataset.train_triples, dataset.num_entities)
model.entities.initialize_with_frequencies(freqs)

train_ds = TripleDataset(
    dataset.train_triples, dataset.num_entities, dataset.num_relations,
    neg_size=config.negative_samples, hr_to_t=dataset.hr_to_t,
    tr_to_h=dataset.tr_to_h, entity_inv_freq=dataset.entity_inv_freq,
)
loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

trainer = PICASOTrainer(model, dataset, config)
trainer.train(loader)

# 4. Evaluate
evaluator = Evaluator(model, dataset, config.device)
results = evaluator.link_prediction(dataset.test_triples)
print(f"MRR: {results['MRR']:.4f}, Hits@10: {results['Hits@10']:.4f}")
```

## Loading Your Own Data

```python
# From JSON (Wikidata extraction format)
dataset = KnowledgeGraph.from_json("entity_relations.json")

# From TSV files (head \t relation \t tail)
dataset = KnowledgeGraph.from_tsv("train.tsv", "valid.tsv", "test.tsv")

# From integer triples directly
dataset = KnowledgeGraph.from_triples(triples, num_entities=100, num_relations=10)
```

## Key Features

- **Uncertainty Quantification** — Every prediction comes with a calibrated confidence score, enabling risk-aware decision-making
- **Gaussian Entity Embeddings** — Entities are distributions (mean + variance), not just point vectors
- **Ensemble Scoring** — Combines geometric, translational, KL-divergence, bilinear, and complex scoring methods
- **Selective Prediction** — Filter predictions by confidence to improve accuracy on the subset you act on
- **Domain Agnostic** — Works with any knowledge graph (defense, biomedical, finance, etc.)

## Package Structure

| Module | Description |
|--------|-------------|
| `PICASOConfig` | All hyperparameters in one dataclass |
| `KnowledgeGraph` | Data loading with `from_json()`, `from_triples()`, `from_tsv()` |
| `PICASO` | The core model |
| `PICASOTrainer` | End-to-end training with early stopping and LR scheduling |
| `Evaluator` | Link prediction, triple classification, uncertainty evaluation |
| `PICASOLoss` | Adversarial + cross-entropy + calibration loss |

## Evaluation Metrics

```python
evaluator = Evaluator(model, dataset, config.device)

# Link prediction: MR, MRR, Hits@1/3/10
lp = evaluator.link_prediction(dataset.test_triples)

# Uncertainty: Brier score, RMSE, Spearman correlation
unc = evaluator.uncertainty_evaluation(dataset.test_triples)

# Triple classification: Accuracy, Precision, Recall, F1, ROC-AUC
tc = evaluator.triple_classification(dataset.test_triples, dataset.valid_triples)
```

## Examples

See the `examples/` directory:
- `quickstart.py` — Minimal working example on synthetic data
- `defense_domain/train_defense.py` — Full training on defense knowledge graph
- `defense_domain/evaluate_defense.py` — Evaluation and selective prediction analysis

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- NumPy, SciPy, scikit-learn, tqdm, psutil
