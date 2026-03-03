"""Train PICASO on the defense-domain Wikidata knowledge graph."""

import os

from torch.utils.data import DataLoader

from picaso import (
    PICASO,
    KnowledgeGraph,
    PICASOConfig,
    PICASOTrainer,
    TripleDataset,
    compute_entity_frequencies,
    set_seed,
)

# ---------- paths ----------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Defense Dataset", "Wikidata_output")
JSON_PATH = os.path.join(DATA_DIR, "entity_relations.json")
SAVE_PATH = "picaso_defense_best.pt"

# ---------- config ----------
config = PICASOConfig(dataset_name="Defense-Wikidata")
set_seed(config.seed)

# ---------- data ----------
dataset = KnowledgeGraph.from_json(JSON_PATH)

train_ds = TripleDataset(
    dataset.train_triples, dataset.num_entities, dataset.num_relations,
    neg_size=config.negative_samples,
    hr_to_t=dataset.hr_to_t, tr_to_h=dataset.tr_to_h,
    rel_head_type=dataset.relation_head_type,
    rel_tail_type=dataset.relation_tail_type,
    entity_types=dataset.entity_types,
    use_type_constraint=config.use_type_constraint,
    entity_inv_freq=dataset.entity_inv_freq,
)
loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

# ---------- model ----------
model = PICASO(dataset.num_entities, dataset.num_relations, dataset.num_types, config)
freqs = compute_entity_frequencies(dataset.train_triples, dataset.num_entities)
model.entities.initialize_with_frequencies(freqs, alpha=3.0)
print(f"Entities: {dataset.num_entities:,} | Relations: {dataset.num_relations} | Types: {dataset.num_types}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---------- train ----------
trainer = PICASOTrainer(model, dataset, config)
history = trainer.train(loader, save_path=SAVE_PATH)
print(f"\nBest validation MRR: {trainer.best_mrr:.4f}")
print(f"Model saved to {SAVE_PATH}")
