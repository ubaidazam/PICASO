"""Minimal PICASO quickstart: train on synthetic triples."""

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

# 1. Configuration
config = PICASOConfig(
    dataset_name="quickstart",
    embedding_dim=100,
    epochs=10,
    eval_every=5,
    batch_size=64,
    negative_samples=32,
)
set_seed(config.seed)

# 2. Build a tiny synthetic KG (replace with your own data)
triples = [(0, 0, 1), (1, 0, 2), (2, 1, 3), (3, 1, 0), (0, 2, 3),
            (1, 2, 0), (2, 0, 3), (3, 2, 1), (0, 1, 2), (1, 1, 3)]
dataset = KnowledgeGraph.from_triples(triples, num_entities=4, num_relations=3)

# 3. Create data loader
train_ds = TripleDataset(
    dataset.train_triples, dataset.num_entities, dataset.num_relations,
    neg_size=config.negative_samples, hr_to_t=dataset.hr_to_t,
    tr_to_h=dataset.tr_to_h, entity_inv_freq=dataset.entity_inv_freq,
)
loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

# 4. Initialize model
model = PICASO(dataset.num_entities, dataset.num_relations, dataset.num_types, config)
freqs = compute_entity_frequencies(dataset.train_triples, dataset.num_entities)
model.entities.initialize_with_frequencies(freqs)

# 5. Train
trainer = PICASOTrainer(model, dataset, config)
trainer.train(loader, save_path="quickstart_best.pt")
print("Done!")
