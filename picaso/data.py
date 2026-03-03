"""PICASO knowledge graph data loading and dataset classes."""

import json
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class KnowledgeGraph:
    """General-purpose knowledge graph with factory methods for various formats."""

    def __init__(self):
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.relation_to_id = {}
        self.id_to_relation = {}
        self.type_to_id = {}
        self.id_to_type = {}
        self.entity_types = {}
        self.type_entities = defaultdict(set)
        self.train_triples = []
        self.valid_triples = []
        self.test_triples = []
        self.all_true_triples = set()
        self.hr_to_t = defaultdict(set)
        self.tr_to_h = defaultdict(set)
        self.relation_head_type = defaultdict(set)
        self.relation_tail_type = defaultdict(set)
        self.entity_frequencies = None
        self.entity_inv_freq = None
        self.entity_labels = {}
        self.relation_labels = {}

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str, train_ratio: float = 0.8, valid_ratio: float = 0.1) -> "KnowledgeGraph":
        """Load a KG from a JSON file of ``{source_id, property_id, target_id, property}`` dicts.

        This matches the format produced by Wikidata extraction pipelines.
        """
        kg = cls()

        with open(path, 'r') as f:
            data = json.load(f)

        print(f"Loading KG from {path}...")
        print(f"  Raw triples: {len(data):,}")

        all_entities = set()
        all_relations = set()

        for item in data:
            all_entities.add(item['source_id'])
            all_entities.add(item['target_id'])
            all_relations.add(item['property_id'])
            kg.relation_labels[item['property_id']] = item.get('property', item['property_id'])

        kg.entity_to_id = {e: i for i, e in enumerate(sorted(all_entities))}
        kg.id_to_entity = {i: e for e, i in kg.entity_to_id.items()}
        kg.relation_to_id = {r: i for i, r in enumerate(sorted(all_relations))}
        kg.id_to_relation = {i: r for r, i in kg.relation_to_id.items()}

        all_triples = list({
            (kg.entity_to_id[item['source_id']],
             kg.relation_to_id[item['property_id']],
             kg.entity_to_id[item['target_id']])
            for item in data
        })
        np.random.shuffle(all_triples)

        kg._split_and_index(all_triples, train_ratio, valid_ratio)
        return kg

    @classmethod
    def from_triples(cls, triples: List[Tuple[int, int, int]], num_entities: int, num_relations: int,
                     train_ratio: float = 0.8, valid_ratio: float = 0.1) -> "KnowledgeGraph":
        """Build a KG from a list of ``(head, relation, tail)`` integer triples."""
        kg = cls()
        kg.entity_to_id = {i: i for i in range(num_entities)}
        kg.id_to_entity = dict(kg.entity_to_id)
        kg.relation_to_id = {i: i for i in range(num_relations)}
        kg.id_to_relation = dict(kg.relation_to_id)

        all_triples = list(set(triples))
        np.random.shuffle(all_triples)

        kg._split_and_index(all_triples, train_ratio, valid_ratio)
        return kg

    @classmethod
    def from_tsv(cls, train_path: str, valid_path: str = None, test_path: str = None) -> "KnowledgeGraph":
        """Load a KG from TSV files (head\\trelation\\ttail per line)."""
        kg = cls()

        def _read_tsv(path):
            triples = []
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        triples.append(tuple(parts))
            return triples

        raw_train = _read_tsv(train_path)
        raw_valid = _read_tsv(valid_path) if valid_path else []
        raw_test = _read_tsv(test_path) if test_path else []

        all_raw = raw_train + raw_valid + raw_test
        all_entities = sorted({e for h, r, t in all_raw for e in (h, t)})
        all_relations = sorted({r for _, r, _ in all_raw})

        kg.entity_to_id = {e: i for i, e in enumerate(all_entities)}
        kg.id_to_entity = {i: e for e, i in kg.entity_to_id.items()}
        kg.relation_to_id = {r: i for i, r in enumerate(all_relations)}
        kg.id_to_relation = {i: r for r, i in kg.relation_to_id.items()}

        def _to_ids(raw):
            return [(kg.entity_to_id[h], kg.relation_to_id[r], kg.entity_to_id[t]) for h, r, t in raw]

        kg.train_triples = _to_ids(raw_train)
        kg.valid_triples = _to_ids(raw_valid) if raw_valid else []
        kg.test_triples = _to_ids(raw_test) if raw_test else []

        if not kg.valid_triples and not kg.test_triples:
            all_triples = list(set(kg.train_triples))
            np.random.shuffle(all_triples)
            kg._split_and_index(all_triples, 0.8, 0.1)
        else:
            for h, r, t in kg.train_triples + kg.valid_triples + kg.test_triples:
                kg.all_true_triples.add((h, r, t))
                kg.hr_to_t[(h, r)].add(t)
                kg.tr_to_h[(t, r)].add(h)
                kg.relation_head_type[r].add(h)
                kg.relation_tail_type[r].add(t)
            kg._create_pseudo_types()
            kg._compute_entity_frequencies()

        kg._print_stats()
        return kg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_and_index(self, all_triples, train_ratio, valid_ratio):
        n = len(all_triples)
        train_end = int(train_ratio * n)
        valid_end = int((train_ratio + valid_ratio) * n)

        self.train_triples = all_triples[:train_end]
        self.valid_triples = all_triples[train_end:valid_end]
        self.test_triples = all_triples[valid_end:]

        for h, r, t in all_triples:
            self.all_true_triples.add((h, r, t))
            self.hr_to_t[(h, r)].add(t)
            self.tr_to_h[(t, r)].add(h)
            self.relation_head_type[r].add(h)
            self.relation_tail_type[r].add(t)

        self._create_pseudo_types()
        self._compute_entity_frequencies()
        self._print_stats()

    def _create_pseudo_types(self):
        type_counter = 0
        for r in range(self.num_relations):
            self.type_to_id[f"head_type_{r}"] = type_counter
            self.id_to_type[type_counter] = f"head_type_{r}"
            for h in self.relation_head_type[r]:
                if h not in self.entity_types:
                    self.entity_types[h] = []
                self.entity_types[h].append(type_counter)
                self.type_entities[type_counter].add(h)
            type_counter += 1
            self.type_to_id[f"tail_type_{r}"] = type_counter
            self.id_to_type[type_counter] = f"tail_type_{r}"
            for t in self.relation_tail_type[r]:
                if t not in self.entity_types:
                    self.entity_types[t] = []
                self.entity_types[t].append(type_counter)
                self.type_entities[type_counter].add(t)
            type_counter += 1

    def _compute_entity_frequencies(self):
        self.entity_frequencies = np.zeros(self.num_entities, dtype=np.float32)
        for h, r, t in self.train_triples:
            self.entity_frequencies[h] += 1
            self.entity_frequencies[t] += 1
        self.entity_inv_freq = 1.0 / (self.entity_frequencies + 1.0)
        self.entity_inv_freq /= self.entity_inv_freq.sum()

    def _print_stats(self):
        print(f"  Entities:  {self.num_entities:,}")
        print(f"  Relations: {self.num_relations}")
        print(f"  Types:     {self.num_types}")
        print(f"  Train:     {len(self.train_triples):,}")
        print(f"  Valid:     {len(self.valid_triples):,}")
        print(f"  Test:      {len(self.test_triples):,}")

    def get_relation_name(self, rel_id: int) -> str:
        """Get human-readable relation name."""
        prop_id = self.id_to_relation[rel_id]
        return self.relation_labels.get(prop_id, prop_id)

    @property
    def num_entities(self):
        return len(self.entity_to_id)

    @property
    def num_relations(self):
        return len(self.relation_to_id)

    @property
    def num_types(self):
        return len(self.type_to_id)


class TripleDataset(Dataset):
    """Dataset with uncertainty-aware negative sampling."""

    def __init__(self, triples, num_entities, num_relations, neg_size=128,
                 hr_to_t=None, tr_to_h=None, rel_head_type=None, rel_tail_type=None,
                 entity_types=None, use_type_constraint=True, entity_inv_freq=None):
        self.triples = triples
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.neg_size = neg_size
        self.hr_to_t = hr_to_t or defaultdict(set)
        self.tr_to_h = tr_to_h or defaultdict(set)
        self.rel_head_type = rel_head_type or defaultdict(set)
        self.rel_tail_type = rel_tail_type or defaultdict(set)
        self.entity_types = entity_types or {}
        self.use_type_constraint = use_type_constraint
        self.rel_head_list = {r: list(heads) for r, heads in self.rel_head_type.items()}
        self.rel_tail_list = {r: list(tails) for r, tails in self.rel_tail_type.items()}
        self.entity_inv_freq = entity_inv_freq
        self.use_uncertainty_sampling = entity_inv_freq is not None

    def __len__(self):
        return len(self.triples)

    def _sample_negatives(self, h, r, t, corrupt_tail=True):
        negatives = []
        positive_set = self.hr_to_t[(h, r)] if corrupt_tail else self.tr_to_h[(t, r)]

        if self.use_type_constraint:
            type_list = self.rel_tail_list.get(r, []) if corrupt_tail else self.rel_head_list.get(r, [])
            if type_list:
                for _ in range(self.neg_size // 2 * 3):
                    if len(negatives) >= self.neg_size // 2:
                        break
                    neg = type_list[np.random.randint(len(type_list))]
                    if neg not in positive_set:
                        negatives.append(neg)

        if self.use_uncertainty_sampling and len(negatives) < self.neg_size:
            pos_entity = t if corrupt_tail else h
            pos_inv_freq = self.entity_inv_freq[pos_entity]
            freq_diff = np.abs(self.entity_inv_freq - pos_inv_freq)
            sampling_weights = np.exp(-freq_diff * 1000)
            sampling_weights /= sampling_weights.sum()
            n_remaining = self.neg_size - len(negatives)
            candidates = np.random.choice(
                self.num_entities, size=n_remaining * 2, replace=True, p=sampling_weights
            )
            for neg in candidates:
                if len(negatives) >= self.neg_size:
                    break
                if neg not in positive_set:
                    negatives.append(neg)

        while len(negatives) < self.neg_size:
            neg = np.random.randint(0, self.num_entities)
            if neg not in positive_set:
                negatives.append(neg)

        return negatives[:self.neg_size]

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        h_types = self.entity_types.get(h, [])
        t_types = self.entity_types.get(t, [])
        max_types = 5
        h_types_padded = (h_types[:max_types] + [-1] * max_types)[:max_types]
        t_types_padded = (t_types[:max_types] + [-1] * max_types)[:max_types]
        return {
            'head': torch.tensor(h, dtype=torch.long),
            'relation': torch.tensor(r, dtype=torch.long),
            'tail': torch.tensor(t, dtype=torch.long),
            'neg_tails': torch.tensor(self._sample_negatives(h, r, t, True), dtype=torch.long),
            'neg_heads': torch.tensor(self._sample_negatives(h, r, t, False), dtype=torch.long),
            'head_types': torch.tensor(h_types_padded, dtype=torch.long),
            'tail_types': torch.tensor(t_types_padded, dtype=torch.long),
        }
