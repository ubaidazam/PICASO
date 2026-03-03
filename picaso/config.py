"""PICASO configuration."""

from dataclasses import dataclass

import torch


@dataclass
class PICASOConfig:
    dataset_name: str = "custom"
    embedding_dim: int = 200
    entity_init_std: float = 1e-3
    entity_init_var: float = 0.5
    min_var: float = 0.01
    max_var: float = 10.0
    relation_init_var: float = 0.3
    use_rotation: bool = True
    use_scaling: bool = True
    use_reciprocal: bool = True
    use_semantic_types: bool = True
    type_embedding_dim: int = None
    type_kernel_bandwidth: float = 1.0
    type_loss_weight: float = 0.1
    scoring_method: str = "kl_divergence"
    use_ensemble: bool = True
    geometric_weight: float = 0.30
    translational_weight: float = 0.20
    kl_weight: float = 0.25
    bilinear_weight: float = 0.15
    complex_weight: float = 0.10
    epochs: int = 200
    batch_size: int = 256
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.0005
    weight_decay: float = 0.0
    negative_samples: int = 128
    adversarial_temperature: float = 2.0
    use_type_constraint: bool = True
    margin: float = 9.0
    use_label_smoothing: bool = True
    label_smoothing: float = 0.1
    adversarial_loss_weight: float = 0.5
    ce_loss_weight: float = 0.5
    calibration_loss_weight: float = 2.0
    uncertainty_reg_weight: float = 0.01
    spread_loss_weight: float = 0.5
    aleatoric_weight: float = 5.0
    use_warmup: bool = True
    warmup_epochs: int = 30
    cosine_T0: int = 20
    cosine_T_mult: int = 2
    patience: int = 100
    eval_batch_size: int = 32
    eval_every: int = 25
    seed: int = 42
    num_workers: int = 0
    device: str = None

    def __post_init__(self):
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.type_embedding_dim is None:
            self.type_embedding_dim = self.embedding_dim
