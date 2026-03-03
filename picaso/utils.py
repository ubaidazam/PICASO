"""PICASO utility functions."""

import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_entity_frequencies(triples, num_entities):
    """Compute how often each entity appears in the given triples."""
    frequencies = np.zeros(num_entities, dtype=np.int32)
    for h, r, t in triples:
        frequencies[h] += 1
        frequencies[t] += 1
    return frequencies


def load_model_safe(model, path, device):
    """Load a model state dict, stripping ``_orig_mod.`` prefixes if present."""
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model


def gaussian_nll_calibration(model, h, r, t):
    """Gaussian NLL proper scoring rule for uncertainty calibration.

    Optimal solution: variance = squared prediction error per dimension.
    """
    h_mu, h_var = model.entities(h)
    t_mu, t_var = model.entities(t)
    pred_mu, pred_var = model.rel_transform.transform(h_mu, h_var, r)
    combined_var = pred_var + t_var
    sq_error = (pred_mu - t_mu) ** 2
    nll = 0.5 * (torch.log(combined_var + 1e-8) + sq_error / (combined_var + 1e-8))
    return nll.mean()
