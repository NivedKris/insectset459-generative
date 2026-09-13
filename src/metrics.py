"""
Core metrics for Species-Identity Preservation and Evaluation:
1. Species-Identity Preservation Rate (SIPR, Eq. 3)
2. Genus Confusion Rate (GCR)
3. Maximum Mean Discrepancy (MMD^2) with median heuristic Gaussian RBF kernel
4. Tier-stratified Macro-F1
"""
import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_sipr_and_gcr(oracle_model: torch.nn.Module,
                         spectrograms: torch.Tensor,
                         target_class: int,
                         target_genus: int,
                         class_to_genus: List[int],
                         device: torch.device) -> Tuple[float, float, int]:
    """
    Computes SIPR and Genus Confusion Rate for a batch of synthetic spectrograms
    generated for target_class.
    """
    oracle_model.eval()
    with torch.no_grad():
        specs = spectrograms.to(device)
        logits = oracle_model(specs)
        preds = logits.argmax(dim=-1).cpu().numpy()

    correct_species = int(np.sum(preds == target_class))
    wrong_species_same_genus = 0

    for p in preds:
        if p != target_class and class_to_genus[p] == target_genus:
            wrong_species_same_genus += 1

    total = len(preds)
    sipr = correct_species / total if total > 0 else 0.0
    gcr = wrong_species_same_genus / total if total > 0 else 0.0
    return sipr, gcr, total


def compute_mmd2_rbf(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes Maximum Mean Discrepancy (MMD^2) with Gaussian RBF kernel using
    the median heuristic bandwidth over embedding spaces.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Subsample if large to maintain speed
    if len(x) > 1000:
        idx_x = np.random.choice(len(x), 1000, replace=False)
        x = x[idx_x]
    if len(y) > 1000:
        idx_y = np.random.choice(len(y), 1000, replace=False)
        y = y[idx_y]

    # Compute pairwise Euclidean distances to determine median heuristic bandwidth sigma
    xy = np.vstack([x, y])
    dists = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    median_dist = np.median(dists[dists > 0])
    gamma = 1.0 / (2.0 * max(median_dist, 1e-5))

    # Kernel matrices
    K_xx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    K_yy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    K_xy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))

    m = len(x)
    n = len(y)
    # Unbiased MMD^2 estimator
    np.fill_diagonal(K_xx, 0.0)
    np.fill_diagonal(K_yy, 0.0)
    mmd2 = (np.sum(K_xx) / (m * (m - 1)) +
            np.sum(K_yy) / (n * (n - 1)) -
            2.0 * np.sum(K_xy) / (m * n))
    return float(max(0.0, mmd2))


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 459) -> Tuple[float, np.ndarray]:
    """Computes Macro-F1 across all classes with zero-division handling."""
    per_class_f1 = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        denom = 2 * tp + fp + fn
        if denom > 0:
            per_class_f1[c] = (2.0 * tp) / denom
        else:
            per_class_f1[c] = 0.0

    macro_f1 = float(np.mean(per_class_f1))
    return macro_f1, per_class_f1
