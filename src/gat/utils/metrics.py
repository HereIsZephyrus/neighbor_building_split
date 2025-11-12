"""Evaluation metrics for node classification."""

import torch
from sklearn.metrics import f1_score
from typing import Tuple, Optional

from .logger import get_logger

logger = get_logger(__name__)


def node_classification_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    Compute node classification accuracy.

    Args:
        pred: Predicted class logits or labels (N, C) or (N,)
        target: Ground truth labels (N,)
        mask: Optional mask for which nodes to evaluate (N,)

    Returns:
        Accuracy as float
    """
    # Convert logits to labels if needed
    if pred.dim() > 1:
        pred = pred.argmax(dim=1)

    # Apply mask if provided
    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    # Compute accuracy
    correct = (pred == target).sum().item()
    total = target.size(0)

    accuracy = correct / total if total > 0 else 0.0

    return accuracy


def compute_f1_scores(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[float, float]:
    """
    Compute F1 scores (macro and micro).

    Args:
        pred: Predicted class logits or labels (N, C) or (N,)
        target: Ground truth labels (N,)
        mask: Optional mask for which nodes to evaluate (N,)

    Returns:
        Tuple of (macro_f1, micro_f1)
    """
    # Convert to numpy
    if pred.dim() > 1:
        pred = pred.argmax(dim=1)

    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    # Compute F1 scores
    macro_f1 = f1_score(target_np, pred_np, average='macro', zero_division=0)
    micro_f1 = f1_score(target_np, pred_np, average='micro', zero_division=0)

    return macro_f1, micro_f1


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> dict:
    """
    Compute all metrics (accuracy, F1 scores).

    Args:
        pred: Predicted class logits (N, C)
        target: Ground truth labels (N,)
        mask: Optional mask for which nodes to evaluate (N,)

    Returns:
        Dictionary with metrics
    """
    accuracy = node_classification_accuracy(pred, target, mask)
    macro_f1, micro_f1 = compute_f1_scores(pred, target, mask)

    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
    }

    return metrics

