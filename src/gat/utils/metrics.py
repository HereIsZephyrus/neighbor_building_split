"""Evaluation metrics for node classification."""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from .logger import get_logger

logger = get_logger()


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
    mask: Optional[torch.Tensor] = None,
    average: str = 'macro'
) -> Tuple[float, float]:
    """
    Compute F1 scores (macro and micro).

    Args:
        pred: Predicted class logits or labels (N, C) or (N,)
        target: Ground truth labels (N,)
        mask: Optional mask for which nodes to evaluate (N,)
        average: Averaging method ('macro', 'micro', 'weighted')

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


def compute_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        pred: Predicted class logits or labels (N, C) or (N,)
        target: Ground truth labels (N,)
        mask: Optional mask for which nodes to evaluate (N,)

    Returns:
        Confusion matrix as numpy array
    """
    if pred.dim() > 1:
        pred = pred.argmax(dim=1)

    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    cm = confusion_matrix(target_np, pred_np)

    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[list] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: Optional list of class names
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


def print_classification_report(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    class_names: Optional[list] = None
) -> str:
    """
    Print classification report.

    Args:
        pred: Predicted class logits or labels (N, C) or (N,)
        target: Ground truth labels (N,)
        mask: Optional mask for which nodes to evaluate (N,)
        class_names: Optional list of class names

    Returns:
        Classification report as string
    """
    if pred.dim() > 1:
        pred = pred.argmax(dim=1)

    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    report = classification_report(
        target_np, pred_np,
        target_names=class_names,
        zero_division=0
    )

    return report


def embedding_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    perplexity: int = 30,
    figsize: Tuple[int, int] = (13, 10)
) -> None:
    """
    Visualize embeddings using t-SNE.

    Args:
        embeddings: Node embeddings (N, D)
        labels: Node labels (N,)
        save_path: Path to save figure
        perplexity: t-SNE perplexity parameter
        figsize: Figure size
    """
    from sklearn.manifold import TSNE

    logger.info(f"Computing t-SNE with perplexity={perplexity}...")

    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=figsize)

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=f'Class {label}',
            alpha=0.6,
            s=50
        )

    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.title('Node Embeddings Visualization (t-SNE)')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"t-SNE plot saved to {save_path}")
    else:
        plt.show()

    plt.close()

