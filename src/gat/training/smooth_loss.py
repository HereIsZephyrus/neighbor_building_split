"""Spatial smoothness loss for GAT training.

Implements edge-based smoothness loss to encourage neighboring buildings
to be classified into the same category, forming continuous spatial regions.
"""

import torch
import torch.nn.functional as F


def edge_smoothness_loss(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor = None,
    temperature: float = 1.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute edge-based smoothness loss for spatial regularization.

    Encourages neighboring nodes (buildings) to have similar prediction distributions,
    with closer neighbors (smaller distances) having stronger constraints.

    Args:
        logits: Node classification logits, shape (N, C) where N is number of nodes
                and C is number of classes
        edge_index: Edge indices, shape (2, E) where E is number of edges.
                   edge_index[0] contains source nodes, edge_index[1] contains target nodes
        edge_weight: Optional edge weights (distances), shape (E,) or (E, 1).
                    If provided, will be converted to similarity weights using exp(-distance/scale).
                    If None, all edges are weighted equally.
        temperature: Softmax temperature parameter for prediction distributions.
                    Lower values make the constraint more strict.
        eps: Small epsilon for numerical stability (default: 1e-6)

    Returns:
        Scalar loss value representing the average smoothness penalty across all edges

    References:
        Based on the attention mechanism idea from GAT (Veličković et al., ICLR 2018):
        https://github.com/PetarV-/GAT
    """
    if edge_index.shape[1] == 0:
        # No edges, return zero loss
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # Check for NaN/Inf in input logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # Convert distance weights to similarity weights
    # Distance metric: smaller values = closer neighbors = higher similarity
    similarity = None
    if edge_weight is not None:
        # Flatten if edge_weight has shape (E, 1)
        if edge_weight.dim() > 1:
            edge_weight = edge_weight.squeeze(-1)

        # Check for NaN/Inf in edge weights
        if not (torch.isnan(edge_weight).any() or torch.isinf(edge_weight).any()):
            # Clamp edge weights to prevent extreme values
            edge_weight = torch.clamp(edge_weight, min=eps, max=1e6)

            # Convert to similarity: similarity = exp(-distance / scale)
            # Scale by mean distance for normalization
            scale = torch.clamp(edge_weight.mean(), min=eps)
            # Clamp the exponential input to prevent overflow
            exp_input = torch.clamp(-edge_weight / scale, min=-10.0, max=10.0)
            similarity = torch.exp(exp_input)

    if similarity is None:
        # Uniform weights if no edge attributes provided or if invalid
        similarity = torch.ones(edge_index.shape[1], device=logits.device, dtype=logits.dtype)

    # Get source and target node indices
    src_idx = edge_index[0]
    dst_idx = edge_index[1]

    # Clamp logits to prevent extreme values before softmax
    logits_clamped = torch.clamp(logits, min=-10.0, max=10.0)

    # Compute softmax probability distributions for source and target nodes
    # Using temperature scaling: lower temperature → sharper distributions
    # Use log_softmax for numerical stability (avoids log(0) = -inf)
    temperature = max(temperature, eps)  # Ensure temperature is not too small
    src_log_probs = F.log_softmax(logits_clamped[src_idx] / temperature, dim=-1)
    dst_probs = F.softmax(logits_clamped[dst_idx] / temperature, dim=-1)

    # Add small epsilon to dst_probs to prevent log(0) in KL divergence
    dst_probs = dst_probs + eps

    # Compute KL divergence between source and target distributions
    # KL(src || dst) measures how different the two distributions are
    # F.kl_div expects log probabilities as first argument and probabilities as second
    kl_div = F.kl_div(
        src_log_probs,
        dst_probs,
        reduction='none'
    ).sum(dim=-1)

    # Clamp KL divergence to prevent extreme values
    kl_div = torch.clamp(kl_div, min=0.0, max=100.0)

    # Check for NaN/Inf in KL divergence
    if torch.isnan(kl_div).any() or torch.isinf(kl_div).any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # Weight KL divergence by similarity (closer neighbors have higher penalty)
    weighted_loss = (kl_div * similarity).mean()

    # Final check for NaN/Inf
    if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    return weighted_loss


def bidirectional_smoothness_loss(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor = None,
    temperature: float = 1.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute bidirectional smoothness loss (symmetric version).

    Computes both KL(src || dst) and KL(dst || src) and averages them.
    This makes the loss symmetric with respect to edge direction.

    Args:
        logits: Node classification logits, shape (N, C)
        edge_index: Edge indices, shape (2, E)
        edge_weight: Optional edge weights (distances), shape (E,) or (E, 1)
        temperature: Softmax temperature parameter
        eps: Small epsilon for numerical stability (default: 1e-6)

    Returns:
        Scalar loss value
    """
    # Forward direction: KL(src || dst)
    loss_forward = edge_smoothness_loss(logits, edge_index, edge_weight, temperature, eps)

    # Backward direction: KL(dst || src)
    # Swap source and target indices
    edge_index_reverse = edge_index[[1, 0]]
    loss_backward = edge_smoothness_loss(logits, edge_index_reverse, edge_weight, temperature, eps)

    # Average of both directions
    return (loss_forward + loss_backward) / 2.0

