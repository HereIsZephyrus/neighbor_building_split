"""Graph construction and conversion utilities."""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional

from .logger import get_logger

logger = get_logger(__name__)


def similarity_matrix_to_edge_index(
    sim_matrix: pd.DataFrame,
    threshold: Optional[float] = None,
    add_self_loops: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert similarity/adjacency matrix to PyG edge_index format.

    Args:
        sim_matrix: Pandas DataFrame with similarity values (N×N)
                   Non-zero values indicate edges between buildings
        threshold: Optional threshold to filter weak edges (default: None, keep non-zero)
        add_self_loops: Whether to add self-loops (default: False)

    Returns:
        edge_index: LongTensor of shape (2, E) with source and target indices
        edge_attr: FloatTensor of shape (E,) with edge weights (similarity values)
    """
    logger.debug(f"Converting similarity matrix of shape {sim_matrix.shape} to edge_index")

    # Convert to numpy array
    matrix = sim_matrix.values

    # Apply threshold if specified
    if threshold is not None:
        matrix = matrix * (matrix >= threshold)
        logger.debug(f"Applied threshold {threshold}, keeping edges with similarity >= {threshold}")

    # Find non-zero entries (edges)
    rows, cols = np.nonzero(matrix)
    edge_weights = matrix[rows, cols]

    # Map to consecutive indices (0 to N-1)
    edge_list = []
    edge_values = []

    for i, j, weight in zip(rows, cols, edge_weights):
        if i != j or add_self_loops:  # Skip self-loops unless requested
            edge_list.append([i, j])
            edge_values.append(weight)

    if len(edge_list) == 0:
        logger.warning("No edges found in similarity matrix!")
        # Return empty edge index
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0,), dtype=torch.float)
        return edge_index, edge_attr

    # Convert to PyTorch tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_values, dtype=torch.float)

    logger.debug(f"Created edge_index with shape {edge_index.shape}, {edge_index.shape[1]} edges")
    logger.debug(f"Edge weights - min: {edge_attr.min():.4f}, max: {edge_attr.max():.4f}, mean: {edge_attr.mean():.4f}")

    return edge_index, edge_attr


def global_pool(x: torch.Tensor, method: str = 'mean_max') -> torch.Tensor:
    """
    Global pooling over all nodes in a graph.

    Args:
        x: Node features tensor of shape (N, D)
        method: Pooling method - 'mean', 'max', or 'mean_max' (concatenation)

    Returns:
        Graph-level feature vector of shape (1, D) or (1, 2*D) for mean_max
    """
    if method == 'mean':
        return x.mean(dim=0, keepdim=True)
    elif method == 'max':
        return x.max(dim=0, keepdim=True)[0]
    elif method == 'mean_max':
        mean_pool = x.mean(dim=0, keepdim=True)
        max_pool = x.max(dim=0, keepdim=True)[0]
        return torch.cat([mean_pool, max_pool], dim=1)
    else:
        raise ValueError(f"Unknown pooling method: {method}. Choose from 'mean', 'max', 'mean_max'.")

