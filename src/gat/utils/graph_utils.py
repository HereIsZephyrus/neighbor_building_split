"""Graph construction and conversion utilities."""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional
from scipy.sparse import coo_matrix

from .logger import get_logger

logger = get_logger()


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

    # Create mapping from original building IDs to consecutive indices
    original_ids = sim_matrix.index.values
    id_to_idx = {orig_id: idx for idx, orig_id in enumerate(original_ids)}

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


def add_self_loops_to_edge_index(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
    fill_value: float = 1.0
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Add self-loops to edge_index.

    Args:
        edge_index: Edge index tensor of shape (2, E)
        edge_attr: Optional edge attributes of shape (E,) or (E, D)
        num_nodes: Number of nodes. If None, inferred from edge_index
        fill_value: Value for self-loop edges

    Returns:
        edge_index: Edge index with self-loops (2, E + N)
        edge_attr: Edge attributes with self-loops (E + N,) or (E + N, D)
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    # Create self-loop edges
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    # Concatenate with existing edges
    edge_index = torch.cat([edge_index, loop_index], dim=1)

    # Add self-loop attributes if edge_attr is provided
    if edge_attr is not None:
        if edge_attr.dim() == 1:
            loop_attr = torch.full((num_nodes,), fill_value, dtype=edge_attr.dtype, device=edge_attr.device)
        else:
            loop_attr = torch.full((num_nodes, edge_attr.size(1)), fill_value, dtype=edge_attr.dtype, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

    return edge_index, edge_attr


def normalize_edge_weights(edge_attr: torch.Tensor, method: str = 'sum') -> torch.Tensor:
    """
    Normalize edge weights.

    Args:
        edge_attr: Edge weights tensor of shape (E,)
        method: Normalization method ('sum', 'max', 'minmax', 'standard')

    Returns:
        Normalized edge weights
    """
    if method == 'sum':
        # Normalize so weights sum to 1
        return edge_attr / edge_attr.sum()
    elif method == 'max':
        # Normalize by maximum weight
        return edge_attr / edge_attr.max()
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = edge_attr.min()
        max_val = edge_attr.max()
        if max_val > min_val:
            return (edge_attr - min_val) / (max_val - min_val)
        else:
            return torch.ones_like(edge_attr)
    elif method == 'standard':
        # Standardization (z-score)
        mean = edge_attr.mean()
        std = edge_attr.std()
        if std > 0:
            return (edge_attr - mean) / std
        else:
            return edge_attr - mean
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_graph_stats(edge_index: torch.Tensor, num_nodes: int) -> dict:
    """
    Compute graph statistics.

    Args:
        edge_index: Edge index tensor of shape (2, E)
        num_nodes: Number of nodes

    Returns:
        Dictionary with graph statistics
    """
    num_edges = edge_index.shape[1]

    # Compute node degrees
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(num_nodes):
        degrees[i] = (edge_index[0] == i).sum()

    # Compute statistics
    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': degrees.float().mean().item(),
        'max_degree': degrees.max().item(),
        'min_degree': degrees.min().item(),
        'median_degree': degrees.float().median().item(),
        'density': num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0,
    }

    logger.debug(f"Graph stats: {stats}")

    return stats


def get_connected_components(edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, int]:
    """
    Find connected components in the graph.

    Args:
        edge_index: Edge index tensor of shape (2, E)
        num_nodes: Number of nodes

    Returns:
        component_labels: Tensor of shape (N,) with component ID for each node
        num_components: Number of connected components
    """
    # Use DFS to find connected components
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    component_labels = torch.zeros(num_nodes, dtype=torch.long)
    current_component = 0

    # Build adjacency list
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[src].append(dst)
        adj_list[dst].append(src)

    # DFS
    def dfs(node, component):
        visited[node] = True
        component_labels[node] = component
        for neighbor in adj_list[node]:
            if not visited[neighbor]:
                dfs(neighbor, component)

    # Find all components
    for node in range(num_nodes):
        if not visited[node]:
            dfs(node, current_component)
            current_component += 1

    logger.debug(f"Found {current_component} connected components")

    return component_labels, current_component

