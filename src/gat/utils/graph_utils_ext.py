"""
Extended graph utilities for connected component processing.
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from typing import Tuple, List, Dict


def extract_subgraph(
    data: Data,
    node_mask: np.ndarray,
    building_ids: List[int] = None
) -> Tuple[Data, Dict]:
    """
    Extract a subgraph from a PyG Data object based on node mask.

    Args:
        data: Original graph data
        node_mask: Boolean mask indicating which nodes to keep
        building_ids: Optional list of building IDs (for mapping)

    Returns:
        sub_data: Subgraph Data object
        node_mapping: Dict mapping old node indices to new indices
    """
    # Convert to tensor if numpy array
    if isinstance(node_mask, np.ndarray):
        node_mask_tensor = torch.from_numpy(node_mask)
    else:
        node_mask_tensor = node_mask

    # Create mapping from old indices to new indices
    old_to_new = {}
    new_idx = 0
    for old_idx in range(len(node_mask)):
        if node_mask[old_idx]:
            old_to_new[old_idx] = new_idx
            new_idx += 1

    # Extract node features
    sub_x = data.x[node_mask_tensor]
    sub_y = data.y[node_mask_tensor] if data.y is not None else None

    # Extract edges: keep only edges where both endpoints are in the subgraph
    edge_mask = node_mask_tensor[data.edge_index[0]] & node_mask_tensor[data.edge_index[1]]
    sub_edge_index = data.edge_index[:, edge_mask]

    # Remap edge indices to new node indices
    remapped_edges = []
    for i in range(sub_edge_index.shape[1]):
        src = int(sub_edge_index[0, i])
        dst = int(sub_edge_index[1, i])
        if src in old_to_new and dst in old_to_new:
            remapped_edges.append([old_to_new[src], old_to_new[dst]])

    if remapped_edges:
        sub_edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()
    else:
        sub_edge_index = torch.empty((2, 0), dtype=torch.long)

    # Extract edge attributes if present
    sub_edge_attr = None
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        sub_edge_attr = data.edge_attr[edge_mask]

    # Create subgraph Data object
    sub_data = Data(
        x=sub_x,
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        y=sub_y,
        num_nodes=sub_x.size(0)
    )

    return sub_data, old_to_new


def extract_subgraph_from_adjacency(
    adjacency_matrix: pd.DataFrame,
    building_ids: List[int]
) -> pd.DataFrame:
    """
    Extract a sub-adjacency matrix for a subset of buildings.

    Args:
        adjacency_matrix: Original adjacency matrix (building_id × building_id)
        building_ids: List of building IDs to keep

    Returns:
        Sub-adjacency matrix (len(building_ids) × len(building_ids))
    """
    # Filter rows and columns
    sub_adjacency = adjacency_matrix.loc[building_ids, building_ids]
    return sub_adjacency


def merge_component_results(
    component_results: List[Dict],
    original_order: np.ndarray
) -> Dict:
    """
    Merge results from multiple connected components back into original order.

    Args:
        component_results: List of result dicts from each component
        original_order: Mapping from component results to original indices

    Returns:
        Merged results dict with arrays in original node order
    """
    total_nodes = sum(r['num_nodes'] for r in component_results)
    num_classes = component_results[0]['logits'].shape[1] if len(component_results) > 0 else 0

    # Initialize output arrays
    merged = {
        'embeddings': np.zeros((total_nodes, component_results[0]['embeddings'].shape[1])),
        'logits': np.zeros((total_nodes, num_classes)),
        'gat_labels': np.zeros(total_nodes, dtype=int),
        'cluster_assignments': np.zeros(total_nodes, dtype=int),
        'final_labels': np.zeros(total_nodes, dtype=int),
        'component_id': np.zeros(total_nodes, dtype=int)
    }

    # Fill in results from each component
    current_idx = 0
    cluster_offset = 0  # Offset cluster IDs to keep them unique globally

    for comp_id, result in enumerate(component_results):
        n = result['num_nodes']
        end_idx = current_idx + n

        merged['embeddings'][current_idx:end_idx] = result['embeddings']
        merged['logits'][current_idx:end_idx] = result['logits']
        merged['gat_labels'][current_idx:end_idx] = result['gat_labels']

        # Offset cluster IDs to avoid conflicts between components
        merged['cluster_assignments'][current_idx:end_idx] = result['cluster_assignments'] + cluster_offset
        merged['final_labels'][current_idx:end_idx] = result['final_labels']
        merged['component_id'][current_idx:end_idx] = comp_id

        cluster_offset += (result['cluster_assignments'].max() + 1) if len(result['cluster_assignments']) > 0 else 0
        current_idx = end_idx

    return merged


def get_component_statistics(
    component_labels: np.ndarray,
    building_ids: List[int] = None,
    voronoi_areas: Dict[int, float] = None
) -> List[Dict]:
    """
    Compute statistics for each connected component.

    Args:
        component_labels: Array of component IDs for each node
        building_ids: Optional list of building IDs
        voronoi_areas: Optional dict of building ID -> voronoi area

    Returns:
        List of statistics dicts for each component
    """
    unique_components = np.unique(component_labels)
    stats = []

    for comp_id in unique_components:
        mask = (component_labels == comp_id)
        comp_size = mask.sum()

        stat_dict = {
            'component_id': int(comp_id),
            'num_buildings': int(comp_size),
        }

        # Calculate total area if voronoi areas available
        if building_ids is not None and voronoi_areas is not None:
            comp_building_ids = [building_ids[i] for i in range(len(mask)) if mask[i]]
            total_area = sum(voronoi_areas.get(bid, 0.0) for bid in comp_building_ids)
            stat_dict['total_area_m2'] = total_area
            stat_dict['total_area_km2'] = total_area / 1_000_000

        stats.append(stat_dict)

    return stats

