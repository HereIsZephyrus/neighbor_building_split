"""Mini-batch sampling for large graphs using NeighborLoader."""

import torch
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.data import Data, Batch
from typing import List, Optional

from ..utils.logger import get_logger

logger = get_logger()


def create_neighbor_loader(
    data: Data,
    num_neighbors: List[int] = [15, 10],
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> NeighborLoader:
    """
    Create NeighborLoader for mini-batch sampling on a single large graph.
    
    Args:
        data: PyG Data object
        num_neighbors: Number of neighbors to sample at each layer [layer1, layer2, ...]
        batch_size: Number of seed nodes per batch
        shuffle: Whether to shuffle nodes
        num_workers: Number of worker processes
        **kwargs: Additional arguments for NeighborLoader
        
    Returns:
        NeighborLoader instance
    """
    logger.debug(f"Creating NeighborLoader with batch_size={batch_size}, num_neighbors={num_neighbors}")
    
    # Create input nodes (all nodes by default)
    input_nodes = torch.arange(data.num_nodes)
    
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        input_nodes=input_nodes,
        **kwargs
    )
    
    logger.info(f"Created NeighborLoader: {len(loader)} batches for {data.num_nodes} nodes")
    
    return loader


def create_dataloader_for_districts(
    data_list: List[Data],
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader for batching multiple district graphs.
    
    Use this when each district graph is small enough to fit in memory,
    and you want to process multiple districts in a batch.
    
    Args:
        data_list: List of PyG Data objects (one per district)
        batch_size: Number of graphs per batch
        shuffle: Whether to shuffle graphs
        num_workers: Number of worker processes
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader instance
    """
    logger.debug(f"Creating DataLoader for {len(data_list)} districts with batch_size={batch_size}")
    
    loader = DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )
    
    logger.info(f"Created DataLoader: {len(loader)} batches for {len(data_list)} graphs")
    
    return loader


def should_use_neighbor_sampling(data: Data, threshold: int = 2000) -> bool:
    """
    Determine if neighbor sampling should be used based on graph size.
    
    Args:
        data: PyG Data object
        threshold: Number of nodes above which to use sampling
        
    Returns:
        True if neighbor sampling is recommended
    """
    return data.num_nodes > threshold


def create_adaptive_loader(
    data: Data,
    batch_size: int = 1024,
    num_neighbors: List[int] = [15, 10],
    node_threshold: int = 2000,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
):
    """
    Create appropriate loader based on graph size.
    
    For large graphs (>threshold nodes), use NeighborLoader for mini-batch sampling.
    For small graphs, return None (process full graph at once).
    
    Args:
        data: PyG Data object
        batch_size: Number of seed nodes per batch (for NeighborLoader)
        num_neighbors: Neighbor sampling strategy
        node_threshold: Threshold for using neighbor sampling
        shuffle: Whether to shuffle
        num_workers: Number of workers
        **kwargs: Additional arguments
        
    Returns:
        NeighborLoader or None (if full-graph training is appropriate)
    """
    if should_use_neighbor_sampling(data, threshold=node_threshold):
        logger.info(f"Graph has {data.num_nodes} nodes (>{node_threshold}), using NeighborLoader")
        return create_neighbor_loader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
    else:
        logger.info(f"Graph has {data.num_nodes} nodes (<={node_threshold}), using full-graph training")
        return None


class MultiGraphNeighborLoader:
    """
    Wrapper for handling multiple graphs with different sizes.
    
    Creates appropriate loaders for each graph based on size:
    - Small graphs: full-graph training
    - Large graphs: NeighborLoader with mini-batch sampling
    """
    
    def __init__(
        self,
        data_list: List[Data],
        batch_size: int = 1024,
        num_neighbors: List[int] = [15, 10],
        node_threshold: int = 2000,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        """
        Initialize multi-graph loader.
        
        Args:
            data_list: List of PyG Data objects
            batch_size: Batch size for NeighborLoader
            num_neighbors: Neighbor sampling strategy
            node_threshold: Threshold for using sampling
            shuffle: Whether to shuffle
            num_workers: Number of workers
        """
        self.data_list = data_list
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.node_threshold = node_threshold
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Create loaders for each graph
        self.loaders = []
        self.is_full_graph = []
        
        for data in data_list:
            if should_use_neighbor_sampling(data, threshold=node_threshold):
                loader = create_neighbor_loader(
                    data,
                    num_neighbors=num_neighbors,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers
                )
                self.loaders.append(loader)
                self.is_full_graph.append(False)
            else:
                # Store the full graph as a single-item list
                self.loaders.append([data])
                self.is_full_graph.append(True)
        
        logger.info(
            f"Created MultiGraphNeighborLoader: "
            f"{sum(self.is_full_graph)} full-graph, "
            f"{len(self.is_full_graph) - sum(self.is_full_graph)} sampled"
        )
    
    def __iter__(self):
        """Iterate over all batches from all graphs."""
        for loader, is_full in zip(self.loaders, self.is_full_graph):
            for batch in loader:
                yield batch, is_full
    
    def __len__(self):
        """Return total number of batches."""
        return sum(len(loader) for loader in self.loaders)

