"""PyTorch Geometric Dataset for building graphs."""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset, Data
from pathlib import Path
from typing import Optional, List, Dict
from sklearn.preprocessing import StandardScaler

from .data_utils import load_district_graph
from ..utils.logger import get_logger
from .building import BuildingDataset
from .district import DistrictDataset

logger = get_logger()


class BuildingGraphDataset(Dataset):
    """
    Dataset for loading building graphs from districts.

    Each sample is a graph representing one district, where:
    - Nodes are buildings with standardized features
    - Edges are adjacency/similarity relationships
    - Labels are building categories/clusters
    """

    def __init__(
        self,
        adjacency_dir: str,
        district_dataset: DistrictDataset,
        building_dataset: BuildingDataset,
        dataset_dir: str,
    ):
        """
        Initialize BuildingGraphDataset and load all data.

        Args:
            adjacency_dir: Folder containing adjacency matrices (district_{district_id}_adjacency.pkl)
            district_ids: List of district IDs to load
            building_dataset: BuildingDataset containing building information
            dataset_dir: Directory to save/load processed dataset
        """
        # Initialize parent class
        super().__init__()

        self.adjacency_dir = Path(adjacency_dir)
        self.district_dataset = district_dataset
        self.building_dataset = building_dataset
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.scaler: Optional[StandardScaler] = None
        self.graphs: List[Data] = []
        self._num_features: int = 0  # Use private attribute to avoid property conflict

        # Load and construct dataset
        self._construct_dataset()

    def _construct_dataset(self):
        """Construct the dataset by loading adjacency matrices and building features."""
        logger.info(f"Constructing dataset for {len(self.district_dataset)} districts...")

        # First pass: collect all features (including degree) for standardization
        all_features = []
        district_data = []  # Store (district_id, adjacency_df, building_df, feature_cols, degrees) tuples

        for district_id in self.district_dataset.get_district_ids():
            try:
                # Load adjacency matrix
                adjacency_file = self.adjacency_dir / f"district_{district_id}_adjacency.pkl"
                if not adjacency_file.exists():
                    logger.warning(f"Adjacency file not found: {adjacency_file}")
                    continue

                adjacency_df = pd.read_pickle(adjacency_file)

                # Get buildings for this district
                buildings_df = self.building_dataset.get_buildings(self.district_dataset, district_id)

                if len(buildings_df) == 0:
                    logger.warning(f"No buildings found for district {district_id}")
                    continue

                # Extract features (all columns except 'id' and 'label')
                feature_cols = [col for col in buildings_df.columns 
                               if col not in ['fid', 'OBJECTID', 'id', 'label', 'geometry']]

                if not feature_cols:
                    logger.error(f"No feature columns found for district {district_id}")
                    continue

                features = buildings_df[feature_cols].values
                
                # Calculate degrees for this district
                # Build ID to index mapping
                if 'id' in buildings_df.columns:
                    building_ids = buildings_df['id'].values
                else:
                    building_ids = buildings_df.index.values
                id_to_idx = {bid: idx for idx, bid in enumerate(building_ids)}
                
                # Count edges for degree calculation
                num_nodes = len(buildings_df)
                degrees = np.zeros((num_nodes, 1))
                row_ids = adjacency_df.index.tolist()
                col_ids = adjacency_df.columns.tolist()
                
                for i, source_id in enumerate(row_ids):
                    if source_id not in id_to_idx:
                        continue
                    for j, target_id in enumerate(col_ids):
                        if target_id not in id_to_idx:
                            continue
                        weight = adjacency_df.iloc[i, j]
                        if weight > 1e-6:
                            source_idx = id_to_idx[source_id]
                            target_idx = id_to_idx[target_id]
                            if source_idx != target_idx:
                                degrees[source_idx] += 1
                
                # Concatenate original features with degree
                features_with_degree = np.hstack([features, degrees])
                all_features.append(features_with_degree)
                district_data.append((district_id, adjacency_df, buildings_df, feature_cols))

            except Exception as e:
                logger.error(f"Failed to load data for district {district_id}: {e}")
                continue

        if not all_features:
            raise ValueError("No valid district data found!")

        # Fit scaler on all features (including degree)
        all_features_concat = np.concatenate(all_features, axis=0)
        self.scaler = StandardScaler()
        self.scaler.fit(all_features_concat)
        self._num_features = all_features_concat.shape[1]
        logger.info(f"Fitted StandardScaler on {all_features_concat.shape[0]} buildings, {self._num_features} features (including degree)")

        # Second pass: construct graphs with normalized features
        for district_id, adjacency_df, buildings_df, feature_cols in district_data:
            try:
                graph = self._construct_graph(district_id, adjacency_df, buildings_df, feature_cols)
                self.graphs.append(graph)
                logger.info(f"Constructed graph for district {district_id}: "
                           f"{graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
            except Exception as e:
                logger.error(f"Failed to construct graph for district {district_id}: {e}")
                continue

        logger.info(f"Successfully constructed {len(self.graphs)} graphs")

    def _construct_graph(
        self, 
        district_id: int, 
        adjacency_df: pd.DataFrame,
        buildings_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Data:
        """
        Construct a PyG Data object for one district.

        Args:
            district_id: District ID
            adjacency_df: DataFrame containing adjacency information
            buildings_df: DataFrame containing building information
            feature_cols: List of feature column names

        Returns:
            PyG Data object
        """
        # Extract features (will add degree feature later)
        features = buildings_df[feature_cols].values
        num_nodes = len(buildings_df)

        # Extract labels (convert from 1-based to 0-based indexing)
        if 'label' in buildings_df.columns:
            labels = buildings_df['label'].values - 1  # Convert 1-based to 0-based
            y = torch.tensor(labels, dtype=torch.long)
        else:
            # If no labels, use zeros
            y = torch.zeros(len(buildings_df), dtype=torch.long)
            logger.warning(f"No 'label' column found for district {district_id}, using zeros")

        # Extract building IDs
        if 'id' in buildings_df.columns:
            building_ids = buildings_df['id'].values
        else:
            building_ids = buildings_df.index.values
            logger.warning(f"No 'id' column found for district {district_id}, using index")

        # Create ID to index mapping
        id_to_idx = {bid: idx for idx, bid in enumerate(building_ids)}

        # Build edge index from adjacency matrix
        # adjacency_df is a square matrix where rows and columns are building IDs
        edge_list = []
        edge_weights = []

        # Get row and column indices (building IDs)
        row_ids = adjacency_df.index.tolist()
        col_ids = adjacency_df.columns.tolist()

        # Iterate through the adjacency matrix
        for i, source_id in enumerate(row_ids):
            # Skip if source building not in our building dataset
            if source_id not in id_to_idx:
                continue

            for j, target_id in enumerate(col_ids):
                # Skip if target building not in our building dataset
                if target_id not in id_to_idx:
                    continue

                # Get adjacency value (distance/weight)
                weight = adjacency_df.iloc[i, j]

                # Skip if no edge (weight is 0 or very small)
                if weight > 1e-6:  # Threshold to avoid self-loops and zero weights
                    source_idx = id_to_idx[source_id]
                    target_idx = id_to_idx[target_id]

                    # Skip self-loops
                    if source_idx != target_idx:
                        edge_list.append([source_idx, target_idx])
                        edge_weights.append(weight)

        logger.info(f"District {district_id}: Found {len(edge_list)} edges from adjacency matrix")

        # Convert to tensor
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            # Empty graph
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Calculate node degrees (undirected graph)
        # For each node, count the number of unique neighbors
        degree = torch.zeros(num_nodes, dtype=torch.float)
        if edge_list:
            # Count degree for each node
            for src_idx, tgt_idx in edge_list:
                degree[src_idx] += 1
                # If this is a directed edge, also count reverse direction
                # For undirected graphs, adjacency matrix should be symmetric
        
        # Add degree as a new feature
        degree_feature = degree.numpy().reshape(-1, 1)
        features_with_degree = np.hstack([features, degree_feature])
        
        # Normalize features including degree
        features_normalized = self.scaler.transform(features_with_degree)
        x = torch.tensor(features_normalized, dtype=torch.float)
        
        logger.debug(f"District {district_id}: Added degree feature. Features shape: {x.shape}")

        # Create Data object
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            district_id=district_id,
            building_ids=building_ids,
        )

        # Add edge weights if available
        if edge_weights:
            data.edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)

        return data

    def len(self) -> int:
        """Return number of graphs in dataset."""
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        """
        Get graph data for district at index.

        Args:
            idx: Index of district in the dataset

        Returns:
            Data object for the district
        """
        if idx < 0 or idx >= len(self.graphs):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.graphs)}")

        return self.graphs[idx]

    @property
    def num_features(self) -> int:
        """Get number of node features (override parent property)."""
        return self._num_features

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        if not self.graphs:
            logger.warning("No graphs in dataset")
            return {
                'num_graphs': 0,
                'total_nodes': 0,
                'total_edges': 0,
                'avg_nodes_per_graph': 0,
                'avg_edges_per_graph': 0,
                'num_classes': 0,
                'num_features': self.num_features,
            }

        total_nodes = 0
        total_edges = 0
        all_labels = []

        # Directly iterate over the loaded graphs
        for graph in self.graphs:
            total_nodes += graph.num_nodes
            total_edges += graph.edge_index.shape[1]
            all_labels.append(graph.y.numpy())

        import numpy as np
        all_labels = np.concatenate(all_labels)

        num_graphs = len(self.graphs)
        stats = {
            'num_graphs': num_graphs,
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'avg_nodes_per_graph': total_nodes / num_graphs,
            'avg_edges_per_graph': total_edges / num_graphs,
            'num_classes': len(np.unique(all_labels)),
            'num_features': self.num_features,
        }

        logger.info(f"Dataset statistics: {stats}")

        return stats
