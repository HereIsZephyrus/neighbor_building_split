"""PyTorch Geometric Dataset for building graphs."""

import torch
from torch_geometric.data import Dataset, Data
from pathlib import Path
from typing import Optional, List, Tuple
import geopandas as gpd
from sklearn.preprocessing import StandardScaler

from .data_utils import load_district_graph
from ..utils.logger import get_logger

logger = get_logger()


class BuildingGraphDataset(Dataset):
    """
    Dataset for loading building graphs from districts.
    
    Each sample is a graph representing one district, where:
    - Nodes are buildings with 12-dimensional features
    - Edges are adjacency/similarity relationships
    - Labels are building categories/clusters
    """
    
    def __init__(
        self,
        root: str,
        district_ids: List[int],
        building_shapefile_path: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        normalize_features: bool = True
    ):
        """
        Initialize BuildingGraphDataset.
        
        Args:
            root: Root directory containing voronoi output
            district_ids: List of district IDs to load
            building_shapefile_path: Path to building shapefile
            transform: Optional transform to apply to each graph
            pre_transform: Optional transform to apply before saving
            pre_filter: Optional filter to apply before saving
            normalize_features: Whether to normalize features with StandardScaler
        """
        self.district_ids = district_ids
        self.building_shapefile_path = Path(building_shapefile_path)
        self.normalize_features = normalize_features
        self.scaler: Optional[StandardScaler] = None
        
        # Convert root to Path
        self.data_dir = Path(root)
        
        super().__init__(str(root), transform, pre_transform, pre_filter)
        
        # Load all districts and fit scaler
        self._load_and_prepare_data()
    
    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names."""
        return [f"district_{did}_adjacency.pkl" for did in self.district_ids]
    
    @property
    def processed_file_names(self) -> List[str]:
        """Return list of processed file names."""
        return [f"district_{did}.pt" for did in self.district_ids]
    
    def download(self):
        """Download data (not needed, data should already exist)."""
        pass
    
    def process(self):
        """Process raw data and save."""
        logger.info("Processing districts...")
        
        # First pass: collect all features to fit scaler
        all_features = []
        temp_data_list = []
        
        for idx, district_id in enumerate(self.district_ids):
            try:
                data, _ = load_district_graph(
                    district_id,
                    self.data_dir,
                    self.building_shapefile_path,
                    normalize_features=False
                )
                temp_data_list.append(data)
                all_features.append(data.x.numpy())
            except Exception as e:
                logger.error(f"Failed to process district {district_id}: {e}")
                continue
        
        # Fit scaler on all features
        if self.normalize_features and len(all_features) > 0:
            import numpy as np
            all_features_concat = np.concatenate(all_features, axis=0)
            self.scaler = StandardScaler()
            self.scaler.fit(all_features_concat)
            logger.info("Fitted StandardScaler on all district features")
            
            # Apply normalization
            for data in temp_data_list:
                data.x = torch.tensor(
                    self.scaler.transform(data.x.numpy()),
                    dtype=torch.float
                )
        
        # Apply pre_filter and pre_transform if needed
        for idx, data in enumerate(temp_data_list):
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            # Save processed data
            district_id = data.district_id
            torch.save(data, self.processed_dir / f"district_{district_id}.pt")
        
        logger.info(f"Processed {len(temp_data_list)} districts")
    
    def _load_and_prepare_data(self):
        """Load and prepare all data."""
        # Check if processing is needed
        if not self._check_processed():
            logger.info("Processed data not found, processing...")
            self.process()
        else:
            # Load scaler from first district to check if we need to fit it
            if self.normalize_features:
                try:
                    # We'll fit the scaler when first loading data
                    # For now, we'll recompute it in len() or get()
                    pass
                except Exception as e:
                    logger.warning(f"Could not load scaler info: {e}")
    
    def _check_processed(self) -> bool:
        """Check if all processed files exist."""
        for fname in self.processed_file_names:
            if not (Path(self.processed_dir) / fname).exists():
                return False
        return True
    
    def len(self) -> int:
        """Return number of graphs in dataset."""
        return len(self.district_ids)
    
    def get(self, idx: int) -> Data:
        """
        Get graph data for district at index.
        
        Args:
            idx: Index of district
            
        Returns:
            Data object for the district
        """
        district_id = self.district_ids[idx]
        
        # Load processed data
        data_path = Path(self.processed_dir) / f"district_{district_id}.pt"
        
        if data_path.exists():
            data = torch.load(data_path)
        else:
            # Fallback: load and process on the fly
            logger.warning(f"Processed file not found for district {district_id}, loading on the fly")
            data, _ = load_district_graph(
                district_id,
                self.data_dir,
                self.building_shapefile_path,
                normalize_features=self.normalize_features,
                scaler=self.scaler
            )
        
        return data
    
    def get_num_classes(self) -> int:
        """Get number of unique classes in dataset."""
        all_labels = []
        for idx in range(len(self)):
            data = self.get(idx)
            all_labels.append(data.y.numpy())
        
        import numpy as np
        all_labels = np.concatenate(all_labels)
        num_classes = len(np.unique(all_labels))
        
        return num_classes
    
    def get_num_features(self) -> int:
        """Get number of node features."""
        # All graphs have same number of features (12)
        return 12
    
    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        total_nodes = 0
        total_edges = 0
        all_labels = []
        
        for idx in range(len(self)):
            data = self.get(idx)
            total_nodes += data.num_nodes
            total_edges += data.edge_index.shape[1]
            all_labels.append(data.y.numpy())
        
        import numpy as np
        all_labels = np.concatenate(all_labels)
        
        stats = {
            'num_graphs': len(self),
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'avg_nodes_per_graph': total_nodes / len(self),
            'avg_edges_per_graph': total_edges / len(self),
            'num_classes': len(np.unique(all_labels)),
            'num_features': self.get_num_features(),
        }
        
        logger.info(f"Dataset statistics: {stats}")
        
        return stats

