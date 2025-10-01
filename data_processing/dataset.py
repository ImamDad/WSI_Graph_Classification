import os
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional

from .build_graphs import GraphBuilder

class WSIGraphDataset(Dataset):
    """Dataset for WSI graph classification using hierarchical graphs."""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform=None,
                 config: Dict = None):
        """
        Args:
            data_dir: Directory containing WSI patches
            split: 'train', 'val', or 'test'
            transform: Graph transformation functions
            config: Configuration dictionary
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.config = config or {}
        
        self.graph_builder = GraphBuilder(config)
        
        # Load data split
        self.samples = self._load_split_info()
        
        # Precompute graphs or load from cache
        self.graph_cache = {}
        self.use_cache = self.config.get('use_cache', True)
        
    def _load_split_info(self) -> List[Dict]:
        """Load dataset split information."""
        split_file = os.path.join(self.data_dir, f'{self.split}_split.txt')
        samples = []
        
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    sample = {
                        'patch_path': parts[0],
                        'mask_path': parts[1],
                        'label': int(parts[2]),
                        'tissue_type': parts[3] if len(parts) > 3 else 'unknown'
                    }
                    samples.append(sample)
                    
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Dict, int]:
        """Get hierarchical graphs and label for a sample."""
        sample = self.samples[idx]
        
        # Check cache first
        cache_key = f"{sample['patch_path']}_{sample['mask_path']}"
        if self.use_cache and cache_key in self.graph_cache:
            cell_graph, tissue_graph = self.graph_cache[cache_key]
        else:
            # Build graphs
            cell_graph, tissue_graph = self.graph_builder.build_hierarchical_graph(
                sample['patch_path'], sample['mask_path']
            )
            
            if self.use_cache:
                self.graph_cache[cache_key] = (cell_graph, tissue_graph)
        
        # Handle case where graph building fails
        if cell_graph is None or tissue_graph is None:
            # Return empty graphs (will be filtered later)
            cell_graph = self._create_empty_graph()
            tissue_graph = self._create_empty_graph()
        
        # Apply transforms
        if self.transform:
            cell_graph, tissue_graph = self.transform(cell_graph, tissue_graph)
            
        # Package graphs
        graphs = {
            'cell_graph': cell_graph,
            'tissue_graph': tissue_graph,
            'metadata': {
                'patch_path': sample['patch_path'],
                'tissue_type': sample['tissue_type'],
                'num_cells': cell_graph.num_nodes if cell_graph else 0,
                'num_tissues': tissue_graph.num_nodes if tissue_graph else 0
            }
        }
        
        return graphs, sample['label']
    
    def _create_empty_graph(self) -> torch.Tensor:
        """Create an empty graph placeholder."""
        return torch.zeros(0, 768)  # Empty feature matrix
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced dataset."""
        labels = [sample['label'] for sample in self.samples]
        class_counts = torch.bincount(torch.tensor(labels))
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        return class_weights
    
    def get_tissue_type_distribution(self) -> Dict[str, int]:
        """Get distribution of tissue types in dataset."""
        tissue_counts = {}
        for sample in self.samples:
            tissue_type = sample['tissue_type']
            tissue_counts[tissue_type] = tissue_counts.get(tissue_type, 0) + 1
        return tissue_counts

class HierarchicalGraphCollator:
    """Custom collator for hierarchical graph batches."""
    
    def __init__(self, follow_batch=None, exclude_keys=None):
        self.follow_batch = follow_batch or []
        self.exclude_keys = exclude_keys or []
    
    def __call__(self, batch):
        """Collate batch of hierarchical graphs."""
        from torch_geometric.data import Batch
        
        graphs_list, labels_list = zip(*batch)
        
        # Separate cell and tissue graphs
        cell_graphs = [item['cell_graph'] for item in graphs_list]
        tissue_graphs = [item['tissue_graph'] for item in graphs_list]
        metadata_list = [item['metadata'] for item in graphs_list]
        
        # Batch graphs
        cell_batch = Batch.from_data_list(cell_graphs, 
                                        follow_batch=self.follow_batch,
                                        exclude_keys=self.exclude_keys)
        tissue_batch = Batch.from_data_list(tissue_graphs,
                                          follow_batch=self.follow_batch,
                                          exclude_keys=self.exclude_keys)
        
        # Batch labels
        labels = torch.tensor(labels_list, dtype=torch.long)
        
        return {
            'cell_graph': cell_batch,
            'tissue_graph': tissue_batch,
            'labels': labels,
            'metadata': metadata_list
        }
