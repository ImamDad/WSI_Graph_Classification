import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T

class GraphTransform(BaseTransform):
    """Transformations for hierarchical graphs."""
    
    def __init__(self, config):
        self.config = config
        self.transforms = self._build_transforms()
        
    def _build_transforms(self):
        """Build sequence of graph transformations."""
        transforms = []
        
        # Feature normalization
        if self.config.get('normalize_features', True):
            transforms.append(NormalizeFeatures())
            
        # Graph augmentation for training
        if self.config.get('augment', False):
            transforms.extend([
                RandomNodeDrop(p=0.1),
                RandomEdgeAdd(p=0.1),
                FeatureNoise(std=0.01)
            ])
            
        return transforms
    
    def __call__(self, cell_graph, tissue_graph):
        """Apply transformations to both graphs."""
        for transform in self.transforms:
            cell_graph = transform(cell_graph)
            tissue_graph = transform(tissue_graph)
            
        return cell_graph, tissue_graph

class NormalizeFeatures(BaseTransform):
    """Normalize node features to zero mean and unit variance."""
    
    def __call__(self, data):
        if data.x is not None and data.x.numel() > 0:
            # Skip normalization if all features are zero
            if not torch.all(data.x == 0):
                data.x = (data.x - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-8)
        return data

class RandomNodeDrop(BaseTransform):
    """Randomly drop nodes from graph with probability p."""
    
    def __init__(self, p=0.1):
        self.p = p
        
    def __call__(self, data):
        if torch.rand(1) < self.p and data.num_nodes > 10:
            # Keep at least 10 nodes
            num_to_drop = max(1, int(self.p * data.num_nodes))
            keep_mask = torch.ones(data.num_nodes, dtype=torch.bool)
            drop_indices = torch.randperm(data.num_nodes)[:num_to_drop]
            keep_mask[drop_indices] = False
            
            # Update graph
            data = self._subgraph(data, keep_mask)
            
        return data
    
    def _subgraph(self, data, keep_mask):
        """Create subgraph with kept nodes."""
        from torch_geometric.utils import subgraph
        
        # Filter nodes
        data.x = data.x[keep_mask]
        data.pos = data.pos[keep_mask] if data.pos is not None else None
        
        # Filter edges
        edge_index, edge_attr = subgraph(
            keep_mask, data.edge_index, data.edge_attr, 
            relabel_nodes=True, num_nodes=len(keep_mask)
        )
        
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        
        return data

class RandomEdgeAdd(BaseTransform):
    """Randomly add edges between unconnected nodes."""
    
    def __init__(self, p=0.1):
        self.p = p
        
    def __call__(self, data):
        if data.num_nodes > 2 and torch.rand(1) < self.p:
            num_new_edges = max(1, int(self.p * data.num_edges))
            
            # Get all possible edges
            all_possible_edges = []
            for i in range(data.num_nodes):
                for j in range(data.num_nodes):
                    if i != j:
                        all_possible_edges.append([i, j])
            
            # Remove existing edges
            existing_edges = set([
                tuple(edge.tolist()) for edge in data.edge_index.t().numpy()
            ])
            
            possible_new_edges = [
                edge for edge in all_possible_edges 
                if tuple(edge) not in existing_edges
            ]
            
            # Add random new edges
            if possible_new_edges:
                new_edge_indices = np.random.choice(
                    len(possible_new_edges), 
                    min(num_new_edges, len(possible_new_edges)), 
                    replace=False
                )
                
                new_edges = torch.tensor(
                    [possible_new_edges[i] for i in new_edge_indices],
                    dtype=torch.long
                ).t()
                
                # Compute distances for new edge attributes
                new_positions_i = data.pos[new_edges[0]]
                new_positions_j = data.pos[new_edges[1]]
                new_distances = torch.norm(
                    new_positions_i - new_positions_j, dim=1
                ).unsqueeze(1)
                
                # Add to existing edges
                data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
                if data.edge_attr is not None:
                    data.edge_attr = torch.cat([data.edge_attr, new_distances], dim=0)
                else:
                    data.edge_attr = new_distances
                    
        return data

class FeatureNoise(BaseTransform):
    """Add Gaussian noise to node features."""
    
    def __init__(self, std=0.01):
        self.std = std
        
    def __call__(self, data):
        if data.x is not None:
            noise = torch.randn_like(data.x) * self.std
            data.x = data.x + noise
            
        return data

class GraphNormalization(BaseTransform):
    """Normalize graph structure and features."""
    
    def __init__(self):
        self.add_self_loops = T.AddSelfLoops()
        self.normalize_adj = T.NormalizeFeatures()
        
    def __call__(self, data):
        # Add self-loops
        data = self.add_self_loops(data)
        
        # Normalize adjacency
        data = self.normalize_adj(data)
        
        return data
