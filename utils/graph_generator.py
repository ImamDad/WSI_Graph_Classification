import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

class SyntheticGraphGenerator:
    """Generate synthetic hierarchical graphs for testing and development."""
    
    def __init__(self, config):
        self.config = config
        
    def generate_synthetic_cell_graph(self, num_cells=100, feature_dim=787):
        """Generate synthetic cell graph with realistic properties."""
        # Generate random cell positions
        positions = np.random.rand(num_cells, 2) * 100
        
        # Generate synthetic features (mimicking real nuclear features)
        features = self._generate_synthetic_features(num_cells, feature_dim)
        
        # Build k-NN graph
        edge_index = self._build_knn_edges(positions, k=8)
        
        # Create PyG data object
        data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=edge_index,
            pos=torch.tensor(positions, dtype=torch.float32),
            num_nodes=num_cells
        )
        
        return data
    
    def generate_synthetic_tissue_graph(self, cell_graph, num_tissues=20):
        """Generate synthetic tissue graph from cell graph."""
        if cell_graph.num_nodes == 0:
            return self._create_empty_graph()
            
        # Cluster cells into tissue regions
        clusters = self._synthetic_clustering(cell_graph.pos.numpy(), num_tissues)
        
        # Aggregate tissue nodes
        tissue_features = []
        tissue_positions = []
        
        for cluster_indices in clusters:
            if len(cluster_indices) == 0:
                continue
                
            # Mean pool cell features
            cluster_features = cell_graph.x[cluster_indices]
            tissue_feature = cluster_features.mean(dim=0).numpy()
            tissue_features.append(tissue_feature)
            
            # Compute tissue centroid
            cluster_positions = cell_graph.pos[cluster_indices]
            tissue_position = cluster_positions.mean(dim=0).numpy()
            tissue_positions.append(tissue_position)
            
        if len(tissue_features) == 0:
            return self._create_empty_graph()
            
        # Build tissue graph edges (Delaunay triangulation)
        edge_index = self._build_delaunay_edges(np.array(tissue_positions))
        
        # Create tissue graph
        data = Data(
            x=torch.tensor(tissue_features, dtype=torch.float32),
            edge_index=edge_index,
            pos=torch.tensor(tissue_positions, dtype=torch.float32),
            num_nodes=len(tissue_features)
        )
        
        return data, clusters
    
    def _generate_synthetic_features(self, num_cells, feature_dim):
        """Generate synthetic nuclear features with realistic distributions."""
        features = np.zeros((num_cells, feature_dim))
        
        # DINOv2 features (first 768 dimensions)
        features[:, :768] = np.random.normal(0, 1, (num_cells, 768))
        
        # Morphological features (next 7 dimensions)
        # Area, perimeter, circularity, solidity, eccentricity, extent, orientation
        features[:, 768:775] = np.random.uniform(0, 1, (num_cells, 7))
        
        # Nuclear morphometric features (last 12 dimensions)
        # Radial distances + additional features
        features[:, 775:] = np.random.uniform(0, 1, (num_cells, feature_dim - 775))
        
        return features
    
    def _build_knn_edges(self, positions, k=8):
        """Build k-NN edges from positions."""
        from sklearn.neighbors import kneighbors_graph
        
        adj_matrix = kneighbors_graph(positions, k, mode='connectivity', 
                                    include_self=False)
        edge_index = np.array(adj_matrix.nonzero())
        return torch.tensor(edge_index, dtype=torch.long)
    
    def _synthetic_clustering(self, positions, num_clusters):
        """Perform synthetic clustering of cells."""
        from sklearn.cluster import KMeans
        
        if len(positions) < num_clusters:
            num_clusters = len(positions)
            
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(positions)
        
        clusters = []
        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]
            clusters.append(cluster_indices)
            
        return clusters
    
    def _build_delaunay_edges(self, positions):
        """Build edges using Delaunay triangulation."""
        if len(positions) < 3:
            return self._build_complete_graph(len(positions))
            
        try:
            tri = Delaunay(positions)
            edges = set()
            
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        edge = tuple(sorted([simplex[i], simplex[j]]))
                        edges.add(edge)
                        
            if edges:
                edge_index = np.array(list(zip(*edges))).T
                edge_index = np.concatenate([edge_index, edge_index[:, [1,0]]], axis=0)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long)
                
        except:
            edge_index = self._build_complete_graph(len(positions))
            
        return edge_index
    
    def _build_complete_graph(self, n_nodes):
        """Build complete graph for small numbers of nodes."""
        if n_nodes <= 1:
            return torch.zeros(2, 0, dtype=torch.long)
            
        edges = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edges.append([i, j])
                    
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def _create_empty_graph(self):
        """Create empty graph placeholder."""
        return Data(
            x=torch.zeros(0, 787, dtype=torch.float32),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            pos=torch.zeros(0, 2, dtype=torch.float32)
        )

class GraphVisualizer:
    """Visualization utilities for hierarchical graphs."""
    
    @staticmethod
    def visualize_hierarchical_graph(cell_graph, tissue_graph, clusters, 
                                   cell_importance=None, tissue_importance=None,
                                   save_path=None):
        """Visualize hierarchical graph structure."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Cell graph visualization
        cell_pos = cell_graph.pos.numpy()
        
        if cell_importance is not None:
            sc1 = ax1.scatter(cell_pos[:, 0], cell_pos[:, 1], 
                             c=cell_importance, cmap='Reds', s=30, alpha=0.7)
            plt.colorbar(sc1, ax=ax1)
        else:
            ax1.scatter(cell_pos[:, 0], cell_pos[:, 1], s=30, alpha=0.6)
            
        # Plot cell graph edges
        edge_index = cell_graph.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            source = cell_pos[edge_index[0, i]]
            target = cell_pos[edge_index[1, i]]
            ax1.plot([source[0], target[0]], [source[1], target[1]], 
                    'k-', alpha=0.2, linewidth=0.5)
            
        ax1.set_title('Cell Graph')
        ax1.set_aspect('equal')
        
        # Tissue graph visualization with hierarchical relationships
        tissue_pos = tissue_graph.pos.numpy()
        
        if tissue_importance is not None:
            sc2 = ax2.scatter(tissue_pos[:, 0], tissue_pos[:, 1], 
                             c=tissue_importance, cmap='Blues', s=100, alpha=0.7)
            plt.colorbar(sc2, ax=ax2)
        else:
            ax2.scatter(tissue_pos[:, 0], tissue_pos[:, 1], s=100, alpha=0.7)
            
        # Plot tissue graph edges
        tissue_edges = tissue_graph.edge_index.numpy()
        for i in range(tissue_edges.shape[1]):
            source = tissue_pos[tissue_edges[0, i]]
            target = tissue_pos[tissue_edges[1, i]]
            ax2.plot([source[0], target[0]], [source[1], target[1]], 
                    'b-', alpha=0.5, linewidth=1)
            
        # Plot hierarchical relationships
        for i, tissue_node in enumerate(tissue_pos):
            cell_indices = clusters[i] if i < len(clusters) else []
            for cell_idx in cell_indices:
                if cell_idx < len(cell_pos):
                    cell_node = cell_pos[cell_idx]
                    ax2.plot([cell_node[0], tissue_node[0]], 
                            [cell_node[1], tissue_node[1]], 
                            'r-', alpha=0.3, linewidth=0.5)
                    
        ax2.set_title('Tissue Graph with Hierarchical Relationships')
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_feature_distributions(cell_graph, tissue_graph, feature_indices=None, 
                                 save_path=None):
        """Plot distributions of key features."""
        if feature_indices is None:
            feature_indices = [0, 1, 2, 768, 769]  # Sample important features
            
        num_features = len(feature_indices)
        fig, axes = plt.subplots(2, num_features, figsize=(4*num_features, 8))
        
        if num_features == 1:
            axes = axes.reshape(2, 1)
            
        cell_features = cell_graph.x.numpy()
        tissue_features = tissue_graph.x.numpy()
        
        for i, feat_idx in enumerate(feature_indices):
            # Cell feature distribution
            if cell_features.shape[0] > 0 and feat_idx < cell_features.shape[1]:
                axes[0, i].hist(cell_features[:, feat_idx], bins=50, alpha=0.7, 
                               color='red', density=True)
                axes[0, i].set_title(f'Cell Feature {feat_idx}')
                
            # Tissue feature distribution  
            if tissue_features.shape[0] > 0 and feat_idx < tissue_features.shape[1]:
                axes[1, i].hist(tissue_features[:, feat_idx], bins=50, alpha=0.7,
                               color='blue', density=True)
                axes[1, i].set_title(f'Tissue Feature {feat_idx}')
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def create_dummy_hierarchical_graph(batch_size=1, num_cells=50, num_tissues=10):
    """Create dummy hierarchical graphs for testing."""
    generator = SyntheticGraphGenerator({})
    
    all_cell_graphs = []
    all_tissue_graphs = []
    all_clusters = []
    
    for _ in range(batch_size):
        cell_graph = generator.generate_synthetic_cell_graph(num_cells)
        tissue_graph, clusters = generator.generate_synthetic_tissue_graph(
            cell_graph, num_tissues
        )
        
        all_cell_graphs.append(cell_graph)
        all_tissue_graphs.append(tissue_graph)
        all_clusters.append(clusters)
        
    return all_cell_graphs, all_tissue_graphs, all_clusters
