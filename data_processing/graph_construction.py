import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import Delaunay, KDTree
from sklearn.cluster import OPTICS, DBSCAN
import networkx as nx

def build_cell_graph(nuclei_features: np.ndarray, 
                    nuclei_centroids: np.ndarray,
                    config: dict) -> Data:
    """
    Build cell graph from nuclear features and centroids.
    
    Args:
        nuclei_features: Array of shape (N, D) for N nuclei
        nuclei_centroids: Array of shape (N, 2) for centroid coordinates
        config: Configuration dictionary
        
    Returns:
        PyG Data object representing cell graph
    """
    N = len(nuclei_features)
    
    if N == 0:
        return create_empty_graph()
    
    # Adaptive k-NN connectivity
    k = compute_adaptive_k(N, config)
    
    # Build k-NN graph
    edge_index, edge_attr = build_knn_graph(nuclei_centroids, k)
    
    # Node features
    x = torch.tensor(nuclei_features, dtype=torch.float32)
    
    # Node positions (normalized)
    pos = torch.tensor(nuclei_centroids, dtype=torch.float32)
    pos = (pos - pos.mean(dim=0)) / (pos.std(dim=0) + 1e-8)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

def build_tissue_graph(cell_graph: Data,
                      nuclei_centroids: np.ndarray,
                      config: dict) -> Data:
    """
    Build tissue graph from cell graph using adaptive clustering.
    
    Args:
        cell_graph: Cell graph Data object
        nuclei_centroids: Nuclear centroid coordinates
        config: Configuration dictionary
        
    Returns:
        PyG Data object representing tissue graph
    """
    if cell_graph.num_nodes == 0:
        return create_empty_graph()
    
    # Project to latent space for clustering
    cell_features = cell_graph.x.detach().numpy()
    projected_features = project_to_latent_space(cell_features, config)
    
    # Adaptive clustering
    clusters = adaptive_density_clustering(projected_features, 
                                         nuclei_centroids, 
                                         config)
    
    if len(clusters) == 0:
        return create_empty_graph()
    
    # Aggregate tissue nodes
    tissue_features, tissue_centroids = aggregate_tissue_nodes(
        cell_graph, clusters, nuclei_centroids
    )
    
    # Build tissue graph edges using Delaunay triangulation
    edge_index = build_delaunay_edges(tissue_centroids)
    
    # Create tissue graph
    x = torch.tensor(tissue_features, dtype=torch.float32)
    pos = torch.tensor(tissue_centroids, dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index, pos=pos)

def compute_adaptive_k(N: int, config: dict) -> int:
    """Compute adaptive k for k-NN based on cell density."""
    k_min = config.get('k_min', 3)
    k_max = config.get('k_max', 15)
    
    k = min(k_max, max(k_min, int(np.sqrt(N))))
    return k

def build_knn_graph(centroids: np.ndarray, k: int) -> tuple:
    """Build k-NN graph from centroids."""
    from sklearn.neighbors import kneighbors_graph
    
    # Build k-NN graph
    adj_matrix = kneighbors_graph(centroids, k, mode='connectivity', 
                                 include_self=True)
    
    # Convert to edge index format
    edge_index = np.array(adj_matrix.nonzero())
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # Compute edge attributes (distances)
    edge_attr = compute_edge_attributes(centroids, edge_index)
    
    return edge_index, edge_attr

def compute_edge_attributes(centroids: np.ndarray, edge_index: torch.Tensor) -> torch.Tensor:
    """Compute edge attributes as normalized distances."""
    edge_index_np = edge_index.numpy()
    distances = np.linalg.norm(
        centroids[edge_index_np[0]] - centroids[edge_index_np[1]], 
        axis=1
    )
    
    # Normalize distances
    if len(distances) > 0:
        distances = distances / (np.max(distances) + 1e-8)
    
    return torch.tensor(distances, dtype=torch.float32).unsqueeze(1)

def project_to_latent_space(features: np.ndarray, config: dict) -> np.ndarray:
    """Project features to lower-dimensional latent space."""
    from sklearn.decomposition import PCA
    
    n_components = config.get('latent_dim', 32)
    pca = PCA(n_components=n_components)
    
    if len(features) > n_components:
        projected = pca.fit_transform(features)
    else:
        projected = features  # Use original features if too few samples
        
    return projected

def adaptive_density_clustering(features: np.ndarray,
                              centroids: np.ndarray,
                              config: dict) -> list:
    """Perform adaptive density-based clustering."""
    min_samples = config.get('min_samples', 5)
    eps = config.get('eps', 0.5)
    
    # Combine feature and spatial similarity
    combined_features = np.concatenate([
        features / (np.std(features, axis=0) + 1e-8),
        centroids / (np.std(centroids, axis=0) + 1e-8)
    ], axis=1)
    
    # Use OPTICS for adaptive clustering
    clustering = OPTICS(min_samples=min_samples, max_eps=eps)
    labels = clustering.fit_predict(combined_features)
    
    # Group points by cluster
    clusters = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        cluster_indices = np.where(labels == label)[0]
        clusters.append(cluster_indices)
    
    return clusters

def aggregate_tissue_nodes(cell_graph: Data,
                          clusters: list,
                          centroids: np.ndarray) -> tuple:
    """Aggregate cell features into tissue nodes."""
    tissue_features = []
    tissue_centroids = []
    
    for cluster_indices in clusters:
        # Mean pool cell features
        cluster_features = cell_graph.x[cluster_indices]
        tissue_feature = cluster_features.mean(dim=0).numpy()
        tissue_features.append(tissue_feature)
        
        # Compute tissue centroid
        cluster_centroids = centroids[cluster_indices]
        tissue_centroid = cluster_centroids.mean(axis=0)
        tissue_centroids.append(tissue_centroid)
    
    return np.array(tissue_features), np.array(tissue_centroids)

def build_delaunay_edges(centroids: np.ndarray) -> torch.Tensor:
    """Build edges using Delaunay triangulation."""
    if len(centroids) < 3:
        # Fall back to complete graph for small numbers
        return build_complete_graph(len(centroids))
    
    try:
        tri = Delaunay(centroids)
        edges = set()
        
        # Get edges from Delaunay triangulation
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
        
        # Convert to edge index format
        if edges:
            edge_index = np.array(list(zip(*edges))).T
            edge_index = np.concatenate([edge_index, edge_index[:, [1,0]]], axis=0)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            
    except:
        # Fallback if Delaunay fails
        edge_index = build_complete_graph(len(centroids))
    
    return edge_index

def build_complete_graph(n_nodes: int) -> torch.Tensor:
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

def create_empty_graph() -> Data:
    """Create an empty graph placeholder."""
    return Data(
        x=torch.zeros(0, 768, dtype=torch.float32),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 1, dtype=torch.float32),
        pos=torch.zeros(0, 2, dtype=torch.float32)
    )
