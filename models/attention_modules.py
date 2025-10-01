import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import softmax

class MultiHeadGraphAttention(nn.Module):
    """Multi-head Graph Attention Network for cellular-level modeling."""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Multi-head attention layers
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_dim, hidden_dim, dropout=dropout) 
            for _ in range(num_heads)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(num_heads * hidden_dim, out_dim)
        self.activation = nn.ELU()
        
    def forward(self, x, edge_index):
        # Apply multi-head attention
        head_outputs = []
        for attn in self.attentions:
            head_out = attn(x, edge_index)
            head_outputs.append(head_out)
            
        # Concatenate and project
        x = torch.cat(head_outputs, dim=-1)
        x = self.output_proj(x)
        x = self.activation(x)
        
        return x

class GraphAttentionLayer(nn.Module):
    """Single-head graph attention layer."""
    
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        
        # Linear transformations
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        
        # LeakyReLU for attention computation
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)
        
    def forward(self, x, edge_index):
        # Linear transformation
        h = self.W(x)  # [N, out_dim]
        
        # Compute attention coefficients
        source_nodes = h[edge_index[0]]  # [E, out_dim]
        target_nodes = h[edge_index[1]]  # [E, out_dim]
        
        # Concatenate and compute attention
        edge_features = torch.cat([source_nodes, target_nodes], dim=-1)  # [E, 2*out_dim]
        e = self.leaky_relu(self.a(edge_features)).squeeze(-1)  # [E]
        
        # Softmax attention weights
        alpha = softmax(e, edge_index[1], num_nodes=x.size(0))  # [E]
        
        # Apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Aggregate neighborhood features
        out = torch.zeros_like(h)
        source_features = h[edge_index[0]]  # [E, out_dim]
        
        # Scatter add with attention weights
        index = edge_index[1].unsqueeze(-1).expand_as(source_features)
        out = out.scatter_add_(0, index, source_features * alpha.unsqueeze(-1))
        
        return out

class CrossLevelAttention(nn.Module):
    """Bidirectional Cross-Level Attention between cell and tissue graphs."""
    
    def __init__(self, cell_dim, tissue_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.cell_dim = cell_dim
        self.tissue_dim = tissue_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Bottom-up attention (cell-to-tissue)
        self.bottom_up_attn = BottomUpAttention(cell_dim, tissue_dim, hidden_dim, dropout)
        
        # Top-down attention (tissue-to-cell)  
        self.top_down_attn = TopDownAttention(tissue_dim, cell_dim, hidden_dim, dropout)
        
    def forward(self, cell_graph, tissue_graph, cluster_assignments):
        """
        Args:
            cell_graph: Cell graph features [N_cell, cell_dim]
            tissue_graph: Tissue graph features [N_tissue, tissue_dim] 
            cluster_assignments: List of lists mapping tissue nodes to constituent cells
        """
        # Bottom-up: Tissue nodes attend to their constituent cells
        tissue_updated = self.bottom_up_attn(cell_graph, tissue_graph, cluster_assignments)
        
        # Top-down: Cells attend to their parent tissue regions
        cell_updated = self.top_down_attn(cell_graph, tissue_updated, cluster_assignments)
        
        return cell_updated, tissue_updated

class BottomUpAttention(nn.Module):
    """Bottom-up attention: Tissue nodes attend to constituent cells."""
    
    def __init__(self, cell_dim, tissue_dim, hidden_dim, dropout):
        super().__init__()
        self.cell_proj = nn.Linear(cell_dim, hidden_dim)
        self.tissue_query = nn.Linear(tissue_dim, hidden_dim)
        self.attention_weights = nn.Parameter(torch.Tensor(1, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.cell_proj.weight)
        nn.init.xavier_uniform_(self.tissue_query.weight)
        nn.init.xavier_uniform_(self.attention_weights.unsqueeze(0))
        
    def forward(self, cell_features, tissue_features, cluster_assignments):
        """
        Args:
            cell_features: [N_cell, cell_dim]
            tissue_features: [N_tissue, tissue_dim]
            cluster_assignments: List of lists, each containing cell indices for a tissue node
        """
        batch_size = len(cluster_assignments)
        updated_tissues = []
        
        for i, cell_indices in enumerate(cluster_assignments):
            if len(cell_indices) == 0:
                updated_tissues.append(tissue_features[i])
                continue
                
            # Get constituent cells
            constituent_cells = cell_features[cell_indices]  # [K, cell_dim]
            
            # Project cells to hidden space
            cell_keys = self.cell_proj(constituent_cells)  # [K, hidden_dim]
            
            # Tissue node as query
            tissue_query = self.tissue_query(tissue_features[i].unsqueeze(0))  # [1, hidden_dim]
            
            # Compute attention scores
            attention_scores = torch.tanh(cell_keys + tissue_query)  # [K, hidden_dim]
            attention_scores = torch.matmul(attention_scores, self.attention_weights.t())  # [K, 1]
            attention_weights = self.softmax(attention_scores)  # [K, 1]
            
            # Weighted aggregation
            weighted_cells = constituent_cells * attention_weights  # [K, cell_dim]
            aggregated = weighted_cells.sum(dim=0)  # [cell_dim]
            
            # Residual connection
            updated_tissue = tissue_features[i] + self.dropout(aggregated)
            updated_tissues.append(updated_tissue)
            
        return torch.stack(updated_tissues)

class TopDownAttention(nn.Module):
    """Top-down attention: Cells attend to parent tissue regions."""
    
    def __init__(self, tissue_dim, cell_dim, hidden_dim, dropout):
        super().__init__()
        self.tissue_proj = nn.Linear(tissue_dim, hidden_dim)
        self.cell_query = nn.Linear(cell_dim, hidden_dim)
        self.attention_weights = nn.Parameter(torch.Tensor(1, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.tissue_proj.weight)
        nn.init.xavier_uniform_(self.cell_query.weight)
        nn.init.xavier_uniform_(self.attention_weights.unsqueeze(0))
        
    def forward(self, cell_features, tissue_features, cluster_assignments):
        """
        Args:
            cell_features: [N_cell, cell_dim]
            tissue_features: [N_tissue, tissue_dim]
            cluster_assignments: List of lists mapping tissue nodes to constituent cells
        """
        # Create mapping from cells to their tissue parents
        cell_to_tissue = {}
        for tissue_idx, cell_indices in enumerate(cluster_assignments):
            for cell_idx in cell_indices:
                cell_to_tissue[cell_idx] = tissue_idx
        
        updated_cells = []
        
        for cell_idx in range(len(cell_features)):
            if cell_idx not in cell_to_tissue:
                updated_cells.append(cell_features[cell_idx])
                continue
                
            tissue_idx = cell_to_tissue[cell_idx]
            tissue_feature = tissue_features[tissue_idx]  # [tissue_dim]
            
            # Project tissue to hidden space
            tissue_key = self.tissue_proj(tissue_feature.unsqueeze(0))  # [1, hidden_dim]
            
            # Cell as query
            cell_query = self.cell_query(cell_features[cell_idx].unsqueeze(0))  # [1, hidden_dim]
            
            # Compute attention score
            attention_score = torch.tanh(tissue_key + cell_query)  # [1, hidden_dim]
            attention_score = torch.matmul(attention_score, self.attention_weights.t())  # [1, 1]
            attention_weight = torch.sigmoid(attention_score)  # [1, 1]
            
            # Residual connection
            updated_cell = cell_features[cell_idx] + self.dropout(tissue_feature * attention_weight)
            updated_cells.append(updated_cell)
            
        return torch.stack(updated_cells)

class HierarchicalAttentionPooling(nn.Module):
    """Hierarchical attention pooling for graph-level representation."""
    
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        """Compute attention-weighted graph representation."""
        # Compute attention scores
        attention_scores = self.attention_net(x)  # [N, 1]
        attention_weights = F.softmax(attention_scores, dim=0)  # [N, 1]
        
        # Weighted sum
        graph_representation = torch.sum(x * attention_weights, dim=0)  # [in_dim]
        
        return graph_representation
