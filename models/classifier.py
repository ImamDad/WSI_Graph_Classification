import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

from .attention_modules import MultiHeadGraphAttention, CrossLevelAttention, HierarchicalAttentionPooling

class HiGATEClassifier(nn.Module):
    """
    Hierarchical Graph Attention Network for WSI Classification.
    Implements the complete HiGATE architecture from the paper.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature dimensions
        self.cell_in_dim = config['cell_in_dim']  # 768 (DINOv2) + 7 (morph) + 12 (nuclear) = 787
        self.tissue_in_dim = config['tissue_in_dim']  # Same as cell_out_dim
        self.hidden_dim = config['hidden_dim']
        self.num_classes = config['num_classes']
        
        # Cell-level encoder
        self.cell_encoder = CellGraphEncoder(
            self.cell_in_dim, 
            self.hidden_dim,
            num_layers=config.get('cell_layers', 3),
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1)
        )
        
        # Tissue-level encoder  
        self.tissue_encoder = TissueGraphEncoder(
            self.hidden_dim,  # Input is cell encoder output
            self.hidden_dim,
            num_layers=config.get('tissue_layers', 2),
            dropout=config.get('dropout', 0.1)
        )
        
        # Cross-level attention
        self.cross_attention = CrossLevelAttention(
            cell_dim=self.hidden_dim,
            tissue_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            dropout=config.get('dropout', 0.1)
        )
        
        # Hierarchical pooling
        self.cell_pooling = HierarchicalAttentionPooling(self.hidden_dim, self.hidden_dim)
        self.tissue_pooling = HierarchicalAttentionPooling(self.hidden_dim, self.hidden_dim)
        
        # Multi-scale fusion
        fusion_dim = self.hidden_dim * 2  # cell + tissue
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.hidden_dim // 4, self.num_classes)
        )
        
        # Count predictor (auxiliary task)
        if config.get('use_count_predictor', True):
            self.count_predictor = CountPredictor(
                self.hidden_dim,
                num_count_bins=config.get('num_count_bins', 10)
            )
        
    def forward(self, cell_graph, tissue_graph, cluster_assignments, return_attention=False):
        """
        Forward pass through HiGATE.
        
        Args:
            cell_graph: PyG Data object for cell graph
            tissue_graph: PyG Data object for tissue graph  
            cluster_assignments: List mapping tissue nodes to constituent cells
            return_attention: Whether to return attention weights for interpretability
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
            count_pred: Optional count predictions for auxiliary task
        """
        # Encode cell graph
        cell_features = self.cell_encoder(
            cell_graph.x, 
            cell_graph.edge_index,
            cell_graph.batch if hasattr(cell_graph, 'batch') else None
        )  # [N_cell, hidden_dim]
        
        # Encode tissue graph
        tissue_features = self.tissue_encoder(
            tissue_graph.x,
            tissue_graph.edge_index, 
            tissue_graph.batch if hasattr(tissue_graph, 'batch') else None
        )  # [N_tissue, hidden_dim]
        
        # Apply cross-level attention
        cell_features_attn, tissue_features_attn = self.cross_attention(
            cell_features, tissue_features, cluster_assignments
        )
        
        # Hierarchical pooling to get graph-level representations
        cell_graph_repr = self._hierarchical_pooling(
            cell_features_attn, cell_graph.batch, self.cell_pooling
        )  # [batch_size, hidden_dim]
        
        tissue_graph_repr = self._hierarchical_pooling(
            tissue_features_attn, tissue_graph.batch, self.tissue_pooling  
        )  # [batch_size, hidden_dim]
        
        # Multi-scale fusion
        fused_repr = torch.cat([cell_graph_repr, tissue_graph_repr], dim=-1)  # [batch_size, 2*hidden_dim]
        fused_repr = self.fusion_net(fused_repr)  # [batch_size, hidden_dim//2]
        
        # Classification
        logits = self.classifier(fused_repr)  # [batch_size, num_classes]
        
        outputs = {'logits': logits}
        
        # Auxiliary count prediction
        if hasattr(self, 'count_predictor'):
            count_pred = self.count_predictor(fused_repr)
            outputs['count_pred'] = count_pred
            
        if return_attention:
            outputs['attention_weights'] = {
                'cell_attention': cell_features_attn,
                'tissue_attention': tissue_features_attn
            }
            
        return outputs
    
    def _hierarchical_pooling(self, features, batch, attention_pooling):
        """Apply hierarchical pooling considering batch structure."""
        if batch is None:
            # Single graph
            return attention_pooling(features).unsqueeze(0)
        else:
            # Batched graphs
            batch_size = batch.max().item() + 1
            graph_reprs = []
            
            for i in range(batch_size):
                graph_mask = batch == i
                graph_features = features[graph_mask]
                if len(graph_features) > 0:
                    graph_repr = attention_pooling(graph_features)
                else:
                    graph_repr = torch.zeros_like(features[0])
                graph_reprs.append(graph_repr)
                
            return torch.stack(graph_reprs)

class CellGraphEncoder(nn.Module):
    """Encoder for cell-level graph using multi-head graph attention."""
    
    def __init__(self, in_dim, hidden_dim, num_layers=3, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Graph attention layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = MultiHeadGraphAttention(
                hidden_dim, hidden_dim, hidden_dim, 
                num_heads=num_heads, dropout=dropout
            )
            self.layers.append(layer)
            
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, batch=None):
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph attention layers with residual connections
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = x
            x = layer(x, edge_index)
            x = norm(x + residual)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Output projection
        x = self.output_proj(x)
        
        return x

class TissueGraphEncoder(nn.Module):
    """Encoder for tissue-level graph using GCN."""
    
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        
        from torch_geometric.nn import GCNConv
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            residual = x
            x = conv(x, edge_index)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual  # Residual connection
            
        return x
