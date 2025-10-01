import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients, Saliency
import networkx as nx

class HiGATEExplainer:
    """Explainability module for HiGATE model interpretations."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def compute_node_importance(self, cell_graph, tissue_graph, cluster_assignments, target_class=None):
        """
        Compute node importance scores using gradient-based attribution.
        
        Args:
            cell_graph: Cell graph data
            tissue_graph: Tissue graph data
            cluster_assignments: Cluster mapping
            target_class: Target class for attribution (None for predicted class)
            
        Returns:
            cell_importance: Node importance scores for cell graph
            tissue_importance: Node importance scores for tissue graph
        """
        # Ensure graphs are on correct device
        cell_graph = cell_graph.to(self.device)
        tissue_graph = tissue_graph.to(self.device)
        
        # Forward pass with gradient tracking
        cell_graph.x.requires_grad_(True)
        tissue_graph.x.requires_grad_(True)
        
        outputs = self.model(cell_graph, tissue_graph, cluster_assignments)
        logits = outputs['logits']
        
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()
            
        # Compute gradients
        self.model.zero_grad()
        logits[0, target_class].backward()
        
        # Node importance as gradient magnitude
        cell_importance = torch.norm(cell_graph.x.grad, dim=1).cpu().detach().numpy()
        tissue_importance = torch.norm(tissue_graph.x.grad, dim=1).cpu().detach().numpy()
        
        return cell_importance, tissue_importance
    
    def compute_attention_rollout(self, cell_graph, tissue_graph, cluster_assignments):
        """
        Compute attention rollout for hierarchical attention visualization.
        
        Args:
            cell_graph: Cell graph data
            tissue_graph: Tissue graph data
            cluster_assignments: Cluster mapping
            
        Returns:
            cell_attention: Attention weights for cell nodes
            tissue_attention: Attention weights for tissue nodes
        """
        # Get attention weights from model
        with torch.no_grad():
            outputs = self.model(
                cell_graph, tissue_graph, cluster_assignments, return_attention=True
            )
            
        attention_weights = outputs['attention_weights']
        cell_attention = attention_weights['cell_attention'].mean(dim=1).cpu().numpy()
        tissue_attention = attention_weights['tissue_attention'].mean(dim=1).cpu().numpy()
        
        return cell_attention, tissue_attention
    
    def visualize_heatmap(self, cell_importance, tissue_importance, 
                         cell_positions, tissue_positions, 
                         cluster_assignments, save_path=None):
        """
        Visualize importance heatmaps for cell and tissue graphs.
        
        Args:
            cell_importance: Importance scores for cell nodes
            tissue_importance: Importance scores for tissue nodes
            cell_positions: Cell node positions
            tissue_positions: Tissue node positions
            cluster_assignments: Cluster mapping
            save_path: Path to save visualization
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Cell-level heatmap
        sc1 = ax1.scatter(cell_positions[:, 0], cell_positions[:, 1], 
                         c=cell_importance, cmap='Reds', s=50, alpha=0.7)
        ax1.set_title('Cell-level Importance')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(sc1, ax=ax1)
        
        # Tissue-level heatmap
        sc2 = ax2.scatter(tissue_positions[:, 0], tissue_positions[:, 1], 
                         c=tissue_importance, cmap='Blues', s=100, alpha=0.7)
        ax2.set_title('Tissue-level Importance')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(sc2, ax=ax2)
        
        # Combined hierarchical visualization
        for i, (tissue_pos, tissue_imp) in enumerate(zip(tissue_positions, tissue_importance)):
            # Draw tissue region
            tissue_circle = plt.Circle(tissue_pos, tissue_imp * 50, 
                                     color='blue', alpha=0.3)
            ax3.add_patch(tissue_circle)
            
            # Draw constituent cells
            cell_indices = cluster_assignments[i]
            for cell_idx in cell_indices:
                cell_pos = cell_positions[cell_idx]
                cell_imp = cell_importance[cell_idx]
                
                ax3.scatter(cell_pos[0], cell_pos[1], 
                           c='red', s=cell_imp * 100, alpha=0.6)
                
                # Draw connection to tissue center
                ax3.plot([cell_pos[0], tissue_pos[0]], 
                        [cell_pos[1], tissue_pos[1]], 
                        'k-', alpha=0.2, linewidth=0.5)
        
        ax3.set_title('Hierarchical Importance')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_saliency_maps(self, cell_graph, tissue_graph, cluster_assignments, 
                              target_class=None, method='integrated_gradients'):
        """
        Generate saliency maps using different attribution methods.
        
        Args:
            cell_graph: Cell graph data
            tissue_graph: Tissue graph data
            cluster_assignments: Cluster mapping
            target_class: Target class for attribution
            method: Attribution method ('integrated_gradients' or 'saliency')
            
        Returns:
            cell_saliency: Saliency scores for cell nodes
            tissue_saliency: Saliency scores for tissue nodes
        """
        if method == 'integrated_gradients':
            ig = IntegratedGradients(self.model)
            
            def forward_fn(cell_x, tissue_x):
                cell_graph.x = cell_x
                tissue_graph.x = tissue_x
                outputs = self.model(cell_graph, tissue_graph, cluster_assignments)
                return outputs['logits']
            
            # Compute attributions
            cell_attr = ig.attribute(
                cell_graph.x.unsqueeze(0), 
                target=target_class,
                additional_forward_args=(tissue_graph.x.unsqueeze(0),)
            )
            
            tissue_attr = ig.attribute(
                tissue_graph.x.unsqueeze(0),
                target=target_class,
                additional_forward_args=(cell_graph.x.unsqueeze(0),)
            )
            
        elif method == 'saliency':
            saliency = Saliency(self.model)
            
            cell_attr = saliency.attribute(
                cell_graph.x.unsqueeze(0),
                target=target_class,
                additional_forward_args=(tissue_graph, cluster_assignments)
            )
            
            tissue_attr = saliency.attribute(
                tissue_graph.x.unsqueeze(0),
                target=target_class,
                additional_forward_args=(cell_graph, cluster_assignments)
            )
            
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Aggregate attributions
        cell_saliency = torch.norm(cell_attr, dim=2).squeeze().cpu().detach().numpy()
        tissue_saliency = torch.norm(tissue_attr, dim=2).squeeze().cpu().detach().numpy()
        
        return cell_saliency, tissue_saliency
    
    def identify_critical_regions(self, cell_importance, tissue_importance, 
                                 cluster_assignments, top_k=10):
        """
        Identify top-k critical regions based on importance scores.
        
        Args:
            cell_importance: Cell node importance
            tissue_importance: Tissue node importance
            cluster_assignments: Cluster mapping
            top_k: Number of top regions to identify
            
        Returns:
            critical_regions: List of critical region information
        """
        # Compute region scores (average of tissue and constituent cell importance)
        region_scores = []
        
        for i, cell_indices in enumerate(cluster_assignments):
            if len(cell_indices) == 0:
                continue
                
            tissue_score = tissue_importance[i]
            cell_scores = cell_importance[cell_indices]
            region_score = 0.7 * tissue_score + 0.3 * np.mean(cell_scores)
            
            region_scores.append({
                'region_id': i,
                'score': region_score,
                'tissue_importance': tissue_score,
                'avg_cell_importance': np.mean(cell_scores),
                'num_cells': len(cell_indices),
                'cell_indices': cell_indices
            })
        
        # Sort by score and get top-k
        region_scores.sort(key=lambda x: x['score'], reverse=True)
        critical_regions = region_scores[:top_k]
        
        return critical_regions

class GraphGradCAM:
    """Graph Grad-CAM for visualizing important graph substructures."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        # Register hooks on target layer
        target_module = getattr(self.model, self.target_layer)
        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)
        
    def compute_cam(self, cell_graph, tissue_graph, cluster_assignments, target_class=None):
        """
        Compute Graph Grad-CAM.
        
        Args:
            cell_graph: Cell graph data
            tissue_graph: Tissue graph data
            cluster_assignments: Cluster mapping
            target_class: Target class for Grad-CAM
            
        Returns:
            cam_weights: Grad-CAM weights
        """
        # Forward pass
        outputs = self.model(cell_graph, tissue_graph, cluster_assignments)
        
        if target_class is None:
            target_class = outputs['logits'].argmax(dim=-1).item()
            
        # Backward pass
        self.model.zero_grad()
        outputs['logits'][0, target_class].backward()
        
        # Compute Grad-CAM weights
        if self.gradients is not None and self.activations is not None:
            weights = torch.mean(self.gradients, dim=1, keepdim=True)  # [N, 1]
            cam = torch.sum(weights * self.activations, dim=-1)  # [N]
            cam = F.relu(cam)  # Apply ReLU to focus on positive influences
            
            return cam.detach().cpu().numpy()
        else:
            raise ValueError("Gradients or activations not captured")
