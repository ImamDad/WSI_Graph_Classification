import torch
import torch.nn as nn
import torch.nn.functional as F

class CountPredictor(nn.Module):
    """
    Auxiliary count predictor for nuclear composition.
    Predicts distribution over count bins for different nuclear types.
    """
    
    def __init__(self, input_dim, num_count_bins=10, num_nuclear_types=5):
        super().__init__()
        self.num_count_bins = num_count_bins
        self.num_nuclear_types = num_nuclear_types
        
        # Count prediction heads for different nuclear types
        self.count_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim // 2, num_count_bins)
            ) for _ in range(num_nuclear_types)
        ])
        
        # Nuclear type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_nuclear_types)
        )
        
    def forward(self, x):
        """
        Args:
            x: Graph-level representation [batch_size, input_dim]
            
        Returns:
            count_predictions: List of count distributions for each nuclear type
            type_logits: Nuclear type classification logits
        """
        batch_size = x.shape[0]
        
        # Predict nuclear type distribution
        type_logits = self.type_classifier(x)  # [batch_size, num_nuclear_types]
        
        # Predict count distributions for each nuclear type
        count_predictions = []
        for head in self.count_heads:
            count_logits = head(x)  # [batch_size, num_count_bins]
            count_probs = F.softmax(count_logits, dim=-1)
            count_predictions.append(count_probs)
            
        return {
            'count_predictions': count_predictions,  # List of [batch_size, num_count_bins]
            'type_logits': type_logits  # [batch_size, num_nuclear_types]
        }

class CountLoss(nn.Module):
    """Loss function for auxiliary count prediction task."""
    
    def __init__(self, alpha=0.1, count_loss_weight=1.0, type_loss_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.count_loss_weight = count_loss_weight
        self.type_loss_weight = type_loss_weight
        
        # Count loss (KL divergence between predicted and target distributions)
        self.count_loss_fn = nn.KLDivLoss(reduction='batchmean')
        
        # Type classification loss
        self.type_loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Output from CountPredictor
            targets: Dictionary with 'counts' and 'types'
                counts: List of count bin distributions [num_types, batch_size, num_bins]
                types: Nuclear type labels [batch_size]
        """
        total_loss = 0
        
        # Count prediction loss
        count_predictions = predictions['count_predictions']
        count_targets = targets['counts']
        
        count_loss = 0
        for pred, target in zip(count_predictions, count_targets):
            # Add small epsilon to avoid log(0)
            pred = torch.clamp(pred, min=1e-8)
            target = torch.clamp(target, min=1e-8)
            
            count_loss += self.count_loss_fn(
                torch.log(pred), target
            )
        
        count_loss = count_loss / len(count_predictions)
        total_loss += self.count_loss_weight * count_loss
        
        # Type classification loss
        type_logits = predictions['type_logits']
        type_targets = targets['types']
        
        type_loss = self.type_loss_fn(type_logits, type_targets)
        total_loss += self.type_loss_weight * type_loss
        
        return total_loss

def create_count_bins(max_count=1000, num_bins=10):
    """Create logarithmic count bins for nuclear counting."""
    import numpy as np
    
    # Logarithmic bins for better distribution across scales
    log_bins = np.logspace(0, np.log10(max_count + 1), num_bins + 1)
    bin_edges = np.unique(log_bins.astype(int))
    
    return bin_edges

def convert_counts_to_distribution(counts, bin_edges):
    """
    Convert continuous counts to distribution over count bins.
    
    Args:
        counts: Tensor of actual counts [batch_size]
        bin_edges: Array defining count bin edges
        
    Returns:
        distributions: Tensor of count distributions [batch_size, num_bins]
    """
    batch_size = counts.shape[0]
    num_bins = len(bin_edges) - 1
    
    distributions = torch.zeros(batch_size, num_bins)
    
    for i in range(batch_size):
        count = counts[i].item()
        
        # Find which bin the count falls into
        bin_idx = 0
        for j in range(num_bins):
            if bin_edges[j] <= count < bin_edges[j + 1]:
                bin_idx = j
                break
            elif count >= bin_edges[-1]:
                bin_idx = num_bins - 1
                break
                
        distributions[i, bin_idx] = 1.0
        
    return distributions
