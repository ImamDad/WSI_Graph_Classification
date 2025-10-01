import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json

class ExternalValidator:
    """External validation framework for HiGATE model."""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
    def validate_on_external_dataset(self, test_loader, dataset_name, save_dir=None):
        """
        Comprehensive external validation on a test dataset.
        
        Args:
            test_loader: DataLoader for external test set
            dataset_name: Name of external dataset for reporting
            save_dir: Directory to save results
            
        Returns:
            results: Dictionary containing all evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_metadata = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Validating on {dataset_name}"):
                # Move data to device
                cell_graph = batch['cell_graph'].to(self.device)
                tissue_graph = batch['tissue_graph'].to(self.device)
                labels = batch['labels'].to(self.device)
                metadata = batch['metadata']
                
                # Forward pass
                outputs = self.model(cell_graph, tissue_graph, 
                                   self._get_cluster_assignments(metadata))
                
                # Get predictions
                logits = outputs['logits']
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_metadata.extend(metadata)
        
        # Compute metrics
        results = self._compute_comprehensive_metrics(
            all_predictions, all_targets, all_probabilities, dataset_name
        )
        
        # Generate visualizations
        if save_dir:
            self._generate_validation_plots(results, all_metadata, save_dir, dataset_name)
            self._save_results(results, save_dir, dataset_name)
            
        return results
    
    def _get_cluster_assignments(self, metadata):
        """Extract cluster assignments from metadata."""
        # This depends on how cluster assignments are stored in your data
        # Assuming they're stored in metadata under 'cluster_assignments'
        cluster_assignments = []
        for meta in metadata:
            if 'cluster_assignments' in meta:
                cluster_assignments.append(meta['cluster_assignments'])
            else:
                # Fallback: create dummy assignments (modify based on your data structure)
                cluster_assignments.append(self._create_dummy_assignments(meta))
                
        return cluster_assignments
    
    def _create_dummy_assignments(self, metadata):
        """Create dummy cluster assignments when not available."""
        # This is a placeholder - modify based on your actual data structure
        num_cells = metadata.get('num_cells', 0)
        num_tissues = metadata.get('num_tissues', 1)
        
        if num_cells == 0 or num_tissues == 0:
            return []
            
        # Simple assignment: divide cells evenly among tissue regions
        assignments = []
        cells_per_tissue = max(1, num_cells // num_tissues)
        
        for i in range(num_tissues):
            start_idx = i * cells_per_tissue
            end_idx = min((i + 1) * cells_per_tissue, num_cells)
            if start_idx < num_cells:
                assignments.append(list(range(start_idx, end_idx)))
                
        return assignments
    
    def _compute_comprehensive_metrics(self, predictions, targets, probabilities, dataset_name):
        """Compute comprehensive evaluation metrics."""
        predictions = np.array(predictions)
        targets = np.array(targets)
        probabilities = np.array(probabilities)
        
        num_classes = probabilities.shape[1]
        
        results = {
            'dataset': dataset_name,
            'accuracy': accuracy_score(targets, predictions),
            'macro_f1': f1_score(targets, predictions, average='macro'),
            'weighted_f1': f1_score(targets, predictions, average='weighted'),
            'per_class_f1': f1_score(targets, predictions, average=None).tolist(),
        }
        
        # ROC-AUC for binary and multiclass
        if num_classes == 2:
            results['roc_auc'] = roc_auc_score(targets, probabilities[:, 1])
        else:
            results['macro_auc'] = roc_auc
