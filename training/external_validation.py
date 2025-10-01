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
            results['macro_auc'] = roc_auc_score(targets, probabilities, 
                                               multi_class='ovo', average='macro')
            results['weighted_auc'] = roc_auc_score(targets, probabilities,
                                                  multi_class='ovo', average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        results['confusion_matrix'] = cm.tolist()
        
        # Per-class accuracy
        per_class_accuracy = []
        for i in range(num_classes):
            class_mask = targets == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(targets[class_mask], predictions[class_mask])
                per_class_accuracy.append(class_acc)
            else:
                per_class_accuracy.append(0.0)
                
        results['per_class_accuracy'] = per_class_accuracy
        
        # Additional metrics for clinical relevance
        results['sensitivity'] = self._compute_sensitivity(cm)
        results['specificity'] = self._compute_specificity(cm)
        results['ppv'] = self._compute_ppv(cm)  # Positive Predictive Value
        results['npv'] = self._compute_npv(cm)  # Negative Predictive Value
        
        return results
    
    def _compute_sensitivity(self, cm):
        """Compute sensitivity (recall) for each class."""
        sensitivities = []
        for i in range(len(cm)):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            sensitivities.append(sensitivity)
        return sensitivities
    
    def _compute_specificity(self, cm):
        """Compute specificity for each class."""
        specificities = []
        n_classes = len(cm)
        for i in range(n_classes):
            tn = 0
            for j in range(n_classes):
                for k in range(n_classes):
                    if j != i and k != i:
                        tn += cm[j, k]
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(specificity)
        return specificities
    
    def _compute_ppv(self, cm):
        """Compute Positive Predictive Value for each class."""
        ppvs = []
        for i in range(len(cm)):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            ppvs.append(ppv)
        return ppvs
    
    def _compute_npv(self, cm):
        """Compute Negative Predictive Value for each class."""
        npvs = []
        n_classes = len(cm)
        for i in range(n_classes):
            tn = 0
            for j in range(n_classes):
                for k in range(n_classes):
                    if j != i and k != i:
                        tn += cm[j, k]
            fn = np.sum(cm[i, :]) - cm[i, i]
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            npvs.append(npv)
        return npvs
    
    def _generate_validation_plots(self, results, metadata, save_dir, dataset_name):
        """Generate comprehensive validation plots."""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_{dataset_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Metric Comparison Bar Plot
        metrics = ['accuracy', 'macro_f1', 'weighted_f1']
        if 'roc_auc' in results:
            metrics.append('roc_auc')
        else:
            metrics.extend(['macro_auc', 'weighted_auc'])
            
        values = [results[metric] for metric in metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.ylim(0, 1)
        plt.title(f'Performance Metrics - {dataset_name}')
        plt.ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
            
        plt.savefig(os.path.join(save_dir, f'metrics_comparison_{dataset_name}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Per-class Performance
        if len(results['per_class_f1']) > 1:
            classes = range(len(results['per_class_f1']))
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # F1 scores
            ax1.bar(classes, results['per_class_f1'], color='lightseagreen')
            ax1.set_title(f'Per-class F1 Score - {dataset_name}')
            ax1.set_xlabel('Class')
            ax1.set_ylabel('F1 Score')
            ax1.set_ylim(0, 1)
            
            # Accuracy
            ax2.bar(classes, results['per_class_accuracy'], color='coral')
            ax2.set_title(f'Per-class Accuracy - {dataset_name}')
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'per_class_performance_{dataset_name}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_results(self, results, save_dir, dataset_name):
        """Save validation results to JSON file."""
        results_file = os.path.join(save_dir, f'validation_results_{dataset_name}.json')
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {results_file}")
        
        # Also save as CSV for easy analysis
        csv_file = os.path.join(save_dir, f'validation_results_{dataset_name}.csv')
        
        # Flatten results for CSV
        flat_results = {}
        for key, value in results.items():
            if isinstance(value, list):
                for i, v in enumerate(value):
                    flat_results[f'{key}_{i}'] = v
            else:
                flat_results[key] = value
                
        pd.DataFrame([flat_results]).to_csv(csv_file, index=False)
        print(f"CSV results saved to {csv_file}")

class CrossDomainValidator:
    """Cross-domain validation for assessing model generalization."""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
    def run_cross_domain_analysis(self, domain_loaders):
        """
        Run cross-domain validation across multiple datasets.
        
        Args:
            domain_loaders: Dictionary of {domain_name: data_loader}
            
        Returns:
            cross_domain_results: Results across all domains
        """
        cross_domain_results = {}
        
        # Validate on each domain
        for domain_name, loader in domain_loaders.items():
            validator = ExternalValidator(self.model, self.device, self.config)
            results = validator.validate_on_external_dataset(loader, domain_name)
            cross_domain_results[domain_name] = results
            
        # Generate cross-domain comparison
        self._generate_cross_domain_comparison(cross_domain_results)
        
        return cross_domain_results
    
    def _generate_cross_domain_comparison(self, cross_domain_results):
        """Generate comparison plots across domains."""
        domains = list(cross_domain_results.keys())
        metrics = ['accuracy', 'macro_f1', 'weighted_f1']
        
        # Extract metric values for each domain
        metric_data = {metric: [] for metric in metrics}
        
        for domain in domains:
            results = cross_domain_results[domain]
            for metric in metrics:
                metric_data[metric].append(results[metric])
        
        # Create comparison plot
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            axes[i].bar(domains, metric_data[metric], color='lightblue')
            axes[i].set_title(f'{metric.upper()} across Domains')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylim(0, 1)
            
            # Add value labels
            for j, value in enumerate(metric_data[metric]):
                axes[i].text(j, value + 0.01, f'{value:.3f}', 
                           ha='center', va='bottom')
                
        plt.tight_layout()
        plt.savefig('cross_domain_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
