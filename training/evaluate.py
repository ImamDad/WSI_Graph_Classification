import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import json

from config import config
from models.hierarchical_gnn import HierarchicalGNN
from data_processing.dataset import GraphDataset

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, model_path: str):
        self.device = torch.device(config.DEVICE)
        self.model_path = Path(model_path)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Results directory
        self.results_dir = config.RESULTS_PATH / "evaluation"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> HierarchicalGNN:
        """Load trained model with error handling"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Create model
            model = HierarchicalGNN(
                cnn_feature_dim=config.CNN_FEATURE_DIM,
                morph_feature_dim=config.MORPH_FEATURE_DIM,
                num_classes=config.NUM_CLASSES,
                hidden_dim=config.GNN_HIDDEN_DIM,
                dropout=config.DROPOUT_RATE
            ).to(self.device)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Successfully loaded model from {self.model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def evaluate(self):
        """Run comprehensive evaluation"""
        logger.info("Starting evaluation...")
        
        # Load test dataset
        test_dataset = GraphDataset(config.GRAPH_DATA_PATH, "test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            collate_fn=self._collate_fn
        )
        
        # Run evaluation
        results = self._evaluate_epoch(test_loader)
        
        # Save results
        self._save_results(results)
        
        # Generate plots
        self._generate_plots(results)
        
        logger.info("Evaluation completed!")
        return results

    def _evaluate_epoch(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set"""
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_image_ids = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                try:
                    # Prepare data
                    cell_graphs = batch['cell_graph']
                    labels = batch['label'].to(self.device)
                    image_ids = batch['image_id']
                    
                    # Create batch
                    cell_batch = Batch.from_data_list(cell_graphs).to(self.device)
                    
                    # Forward pass
                    outputs = self.model.forward_simple({
                        'cnn_features': cell_batch.x[:, :config.CNN_FEATURE_DIM],
                        'morph_features': cell_batch.x[:, config.CNN_FEATURE_DIM:],
                        'cell_edge_index': cell_batch.edge_index,
                        'batch': cell_batch.batch
                    })
                    
                    # Get predictions
                    probabilities = F.softmax(outputs, dim=1)
                    _, predictions = outputs.max(1)
                    
                    # Store results
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_image_ids.extend(image_ids)
                    
                except Exception as e:
                    logger.error(f"Error in evaluation batch: {str(e)}")
                    continue
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_targets, all_probabilities)
        
        return {
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'image_ids': all_image_ids,
            'metrics': metrics
        }

    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                          probabilities: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        # Classification report
        class_report = classification_report(
            targets, predictions, 
            target_names=config.CLASS_NAMES,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # AUC-ROC (if multi-class)
        try:
            if config.NUM_CLASSES > 2:
                auc_roc = roc_auc_score(
                    F.one_hot(torch.from_numpy(targets), probabilities, 
                    multi_class='ovr', average='macro'
                )
            else:
                auc_roc = roc_auc_score(targets, probabilities[:, 1])
        except:
            auc_roc = 0.0
        
        # Overall accuracy
        accuracy = (predictions == targets).mean()
        
        return {
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'auc_roc': auc_roc,
            'accuracy': accuracy,
            'per_class_accuracy': cm.diagonal() / cm.sum(axis=1)
        }

    def _save_results(self, results: Dict):
        """Save evaluation results to files"""
        # Save metrics as JSON
        metrics_file = self.results_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        # Save predictions as CSV
        predictions_df = pd.DataFrame({
            'image_id': results['image_ids'],
            'true_label': results['targets'],
            'predicted_label': results['predictions'],
            'confidence': np.max(results['probabilities'], axis=1)
        })
        predictions_file = self.results_dir / "predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)
        
        logger.info(f"Results saved to {self.results_dir}")

    def _generate_plots(self, results: Dict):
        """Generate evaluation plots"""
        # Confusion matrix heatmap
        plt.figure(figsize=(10, 8))
        cm = np.array(results['metrics']['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=config.CLASS_NAMES,
                   yticklabels=config.CLASS_NAMES)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Accuracy per class
        plt.figure(figsize=(10, 6))
        class_acc = results['metrics']['per_class_accuracy']
        plt.bar(config.CLASS_NAMES, class_acc)
        plt.title('Accuracy per Class')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'class_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _collate_fn(self, batch):
        """Custom collate function"""
        return {
            'cell_graph': [item['cell_graph'] for item in batch],
            'label': torch.tensor([item['label'] for item in batch]),
            'image_id': [item['image_id'] for item in batch]
        }

def main():
    """Main evaluation function"""
    model_path = config.MODEL_SAVE_PATH / "best_model.pth"
    evaluator = Evaluator(model_path)
    results = evaluator.evaluate()
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"AUC-ROC: {results['metrics']['auc_roc']:.4f}")
    print("\nPer-class Accuracy:")
    for class_name, acc in zip(config.CLASS_NAMES, results['metrics']['per_class_accuracy']):
        print(f"  {class_name}: {acc:.4f}")

if __name__ == "__main__":
    main()
