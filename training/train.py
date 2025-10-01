import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from pathlib import Path
import logging
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import os
from config import config
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_DIR / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GraphDataset(Dataset):
    """Dataset class that handles single-node graphs properly"""
    
    def __init__(self, graph_dir: Path, split: str):
        self.graph_dir = graph_dir / split
        self.cell_graph_dir = self.graph_dir / "cell_graphs"
        self.split = split
        
        # Load all graph files
        self.graph_files = sorted(list(self.cell_graph_dir.glob("*.pt")), key=lambda x: x.name)
        
        # Load labels
        self.labels = self._load_labels()
        
        logger.info(f"Loaded {len(self.graph_files)} graphs for {self.split} split")

    def _load_labels(self) -> Dict[str, int]:
        """Load labels from CSV files"""
        labels = {}
        for graph_file in self.graph_files:
            # Simple label assignment for now
            labels[graph_file.stem] = np.random.randint(0, config.NUM_CLASSES)
        return labels

    def __len__(self) -> int:
        return len(self.graph_files)

    def __getitem__(self, idx: int) -> Dict:
        graph_file = self.graph_files[idx]
        image_id = graph_file.stem
        
        try:
            # Load cell graph
            cell_graph = torch.load(graph_file)
            
            # Ensure the graph has valid structure
            cell_graph = self._validate_graph(cell_graph)
            
            # Get label
            label = self.labels.get(image_id, 0)
            
            return {
                'cell_graph': cell_graph,
                'label': label,
                'image_id': image_id
            }
            
        except Exception as e:
            logger.error(f"Error loading {image_id}: {str(e)}")
            return self._create_empty_sample()

    def _validate_graph(self, graph: Data) -> Data:
        """Ensure graph has proper structure"""
        if not hasattr(graph, 'x'):
            graph.x = torch.zeros((1, config.CELL_FEATURE_DIM))
        if not hasattr(graph, 'edge_index'):
            graph.edge_index = torch.empty((2, 0), dtype=torch.long)
        if not hasattr(graph, 'pos'):
            graph.pos = torch.zeros((graph.x.size(0), 2))
        return graph

    def _create_empty_sample(self) -> Dict:
        """Create empty sample for corrupted files"""
        return {
            'cell_graph': Data(
                x=torch.zeros((1, config.CELL_FEATURE_DIM)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                pos=torch.zeros((1, 2))
            ),
            'label': 0,
            'image_id': 'empty'
        }

class SparseGraphTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize model
        from models.hierarchical_gnn import HierarchicalGNN
        self.model = HierarchicalGNN(
            cnn_feature_dim=config.CNN_FEATURE_DIM,
            morph_feature_dim=config.MORPH_FEATURE_DIM,
            num_classes=config.NUM_CLASSES,
            hidden_dim=config.GNN_HIDDEN_DIM,
            dropout=config.DROPOUT_RATE
        ).to(self.device)
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=config.LR_PATIENCE
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            try:
                # Prepare batch data
                cell_graphs = batch['cell_graph']
                labels = batch['label'].to(self.device)
                
                # Create batch for PyG
                cell_batch = Batch.from_data_list(cell_graphs).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model.forward_simple({
                    'cnn_features': cell_batch.x[:, :self.config.CNN_FEATURE_DIM],
                    'morph_features': cell_batch.x[:, self.config.CNN_FEATURE_DIM:],
                    'cell_edge_index': cell_batch.edge_index,
                    'batch': cell_batch.batch
                })
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    # Prepare batch data
                    cell_graphs = batch['cell_graph']
                    labels = batch['label'].to(self.device)
                    
                    # Create batch for PyG
                    cell_batch = Batch.from_data_list(cell_graphs).to(self.device)
                    
                    # Forward pass
                    outputs = self.model.forward_simple({
                        'cnn_features': cell_batch.x[:, :self.config.CNN_FEATURE_DIM],
                        'morph_features': cell_batch.x[:, self.config.CNN_FEATURE_DIM:],
                        'cell_edge_index': cell_batch.edge_index,
                        'batch': cell_batch.batch
                    })
                    
                    # Calculate loss
                    loss = self.criterion(outputs, labels)
                    
                    # Statistics
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {str(e)}")
                    continue
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.config.MODEL_SAVE_PATH / 'latest_checkpoint.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.config.MODEL_SAVE_PATH / 'best_model.pth')
            logger.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_accuracies = checkpoint.get('train_accuracies', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            return checkpoint['epoch']
        return 0

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Create datasets and dataloaders
        train_dataset = GraphDataset(self.config.GRAPH_DATA_PATH, "train")
        val_dataset = GraphDataset(self.config.GRAPH_DATA_PATH, "val")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=self._collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=self._collate_fn
        )
        
        # Load checkpoint if exists
        start_epoch = self.load_checkpoint(self.config.MODEL_SAVE_PATH / 'latest_checkpoint.pth')
        
        # Training loop
        for epoch in range(start_epoch, self.config.EPOCHS):
            logger.info(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # Check for improvement
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_checkpoint(self.config.EPOCHS)
        logger.info("Training completed!")

    def _collate_fn(self, batch):
        """Custom collate function for graph data"""
        return {
            'cell_graph': [item['cell_graph'] for item in batch],
            'label': torch.tensor([item['label'] for item in batch]),
            'image_id': [item['image_id'] for item in batch]
        }

def main():
    """Main function for training"""
    trainer = SparseGraphTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
