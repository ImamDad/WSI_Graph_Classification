import numpy as np
import torch
from pathlib import Path
import logging
from typing import Tuple, List, Dict
import pandas as pd

logger = logging.getLogger(__name__)

class Config:
    """Configuration class for WSI Graph Classification project"""
    
    def __init__(self):
        # Hardware configuration
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_WORKERS = min(4, torch.multiprocessing.cpu_count() // 2)
        
        # Data paths
        self.DATA_ROOT = Path("F:/PanNuke")
        self.TRAIN_FOLD = self.DATA_ROOT / "fold0"
        self.VAL_FOLD = self.DATA_ROOT / "fold1"
        self.TEST_FOLD = self.DATA_ROOT / "fold2"
        
        # Directory names
        self.IMAGES_DIR = "extracted_images_npy"
        self.MASKS_DIR = "extracted_masks"
        self.LABELS_FILE = "cell_counts.csv"
        
        # Model parameters
        self.NUM_CLASSES = 5
        self.CLASS_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
        self.MULTI_LABEL = True
        
        # Feature dimensions
        self.CNN_FEATURE_DIM = 768  # DINOv2 features
        self.MORPH_FEATURE_DIM = 6  # Morphological features
        self.CELL_FEATURE_DIM = self.CNN_FEATURE_DIM + self.MORPH_FEATURE_DIM
        self.TISSUE_FEATURE_DIM = 256
        
        # Graph parameters
        self.GNN_HIDDEN_DIM = 128
        self.K_MIN = 3
        self.K_MAX = 15
        self.MAX_NODES = 100
        self.NUM_HEADS = 4
        self.DROPOUT_RATE = 0.2
        self.ATTENTION_DROPOUT = 0.1

        # Training parameters
        self.BATCH_SIZE = 16
        self.USE_AMP = False  # Automatic Mixed Precision
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 1e-5
        self.EPOCHS = 100
        self.EARLY_STOPPING_PATIENCE = 10
        self.GRADIENT_CLIP = 1.0
        self.LR_PATIENCE = 5
        
        # Visualization and debugging
        self.VISUALIZE_SAMPLES = False
        self.DEBUG = False
        
        # Output directories
        self.RESULTS_PATH = Path("results")
        self.MODEL_SAVE_PATH = Path("saved_models")
        self.LOG_DIR = Path("logs")
        self.GRAPH_DATA_PATH = Path(__file__).parent / "graph_data"
        
        # Initialize with default morphological stats
        self.MORPH_MEAN = [0.0] * self.MORPH_FEATURE_DIM
        self.MORPH_STD = [1.0] * self.MORPH_FEATURE_DIM
        
        # Setup directories and validate paths
        self._initialize()
        self.calculate_morph_stats()

    def _initialize(self):
        """Ensure directories exist and validate paths"""
        # Create output directories
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        self.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        self.GRAPH_DATA_PATH.mkdir(parents=True, exist_ok=True)
        
        # Validate dataset paths
        for fold in [self.TRAIN_FOLD, self.VAL_FOLD, self.TEST_FOLD]:
            if not fold.exists():
                raise FileNotFoundError(f"Dataset folder not found: {fold}")
            
            required_dirs = [fold/self.IMAGES_DIR, fold/self.MASKS_DIR]
            for dir_path in required_dirs:
                if not dir_path.exists():
                    raise FileNotFoundError(f"Required directory not found: {dir_path}")
                
            labels_path = fold/self.LABELS_FILE
            if not labels_path.exists():
                raise FileNotFoundError(f"Labels file not found: {labels_path}")

    def calculate_morph_stats(self):
        """Calculate morphological feature statistics from training data"""
        try:
            from data_processing.feature_extraction import CachedFeatureExtractor
            
            extractor = CachedFeatureExtractor()
            all_features = []
            sample_count = 0
            max_samples = 100
            
            for img_file in (self.TRAIN_FOLD/self.IMAGES_DIR).glob('*.npy'):
                if sample_count >= max_samples:
                    break
                    
                try:
                    image = np.load(img_file)
                    mask_file = self.TRAIN_FOLD/self.MASKS_DIR/img_file.name
                    mask = np.load(mask_file) if mask_file.exists() else None
                    
                    if mask is not None:
                        features = extractor.extract_features(image, mask)
                        if 'morph_features' in features and features['morph_features'].shape[0] > 0:
                            all_features.append(features['morph_features'])
                            sample_count += 1
                            
                except Exception as e:
                    logger.warning(f"Error processing {img_file.name}: {str(e)}")
                    continue
            
            if all_features:
                all_features = torch.cat(all_features, dim=0)
                self.MORPH_MEAN = all_features.mean(dim=0).tolist()
                self.MORPH_STD = all_features.std(dim=0).tolist()
                logger.info(f"Calculated morph stats from {sample_count} samples")
                
        except Exception as e:
            logger.warning(f"Could not calculate morph stats: {str(e)}")

# Instantiate the config
config = Config()
