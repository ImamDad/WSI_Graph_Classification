import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import regionprops
from torch.cuda.amp import autocast
from typing import Dict, List, Optional, Tuple, Any, Union
import hashlib
import warnings
import logging
from torchvision import transforms
from pathlib import Path

logger = logging.getLogger(__name__)

# Type alias for region properties
RegionProperties = Any

class LocalDinoFeatureExtractor:
    def __init__(self):
        self._model = None
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(518),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        self._initialized = False
        self._device = None
        
    def initialize(self, device: torch.device) -> bool:
        """Initialize DINOv2 model with robust error handling"""
        if self._initialized:
            return True
            
        try:
            self._model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self._model = self._model.to(device).eval()
            self._device = device
            self._initialized = True
            logger.info("DINOv2 model successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize DINOv2 model: {str(e)}")
            self._model = None
            return False

    @torch.no_grad()
    def extract_features(self, patches: torch.Tensor) -> torch.Tensor:
        """Safe feature extraction with fallback"""
        if not self._initialized or len(patches) == 0:
            return torch.zeros((len(patches), 768), device='cpu')
            
        try:
            # Process patches in batches to avoid OOM errors
            batch_size = 32
            features = []
            
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i+batch_size]
                processed_batch = torch.stack([self._transform(patch) for patch in batch])
                processed_batch = processed_batch.to(self._device)
                
                with autocast():
                    features.append(self._model(processed_batch).cpu())
            
            return torch.cat(features, dim=0)
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return torch.zeros((len(patches), 768), device='cpu')

class CachedFeatureExtractor:
    def __init__(self, 
                 morph_mean: List[float] = [0.0]*6,
                 morph_std: List[float] = [1.0]*6):
        """
        Initialize feature extractor with morphological normalization parameters
        
        Args:
            morph_mean: Mean values for 6 morphological features
            morph_std: Std values for 6 morphological features
        """
        self.morph_mean = torch.tensor(morph_mean)
        self.morph_std = torch.tensor(morph_std)
        self.patch_size = 224
        self.feature_cache = {}
        self._dino_extractor = LocalDinoFeatureExtractor()
        self._device = None
        self._initialized = False

    def initialize(self, device: torch.device) -> bool:
        """Initialize with device placement and status checking"""
        self._device = device
        self.morph_mean = self.morph_mean.to(device)
        self.morph_std = self.morph_std.to(device)
        self._initialized = self._dino_extractor.initialize(device)
        return self._initialized

    def _get_region_properties(self, mask: np.ndarray) -> List[RegionProperties]:
        """Get region properties with robust handling"""
        try:
            from skimage.measure import label
            labeled = label(mask > 0)
            return regionprops(labeled)
        except Exception as e:
            logger.warning(f"Region props extraction failed: {str(e)}")
            return []

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert mask to consistent uint8 format"""
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
        return mask

    def _extract_patches(self, image: np.ndarray, props: List[RegionProperties]) -> Tuple:
        """Extract image patches around each region with validation"""
        patches = []
        patch_hashes = []
        valid_indices = []
        
        for idx, prop in enumerate(props):
            if prop.area < 10:  # Skip small regions
                continue
                
            try:
                # Get bounding box coordinates safely
                bbox = prop.bbox
                if len(bbox) != 4:  # Should be (min_row, min_col, max_row, max_col)
                    continue
                    
                minr, minc, maxr, maxc = bbox
                
                # Ensure coordinates are within image bounds
                h, w = image.shape[:2]
                minr = max(0, minr)
                minc = max(0, minc)
                maxr = min(h, maxr)
                maxc = min(w, maxc)
                
                # Skip if region is invalid
                if minr >= maxr or minc >= maxc:
                    continue
                    
                patch = image[minr:maxr, minc:maxc]
                
                # Handle empty or grayscale patches
                if patch.size == 0:
                    patch = np.zeros((10, 10, 3), dtype=np.uint8)
                elif patch.ndim == 2:
                    patch = np.stack([patch]*3, axis=-1)
                elif patch.ndim == 3 and patch.shape[2] == 1:
                    patch = np.concatenate([patch]*3, axis=-1)
                    
                patches.append(patch)
                patch_hashes.append(self._get_patch_hash(bbox, image.shape))
                valid_indices.append(idx)
                
            except Exception as e:
                logger.warning(f"Patch extraction failed for region {idx}: {str(e)}")
                continue
                
        return patches, patch_hashes, valid_indices

    def _get_patch_hash(self, bbox: Tuple, image_shape: Tuple) -> str:
        """Generate consistent hash for caching"""
        return hashlib.md5(f"{bbox}_{image_shape}".encode()).hexdigest()

    def _get_cnn_features(self, patches: List[np.ndarray], hashes: List[str]) -> torch.Tensor:
        """Get CNN features with caching mechanism"""
        if not patches or not self._initialized:
            return torch.zeros((len(patches), 768), device=self._device)
            
        # Find uncached patches
        uncached_idx = [i for i, h in enumerate(hashes) if h not in self.feature_cache]
        
        if uncached_idx:
            # Process uncached patches
            uncached_patches = [patches[i] for i in uncached_idx]
            patch_tensors = torch.stack([self._process_patch(p) for p in uncached_patches])
            
            # Extract features
            features = self._dino_extractor.extract_features(patch_tensors)
            
            # Update cache
            for i, feat in zip(uncached_idx, features):
                self.feature_cache[hashes[i]] = feat.to('cpu')
                
        # Retrieve all features
        return torch.stack([self.feature_cache[h].to(self._device) for h in hashes])

    def _process_patch(self, patch: np.ndarray) -> torch.Tensor:
        """Convert patch to normalized tensor"""
        try:
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
            return F.interpolate(
                patch_tensor.unsqueeze(0), 
                size=self.patch_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        except Exception:
            return torch.zeros((3, self.patch_size, self.patch_size))

    def _get_morph_features(self, props: List[RegionProperties]) -> torch.Tensor:
        """Extract 6 standardized morphological features"""
        features = []
        for prop in props:
            features.append([
                prop.area / 1000.0,  # Scale down large areas
                self._safe_get(prop, 'eccentricity', 0.0),
                self._safe_get(prop, 'solidity', 1.0),
                self._safe_get(prop, 'perimeter', 0.0) / 100.0,  # Scale down perimeters
                self._safe_get(prop, 'extent', 0.0),
                self._safe_axis_ratio(prop)
            ])
        return torch.tensor(features, device=self._device, dtype=torch.float32)

    def _safe_get(self, prop, attr, default):
        """Safely get property attribute with fallback"""
        try:
            val = getattr(prop, attr)
            return val if not np.isnan(val) else default
        except (AttributeError, NotImplementedError):
            return default

    def _safe_axis_ratio(self, prop: RegionProperties) -> float:
        """Calculate axis ratio with fallback"""
        try:
            ratio = prop.major_axis_length / max(prop.minor_axis_length, 1e-6)
            return ratio if not np.isnan(ratio) else 1.0
        except (AttributeError, NotImplementedError):
            return 1.0

    def extract_features(self, image: Union[np.ndarray, torch.Tensor],
                        mask: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Main feature extraction with proper device handling"""
        try:
            # Convert inputs to numpy if they are tensors
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy() if image.is_cuda else image.numpy()
                if image.ndim == 3 and image.shape[0] == 3:  # CHW to HWC
                    image = image.transpose(1, 2, 0)
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy() if mask.is_cuda else mask.numpy()
                mask = mask.squeeze()  # Remove channel dim if exists
            
            # Validate inputs
            if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
                raise ValueError("Image and mask must be numpy arrays or torch tensors")
                
            if image.shape[:2] != mask.shape[:2]:
                raise ValueError(f"Image and mask spatial dimensions mismatch. "
                            f"Image: {image.shape[:2]}, Mask: {mask.shape[:2]}")
                
            # Ensure proper data types
            image = image.astype(np.float32) if image.dtype != np.float32 else image.copy()
            mask = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)

            # Process mask and extract regions
            processed_mask = self._preprocess_mask(mask)
            props = self._get_region_properties(processed_mask)
            if not props:
                return self._get_empty_features()

            # Extract patches and features
            patches, patch_hashes, valid_indices = self._extract_patches(image, props)
            
            if not patches:
                return self._get_empty_features()

            # Extract CNN features
            cnn_features = self._get_cnn_features(patches, patch_hashes).float()
            
            # Extract morphological features
            morph_features = self._get_morph_features([props[i] for i in valid_indices]).float()
            
            # Normalize morphological features
            morph_features = (morph_features - self.morph_mean) / self.morph_std
            
            # Get centroids
            centroids = np.array([props[i].centroid for i in valid_indices], dtype=np.float32)
            
            # Combine features
            combined_features = torch.cat([cnn_features, morph_features], dim=1)
            
            return {
                'cnn_features': cnn_features,
                'morph_features': morph_features,
                'centroids': torch.from_numpy(centroids).to(self._device),
                'combined_features': combined_features
            }
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}", exc_info=True)
            return self._get_empty_features()

    def _get_empty_features(self) -> Dict[str, torch.Tensor]:
        """Return empty features with correct dimensions and device"""
        device = self._device if self._device else 'cpu'
        empty = torch.zeros(0, device=device)
        return {
            'cnn_features': empty.view(0, 768),
            'morph_features': empty.view(0, 6),
            'centroids': torch.empty((0, 2), device=device),
            'combined_features': empty.view(0, 774)
        }
