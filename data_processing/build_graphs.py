import os
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN, OPTICS
import h5py

from .graph_construction import build_cell_graph, build_tissue_graph
from .transforms import GraphTransform

class GraphBuilder:
    """Builds hierarchical graphs from WSI patches and nuclear segmentation data."""
    
    def __init__(self, config):
        self.config = config
        self.transform = GraphTransform(config)
        
    def load_patch_data(self, patch_path, mask_path):
        """Load patch image and corresponding nuclear segmentation mask."""
        # Implementation depends on your data format
        # Assuming HDF5 format with 'image' and 'mask' datasets
        with h5py.File(patch_path, 'r') as f:
            image = f['image'][:]
            mask = f['mask'][:]
            
        return image, mask
    
    def extract_nuclear_features(self, mask, image):
        """Extract multimodal features for each nucleus."""
        from cellpose import models
        import cv2
        
        # Instance segmentation refinement
        model = models.Cellpose(gpu=False, model_type='cyto')
        masks, flows, styles = model.eval(image, diameter=30)
        
        nuclei_features = []
        nuclei_centroids = []
        
        for nucleus_id in np.unique(masks):
            if nucleus_id == 0:  # Skip background
                continue
                
            # Get nucleus mask
            nucleus_mask = masks == nucleus_id
            
            # Extract centroid
            moments = cv2.moments(nucleus_mask.astype(np.uint8))
            if moments["m00"] != 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
                nuclei_centroids.append([cx, cy])
            else:
                continue
                
            # Extract morphological features
            morph_features = self._extract_morphological_features(nucleus_mask)
            
            # Extract visual features from image patch
            vis_features = self._extract_visual_features(image, nucleus_mask)
            
            # Extract fine-grained morphometric features
            nuc_features = self._extract_nuclear_morphometrics(nucleus_mask, image)
            
            # Combine features
            combined_features = np.concatenate([
                vis_features, morph_features, nuc_features
            ])
            
            nuclei_features.append(combined_features)
            
        return np.array(nuclei_features), np.array(nuclei_centroids)
    
    def _extract_morphological_features(self, mask):
        """Extract 7-dimensional morphological features."""
        import cv2
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return np.zeros(7)
            
        contour = contours[0]
        
        # Area
        area = cv2.contourArea(contour)
        
        # Perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Eccentricity from fitted ellipse
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
        else:
            eccentricity = 0
            
        # Extent (area / bounding box area)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0
        
        # Orientation
        if len(contour) >= 5:
            _, _, angle = cv2.fitEllipse(contour)
            orientation = angle
        else:
            orientation = 0
            
        return np.array([area, perimeter, circularity, solidity, 
                        eccentricity, extent, orientation])
    
    def _extract_visual_features(self, image, mask):
        """Extract visual features using DINOv2."""
        import torchvision.transforms as T
        from transformers import AutoImageProcessor, AutoModel
        
        # Load pre-trained DINOv2
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        
        # Extract patch containing the nucleus
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return np.zeros(768)
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add padding
        pad = 10
        y_min = max(0, y_min - pad)
        y_max = min(image.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(image.shape[1], x_max + pad)
        
        nucleus_patch = image[y_min:y_max, x_min:x_max]
        
        if nucleus_patch.size == 0:
            return np.zeros(768)
        
        # Preprocess and extract features
        inputs = processor(images=nucleus_patch, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
        return features
    
    def _extract_nuclear_morphometrics(self, mask, image):
        """Extract 12-dimensional fine-grained nuclear morphometric features."""
        import cv2
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return np.zeros(12)
            
        contour = contours[0]
        
        # Radial distance distribution (8 distances)
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return np.zeros(12)
            
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        
        # Sample 8 radial distances
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        radial_distances = []
        
        for angle in angles:
            # Find intersection with contour in this direction
            max_dist = max(mask.shape)
            x_end = cx + max_dist * np.cos(angle)
            y_end = cy + max_dist * np.sin(angle)
            
            # Simple ray casting (simplified)
            dist = self._ray_cast_distance(mask, (cx, cy), (x_end, y_end))
            radial_distances.append(dist)
        
        radial_distances = np.array(radial_distances)
        if np.max(radial_distances) > 0:
            radial_distances = radial_distances / np.max(radial_distances)
        
        # Major-minor axis ratio
        if len(contour) >= 5:
            (x, y), (major, minor), angle = cv2.fitEllipse(contour)
            axis_ratio = minor / major if major > 0 else 0
        else:
            axis_ratio = 0
            
        # Area discrepancy (polygon area vs convex hull area)
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)
        area_discrepancy = (hull_area - area) / hull_area if hull_area > 0 else 0
        
        # Intensity homogeneity (from image)
        nucleus_pixels = image[mask > 0]
        intensity_variance = np.var(nucleus_pixels) if len(nucleus_pixels) > 0 else 0
        
        # Nuclear hyperchromasia
        max_intensity = np.max(nucleus_pixels) if len(nucleus_pixels) > 0 else 0
        
        return np.concatenate([
            radial_distances,
            [axis_ratio, area_discrepancy, intensity_variance, max_intensity]
        ])
    
    def _ray_cast_distance(self, mask, start, end):
        """Calculate distance from centroid to boundary in given direction."""
        # Simplified ray casting implementation
        x0, y0 = start
        x1, y1 = end
        
        distance = 0
        steps = 100
        for i in range(steps):
            t = i / steps
            x = int(x0 + t * (x1 - x0))
            y = int(y0 + t * (y1 - y0))
            
            if (x < 0 or x >= mask.shape[1] or 
                y < 0 or y >= mask.shape[0] or 
                not mask[y, x]):
                break
            distance = t * np.sqrt((x1-x0)**2 + (y1-y0)**2)
            
        return distance
    
    def build_hierarchical_graph(self, patch_path, mask_path):
        """Build complete hierarchical graph from WSI patch."""
        # Load data
        image, mask = self.load_patch_data(patch_path, mask_path)
        
        # Extract nuclear features
        nuclei_features, nuclei_centroids = self.extract_nuclear_features(mask, image)
        
        if len(nuclei_features) == 0:
            return None
            
        # Build cell graph
        cell_graph = build_cell_graph(nuclei_features, nuclei_centroids, self.config)
        
        # Build tissue graph
        tissue_graph = build_tissue_graph(cell_graph, nuclei_centroids, self.config)
        
        # Apply transforms
        if self.transform:
            cell_graph, tissue_graph = self.transform(cell_graph, tissue_graph)
            
        return cell_graph, tissue_graph
