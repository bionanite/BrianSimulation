#!/usr/bin/env python3
"""
Hierarchical Feature Learning
Implements multi-layer hierarchical feature detection and abstraction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

@dataclass
class FeatureLayer:
    """Represents a layer in the hierarchical feature network"""
    layer_id: int
    layer_size: int
    input_size: int
    feature_detectors: np.ndarray  # (layer_size, input_size) weights
    activations: np.ndarray  # (layer_size,) current activations
    biases: np.ndarray  # (layer_size,) biases
    layer_type: str = 'feedforward'  # 'feedforward', 'feedback', 'lateral'
    
@dataclass
class HierarchicalNetwork:
    """Represents a hierarchical feature learning network"""
    layers: List[FeatureLayer]
    layer_connections: Dict[Tuple[int, int], np.ndarray]  # (from_layer, to_layer) -> weights
    learning_history: List[Dict] = field(default_factory=list)

class HierarchicalFeatureLearning:
    """
    Hierarchical Feature Learning
    
    Learns features at multiple levels of abstraction
    Lower layers: simple features (edges, corners)
    Higher layers: complex patterns (objects, concepts)
    """
    
    def __init__(self,
                 layer_sizes: List[int],
                 input_size: int,
                 learning_rate: float = 0.01,
                 sparsity_target: float = 0.1):
        self.layer_sizes = layer_sizes
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.sparsity_target = sparsity_target
        
        # Create hierarchical network
        self.network = self._create_network()
    
    def _create_network(self) -> HierarchicalNetwork:
        """Create hierarchical network structure"""
        layers = []
        layer_connections = {}
        
        # Create layers
        prev_size = self.input_size
        for i, layer_size in enumerate(self.layer_sizes):
            layer = FeatureLayer(
                layer_id=i,
                layer_size=layer_size,
                input_size=prev_size,
                feature_detectors=np.random.normal(0, 0.1, (layer_size, prev_size)),
                activations=np.zeros(layer_size),
                biases=np.zeros(layer_size)
            )
            
            # Normalize detectors
            for j in range(layer_size):
                norm = np.linalg.norm(layer.feature_detectors[j])
                if norm > 0:
                    layer.feature_detectors[j] /= norm
            
            layers.append(layer)
            
            # Create connections between layers
            if i > 0:
                # Feedforward connection
                layer_connections[(i-1, i)] = np.random.normal(0, 0.1, (layer_size, prev_size))
                # Feedback connection (weaker)
                layer_connections[(i, i-1)] = np.random.normal(0, 0.05, (prev_size, layer_size))
            
            prev_size = layer_size
        
        return HierarchicalNetwork(layers=layers, layer_connections=layer_connections)
    
    def forward_pass(self, input_pattern: np.ndarray) -> List[np.ndarray]:
        """
        Forward pass through hierarchical network
        
        Returns:
            List of activations at each layer
        """
        activations = []
        current_input = input_pattern.copy()
        
        for layer in self.network.layers:
            # Ensure input size matches
            if len(current_input) != layer.input_size:
                # Pad or truncate to match
                if len(current_input) < layer.input_size:
                    current_input = np.pad(current_input, (0, layer.input_size - len(current_input)))
                else:
                    current_input = current_input[:layer.input_size]
            
            # Compute layer activations
            layer.activations = np.dot(layer.feature_detectors, current_input) + layer.biases
            
            # Apply activation function (ReLU with sparsity)
            layer.activations = np.maximum(0, layer.activations)
            
            # Sparsity constraint
            if self.sparsity_target > 0:
                # Keep only top k% active
                k = int(self.sparsity_target * layer.layer_size)
                if k > 0:
                    threshold = np.partition(layer.activations, -k)[-k]
                    layer.activations = np.where(layer.activations >= threshold, 
                                               layer.activations, 0)
            
            activations.append(layer.activations.copy())
            current_input = layer.activations.copy()
        
        return activations
    
    def backward_pass(self,
                     input_pattern: np.ndarray,
                     target_pattern: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Backward pass for learning (predictive coding)
        
        Computes prediction errors at each layer
        """
        errors = []
        
        # Forward pass first
        activations = self.forward_pass(input_pattern)
        
        # Compute errors top-down
        if target_pattern is not None:
            # Top layer error
            if len(target_pattern) == len(activations[-1]):
                top_error = target_pattern - activations[-1]
            else:
                top_error = np.zeros(len(activations[-1]))
            errors.append(top_error)
        else:
            # Use input as target for bottom layer
            if len(input_pattern) == len(activations[0]):
                errors.append(input_pattern - activations[0])
            else:
                errors.append(np.zeros(len(activations[0])))
        
        # Propagate errors down
        for i in range(len(self.network.layers) - 1, 0, -1):
            layer = self.network.layers[i]
            prev_layer = self.network.layers[i-1]
            
            # Predict lower layer from current layer
            if (i, i-1) in self.network.layer_connections:
                feedback_weights = self.network.layer_connections[(i, i-1)]
                prediction = np.dot(feedback_weights, layer.activations)
                # Ensure sizes match
                if len(prediction) == len(prev_layer.activations):
                    error = prev_layer.activations - prediction
                else:
                    error = np.zeros(len(prev_layer.activations))
                errors.insert(0, error)
        
        return errors
    
    def learn_from_pattern(self, input_pattern: np.ndarray):
        """Learn hierarchical features from input pattern"""
        # Forward pass
        activations = self.forward_pass(input_pattern)
        
        # Compute errors (predictive coding)
        errors = self.backward_pass(input_pattern)
        
        # Update weights at each layer
        for i, layer in enumerate(self.network.layers):
            error = errors[i] if i < len(errors) else np.zeros(layer.layer_size)
            
            # Ensure error matches layer size
            if len(error) != layer.layer_size:
                error = np.zeros(layer.layer_size)
            
            # Get input for this layer
            if i == 0:
                layer_input = input_pattern.copy()
            else:
                layer_input = activations[i-1].copy()
            
            # Ensure input size matches
            if len(layer_input) != layer.input_size:
                # Pad or truncate to match
                if len(layer_input) < layer.input_size:
                    layer_input = np.pad(layer_input, (0, layer.input_size - len(layer_input)))
                else:
                    layer_input = layer_input[:layer.input_size]
            
            # Update feature detectors
            for j in range(layer.layer_size):
                # Hebbian-like update
                delta_w = self.learning_rate * layer.activations[j] * layer_input
                layer.feature_detectors[j] += delta_w
                
                # Normalize to prevent unbounded growth
                norm = np.linalg.norm(layer.feature_detectors[j])
                if norm > 0:
                    layer.feature_detectors[j] /= norm
            
            # Update biases (sparsity)
            sparsity_error = np.mean(layer.activations) - self.sparsity_target
            layer.biases -= self.learning_rate * sparsity_error
            
            # Update layer connections
            if i > 0 and (i-1, i) in self.network.layer_connections:
                # Feedforward connection update
                prev_activations = activations[i-1]
                delta_conn = self.learning_rate * np.outer(layer.activations, prev_activations)
                self.network.layer_connections[(i-1, i)] += delta_conn
            
            if i < len(self.network.layers) - 1 and (i, i+1) in self.network.layer_connections:
                # Feedback connection update
                next_activations = activations[i+1] if i+1 < len(activations) else layer.activations
                delta_conn = self.learning_rate * 0.1 * np.outer(next_activations, layer.activations)
                self.network.layer_connections[(i, i+1)] += delta_conn
    
    def get_layer_features(self, layer_id: int) -> np.ndarray:
        """Get learned features at a specific layer"""
        if 0 <= layer_id < len(self.network.layers):
            return self.network.layers[layer_id].feature_detectors.copy()
        return None
    
    def get_abstraction_level(self, layer_id: int) -> float:
        """Get abstraction level of a layer (0 = low, 1 = high)"""
        if layer_id < 0 or layer_id >= len(self.network.layers):
            return 0.0
        return layer_id / max(1, len(self.network.layers) - 1)

class MultiScaleFeatureLearning:
    """
    Multi-Scale Feature Learning
    
    Learns features at different scales simultaneously
    """
    
    def __init__(self,
                 scales: List[int],
                 base_feature_size: int,
                 learning_rate: float = 0.01):
        self.scales = scales  # e.g., [1, 2, 4] for different receptive field sizes
        self.base_feature_size = base_feature_size
        self.learning_rate = learning_rate
        
        self.scale_detectors: Dict[int, np.ndarray] = {}
        
        # Initialize detectors for each scale
        for scale in scales:
            detector_size = base_feature_size * scale
            self.scale_detectors[scale] = np.random.normal(0, 0.1, (base_feature_size, detector_size))
    
    def extract_features(self, input_pattern: np.ndarray) -> Dict[int, np.ndarray]:
        """Extract features at different scales"""
        features = {}
        
        for scale in self.scales:
            # Resize input to scale
            if scale > 1:
                # Downsample
                step = len(input_pattern) // (self.base_feature_size * scale)
                scaled_input = input_pattern[::max(1, step)]
            else:
                scaled_input = input_pattern[:self.base_feature_size * scale]
            
            # Pad if needed
            target_size = self.base_feature_size * scale
            if len(scaled_input) < target_size:
                scaled_input = np.pad(scaled_input, (0, target_size - len(scaled_input)))
            elif len(scaled_input) > target_size:
                scaled_input = scaled_input[:target_size]
            
            # Extract features
            features[scale] = np.dot(self.scale_detectors[scale], scaled_input)
        
        return features
    
    def learn_from_pattern(self, input_pattern: np.ndarray):
        """Learn features at multiple scales"""
        features = self.extract_features(input_pattern)
        
        for scale in self.scales:
            # Get scaled input
            if scale > 1:
                step = len(input_pattern) // (self.base_feature_size * scale)
                scaled_input = input_pattern[::max(1, step)]
            else:
                scaled_input = input_pattern[:self.base_feature_size * scale]
            
            target_size = self.base_feature_size * scale
            if len(scaled_input) < target_size:
                scaled_input = np.pad(scaled_input, (0, target_size - len(scaled_input)))
            elif len(scaled_input) > target_size:
                scaled_input = scaled_input[:target_size]
            
            # Update detectors
            for i in range(self.base_feature_size):
                delta_w = self.learning_rate * features[scale][i] * scaled_input
                self.scale_detectors[scale][i] += delta_w
                
                # Normalize
                norm = np.linalg.norm(self.scale_detectors[scale][i])
                if norm > 0:
                    self.scale_detectors[scale][i] /= norm

class HierarchicalLearningManager:
    """
    Manages hierarchical feature learning
    """
    
    def __init__(self,
                 enable_hierarchical: bool = True,
                 enable_multiscale: bool = True,
                 layer_sizes: List[int] = [50, 30, 10],
                 input_size: int = 100):
        self.enable_hierarchical = enable_hierarchical
        self.enable_multiscale = enable_multiscale
        
        self.hierarchical = HierarchicalFeatureLearning(
            layer_sizes=layer_sizes,
            input_size=input_size
        ) if enable_hierarchical else None
        
        self.multiscale = MultiScaleFeatureLearning(
            scales=[1, 2, 4],
            base_feature_size=20
        ) if enable_multiscale else None
        
        self.learning_history = []
    
    def learn_from_pattern(self, input_pattern: np.ndarray):
        """Learn from input pattern using all enabled mechanisms"""
        if self.enable_hierarchical and self.hierarchical:
            self.hierarchical.learn_from_pattern(input_pattern)
        
        if self.enable_multiscale and self.multiscale:
            self.multiscale.learn_from_pattern(input_pattern)
    
    def get_hierarchical_features(self, layer_id: int) -> Optional[np.ndarray]:
        """Get features from hierarchical network"""
        if self.enable_hierarchical and self.hierarchical:
            return self.hierarchical.get_layer_features(layer_id)
        return None
    
    def get_multiscale_features(self, input_pattern: np.ndarray) -> Dict[int, np.ndarray]:
        """Get features at multiple scales"""
        if self.enable_multiscale and self.multiscale:
            return self.multiscale.extract_features(input_pattern)
        return {}
    
    def get_statistics(self) -> Dict:
        """Get statistics about hierarchical learning"""
        stats = {}
        
        if self.enable_hierarchical and self.hierarchical:
            network = self.hierarchical.network
            stats['hierarchical'] = {
                'num_layers': len(network.layers),
                'layer_sizes': [layer.layer_size for layer in network.layers],
                'total_connections': sum(len(conn) for conn in network.layer_connections.values())
            }
        
        if self.enable_multiscale and self.multiscale:
            stats['multiscale'] = {
                'scales': self.multiscale.scales,
                'num_detectors_per_scale': self.multiscale.base_feature_size
            }
        
        return stats

