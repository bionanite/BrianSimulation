#!/usr/bin/env python3
"""
Unsupervised Learning Mechanisms
Implements Hebbian learning, competitive learning, and predictive coding
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

@dataclass
class FeatureDetector:
    """Represents a feature detector neuron"""
    detector_id: int
    weights: np.ndarray
    activation: float = 0.0
    win_count: int = 0
    last_update_time: float = 0.0

class HebbianLearning:
    """
    Hebbian Learning: "Neurons that fire together, wire together"
    
    Updates weights based on correlation between pre and post-synaptic activity
    """
    
    def __init__(self,
                 learning_rate: float = 0.01,
                 normalization: str = 'Oja'):  # 'Oja', 'subtractive', 'none'
        self.learning_rate = learning_rate
        self.normalization = normalization
    
    def update_weights(self,
                      weights: np.ndarray,
                      pre_activity: np.ndarray,
                      post_activity: float) -> np.ndarray:
        """
        Update weights using Hebbian rule
        
        Standard: Δw = η * pre_activity * post_activity
        Oja: Δw = η * post_activity * (pre_activity - post_activity * w)
        """
        if self.normalization == 'Oja':
            # Oja's rule: prevents unbounded growth
            delta_w = self.learning_rate * post_activity * (
                pre_activity - post_activity * weights
            )
        elif self.normalization == 'subtractive':
            # Subtractive normalization
            delta_w = self.learning_rate * post_activity * pre_activity
            # Normalize: subtract mean
            delta_w = delta_w - np.mean(delta_w)
        else:
            # Standard Hebbian (no normalization)
            delta_w = self.learning_rate * post_activity * pre_activity
        
        new_weights = weights + delta_w
        
        # Ensure non-negative weights (optional)
        new_weights = np.maximum(new_weights, 0.0)
        
        return new_weights
    
    def calculate_correlation(self,
                             pre_activity: np.ndarray,
                             post_activity: float) -> float:
        """Calculate correlation between pre and post activity"""
        if len(pre_activity) == 0:
            return 0.0
        
        # Simple correlation measure
        correlation = np.mean(pre_activity) * post_activity
        return correlation

class CompetitiveLearning:
    """
    Competitive Learning: Winner-Take-All
    
    Feature detectors compete, winner updates weights
    """
    
    def __init__(self,
                 learning_rate: float = 0.1,
                 learning_rate_loser: float = 0.01,
                 neighborhood_size: int = 1):
        self.learning_rate = learning_rate
        self.learning_rate_loser = learning_rate_loser
        self.neighborhood_size = neighborhood_size
    
    def select_winner(self,
                     detectors: List[FeatureDetector],
                     input_pattern: np.ndarray) -> Tuple[int, float]:
        """
        Select winner detector based on response strength
        
        Returns:
            (winner_id, response_strength)
        """
        best_id = None
        best_response = float('-inf')
        
        for detector in detectors:
            # Calculate response (dot product)
            response = np.dot(detector.weights, input_pattern)
            detector.activation = response
            
            if response > best_response:
                best_response = response
                best_id = detector.detector_id
        
        return best_id, best_response
    
    def update_winner(self,
                     detector: FeatureDetector,
                     input_pattern: np.ndarray):
        """Update winner detector weights"""
        # Move weights toward input pattern
        delta_w = self.learning_rate * (input_pattern - detector.weights)
        detector.weights += delta_w
        
        # Normalize weights
        norm = np.linalg.norm(detector.weights)
        if norm > 0:
            detector.weights = detector.weights / norm
        
        detector.win_count += 1
        detector.last_update_time = time.time()
    
    def update_losers(self,
                     detectors: List[FeatureDetector],
                     winner_id: int,
                     input_pattern: np.ndarray):
        """Update loser detectors (weaker learning)"""
        for detector in detectors:
            if detector.detector_id != winner_id:
                # Weaker update for losers
                delta_w = self.learning_rate_loser * (input_pattern - detector.weights)
                detector.weights += delta_w
                
                # Normalize
                norm = np.linalg.norm(detector.weights)
                if norm > 0:
                    detector.weights = detector.weights / norm
    
    def create_feature_detectors(self,
                                num_detectors: int,
                                input_size: int,
                                initialization: str = 'random') -> List[FeatureDetector]:
        """Create feature detectors"""
        detectors = []
        
        for i in range(num_detectors):
            if initialization == 'random':
                weights = np.random.random(input_size)
            elif initialization == 'uniform':
                weights = np.ones(input_size) / input_size
            else:
                weights = np.random.normal(0, 0.1, input_size)
            
            # Normalize
            norm = np.linalg.norm(weights)
            if norm > 0:
                weights = weights / norm
            
            detector = FeatureDetector(
                detector_id=i,
                weights=weights
            )
            detectors.append(detector)
        
        return detectors

class PredictiveCoding:
    """
    Predictive Coding: Predict next input, minimize prediction error
    
    Learns hierarchical representations by predicting and minimizing errors
    """
    
    def __init__(self,
                 learning_rate: float = 0.01,
                 prediction_error_weight: float = 1.0):
        self.learning_rate = learning_rate
        self.prediction_error_weight = prediction_error_weight
        
        self.predictions: Dict[int, np.ndarray] = {}  # layer_id -> prediction
        self.errors: Dict[int, np.ndarray] = {}  # layer_id -> error
    
    def predict(self,
               current_input: np.ndarray,
               layer_id: int,
               prediction_weights: np.ndarray) -> np.ndarray:
        """
        Predict next input based on current input
        
        prediction = weights @ current_input
        """
        if prediction_weights.shape[1] != len(current_input):
            # Resize if needed
            if prediction_weights.shape[1] > len(current_input):
                current_input = np.pad(current_input, (0, prediction_weights.shape[1] - len(current_input)))
            else:
                current_input = current_input[:prediction_weights.shape[1]]
        
        prediction = prediction_weights @ current_input
        
        # Store prediction
        self.predictions[layer_id] = prediction
        
        return prediction
    
    def calculate_error(self,
                       actual_input: np.ndarray,
                       predicted_input: np.ndarray,
                       layer_id: int) -> np.ndarray:
        """
        Calculate prediction error
        
        error = actual - predicted
        """
        # Ensure same size
        min_len = min(len(actual_input), len(predicted_input))
        error = actual_input[:min_len] - predicted_input[:min_len]
        
        # Store error
        self.errors[layer_id] = error
        
        return error
    
    def update_weights(self,
                      weights: np.ndarray,
                      error: np.ndarray,
                      input_pattern: np.ndarray) -> np.ndarray:
        """
        Update prediction weights to reduce error
        
        Δw = learning_rate * error * input
        """
        # Ensure dimensions match
        if weights.shape[0] != len(error):
            error = error[:weights.shape[0]]
        if weights.shape[1] != len(input_pattern):
            input_pattern = input_pattern[:weights.shape[1]]
        
        # Update weights
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                delta_w = self.learning_rate * error[i] * input_pattern[j]
                weights[i, j] += delta_w
        
        return weights
    
    def get_prediction_error_magnitude(self, layer_id: int) -> float:
        """Get magnitude of prediction error"""
        if layer_id not in self.errors:
            return 0.0
        return np.linalg.norm(self.errors[layer_id])

class SelfOrganizingMap:
    """
    Self-Organizing Map (SOM)
    
    Creates topographic map of input space
    """
    
    def __init__(self,
                 map_size: Tuple[int, int],
                 input_size: int,
                 learning_rate: float = 0.1,
                 neighborhood_radius: float = 2.0):
        self.map_size = map_size
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.initial_radius = neighborhood_radius
        self.current_radius = neighborhood_radius
        
        # Create map
        self.map = np.random.random((map_size[0], map_size[1], input_size))
        self.activation_map = np.zeros(map_size)
    
    def find_best_matching_unit(self, input_pattern: np.ndarray) -> Tuple[int, int]:
        """Find BMU (Best Matching Unit)"""
        best_distance = float('inf')
        bmu = (0, 0)
        
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                distance = np.linalg.norm(self.map[i, j] - input_pattern)
                if distance < best_distance:
                    best_distance = distance
                    bmu = (i, j)
        
        return bmu
    
    def update_map(self,
                  input_pattern: np.ndarray,
                  iteration: int,
                  max_iterations: int = 1000):
        """Update SOM map"""
        # Find BMU
        bmu_i, bmu_j = self.find_best_matching_unit(input_pattern)
        
        # Update learning rate and radius (decay over time)
        current_lr = self.learning_rate * (1 - iteration / max_iterations)
        self.current_radius = self.initial_radius * (1 - iteration / max_iterations)
        
        # Update weights for BMU and neighbors
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                # Calculate distance from BMU
                distance = np.sqrt((i - bmu_i)**2 + (j - bmu_j)**2)
                
                if distance <= self.current_radius:
                    # Calculate neighborhood function
                    neighborhood = np.exp(-distance**2 / (2 * self.current_radius**2))
                    
                    # Update weight
                    delta_w = current_lr * neighborhood * (input_pattern - self.map[i, j])
                    self.map[i, j] += delta_w
                    
                    # Normalize
                    norm = np.linalg.norm(self.map[i, j])
                    if norm > 0:
                        self.map[i, j] = self.map[i, j] / norm
        
        # Update activation
        self.activation_map[bmu_i, bmu_j] += 1.0
        self.activation_map *= 0.9  # Decay

class UnsupervisedLearningManager:
    """
    Manages all unsupervised learning mechanisms
    """
    
    def __init__(self,
                 enable_hebbian: bool = True,
                 enable_competitive: bool = True,
                 enable_predictive: bool = True,
                 enable_som: bool = False):
        self.enable_hebbian = enable_hebbian
        self.enable_competitive = enable_competitive
        self.enable_predictive = enable_predictive
        self.enable_som = enable_som
        
        self.hebbian = HebbianLearning() if enable_hebbian else None
        self.competitive = CompetitiveLearning() if enable_competitive else None
        self.predictive = PredictiveCoding() if enable_predictive else None
        self.som = None  # Created on demand
        
        # Feature detectors for competitive learning
        self.feature_detectors: List[FeatureDetector] = []
        
        # Learning history
        self.learning_history = []
    
    def initialize_feature_detectors(self,
                                    num_detectors: int,
                                    input_size: int):
        """Initialize feature detectors for competitive learning"""
        if self.competitive:
            self.feature_detectors = self.competitive.create_feature_detectors(
                num_detectors, input_size
            )
    
    def learn_from_pattern(self,
                          input_pattern: np.ndarray,
                          current_time: float = 0.0) -> Dict:
        """
        Learn from an input pattern using all enabled mechanisms
        
        Returns:
            Dictionary with learning updates
        """
        updates = {
            'hebbian_updates': 0,
            'competitive_winner': None,
            'prediction_error': 0.0,
            'som_updates': 0
        }
        
        # Hebbian learning
        if self.enable_hebbian and self.hebbian and self.feature_detectors:
            # Use first detector as example
            if self.feature_detectors:
                detector = self.feature_detectors[0]
                post_activity = np.dot(detector.weights, input_pattern)
                detector.weights = self.hebbian.update_weights(
                    detector.weights, input_pattern, post_activity
                )
                updates['hebbian_updates'] = 1
        
        # Competitive learning
        if self.enable_competitive and self.competitive and self.feature_detectors:
            winner_id, response = self.competitive.select_winner(
                self.feature_detectors, input_pattern
            )
            winner = next(d for d in self.feature_detectors if d.detector_id == winner_id)
            self.competitive.update_winner(winner, input_pattern)
            self.competitive.update_losers(self.feature_detectors, winner_id, input_pattern)
            updates['competitive_winner'] = winner_id
        
        # Predictive coding
        if self.enable_predictive and self.predictive:
            layer_id = 0
            # Simple prediction weights
            if not hasattr(self, 'prediction_weights'):
                self.prediction_weights = np.random.random((len(input_pattern), len(input_pattern))) * 0.1
            
            prediction = self.predictive.predict(input_pattern, layer_id, self.prediction_weights)
            error = self.predictive.calculate_error(input_pattern, prediction, layer_id)
            self.prediction_weights = self.predictive.update_weights(
                self.prediction_weights, error, input_pattern
            )
            updates['prediction_error'] = np.linalg.norm(error)
        
        # SOM (if enabled)
        if self.enable_som and self.som:
            if not hasattr(self, 'som_iteration'):
                self.som_iteration = 0
            self.som.update_map(input_pattern, self.som_iteration)
            self.som_iteration += 1
            updates['som_updates'] = 1
        
        return updates
    
    def get_statistics(self) -> Dict:
        """Get statistics about unsupervised learning"""
        stats = {}
        
        if self.feature_detectors:
            win_counts = [d.win_count for d in self.feature_detectors]
            stats['competitive'] = {
                'num_detectors': len(self.feature_detectors),
                'total_wins': sum(win_counts),
                'win_distribution': win_counts,
                'most_active': max(self.feature_detectors, key=lambda d: d.win_count).detector_id if win_counts else None
            }
        
        if self.predictive:
            errors = list(self.predictive.errors.values())
            if errors:
                stats['predictive'] = {
                    'num_layers': len(self.predictive.errors),
                    'avg_error': np.mean([np.linalg.norm(e) for e in errors]),
                    'total_error': sum(np.linalg.norm(e) for e in errors)
                }
        
        return stats

