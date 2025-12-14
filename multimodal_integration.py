#!/usr/bin/env python3
"""
Multi-Modal Integration
Implements cross-modal learning, sensory fusion, unified representations, and attention
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
from collections import defaultdict

@dataclass
class Modality:
    """Represents a sensory modality"""
    modality_id: int
    name: str
    feature_size: int
    features: np.ndarray
    attention_weight: float = 1.0
    reliability: float = 1.0

@dataclass
class CrossModalMapping:
    """Represents a cross-modal mapping"""
    from_modality_id: int
    to_modality_id: int
    mapping_weights: np.ndarray  # (to_size, from_size)
    strength: float = 1.0
    confidence: float = 1.0

@dataclass
class UnifiedRepresentation:
    """Represents a unified multi-modal representation"""
    representation_id: int
    unified_features: np.ndarray
    source_modalities: List[int]
    fusion_weights: Dict[int, float]  # modality_id -> weight
    created_time: float = 0.0

class CrossModalLearning:
    """
    Cross-Modal Learning
    
    Learns mappings between different modalities
    Enables translation between senses
    """
    
    def __init__(self,
                 learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        
        self.mappings: Dict[Tuple[int, int], CrossModalMapping] = {}
    
    def learn_mapping(self,
                     from_modality_id: int,
                     to_modality_id: int,
                     from_features: np.ndarray,
                     to_features: np.ndarray):
        """Learn mapping between modalities"""
        key = (from_modality_id, to_modality_id)
        
        if key not in self.mappings:
            # Create new mapping
            mapping = CrossModalMapping(
                from_modality_id=from_modality_id,
                to_modality_id=to_modality_id,
                mapping_weights=np.random.normal(0, 0.1, (len(to_features), len(from_features)))
            )
            self.mappings[key] = mapping
        
        mapping = self.mappings[key]
        
        # Update mapping weights
        error = to_features - np.dot(mapping.mapping_weights, from_features)
        delta_w = self.learning_rate * np.outer(error, from_features)
        mapping.mapping_weights += delta_w
        
        # Update confidence
        prediction_error = np.linalg.norm(error)
        mapping.confidence = 1.0 / (1.0 + prediction_error)
        mapping.strength = min(1.0, mapping.strength + 0.01)
    
    def translate(self,
                  from_modality_id: int,
                  to_modality_id: int,
                  from_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Translate features from one modality to another
        
        Returns:
            (translated_features, confidence)
        """
        key = (from_modality_id, to_modality_id)
        
        if key not in self.mappings:
            # No mapping available
            return np.zeros(len(from_features)), 0.0
        
        mapping = self.mappings[key]
        translated = np.dot(mapping.mapping_weights, from_features)
        confidence = mapping.confidence
        
        return translated, confidence

class SensoryFusion:
    """
    Sensory Fusion
    
    Combines information from multiple modalities
    Creates unified representations
    """
    
    def __init__(self,
                 fusion_method: str = 'weighted_average'):  # 'weighted_average', 'max', 'concat'
        self.fusion_method = fusion_method
    
    def fuse_modalities(self,
                      modalities: List[Modality],
                      fusion_weights: Optional[Dict[int, float]] = None) -> UnifiedRepresentation:
        """
        Fuse multiple modalities into unified representation
        
        Returns:
            Unified representation
        """
        if not modalities:
            return None
        
        # Default weights: equal or based on reliability
        if fusion_weights is None:
            fusion_weights = {}
            total_reliability = sum(m.reliability for m in modalities)
            for m in modalities:
                fusion_weights[m.modality_id] = m.reliability / total_reliability if total_reliability > 0 else 1.0 / len(modalities)
        
        # Normalize weights
        total_weight = sum(fusion_weights.values())
        fusion_weights = {k: v / total_weight for k, v in fusion_weights.items()}
        
        # Fuse based on method
        if self.fusion_method == 'weighted_average':
            # Weighted average of features (handle different sizes)
            max_size = max(m.feature_size for m in modalities)
            unified = np.zeros(max_size)
            total_weight = 0.0
            
            for m in modalities:
                weight = fusion_weights.get(m.modality_id, 0.0)
                # Pad or truncate to max_size
                features = m.features.copy()
                if len(features) < max_size:
                    features = np.pad(features, (0, max_size - len(features)))
                elif len(features) > max_size:
                    features = features[:max_size]
                
                unified += weight * features
                total_weight += weight
            
            if total_weight > 0:
                unified /= total_weight
        elif self.fusion_method == 'max':
            # Element-wise maximum (handle different sizes)
            max_size = max(m.feature_size for m in modalities)
            unified = np.zeros(max_size)
            
            for m in modalities:
                features = m.features.copy()
                if len(features) < max_size:
                    features = np.pad(features, (0, max_size - len(features)))
                elif len(features) > max_size:
                    features = features[:max_size]
                
                unified = np.maximum(unified, features)
        elif self.fusion_method == 'concat':
            # Concatenation
            unified = np.concatenate([m.features for m in modalities])
        else:
            unified = modalities[0].features.copy()
        
        representation = UnifiedRepresentation(
            representation_id=hash(tuple(unified)) % 1000000,
            unified_features=unified,
            source_modalities=[m.modality_id for m in modalities],
            fusion_weights=fusion_weights,
            created_time=time.time()
        )
        
        return representation
    
    def compute_fusion_confidence(self,
                                 modalities: List[Modality],
                                 fusion_weights: Dict[int, float]) -> float:
        """Compute confidence of fusion"""
        if not modalities:
            return 0.0
        
        # Weighted average of reliabilities
        confidence = sum(
            m.reliability * fusion_weights.get(m.modality_id, 0.0)
            for m in modalities
        )
        
        return confidence

class MultiModalAttention:
    """
    Multi-Modal Attention
    
    Selects and weights modalities based on relevance
    """
    
    def __init__(self,
                 attention_temperature: float = 1.0):
        self.attention_temperature = attention_temperature
        self.attention_history: List[Dict[int, float]] = []
    
    def compute_attention_weights(self,
                                 modalities: List[Modality],
                                 context: Optional[np.ndarray] = None) -> Dict[int, float]:
        """
        Compute attention weights for modalities
        
        Returns:
            Dictionary mapping modality_id -> attention_weight
        """
        if not modalities:
            return {}
        
        # Compute relevance scores
        scores = {}
        for m in modalities:
            # Base score on reliability and current attention
            score = m.reliability * m.attention_weight
            
            # Context-based adjustment (if context provided)
            if context is not None and len(context) == len(m.features):
                context_similarity = np.dot(context, m.features) / (
                    np.linalg.norm(context) * np.linalg.norm(m.features) + 1e-10
                )
                score *= (1.0 + context_similarity)
            
            scores[m.modality_id] = score
        
        # Softmax normalization
        max_score = max(scores.values()) if scores else 0.0
        exp_scores = {k: np.exp((v - max_score) / self.attention_temperature) 
                     for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        
        attention_weights = {k: v / total_exp if total_exp > 0 else 1.0 / len(scores)
                           for k, v in exp_scores.items()}
        
        self.attention_history.append(attention_weights.copy())
        
        return attention_weights
    
    def update_modality_attention(self,
                                 modality: Modality,
                                 relevance: float):
        """Update attention weight for a modality"""
        # Decay and update
        modality.attention_weight = 0.9 * modality.attention_weight + 0.1 * relevance

class UnifiedRepresentationLearning:
    """
    Unified Representation Learning
    
    Learns unified representations across modalities
    """
    
    def __init__(self,
                 unified_size: int = 50,
                 learning_rate: float = 0.01):
        self.unified_size = unified_size
        self.learning_rate = learning_rate
        
        self.unified_space: np.ndarray = np.random.normal(0, 0.1, (unified_size, unified_size))
        self.modality_projections: Dict[int, np.ndarray] = {}  # modality_id -> projection matrix
    
    def project_to_unified_space(self,
                                modality: Modality) -> np.ndarray:
        """Project modality features to unified space"""
        if modality.modality_id not in self.modality_projections:
            # Initialize projection
            self.modality_projections[modality.modality_id] = np.random.normal(
                0, 0.1, (self.unified_size, modality.feature_size)
            )
        
        projection = self.modality_projections[modality.modality_id]
        unified = np.dot(projection, modality.features)
        
        return unified
    
    def learn_unified_representation(self,
                                     modalities: List[Modality],
                                     target_unified: Optional[np.ndarray] = None):
        """Learn unified representation from modalities"""
        if not modalities:
            return None
        
        # Project each modality to unified space
        unified_representations = []
        for m in modalities:
            unified = self.project_to_unified_space(m)
            unified_representations.append(unified)
        
        # Average or use target
        if target_unified is not None:
            target = target_unified
        else:
            target = np.mean(unified_representations, axis=0)
        
        # Update projections to minimize reconstruction error
        for i, m in enumerate(modalities):
            projection = self.modality_projections[m.modality_id]
            error = target - unified_representations[i]
            delta_p = self.learning_rate * np.outer(error, m.features)
            projection += delta_p
        
        return target

class MultiModalIntegrationManager:
    """
    Manages all multi-modal integration mechanisms
    """
    
    def __init__(self):
        self.modalities: Dict[int, Modality] = {}
        self.next_modality_id = 0
        
        self.cross_modal_learning = CrossModalLearning()
        self.sensory_fusion = SensoryFusion()
        self.attention = MultiModalAttention()
        self.unified_learning = UnifiedRepresentationLearning()
        
        self.unified_representations: List[UnifiedRepresentation] = []
    
    def register_modality(self,
                         name: str,
                         feature_size: int,
                         initial_features: Optional[np.ndarray] = None) -> Modality:
        """Register a new modality"""
        if initial_features is None:
            initial_features = np.random.random(feature_size)
        
        modality = Modality(
            modality_id=self.next_modality_id,
            name=name,
            feature_size=feature_size,
            features=initial_features.copy()
        )
        
        self.modalities[self.next_modality_id] = modality
        self.next_modality_id += 1
        
        return modality
    
    def update_modality(self,
                       modality_id: int,
                       new_features: np.ndarray):
        """Update modality features"""
        if modality_id in self.modalities:
            self.modalities[modality_id].features = new_features.copy()
    
    def fuse_modalities(self,
                       modality_ids: List[int],
                       context: Optional[np.ndarray] = None) -> UnifiedRepresentation:
        """Fuse multiple modalities"""
        modalities = [self.modalities[mid] for mid in modality_ids if mid in self.modalities]
        
        if not modalities:
            return None
        
        # Compute attention weights
        attention_weights = self.attention.compute_attention_weights(modalities, context)
        
        # Use attention weights for fusion
        fusion_weights = attention_weights.copy()
        
        # Fuse
        unified = self.sensory_fusion.fuse_modalities(modalities, fusion_weights)
        
        if unified:
            self.unified_representations.append(unified)
        
        return unified
    
    def learn_cross_modal_mapping(self,
                                 from_modality_id: int,
                                 to_modality_id: int):
        """Learn cross-modal mapping"""
        if from_modality_id not in self.modalities or to_modality_id not in self.modalities:
            return
        
        from_mod = self.modalities[from_modality_id]
        to_mod = self.modalities[to_modality_id]
        
        self.cross_modal_learning.learn_mapping(
            from_modality_id, to_modality_id,
            from_mod.features, to_mod.features
        )
    
    def translate_modality(self,
                          from_modality_id: int,
                          to_modality_id: int) -> Tuple[np.ndarray, float]:
        """Translate between modalities"""
        if from_modality_id not in self.modalities:
            return np.zeros(10), 0.0
        
        from_mod = self.modalities[from_modality_id]
        translated, confidence = self.cross_modal_learning.translate(
            from_modality_id, to_modality_id, from_mod.features
        )
        
        return translated, confidence
    
    def get_statistics(self) -> Dict:
        """Get statistics about multi-modal integration"""
        return {
            'modalities': len(self.modalities),
            'cross_modal_mappings': len(self.cross_modal_learning.mappings),
            'unified_representations': len(self.unified_representations),
            'attention_history_length': len(self.attention.attention_history)
        }

