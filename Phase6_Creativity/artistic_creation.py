#!/usr/bin/env python3
"""
Artistic & Aesthetic Creation - Phase 6.3
Implements aesthetic evaluation, style learning, composition generation,
emotional expression, and style transfer
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
import random
from collections import defaultdict

# Import dependencies
try:
    from multimodal_integration import MultiModalIntegrationManager
    from qualia_subjective import QualiaSubjectiveExperience
    from self_model import SelfModelSystem
except ImportError:
    MultiModalIntegrationManager = None
    QualiaSubjectiveExperience = None
    SelfModelSystem = None


@dataclass
class Artwork:
    """Represents an artistic creation"""
    artwork_id: int
    description: str
    features: np.ndarray  # Feature vector representing the artwork
    style: str
    emotion: str
    aesthetic_score: float
    creativity_score: float
    created_time: float = 0.0
    medium: str = 'abstract'  # 'visual', 'musical', 'textual', 'abstract'


@dataclass
class ArtisticStyle:
    """Represents an artistic style"""
    style_id: int
    name: str
    features: np.ndarray  # Style feature vector
    examples: List[np.ndarray]  # Example artworks in this style
    learned_from: List[int]  # Artwork IDs used to learn this style
    strength: float = 1.0


class AestheticEvaluation:
    """
    Aesthetic Evaluation
    
    Assesses beauty, harmony, balance in artworks
    Based on aesthetic principles
    """
    
    def __init__(self,
                 harmony_weight: float = 0.3,
                 balance_weight: float = 0.3,
                 complexity_weight: float = 0.2,
                 unity_weight: float = 0.2):
        self.harmony_weight = harmony_weight
        self.balance_weight = balance_weight
        self.complexity_weight = complexity_weight
        self.unity_weight = unity_weight
        self.evaluation_history: List[Dict] = []
    
    def evaluate_aesthetics(self, artwork: Artwork) -> Dict:
        """
        Evaluate aesthetic qualities of an artwork
        
        Returns:
            Dictionary with aesthetic scores
        """
        features = artwork.features
        
        # Compute harmony (similarity between related features)
        harmony = self._compute_harmony(features)
        
        # Compute balance (symmetry and distribution)
        balance = self._compute_balance(features)
        
        # Compute complexity (variety and detail)
        complexity = self._compute_complexity(features)
        
        # Compute unity (coherence and consistency)
        unity = self._compute_unity(features)
        
        # Overall aesthetic score
        aesthetic_score = (
            self.harmony_weight * harmony +
            self.balance_weight * balance +
            self.complexity_weight * complexity +
            self.unity_weight * unity
        )
        
        evaluation = {
            'artwork_id': artwork.artwork_id,
            'aesthetic_score': aesthetic_score,
            'harmony': harmony,
            'balance': balance,
            'complexity': complexity,
            'unity': unity
        }
        
        self.evaluation_history.append(evaluation)
        return evaluation
    
    def _compute_harmony(self, features: np.ndarray) -> float:
        """Compute harmony score"""
        # Harmony: related features should be similar
        # Group features into pairs and check similarity
        if len(features) < 2:
            return 0.5
        
        similarities = []
        for i in range(0, len(features) - 1, 2):
            feat1 = features[i]
            feat2 = features[i + 1] if i + 1 < len(features) else features[0]
            
            # Normalize for comparison
            norm1 = np.linalg.norm([feat1])
            norm2 = np.linalg.norm([feat2])
            
            if norm1 > 0 and norm2 > 0:
                sim = min(feat1, feat2) / max(feat1, feat2) if max(feat1, feat2) > 0 else 0.0
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _compute_balance(self, features: np.ndarray) -> float:
        """Compute balance score"""
        # Balance: features should be evenly distributed
        if len(features) == 0:
            return 0.5
        
        # Check symmetry (if even number of features)
        if len(features) % 2 == 0:
            mid = len(features) // 2
            left = features[:mid]
            right = features[mid:]
            symmetry = 1.0 - np.mean(np.abs(left - right[::-1]))
        else:
            symmetry = 0.5
        
        # Check distribution uniformity
        variance = np.var(features)
        uniformity = 1.0 / (1.0 + variance)
        
        balance = (symmetry + uniformity) / 2.0
        return balance
    
    def _compute_complexity(self, features: np.ndarray) -> float:
        """Compute complexity score"""
        # Complexity: variety and detail
        if len(features) == 0:
            return 0.0
        
        # Variance indicates complexity
        variance = np.var(features)
        
        # Number of distinct values
        unique_ratio = len(np.unique(features)) / len(features)
        
        # Entropy-like measure
        hist, _ = np.histogram(features, bins=min(10, len(features)))
        hist = hist[hist > 0]
        if len(hist) > 0:
            probs = hist / np.sum(hist)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            normalized_entropy = entropy / np.log(len(hist) + 1e-10)
        else:
            normalized_entropy = 0.0
        
        complexity = (variance + unique_ratio + normalized_entropy) / 3.0
        return min(1.0, complexity)
    
    def _compute_unity(self, features: np.ndarray) -> float:
        """Compute unity score"""
        # Unity: coherence and consistency
        if len(features) < 2:
            return 1.0
        
        # Check consistency (low variance indicates unity)
        variance = np.var(features)
        consistency = 1.0 / (1.0 + variance * 5.0)
        
        # Check coherence (features should relate to each other)
        mean_feat = np.mean(features)
        deviations = np.abs(features - mean_feat)
        coherence = 1.0 - np.mean(deviations)
        
        unity = (consistency + coherence) / 2.0
        return max(0.0, min(1.0, unity))


class StyleLearning:
    """
    Style Learning
    
    Learns and replicates artistic styles
    Extracts style features from examples
    """
    
    def __init__(self,
                 learning_rate: float = 0.1,
                 min_examples: int = 3):
        self.learning_rate = learning_rate
        self.min_examples = min_examples
        self.styles: Dict[int, ArtisticStyle] = {}
        self.next_style_id = 0
    
    def learn_style(self,
                   style_name: str,
                   example_artworks: List[Artwork]) -> Optional[ArtisticStyle]:
        """
        Learn a style from example artworks
        
        Returns:
            Learned style or None
        """
        if len(example_artworks) < self.min_examples:
            return None
        
        # Extract style features (average of example features)
        style_features = np.mean([art.features for art in example_artworks], axis=0)
        
        # Normalize
        norm = np.linalg.norm(style_features)
        if norm > 0:
            style_features = style_features / norm
        
        # Create style
        style = ArtisticStyle(
            style_id=self.next_style_id,
            name=style_name,
            features=style_features,
            examples=[art.features.copy() for art in example_artworks],
            learned_from=[art.artwork_id for art in example_artworks],
            strength=1.0
        )
        
        self.styles[self.next_style_id] = style
        self.next_style_id += 1
        
        return style
    
    def apply_style(self,
                   base_features: np.ndarray,
                   style: ArtisticStyle,
                   strength: float = 0.7) -> np.ndarray:
        """
        Apply a style to base features
        
        Returns:
            Styled feature vector
        """
        # Blend base features with style features
        styled = (1.0 - strength) * base_features + strength * style.features
        
        # Normalize
        norm = np.linalg.norm(styled)
        if norm > 0:
            styled = styled / norm
        
        return styled
    
    def update_style(self, style: ArtisticStyle, new_example: Artwork):
        """Update style with new example"""
        # Update features using learning rate
        style.features = (
            (1.0 - self.learning_rate) * style.features +
            self.learning_rate * new_example.features
        )
        
        # Normalize
        norm = np.linalg.norm(style.features)
        if norm > 0:
            style.features = style.features / norm
        
        # Add to examples
        style.examples.append(new_example.features.copy())
        style.learned_from.append(new_example.artwork_id)
        style.strength = min(1.0, style.strength + 0.1)


class CompositionGeneration:
    """
    Composition Generation
    
    Creates visual/musical compositions
    Generates structured artistic works
    """
    
    def __init__(self,
                 composition_structure: str = 'balanced'):
        self.composition_structure = composition_structure
        self.compositions_created: List[Artwork] = []
    
    def generate_composition(self,
                           base_theme: np.ndarray,
                           medium: str = 'visual',
                           num_elements: int = 10) -> Artwork:
        """
        Generate an artistic composition
        
        Returns:
            Generated artwork
        """
        if self.composition_structure == 'balanced':
            features = self._generate_balanced_composition(base_theme, num_elements)
        elif self.composition_structure == 'rhythmic':
            features = self._generate_rhythmic_composition(base_theme, num_elements)
        elif self.composition_structure == 'harmonic':
            features = self._generate_harmonic_composition(base_theme, num_elements)
        else:
            features = self._generate_balanced_composition(base_theme, num_elements)
        
        artwork = Artwork(
            artwork_id=-1,  # Will be assigned later
            description=f"{medium} composition",
            features=features,
            style='generated',
            emotion='neutral',
            aesthetic_score=0.5,
            creativity_score=0.6,
            medium=medium
        )
        
        self.compositions_created.append(artwork)
        return artwork
    
    def _generate_balanced_composition(self, base_theme: np.ndarray, num_elements: int) -> np.ndarray:
        """Generate balanced composition"""
        features = []
        
        for i in range(num_elements):
            # Vary around base theme
            variation = base_theme.copy() if len(base_theme) > 0 else np.random.normal(0, 1, 5)
            
            # Add structured variation
            phase = 2 * np.pi * i / num_elements
            variation = variation + 0.3 * np.sin(phase)
            
            features.append(variation[0] if len(variation) > 0 else variation)
        
        return np.array(features)
    
    def _generate_rhythmic_composition(self, base_theme: np.ndarray, num_elements: int) -> np.ndarray:
        """Generate rhythmic composition"""
        features = []
        
        rhythm_pattern = [1.0, 0.5, 0.8, 0.5, 1.0, 0.3, 0.9, 0.4]
        
        for i in range(num_elements):
            rhythm_value = rhythm_pattern[i % len(rhythm_pattern)]
            base_value = base_theme[0] if len(base_theme) > 0 else 0.5
            features.append(base_value * rhythm_value)
        
        return np.array(features)
    
    def _generate_harmonic_composition(self, base_theme: np.ndarray, num_elements: int) -> np.ndarray:
        """Generate harmonic composition"""
        features = []
        
        # Harmonic series
        harmonics = [1.0, 0.5, 0.33, 0.25, 0.2]
        
        for i in range(num_elements):
            harmonic_idx = i % len(harmonics)
            base_value = base_theme[0] if len(base_theme) > 0 else 0.5
            features.append(base_value * harmonics[harmonic_idx])
        
        return np.array(features)


class EmotionalExpression:
    """
    Emotional Expression
    
    Expresses emotions through art
    Maps emotions to artistic features
    """
    
    def __init__(self):
        self.emotion_mappings: Dict[str, np.ndarray] = {}
        self._initialize_emotion_mappings()
    
    def _initialize_emotion_mappings(self):
        """Initialize emotion to feature mappings"""
        # Simple emotion feature vectors
        self.emotion_mappings = {
            'joy': np.array([1.0, 0.8, 0.6, 0.9, 0.7]),
            'sadness': np.array([0.2, 0.3, 0.4, 0.1, 0.3]),
            'anger': np.array([0.9, 0.1, 0.8, 0.2, 0.6]),
            'fear': np.array([0.3, 0.7, 0.2, 0.4, 0.5]),
            'love': np.array([0.8, 0.9, 0.7, 0.8, 0.9]),
            'peace': np.array([0.5, 0.6, 0.5, 0.7, 0.6]),
            'excitement': np.array([0.9, 0.7, 0.8, 0.6, 0.8]),
            'calm': np.array([0.4, 0.5, 0.4, 0.6, 0.5]),
        }
        
        # Normalize all emotion vectors
        for emotion, features in self.emotion_mappings.items():
            norm = np.linalg.norm(features)
            if norm > 0:
                self.emotion_mappings[emotion] = features / norm
    
    def express_emotion(self,
                       base_features: np.ndarray,
                       emotion: str,
                       intensity: float = 0.7) -> np.ndarray:
        """
        Express emotion in artwork
        
        Returns:
            Emotionally expressive feature vector
        """
        if emotion not in self.emotion_mappings:
            emotion = 'neutral'
            intensity = 0.0
        
        if emotion == 'neutral':
            return base_features
        
        emotion_features = self.emotion_mappings[emotion]
        
        # Ensure same dimensionality before blending
        min_dim = min(len(base_features), len(emotion_features))
        base_truncated = base_features[:min_dim]
        emotion_truncated = emotion_features[:min_dim]
        
        # Blend base with emotion
        expressive = (1.0 - intensity) * base_truncated + intensity * emotion_truncated
        
        # If base_features was longer, pad with zeros or extend
        if len(base_features) > min_dim:
            expressive = np.concatenate([expressive, base_features[min_dim:] * (1.0 - intensity)])
        
        # Normalize
        norm = np.linalg.norm(expressive)
        if norm > 0:
            expressive = expressive / norm
        
        return expressive
    
    def detect_emotion(self, artwork: Artwork) -> str:
        """Detect emotion expressed in artwork"""
        best_emotion = 'neutral'
        best_similarity = 0.0
        
        for emotion, emotion_features in self.emotion_mappings.items():
            similarity = np.dot(artwork.features, emotion_features) / (
                np.linalg.norm(artwork.features) * np.linalg.norm(emotion_features) + 1e-10
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_emotion = emotion
        
        return best_emotion if best_similarity > 0.5 else 'neutral'


class StyleTransfer:
    """
    Style Transfer
    
    Applies one style to another medium or artwork
    Transfers stylistic features
    """
    
    def __init__(self,
                 transfer_strength: float = 0.6):
        self.transfer_strength = transfer_strength
        self.transfers_performed: List[Tuple[int, int]] = []  # (source_id, target_id)
    
    def transfer_style(self,
                      source_artwork: Artwork,
                      target_base: np.ndarray,
                      strength: Optional[float] = None) -> np.ndarray:
        """
        Transfer style from source artwork to target base
        
        Returns:
            Styled target vector
        """
        if strength is None:
            strength = self.transfer_strength
        
        # Extract style features (can be the artwork features themselves)
        style_features = source_artwork.features
        
        # Blend target with style
        styled = (1.0 - strength) * target_base + strength * style_features
        
        # Ensure same dimensionality
        min_dim = min(len(styled), len(style_features))
        styled = styled[:min_dim]
        style_features = style_features[:min_dim]
        
        # Blend again
        styled = (1.0 - strength) * styled + strength * style_features
        
        # Normalize
        norm = np.linalg.norm(styled)
        if norm > 0:
            styled = styled / norm
        
        return styled
    
    def cross_modal_transfer(self,
                            source_artwork: Artwork,
                            target_medium: str) -> Artwork:
        """
        Transfer style across different media
        
        Returns:
            Artwork in target medium with transferred style
        """
        # Create base for target medium
        if target_medium == 'visual':
            base = np.random.normal(0, 1, len(source_artwork.features))
        elif target_medium == 'musical':
            base = np.random.uniform(0, 1, len(source_artwork.features))
        else:
            base = np.random.normal(0, 1, len(source_artwork.features))
        
        # Normalize base
        norm = np.linalg.norm(base)
        if norm > 0:
            base = base / norm
        
        # Transfer style
        styled_features = self.transfer_style(source_artwork, base)
        
        transferred_artwork = Artwork(
            artwork_id=-1,
            description=f"Style transfer from {source_artwork.medium} to {target_medium}",
            features=styled_features,
            style=source_artwork.style,
            emotion=source_artwork.emotion,
            aesthetic_score=source_artwork.aesthetic_score * 0.9,
            creativity_score=source_artwork.creativity_score * 0.8,
            medium=target_medium
        )
        
        return transferred_artwork


class ArtisticCreation:
    """
    Artistic Creation Manager
    
    Integrates all artistic creation components
    """
    
    def __init__(self,
                 brain_system=None,
                 multimodal_integration: Optional[MultiModalIntegrationManager] = None,
                 qualia_system: Optional[QualiaSubjectiveExperience] = None,
                 self_model: Optional[SelfModelSystem] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.aesthetic_evaluation = AestheticEvaluation()
        self.style_learning = StyleLearning()
        self.composition_generation = CompositionGeneration()
        self.emotional_expression = EmotionalExpression()
        self.style_transfer = StyleTransfer()
        
        # Integration with existing systems
        self.multimodal_integration = multimodal_integration
        self.qualia_system = qualia_system
        self.self_model = self_model
        
        # Artwork tracking
        self.artworks: List[Artwork] = []
        self.next_artwork_id = 0
        
        # Statistics
        self.stats = {
            'artworks_created': 0,
            'styles_learned': 0,
            'average_aesthetic': 0.0,
            'average_creativity': 0.0
        }
    
    def create_artwork(self,
                      base_theme: Optional[np.ndarray] = None,
                      style: Optional[str] = None,
                      emotion: Optional[str] = None,
                      medium: str = 'visual') -> Artwork:
        """
        Create an artistic work
        
        Returns:
            Created artwork
        """
        # Generate base composition
        if base_theme is None:
            base_theme = np.random.normal(0, 1, 5)  # Use 5D to match emotion features
            norm = np.linalg.norm(base_theme)
            if norm > 0:
                base_theme = base_theme / norm
        
        # Ensure base_theme is at least 1D
        if len(base_theme.shape) == 0:
            base_theme = np.array([base_theme])
        
        composition = self.composition_generation.generate_composition(
            base_theme, medium=medium
        )
        
        features = composition.features
        
        # Apply style if specified
        if style:
            style_obj = self.style_learning.styles.get(style)
            if style_obj:
                features = self.style_learning.apply_style(features, style_obj)
        
        # Express emotion if specified
        if emotion:
            features = self.emotional_expression.express_emotion(features, emotion)
        
        # Create artwork
        artwork = Artwork(
            artwork_id=self.next_artwork_id,
            description=f"{medium} artwork",
            features=features,
            style=style or 'original',
            emotion=emotion or 'neutral',
            aesthetic_score=0.5,
            creativity_score=0.6,
            medium=medium
        )
        
        self.next_artwork_id += 1
        
        # Evaluate aesthetics
        evaluation = self.aesthetic_evaluation.evaluate_aesthetics(artwork)
        artwork.aesthetic_score = evaluation['aesthetic_score']
        
        # Detect emotion if not specified
        if not emotion:
            artwork.emotion = self.emotional_expression.detect_emotion(artwork)
        
        # Store and update stats
        self.artworks.append(artwork)
        self._update_stats(artwork)
        
        return artwork
    
    def learn_style_from_examples(self,
                                 style_name: str,
                                 example_artworks: List[Artwork]) -> Optional[ArtisticStyle]:
        """Learn a style from example artworks"""
        style = self.style_learning.learn_style(style_name, example_artworks)
        if style:
            self.stats['styles_learned'] += 1
        return style
    
    def transfer_style_between_artworks(self,
                                       source_artwork: Artwork,
                                       target_base: np.ndarray) -> Artwork:
        """Transfer style between artworks"""
        styled_features = self.style_transfer.transfer_style(source_artwork, target_base)
        
        transferred = Artwork(
            artwork_id=self.next_artwork_id,
            description=f"Style transfer from artwork {source_artwork.artwork_id}",
            features=styled_features,
            style=source_artwork.style,
            emotion=source_artwork.emotion,
            aesthetic_score=source_artwork.aesthetic_score * 0.9,
            creativity_score=source_artwork.creativity_score * 0.8,
            medium=source_artwork.medium
        )
        
        self.next_artwork_id += 1
        self.artworks.append(transferred)
        self._update_stats(transferred)
        
        return transferred
    
    def _update_stats(self, artwork: Artwork):
        """Update statistics"""
        self.stats['artworks_created'] += 1
        
        total = self.stats['artworks_created']
        self.stats['average_aesthetic'] = (
            (self.stats['average_aesthetic'] * (total - 1) + artwork.aesthetic_score) / total
        )
        self.stats['average_creativity'] = (
            (self.stats['average_creativity'] * (total - 1) + artwork.creativity_score) / total
        )
    
    def get_statistics(self) -> Dict:
        """Get artistic creation statistics"""
        return self.stats.copy()
    
    def get_best_artworks(self, top_k: int = 5, metric: str = 'aesthetic') -> List[Artwork]:
        """Get top artworks by metric"""
        if not self.artworks:
            return []
        
        if metric == 'aesthetic':
            sorted_artworks = sorted(self.artworks, key=lambda x: x.aesthetic_score, reverse=True)
        elif metric == 'creativity':
            sorted_artworks = sorted(self.artworks, key=lambda x: x.creativity_score, reverse=True)
        else:
            sorted_artworks = sorted(self.artworks, key=lambda x: x.aesthetic_score, reverse=True)
        
        return sorted_artworks[:top_k]

