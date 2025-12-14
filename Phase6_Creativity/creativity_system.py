#!/usr/bin/env python3
"""
Creativity System - Phase 6.1: Creative Idea Generation
Implements conceptual blending, divergent thinking, associative networks,
randomness injection, and novelty detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
import random
from collections import defaultdict, deque

# Import dependencies from existing phases
try:
    from semantic_representations import SemanticNetwork, ConceptFormation, Concept
    from hierarchical_learning import HierarchicalFeatureLearning
    from intrinsic_motivation import CuriosityDrive, NoveltySeeking
except ImportError:
    # Fallback if imports fail
    SemanticNetwork = None
    ConceptFormation = None
    Concept = None
    HierarchicalFeatureLearning = None
    CuriosityDrive = None
    NoveltySeeking = None


@dataclass
class CreativeIdea:
    """Represents a creative idea"""
    idea_id: int
    description: str
    blended_concepts: List[int]  # Concept IDs that were blended
    novelty_score: float
    feasibility_score: float
    creativity_score: float
    created_time: float = 0.0
    exploration_path: List[int] = field(default_factory=list)  # Path through semantic network


class ConceptualBlending:
    """
    Conceptual Blending
    
    Combines unrelated concepts to generate novel ideas
    Based on Fauconnier & Turner's conceptual blending theory
    """
    
    def __init__(self,
                 blend_strength: float = 0.5,
                 novelty_threshold: float = 0.3):
        self.blend_strength = blend_strength
        self.novelty_threshold = novelty_threshold
        self.blends_created: List[CreativeIdea] = []
        self.blend_history: List[Tuple[int, int]] = []  # (concept1_id, concept2_id)
    
    def blend_concepts(self,
                      concept1: Concept,
                      concept2: Concept,
                      blend_type: str = 'average') -> np.ndarray:
        """
        Blend two concepts into a new combined concept
        
        Args:
            concept1: First concept to blend
            concept2: Second concept to blend
            blend_type: Type of blending ('average', 'weighted', 'projection')
        
        Returns:
            Blended feature vector
        """
        features1 = concept1.features
        features2 = concept2.features
        
        # Ensure same dimensionality
        min_dim = min(len(features1), len(features2))
        features1 = features1[:min_dim]
        features2 = features2[:min_dim]
        
        if blend_type == 'average':
            # Simple average blending
            blended = (features1 + features2) / 2.0
        elif blend_type == 'weighted':
            # Weighted blend based on concept strength
            weight1 = concept1.strength / (concept1.strength + concept2.strength + 1e-10)
            weight2 = concept2.strength / (concept1.strength + concept2.strength + 1e-10)
            blended = weight1 * features1 + weight2 * features2
        elif blend_type == 'projection':
            # Projection: take some features from each
            mid_point = min_dim // 2
            blended = np.concatenate([features1[:mid_point], features2[mid_point:]])
        else:
            # Default to average
            blended = (features1 + features2) / 2.0
        
        # Add controlled randomness
        noise = np.random.normal(0, 0.1 * self.blend_strength, len(blended))
        blended = blended + noise
        
        # Normalize
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm
        
        return blended
    
    def find_unrelated_concepts(self,
                               concept1_id: int,
                               semantic_network: SemanticNetwork,
                               concept_formation: ConceptFormation,
                               max_distance: float = 0.3) -> List[int]:
        """
        Find concepts that are unrelated to concept1
        
        Args:
            concept1_id: ID of the source concept
            semantic_network: Semantic network to search
            concept_formation: Concept formation system
            max_distance: Maximum similarity to consider unrelated
        
        Returns:
            List of unrelated concept IDs
        """
        unrelated = []
        concept1 = concept_formation.concepts[concept1_id]
        
        for concept_id, concept2 in concept_formation.concepts.items():
            if concept_id == concept1_id:
                continue
            
            # Check semantic distance
            similarity = concept_formation.compute_similarity(concept1.features, concept2.features)
            
            # Check if there's a direct relation
            has_relation = False
            if semantic_network and concept_id in semantic_network.concept_relations:
                for rel in semantic_network.concept_relations[concept_id]:
                    if rel.to_concept_id == concept1_id:
                        has_relation = True
                        break
            
            # Consider unrelated if similarity is low and no direct relation
            if similarity < max_distance and not has_relation:
                unrelated.append(concept_id)
        
        return unrelated


class DivergentThinking:
    """
    Divergent Thinking
    
    Generates multiple alternative solutions or ideas
    Explores solution space broadly
    """
    
    def __init__(self,
                 num_alternatives: int = 10,
                 diversity_strength: float = 0.5):
        self.num_alternatives = num_alternatives
        self.diversity_strength = diversity_strength
        self.generated_alternatives: List[List[np.ndarray]] = []
    
    def generate_alternatives(self,
                             base_concept: np.ndarray,
                             num_alternatives: Optional[int] = None) -> List[np.ndarray]:
        """
        Generate multiple alternative variations of a concept
        
        Args:
            base_concept: Base concept to vary
            num_alternatives: Number of alternatives to generate
        
        Returns:
            List of alternative concept vectors
        """
        if num_alternatives is None:
            num_alternatives = self.num_alternatives
        
        alternatives = []
        
        for i in range(num_alternatives):
            # Create variation
            variation = self._create_variation(base_concept, i)
            alternatives.append(variation)
        
        self.generated_alternatives.append(alternatives)
        return alternatives
    
    def _create_variation(self, base: np.ndarray, index: int) -> np.ndarray:
        """Create a single variation of the base concept"""
        # Different variation strategies
        strategy = index % 4
        
        if strategy == 0:
            # Additive noise
            noise = np.random.normal(0, self.diversity_strength, len(base))
            variation = base + noise
        elif strategy == 1:
            # Multiplicative variation
            multiplier = 1.0 + np.random.normal(0, self.diversity_strength, len(base))
            variation = base * multiplier
        elif strategy == 2:
            # Feature swapping (if multiple features)
            variation = base.copy()
            if len(variation) > 1:
                idx1, idx2 = np.random.choice(len(variation), 2, replace=False)
                variation[idx1], variation[idx2] = variation[idx2], variation[idx1]
        else:
            # Combination approach
            noise = np.random.normal(0, self.diversity_strength * 0.5, len(base))
            multiplier = 1.0 + np.random.normal(0, self.diversity_strength * 0.3, len(base))
            variation = (base + noise) * multiplier
        
        # Normalize
        norm = np.linalg.norm(variation)
        if norm > 0:
            variation = variation / norm
        
        return variation
    
    def compute_diversity(self, alternatives: List[np.ndarray]) -> float:
        """
        Compute diversity score of alternatives
        
        Returns:
            Diversity score (higher = more diverse)
        """
        if len(alternatives) < 2:
            return 0.0
        
        pairwise_distances = []
        for i in range(len(alternatives)):
            for j in range(i + 1, len(alternatives)):
                dist = np.linalg.norm(alternatives[i] - alternatives[j])
                pairwise_distances.append(dist)
        
        return np.mean(pairwise_distances)


class AssociativeNetworks:
    """
    Associative Networks
    
    Traverses semantic networks to find unexpected connections
    Explores distant relationships
    """
    
    def __init__(self,
                 max_path_length: int = 5,
                 exploration_probability: float = 0.3):
        self.max_path_length = max_path_length
        self.exploration_probability = exploration_probability
        self.explored_paths: List[List[int]] = []
    
    def find_unexpected_connections(self,
                                   start_concept_id: int,
                                   semantic_network: SemanticNetwork,
                                   concept_formation: ConceptFormation) -> List[Tuple[int, List[int]]]:
        """
        Find unexpected connections by traversing semantic network
        
        Args:
            start_concept_id: Starting concept ID
            semantic_network: Semantic network to traverse
            concept_formation: Concept formation system
        
        Returns:
            List of (target_concept_id, path) tuples
        """
        if not semantic_network or start_concept_id not in concept_formation.concepts:
            return []
        
        unexpected_connections = []
        visited = set()
        
        # BFS traversal with exploration
        queue = deque([(start_concept_id, [start_concept_id])])
        visited.add(start_concept_id)
        
        while queue and len(unexpected_connections) < 20:
            current_id, path = queue.popleft()
            
            # Check if path is long enough to be unexpected
            if len(path) >= 3:
                # Compute semantic distance
                start_concept = concept_formation.concepts[start_concept_id]
                current_concept = concept_formation.concepts[current_id]
                distance = 1.0 - concept_formation.compute_similarity(
                    start_concept.features, current_concept.features
                )
                
                if distance > 0.5:  # Unexpected if distant
                    unexpected_connections.append((current_id, path.copy()))
            
            # Continue exploration
            if len(path) < self.max_path_length:
                # Get related concepts
                if current_id in semantic_network.concept_relations:
                    related = semantic_network.concept_relations[current_id]
                    
                    for rel in related:
                        next_id = rel.to_concept_id
                        
                        # Exploration vs exploitation
                        if (next_id not in visited and 
                            (random.random() < self.exploration_probability or 
                             len(path) < 2)):
                            queue.append((next_id, path + [next_id]))
                            visited.add(next_id)
        
        self.explored_paths.extend([path for _, path in unexpected_connections])
        return unexpected_connections
    
    def random_walk(self,
                   start_concept_id: int,
                   semantic_network: SemanticNetwork,
                   steps: int = 5) -> List[int]:
        """
        Perform random walk through semantic network
        
        Returns:
            Path of concept IDs
        """
        if not semantic_network or start_concept_id not in semantic_network.concept_relations:
            return [start_concept_id]
        
        path = [start_concept_id]
        current_id = start_concept_id
        
        for _ in range(steps):
            if current_id not in semantic_network.concept_relations:
                break
            
            relations = semantic_network.concept_relations[current_id]
            if not relations:
                break
            
            # Choose next concept (weighted by relation strength)
            weights = [rel.strength for rel in relations]
            total_weight = sum(weights)
            
            if total_weight > 0:
                probs = [w / total_weight for w in weights]
                next_rel = np.random.choice(len(relations), p=probs)
                current_id = relations[next_rel].to_concept_id
                path.append(current_id)
            else:
                break
        
        return path


class RandomnessInjection:
    """
    Randomness Injection
    
    Controlled randomness for exploration
    Balances exploration vs exploitation
    """
    
    def __init__(self,
                 randomness_strength: float = 0.2,
                 adaptive_randomness: bool = True):
        self.randomness_strength = randomness_strength
        self.adaptive_randomness = adaptive_randomness
        self.randomness_history: deque = deque(maxlen=100)
    
    def inject_randomness(self,
                         base_vector: np.ndarray,
                         randomness_type: str = 'gaussian') -> np.ndarray:
        """
        Inject controlled randomness into a vector
        
        Args:
            base_vector: Base vector to randomize
            randomness_type: Type of randomness ('gaussian', 'uniform', 'sparse')
        
        Returns:
            Randomized vector
        """
        if randomness_type == 'gaussian':
            noise = np.random.normal(0, self.randomness_strength, len(base_vector))
            randomized = base_vector + noise
        elif randomness_type == 'uniform':
            noise = np.random.uniform(-self.randomness_strength, 
                                      self.randomness_strength, 
                                      len(base_vector))
            randomized = base_vector + noise
        elif randomness_type == 'sparse':
            # Sparse: only randomize some dimensions
            mask = np.random.random(len(base_vector)) < 0.3
            noise = np.random.normal(0, self.randomness_strength, len(base_vector))
            randomized = base_vector.copy()
            randomized[mask] += noise[mask]
        else:
            randomized = base_vector
        
        # Normalize
        norm = np.linalg.norm(randomized)
        if norm > 0:
            randomized = randomized / norm
        
        self.randomness_history.append(np.linalg.norm(noise))
        return randomized
    
    def adaptive_randomness_strength(self, novelty_score: float) -> float:
        """
        Adapt randomness strength based on novelty
        
        More randomness when novelty is low (need more exploration)
        """
        if not self.adaptive_randomness:
            return self.randomness_strength
        
        # Increase randomness if novelty is low
        if novelty_score < 0.3:
            return min(0.5, self.randomness_strength * 1.5)
        elif novelty_score > 0.7:
            return max(0.1, self.randomness_strength * 0.7)
        else:
            return self.randomness_strength


class NoveltyDetection:
    """
    Novelty Detection
    
    Identifies truly novel vs recombined ideas
    Tracks idea history
    """
    
    def __init__(self,
                 novelty_threshold: float = 0.3,
                 history_size: int = 1000):
        self.novelty_threshold = novelty_threshold
        self.history_size = history_size
        self.idea_history: deque = deque(maxlen=history_size)
        self.novelty_scores: deque = deque(maxlen=history_size)
    
    def compute_novelty(self, idea_vector: np.ndarray) -> float:
        """
        Compute novelty score for an idea
        
        Returns:
            Novelty score (0-1, higher = more novel)
        """
        if len(self.idea_history) == 0:
            # First idea is maximally novel
            self.idea_history.append(idea_vector.copy())
            self.novelty_scores.append(1.0)
            return 1.0
        
        # Compare with all previous ideas
        similarities = []
        for past_idea in self.idea_history:
            # Handle dimension mismatch
            min_dim = min(len(idea_vector), len(past_idea))
            idea_truncated = idea_vector[:min_dim]
            past_truncated = past_idea[:min_dim]
            
            similarity = np.dot(idea_truncated, past_truncated) / (
                np.linalg.norm(idea_truncated) * np.linalg.norm(past_truncated) + 1e-10
            )
            similarities.append(similarity)
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        novelty = 1.0 - max_similarity
        
        # Store
        self.idea_history.append(idea_vector.copy())
        self.novelty_scores.append(novelty)
        
        return novelty
    
    def is_novel(self, idea_vector: np.ndarray) -> bool:
        """Check if idea is novel enough"""
        novelty = self.compute_novelty(idea_vector)
        return novelty >= self.novelty_threshold
    
    def compute_creativity_score(self,
                                idea_vector: np.ndarray,
                                novelty_weight: float = 0.6,
                                feasibility_weight: float = 0.4) -> float:
        """
        Compute overall creativity score
        
        Args:
            idea_vector: Idea feature vector
            novelty_weight: Weight for novelty component
            feasibility_weight: Weight for feasibility component
        
        Returns:
            Creativity score (0-1)
        """
        novelty = self.compute_novelty(idea_vector)
        
        # Simple feasibility heuristic: ideas closer to known concepts are more feasible
        feasibility = 1.0 - novelty  # Inverse of novelty as proxy for feasibility
        
        creativity = (novelty_weight * novelty + 
                     feasibility_weight * feasibility)
        
        return creativity


class CreativitySystem:
    """
    Creativity System Manager
    
    Integrates all creativity components
    Generates creative ideas
    """
    
    def __init__(self,
                 brain_system=None,
                 blend_strength: float = 0.5,
                 num_alternatives: int = 10,
                 randomness_strength: float = 0.2):
        self.brain_system = brain_system
        
        # Initialize components
        self.conceptual_blending = ConceptualBlending(blend_strength=blend_strength)
        self.divergent_thinking = DivergentThinking(num_alternatives=num_alternatives)
        self.associative_networks = AssociativeNetworks()
        self.randomness_injection = RandomnessInjection(randomness_strength=randomness_strength)
        self.novelty_detection = NoveltyDetection()
        
        # Integration with existing systems
        self.semantic_network = None
        self.concept_formation = None
        self.hierarchical_learner = None
        self.curiosity_drive = None
        
        # Idea tracking
        self.ideas: List[CreativeIdea] = []
        self.next_idea_id = 0
        
        # Statistics
        self.stats = {
            'ideas_generated': 0,
            'novel_ideas': 0,
            'blends_created': 0,
            'average_novelty': 0.0,
            'average_creativity': 0.0
        }
    
    def initialize_integrations(self,
                               semantic_network: Optional[SemanticNetwork] = None,
                               concept_formation: Optional[ConceptFormation] = None,
                               hierarchical_learner: Optional[HierarchicalFeatureLearning] = None,
                               curiosity_drive: Optional[CuriosityDrive] = None):
        """Initialize integrations with existing systems"""
        self.semantic_network = semantic_network
        self.concept_formation = concept_formation
        self.hierarchical_learner = hierarchical_learner
        self.curiosity_drive = curiosity_drive
    
    def generate_ideas(self,
                      input_data: Optional[np.ndarray] = None,
                      context: Optional[Dict] = None,
                      num_ideas: int = 5,
                      method: str = 'blending') -> List[CreativeIdea]:
        """
        Generate creative ideas
        
        Args:
            input_data: Optional input pattern to start from
            context: Optional context dictionary
            num_ideas: Number of ideas to generate
            method: Generation method ('blending', 'divergent', 'associative', 'random')
        
        Returns:
            List of creative ideas
        """
        ideas = []
        
        for i in range(num_ideas):
            if method == 'blending':
                idea = self._generate_blended_idea(input_data, context)
            elif method == 'divergent':
                idea = self._generate_divergent_idea(input_data, context)
            elif method == 'associative':
                idea = self._generate_associative_idea(input_data, context)
            elif method == 'random':
                idea = self._generate_random_idea(input_data, context)
            else:
                # Try all methods
                methods = ['blending', 'divergent', 'associative']
                method = methods[i % len(methods)]
                idea = self._generate_blended_idea(input_data, context)
            
            if idea:
                ideas.append(idea)
                self.ideas.append(idea)
                self._update_stats(idea)
        
        return ideas
    
    def _generate_blended_idea(self,
                               input_data: Optional[np.ndarray],
                               context: Optional[Dict]) -> Optional[CreativeIdea]:
        """Generate idea through conceptual blending"""
        if not self.concept_formation or len(self.concept_formation.concepts) < 2:
            return None
        
        # Select two concepts to blend
        concept_ids = list(self.concept_formation.concepts.keys())
        if len(concept_ids) < 2:
            return None
        
        concept1_id = random.choice(concept_ids)
        unrelated = self.conceptual_blending.find_unrelated_concepts(
            concept1_id, self.semantic_network, self.concept_formation
        )
        
        if unrelated:
            concept2_id = random.choice(unrelated)
        else:
            # Fallback: choose any other concept
            concept2_id = random.choice([c for c in concept_ids if c != concept1_id])
        
        concept1 = self.concept_formation.concepts[concept1_id]
        concept2 = self.concept_formation.concepts[concept2_id]
        
        # Blend concepts
        blended_features = self.conceptual_blending.blend_concepts(concept1, concept2)
        
        # Add randomness
        blended_features = self.randomness_injection.inject_randomness(blended_features)
        
        # Compute novelty
        novelty = self.novelty_detection.compute_novelty(blended_features)
        creativity = self.novelty_detection.compute_creativity_score(blended_features)
        
        # Create idea
        idea = CreativeIdea(
            idea_id=self.next_idea_id,
            description=f"Blend of {concept1.name} and {concept2.name}",
            blended_concepts=[concept1_id, concept2_id],
            novelty_score=novelty,
            feasibility_score=1.0 - novelty,
            creativity_score=creativity,
            created_time=time.time()
        )
        
        self.next_idea_id += 1
        return idea
    
    def _generate_divergent_idea(self,
                                input_data: Optional[np.ndarray],
                                context: Optional[Dict]) -> Optional[CreativeIdea]:
        """Generate idea through divergent thinking"""
        if input_data is None:
            # Use a random concept as base
            if not self.concept_formation or len(self.concept_formation.concepts) == 0:
                return None
            base_concept = random.choice(list(self.concept_formation.concepts.values()))
            base_vector = base_concept.features
        else:
            base_vector = input_data
        
        # Generate alternatives
        alternatives = self.divergent_thinking.generate_alternatives(base_vector, num_alternatives=1)
        
        if not alternatives:
            return None
        
        idea_vector = alternatives[0]
        
        # Compute novelty
        novelty = self.novelty_detection.compute_novelty(idea_vector)
        creativity = self.novelty_detection.compute_creativity_score(idea_vector)
        
        idea = CreativeIdea(
            idea_id=self.next_idea_id,
            description="Divergent variation",
            blended_concepts=[],
            novelty_score=novelty,
            feasibility_score=1.0 - novelty,
            creativity_score=creativity,
            created_time=time.time()
        )
        
        self.next_idea_id += 1
        return idea
    
    def _generate_associative_idea(self,
                                  input_data: Optional[np.ndarray],
                                  context: Optional[Dict]) -> Optional[CreativeIdea]:
        """Generate idea through associative network traversal"""
        if not self.concept_formation or len(self.concept_formation.concepts) == 0:
            return None
        
        # Start from random or input concept
        if input_data is not None and self.concept_formation:
            # Find closest concept
            best_match = self.concept_formation.find_best_matching_concept(input_data)
            if best_match:
                start_id = best_match[0]
            else:
                start_id = random.choice(list(self.concept_formation.concepts.keys()))
        else:
            start_id = random.choice(list(self.concept_formation.concepts.keys()))
        
        # Find unexpected connections
        connections = self.associative_networks.find_unexpected_connections(
            start_id, self.semantic_network, self.concept_formation
        )
        
        if not connections:
            # Fallback to random walk
            path = self.associative_networks.random_walk(
                start_id, self.semantic_network, steps=3
            )
            if len(path) > 1:
                target_id = path[-1]
            else:
                return None
        else:
            target_id, path = random.choice(connections)
        
        target_concept = self.concept_formation.concepts[target_id]
        idea_vector = target_concept.features.copy()
        
        # Add randomness
        idea_vector = self.randomness_injection.inject_randomness(idea_vector)
        
        # Compute novelty
        novelty = self.novelty_detection.compute_novelty(idea_vector)
        creativity = self.novelty_detection.compute_creativity_score(idea_vector)
        
        idea = CreativeIdea(
            idea_id=self.next_idea_id,
            description=f"Associative connection from concept {start_id}",
            blended_concepts=[start_id, target_id],
            novelty_score=novelty,
            feasibility_score=1.0 - novelty,
            creativity_score=creativity,
            created_time=time.time(),
            exploration_path=[start_id, target_id]
        )
        
        self.next_idea_id += 1
        return idea
    
    def _generate_random_idea(self,
                             input_data: Optional[np.ndarray],
                             context: Optional[Dict]) -> Optional[CreativeIdea]:
        """Generate idea through pure randomness"""
        if input_data is not None:
            base_vector = input_data
        else:
            # Random vector
            base_vector = np.random.normal(0, 1, 10)
            norm = np.linalg.norm(base_vector)
            if norm > 0:
                base_vector = base_vector / norm
        
        # Heavy randomness injection
        idea_vector = self.randomness_injection.inject_randomness(
            base_vector, randomness_type='gaussian'
        )
        
        # Compute novelty
        novelty = self.novelty_detection.compute_novelty(idea_vector)
        creativity = self.novelty_detection.compute_creativity_score(idea_vector)
        
        idea = CreativeIdea(
            idea_id=self.next_idea_id,
            description="Random exploration",
            blended_concepts=[],
            novelty_score=novelty,
            feasibility_score=1.0 - novelty,
            creativity_score=creativity,
            created_time=time.time()
        )
        
        self.next_idea_id += 1
        return idea
    
    def _update_stats(self, idea: CreativeIdea):
        """Update statistics"""
        self.stats['ideas_generated'] += 1
        if idea.novelty_score >= self.novelty_detection.novelty_threshold:
            self.stats['novel_ideas'] += 1
        if len(idea.blended_concepts) >= 2:
            self.stats['blends_created'] += 1
        
        # Update averages
        total = self.stats['ideas_generated']
        self.stats['average_novelty'] = (
            (self.stats['average_novelty'] * (total - 1) + idea.novelty_score) / total
        )
        self.stats['average_creativity'] = (
            (self.stats['average_creativity'] * (total - 1) + idea.creativity_score) / total
        )
    
    def get_statistics(self) -> Dict:
        """Get creativity statistics"""
        return self.stats.copy()
    
    def get_best_ideas(self, top_k: int = 5, metric: str = 'creativity') -> List[CreativeIdea]:
        """
        Get top ideas by metric
        
        Args:
            top_k: Number of ideas to return
            metric: Metric to sort by ('creativity', 'novelty', 'feasibility')
        
        Returns:
            List of top ideas
        """
        if not self.ideas:
            return []
        
        if metric == 'creativity':
            sorted_ideas = sorted(self.ideas, key=lambda x: x.creativity_score, reverse=True)
        elif metric == 'novelty':
            sorted_ideas = sorted(self.ideas, key=lambda x: x.novelty_score, reverse=True)
        elif metric == 'feasibility':
            sorted_ideas = sorted(self.ideas, key=lambda x: x.feasibility_score, reverse=True)
        else:
            sorted_ideas = sorted(self.ideas, key=lambda x: x.creativity_score, reverse=True)
        
        return sorted_ideas[:top_k]

