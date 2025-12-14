#!/usr/bin/env python3
"""
Semantic Representations
Implements concept formation, symbol grounding, meaning extraction, and semantic networks
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
from collections import defaultdict

@dataclass
class Concept:
    """Represents a semantic concept"""
    concept_id: int
    name: str
    features: np.ndarray  # Feature vector
    instances: List[np.ndarray]  # Examples of this concept
    frequency: int = 0  # How often encountered
    strength: float = 1.0  # Concept strength
    created_time: float = 0.0
    
@dataclass
class Symbol:
    """Represents a grounded symbol"""
    symbol_id: int
    label: str
    concept_id: int  # Grounded to concept
    grounding_strength: float = 1.0  # How well grounded
    usage_count: int = 0
    
@dataclass
class SemanticRelation:
    """Represents a semantic relationship"""
    from_concept_id: int
    to_concept_id: int
    relation_type: str  # 'is_a', 'part_of', 'similar_to', 'causes', etc.
    strength: float = 1.0
    frequency: int = 0

class ConceptFormation:
    """
    Concept Formation
    
    Forms concepts by clustering similar patterns
    Uses prototype-based learning
    """
    
    def __init__(self,
                 similarity_threshold: float = 0.7,
                 min_instances: int = 3,
                 learning_rate: float = 0.1):
        self.similarity_threshold = similarity_threshold
        self.min_instances = min_instances
        self.learning_rate = learning_rate
        
        self.concepts: Dict[int, Concept] = {}
        self.next_concept_id = 0
    
    def compute_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Compute cosine similarity between patterns"""
        norm1 = np.linalg.norm(pattern1)
        norm2 = np.linalg.norm(pattern2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(pattern1, pattern2) / (norm1 * norm2)
    
    def find_best_matching_concept(self, pattern: np.ndarray) -> Optional[Tuple[int, float]]:
        """Find concept that best matches pattern"""
        best_concept_id = None
        best_similarity = -1.0
        
        for concept_id, concept in self.concepts.items():
            similarity = self.compute_similarity(pattern, concept.features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_concept_id = concept_id
        
        return (best_concept_id, best_similarity) if best_similarity >= self.similarity_threshold else None
    
    def create_concept(self, pattern: np.ndarray, name: Optional[str] = None) -> Concept:
        """Create a new concept from pattern"""
        concept = Concept(
            concept_id=self.next_concept_id,
            name=name or f"Concept_{self.next_concept_id}",
            features=pattern.copy(),
            instances=[pattern.copy()],
            created_time=time.time()
        )
        
        self.concepts[self.next_concept_id] = concept
        self.next_concept_id += 1
        
        return concept
    
    def update_concept(self, concept: Concept, pattern: np.ndarray):
        """Update concept with new instance"""
        # Update prototype (moving average)
        concept.features = (1 - self.learning_rate) * concept.features + \
                          self.learning_rate * pattern
        
        # Add instance
        concept.instances.append(pattern.copy())
        concept.frequency += 1
        
        # Strengthen concept
        concept.strength = min(1.0, concept.strength + 0.01)
    
    def form_concept(self, pattern: np.ndarray, name: Optional[str] = None) -> Concept:
        """
        Form or update concept from pattern
        
        Returns:
            Concept that matches or was created
        """
        # Try to find matching concept
        match = self.find_best_matching_concept(pattern)
        
        if match:
            concept_id, similarity = match
            concept = self.concepts[concept_id]
            self.update_concept(concept, pattern)
            return concept
        else:
            # Create new concept
            return self.create_concept(pattern, name)

class SymbolGrounding:
    """
    Symbol Grounding
    
    Grounds symbols (labels) to concepts (meanings)
    Links linguistic symbols to perceptual concepts
    """
    
    def __init__(self,
                 grounding_threshold: float = 0.8):
        self.grounding_threshold = grounding_threshold
        
        self.symbols: Dict[int, Symbol] = {}
        self.symbol_to_concept: Dict[str, int] = {}  # label -> concept_id
        self.next_symbol_id = 0
    
    def ground_symbol(self,
                     label: str,
                     concept_id: int,
                     pattern: np.ndarray,
                     concept_features: np.ndarray) -> Symbol:
        """
        Ground a symbol to a concept
        
        Returns:
            Created or updated symbol
        """
        # Check if symbol already exists
        if label in self.symbol_to_concept:
            symbol_id = self.symbol_to_concept[label]
            symbol = self.symbols[symbol_id]
            
            # Update grounding if concept matches
            if symbol.concept_id == concept_id:
                symbol.grounding_strength = min(1.0, symbol.grounding_strength + 0.1)
            else:
                # Re-ground to new concept if better match
                similarity = np.dot(pattern, concept_features) / (
                    np.linalg.norm(pattern) * np.linalg.norm(concept_features) + 1e-10
                )
                if similarity > self.grounding_threshold:
                    symbol.concept_id = concept_id
                    symbol.grounding_strength = similarity
            
            symbol.usage_count += 1
            return symbol
        
        # Create new symbol
        similarity = np.dot(pattern, concept_features) / (
            np.linalg.norm(pattern) * np.linalg.norm(concept_features) + 1e-10
        )
        
        symbol = Symbol(
            symbol_id=self.next_symbol_id,
            label=label,
            concept_id=concept_id,
            grounding_strength=similarity,
            usage_count=1
        )
        
        self.symbols[self.next_symbol_id] = symbol
        self.symbol_to_concept[label] = self.next_symbol_id
        self.next_symbol_id += 1
        
        return symbol
    
    def get_concept_for_symbol(self, label: str) -> Optional[int]:
        """Get concept ID for a symbol"""
        if label in self.symbol_to_concept:
            symbol_id = self.symbol_to_concept[label]
            return self.symbols[symbol_id].concept_id
        return None
    
    def get_symbols_for_concept(self, concept_id: int) -> List[str]:
        """Get all symbols grounded to a concept"""
        symbols = []
        for symbol in self.symbols.values():
            if symbol.concept_id == concept_id:
                symbols.append(symbol.label)
        return symbols

class MeaningExtraction:
    """
    Meaning Extraction
    
    Extracts meaning from patterns and relationships
    Builds semantic understanding
    """
    
    def __init__(self):
        self.meanings: Dict[int, Dict] = {}  # concept_id -> meaning dict
        self.context_vectors: Dict[int, np.ndarray] = {}  # concept_id -> context
    
    def extract_meaning(self,
                       concept: Concept,
                       context: Optional[np.ndarray] = None) -> Dict:
        """
        Extract meaning from concept
        
        Returns:
            Dictionary with meaning components
        """
        meaning = {
            'concept_id': concept.concept_id,
            'name': concept.name,
            'prototype': concept.features.copy(),
            'frequency': concept.frequency,
            'strength': concept.strength,
            'num_instances': len(concept.instances),
            'variability': self._compute_variability(concept),
            'context': context.copy() if context is not None else None
        }
        
        self.meanings[concept.concept_id] = meaning
        
        if context is not None:
            self.context_vectors[concept.concept_id] = context.copy()
        
        return meaning
    
    def _compute_variability(self, concept: Concept) -> float:
        """Compute variability of concept instances"""
        if len(concept.instances) < 2:
            return 0.0
        
        variances = []
        for instance in concept.instances:
            diff = instance - concept.features
            variances.append(np.var(diff))
        
        return np.mean(variances)
    
    def compute_semantic_similarity(self,
                                   concept1_id: int,
                                   concept2_id: int) -> float:
        """Compute semantic similarity between concepts"""
        if concept1_id not in self.meanings or concept2_id not in self.meanings:
            return 0.0
        
        meaning1 = self.meanings[concept1_id]
        meaning2 = self.meanings[concept2_id]
        
        # Compare prototypes
        prototype_sim = np.dot(meaning1['prototype'], meaning2['prototype']) / (
            np.linalg.norm(meaning1['prototype']) * np.linalg.norm(meaning2['prototype']) + 1e-10
        )
        
        # Compare contexts if available
        context_sim = 0.0
        if concept1_id in self.context_vectors and concept2_id in self.context_vectors:
            ctx1 = self.context_vectors[concept1_id]
            ctx2 = self.context_vectors[concept2_id]
            context_sim = np.dot(ctx1, ctx2) / (
                np.linalg.norm(ctx1) * np.linalg.norm(ctx2) + 1e-10
            )
        
        return 0.7 * prototype_sim + 0.3 * context_sim

class SemanticNetwork:
    """
    Semantic Network
    
    Represents concepts and their relationships
    Builds knowledge graph
    """
    
    def __init__(self):
        self.relations: List[SemanticRelation] = []
        self.concept_relations: Dict[int, List[SemanticRelation]] = defaultdict(list)
    
    def add_relation(self,
                    from_concept_id: int,
                    to_concept_id: int,
                    relation_type: str,
                    strength: float = 1.0):
        """Add or update semantic relation"""
        # Check if relation exists
        existing = None
        for rel in self.concept_relations[from_concept_id]:
            if rel.to_concept_id == to_concept_id and rel.relation_type == relation_type:
                existing = rel
                break
        
        if existing:
            # Update existing relation
            existing.strength = min(1.0, existing.strength + 0.1)
            existing.frequency += 1
        else:
            # Create new relation
            relation = SemanticRelation(
                from_concept_id=from_concept_id,
                to_concept_id=to_concept_id,
                relation_type=relation_type,
                strength=strength
            )
            self.relations.append(relation)
            self.concept_relations[from_concept_id].append(relation)
    
    def infer_relation(self,
                      concept1_id: int,
                      concept2_id: int,
                      meaning_extractor: MeaningExtraction) -> Optional[str]:
        """
        Infer relationship between concepts
        
        Returns:
            Inferred relation type or None
        """
        similarity = meaning_extractor.compute_semantic_similarity(concept1_id, concept2_id)
        
        if similarity > 0.8:
            return 'similar_to'
        elif similarity > 0.6:
            return 'related_to'
        else:
            return None
    
    def get_related_concepts(self,
                           concept_id: int,
                           relation_type: Optional[str] = None) -> List[int]:
        """Get concepts related to given concept"""
        related = []
        for rel in self.concept_relations[concept_id]:
            if relation_type is None or rel.relation_type == relation_type:
                related.append(rel.to_concept_id)
        return related
    
    def get_statistics(self) -> Dict:
        """Get network statistics"""
        relation_types = defaultdict(int)
        for rel in self.relations:
            relation_types[rel.relation_type] += 1
        
        return {
            'total_relations': len(self.relations),
            'relation_types': dict(relation_types),
            'avg_relation_strength': np.mean([r.strength for r in self.relations]) if self.relations else 0.0
        }

class SemanticRepresentationManager:
    """
    Manages all semantic representation mechanisms
    """
    
    def __init__(self):
        self.concept_formation = ConceptFormation()
        self.symbol_grounding = SymbolGrounding()
        self.meaning_extraction = MeaningExtraction()
        self.semantic_network = SemanticNetwork()
    
    def learn_concept(self,
                    pattern: np.ndarray,
                    label: Optional[str] = None,
                    context: Optional[np.ndarray] = None) -> Concept:
        """
        Learn a concept from pattern
        
        Returns:
            Concept that was formed or updated
        """
        # Form concept
        concept = self.concept_formation.form_concept(pattern, name=label)
        
        # Ground symbol if label provided
        if label:
            self.symbol_grounding.ground_symbol(
                label, concept.concept_id, pattern, concept.features
            )
        
        # Extract meaning
        self.meaning_extraction.extract_meaning(concept, context)
        
        return concept
    
    def relate_concepts(self,
                      concept1_id: int,
                      concept2_id: int,
                      relation_type: str,
                      auto_infer: bool = True):
        """Relate two concepts"""
        # Add explicit relation
        self.semantic_network.add_relation(concept1_id, concept2_id, relation_type)
        
        # Auto-infer reverse relation if similar
        if auto_infer:
            inferred = self.semantic_network.infer_relation(
                concept2_id, concept1_id, self.meaning_extraction
            )
            if inferred:
                self.semantic_network.add_relation(concept2_id, concept1_id, inferred, strength=0.5)
    
    def query_concept(self, label: str) -> Optional[Dict]:
        """Query concept by symbol"""
        concept_id = self.symbol_grounding.get_concept_for_symbol(label)
        if concept_id is None:
            return None
        
        concept = self.concept_formation.concepts[concept_id]
        meaning = self.meaning_extraction.meanings.get(concept_id, {})
        relations = self.semantic_network.get_related_concepts(concept_id)
        
        return {
            'concept': concept,
            'meaning': meaning,
            'related_concepts': relations,
            'symbols': self.symbol_grounding.get_symbols_for_concept(concept_id)
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about semantic representations"""
        return {
            'concepts': len(self.concept_formation.concepts),
            'symbols': len(self.symbol_grounding.symbols),
            'meanings': len(self.meaning_extraction.meanings),
            'semantic_network': self.semantic_network.get_statistics()
        }

