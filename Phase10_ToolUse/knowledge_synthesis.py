#!/usr/bin/env python3
"""
Knowledge Integration & Synthesis - Phase 10.3
Implements cross-domain integration, knowledge synthesis, contradiction resolution,
knowledge validation, and knowledge graph construction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
from collections import defaultdict

# Import dependencies
try:
    from semantic_representations import SemanticNetwork, ConceptFormation, Concept
    from global_workspace import GlobalWorkspace
    from memory_consolidation import MemoryConsolidationManager
except ImportError:
    SemanticNetwork = None
    ConceptFormation = None
    Concept = None
    GlobalWorkspace = None
    MemoryConsolidationManager = None


@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge"""
    knowledge_id: int
    content: str
    domain: str
    confidence: float = 0.5
    source: str = 'unknown'
    created_time: float = 0.0


@dataclass
class Contradiction:
    """Represents a contradiction between knowledge items"""
    contradiction_id: int
    knowledge1_id: int
    knowledge2_id: int
    conflict_type: str  # 'direct', 'implicit', 'temporal'
    severity: float = 0.5
    resolved: bool = False


class CrossDomainIntegration:
    """
    Cross-Domain Integration
    
    Integrates knowledge across domains
    Finds connections between domains
    """
    
    def __init__(self):
        self.domain_knowledge: Dict[str, List[KnowledgeItem]] = defaultdict(list)
        self.cross_domain_connections: List[Tuple[str, str, float]] = []  # (domain1, domain2, strength)
    
    def add_knowledge(self,
                     knowledge: KnowledgeItem):
        """Add knowledge to domain"""
        self.domain_knowledge[knowledge.domain].append(knowledge)
    
    def find_cross_domain_connections(self,
                                     domain1: str,
                                     domain2: str,
                                     semantic_network: Optional[SemanticNetwork] = None) -> float:
        """
        Find connections between domains
        
        Returns:
            Connection strength
        """
        knowledge1 = self.domain_knowledge.get(domain1, [])
        knowledge2 = self.domain_knowledge.get(domain2, [])
        
        if not knowledge1 or not knowledge2:
            return 0.0
        
        # Compute similarity between domains
        # Simplified: count shared concepts
        shared_concepts = 0
        total_concepts = len(knowledge1) + len(knowledge2)
        
        # Simple keyword matching (in practice would use semantic similarity)
        keywords1 = set()
        keywords2 = set()
        
        for k in knowledge1:
            keywords1.update(k.content.lower().split())
        
        for k in knowledge2:
            keywords2.update(k.content.lower().split())
        
        shared_keywords = keywords1.intersection(keywords2)
        connection_strength = len(shared_keywords) / max(1, len(keywords1.union(keywords2)))
        
        self.cross_domain_connections.append((domain1, domain2, connection_strength))
        
        return connection_strength
    
    def integrate_domains(self,
                         domain1: str,
                         domain2: str) -> List[KnowledgeItem]:
        """
        Integrate knowledge from two domains
        
        Returns:
            Integrated knowledge items
        """
        knowledge1 = self.domain_knowledge.get(domain1, [])
        knowledge2 = self.domain_knowledge.get(domain2, [])
        
        integrated = []
        
        # Combine knowledge from both domains
        for k in knowledge1:
            integrated.append(k)
        
        for k in knowledge2:
            # Check if similar knowledge already exists
            is_duplicate = False
            for existing in integrated:
                if k.content.lower() == existing.content.lower():
                    is_duplicate = True
                    # Merge confidence
                    existing.confidence = max(existing.confidence, k.confidence)
                    break
            
            if not is_duplicate:
                integrated.append(k)
        
        return integrated


class KnowledgeSynthesis:
    """
    Knowledge Synthesis
    
    Synthesizes new knowledge from existing
    Creates novel insights
    """
    
    def __init__(self):
        self.synthesized_knowledge: List[KnowledgeItem] = []
    
    def synthesize(self,
                  knowledge_items: List[KnowledgeItem],
                  synthesis_method: str = 'combination') -> KnowledgeItem:
        """
        Synthesize new knowledge from existing
        
        Returns:
            Synthesized knowledge
        """
        if not knowledge_items:
            return None
        
        if synthesis_method == 'combination':
            # Combine knowledge items
            combined_content = " AND ".join([k.content for k in knowledge_items])
            avg_confidence = np.mean([k.confidence for k in knowledge_items])
            domains = list(set([k.domain for k in knowledge_items]))
            combined_domain = "_".join(domains) if len(domains) > 1 else domains[0]
        
        elif synthesis_method == 'generalization':
            # Generalize from specific knowledge
            combined_content = f"General principle from: {knowledge_items[0].content}"
            avg_confidence = knowledge_items[0].confidence * 0.8  # Lower confidence for generalization
            combined_domain = knowledge_items[0].domain
        
        else:
            # Default combination
            combined_content = " AND ".join([k.content for k in knowledge_items])
            avg_confidence = np.mean([k.confidence for k in knowledge_items])
            combined_domain = knowledge_items[0].domain
        
        synthesized = KnowledgeItem(
            knowledge_id=-1,  # Will be assigned later
            content=combined_content,
            domain=combined_domain,
            confidence=avg_confidence,
            source='synthesis',
            created_time=time.time()
        )
        
        self.synthesized_knowledge.append(synthesized)
        return synthesized
    
    def infer_new_knowledge(self,
                           knowledge_items: List[KnowledgeItem],
                           inference_rules: List[Dict]) -> List[KnowledgeItem]:
        """
        Infer new knowledge using rules
        
        Returns:
            Inferred knowledge items
        """
        inferred = []
        
        for rule in inference_rules:
            # Check if rule applies
            if self._rule_applies(rule, knowledge_items):
                # Generate inferred knowledge
                inferred_knowledge = KnowledgeItem(
                    knowledge_id=-1,
                    content=rule.get('conclusion', 'Inferred knowledge'),
                    domain=knowledge_items[0].domain if knowledge_items else 'unknown',
                    confidence=rule.get('confidence', 0.5),
                    source='inference',
                    created_time=time.time()
                )
                inferred.append(inferred_knowledge)
        
        return inferred
    
    def _rule_applies(self, rule: Dict, knowledge_items: List[KnowledgeItem]) -> bool:
        """Check if inference rule applies"""
        premises = rule.get('premises', [])
        
        # Check if all premises are satisfied
        for premise in premises:
            satisfied = any(premise.lower() in k.content.lower() for k in knowledge_items)
            if not satisfied:
                return False
        
        return True


class ContradictionResolution:
    """
    Contradiction Resolution
    
    Resolves conflicting knowledge
    Maintains knowledge consistency
    """
    
    def __init__(self):
        self.contradictions: Dict[int, Contradiction] = {}
        self.next_contradiction_id = 0
        self.resolutions: List[Dict] = []
    
    def detect_contradiction(self,
                           knowledge1: KnowledgeItem,
                           knowledge2: KnowledgeItem) -> Optional[Contradiction]:
        """
        Detect contradiction between knowledge items
        
        Returns:
            Contradiction object or None
        """
        # Simple contradiction detection (in practice would use semantic analysis)
        content1 = knowledge1.content.lower()
        content2 = knowledge2.content.lower()
        
        # Check for direct contradictions (simplified)
        contradiction_keywords = [
            ('is', 'is not'),
            ('always', 'never'),
            ('true', 'false'),
            ('yes', 'no')
        ]
        
        has_contradiction = False
        conflict_type = 'implicit'
        
        for keyword1, keyword2 in contradiction_keywords:
            if keyword1 in content1 and keyword2 in content2:
                has_contradiction = True
                conflict_type = 'direct'
                break
        
        if has_contradiction:
            contradiction = Contradiction(
                contradiction_id=self.next_contradiction_id,
                knowledge1_id=knowledge1.knowledge_id,
                knowledge2_id=knowledge2.knowledge_id,
                conflict_type=conflict_type,
                severity=abs(knowledge1.confidence - knowledge2.confidence)
            )
            
            self.contradictions[self.next_contradiction_id] = contradiction
            self.next_contradiction_id += 1
            
            return contradiction
        
        return None
    
    def resolve_contradiction(self,
                            contradiction: Contradiction,
                            knowledge_items: Dict[int, KnowledgeItem],
                            resolution_strategy: str = 'confidence') -> Dict:
        """
        Resolve a contradiction
        
        Returns:
            Resolution result
        """
        k1 = knowledge_items.get(contradiction.knowledge1_id)
        k2 = knowledge_items.get(contradiction.knowledge2_id)
        
        if not k1 or not k2:
            return {'success': False}
        
        if resolution_strategy == 'confidence':
            # Keep knowledge with higher confidence
            if k1.confidence > k2.confidence:
                resolved_knowledge = k1
                removed_knowledge = k2
            else:
                resolved_knowledge = k2
                removed_knowledge = k1
        
        elif resolution_strategy == 'recency':
            # Keep more recent knowledge
            if k1.created_time > k2.created_time:
                resolved_knowledge = k1
                removed_knowledge = k2
            else:
                resolved_knowledge = k2
                removed_knowledge = k1
        
        else:
            # Default: keep first
            resolved_knowledge = k1
            removed_knowledge = k2
        
        contradiction.resolved = True
        
        resolution = {
            'contradiction_id': contradiction.contradiction_id,
            'resolved_knowledge_id': resolved_knowledge.knowledge_id,
            'removed_knowledge_id': removed_knowledge.knowledge_id,
            'strategy': resolution_strategy,
            'timestamp': time.time()
        }
        
        self.resolutions.append(resolution)
        return resolution


class KnowledgeValidation:
    """
    Knowledge Validation
    
    Validates knowledge consistency
    Checks knowledge quality
    """
    
    def __init__(self):
        self.validation_rules: List[Callable] = []
        self.validation_history: List[Dict] = []
    
    def validate_knowledge(self,
                          knowledge: KnowledgeItem,
                          existing_knowledge: List[KnowledgeItem]) -> Dict:
        """
        Validate knowledge item
        
        Returns:
            Validation results
        """
        validation_result = {
            'knowledge_id': knowledge.knowledge_id,
            'is_valid': True,
            'issues': [],
            'confidence_adjustment': 0.0
        }
        
        # Check consistency with existing knowledge
        contradictions = []
        for existing in existing_knowledge:
            if existing.knowledge_id == knowledge.knowledge_id:
                continue
            
            # Simple consistency check
            if existing.content.lower() == knowledge.content.lower():
                if abs(existing.confidence - knowledge.confidence) > 0.3:
                    contradictions.append(existing.knowledge_id)
        
        if contradictions:
            validation_result['issues'].append('contradictions_detected')
            validation_result['confidence_adjustment'] = -0.1
        
        # Check confidence level
        if knowledge.confidence < 0.3:
            validation_result['issues'].append('low_confidence')
            validation_result['is_valid'] = False
        
        # Check domain consistency
        if knowledge.domain == 'unknown':
            validation_result['issues'].append('unknown_domain')
        
        self.validation_history.append(validation_result)
        return validation_result


class KnowledgeGraphConstruction:
    """
    Knowledge Graph Construction
    
    Builds comprehensive knowledge graphs
    Represents knowledge relationships
    """
    
    def __init__(self):
        self.knowledge_graph: Dict[int, Set[int]] = {}  # knowledge_id -> related_ids
        self.relation_types: Dict[Tuple[int, int], str] = {}  # (id1, id2) -> relation_type
    
    def add_knowledge_node(self, knowledge_id: int):
        """Add a knowledge node to graph"""
        if knowledge_id not in self.knowledge_graph:
            self.knowledge_graph[knowledge_id] = set()
    
    def add_relation(self,
                    knowledge1_id: int,
                    knowledge2_id: int,
                    relation_type: str = 'related_to'):
        """Add relation between knowledge items"""
        self.add_knowledge_node(knowledge1_id)
        self.add_knowledge_node(knowledge2_id)
        
        self.knowledge_graph[knowledge1_id].add(knowledge2_id)
        self.knowledge_graph[knowledge2_id].add(knowledge1_id)
        
        self.relation_types[(knowledge1_id, knowledge2_id)] = relation_type
        self.relation_types[(knowledge2_id, knowledge1_id)] = relation_type
    
    def build_graph_from_knowledge(self,
                                  knowledge_items: List[KnowledgeItem],
                                  semantic_network: Optional[SemanticNetwork] = None):
        """Build knowledge graph from knowledge items"""
        # Add all nodes
        for knowledge in knowledge_items:
            self.add_knowledge_node(knowledge.knowledge_id)
        
        # Add relations based on similarity
        for i, k1 in enumerate(knowledge_items):
            for k2 in knowledge_items[i+1:]:
                # Compute similarity (simplified)
                similarity = self._compute_similarity(k1, k2)
                
                if similarity > 0.5:
                    self.add_relation(k1.knowledge_id, k2.knowledge_id, 'similar_to')
                elif similarity > 0.3:
                    self.add_relation(k1.knowledge_id, k2.knowledge_id, 'related_to')
    
    def _compute_similarity(self, k1: KnowledgeItem, k2: KnowledgeItem) -> float:
        """Compute similarity between knowledge items"""
        # Simple keyword-based similarity
        words1 = set(k1.content.lower().split())
        words2 = set(k2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_related_knowledge(self, knowledge_id: int) -> List[int]:
        """Get related knowledge items"""
        return list(self.knowledge_graph.get(knowledge_id, set()))


class KnowledgeSynthesisSystem:
    """
    Knowledge Synthesis System Manager
    
    Integrates all knowledge synthesis components
    """
    
    def __init__(self,
                 brain_system=None,
                 semantic_network: Optional[SemanticNetwork] = None,
                 concept_formation: Optional[ConceptFormation] = None,
                 global_workspace: Optional[GlobalWorkspace] = None,
                 memory_consolidation: Optional[MemoryConsolidationManager] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.cross_domain_integration = CrossDomainIntegration()
        self.knowledge_synthesis = KnowledgeSynthesis()
        self.contradiction_resolution = ContradictionResolution()
        self.knowledge_validation = KnowledgeValidation()
        self.knowledge_graph = KnowledgeGraphConstruction()
        
        # Integration with existing systems
        self.semantic_network = semantic_network
        self.concept_formation = concept_formation
        self.global_workspace = global_workspace
        self.memory_consolidation = memory_consolidation
        
        # Knowledge tracking
        self.knowledge_items: Dict[int, KnowledgeItem] = {}
        self.next_knowledge_id = 0
        
        # Statistics
        self.stats = {
            'knowledge_items_added': 0,
            'cross_domain_connections': 0,
            'knowledge_synthesized': 0,
            'contradictions_resolved': 0,
            'graph_nodes': 0
        }
    
    def add_knowledge(self,
                     content: str,
                     domain: str,
                     confidence: float = 0.5) -> KnowledgeItem:
        """Add knowledge to system"""
        knowledge = KnowledgeItem(
            knowledge_id=self.next_knowledge_id,
            content=content,
            domain=domain,
            confidence=confidence,
            created_time=time.time()
        )
        
        self.knowledge_items[self.next_knowledge_id] = knowledge
        self.next_knowledge_id += 1
        
        # Add to domain integration
        self.cross_domain_integration.add_knowledge(knowledge)
        
        # Validate
        validation = self.knowledge_validation.validate_knowledge(
            knowledge, list(self.knowledge_items.values())
        )
        
        if validation['is_valid']:
            # Add to knowledge graph
            self.knowledge_graph.add_knowledge_node(knowledge.knowledge_id)
            self.stats['knowledge_items_added'] += 1
            self.stats['graph_nodes'] += 1
        
        return knowledge
    
    def synthesize_knowledge(self,
                            knowledge_ids: List[int],
                            synthesis_method: str = 'combination') -> Optional[KnowledgeItem]:
        """Synthesize knowledge from multiple items"""
        knowledge_items = [self.knowledge_items[kid] for kid in knowledge_ids if kid in self.knowledge_items]
        
        if not knowledge_items:
            return None
        
        synthesized = self.knowledge_synthesis.synthesize(knowledge_items, synthesis_method)
        
        if synthesized:
            synthesized.knowledge_id = self.next_knowledge_id
            self.next_knowledge_id += 1
            self.knowledge_items[synthesized.knowledge_id] = synthesized
            self.stats['knowledge_synthesized'] += 1
        
        return synthesized
    
    def resolve_all_contradictions(self) -> List[Dict]:
        """Detect and resolve all contradictions"""
        resolutions = []
        
        # Detect contradictions
        knowledge_list = list(self.knowledge_items.values())
        for i, k1 in enumerate(knowledge_list):
            for k2 in knowledge_list[i+1:]:
                contradiction = self.contradiction_resolution.detect_contradiction(k1, k2)
                
                if contradiction:
                    # Resolve
                    resolution = self.contradiction_resolution.resolve_contradiction(
                        contradiction, self.knowledge_items
                    )
                    resolutions.append(resolution)
                    self.stats['contradictions_resolved'] += 1
        
        return resolutions
    
    def build_knowledge_graph(self):
        """Build comprehensive knowledge graph"""
        knowledge_list = list(self.knowledge_items.values())
        self.knowledge_graph.build_graph_from_knowledge(
            knowledge_list, self.semantic_network
        )
        self.stats['graph_nodes'] = len(self.knowledge_graph.knowledge_graph)
    
    def get_statistics(self) -> Dict:
        """Get knowledge synthesis statistics"""
        return self.stats.copy()

