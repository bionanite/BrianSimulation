#!/usr/bin/env python3
"""
Long-Term Memory Systems - Phase 12.3
Implements autobiographical memory, semantic memory enhancement,
memory retrieval, memory organization, and memory forgetting
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from collections import defaultdict

# Import dependencies
try:
    from memory_consolidation import MemoryConsolidationManager
    from semantic_representations import SemanticNetwork, Concept
    from hierarchical_learning import HierarchicalFeatureLearning
except ImportError:
    MemoryConsolidationManager = None
    SemanticNetwork = None
    Concept = None
    HierarchicalFeatureLearning = None


@dataclass
class AutobiographicalMemory:
    """Represents an autobiographical memory"""
    memory_id: int
    event_description: str
    timestamp: float
    location: str
    emotional_significance: float = 0.5
    personal_importance: float = 0.5
    associated_concepts: List[str] = field(default_factory=list)
    retrieval_count: int = 0
    last_retrieved: float = 0.0


@dataclass
class SemanticMemoryEntry:
    """Represents a semantic memory entry"""
    entry_id: int
    concept: str
    definition: str
    properties: Dict[str, any]
    relationships: List[Tuple[str, str]]  # (relation_type, related_concept)
    strength: float = 1.0
    last_accessed: float = 0.0


class AutobiographicalMemorySystem:
    """
    Autobiographical Memory
    
    Stores personal life events
    Maintains life narrative
    """
    
    def __init__(self):
        self.memories: Dict[int, AutobiographicalMemory] = {}
        self.next_memory_id = 0
        self.life_timeline: List[int] = []  # Ordered memory IDs
    
    def store_life_event(self,
                       event_description: str,
                       location: str,
                       emotional_significance: float = 0.5,
                       personal_importance: float = 0.5) -> AutobiographicalMemory:
        """
        Store a life event
        
        Returns:
            Stored memory
        """
        memory = AutobiographicalMemory(
            memory_id=self.next_memory_id,
            event_description=event_description,
            timestamp=time.time(),
            location=location,
            emotional_significance=emotional_significance,
            personal_importance=personal_importance
        )
        
        self.memories[self.next_memory_id] = memory
        self.life_timeline.append(self.next_memory_id)
        self.next_memory_id += 1
        
        return memory
    
    def retrieve_life_events(self,
                           time_range: Optional[Tuple[float, float]] = None,
                           location: Optional[str] = None) -> List[AutobiographicalMemory]:
        """
        Retrieve life events
        
        Returns:
            List of memories
        """
        retrieved = []
        
        for memory_id in self.life_timeline:
            memory = self.memories[memory_id]
            
            # Filter by time range
            if time_range:
                if not (time_range[0] <= memory.timestamp <= time_range[1]):
                    continue
            
            # Filter by location
            if location:
                if memory.location.lower() != location.lower():
                    continue
            
            retrieved.append(memory)
            memory.retrieval_count += 1
            memory.last_retrieved = time.time()
        
        return retrieved
    
    def get_life_narrative(self) -> str:
        """Get life narrative as text"""
        narrative_parts = []
        
        for memory_id in self.life_timeline[:10]:  # First 10 events
            memory = self.memories[memory_id]
            narrative_parts.append(f"At {memory.location}: {memory.event_description}")
        
        return ". ".join(narrative_parts)


class SemanticMemoryEnhancement:
    """
    Semantic Memory Enhancement
    
    Enhanced semantic memory storage
    Rich concept representations
    """
    
    def __init__(self, semantic_network: Optional[SemanticNetwork] = None):
        self.semantic_network = semantic_network
        self.semantic_entries: Dict[int, SemanticMemoryEntry] = {}
        self.next_entry_id = 0
        self.concept_index: Dict[str, int] = {}  # concept -> entry_id
    
    def store_semantic_knowledge(self,
                                 concept: str,
                                 definition: str,
                                 properties: Dict[str, any] = None,
                                 relationships: List[Tuple[str, str]] = None) -> SemanticMemoryEntry:
        """
        Store semantic knowledge
        
        Returns:
            Stored entry
        """
        if properties is None:
            properties = {}
        if relationships is None:
            relationships = []
        
        entry = SemanticMemoryEntry(
            entry_id=self.next_entry_id,
            concept=concept,
            definition=definition,
            properties=properties,
            relationships=relationships,
            last_accessed=time.time()
        )
        
        self.semantic_entries[self.next_entry_id] = entry
        self.concept_index[concept] = self.next_entry_id
        self.next_entry_id += 1
        
        return entry
    
    def retrieve_semantic_knowledge(self, concept: str) -> Optional[SemanticMemoryEntry]:
        """Retrieve semantic knowledge"""
        if concept in self.concept_index:
            entry_id = self.concept_index[concept]
            entry = self.semantic_entries[entry_id]
            entry.last_accessed = time.time()
            return entry
        
        return None
    
    def enhance_semantic_entry(self,
                              entry_id: int,
                              new_properties: Dict[str, any] = None,
                              new_relationships: List[Tuple[str, str]] = None):
        """Enhance existing semantic entry"""
        if entry_id not in self.semantic_entries:
            return
        
        entry = self.semantic_entries[entry_id]
        
        if new_properties:
            entry.properties.update(new_properties)
        
        if new_relationships:
            entry.relationships.extend(new_relationships)


class MemoryRetrieval:
    """
    Memory Retrieval
    
    Efficient memory search and retrieval
    Cue-based retrieval
    """
    
    def __init__(self):
        self.retrieval_history: List[Dict] = []
    
    def retrieve_by_cue(self,
                       cue: str,
                       autobiographical_memories: Dict[int, AutobiographicalMemory],
                       semantic_entries: Dict[int, SemanticMemoryEntry]) -> Dict:
        """
        Retrieve memories by cue
        
        Returns:
            Retrieval results
        """
        cue_lower = cue.lower()
        
        # Search autobiographical memories
        autobiographical_results = []
        for memory in autobiographical_memories.values():
            if cue_lower in memory.event_description.lower():
                autobiographical_results.append(memory)
        
        # Search semantic entries
        semantic_results = []
        for entry in semantic_entries.values():
            if cue_lower in entry.concept.lower() or cue_lower in entry.definition.lower():
                semantic_results.append(entry)
        
        results = {
            'cue': cue,
            'autobiographical_memories': autobiographical_results,
            'semantic_entries': semantic_results,
            'total_results': len(autobiographical_results) + len(semantic_results),
            'timestamp': time.time()
        }
        
        self.retrieval_history.append(results)
        return results
    
    def retrieve_by_association(self,
                               concept: str,
                               semantic_entries: Dict[int, SemanticMemoryEntry],
                               max_depth: int = 2) -> List[SemanticMemoryEntry]:
        """
        Retrieve by association
        
        Returns:
            Associated entries
        """
        if concept not in [e.concept for e in semantic_entries.values()]:
            return []
        
        # Find entry
        entry = None
        for e in semantic_entries.values():
            if e.concept == concept:
                entry = e
                break
        
        if not entry:
            return []
        
        # Find related entries
        associated = [entry]
        visited = {entry.entry_id}
        
        for depth in range(max_depth):
            new_associated = []
            for e in associated:
                for rel_type, related_concept in e.relationships:
                    # Find related entry
                    for related_entry in semantic_entries.values():
                        if related_entry.concept == related_concept and related_entry.entry_id not in visited:
                            new_associated.append(related_entry)
                            visited.add(related_entry.entry_id)
            
            associated.extend(new_associated)
        
        return associated


class MemoryOrganization:
    """
    Memory Organization
    
    Organizes memories hierarchically
    Creates memory structures
    """
    
    def __init__(self, hierarchical_learner: Optional[HierarchicalFeatureLearning] = None):
        self.hierarchical_learner = hierarchical_learner
        self.memory_hierarchies: Dict[str, Dict] = {}
    
    def organize_hierarchically(self,
                               memories: List[AutobiographicalMemory],
                               semantic_entries: List[SemanticMemoryEntry]) -> Dict:
        """
        Organize memories hierarchically
        
        Returns:
            Hierarchical organization
        """
        # Create hierarchy by time periods
        time_periods = defaultdict(list)
        
        for memory in memories:
            # Categorize by time period (simplified)
            age = time.time() - memory.timestamp
            if age < 86400:  # 1 day
                period = 'recent'
            elif age < 604800:  # 1 week
                period = 'this_week'
            elif age < 2592000:  # 1 month
                period = 'this_month'
            else:
                period = 'older'
            
            time_periods[period].append(memory.memory_id)
        
        # Create hierarchy by importance
        importance_levels = defaultdict(list)
        for memory in memories:
            if memory.personal_importance > 0.8:
                level = 'high'
            elif memory.personal_importance > 0.5:
                level = 'medium'
            else:
                level = 'low'
            
            importance_levels[level].append(memory.memory_id)
        
        organization = {
            'by_time': dict(time_periods),
            'by_importance': dict(importance_levels),
            'total_memories': len(memories),
            'total_semantic_entries': len(semantic_entries)
        }
        
        self.memory_hierarchies['default'] = organization
        return organization


class MemoryForgetting:
    """
    Memory Forgetting
    
    Intelligent forgetting mechanisms
    Forgets less important memories
    """
    
    def __init__(self):
        self.forgetting_history: List[Dict] = []
    
    def forget_memories(self,
                       memories: Dict[int, AutobiographicalMemory],
                       forgetting_criterion: str = 'importance') -> List[int]:
        """
        Forget memories based on criterion
        
        Returns:
            List of forgotten memory IDs
        """
        forgotten_ids = []
        
        if forgetting_criterion == 'importance':
            # Forget low-importance memories
            for memory_id, memory in memories.items():
                if memory.personal_importance < 0.3:
                    forgotten_ids.append(memory_id)
        
        elif forgetting_criterion == 'recency':
            # Forget old, rarely accessed memories
            current_time = time.time()
            for memory_id, memory in memories.items():
                age = current_time - memory.timestamp
                if age > 31536000 and memory.retrieval_count < 2:  # 1 year old, rarely accessed
                    forgotten_ids.append(memory_id)
        
        elif forgetting_criterion == 'emotional':
            # Forget low-emotional-significance memories
            for memory_id, memory in memories.items():
                if memory.emotional_significance < 0.2:
                    forgotten_ids.append(memory_id)
        
        # Record forgetting
        for memory_id in forgotten_ids:
            self.forgetting_history.append({
                'memory_id': memory_id,
                'criterion': forgetting_criterion,
                'timestamp': time.time()
            })
        
        return forgotten_ids
    
    def decay_memory_strength(self,
                            entry: SemanticMemoryEntry,
                            decay_rate: float = 0.01) -> SemanticMemoryEntry:
        """Decay memory strength over time"""
        time_since_access = time.time() - entry.last_accessed
        decay = decay_rate * time_since_access / 86400  # Per day
        entry.strength = max(0.0, entry.strength - decay)
        return entry


class LongTermMemorySystem:
    """
    Long-Term Memory System Manager
    
    Integrates all long-term memory components
    """
    
    def __init__(self,
                 brain_system=None,
                 memory_consolidation: Optional[MemoryConsolidationManager] = None,
                 semantic_network: Optional[SemanticNetwork] = None,
                 hierarchical_learner: Optional[HierarchicalFeatureLearning] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.autobiographical_memory = AutobiographicalMemorySystem()
        self.semantic_enhancement = SemanticMemoryEnhancement(semantic_network)
        self.memory_retrieval = MemoryRetrieval()
        self.memory_organization = MemoryOrganization(hierarchical_learner)
        self.memory_forgetting = MemoryForgetting()
        
        # Integration with existing systems
        self.memory_consolidation = memory_consolidation
        self.semantic_network = semantic_network
        self.hierarchical_learner = hierarchical_learner
        
        # Statistics
        self.stats = {
            'life_events_stored': 0,
            'semantic_entries_stored': 0,
            'memories_retrieved': 0,
            'memories_organized': 0,
            'memories_forgotten': 0
        }
    
    def store_life_event(self,
                       event_description: str,
                       location: str,
                       importance: float = 0.5) -> AutobiographicalMemory:
        """Store life event"""
        memory = self.autobiographical_memory.store_life_event(
            event_description, location, personal_importance=importance
        )
        self.stats['life_events_stored'] += 1
        return memory
    
    def store_semantic_knowledge(self,
                               concept: str,
                               definition: str,
                               properties: Dict[str, any] = None) -> SemanticMemoryEntry:
        """Store semantic knowledge"""
        entry = self.semantic_enhancement.store_semantic_knowledge(
            concept, definition, properties
        )
        self.stats['semantic_entries_stored'] += 1
        return entry
    
    def retrieve_memories(self, cue: str) -> Dict:
        """Retrieve memories by cue"""
        results = self.memory_retrieval.retrieve_by_cue(
            cue,
            self.autobiographical_memory.memories,
            self.semantic_enhancement.semantic_entries
        )
        self.stats['memories_retrieved'] += results['total_results']
        return results
    
    def organize_memories(self) -> Dict:
        """Organize memories hierarchically"""
        memories = list(self.autobiographical_memory.memories.values())
        entries = list(self.semantic_enhancement.semantic_entries.values())
        organization = self.memory_organization.organize_hierarchically(memories, entries)
        self.stats['memories_organized'] += 1
        return organization
    
    def forget_old_memories(self, criterion: str = 'importance') -> List[int]:
        """Forget memories"""
        forgotten = self.memory_forgetting.forget_memories(
            self.autobiographical_memory.memories, criterion
        )
        self.stats['memories_forgotten'] += len(forgotten)
        return forgotten
    
    def get_statistics(self) -> Dict:
        """Get long-term memory statistics"""
        return self.stats.copy()

