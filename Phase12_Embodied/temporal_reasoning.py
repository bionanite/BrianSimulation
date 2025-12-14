#!/usr/bin/env python3
"""
Temporal Reasoning & Episodic Memory - Phase 12.2
Implements episodic memory, temporal sequences, time perception,
event causality, and temporal planning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from collections import defaultdict

# Import dependencies
try:
    from memory_consolidation import MemoryConsolidationManager
    from world_models import WorldModelManager
    from Phase8_AdvancedReasoning.probabilistic_causal_reasoning import ProbabilisticCausalReasoning
except ImportError:
    MemoryConsolidationManager = None
    WorldModelManager = None
    ProbabilisticCausalReasoning = None


@dataclass
class EpisodicMemory:
    """Represents an episodic memory"""
    memory_id: int
    event_description: str
    timestamp: float
    location: np.ndarray
    participants: List[str]
    emotional_valence: float = 0.0
    importance: float = 0.5
    retrieved_count: int = 0


@dataclass
class TemporalSequence:
    """Represents a temporal sequence"""
    sequence_id: int
    events: List[int]  # Event IDs
    temporal_order: List[float]  # Timestamps
    causal_links: List[Tuple[int, int]]  # (cause_event_id, effect_event_id)


@dataclass
class TimePerception:
    """Represents time perception"""
    perceived_time: float
    actual_time: float
    time_dilation_factor: float = 1.0


class EpisodicMemorySystem:
    """
    Episodic Memory
    
    Stores and recalls specific events
    Maintains event memories
    """
    
    def __init__(self, memory_consolidation: Optional[MemoryConsolidationManager] = None):
        self.memory_consolidation = memory_consolidation
        self.episodic_memories: Dict[int, EpisodicMemory] = {}
        self.next_memory_id = 0
        self.memory_index: Dict[str, List[int]] = defaultdict(list)  # keyword -> memory_ids
    
    def store_episode(self,
                     event_description: str,
                     location: np.ndarray,
                     participants: List[str] = None,
                     emotional_valence: float = 0.0) -> EpisodicMemory:
        """
        Store an episodic memory
        
        Returns:
            Stored memory
        """
        if participants is None:
            participants = []
        
        memory = EpisodicMemory(
            memory_id=self.next_memory_id,
            event_description=event_description,
            timestamp=time.time(),
            location=location.copy(),
            participants=participants,
            emotional_valence=emotional_valence,
            importance=abs(emotional_valence)  # More emotional = more important
        )
        
        self.episodic_memories[self.next_memory_id] = memory
        
        # Index by keywords
        keywords = event_description.lower().split()
        for keyword in keywords:
            self.memory_index[keyword].append(self.next_memory_id)
        
        self.next_memory_id += 1
        
        return memory
    
    def recall_episode(self,
                      query: str,
                      time_range: Optional[Tuple[float, float]] = None) -> List[EpisodicMemory]:
        """
        Recall episodic memories
        
        Returns:
            List of recalled memories
        """
        # Find memories matching query
        query_keywords = query.lower().split()
        matching_memory_ids = set()
        
        for keyword in query_keywords:
            if keyword in self.memory_index:
                matching_memory_ids.update(self.memory_index[keyword])
        
        # Filter by time range
        recalled_memories = []
        for memory_id in matching_memory_ids:
            memory = self.episodic_memories[memory_id]
            
            if time_range:
                if time_range[0] <= memory.timestamp <= time_range[1]:
                    recalled_memories.append(memory)
            else:
                recalled_memories.append(memory)
            
            memory.retrieved_count += 1
        
        # Sort by importance
        recalled_memories.sort(key=lambda m: m.importance, reverse=True)
        
        return recalled_memories


class TemporalSequenceLearning:
    """
    Temporal Sequences
    
    Learns and predicts temporal sequences
    Understands event ordering
    """
    
    def __init__(self):
        self.sequences: Dict[int, TemporalSequence] = {}
        self.next_sequence_id = 0
        self.sequence_patterns: Dict[Tuple[int, ...], float] = {}  # event_pattern -> probability
    
    def learn_sequence(self,
                      events: List[int],
                      timestamps: List[float]) -> TemporalSequence:
        """
        Learn a temporal sequence
        
        Returns:
            Learned sequence
        """
        sequence = TemporalSequence(
            sequence_id=self.next_sequence_id,
            events=events.copy(),
            temporal_order=timestamps.copy(),
            causal_links=[]
        )
        
        self.sequences[self.next_sequence_id] = sequence
        self.next_sequence_id += 1
        
        # Learn sequence patterns
        if len(events) >= 2:
            pattern = tuple(events)
            if pattern not in self.sequence_patterns:
                self.sequence_patterns[pattern] = 0.0
            self.sequence_patterns[pattern] += 1.0
        
        return sequence
    
    def predict_next_event(self,
                         current_events: List[int]) -> Optional[int]:
        """
        Predict next event in sequence
        
        Returns:
            Predicted event ID or None
        """
        if not current_events:
            return None
        
        # Find matching patterns
        best_pattern = None
        best_probability = 0.0
        
        for pattern, probability in self.sequence_patterns.items():
            if len(pattern) > len(current_events):
                # Check if current events match beginning of pattern
                if pattern[:len(current_events)] == tuple(current_events):
                    if probability > best_probability:
                        best_probability = probability
                        best_pattern = pattern
        
        if best_pattern and len(best_pattern) > len(current_events):
            return best_pattern[len(current_events)]
        
        return None


class TimePerceptionSystem:
    """
    Time Perception
    
    Perceives and reasons about time
    Maintains temporal awareness
    """
    
    def __init__(self):
        self.time_perceptions: List[TimePerception] = []
        self.time_dilation_history: List[float] = []
    
    def perceive_time(self,
                     perceived_duration: float,
                     actual_duration: float) -> TimePerception:
        """
        Perceive time duration
        
        Returns:
            Time perception
        """
        dilation_factor = perceived_duration / (actual_duration + 1e-10)
        
        perception = TimePerception(
            perceived_time=perceived_duration,
            actual_time=actual_duration,
            time_dilation_factor=dilation_factor
        )
        
        self.time_perceptions.append(perception)
        self.time_dilation_history.append(dilation_factor)
        
        return perception
    
    def estimate_duration(self,
                         start_time: float,
                         end_time: float) -> float:
        """Estimate duration between times"""
        return end_time - start_time
    
    def get_time_dilation_trend(self) -> float:
        """Get average time dilation factor"""
        if not self.time_dilation_history:
            return 1.0
        
        return np.mean(self.time_dilation_history[-10:])


class EventCausality:
    """
    Event Causality
    
    Understands causal relationships in time
    Identifies cause-effect chains
    """
    
    def __init__(self, causal_reasoning: Optional[ProbabilisticCausalReasoning] = None):
        self.causal_reasoning = causal_reasoning
        self.causal_chains: List[List[int]] = []  # List of event chains
        self.causal_relationships: Dict[Tuple[int, int], float] = {}  # (cause, effect) -> strength
    
    def identify_causality(self,
                          event1_id: int,
                          event2_id: int,
                          temporal_gap: float) -> float:
        """
        Identify causal relationship between events
        
        Returns:
            Causal strength
        """
        # Causal strength decreases with temporal gap
        base_strength = 1.0 / (1.0 + temporal_gap)
        
        # Use causal reasoning if available
        if self.causal_reasoning:
            # Simplified: use temporal proximity as proxy
            causal_strength = base_strength
        else:
            causal_strength = base_strength
        
        self.causal_relationships[(event1_id, event2_id)] = causal_strength
        
        return causal_strength
    
    def build_causal_chain(self, events: List[int], timestamps: List[float]) -> List[int]:
        """
        Build causal chain from events
        
        Returns:
            Ordered causal chain
        """
        if len(events) < 2:
            return events
        
        # Identify causal relationships
        causal_links = []
        for i in range(len(events) - 1):
            temporal_gap = timestamps[i+1] - timestamps[i]
            strength = self.identify_causality(events[i], events[i+1], temporal_gap)
            
            if strength > 0.3:  # Threshold
                causal_links.append((events[i], events[i+1]))
        
        # Build chain following causal links
        chain = [events[0]]
        current_event = events[0]
        
        while True:
            next_event = None
            best_strength = 0.0
            
            for cause, effect in causal_links:
                if cause == current_event:
                    strength = self.causal_relationships.get((cause, effect), 0.0)
                    if strength > best_strength:
                        best_strength = strength
                        next_event = effect
            
            if next_event and next_event not in chain:
                chain.append(next_event)
                current_event = next_event
            else:
                break
        
        return chain


class TemporalPlanning:
    """
    Temporal Planning
    
    Plans actions across time
    Sequences actions temporally
    """
    
    def __init__(self, world_model: Optional[WorldModelManager] = None):
        self.world_model = world_model
        self.temporal_plans: List[Dict] = []
    
    def create_temporal_plan(self,
                            actions: List[int],
                            time_constraints: List[Tuple[float, float]] = None) -> Dict:
        """
        Create temporal plan
        
        Returns:
            Temporal plan
        """
        if time_constraints is None:
            time_constraints = [(0.0, float('inf'))] * len(actions)
        
        plan = {
            'actions': actions.copy(),
            'time_constraints': time_constraints,
            'scheduled_times': [],
            'created_time': time.time()
        }
        
        # Schedule actions
        current_time = time.time()
        for i, action_id in enumerate(actions):
            start_time = current_time + i * 1.0  # 1 second between actions
            end_time = start_time + 1.0
            plan['scheduled_times'].append((start_time, end_time))
        
        self.temporal_plans.append(plan)
        return plan
    
    def check_temporal_feasibility(self, plan: Dict) -> bool:
        """Check if plan is temporally feasible"""
        scheduled_times = plan['scheduled_times']
        
        # Check for overlaps
        for i in range(len(scheduled_times) - 1):
            if scheduled_times[i][1] > scheduled_times[i+1][0]:
                return False
        
        return True


class TemporalReasoningSystem:
    """
    Temporal Reasoning System Manager
    
    Integrates all temporal reasoning components
    """
    
    def __init__(self,
                 brain_system=None,
                 memory_consolidation: Optional[MemoryConsolidationManager] = None,
                 world_model: Optional[WorldModelManager] = None,
                 causal_reasoning: Optional[ProbabilisticCausalReasoning] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.episodic_memory = EpisodicMemorySystem(memory_consolidation)
        self.temporal_sequences = TemporalSequenceLearning()
        self.time_perception = TimePerceptionSystem()
        self.event_causality = EventCausality(causal_reasoning)
        self.temporal_planning = TemporalPlanning(world_model)
        
        # Integration with existing systems
        self.memory_consolidation = memory_consolidation
        self.world_model = world_model
        self.causal_reasoning = causal_reasoning
        
        # Statistics
        self.stats = {
            'episodes_stored': 0,
            'sequences_learned': 0,
            'time_perceptions': 0,
            'causal_chains_built': 0,
            'temporal_plans_created': 0
        }
    
    def store_episode(self,
                     event_description: str,
                     location: np.ndarray,
                     participants: List[str] = None) -> EpisodicMemory:
        """Store episodic memory"""
        memory = self.episodic_memory.store_episode(
            event_description, location, participants
        )
        self.stats['episodes_stored'] += 1
        return memory
    
    def learn_sequence(self,
                     events: List[int],
                     timestamps: List[float]) -> TemporalSequence:
        """Learn temporal sequence"""
        sequence = self.temporal_sequences.learn_sequence(events, timestamps)
        self.stats['sequences_learned'] += 1
        return sequence
    
    def perceive_time(self,
                     perceived_duration: float,
                     actual_duration: float) -> TimePerception:
        """Perceive time"""
        perception = self.time_perception.perceive_time(perceived_duration, actual_duration)
        self.stats['time_perceptions'] += 1
        return perception
    
    def build_causal_chain(self,
                         events: List[int],
                         timestamps: List[float]) -> List[int]:
        """Build causal chain"""
        chain = self.event_causality.build_causal_chain(events, timestamps)
        self.stats['causal_chains_built'] += 1
        return chain
    
    def create_temporal_plan(self,
                            actions: List[int],
                            time_constraints: List[Tuple[float, float]] = None) -> Dict:
        """Create temporal plan"""
        plan = self.temporal_planning.create_temporal_plan(actions, time_constraints)
        self.stats['temporal_plans_created'] += 1
        return plan
    
    def get_statistics(self) -> Dict:
        """Get temporal reasoning statistics"""
        return self.stats.copy()

