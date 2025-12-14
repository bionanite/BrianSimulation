#!/usr/bin/env python3
"""
Memory Consolidation Mechanisms
Implements sleep-like consolidation, memory reconsolidation, and forgetting
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

@dataclass
class MemoryTrace:
    """Represents a memory trace"""
    memory_id: int
    content: np.ndarray
    strength: float = 1.0
    age: float = 0.0  # Time since creation
    access_count: int = 0
    last_access_time: float = 0.0
    consolidation_level: float = 0.0  # 0 = working memory, 1 = fully consolidated
    importance: float = 0.5  # Subjective importance
    
@dataclass
class ConsolidationEvent:
    """Represents a consolidation event"""
    memory_id: int
    consolidation_type: str  # 'sleep', 'reconsolidation', 'forgetting'
    strength_change: float
    timestamp: float

class SleepLikeConsolidation:
    """
    Sleep-Like Consolidation
    
    During "sleep" periods, memories are replayed and strengthened
    Mimics slow-wave sleep and REM sleep consolidation
    """
    
    def __init__(self,
                 consolidation_rate: float = 0.1,
                 replay_probability: float = 0.3,
                 strength_increase: float = 0.05):
        self.consolidation_rate = consolidation_rate
        self.replay_probability = replay_probability
        self.strength_increase = strength_increase
    
    def select_memories_for_replay(self,
                                  memories: List[MemoryTrace],
                                  num_to_replay: Optional[int] = None) -> List[MemoryTrace]:
        """
        Select memories for replay during sleep
        
        Prioritizes recent, important, or weakly consolidated memories
        """
        if not memories:
            return []
        
        # Score memories for replay priority
        scores = []
        for memory in memories:
            # Recent memories have higher priority
            recency_score = np.exp(-memory.age / 1000.0)
            
            # Important memories have higher priority
            importance_score = memory.importance
            
            # Weakly consolidated memories have higher priority
            consolidation_score = 1.0 - memory.consolidation_level
            
            # Frequently accessed memories have higher priority
            access_score = min(1.0, memory.access_count / 10.0)
            
            # Combined score
            score = (recency_score * 0.3 + 
                    importance_score * 0.3 + 
                    consolidation_score * 0.2 + 
                    access_score * 0.2)
            scores.append(score)
        
        # Select top memories
        if num_to_replay is None:
            num_to_replay = max(1, int(len(memories) * self.replay_probability))
        
        indices = np.argsort(scores)[-num_to_replay:]
        return [memories[i] for i in indices]
    
    def consolidate_memory(self,
                           memory: MemoryTrace,
                           replay_count: int = 1) -> float:
        """
        Consolidate a memory through replay
        
        Returns:
            Strength change
        """
        initial_strength = memory.strength
        
        # Increase consolidation level
        memory.consolidation_level = min(1.0, 
                                        memory.consolidation_level + 
                                        self.consolidation_rate * replay_count)
        
        # Increase strength
        strength_gain = self.strength_increase * replay_count
        memory.strength = min(1.0, memory.strength + strength_gain)
        
        # Update age (consolidated memories age slower)
        memory.age *= (1 - memory.consolidation_level * 0.1)
        
        return memory.strength - initial_strength
    
    def simulate_sleep_cycle(self,
                            memories: List[MemoryTrace],
                            cycle_duration: float = 1000.0) -> List[ConsolidationEvent]:
        """
        Simulate a sleep cycle with memory replay
        
        Returns:
            List of consolidation events
        """
        events = []
        
        # Select memories for replay
        memories_to_replay = self.select_memories_for_replay(memories)
        
        # Replay and consolidate
        for memory in memories_to_replay:
            replay_count = np.random.poisson(2) + 1  # Multiple replays
            strength_change = self.consolidate_memory(memory, replay_count)
            
            event = ConsolidationEvent(
                memory_id=memory.memory_id,
                consolidation_type='sleep',
                strength_change=strength_change,
                timestamp=time.time()
            )
            events.append(event)
        
        return events

class MemoryReconsolidation:
    """
    Memory Reconsolidation
    
    When a memory is recalled, it becomes labile and can be updated
    Allows memories to be modified based on new information
    """
    
    def __init__(self,
                 reconsolidation_window: float = 100.0,  # ms
                 update_strength: float = 0.1):
        self.reconsolidation_window = reconsolidation_window
        self.update_strength = update_strength
    
    def is_labile(self, memory: MemoryTrace, current_time: float) -> bool:
        """Check if memory is in labile state (recently accessed)"""
        time_since_access = current_time - memory.last_access_time
        return time_since_access < self.reconsolidation_window
    
    def update_memory(self,
                     memory: MemoryTrace,
                     new_content: np.ndarray,
                     update_strength: Optional[float] = None) -> float:
        """
        Update memory content during reconsolidation
        
        Blends old and new content
        """
        if update_strength is None:
            update_strength = self.update_strength
        
        # Blend old and new content
        memory.content = (1 - update_strength) * memory.content + update_strength * new_content
        
        # Slightly reduce consolidation level (memory is being modified)
        memory.consolidation_level *= 0.95
        
        # Update access
        memory.access_count += 1
        memory.last_access_time = time.time()
        
        return np.linalg.norm(memory.content - new_content)
    
    def reconsolidate(self,
                     memory: MemoryTrace,
                     current_time: float) -> bool:
        """
        Reconsolidate a memory after it becomes stable again
        
        Returns:
            True if reconsolidation occurred
        """
        if not self.is_labile(memory, current_time):
            # Memory is stable, reconsolidate
            memory.consolidation_level = min(1.0, memory.consolidation_level + 0.1)
            return True
        return False

class ForgettingMechanisms:
    """
    Forgetting Mechanisms
    
    Memories decay over time if not accessed
    Important or frequently accessed memories decay slower
    """
    
    def __init__(self,
                 decay_rate: float = 0.0001,  # Reduced default decay rate
                 forgetting_threshold: float = 0.1,
                 importance_factor: float = 0.5):
        self.decay_rate = decay_rate
        self.forgetting_threshold = forgetting_threshold
        self.importance_factor = importance_factor
    
    def decay_memory(self,
                    memory: MemoryTrace,
                    time_elapsed: float) -> float:
        """
        Decay memory strength over time
        
        Returns:
            Strength change
        """
        initial_strength = memory.strength
        
        # Calculate decay rate (slower for important/frequently accessed memories)
        effective_decay = self.decay_rate * (1 - memory.importance * self.importance_factor)
        effective_decay *= (1 - min(1.0, memory.access_count / 100.0) * 0.5)
        
        # Decay strength
        memory.strength -= effective_decay * time_elapsed
        memory.strength = max(0.0, memory.strength)
        
        # Update age
        memory.age += time_elapsed
        
        return memory.strength - initial_strength
    
    def should_forget(self, memory: MemoryTrace) -> bool:
        """Determine if memory should be forgotten"""
        return memory.strength < self.forgetting_threshold
    
    def forget_memory(self, memory: MemoryTrace) -> Dict:
        """Forget a memory (remove or mark for deletion)"""
        return {
            'memory_id': memory.memory_id,
            'strength': memory.strength,
            'age': memory.age,
            'access_count': memory.access_count
        }

class MemoryConsolidationManager:
    """
    Manages all memory consolidation mechanisms
    """
    
    def __init__(self,
                 enable_sleep_consolidation: bool = True,
                 enable_reconsolidation: bool = True,
                 enable_forgetting: bool = True):
        self.enable_sleep_consolidation = enable_sleep_consolidation
        self.enable_reconsolidation = enable_reconsolidation
        self.enable_forgetting = enable_forgetting
        
        self.sleep_consolidation = SleepLikeConsolidation() if enable_sleep_consolidation else None
        self.reconsolidation = MemoryReconsolidation() if enable_reconsolidation else None
        self.forgetting = ForgettingMechanisms() if enable_forgetting else None
        
        self.memories: Dict[int, MemoryTrace] = {}
        self.consolidation_history: List[ConsolidationEvent] = []
        self.next_memory_id = 0
    
    def create_memory(self,
                     content: np.ndarray,
                     importance: float = 0.5) -> MemoryTrace:
        """Create a new memory trace"""
        memory = MemoryTrace(
            memory_id=self.next_memory_id,
            content=content.copy(),
            importance=importance,
            last_access_time=time.time()
        )
        self.memories[self.next_memory_id] = memory
        self.next_memory_id += 1
        return memory
    
    def access_memory(self,
                     memory_id: int,
                     current_time: float = None) -> Optional[MemoryTrace]:
        """Access a memory (recall)"""
        if memory_id not in self.memories:
            return None
        
        memory = self.memories[memory_id]
        
        if current_time is None:
            current_time = time.time()
        
        memory.access_count += 1
        memory.last_access_time = current_time
        
        # Trigger reconsolidation if enabled
        if self.enable_reconsolidation and self.reconsolidation:
            self.reconsolidation.reconsolidate(memory, current_time)
        
        return memory
    
    def update_memory(self,
                     memory_id: int,
                     new_content: np.ndarray,
                     current_time: float = None) -> bool:
        """Update memory content during reconsolidation window"""
        if memory_id not in self.memories:
            return False
        
        memory = self.memories[memory_id]
        
        if current_time is None:
            current_time = time.time()
        
        if self.enable_reconsolidation and self.reconsolidation:
            if self.reconsolidation.is_labile(memory, current_time):
                self.reconsolidation.update_memory(memory, new_content)
                return True
        
        return False
    
    def simulate_sleep(self,
                       duration: float = 1000.0,
                       current_time: float = None) -> List[ConsolidationEvent]:
        """Simulate sleep consolidation"""
        if not self.enable_sleep_consolidation or not self.sleep_consolidation:
            return []
        
        if current_time is None:
            current_time = time.time()
        
        memories_list = list(self.memories.values())
        events = self.sleep_consolidation.simulate_sleep_cycle(memories_list, duration)
        
        self.consolidation_history.extend(events)
        return events
    
    def update_memories(self,
                       time_elapsed: float,
                       current_time: float = None) -> Dict:
        """
        Update all memories (decay, forgetting)
        
        Returns:
            Statistics about updates
        """
        if current_time is None:
            current_time = time.time()
        
        stats = {
            'memories_decayed': 0,
            'memories_forgotten': 0,
            'total_strength_loss': 0.0
        }
        
        memories_to_remove = []
        
        for memory_id, memory in self.memories.items():
            # Decay
            if self.enable_forgetting and self.forgetting:
                strength_change = self.forgetting.decay_memory(memory, time_elapsed)
                stats['total_strength_loss'] += abs(strength_change)
                
                if strength_change < 0:
                    stats['memories_decayed'] += 1
                
                # Check if should forget
                if self.forgetting.should_forget(memory):
                    forget_info = self.forgetting.forget_memory(memory)
                    memories_to_remove.append(memory_id)
                    stats['memories_forgotten'] += 1
        
        # Remove forgotten memories
        for memory_id in memories_to_remove:
            del self.memories[memory_id]
        
        return stats
    
    def get_statistics(self) -> Dict:
        """Get statistics about memory consolidation"""
        if not self.memories:
            return {}
        
        memories_list = list(self.memories.values())
        
        strengths = [m.strength for m in memories_list]
        consolidation_levels = [m.consolidation_level for m in memories_list]
        ages = [m.age for m in memories_list]
        access_counts = [m.access_count for m in memories_list]
        
        return {
            'total_memories': len(memories_list),
            'avg_strength': np.mean(strengths),
            'avg_consolidation': np.mean(consolidation_levels),
            'avg_age': np.mean(ages),
            'avg_access_count': np.mean(access_counts),
            'consolidated_memories': sum(1 for m in memories_list if m.consolidation_level > 0.5),
            'weak_memories': sum(1 for m in memories_list if m.strength < 0.3),
            'total_consolidation_events': len(self.consolidation_history)
        }

