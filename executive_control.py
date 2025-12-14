#!/usr/bin/env python3
"""
Executive Control
Implements cognitive control, attention management, task switching, inhibition, and working memory
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
from collections import deque

@dataclass
class Task:
    """Represents a task"""
    task_id: int
    description: str
    priority: float = 1.0
    status: str = 'active'  # 'active', 'completed', 'suspended', 'failed'
    created_time: float = 0.0
    progress: float = 0.0

@dataclass
class AttentionFocus:
    """Represents current attention focus"""
    focus_id: int
    target: str  # What is being attended to
    intensity: float = 1.0
    duration: float = 0.0
    created_time: float = 0.0

class CognitiveControl:
    """
    Cognitive Control
    
    Manages cognitive resources
    Controls attention and inhibition
    """
    
    def __init__(self,
                 control_strength: float = 1.0,
                 inhibition_threshold: float = 0.5):
        self.control_strength = control_strength
        self.inhibition_threshold = inhibition_threshold
        
        self.active_tasks: Dict[int, Task] = {}
        self.inhibited_responses: Set[int] = set()
    
    def create_task(self, description: str, priority: float = 1.0) -> Task:
        """Create a new task"""
        task_id = len(self.active_tasks)
        task = Task(
            task_id=task_id,
            description=description,
            priority=priority,
            created_time=time.time()
        )
        self.active_tasks[task_id] = task
        return task
    
    def inhibit_response(self, response_id: int):
        """Inhibit a response"""
        self.inhibited_responses.add(response_id)
    
    def release_inhibition(self, response_id: int):
        """Release inhibition on a response"""
        self.inhibited_responses.discard(response_id)
    
    def is_inhibited(self, response_id: int) -> bool:
        """Check if response is inhibited"""
        return response_id in self.inhibited_responses
    
    def select_task(self) -> Optional[Task]:
        """Select highest priority active task"""
        active = [t for t in self.active_tasks.values() if t.status == 'active']
        if not active:
            return None
        
        # Select by priority
        return max(active, key=lambda t: t.priority)

class AttentionManagement:
    """
    Attention Management
    
    Manages attention focus
    Controls what is attended to
    """
    
    def __init__(self,
                 attention_capacity: float = 1.0,
                 focus_duration: float = 1000.0):
        self.attention_capacity = attention_capacity
        self.focus_duration = focus_duration
        
        self.current_focuses: List[AttentionFocus] = []
        self.attention_history: deque = deque(maxlen=100)
        self.next_focus_id = 0
    
    def focus_attention(self,
                      target: str,
                      intensity: float = 1.0) -> AttentionFocus:
        """Focus attention on a target"""
        # Check if already focused
        existing = None
        for focus in self.current_focuses:
            if focus.target == target:
                existing = focus
                break
        
        if existing:
            # Update existing focus
            existing.intensity = min(1.0, existing.intensity + 0.1)
            existing.created_time = time.time()
            return existing
        
        # Create new focus
        focus = AttentionFocus(
            focus_id=self.next_focus_id,
            target=target,
            intensity=intensity,
            created_time=time.time()
        )
        
        self.current_focuses.append(focus)
        self.attention_history.append(focus.target)
        self.next_focus_id += 1
        
        return focus
    
    def shift_attention(self, new_target: str, intensity: float = 1.0):
        """Shift attention to new target"""
        # Decay old focuses
        current_time = time.time()
        self.current_focuses = [
            f for f in self.current_focuses
            if current_time - f.created_time < self.focus_duration
        ]
        
        # Focus on new target
        return self.focus_attention(new_target, intensity)
    
    def get_attention_distribution(self) -> Dict[str, float]:
        """Get current attention distribution"""
        if not self.current_focuses:
            return {}
        
        total_intensity = sum(f.intensity for f in self.current_focuses)
        if total_intensity == 0:
            return {}
        
        distribution = {}
        for focus in self.current_focuses:
            distribution[focus.target] = focus.intensity / total_intensity
        
        return distribution
    
    def compute_attention_load(self) -> float:
        """Compute current attention load"""
        return sum(f.intensity for f in self.current_focuses) / self.attention_capacity

class TaskSwitching:
    """
    Task Switching
    
    Manages switching between tasks
    Handles task transitions
    """
    
    def __init__(self,
                 switch_cost: float = 0.1,
                 switch_threshold: float = 0.3):
        self.switch_cost = switch_cost
        self.switch_threshold = switch_threshold
        
        self.current_task_id: Optional[int] = None
        self.switch_history: List[Tuple[int, int, float]] = []  # (from, to, time)
    
    def should_switch_task(self,
                         current_task: Task,
                         candidate_task: Task) -> bool:
        """Determine if should switch tasks"""
        if current_task.task_id == candidate_task.task_id:
            return False
        
        # Switch if candidate has much higher priority
        priority_diff = candidate_task.priority - current_task.priority
        
        return priority_diff > self.switch_threshold
    
    def switch_task(self,
                   from_task_id: int,
                   to_task_id: int) -> float:
        """
        Switch from one task to another
        
        Returns:
            Switch cost
        """
        self.switch_history.append((from_task_id, to_task_id, time.time()))
        self.current_task_id = to_task_id
        
        return self.switch_cost
    
    def get_switch_frequency(self, time_window: float = 1000.0) -> float:
        """Get task switching frequency"""
        if not self.switch_history:
            return 0.0
        
        current_time = time.time()
        recent_switches = [
            s for s in self.switch_history
            if current_time - s[2] < time_window
        ]
        
        return len(recent_switches) / (time_window / 1000.0) if time_window > 0 else 0.0

class InhibitionControl:
    """
    Inhibition Control
    
    Suppresses unwanted responses
    Prevents interference
    """
    
    def __init__(self,
                 inhibition_strength: float = 1.0):
        self.inhibition_strength = inhibition_strength
        
        self.inhibited_items: Dict[int, float] = {}  # item_id -> inhibition_strength
        self.inhibition_history: List[Tuple[int, float]] = []
    
    def inhibit(self, item_id: int, strength: Optional[float] = None):
        """Inhibit an item"""
        if strength is None:
            strength = self.inhibition_strength
        
        self.inhibited_items[item_id] = strength
        self.inhibition_history.append((item_id, time.time()))
    
    def release(self, item_id: int):
        """Release inhibition on an item"""
        if item_id in self.inhibited_items:
            del self.inhibited_items[item_id]
    
    def is_inhibited(self, item_id: int) -> bool:
        """Check if item is inhibited"""
        return item_id in self.inhibited_items
    
    def get_inhibition_strength(self, item_id: int) -> float:
        """Get inhibition strength for an item"""
        return self.inhibited_items.get(item_id, 0.0)
    
    def decay_inhibition(self, decay_rate: float = 0.01):
        """Decay inhibition over time"""
        for item_id in list(self.inhibited_items.keys()):
            self.inhibited_items[item_id] *= (1 - decay_rate)
            if self.inhibited_items[item_id] < 0.01:
                del self.inhibited_items[item_id]

class WorkingMemoryManagement:
    """
    Working Memory Management
    
    Manages working memory capacity
    Maintains and updates information
    """
    
    def __init__(self,
                 capacity: int = 7,  # Miller's magic number
                 decay_rate: float = 0.1):
        self.capacity = capacity
        self.decay_rate = decay_rate
        
        self.memory_items: deque = deque(maxlen=capacity)
        self.item_strengths: Dict[int, float] = {}
        self.next_item_id = 0
    
    def add_to_working_memory(self, item: str, strength: float = 1.0) -> int:
        """Add item to working memory"""
        item_id = self.next_item_id
        self.next_item_id += 1
        
        self.memory_items.append((item_id, item))
        self.item_strengths[item_id] = strength
        
        return item_id
    
    def update_strength(self, item_id: int, new_strength: float):
        """Update strength of memory item"""
        if item_id in self.item_strengths:
            self.item_strengths[item_id] = new_strength
    
    def decay_memory(self):
        """Decay memory strengths"""
        for item_id in list(self.item_strengths.keys()):
            self.item_strengths[item_id] *= (1 - self.decay_rate)
            if self.item_strengths[item_id] < 0.01:
                del self.item_strengths[item_id]
                # Remove from memory items
                self.memory_items = deque(
                    [(id, item) for id, item in self.memory_items if id != item_id],
                    maxlen=self.capacity
                )
    
    def get_working_memory_contents(self) -> List[Tuple[int, str, float]]:
        """Get current working memory contents"""
        contents = []
        for item_id, item in self.memory_items:
            strength = self.item_strengths.get(item_id, 0.0)
            contents.append((item_id, item, strength))
        
        return contents
    
    def is_in_memory(self, item: str) -> bool:
        """Check if item is in working memory"""
        return any(item == stored_item for _, stored_item in self.memory_items)

class ExecutiveControlManager:
    """
    Manages all executive control mechanisms
    """
    
    def __init__(self):
        self.cognitive_control = CognitiveControl()
        self.attention_management = AttentionManagement()
        self.task_switching = TaskSwitching()
        self.inhibition_control = InhibitionControl()
        self.working_memory = WorkingMemoryManagement()
    
    def create_task(self, description: str, priority: float = 1.0) -> Task:
        """Create a new task"""
        return self.cognitive_control.create_task(description, priority)
    
    def focus_attention(self, target: str, intensity: float = 1.0):
        """Focus attention on target"""
        return self.attention_management.focus_attention(target, intensity)
    
    def switch_task(self, from_task_id: int, to_task_id: int) -> float:
        """Switch between tasks"""
        return self.task_switching.switch_task(from_task_id, to_task_id)
    
    def inhibit_response(self, response_id: int):
        """Inhibit a response"""
        self.inhibition_control.inhibit(response_id)
    
    def add_to_working_memory(self, item: str) -> int:
        """Add item to working memory"""
        return self.working_memory.add_to_working_memory(item)
    
    def get_statistics(self) -> Dict:
        """Get statistics about executive control"""
        return {
            'active_tasks': len([t for t in self.cognitive_control.active_tasks.values() if t.status == 'active']),
            'attention_focuses': len(self.attention_management.current_focuses),
            'attention_load': self.attention_management.compute_attention_load(),
            'task_switches': len(self.task_switching.switch_history),
            'inhibited_items': len(self.inhibition_control.inhibited_items),
            'working_memory_items': len(self.working_memory.memory_items),
            'working_memory_load': len(self.working_memory.memory_items) / self.working_memory.capacity
        }

