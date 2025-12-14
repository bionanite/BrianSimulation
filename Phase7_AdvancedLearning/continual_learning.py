#!/usr/bin/env python3
"""
Continual Learning - Phase 7.2
Implements catastrophic forgetting prevention, task rehearsal, elastic weight consolidation,
progressive networks, and lifelong learning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
import random
from collections import defaultdict

# Import dependencies
try:
    from memory_consolidation import MemoryConsolidationManager
    from structural_plasticity import StructuralPlasticityManager
    from global_workspace import GlobalWorkspace
except ImportError:
    MemoryConsolidationManager = None
    StructuralPlasticityManager = None
    GlobalWorkspace = None


@dataclass
class TaskMemory:
    """Represents memory for a specific task"""
    task_id: int
    task_name: str
    weights: Dict[str, np.ndarray]  # Parameter name -> weight values
    importance: Dict[str, float]  # Parameter importance
    examples: List[np.ndarray]  # Stored examples for rehearsal
    last_accessed: float = 0.0
    access_count: int = 0


class CatastrophicForgettingPrevention:
    """
    Catastrophic Forgetting Prevention
    
    Maintains old knowledge while learning new tasks
    Prevents interference between tasks
    """
    
    def __init__(self,
                 consolidation_strength: float = 0.7,
                 interference_threshold: float = 0.3):
        self.consolidation_strength = consolidation_strength
        self.interference_threshold = interference_threshold
        self.task_memories: Dict[int, TaskMemory] = {}
        self.forgetting_events: List[Dict] = []
    
    def protect_important_weights(self,
                                 task_id: int,
                                 weights: Dict[str, np.ndarray],
                                 importance: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Protect important weights from being overwritten
        
        Returns:
            Protected weights
        """
        protected_weights = {}
        
        for param_name, weight_value in weights.items():
            importance_score = importance.get(param_name, 0.5)
            
            # Protect based on importance
            protection_strength = importance_score * self.consolidation_strength
            
            # Blend with old weights if they exist
            if task_id in self.task_memories:
                old_memory = self.task_memories[task_id]
                if param_name in old_memory.weights:
                    old_weight = old_memory.weights[param_name]
                    protected_weight = (
                        (1.0 - protection_strength) * weight_value +
                        protection_strength * old_weight
                    )
                else:
                    protected_weight = weight_value
            else:
                protected_weight = weight_value
            
            protected_weights[param_name] = protected_weight
        
        return protected_weights
    
    def detect_interference(self,
                          old_task_id: int,
                          new_task_id: int,
                          old_weights: Dict[str, np.ndarray],
                          new_weights: Dict[str, np.ndarray]) -> float:
        """
        Detect interference between tasks
        
        Returns:
            Interference score (0-1, higher = more interference)
        """
        interference_scores = []
        
        for param_name in old_weights.keys():
            if param_name in new_weights:
                old_weight = old_weights[param_name]
                new_weight = new_weights[param_name]
                
                # Compute change magnitude
                change = np.linalg.norm(new_weight - old_weight)
                old_magnitude = np.linalg.norm(old_weight)
                
                if old_magnitude > 0:
                    relative_change = change / old_magnitude
                    interference_scores.append(relative_change)
        
        interference = np.mean(interference_scores) if interference_scores else 0.0
        
        if interference > self.interference_threshold:
            self.forgetting_events.append({
                'old_task_id': old_task_id,
                'new_task_id': new_task_id,
                'interference': interference,
                'timestamp': time.time()
            })
        
        return interference


class TaskRehearsal:
    """
    Task Rehearsal
    
    Replays old tasks to prevent forgetting
    Maintains performance on previous tasks
    """
    
    def __init__(self,
                 rehearsal_probability: float = 0.3,
                 rehearsal_frequency: int = 10):  # Rehearse every N new tasks
        self.rehearsal_probability = rehearsal_probability
        self.rehearsal_frequency = rehearsal_frequency
        self.rehearsal_history: List[Dict] = []
        self.tasks_since_rehearsal = 0
    
    def select_tasks_for_rehearsal(self,
                                  task_memories: Dict[int, TaskMemory],
                                  num_to_rehearse: Optional[int] = None) -> List[int]:
        """
        Select tasks to rehearse
        
        Returns:
            List of task IDs to rehearse
        """
        if not task_memories:
            return []
        
        # Check if it's time to rehearse
        self.tasks_since_rehearsal += 1
        if self.tasks_since_rehearsal < self.rehearsal_frequency:
            return []
        
        self.tasks_since_rehearsal = 0
        
        # Score tasks for rehearsal priority
        scores = []
        for task_id, memory in task_memories.items():
            # Older tasks have higher priority
            age = time.time() - memory.last_accessed
            age_score = np.exp(-age / 10000.0)
            
            # Less frequently accessed tasks have higher priority
            access_score = 1.0 / (1.0 + memory.access_count)
            
            # Tasks with stored examples have higher priority
            example_score = min(1.0, len(memory.examples) / 10.0)
            
            score = age_score * 0.4 + access_score * 0.3 + example_score * 0.3
            scores.append((task_id, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if num_to_rehearse is None:
            num_to_rehearse = max(1, int(len(task_memories) * self.rehearsal_probability))
        
        selected = [task_id for task_id, _ in scores[:num_to_rehearse]]
        return selected
    
    def rehearse_task(self,
                     task_id: int,
                     task_memory: TaskMemory) -> Dict:
        """
        Rehearse a task to prevent forgetting
        
        Returns:
            Rehearsal results
        """
        if not task_memory.examples:
            return {'success': False, 'reason': 'No examples stored'}
        
        # Replay examples (simplified - would use actual learning)
        num_replayed = min(len(task_memory.examples), 5)
        replayed_examples = random.sample(task_memory.examples, num_replayed)
        
        # Update access tracking
        task_memory.last_accessed = time.time()
        task_memory.access_count += 1
        
        rehearsal_result = {
            'task_id': task_id,
            'num_examples_replayed': num_replayed,
            'success': True,
            'timestamp': time.time()
        }
        
        self.rehearsal_history.append(rehearsal_result)
        return rehearsal_result


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC)
    
    Protects important weights based on Fisher information
    Prevents catastrophic forgetting
    """
    
    def __init__(self,
                 ewc_lambda: float = 0.4,
                 fisher_samples: int = 100):
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.fisher_information: Dict[int, Dict[str, np.ndarray]] = {}  # task_id -> param_name -> Fisher info
        self.important_weights: Dict[int, Dict[str, np.ndarray]] = {}  # task_id -> param_name -> important weights
    
    def compute_fisher_information(self,
                                 task_id: int,
                                 weights: Dict[str, np.ndarray],
                                 examples: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute Fisher information matrix for weights
        
        Returns:
            Dictionary of Fisher information per parameter
        """
        fisher_info = {}
        
        for param_name, weight_value in weights.items():
            # Simplified Fisher information computation
            # In practice, would compute gradients over examples
            fisher_value = np.ones_like(weight_value) * 0.1
            
            # Increase Fisher info for parameters that vary more
            if len(examples) > 0:
                # Use example variance as proxy
                example_array = np.array(examples)
                if example_array.size > 0:
                    variance = np.var(example_array, axis=0)
                    if variance.size == weight_value.size:
                        fisher_value = variance
            
            fisher_info[param_name] = fisher_value
        
        self.fisher_information[task_id] = fisher_info
        return fisher_info
    
    def compute_ewc_loss(self,
                        current_weights: Dict[str, np.ndarray],
                        important_weights: Dict[str, np.ndarray],
                        fisher_info: Dict[str, np.ndarray]) -> float:
        """
        Compute EWC regularization loss
        
        Returns:
            EWC loss value
        """
        total_loss = 0.0
        
        for param_name in current_weights.keys():
            if param_name in important_weights and param_name in fisher_info:
                current = current_weights[param_name]
                important = important_weights[param_name]
                fisher = fisher_info[param_name]
                
                # EWC loss: lambda * Fisher * (current - important)^2
                diff = current - important
                loss = self.ewc_lambda * np.sum(fisher * diff * diff)
                total_loss += loss
        
        return total_loss
    
    def store_important_weights(self,
                              task_id: int,
                              weights: Dict[str, np.ndarray]):
        """Store important weights for a task"""
        self.important_weights[task_id] = {
            name: weight.copy() for name, weight in weights.items()
        }


class ProgressiveNetworks:
    """
    Progressive Networks
    
    Adds capacity for new tasks
    Prevents interference by adding new pathways
    """
    
    def __init__(self,
                 expansion_rate: float = 0.2):
        self.expansion_rate = expansion_rate
        self.network_layers: Dict[int, Dict[str, np.ndarray]] = {}  # task_id -> layer_name -> weights
        self.connections: Dict[Tuple[int, int], np.ndarray] = {}  # (from_task, to_task) -> connection weights
    
    def add_task_capacity(self,
                         task_id: int,
                         base_layers: Dict[str, np.ndarray],
                         new_layer_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Add new capacity for a task
        
        Returns:
            Expanded layers
        """
        expanded_layers = {}
        
        for layer_name, base_weights in base_layers.items():
            if new_layer_size is None:
                # Expand by expansion_rate
                current_size = base_weights.shape[0] if len(base_weights.shape) > 0 else len(base_weights)
                new_size = int(current_size * (1.0 + self.expansion_rate))
            else:
                new_size = new_layer_size
            
            # Create expanded layer
            if len(base_weights.shape) == 1:
                # 1D weights
                expanded = np.zeros(new_size)
                expanded[:len(base_weights)] = base_weights
                # Initialize new weights randomly
                expanded[len(base_weights):] = np.random.normal(0, 0.1, new_size - len(base_weights))
            else:
                # Multi-dimensional weights
                expanded = np.zeros((new_size, base_weights.shape[1]))
                expanded[:base_weights.shape[0], :] = base_weights
                expanded[base_weights.shape[0]:, :] = np.random.normal(0, 0.1, 
                    (new_size - base_weights.shape[0], base_weights.shape[1]))
            
            expanded_layers[layer_name] = expanded
        
        self.network_layers[task_id] = expanded_layers
        return expanded_layers
    
    def create_lateral_connection(self,
                                from_task_id: int,
                                to_task_id: int,
                                connection_strength: float = 0.3):
        """Create lateral connection between tasks"""
        if from_task_id in self.network_layers and to_task_id in self.network_layers:
            # Create connection matrix
            from_layers = self.network_layers[from_task_id]
            to_layers = self.network_layers[to_task_id]
            
            # Simple connection (would be more sophisticated in practice)
            connection_key = (from_task_id, to_task_id)
            if connection_key not in self.connections:
                # Create random connection
                connection_size = (10, 10)  # Simplified
                self.connections[connection_key] = np.random.normal(0, connection_strength, connection_size)


class ContinualLearningSystem:
    """
    Continual Learning System Manager
    
    Integrates all continual learning components
    Enables lifelong learning
    """
    
    def __init__(self,
                 brain_system=None,
                 memory_consolidation: Optional[MemoryConsolidationManager] = None,
                 structural_plasticity: Optional[StructuralPlasticityManager] = None,
                 global_workspace: Optional[GlobalWorkspace] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.forgetting_prevention = CatastrophicForgettingPrevention()
        self.task_rehearsal = TaskRehearsal()
        self.ewc = ElasticWeightConsolidation()
        self.progressive_networks = ProgressiveNetworks()
        
        # Integration with existing systems
        self.memory_consolidation = memory_consolidation
        self.structural_plasticity = structural_plasticity
        self.global_workspace = global_workspace
        
        # Task tracking
        self.task_memories: Dict[int, TaskMemory] = {}
        self.next_task_id = 0
        
        # Statistics
        self.stats = {
            'tasks_learned': 0,
            'tasks_rehearsed': 0,
            'forgetting_events': 0,
            'average_retention': 1.0
        }
    
    def learn_new_task(self,
                      task_name: str,
                      weights: Dict[str, np.ndarray],
                      examples: List[np.ndarray],
                      importance: Optional[Dict[str, float]] = None) -> int:
        """
        Learn a new task while preserving old knowledge
        
        Returns:
            Task ID
        """
        task_id = self.next_task_id
        self.next_task_id += 1
        
        # Compute importance if not provided
        if importance is None:
            importance = {name: 0.5 for name in weights.keys()}
        
        # Protect important weights from old tasks
        protected_weights = self.forgetting_prevention.protect_important_weights(
            task_id, weights, importance
        )
        
        # Check for interference with old tasks
        for old_task_id, old_memory in self.task_memories.items():
            interference = self.forgetting_prevention.detect_interference(
                old_task_id, task_id, old_memory.weights, protected_weights
            )
            if interference > self.forgetting_prevention.interference_threshold:
                self.stats['forgetting_events'] += 1
        
        # Store task memory
        task_memory = TaskMemory(
            task_id=task_id,
            task_name=task_name,
            weights=protected_weights,
            importance=importance,
            examples=examples[:10],  # Store up to 10 examples
            last_accessed=time.time()
        )
        self.task_memories[task_id] = task_memory
        
        # Compute Fisher information for EWC
        fisher_info = self.ewc.compute_fisher_information(task_id, protected_weights, examples)
        self.ewc.store_important_weights(task_id, protected_weights)
        
        # Add progressive network capacity
        self.progressive_networks.add_task_capacity(task_id, protected_weights)
        
        # Update statistics
        self.stats['tasks_learned'] += 1
        
        return task_id
    
    def rehearse_old_tasks(self):
        """Rehearse old tasks to prevent forgetting"""
        # Select tasks for rehearsal
        tasks_to_rehearse = self.task_rehearsal.select_tasks_for_rehearsal(self.task_memories)
        
        for task_id in tasks_to_rehearse:
            if task_id in self.task_memories:
                memory = self.task_memories[task_id]
                result = self.task_rehearsal.rehearse_task(task_id, memory)
                if result['success']:
                    self.stats['tasks_rehearsed'] += 1
    
    def compute_retention(self, task_id: int) -> float:
        """
        Compute retention score for a task
        
        Returns:
            Retention score (0-1, higher = better retention)
        """
        if task_id not in self.task_memories:
            return 0.0
        
        memory = self.task_memories[task_id]
        
        # Retention based on access frequency and recency
        age = time.time() - memory.last_accessed
        age_factor = np.exp(-age / 10000.0)
        access_factor = min(1.0, memory.access_count / 10.0)
        
        retention = (age_factor + access_factor) / 2.0
        return retention
    
    def get_statistics(self) -> Dict:
        """Get continual learning statistics"""
        # Compute average retention
        if self.task_memories:
            retentions = [self.compute_retention(task_id) for task_id in self.task_memories.keys()]
            self.stats['average_retention'] = np.mean(retentions)
        
        return self.stats.copy()

