#!/usr/bin/env python3
"""
Meta-Learning (Learning to Learn) - Phase 7.1
Implements few-shot learning, rapid adaptation, learning strategy selection,
hyperparameter optimization, and transfer learning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import time
import random
from collections import defaultdict

# Import dependencies
try:
    from metacognition import MetacognitiveMonitoring, StrategySelection
    from memory_consolidation import MemoryConsolidationManager
    from plasticity_mechanisms import PlasticityManager
    from reward_learning import RewardLearningManager
    from unsupervised_learning import UnsupervisedLearningManager
except ImportError:
    MetacognitiveMonitoring = None
    StrategySelection = None
    MemoryConsolidationManager = None
    PlasticityManager = None
    RewardLearningManager = None
    UnsupervisedLearningManager = None


@dataclass
class Task:
    """Represents a learning task"""
    task_id: int
    name: str
    domain: str
    examples: List[Tuple[np.ndarray, Optional[np.ndarray]]]  # (input, target)
    task_type: str  # 'classification', 'regression', 'reinforcement', 'unsupervised'
    created_time: float = 0.0


@dataclass
class LearningStrategy:
    """Represents a learning strategy"""
    strategy_id: int
    name: str
    learning_algorithm: str  # 'stdp', 'reward', 'unsupervised', 'hybrid'
    hyperparameters: Dict[str, float]
    success_rate: float = 0.5
    adaptation_speed: float = 0.5
    usage_count: int = 0
    last_used: float = 0.0


class FewShotLearning:
    """
    Few-Shot Learning
    
    Learns from very few examples
    Rapidly adapts to new tasks
    """
    
    def __init__(self,
                 adaptation_rate: float = 0.3,
                 few_shot_threshold: int = 10):
        self.adaptation_rate = adaptation_rate
        self.few_shot_threshold = few_shot_threshold
        self.few_shot_tasks: List[Task] = []
        self.adaptation_history: List[Dict] = []
    
    def learn_from_few_examples(self,
                               task: Task,
                               max_examples: int = 5) -> Dict:
        """
        Learn a task from very few examples
        
        Returns:
            Dictionary with learning results
        """
        if len(task.examples) > max_examples:
            # Use only first max_examples
            examples = task.examples[:max_examples]
        else:
            examples = task.examples
        
        # Extract patterns from examples
        patterns = [ex[0] for ex in examples]
        targets = [ex[1] for ex in examples if ex[1] is not None]
        
        # Learn prototype (average pattern)
        if patterns:
            prototype = np.mean(patterns, axis=0)
            prototype_norm = np.linalg.norm(prototype)
            if prototype_norm > 0:
                prototype = prototype / prototype_norm
        else:
            prototype = np.random.normal(0, 1, 10)
            norm = np.linalg.norm(prototype)
            if norm > 0:
                prototype = prototype / norm
        
        # Learn from similarity to prototype
        learning_result = {
            'task_id': task.task_id,
            'prototype': prototype,
            'num_examples': len(examples),
            'learning_success': len(examples) >= 2,
            'adaptation_speed': self.adaptation_rate,
            'confidence': min(1.0, len(examples) / max_examples)
        }
        
        self.few_shot_tasks.append(task)
        self.adaptation_history.append(learning_result)
        
        return learning_result
    
    def rapid_adaptation(self,
                        base_knowledge: np.ndarray,
                        new_examples: List[Tuple[np.ndarray, Optional[np.ndarray]]],
                        adaptation_strength: float = 0.5) -> np.ndarray:
        """
        Rapidly adapt base knowledge to new examples
        
        Returns:
            Adapted knowledge vector
        """
        if not new_examples:
            return base_knowledge
        
        # Extract patterns
        patterns = [ex[0] for ex in new_examples]
        
        # Compute new prototype
        new_prototype = np.mean(patterns, axis=0)
        new_norm = np.linalg.norm(new_prototype)
        if new_norm > 0:
            new_prototype = new_prototype / new_norm
        
        # Adapt base knowledge
        adapted = (1.0 - adaptation_strength) * base_knowledge + adaptation_strength * new_prototype
        
        # Ensure same dimensionality
        min_dim = min(len(adapted), len(new_prototype))
        adapted = adapted[:min_dim]
        new_prototype = new_prototype[:min_dim]
        
        adapted = (1.0 - adaptation_strength) * adapted + adaptation_strength * new_prototype
        
        # Normalize
        norm = np.linalg.norm(adapted)
        if norm > 0:
            adapted = adapted / norm
        
        return adapted


class LearningStrategySelection:
    """
    Learning Strategy Selection
    
    Chooses optimal learning algorithms for tasks
    Meta-learns which strategies work best
    """
    
    def __init__(self,
                 exploration_probability: float = 0.2):
        self.exploration_probability = exploration_probability
        self.strategies: Dict[int, LearningStrategy] = {}
        self.strategy_performance: Dict[int, List[float]] = defaultdict(list)
        self.next_strategy_id = 0
    
    def create_strategy(self,
                       name: str,
                       algorithm: str,
                       hyperparameters: Dict[str, float]) -> LearningStrategy:
        """Create a new learning strategy"""
        strategy = LearningStrategy(
            strategy_id=self.next_strategy_id,
            name=name,
            learning_algorithm=algorithm,
            hyperparameters=hyperparameters
        )
        
        self.strategies[self.next_strategy_id] = strategy
        self.next_strategy_id += 1
        
        return strategy
    
    def select_strategy(self,
                       task: Task,
                       available_strategies: Optional[List[LearningStrategy]] = None) -> LearningStrategy:
        """
        Select best strategy for a task
        
        Returns:
            Selected learning strategy
        """
        if available_strategies is None:
            available_strategies = list(self.strategies.values())
        
        if not available_strategies:
            # Create default strategy
            return self.create_strategy("default", "stdp", {})
        
        # Exploration vs exploitation
        if random.random() < self.exploration_probability:
            # Explore: choose random strategy
            return random.choice(available_strategies)
        
        # Exploit: choose best strategy for this task type
        best_strategy = None
        best_score = -1.0
        
        for strategy in available_strategies:
            # Score strategy based on past performance
            if strategy.strategy_id in self.strategy_performance:
                avg_performance = np.mean(self.strategy_performance[strategy.strategy_id])
            else:
                avg_performance = 0.5
            
            # Boost score if strategy matches task type
            type_match = 1.0
            if task.task_type == 'reinforcement' and strategy.learning_algorithm == 'reward':
                type_match = 1.2
            elif task.task_type == 'unsupervised' and strategy.learning_algorithm == 'unsupervised':
                type_match = 1.2
            
            score = avg_performance * type_match
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy or available_strategies[0]
    
    def update_strategy_performance(self,
                                  strategy_id: int,
                                  performance: float):
        """Update performance record for a strategy"""
        self.strategy_performance[strategy_id].append(performance)
        
        # Update strategy success rate
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            if len(self.strategy_performance[strategy_id]) > 0:
                strategy.success_rate = np.mean(self.strategy_performance[strategy_id])
            strategy.usage_count += 1
            strategy.last_used = time.time()


class HyperparameterOptimization:
    """
    Hyperparameter Optimization
    
    Automatically tunes learning parameters
    Uses meta-learning to find optimal hyperparameters
    """
    
    def __init__(self,
                 optimization_method: str = 'grid_search',
                 num_trials: int = 10):
        self.optimization_method = optimization_method
        self.num_trials = num_trials
        self.hyperparameter_history: List[Dict] = []
        self.best_hyperparameters: Dict[str, Dict[str, float]] = {}  # task_type -> hyperparameters
    
    def optimize_hyperparameters(self,
                                task: Task,
                                base_hyperparameters: Dict[str, float],
                                performance_function: Callable) -> Dict[str, float]:
        """
        Optimize hyperparameters for a task
        
        Args:
            task: Learning task
            base_hyperparameters: Starting hyperparameters
            performance_function: Function that evaluates performance given hyperparameters
        
        Returns:
            Optimized hyperparameters
        """
        if self.optimization_method == 'grid_search':
            return self._grid_search(task, base_hyperparameters, performance_function)
        elif self.optimization_method == 'random_search':
            return self._random_search(task, base_hyperparameters, performance_function)
        else:
            return base_hyperparameters
    
    def _grid_search(self,
                    task: Task,
                    base_hyperparameters: Dict[str, float],
                    performance_function: Callable) -> Dict[str, float]:
        """Grid search for hyperparameters"""
        best_params = base_hyperparameters.copy()
        best_performance = performance_function(base_hyperparameters)
        
        # Try variations of each hyperparameter
        for param_name, param_value in base_hyperparameters.items():
            # Try lower and higher values
            for multiplier in [0.5, 0.7, 1.3, 1.5]:
                test_params = base_hyperparameters.copy()
                test_params[param_name] = param_value * multiplier
                
                performance = performance_function(test_params)
                
                if performance > best_performance:
                    best_performance = performance
                    best_params = test_params
        
        return best_params
    
    def _random_search(self,
                      task: Task,
                      base_hyperparameters: Dict[str, float],
                      performance_function: Callable) -> Dict[str, float]:
        """Random search for hyperparameters"""
        best_params = base_hyperparameters.copy()
        best_performance = performance_function(base_hyperparameters)
        
        for _ in range(self.num_trials):
            test_params = {}
            for param_name, param_value in base_hyperparameters.items():
                # Random variation
                multiplier = np.random.uniform(0.5, 1.5)
                test_params[param_name] = param_value * multiplier
            
            performance = performance_function(test_params)
            
            if performance > best_performance:
                best_performance = performance
                best_params = test_params
        
        return best_params
    
    def get_best_hyperparameters(self, task_type: str) -> Optional[Dict[str, float]]:
        """Get best hyperparameters for a task type"""
        return self.best_hyperparameters.get(task_type)


class TransferLearning:
    """
    Transfer Learning
    
    Transfers knowledge across domains
    Applies learned knowledge to new tasks
    """
    
    def __init__(self,
                 transfer_strength: float = 0.6,
                 similarity_threshold: float = 0.5):
        self.transfer_strength = transfer_strength
        self.similarity_threshold = similarity_threshold
        self.transfer_history: List[Dict] = []
        self.knowledge_base: Dict[str, np.ndarray] = {}  # domain -> knowledge
    
    def transfer_knowledge(self,
                          source_domain: str,
                          target_domain: str,
                          source_knowledge: np.ndarray) -> Optional[np.ndarray]:
        """
        Transfer knowledge from source to target domain
        
        Returns:
            Transferred knowledge or None
        """
        # Check if domains are similar enough
        if source_domain in self.knowledge_base:
            source_kb = self.knowledge_base[source_domain]
            similarity = np.dot(source_knowledge, source_kb) / (
                np.linalg.norm(source_knowledge) * np.linalg.norm(source_kb) + 1e-10
            )
        else:
            similarity = 0.5  # Default similarity
        
        if similarity < self.similarity_threshold:
            return None
        
        # Transfer knowledge
        if target_domain in self.knowledge_base:
            target_kb = self.knowledge_base[target_domain]
            transferred = (1.0 - self.transfer_strength) * target_kb + self.transfer_strength * source_knowledge
        else:
            transferred = source_knowledge.copy()
        
        # Normalize
        norm = np.linalg.norm(transferred)
        if norm > 0:
            transferred = transferred / norm
        
        # Store transferred knowledge
        self.knowledge_base[target_domain] = transferred
        
        transfer_record = {
            'source_domain': source_domain,
            'target_domain': target_domain,
            'similarity': similarity,
            'transfer_strength': self.transfer_strength,
            'timestamp': time.time()
        }
        self.transfer_history.append(transfer_record)
        
        return transferred
    
    def store_knowledge(self, domain: str, knowledge: np.ndarray):
        """Store knowledge for a domain"""
        # Normalize
        norm = np.linalg.norm(knowledge)
        if norm > 0:
            knowledge = knowledge / norm
        
        self.knowledge_base[domain] = knowledge
    
    def retrieve_knowledge(self, domain: str) -> Optional[np.ndarray]:
        """Retrieve stored knowledge for a domain"""
        return self.knowledge_base.get(domain)


class MetaLearningSystem:
    """
    Meta-Learning System Manager
    
    Integrates all meta-learning components
    Learns to learn effectively
    """
    
    def __init__(self,
                 brain_system=None,
                 metacognition: Optional[MetacognitiveMonitoring] = None,
                 memory_consolidation: Optional[MemoryConsolidationManager] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.few_shot_learning = FewShotLearning()
        self.strategy_selection = LearningStrategySelection()
        self.hyperparameter_optimization = HyperparameterOptimization()
        self.transfer_learning = TransferLearning()
        
        # Integration with existing systems
        self.metacognition = metacognition
        self.memory_consolidation = memory_consolidation
        
        # Task tracking
        self.tasks: Dict[int, Task] = {}
        self.next_task_id = 0
        
        # Statistics
        self.stats = {
            'tasks_learned': 0,
            'few_shot_tasks': 0,
            'transfers_performed': 0,
            'average_adaptation_speed': 0.0,
            'average_success_rate': 0.0
        }
    
    def learn_task(self,
                  task: Task,
                  use_few_shot: bool = True,
                  optimize_hyperparameters: bool = True) -> Dict:
        """
        Learn a new task using meta-learning
        
        Returns:
            Dictionary with learning results
        """
        # Store task
        task.task_id = self.next_task_id
        self.next_task_id += 1
        self.tasks[task.task_id] = task
        
        # Select learning strategy
        strategy = self.strategy_selection.select_strategy(task)
        
        # Optimize hyperparameters if requested
        if optimize_hyperparameters:
            def performance_fn(params):
                # Simple performance estimate based on task characteristics
                return 0.5 + np.random.normal(0, 0.1)
            
            optimized_params = self.hyperparameter_optimization.optimize_hyperparameters(
                task, strategy.hyperparameters, performance_fn
            )
            strategy.hyperparameters = optimized_params
        
        # Learn task
        if use_few_shot and len(task.examples) <= self.few_shot_learning.few_shot_threshold:
            learning_result = self.few_shot_learning.learn_from_few_examples(task)
            self.stats['few_shot_tasks'] += 1
        else:
            # Standard learning (would use actual learning mechanisms)
            learning_result = {
                'task_id': task.task_id,
                'strategy': strategy.name,
                'learning_success': True,
                'confidence': 0.7
            }
        
        # Update strategy performance
        performance = learning_result.get('confidence', 0.5)
        self.strategy_selection.update_strategy_performance(strategy.strategy_id, performance)
        
        # Store knowledge for transfer
        if 'prototype' in learning_result:
            self.transfer_learning.store_knowledge(task.domain, learning_result['prototype'])
        
        # Update statistics
        self._update_stats(learning_result)
        
        return learning_result
    
    def adapt_to_new_task(self,
                         base_task_id: int,
                         new_examples: List[Tuple[np.ndarray, Optional[np.ndarray]]]) -> Dict:
        """
        Rapidly adapt to a new task based on a similar task
        
        Returns:
            Adaptation results
        """
        if base_task_id not in self.tasks:
            return {'success': False, 'error': 'Base task not found'}
        
        base_task = self.tasks[base_task_id]
        
        # Retrieve base knowledge
        base_knowledge = self.transfer_learning.retrieve_knowledge(base_task.domain)
        if base_knowledge is None:
            # Use prototype from base task
            if base_task.examples:
                patterns = [ex[0] for ex in base_task.examples]
                base_knowledge = np.mean(patterns, axis=0)
                norm = np.linalg.norm(base_knowledge)
                if norm > 0:
                    base_knowledge = base_knowledge / norm
            else:
                base_knowledge = np.random.normal(0, 1, 10)
                norm = np.linalg.norm(base_knowledge)
                if norm > 0:
                    base_knowledge = base_knowledge / norm
        
        # Rapid adaptation
        adapted_knowledge = self.few_shot_learning.rapid_adaptation(
            base_knowledge, new_examples
        )
        
        # Create new task
        new_task = Task(
            task_id=self.next_task_id,
            name=f"Adapted from {base_task.name}",
            domain=base_task.domain,
            examples=new_examples,
            task_type=base_task.task_type
        )
        self.next_task_id += 1
        self.tasks[new_task.task_id] = new_task
        
        # Store adapted knowledge
        self.transfer_learning.store_knowledge(new_task.domain, adapted_knowledge)
        
        adaptation_result = {
            'success': True,
            'base_task_id': base_task_id,
            'new_task_id': new_task.task_id,
            'adapted_knowledge': adapted_knowledge,
            'adaptation_speed': self.few_shot_learning.adaptation_rate
        }
        
        self.stats['transfers_performed'] += 1
        return adaptation_result
    
    def transfer_knowledge_between_domains(self,
                                         source_domain: str,
                                         target_domain: str) -> Optional[np.ndarray]:
        """Transfer knowledge between domains"""
        source_knowledge = self.transfer_learning.retrieve_knowledge(source_domain)
        if source_knowledge is None:
            return None
        
        transferred = self.transfer_learning.transfer_knowledge(
            source_domain, target_domain, source_knowledge
        )
        
        if transferred is not None:
            self.stats['transfers_performed'] += 1
        
        return transferred
    
    def _update_stats(self, learning_result: Dict):
        """Update statistics"""
        self.stats['tasks_learned'] += 1
        
        adaptation_speed = learning_result.get('adaptation_speed', 0.5)
        total = self.stats['tasks_learned']
        self.stats['average_adaptation_speed'] = (
            (self.stats['average_adaptation_speed'] * (total - 1) + adaptation_speed) / total
        )
        
        success_rate = learning_result.get('confidence', 0.5)
        self.stats['average_success_rate'] = (
            (self.stats['average_success_rate'] * (total - 1) + success_rate) / total
        )
    
    def get_statistics(self) -> Dict:
        """Get meta-learning statistics"""
        return self.stats.copy()
    
    def enhance_learning_method(self, learning_method: Callable) -> Callable:
        """
        Enhance an existing learning method with meta-learning
        
        Returns:
            Enhanced learning method
        """
        def enhanced_method(*args, **kwargs):
            # Monitor learning performance
            start_time = time.time()
            
            # Execute original method
            result = learning_method(*args, **kwargs)
            
            # Compute performance metrics
            duration = time.time() - start_time
            performance = 1.0 / (1.0 + duration)  # Faster = better
            
            # Update meta-learning based on performance
            # (In practice, would update strategy selection, etc.)
            
            return result
        
        return enhanced_method

