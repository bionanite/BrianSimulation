#!/usr/bin/env python3
"""
Recursive Self-Improvement - Phase 10.2
Implements architecture search, algorithm discovery, self-modification,
capability expansion, and performance monitoring
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import time
import inspect
from collections import defaultdict

# Import dependencies
try:
    from self_model import SelfModelSystem
    from metacognition import MetacognitiveMonitoring
    from plasticity_mechanisms import PlasticityManager
    from reward_learning import RewardLearningManager
    from unsupervised_learning import UnsupervisedLearningManager
except ImportError:
    SelfModelSystem = None
    MetacognitiveMonitoring = None
    PlasticityManager = None
    RewardLearningManager = None
    UnsupervisedLearningManager = None


@dataclass
class Architecture:
    """Represents a neural architecture"""
    architecture_id: int
    name: str
    layers: List[int]  # Layer sizes
    connections: Dict[Tuple[int, int], float]  # (from_layer, to_layer) -> connection_strength
    performance: float = 0.0
    efficiency: float = 0.5
    created_time: float = 0.0


@dataclass
class Algorithm:
    """Represents a learning algorithm"""
    algorithm_id: int
    name: str
    parameters: Dict[str, float]
    performance: float = 0.0
    usage_count: int = 0
    created_time: float = 0.0


@dataclass
class Capability:
    """Represents a system capability"""
    capability_id: int
    name: str
    description: str
    strength: float = 0.0
    enabled: bool = True
    created_time: float = 0.0


class ArchitectureSearch:
    """
    Architecture Search
    
    Searches for better architectures
    Explores architecture space
    """
    
    def __init__(self,
                 search_strategy: str = 'evolutionary',
                 population_size: int = 10):
        self.search_strategy = search_strategy
        self.population_size = population_size
        self.architectures: Dict[int, Architecture] = {}
        self.next_architecture_id = 0
    
    def create_architecture(self,
                          name: str,
                          layers: List[int],
                          connections: Optional[Dict[Tuple[int, int], float]] = None) -> Architecture:
        """Create a new architecture"""
        if connections is None:
            connections = {}
            for i in range(len(layers) - 1):
                connections[(i, i + 1)] = 1.0
        
        architecture = Architecture(
            architecture_id=self.next_architecture_id,
            name=name,
            layers=layers,
            connections=connections,
            created_time=time.time()
        )
        
        self.architectures[self.next_architecture_id] = architecture
        self.next_architecture_id += 1
        
        return architecture
    
    def mutate_architecture(self, architecture: Architecture) -> Architecture:
        """Create mutated version of architecture"""
        # Mutate layer sizes
        mutated_layers = []
        for layer_size in architecture.layers:
            mutation = np.random.normal(0, layer_size * 0.1)
            new_size = max(1, int(layer_size + mutation))
            mutated_layers.append(new_size)
        
        # Mutate connections
        mutated_connections = {}
        for (from_layer, to_layer), strength in architecture.connections.items():
            mutation = np.random.normal(0, 0.1)
            new_strength = max(0.0, min(1.0, strength + mutation))
            mutated_connections[(from_layer, to_layer)] = new_strength
        
        return self.create_architecture(
            name=f"{architecture.name}_mutated",
            layers=mutated_layers,
            connections=mutated_connections
        )
    
    def search_better_architecture(self,
                                  base_architecture: Architecture,
                                  performance_function: Callable) -> Architecture:
        """
        Search for better architecture
        
        Returns:
            Best architecture found
        """
        best_architecture = base_architecture
        best_performance = performance_function(base_architecture)
        
        # Generate candidate architectures
        candidates = [base_architecture]
        for _ in range(self.population_size - 1):
            candidate = self.mutate_architecture(base_architecture)
            candidate.performance = performance_function(candidate)
            candidates.append(candidate)
            
            if candidate.performance > best_performance:
                best_performance = candidate.performance
                best_architecture = candidate
        
        return best_architecture


class AlgorithmDiscovery:
    """
    Algorithm Discovery
    
    Discovers new learning algorithms
    Creates algorithm variations
    """
    
    def __init__(self):
        self.algorithms: Dict[int, Algorithm] = {}
        self.next_algorithm_id = 0
    
    def create_algorithm(self,
                        name: str,
                        parameters: Dict[str, float]) -> Algorithm:
        """Create a new algorithm"""
        algorithm = Algorithm(
            algorithm_id=self.next_algorithm_id,
            name=name,
            parameters=parameters,
            created_time=time.time()
        )
        
        self.algorithms[self.next_algorithm_id] = algorithm
        self.next_algorithm_id += 1
        
        return algorithm
    
    def discover_algorithm_variation(self,
                                    base_algorithm: Algorithm,
                                    performance_data: Dict[str, float]) -> Algorithm:
        """
        Discover variation of algorithm
        
        Returns:
            New algorithm variation
        """
        # Adjust parameters based on performance
        new_parameters = {}
        for param_name, param_value in base_algorithm.parameters.items():
            # Simple adjustment: increase parameters that correlate with good performance
            if performance_data.get('success', False):
                adjustment = np.random.normal(0.05, 0.02)
            else:
                adjustment = np.random.normal(-0.05, 0.02)
            
            new_value = param_value + adjustment
            new_parameters[param_name] = max(0.0, min(1.0, new_value))
        
        return self.create_algorithm(
            name=f"{base_algorithm.name}_variation",
            parameters=new_parameters
        )


class SelfModification:
    """
    Self-Modification
    
    Modifies own code/parameters
    Updates system configuration
    """
    
    def __init__(self,
                 modification_safety: bool = True):
        self.modification_safety = modification_safety
        self.modifications: List[Dict] = []
    
    def modify_parameter(self,
                        parameter_name: str,
                        current_value: float,
                        new_value: float,
                        reason: str = '') -> bool:
        """
        Modify a system parameter
        
        Returns:
            Success status
        """
        if self.modification_safety:
            # Safety check: ensure value is in reasonable range
            if new_value < 0.0 or new_value > 1.0:
                return False
        
        modification = {
            'parameter_name': parameter_name,
            'old_value': current_value,
            'new_value': new_value,
            'reason': reason,
            'timestamp': time.time()
        }
        
        self.modifications.append(modification)
        return True
    
    def modify_architecture(self,
                          current_architecture: Architecture,
                          new_layers: List[int],
                          architecture_search: 'ArchitectureSearch') -> Architecture:
        """Modify architecture"""
        # Create new architecture
        new_architecture = architecture_search.create_architecture(
            name=f"{current_architecture.name}_modified",
            layers=new_layers
        )
        
        self.modifications.append({
            'type': 'architecture',
            'old_architecture_id': current_architecture.architecture_id,
            'new_architecture_id': new_architecture.architecture_id,
            'timestamp': time.time()
        })
        
        return new_architecture


class CapabilityExpansion:
    """
    Capability Expansion
    
    Adds new capabilities
    Extends system functionality
    """
    
    def __init__(self):
        self.capabilities: Dict[int, Capability] = {}
        self.next_capability_id = 0
    
    def add_capability(self,
                      name: str,
                      description: str,
                      initial_strength: float = 0.5) -> Capability:
        """Add a new capability"""
        capability = Capability(
            capability_id=self.next_capability_id,
            name=name,
            description=description,
            strength=initial_strength,
            enabled=True,
            created_time=time.time()
        )
        
        self.capabilities[self.next_capability_id] = capability
        self.next_capability_id += 1
        
        return capability
    
    def enhance_capability(self,
                          capability_id: int,
                          strength_increase: float = 0.1) -> bool:
        """Enhance an existing capability"""
        if capability_id not in self.capabilities:
            return False
        
        capability = self.capabilities[capability_id]
        capability.strength = min(1.0, capability.strength + strength_increase)
        
        return True
    
    def enable_capability(self, capability_id: int) -> bool:
        """Enable a capability"""
        if capability_id in self.capabilities:
            self.capabilities[capability_id].enabled = True
            return True
        return False
    
    def disable_capability(self, capability_id: int) -> bool:
        """Disable a capability"""
        if capability_id in self.capabilities:
            self.capabilities[capability_id].enabled = False
            return True
        return False


class PerformanceMonitoring:
    """
    Performance Monitoring
    
    Monitors own performance
    Tracks improvement metrics
    """
    
    def __init__(self):
        self.performance_history: Dict[str, List[Tuple[float, float]]] = {}  # metric -> [(value, time)]
        self.baseline_performance: Dict[str, float] = {}
    
    def record_performance(self,
                         metric_name: str,
                         value: float):
        """Record performance metric"""
        if metric_name not in self.performance_history:
            self.performance_history[metric_name] = []
            self.baseline_performance[metric_name] = value
        
        self.performance_history[metric_name].append((value, time.time()))
        
        # Limit history
        if len(self.performance_history[metric_name]) > 1000:
            self.performance_history[metric_name] = self.performance_history[metric_name][-1000:]
    
    def compute_improvement(self, metric_name: str) -> float:
        """
        Compute improvement in a metric
        
        Returns:
            Improvement ratio (1.0 = no change, >1.0 = improvement)
        """
        if metric_name not in self.performance_history:
            return 1.0
        
        history = self.performance_history[metric_name]
        if len(history) < 2:
            return 1.0
        
        baseline = self.baseline_performance.get(metric_name, history[0][0])
        current = history[-1][0]
        
        if baseline == 0:
            return 1.0
        
        return current / baseline
    
    def get_performance_trend(self, metric_name: str) -> str:
        """Get performance trend"""
        if metric_name not in self.performance_history:
            return 'unknown'
        
        history = self.performance_history[metric_name]
        if len(history) < 2:
            return 'stable'
        
        recent_values = [v for v, _ in history[-10:]]
        if len(recent_values) < 2:
            return 'stable'
        
        trend = recent_values[-1] - recent_values[0]
        
        if trend > 0.1:
            return 'improving'
        elif trend < -0.1:
            return 'declining'
        else:
            return 'stable'


class SelfImprovementSystem:
    """
    Self-Improvement System Manager
    
    Integrates all self-improvement components
    """
    
    def __init__(self,
                 brain_system=None,
                 self_model: Optional[SelfModelSystem] = None,
                 metacognition: Optional[MetacognitiveMonitoring] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.architecture_search = ArchitectureSearch()
        self.algorithm_discovery = AlgorithmDiscovery()
        self.self_modification = SelfModification()
        self.capability_expansion = CapabilityExpansion()
        self.performance_monitoring = PerformanceMonitoring()
        
        # Integration with existing systems
        self.self_model = self_model
        self.metacognition = metacognition
        
        # Current architecture and algorithms
        self.current_architecture: Optional[Architecture] = None
        self.current_algorithms: Dict[str, Algorithm] = {}
        
        # Statistics
        self.stats = {
            'architectures_searched': 0,
            'algorithms_discovered': 0,
            'modifications_made': 0,
            'capabilities_added': 0,
            'performance_improvements': 0
        }
    
    def improve_architecture(self,
                           performance_function: Callable) -> Optional[Architecture]:
        """Improve architecture through search"""
        if self.current_architecture is None:
            # Create initial architecture
            self.current_architecture = self.architecture_search.create_architecture(
                name="initial",
                layers=[100, 50, 10]
            )
        
        # Search for better architecture
        improved = self.architecture_search.search_better_architecture(
            self.current_architecture, performance_function
        )
        
        if improved.performance > self.current_architecture.performance:
            self.current_architecture = improved
            self.stats['architectures_searched'] += 1
            return improved
        
        return None
    
    def discover_better_algorithm(self,
                                  algorithm_name: str,
                                  performance_data: Dict[str, float]) -> Optional[Algorithm]:
        """Discover better algorithm variation"""
        if algorithm_name in self.current_algorithms:
            base_algorithm = self.current_algorithms[algorithm_name]
            variation = self.algorithm_discovery.discover_algorithm_variation(
                base_algorithm, performance_data
            )
            
            if variation.performance > base_algorithm.performance:
                self.current_algorithms[algorithm_name] = variation
                self.stats['algorithms_discovered'] += 1
                return variation
        
        return None
    
    def add_new_capability(self,
                          name: str,
                          description: str) -> Capability:
        """Add a new capability"""
        capability = self.capability_expansion.add_capability(name, description)
        self.stats['capabilities_added'] += 1
        return capability
    
    def monitor_and_improve(self,
                           metric_name: str,
                           current_value: float):
        """Monitor performance and trigger improvements"""
        # Record performance
        self.performance_monitoring.record_performance(metric_name, current_value)
        
        # Check for improvement
        improvement = self.performance_monitoring.compute_improvement(metric_name)
        
        if improvement > 1.1:  # 10% improvement
            self.stats['performance_improvements'] += 1
    
    def get_statistics(self) -> Dict:
        """Get self-improvement statistics"""
        stats = self.stats.copy()
        
        # Add performance trends
        stats['performance_trends'] = {}
        for metric_name in self.performance_monitoring.performance_history.keys():
            stats['performance_trends'][metric_name] = (
                self.performance_monitoring.get_performance_trend(metric_name)
            )
        
        return stats

