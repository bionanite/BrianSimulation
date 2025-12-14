#!/usr/bin/env python3
"""
Structural Plasticity Mechanisms
Implements synapse creation/deletion, dendritic growth, and neurogenesis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
from plasticity_mechanisms import SynapticConnection

@dataclass
class DendriticBranch:
    """Represents a dendritic branch"""
    branch_id: int
    parent_neuron_id: int
    length: float  # micrometers
    diameter: float  # micrometers
    activity: float = 0.0
    synapses: List[int] = field(default_factory=list)  # Synapse IDs on this branch
    growth_factor: float = 0.0
    age: float = 0.0  # Time since creation
    
@dataclass
class NeuronStructure:
    """Tracks structural properties of a neuron"""
    neuron_id: int
    dendritic_branches: List[DendriticBranch] = field(default_factory=list)
    total_dendritic_length: float = 0.0
    synapse_count: int = 0
    creation_time: float = 0.0
    last_activity: float = 0.0

class SynapseCreationDeletion:
    """
    Manages synapse creation and deletion based on activity
    
    Creates synapses between frequently co-active neurons
    Removes unused synapses
    """
    
    def __init__(self,
                 creation_threshold: float = 0.7,  # Co-activity threshold for creation
                 deletion_threshold: float = 0.1,  # Activity threshold for deletion
                 max_synapses_per_pair: int = 1,   # Max synapses between neuron pair
                 creation_probability: float = 0.01):  # Probability of creation per check
        self.creation_threshold = creation_threshold
        self.deletion_threshold = deletion_threshold
        self.max_synapses_per_pair = max_synapses_per_pair
        self.creation_probability = creation_probability
        
        # Track co-activity
        self.co_activity_matrix: Dict[Tuple[int, int], float] = {}
        self.synapse_activity: Dict[int, float] = {}  # synapse_id -> activity
    
    def update_co_activity(self,
                          neuron_pairs: List[Tuple[int, int]],
                          current_time: float,
                          time_window: float = 100.0):
        """Update co-activity tracking for neuron pairs"""
        for pre_id, post_id in neuron_pairs:
            pair = (pre_id, post_id)
            if pair not in self.co_activity_matrix:
                self.co_activity_matrix[pair] = 0.0
            
            # Increment co-activity (decay over time)
            self.co_activity_matrix[pair] *= np.exp(-1.0 / time_window)
            self.co_activity_matrix[pair] += 1.0
    
    def should_create_synapse(self,
                             pre_neuron_id: int,
                             post_neuron_id: int,
                             existing_synapses: List[SynapticConnection]) -> bool:
        """Determine if a synapse should be created"""
        # Check if max synapses already exist
        pair_count = sum(1 for s in existing_synapses 
                        if s.pre_neuron_id == pre_neuron_id and 
                           s.post_neuron_id == post_neuron_id)
        
        if pair_count >= self.max_synapses_per_pair:
            return False
        
        # Check co-activity
        pair = (pre_neuron_id, post_neuron_id)
        co_activity = self.co_activity_matrix.get(pair, 0.0)
        
        # Normalize co-activity (rough estimate)
        normalized_co_activity = min(1.0, co_activity / 10.0)
        
        # Create if co-activity high enough and random chance
        if normalized_co_activity >= self.creation_threshold:
            return np.random.random() < self.creation_probability
        
        return False
    
    def create_synapse(self,
                      pre_neuron_id: int,
                      post_neuron_id: int,
                      initial_weight: float = 0.5,
                      delay: float = 1.0,
                      neurotransmitter: str = 'excitatory') -> SynapticConnection:
        """Create a new synapse"""
        synapse = SynapticConnection(
            pre_neuron_id=pre_neuron_id,
            post_neuron_id=post_neuron_id,
            weight=initial_weight,
            delay=delay,
            neurotransmitter=neurotransmitter,
            initial_weight=initial_weight
        )
        
        self.synapse_activity[id(synapse)] = 0.0
        return synapse
    
    def update_synapse_activity(self,
                               synapse: SynapticConnection,
                               activity: float):
        """Update activity tracking for a synapse"""
        synapse_id = id(synapse)
        if synapse_id not in self.synapse_activity:
            self.synapse_activity[synapse_id] = 0.0
        
        # Decay and add new activity
        self.synapse_activity[synapse_id] *= 0.9
        self.synapse_activity[synapse_id] += activity
    
    def should_delete_synapse(self, synapse: SynapticConnection) -> bool:
        """Determine if a synapse should be deleted"""
        synapse_id = id(synapse)
        activity = self.synapse_activity.get(synapse_id, 0.0)
        
        # Delete if activity below threshold
        return activity < self.deletion_threshold
    
    def prune_synapses(self,
                      synapses: List[SynapticConnection],
                      current_time: float) -> List[SynapticConnection]:
        """Remove synapses that should be deleted"""
        to_keep = []
        deleted_count = 0
        
        for synapse in synapses:
            if self.should_delete_synapse(synapse):
                deleted_count += 1
                # Remove from activity tracking
                synapse_id = id(synapse)
                if synapse_id in self.synapse_activity:
                    del self.synapse_activity[synapse_id]
            else:
                to_keep.append(synapse)
        
        return to_keep, deleted_count

class DendriticGrowth:
    """
    Manages dendritic branch growth and pruning
    
    Grows branches in active regions
    Prunes inactive branches
    """
    
    def __init__(self,
                 growth_rate: float = 0.1,  # micrometers per ms per unit activity
                 pruning_threshold: float = 0.05,  # Activity threshold for pruning
                 max_branch_length: float = 1000.0,  # micrometers
                 min_branch_length: float = 10.0):  # micrometers
        self.growth_rate = growth_rate
        self.pruning_threshold = pruning_threshold
        self.max_branch_length = max_branch_length
        self.min_branch_length = min_branch_length
    
    def update_branch_activity(self,
                              branch: DendriticBranch,
                              activity: float):
        """Update activity for a branch"""
        branch.activity *= 0.95  # Decay
        branch.activity += activity * 0.05
    
    def grow_branch(self,
                   branch: DendriticBranch,
                   activity: float,
                   dt: float) -> float:
        """
        Grow a branch based on activity
        
        Returns:
            Length change
        """
        if activity > self.pruning_threshold:
            # Growth proportional to activity
            growth = self.growth_rate * activity * dt
            new_length = branch.length + growth
            branch.length = min(new_length, self.max_branch_length)
            branch.growth_factor = activity
            return growth
        else:
            # No growth if activity too low
            branch.growth_factor = 0.0
            return 0.0
    
    def prune_branch(self,
                    branch: DendriticBranch,
                    current_time: float) -> bool:
        """
        Prune a branch if inactive
        
        Returns:
            True if branch should be pruned
        """
        # Prune if activity very low and branch is old
        age = current_time - branch.age
        if branch.activity < self.pruning_threshold and age > 1000.0:
            return True
        return False
    
    def create_new_branch(self,
                         neuron_id: int,
                         parent_branch_id: Optional[int],
                         initial_length: float = 20.0,
                         initial_diameter: float = 1.0) -> DendriticBranch:
        """Create a new dendritic branch"""
        branch_id = np.random.randint(1000000, 9999999)  # Random ID
        branch = DendriticBranch(
            branch_id=branch_id,
            parent_neuron_id=neuron_id,
            length=initial_length,
            diameter=initial_diameter,
            age=time.time()
        )
        return branch
    
    def update_dendritic_structure(self,
                                   neuron_structure: NeuronStructure,
                                   activity_map: Dict[int, float],
                                   current_time: float,
                                   dt: float) -> Dict:
        """
        Update dendritic structure for a neuron
        
        Returns:
            Statistics about changes
        """
        stats = {
            'branches_grown': 0,
            'branches_pruned': 0,
            'total_length_change': 0.0,
            'new_branches': 0
        }
        
        neuron_id = neuron_structure.neuron_id
        neuron_activity = activity_map.get(neuron_id, 0.0)
        
        # Update existing branches
        branches_to_remove = []
        for branch in neuron_structure.dendritic_branches:
            branch_activity = branch.activity
            
            # Grow or prune branch
            if self.prune_branch(branch, current_time):
                branches_to_remove.append(branch)
                stats['branches_pruned'] += 1
            else:
                length_change = self.grow_branch(branch, branch_activity, dt)
                if length_change > 0:
                    stats['branches_grown'] += 1
                    stats['total_length_change'] += length_change
        
        # Remove pruned branches
        for branch in branches_to_remove:
            neuron_structure.dendritic_branches.remove(branch)
        
        # Create new branches if activity high
        if neuron_activity > 0.5 and len(neuron_structure.dendritic_branches) < 20:
            if np.random.random() < 0.001:  # Low probability
                new_branch = self.create_new_branch(neuron_id, None)
                neuron_structure.dendritic_branches.append(new_branch)
                stats['new_branches'] += 1
        
        # Update total dendritic length
        neuron_structure.total_dendritic_length = sum(
            b.length for b in neuron_structure.dendritic_branches
        )
        
        return stats

class Neurogenesis:
    """
    Manages creation of new neurons
    
    Creates new neurons in active regions
    Integrates new neurons into network
    """
    
    def __init__(self,
                 neurogenesis_rate: float = 0.0001,  # Probability per update
                 integration_time: float = 1000.0,  # Time to integrate (ms)
                 initial_connections: int = 10):  # Initial connections for new neuron
        self.neurogenesis_rate = neurogenesis_rate
        self.integration_time = integration_time
        self.initial_connections = initial_connections
        self.new_neurons: Dict[int, float] = {}  # neuron_id -> creation_time
    
    def should_create_neuron(self,
                            region_activity: float,
                            current_neuron_count: int,
                            max_neurons: int) -> bool:
        """Determine if a new neuron should be created"""
        # Don't create if at max capacity
        if current_neuron_count >= max_neurons:
            return False
        
        # Higher activity increases probability
        activity_factor = min(1.0, region_activity)
        probability = self.neurogenesis_rate * activity_factor
        
        return np.random.random() < probability
    
    def create_neuron(self,
                     neuron_id: int,
                     neuron_type: str = "pyramidal",
                     current_time: float = 0.0) -> Dict:
        """Create a new neuron structure"""
        self.new_neurons[neuron_id] = current_time
        
        return {
            'neuron_id': neuron_id,
            'neuron_type': neuron_type,
            'creation_time': current_time,
            'is_integrated': False,
            'connection_count': 0
        }
    
    def integrate_neuron(self,
                        neuron_id: int,
                        synapses: List[SynapticConnection],
                        target_neurons: List[int],
                        current_time: float,
                        synapse_creator) -> List[SynapticConnection]:
        """
        Integrate a new neuron into the network
        
        Creates initial connections to random target neurons
        """
        if neuron_id not in self.new_neurons:
            return synapses
        
        creation_time = self.new_neurons[neuron_id]
        time_since_creation = current_time - creation_time
        
        # Only integrate after integration time
        if time_since_creation < self.integration_time:
            return synapses
        
        # Create initial connections
        new_synapses = []
        targets = np.random.choice(
            target_neurons,
            size=min(self.initial_connections, len(target_neurons)),
            replace=False
        )
        
        for target_id in targets:
            # Create both incoming and outgoing connections
            synapse_in = synapse_creator(target_id, neuron_id)
            synapse_out = synapse_creator(neuron_id, target_id)
            new_synapses.extend([synapse_in, synapse_out])
        
        synapses.extend(new_synapses)
        
        # Mark as integrated
        if neuron_id in self.new_neurons:
            del self.new_neurons[neuron_id]
        
        return synapses, len(new_synapses)

class StructuralPlasticityManager:
    """
    Manages all structural plasticity mechanisms
    """
    
    def __init__(self,
                 enable_synapse_creation: bool = True,
                 enable_dendritic_growth: bool = True,
                 enable_neurogenesis: bool = True):
        self.enable_synapse_creation = enable_synapse_creation
        self.enable_dendritic_growth = enable_dendritic_growth
        self.enable_neurogenesis = enable_neurogenesis
        
        self.synapse_manager = SynapseCreationDeletion() if enable_synapse_creation else None
        self.dendritic_growth = DendriticGrowth() if enable_dendritic_growth else None
        self.neurogenesis = Neurogenesis() if enable_neurogenesis else None
        
        self.neuron_structures: Dict[int, NeuronStructure] = {}
        self.structural_history = []
    
    def initialize_neuron_structure(self, neuron_id: int):
        """Initialize structural tracking for a neuron"""
        if neuron_id not in self.neuron_structures:
            structure = NeuronStructure(
                neuron_id=neuron_id,
                creation_time=time.time()
            )
            # Create initial dendritic branches
            for i in range(3):
                branch = self.dendritic_growth.create_new_branch(
                    neuron_id, None, initial_length=50.0
                ) if self.dendritic_growth else None
                if branch:
                    structure.dendritic_branches.append(branch)
            
            self.neuron_structures[neuron_id] = structure
    
    def update_structures(self,
                         synapses: List[SynapticConnection],
                         neurons: List[Dict],
                         activity_map: Dict[int, float],
                         current_time: float,
                         dt: float) -> Dict:
        """
        Update all structural plasticity mechanisms
        
        Returns:
            Statistics about structural changes
        """
        stats = {
            'synapses_created': 0,
            'synapses_deleted': 0,
            'branches_grown': 0,
            'branches_pruned': 0,
            'new_branches': 0,
            'neurons_created': 0,
            'total_synapses': len(synapses),
            'total_neurons': len(neurons)
        }
        
        # Initialize structures for all neurons
        for neuron in neurons:
            neuron_id = neuron.get('id', len(self.neuron_structures))
            self.initialize_neuron_structure(neuron_id)
        
        # Synapse creation/deletion
        if self.enable_synapse_creation and self.synapse_manager:
            # Track co-activity
            active_pairs = []
            for synapse in synapses:
                if synapse.weight > 0.1:  # Active synapse
                    active_pairs.append((synapse.pre_neuron_id, synapse.post_neuron_id))
                    self.synapse_manager.update_synapse_activity(
                        synapse, synapse.weight
                    )
            
            self.synapse_manager.update_co_activity(active_pairs, current_time)
            
            # Create new synapses
            neuron_ids = [n.get('id', i) for i, n in enumerate(neurons)]
            for pre_id in neuron_ids:
                for post_id in neuron_ids:
                    if pre_id != post_id:
                        if self.synapse_manager.should_create_synapse(
                            pre_id, post_id, synapses
                        ):
                            new_synapse = self.synapse_manager.create_synapse(
                                pre_id, post_id
                            )
                            synapses.append(new_synapse)
                            stats['synapses_created'] += 1
            
            # Prune synapses
            synapses, deleted_count = self.synapse_manager.prune_synapses(
                synapses, current_time
            )
            stats['synapses_deleted'] = deleted_count
        
        # Dendritic growth
        if self.enable_dendritic_growth and self.dendritic_growth:
            for neuron_id, structure in self.neuron_structures.items():
                branch_stats = self.dendritic_growth.update_dendritic_structure(
                    structure, activity_map, current_time, dt
                )
                stats['branches_grown'] += branch_stats['branches_grown']
                stats['branches_pruned'] += branch_stats['branches_pruned']
                stats['new_branches'] += branch_stats['new_branches']
        
        # Neurogenesis
        if self.enable_neurogenesis and self.neurogenesis:
            total_activity = sum(activity_map.values()) / max(1, len(activity_map))
            if self.neurogenesis.should_create_neuron(
                total_activity, len(neurons), len(neurons) + 100
            ):
                new_neuron_id = len(neurons)
                neuron_data = self.neurogenesis.create_neuron(
                    new_neuron_id, current_time=current_time
                )
                neurons.append(neuron_data)
                stats['neurons_created'] += 1
        
        stats['total_synapses'] = len(synapses)
        stats['total_neurons'] = len(neurons)
        
        self.structural_history.append({
            'time': current_time,
            'stats': stats.copy()
        })
        
        return stats, synapses, neurons
    
    def get_statistics(self) -> Dict:
        """Get statistics about structural changes"""
        if not self.structural_history:
            return {}
        
        latest = self.structural_history[-1]['stats']
        
        total_synapses_created = sum(
            h['stats']['synapses_created'] for h in self.structural_history
        )
        total_synapses_deleted = sum(
            h['stats']['synapses_deleted'] for h in self.structural_history
        )
        
        return {
            'current_synapses': latest['total_synapses'],
            'current_neurons': latest['total_neurons'],
            'total_synapses_created': total_synapses_created,
            'total_synapses_deleted': total_synapses_deleted,
            'net_synapse_change': total_synapses_created - total_synapses_deleted,
            'total_branches': sum(
                len(s.dendritic_branches) 
                for s in self.neuron_structures.values()
            ),
            'avg_dendritic_length': np.mean([
                s.total_dendritic_length 
                for s in self.neuron_structures.values()
            ]) if self.neuron_structures else 0.0
        }

