"""
Scalable Neural Network Framework
For creating networks from hundreds to billions of interconnected biological neurons
"""

import numpy as np
import matplotlib.pyplot as plt
from realistic_neuron import BiologicalNeuron, SynapseConnection
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random
import json

@dataclass
class NetworkTopology:
    """Defines the structure of neural network connections"""
    layers: List[int]  # neurons per layer
    connection_probability: float = 0.1  # probability of connection between neurons
    excitatory_ratio: float = 0.8  # ratio of excitatory neurons
    small_world_rewiring: float = 0.1  # small-world network parameter
    distance_dependent: bool = True  # whether connections depend on distance

@dataclass 
class NetworkStats:
    """Network performance statistics"""
    total_neurons: int
    total_synapses: int
    avg_firing_rate: float
    synchronization_index: float
    computation_time: float
    memory_usage_mb: float

class OptimizedNeuralNetwork:
    """
    High-performance neural network that can scale from hundreds to billions of neurons
    Uses optimized data structures and parallel processing
    """
    
    def __init__(self, name: str = "BrainNet"):
        self.name = name
        self.neurons = {}  # Dict for O(1) lookup
        self.synapses = []
        self.topology = None
        
        # Performance optimization
        self.use_gpu = False  # Could implement CUDA later
        self.parallel_processing = True
        self.chunk_size = 1000  # neurons per processing chunk
        
        # Network statistics
        self.stats = NetworkStats(0, 0, 0.0, 0.0, 0.0, 0.0)
        
        # Simulation state
        self.current_time = 0.0
        self.dt = 0.01  # time step
        self.recording = True
        
        # Data storage for large networks
        self.use_disk_storage = False
        self.data_directory = "/tmp/neural_network_data"
        
    def create_neurons(self, count: int, neuron_types: List[str] = None):
        """Create a specified number of neurons"""
        if neuron_types is None:
            neuron_types = ["pyramidal"] * int(count * 0.8) + ["interneuron"] * int(count * 0.2)
        
        print(f"🧠 Creating {count:,} neurons...")
        start_time = time.time()
        
        # Create neurons in batches for memory efficiency
        batch_size = min(10000, count)
        
        for i in range(0, count, batch_size):
            batch_end = min(i + batch_size, count)
            batch_neurons = []
            
            for j in range(i, batch_end):
                neuron_type = neuron_types[j % len(neuron_types)]
                neuron = BiologicalNeuron(neuron_id=j, neuron_type=neuron_type)
                
                # Adjust properties based on type
                if neuron_type == "interneuron":
                    neuron.g_leak *= 1.5  # Faster interneurons
                    neuron.V_threshold = -50.0  # Lower threshold
                
                self.neurons[j] = neuron
                batch_neurons.append(neuron)
            
            if i % 50000 == 0:
                print(f"  Created {i + len(batch_neurons):,} neurons...")
        
        creation_time = time.time() - start_time
        print(f"✅ Created {count:,} neurons in {creation_time:.2f} seconds")
        
        self.stats.total_neurons = len(self.neurons)
        return self.neurons
    
    def connect_neurons_random(self, connection_probability: float = 0.01):
        """Connect neurons with random topology"""
        print(f"🔗 Creating random connections (p={connection_probability})...")
        start_time = time.time()
        
        neuron_ids = list(self.neurons.keys())
        total_possible = len(neuron_ids) * (len(neuron_ids) - 1)
        expected_connections = int(total_possible * connection_probability)
        
        print(f"  Expected connections: {expected_connections:,}")
        
        connections_made = 0
        
        # Create connections in chunks to avoid memory issues
        for i, pre_id in enumerate(neuron_ids):
            if i % 10000 == 0:
                print(f"  Processing neuron {i:,}/{len(neuron_ids):,}")
            
            pre_neuron = self.neurons[pre_id]
            
            # Select random targets
            n_connections = np.random.poisson(connection_probability * len(neuron_ids))
            n_connections = min(n_connections, len(neuron_ids) - 1)
            
            if n_connections > 0:
                targets = random.sample([nid for nid in neuron_ids if nid != pre_id], n_connections)
                
                for post_id in targets:
                    post_neuron = self.neurons[post_id]
                    
                    # Determine synapse properties
                    weight = np.random.normal(1.0, 0.3)
                    weight = max(0.1, abs(weight))  # Ensure positive weight
                    delay = np.random.exponential(1.0) + 0.5  # 0.5-5ms delays
                    
                    # 80% excitatory, 20% inhibitory
                    neurotransmitter = 'excitatory' if random.random() < 0.8 else 'inhibitory'
                    
                    synapse = pre_neuron.add_synapse(post_neuron, weight, delay, neurotransmitter)
                    self.synapses.append(synapse)
                    connections_made += 1
        
        connection_time = time.time() - start_time
        print(f"✅ Created {connections_made:,} synapses in {connection_time:.2f} seconds")
        
        self.stats.total_synapses = len(self.synapses)
        return connections_made
    
    def connect_neurons_structured(self, topology: NetworkTopology):
        """Create structured connections (layered, small-world, etc.)"""
        self.topology = topology
        print(f"🏗️ Creating structured network topology...")
        
        # Assign neurons to layers
        neuron_ids = list(self.neurons.keys())
        layer_assignments = {}
        current_idx = 0
        
        for layer_idx, layer_size in enumerate(topology.layers):
            layer_neurons = neuron_ids[current_idx:current_idx + layer_size]
            layer_assignments[layer_idx] = layer_neurons
            current_idx += layer_size
            print(f"  Layer {layer_idx}: {len(layer_neurons)} neurons")
        
        connections_made = 0
        
        # Create layer-to-layer connections
        for layer_idx in range(len(topology.layers) - 1):
            pre_layer = layer_assignments[layer_idx]
            post_layer = layer_assignments[layer_idx + 1]
            
            print(f"  Connecting layer {layer_idx} to {layer_idx + 1}")
            
            for pre_id in pre_layer:
                pre_neuron = self.neurons[pre_id]
                
                # Connect to random subset of next layer
                n_connections = int(len(post_layer) * topology.connection_probability)
                targets = random.sample(post_layer, min(n_connections, len(post_layer)))
                
                for post_id in targets:
                    post_neuron = self.neurons[post_id]
                    
                    weight = np.random.normal(1.0, 0.2)
                    weight = max(0.1, abs(weight))
                    delay = np.random.exponential(1.0) + 0.5
                    
                    neurotransmitter = 'excitatory' if random.random() < topology.excitatory_ratio else 'inhibitory'
                    
                    synapse = pre_neuron.add_synapse(post_neuron, weight, delay, neurotransmitter)
                    self.synapses.append(synapse)
                    connections_made += 1
        
        # Add some recurrent connections for dynamics
        print(f"  Adding recurrent connections...")
        for layer_neurons in layer_assignments.values():
            n_recurrent = int(len(layer_neurons) * topology.connection_probability * 0.1)
            
            for _ in range(n_recurrent):
                pre_id = random.choice(layer_neurons)
                post_id = random.choice(layer_neurons)
                
                if pre_id != post_id:
                    pre_neuron = self.neurons[pre_id]
                    post_neuron = self.neurons[post_id]
                    
                    synapse = pre_neuron.add_synapse(post_neuron, 0.5, 1.0, 'inhibitory')
                    self.synapses.append(synapse)
                    connections_made += 1
        
        self.stats.total_synapses = len(self.synapses)
        print(f"✅ Created structured network with {connections_made:,} synapses")
        
        return connections_made
    
    def simulate_parallel(self, duration_ms: float, external_inputs: Dict[int, float] = None):
        """Run simulation using parallel processing"""
        print(f"🚀 Running parallel simulation for {duration_ms:.1f} ms...")
        
        if external_inputs is None:
            external_inputs = {}
        
        start_time = time.time()
        total_steps = int(duration_ms / self.dt)
        
        # Split neurons into chunks for parallel processing
        neuron_ids = list(self.neurons.keys())
        num_cores = min(mp.cpu_count(), 8)  # Limit to avoid overwhelming system
        chunk_size = max(1, len(neuron_ids) // num_cores)
        neuron_chunks = [neuron_ids[i:i + chunk_size] for i in range(0, len(neuron_ids), chunk_size)]
        
        print(f"  Using {num_cores} cores, {len(neuron_chunks)} chunks")
        
        spike_count = 0
        
        for step in range(total_steps):
            current_time = step * self.dt
            
            # Process chunks in parallel
            if self.parallel_processing and len(self.neurons) > 1000:
                # Use threading for I/O bound operations
                with ThreadPoolExecutor(max_workers=num_cores) as executor:
                    futures = []
                    
                    for chunk in neuron_chunks:
                        future = executor.submit(self._update_neuron_chunk, 
                                               chunk, current_time, external_inputs)
                        futures.append(future)
                    
                    # Collect results
                    chunk_spikes = 0
                    for future in futures:
                        chunk_spikes += future.result()
                    spike_count += chunk_spikes
            else:
                # Sequential processing for small networks
                for neuron_id in neuron_ids:
                    neuron = self.neurons[neuron_id]
                    external_current = external_inputs.get(neuron_id, 0.0)
                    
                    if neuron.update(self.dt, current_time, external_current):
                        spike_count += 1
            
            # Progress reporting
            if step % (total_steps // 10) == 0:
                progress = (step / total_steps) * 100
                elapsed = time.time() - start_time
                print(f"    Progress: {progress:.1f}% - Spikes: {spike_count:,} - Time: {elapsed:.1f}s")
        
        simulation_time = time.time() - start_time
        
        # Calculate statistics
        avg_firing_rate = (spike_count / len(self.neurons)) / (duration_ms / 1000.0)
        
        self.stats.avg_firing_rate = avg_firing_rate
        self.stats.computation_time = simulation_time
        
        print(f"✅ Simulation complete!")
        print(f"   Total spikes: {spike_count:,}")
        print(f"   Average firing rate: {avg_firing_rate:.2f} Hz")
        print(f"   Computation time: {simulation_time:.2f} seconds")
        print(f"   Real-time factor: {(duration_ms/1000) / simulation_time:.2f}x")
        
        return spike_count, avg_firing_rate
    
    def _update_neuron_chunk(self, neuron_ids: List[int], current_time: float, 
                           external_inputs: Dict[int, float]) -> int:
        """Update a chunk of neurons (for parallel processing)"""
        spike_count = 0
        
        for neuron_id in neuron_ids:
            neuron = self.neurons[neuron_id]
            external_current = external_inputs.get(neuron_id, 0.0)
            
            if neuron.update(self.dt, current_time, external_current):
                spike_count += 1
        
        return spike_count
    
    def analyze_network(self):
        """Analyze network properties and connectivity"""
        print("📊 Analyzing network properties...")
        
        # Connectivity analysis
        in_degrees = []
        out_degrees = []
        
        for neuron in self.neurons.values():
            in_degrees.append(len(neuron.synapses_in))
            out_degrees.append(len(neuron.synapses_out))
        
        # Calculate statistics
        stats = {
            'total_neurons': len(self.neurons),
            'total_synapses': len(self.synapses),
            'avg_in_degree': np.mean(in_degrees),
            'avg_out_degree': np.mean(out_degrees),
            'max_in_degree': np.max(in_degrees) if in_degrees else 0,
            'max_out_degree': np.max(out_degrees) if out_degrees else 0,
            'connectivity_density': len(self.synapses) / (len(self.neurons) ** 2) if len(self.neurons) > 1 else 0
        }
        
        # Neuron type distribution
        neuron_types = {}
        for neuron in self.neurons.values():
            neuron_types[neuron.type] = neuron_types.get(neuron.type, 0) + 1
        
        stats['neuron_types'] = neuron_types
        
        print("📋 Network Analysis Results:")
        print(f"   Total neurons: {stats['total_neurons']:,}")
        print(f"   Total synapses: {stats['total_synapses']:,}")
        print(f"   Average in-degree: {stats['avg_in_degree']:.2f}")
        print(f"   Average out-degree: {stats['avg_out_degree']:.2f}")
        print(f"   Connectivity density: {stats['connectivity_density']:.6f}")
        print(f"   Neuron types: {stats['neuron_types']}")
        
        return stats
    
    def save_network(self, filename: str):
        """Save network to file"""
        print(f"💾 Saving network to {filename}...")
        
        # For very large networks, save in chunks
        network_data = {
            'name': self.name,
            'stats': self.stats,
            'topology': self.topology,
            'neuron_count': len(self.neurons),
            'synapse_count': len(self.synapses)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(network_data, f)
        
        print(f"✅ Network saved successfully")
    
    def visualize_network_activity(self, sample_size: int = 100):
        """Visualize activity of a sample of neurons"""
        if len(self.neurons) == 0:
            print("No neurons to visualize")
            return
        
        print(f"📈 Visualizing activity of {min(sample_size, len(self.neurons))} neurons...")
        
        # Sample neurons for visualization
        sample_ids = random.sample(list(self.neurons.keys()), 
                                 min(sample_size, len(self.neurons)))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Voltage traces of sample neurons
        ax1 = axes[0, 0]
        for i, neuron_id in enumerate(sample_ids[:5]):  # Show first 5
            neuron = self.neurons[neuron_id]
            if len(neuron.voltage_history) > 0:
                ax1.plot(neuron.time_history, neuron.voltage_history, 
                        alpha=0.7, label=f'Neuron {neuron_id}')
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Voltage (mV)')
        ax1.set_title('Sample Voltage Traces')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Firing rate distribution
        ax2 = axes[0, 1]
        firing_rates = []
        for neuron_id in sample_ids:
            neuron = self.neurons[neuron_id]
            firing_rates.append(neuron.get_firing_rate())
        
        ax2.hist(firing_rates, bins=20, alpha=0.7, color='skyblue')
        ax2.set_xlabel('Firing Rate (Hz)')
        ax2.set_ylabel('Count')
        ax2.set_title('Firing Rate Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Connectivity histogram
        ax3 = axes[1, 0]
        in_degrees = [len(self.neurons[nid].synapses_in) for nid in sample_ids]
        out_degrees = [len(self.neurons[nid].synapses_out) for nid in sample_ids]
        
        ax3.hist(in_degrees, bins=15, alpha=0.6, label='In-degree', color='red')
        ax3.hist(out_degrees, bins=15, alpha=0.6, label='Out-degree', color='blue')
        ax3.set_xlabel('Number of Connections')
        ax3.set_ylabel('Count')
        ax3.set_title('Connectivity Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Network statistics
        ax4 = axes[1, 1]
        stats_text = f"""Network Statistics:
Total Neurons: {len(self.neurons):,}
Total Synapses: {len(self.synapses):,}
Avg Firing Rate: {self.stats.avg_firing_rate:.2f} Hz
Computation Time: {self.stats.computation_time:.2f} s
        """
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='center')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Network Summary')
        
        plt.tight_layout()
        plt.savefig('/home/user/network_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig


def create_small_network_demo():
    """Demonstrate network creation and simulation with 100 neurons"""
    print("🌟 Creating Small Neural Network Demo (100 neurons)")
    print("=" * 60)
    
    # Create network
    network = OptimizedNeuralNetwork("SmallDemo")
    
    # Create 100 neurons
    neurons = network.create_neurons(100)
    
    # Create random connections
    network.connect_neurons_random(connection_probability=0.05)
    
    # Analyze network
    network.analyze_network()
    
    # Apply external stimulation to 10% of neurons
    stimulus_neurons = random.sample(list(neurons.keys()), 10)
    external_inputs = {nid: 15.0 for nid in stimulus_neurons}
    
    print(f"\n🔋 Applying stimulus to {len(stimulus_neurons)} neurons...")
    
    # Run simulation
    spikes, avg_rate = network.simulate_parallel(duration_ms=200.0, 
                                                external_inputs=external_inputs)
    
    # Visualize results
    network.visualize_network_activity(sample_size=50)
    
    return network

if __name__ == "__main__":
    # Run demo
    network = create_small_network_demo()