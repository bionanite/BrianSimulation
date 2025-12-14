"""
Quick demonstration of neural network scaling
Shows the progression from single neuron to billions
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from realistic_neuron import BiologicalNeuron
import psutil

def demonstrate_neural_scaling():
    """Demonstrate the progression of neural network scaling"""
    
    print("🧠 NEURAL NETWORK SCALING DEMONSTRATION")
    print("=" * 50)
    
    # Get system info
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    print(f"System: {memory_gb:.1f} GB RAM, {cpu_count} CPU cores")
    
    # Define scaling levels
    scales = {
        'Single Neuron': {
            'neurons': 1,
            'description': 'Single biological neuron with Hodgkin-Huxley dynamics',
            'feasible': True,
            'time_estimate': '< 1 second'
        },
        'Small Network': {
            'neurons': 100,
            'description': 'Small network showing basic connectivity',
            'feasible': True,
            'time_estimate': '< 10 seconds'
        },
        'Cortical Column': {
            'neurons': 10000,
            'description': 'Approximates a cortical column',
            'feasible': True,
            'time_estimate': '~ 1 minute'
        },
        'Cortical Area': {
            'neurons': 1000000,
            'description': 'Small cortical area (like V1 patch)',
            'feasible': memory_gb > 8,
            'time_estimate': '~ 10 minutes'
        },
        'Whole Cortex': {
            'neurons': 16000000000,
            'description': 'Human cerebral cortex',
            'feasible': False,
            'time_estimate': '~ 10 hours (with optimization)'
        },
        'Human Brain': {
            'neurons': 86000000000,
            'description': 'Complete human brain',
            'feasible': False,
            'time_estimate': '~ 50 hours (distributed system)'
        }
    }
    
    # Calculate requirements for each scale
    results = {}
    
    print(f"\n📊 SCALING ANALYSIS")
    print("-" * 50)
    
    for scale_name, info in scales.items():
        neuron_count = info['neurons']
        
        # Estimate memory requirements
        bytes_per_neuron = 1000  # Approximate
        bytes_per_synapse = 64
        
        # Estimate connections (sparse connectivity)
        if neuron_count == 1:
            synapses = 0
            connection_prob = 0.0
        elif neuron_count <= 100:
            connection_prob = 0.05
        elif neuron_count <= 10000:
            connection_prob = 0.001
        elif neuron_count <= 1000000:
            connection_prob = 0.0001
        else:
            connection_prob = 0.00001
        
        if neuron_count == 1:
            estimated_synapses = 0
        else:
            estimated_synapses = int(neuron_count * neuron_count * connection_prob)
        
        memory_gb = (neuron_count * bytes_per_neuron + estimated_synapses * bytes_per_synapse) / (1024**3)
        
        # Computational requirements
        ops_per_neuron_per_step = 100  # Hodgkin-Huxley
        steps_per_ms = 100  # dt = 0.01
        ops_per_ms = neuron_count * ops_per_neuron_per_step * steps_per_ms
        
        # Estimate time for 1ms simulation
        ops_per_second_per_core = 1e9  # 1 GHz
        total_ops_per_second = ops_per_second_per_core * cpu_count
        time_for_1ms = ops_per_ms / total_ops_per_second
        
        results[scale_name] = {
            'neurons': neuron_count,
            'synapses': estimated_synapses,
            'memory_gb': memory_gb,
            'time_per_ms': time_for_1ms,
            'feasible': info['feasible'],
            'description': info['description']
        }
        
        feasible_str = "✅ Feasible" if info['feasible'] else "❌ Requires optimization"
        
        print(f"\n{scale_name}:")
        print(f"  Neurons: {neuron_count:,}")
        print(f"  Synapses: {estimated_synapses:,}")
        print(f"  Memory: {memory_gb:.3f} GB")
        print(f"  Time/ms: {time_for_1ms*1000:.3f} ms")
        print(f"  Status: {feasible_str}")
        print(f"  Description: {info['description']}")
    
    # Create visualization
    create_scaling_visualization(results)
    
    return results

def create_scaling_visualization(results):
    """Create comprehensive scaling visualization"""
    
    print(f"\n📊 Creating scaling visualization...")
    
    scales = list(results.keys())
    neurons = [results[s]['neurons'] for s in scales]
    synapses = [results[s]['synapses'] for s in scales]
    memory = [results[s]['memory_gb'] for s in scales]
    time_per_ms = [results[s]['time_per_ms'] for s in scales]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Neuron count progression
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(scales)), neurons, color='lightblue', alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_xlabel('Scale Level')
    ax1.set_ylabel('Number of Neurons')
    ax1.set_title('Neural Network Scale Progression')
    ax1.set_xticks(range(len(scales)))
    ax1.set_xticklabels(scales, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Annotate bars
    for i, (bar, count) in enumerate(zip(bars1, neurons)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{count:,.0e}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Memory requirements
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(scales)), memory, color='lightcoral', alpha=0.7)
    ax2.set_yscale('log')
    ax2.set_xlabel('Scale Level')
    ax2.set_ylabel('Memory Requirements (GB)')
    ax2.set_title('Memory Scaling')
    ax2.set_xticks(range(len(scales)))
    ax2.set_xticklabels(scales, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add feasibility line
    available_memory = psutil.virtual_memory().total / (1024**3)
    ax2.axhline(y=available_memory, color='red', linestyle='--', 
                label=f'Available RAM ({available_memory:.1f} GB)')
    ax2.legend()
    
    # Plot 3: Computation time scaling
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(scales)), time_per_ms, color='lightgreen', alpha=0.7)
    ax3.set_yscale('log')
    ax3.set_xlabel('Scale Level')
    ax3.set_ylabel('Computation Time per ms (seconds)')
    ax3.set_title('Computational Complexity')
    ax3.set_xticks(range(len(scales)))
    ax3.set_xticklabels(scales, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add real-time line
    ax3.axhline(y=0.001, color='red', linestyle='--', 
                label='Real-time threshold')
    ax3.legend()
    
    # Plot 4: Biological brain regions
    ax4 = axes[1, 1]
    
    # Human brain neuron distribution
    brain_regions = {
        'Cerebellum': 69e9,
        'Cerebral Cortex': 16e9,
        'Hippocampus': 40e6,
        'Thalamus': 6e6,
        'Brainstem': 1e6
    }
    
    colors = ['gold', 'skyblue', 'lightcoral', 'lightgreen', 'plum']
    
    wedges, texts, autotexts = ax4.pie(brain_regions.values(), 
                                      labels=brain_regions.keys(),
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=90)
    
    ax4.set_title('Human Brain Neuron Distribution\n(86 billion total)')
    
    plt.tight_layout()
    plt.savefig('/home/user/neural_scaling_roadmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def create_optimization_roadmap():
    """Create roadmap for reaching billion-neuron simulation"""
    
    print(f"\n🚀 OPTIMIZATION ROADMAP TO BILLIONS")
    print("=" * 50)
    
    optimizations = {
        'Current State': {
            'neurons_possible': 100000,
            'techniques': ['CPU parallelization', 'Memory optimization'],
            'bottleneck': 'Memory and computation'
        },
        'GPU Acceleration': {
            'neurons_possible': 10000000,
            'techniques': ['CUDA kernels', 'Parallel ODE solving', 'Sparse matrices'],
            'bottleneck': 'Memory bandwidth'
        },
        'Distributed Computing': {
            'neurons_possible': 100000000,
            'techniques': ['MPI clusters', 'Network partitioning', 'Load balancing'],
            'bottleneck': 'Communication overhead'
        },
        'Neuromorphic Hardware': {
            'neurons_possible': 1000000000,
            'techniques': ['Intel Loihi', 'SpiNNaker', 'Event-driven processing'],
            'bottleneck': 'Hardware availability'
        },
        'Quantum Computing': {
            'neurons_possible': 86000000000,
            'techniques': ['Quantum neural networks', 'Superposition states'],
            'bottleneck': 'Technology maturity'
        }
    }
    
    print("Progression Path:")
    for i, (stage, info) in enumerate(optimizations.items()):
        print(f"\n{i+1}. {stage}")
        print(f"   Possible neurons: {info['neurons_possible']:,}")
        print(f"   Techniques: {', '.join(info['techniques'])}")
        print(f"   Main bottleneck: {info['bottleneck']}")
    
    print(f"\n💡 Key Insights:")
    print(f"   • Each optimization stage enables ~100x scaling")
    print(f"   • Current consumer hardware: ~100K neurons realistic")
    print(f"   • GPU acceleration: Essential for millions of neurons")
    print(f"   • Distributed systems: Required for 100M+ neurons")
    print(f"   • Specialized hardware: Needed for biological scale")
    
    return optimizations

def demonstrate_actual_small_network():
    """Actually create and simulate a small network to prove the concept"""
    
    print(f"\n🔬 LIVE DEMONSTRATION - Small Network")
    print("=" * 40)
    
    # Create 10 interconnected neurons
    print("Creating 10 interconnected neurons...")
    
    neurons = {}
    for i in range(10):
        neurons[i] = BiologicalNeuron(neuron_id=i)
    
    # Create connections
    connections = 0
    for i in range(10):
        for j in range(10):
            if i != j and np.random.random() < 0.3:  # 30% connection probability
                neurons[i].add_synapse(neurons[j], 
                                     weight=np.random.uniform(0.5, 1.5),
                                     delay=np.random.uniform(0.5, 2.0),
                                     neurotransmitter='excitatory')
                connections += 1
    
    print(f"Created {connections} synaptic connections")
    
    # Simulate network
    print("Running 50ms simulation...")
    
    dt = 0.01
    total_time = 50.0  # ms
    current_time = 0.0
    
    # Apply stimulus to first 3 neurons
    stimulus_neurons = [0, 1, 2]
    
    spike_counts = {i: 0 for i in range(10)}
    
    while current_time < total_time:
        for neuron_id, neuron in neurons.items():
            # Apply stimulus
            external_current = 12.0 if neuron_id in stimulus_neurons else 0.0
            
            # Update neuron
            if neuron.update(dt, current_time, external_current):
                spike_counts[neuron_id] += 1
        
        current_time += dt
    
    # Results
    total_spikes = sum(spike_counts.values())
    print(f"\nResults:")
    print(f"  Total spikes: {total_spikes}")
    print(f"  Spike distribution: {spike_counts}")
    
    # Calculate network firing rate
    avg_rate = total_spikes / (len(neurons) * (total_time / 1000))
    print(f"  Average firing rate: {avg_rate:.2f} Hz")
    
    # Visualize one neuron
    sample_neuron = neurons[0]
    if len(sample_neuron.voltage_history) > 0:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(sample_neuron.time_history, sample_neuron.voltage_history, 'b-')
        plt.ylabel('Voltage (mV)')
        plt.title(f'Sample Neuron Activity (Neuron {0})')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        if sample_neuron.spike_times:
            plt.eventplot([sample_neuron.spike_times], colors=['red'])
        plt.ylabel('Spikes')
        plt.xlabel('Time (ms)')
        plt.title('Spike Times')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/user/live_network_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("✅ Live demonstration complete!")
    
    return neurons, spike_counts

if __name__ == "__main__":
    # Run complete demonstration
    
    # 1. Scaling analysis
    scaling_results = demonstrate_neural_scaling()
    
    # 2. Optimization roadmap
    roadmap = create_optimization_roadmap()
    
    # 3. Live demonstration
    demo_neurons, demo_spikes = demonstrate_actual_small_network()