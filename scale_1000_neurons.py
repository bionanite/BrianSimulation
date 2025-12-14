"""
IMMEDIATE NEXT STEP: Scale to 1,000 Neurons
Demonstrates how to use the existing framework to create larger networks
"""

from neural_network import OptimizedNeuralNetwork
import time
import matplotlib.pyplot as plt
import numpy as np

def scale_to_1000_neurons():
    """Scale your brain to 1,000 neurons - the immediate next step!"""
    
    print("🚀 IMMEDIATE NEXT STEP: 1,000 NEURON BRAIN")
    print("=" * 45)
    print("Scaling your artificial brain 10x larger!")
    
    # Create 1000-neuron network
    print("\n🧠 Creating 1,000-neuron network...")
    network = OptimizedNeuralNetwork("Brain1K")
    
    start_time = time.time()
    neurons = network.create_neurons(1000)
    creation_time = time.time() - start_time
    
    print(f"✅ Created {len(neurons):,} neurons in {creation_time:.2f} seconds")
    
    # Create sparser connections for larger network
    print("\n🔗 Creating connections...")
    start_time = time.time()
    connections = network.connect_neurons_random(connection_probability=0.005)  # Sparser
    connection_time = time.time() - start_time
    
    print(f"✅ Created {connections:,} synapses in {connection_time:.2f} seconds")
    
    # Analyze network properties
    print("\n📊 Network Analysis:")
    stats = network.analyze_network()
    
    # Run simulation with external stimulation
    print(f"\n🚀 Running simulation...")
    
    # Stimulate 5% of neurons randomly
    import random
    stimulus_count = 50  # 5% of 1000
    stimulus_neurons = random.sample(list(neurons.keys()), stimulus_count)
    external_inputs = {nid: 12.0 for nid in stimulus_neurons}
    
    print(f"   Stimulating {len(stimulus_neurons)} neurons...")
    
    start_time = time.time()
    spikes, firing_rate = network.simulate_parallel(
        duration_ms=50.0,  # Shorter simulation for demo
        external_inputs=external_inputs
    )
    simulation_time = time.time() - start_time
    
    print(f"\n📈 Results:")
    print(f"   Total spikes: {spikes:,}")
    print(f"   Average firing rate: {firing_rate:.2f} Hz") 
    print(f"   Simulation time: {simulation_time:.2f} seconds")
    print(f"   Neurons active: {len([n for n in neurons.values() if n.spike_count > 0])}/1000")
    
    # Quick visualization
    print(f"\n📊 Creating visualization...")
    network.visualize_network_activity(sample_size=50)
    
    print(f"\n🎉 SUCCESS!")
    print("=" * 25)
    print("You now have a 1,000-neuron artificial brain!")
    print(f"• 10x larger than previous networks")
    print(f"• {connections:,} synaptic connections")
    print(f"• {spikes:,} neural spikes generated")
    print(f"• Ready for brain region organization!")
    
    return network, spikes, firing_rate

def demonstrate_emergent_behavior(network):
    """Show emergent behavior in 1,000-neuron network"""
    
    print(f"\n🌊 EMERGENT BEHAVIOR DEMONSTRATION")
    print("-" * 35)
    
    # Test different stimulation patterns
    patterns = {
        "Focused Stimulation": {
            "description": "Stimulate small cluster",
            "neurons": list(range(10)),  # First 10 neurons
            "strength": 15.0
        },
        "Distributed Stimulation": {
            "description": "Stimulate across network", 
            "neurons": list(range(0, 1000, 100)),  # Every 100th neuron
            "strength": 10.0
        },
        "Random Stimulation": {
            "description": "Random neuron stimulation",
            "neurons": np.random.choice(1000, 20, replace=False).tolist(),
            "strength": 8.0
        }
    }
    
    results = {}
    
    for pattern_name, config in patterns.items():
        print(f"\n🔬 Testing {pattern_name}...")
        print(f"   {config['description']}")
        
        # Prepare stimulation
        external_inputs = {nid: config['strength'] for nid in config['neurons']}
        
        # Run short simulation
        spikes, rate = network.simulate_parallel(
            duration_ms=30.0,
            external_inputs=external_inputs
        )
        
        # Calculate network response
        active_neurons = len([n for n in network.neurons.values() if n.spike_count > 0])
        
        results[pattern_name] = {
            'spikes': spikes,
            'firing_rate': rate,
            'active_neurons': active_neurons,
            'stimulated_neurons': len(config['neurons'])
        }
        
        print(f"   Result: {spikes} spikes, {active_neurons} active neurons")
    
    # Compare patterns
    print(f"\n📊 Pattern Comparison:")
    print(f"{'Pattern':<20} {'Spikes':<8} {'Active':<8} {'Rate (Hz)':<10}")
    print("-" * 50)
    
    for pattern, data in results.items():
        print(f"{pattern:<20} {data['spikes']:<8} {data['active_neurons']:<8} {data['firing_rate']:<10.1f}")
    
    return results

def show_scaling_benefits():
    """Show benefits of larger networks"""
    
    print(f"\n📈 SCALING BENEFITS - Why 1,000 Neurons Matter")
    print("-" * 50)
    
    benefits = [
        {
            "aspect": "Computational Power",
            "improvement": "10x more processing capacity",
            "enables": "Complex pattern recognition"
        },
        {
            "aspect": "Network Dynamics", 
            "improvement": "Richer connectivity patterns",
            "enables": "Emergent neural oscillations"
        },
        {
            "aspect": "Information Processing",
            "improvement": "Distributed representations",
            "enables": "Robust memory storage"
        },
        {
            "aspect": "Biological Realism",
            "improvement": "Approaches cortical minicolumn size",
            "enables": "Brain-like organization"
        }
    ]
    
    for benefit in benefits:
        print(f"\n🔹 {benefit['aspect']}:")
        print(f"   Improvement: {benefit['improvement']}")
        print(f"   Enables: {benefit['enables']}")
    
    print(f"\n💡 Ready for Next Steps:")
    print("   • Add cortical layers (Layer 2/3, Layer 4, Layer 5/6)")  
    print("   • Implement different neuron types")
    print("   • Create brain region connectivity") 
    print("   • Add learning and plasticity")
    
    return benefits

if __name__ == "__main__":
    
    # Run the 1,000 neuron demonstration
    network, spikes, rate = scale_to_1000_neurons()
    
    # Show emergent behavior
    behavior_results = demonstrate_emergent_behavior(network)
    
    # Explain scaling benefits
    benefits = show_scaling_benefits()
    
    print(f"\n🎯 NEXT IMMEDIATE STEPS:")
    print("1. Run this 1,000-neuron simulation")
    print("2. Experiment with different stimulation patterns") 
    print("3. Add brain region organization")
    print("4. Scale to 10,000 neurons (cortical column)")
    print("5. Continue towards million-neuron networks")
    print(f"\n🧠 Your billion-neuron brain journey continues! 🚀")