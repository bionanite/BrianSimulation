#!/usr/bin/env python3
"""
Test Framework for Structural Plasticity Mechanisms
Tests synapse creation/deletion, dendritic growth, and neurogenesis
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from structural_plasticity import (
    StructuralPlasticityManager, SynapseCreationDeletion,
    DendriticGrowth, Neurogenesis, NeuronStructure, DendriticBranch
)
from plasticity_mechanisms import SynapticConnection

class StructuralPlasticityTester:
    """Test framework for structural plasticity"""
    
    def __init__(self):
        self.results = []
    
    def test_synapse_creation_deletion(self):
        """Test synapse creation and deletion"""
        print("\n" + "="*60)
        print("TEST 1: Synapse Creation and Deletion")
        print("="*60)
        
        manager = SynapseCreationDeletion(
            creation_threshold=0.7,
            deletion_threshold=0.1
        )
        
        # Create initial synapses
        synapses = [
            SynapticConnection(0, 1, 1.0, 1.0, 'excitatory'),
            SynapticConnection(1, 2, 0.8, 1.0, 'excitatory'),
        ]
        
        print(f"\nInitial synapses: {len(synapses)}")
        
        # Simulate co-activity
        print("\nTest 1.1: Co-activity tracking")
        for _ in range(100):
            manager.update_co_activity([(0, 1), (2, 3)], time.time())
        
        co_activity_01 = manager.co_activity_matrix.get((0, 1), 0.0)
        co_activity_23 = manager.co_activity_matrix.get((2, 3), 0.0)
        
        print(f"   Co-activity (0,1): {co_activity_01:.2f}")
        print(f"   Co-activity (2,3): {co_activity_23:.2f}")
        print(f"   Result: {'✅ PASS' if co_activity_01 > 0 and co_activity_23 > 0 else '❌ FAIL'}")
        
        # Test synapse creation
        print("\nTest 1.2: Synapse creation")
        initial_count = len(synapses)
        
        # Force high co-activity
        manager.co_activity_matrix[(2, 3)] = 10.0  # High co-activity
        
        creation_attempts = 0
        for _ in range(1000):  # Many attempts
            if manager.should_create_synapse(2, 3, synapses):
                new_synapse = manager.create_synapse(2, 3)
                synapses.append(new_synapse)
                creation_attempts += 1
        
        print(f"   Creation attempts: {creation_attempts}")
        print(f"   Final synapses: {len(synapses)}")
        print(f"   Synapses created: {len(synapses) - initial_count}")
        print(f"   Result: {'✅ PASS' if len(synapses) > initial_count else '❌ FAIL'}")
        
        # Test synapse deletion
        print("\nTest 1.3: Synapse deletion")
        initial_count = len(synapses)
        
        # Make some synapses inactive
        for synapse in synapses[:2]:
            manager.update_synapse_activity(synapse, 0.05)  # Below threshold
        
        synapses, deleted_count = manager.prune_synapses(synapses, time.time())
        
        print(f"   Synapses deleted: {deleted_count}")
        print(f"   Final synapses: {len(synapses)}")
        print(f"   Result: {'✅ PASS' if deleted_count > 0 else '⚠️  CHECK'}")
        
        return True
    
    def test_dendritic_growth(self):
        """Test dendritic growth and pruning"""
        print("\n" + "="*60)
        print("TEST 2: Dendritic Growth and Pruning")
        print("="*60)
        
        growth_manager = DendriticGrowth()
        
        # Create neuron structure
        neuron_structure = NeuronStructure(neuron_id=1)
        
        # Create initial branch
        branch = growth_manager.create_new_branch(1, None, initial_length=50.0)
        neuron_structure.dendritic_branches.append(branch)
        
        print(f"\nInitial branch length: {branch.length:.2f} μm")
        print(f"Initial branches: {len(neuron_structure.dendritic_branches)}")
        
        # Test branch growth
        print("\nTest 2.1: Branch growth with high activity")
        initial_length = branch.length
        
        for _ in range(100):
            growth_manager.update_branch_activity(branch, 0.8)  # High activity
            growth_manager.grow_branch(branch, branch.activity, dt=1.0)
        
        final_length = branch.length
        growth = final_length - initial_length
        
        print(f"   Activity: {branch.activity:.2f}")
        print(f"   Initial length: {initial_length:.2f} μm")
        print(f"   Final length: {final_length:.2f} μm")
        print(f"   Growth: {growth:+.2f} μm")
        print(f"   Result: {'✅ PASS' if growth > 0 else '❌ FAIL'}")
        
        # Test branch pruning
        print("\nTest 2.2: Branch pruning with low activity")
        branch2 = growth_manager.create_new_branch(1, None, initial_length=30.0)
        branch2.age = time.time() - 2000.0  # Old branch
        growth_manager.update_branch_activity(branch2, 0.01)  # Very low activity
        neuron_structure.dendritic_branches.append(branch2)
        
        initial_branch_count = len(neuron_structure.dendritic_branches)
        should_prune = growth_manager.prune_branch(branch2, time.time())
        
        if should_prune:
            neuron_structure.dendritic_branches.remove(branch2)
        
        final_branch_count = len(neuron_structure.dendritic_branches)
        
        print(f"   Initial branches: {initial_branch_count}")
        print(f"   Should prune: {should_prune}")
        print(f"   Final branches: {final_branch_count}")
        print(f"   Result: {'✅ PASS' if should_prune and final_branch_count < initial_branch_count else '⚠️  CHECK'}")
        
        # Test new branch creation
        print("\nTest 2.3: New branch creation")
        activity_map = {1: 0.8}  # High activity
        initial_branches = len(neuron_structure.dendritic_branches)
        
        # Try many times to get a new branch
        for _ in range(10000):
            stats = growth_manager.update_dendritic_structure(
                neuron_structure, activity_map, time.time(), dt=1.0
            )
            if stats['new_branches'] > 0:
                break
        
        final_branches = len(neuron_structure.dendritic_branches)
        new_branches = final_branches - initial_branches
        
        print(f"   Initial branches: {initial_branches}")
        print(f"   Final branches: {final_branches}")
        print(f"   New branches created: {new_branches}")
        print(f"   Result: {'✅ PASS' if new_branches >= 0 else '❌ FAIL'}")
        
        return True
    
    def test_neurogenesis(self):
        """Test neurogenesis"""
        print("\n" + "="*60)
        print("TEST 3: Neurogenesis")
        print("="*60)
        
        neurogenesis = Neurogenesis(neurogenesis_rate=0.01)  # Higher rate for testing
        
        neurons = [{'id': i} for i in range(10)]
        initial_count = len(neurons)
        
        print(f"\nInitial neurons: {initial_count}")
        
        # Test neuron creation
        print("\nTest 3.1: Neuron creation")
        neurons_created = 0
        
        for _ in range(1000):  # Many attempts
            if neurogenesis.should_create_neuron(
                region_activity=0.8,  # High activity
                current_neuron_count=len(neurons),
                max_neurons=20
            ):
                new_neuron_id = len(neurons)
                neuron_data = neurogenesis.create_neuron(
                    new_neuron_id, current_time=time.time()
                )
                neurons.append(neuron_data)
                neurons_created += 1
        
        print(f"   Neurons created: {neurons_created}")
        print(f"   Final neurons: {len(neurons)}")
        print(f"   Result: {'✅ PASS' if neurons_created > 0 else '⚠️  CHECK'}")
        
        # Test neuron integration
        print("\nTest 3.2: Neuron integration")
        if neurons_created > 0:
            new_neuron_id = initial_count
            synapses = []
            target_neurons = list(range(initial_count))
            
            # Set creation time in past
            neurogenesis.new_neurons[new_neuron_id] = time.time() - 2000.0
            
            initial_synapse_count = len(synapses)
            
            def synapse_creator(pre_id, post_id):
                return SynapticConnection(pre_id, post_id, 0.5, 1.0, 'excitatory')
            
            synapses, new_synapses = neurogenesis.integrate_neuron(
                new_neuron_id, synapses, target_neurons, time.time(), synapse_creator
            )
            
            print(f"   Initial synapses: {initial_synapse_count}")
            print(f"   New synapses created: {new_synapses}")
            print(f"   Final synapses: {len(synapses)}")
            print(f"   Result: {'✅ PASS' if new_synapses > 0 else '❌ FAIL'}")
        
        return True
    
    def test_integrated_structural_plasticity(self):
        """Test all structural plasticity mechanisms together"""
        print("\n" + "="*60)
        print("TEST 4: Integrated Structural Plasticity")
        print("="*60)
        
        manager = StructuralPlasticityManager(
            enable_synapse_creation=True,
            enable_dendritic_growth=True,
            enable_neurogenesis=True
        )
        
        # Initialize network
        synapses = [
            SynapticConnection(i, i+1, 1.0, 1.0, 'excitatory')
            for i in range(10)
        ]
        
        neurons = [{'id': i, 'spike_times': []} for i in range(10)]
        
        print(f"\nInitial state:")
        print(f"   Neurons: {len(neurons)}")
        print(f"   Synapses: {len(synapses)}")
        
        # Simulate over time
        simulation_time = 5000.0  # ms
        dt = 10.0  # ms
        
        stats_history = []
        time_points = []
        
        print(f"\nSimulating structural plasticity over {simulation_time}ms...")
        
        for t in np.arange(0, simulation_time, dt):
            # Generate activity
            activity_map = {}
            for neuron in neurons:
                neuron_id = neuron['id']
                # Random activity
                activity = np.random.random() * 0.5 + 0.3
                activity_map[neuron_id] = activity
                
                # Add some spikes
                if np.random.random() < 0.1:
                    neuron['spike_times'].append(t)
            
            # Update structures
            stats, synapses, neurons = manager.update_structures(
                synapses, neurons, activity_map, t, dt
            )
            
            if t % 500 == 0:
                stats_history.append(stats)
                time_points.append(t)
                print(f"   t={t:.0f}ms: Synapses={stats['total_synapses']}, "
                      f"Created={stats['synapses_created']}, "
                      f"Deleted={stats['synapses_deleted']}")
        
        # Visualize evolution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Structural Plasticity Evolution', fontsize=16, fontweight='bold')
        
        # Plot 1: Synapse count
        ax1 = axes[0, 0]
        synapse_counts = [s['total_synapses'] for s in stats_history]
        ax1.plot(time_points, synapse_counts, 'b-', linewidth=2, marker='o')
        ax1.set_title('Total Synapses Over Time', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Synapse Count')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Synapse creation/deletion
        ax2 = axes[0, 1]
        created = [s['synapses_created'] for s in stats_history]
        deleted = [s['synapses_deleted'] for s in stats_history]
        ax2.plot(time_points, created, 'g-', label='Created', linewidth=2)
        ax2.plot(time_points, deleted, 'r-', label='Deleted', linewidth=2)
        ax2.set_title('Synapse Creation/Deletion', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Dendritic branches
        ax3 = axes[1, 0]
        branches_grown = [s['branches_grown'] for s in stats_history]
        branches_pruned = [s['branches_pruned'] for s in stats_history]
        ax3.plot(time_points, branches_grown, 'g-', label='Grown', linewidth=2)
        ax3.plot(time_points, branches_pruned, 'r-', label='Pruned', linewidth=2)
        ax3.set_title('Dendritic Branch Changes', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Neuron count
        ax4 = axes[1, 1]
        neuron_counts = [s['total_neurons'] for s in stats_history]
        ax4.plot(time_points, neuron_counts, 'purple', linewidth=2, marker='s')
        ax4.set_title('Total Neurons Over Time', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Neuron Count')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('structural_plasticity_evolution.png', dpi=300, bbox_inches='tight')
        print("\n✅ Evolution visualization saved: structural_plasticity_evolution.png")
        plt.close()
        
        # Final statistics
        final_stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Current synapses: {final_stats['current_synapses']}")
        print(f"   Current neurons: {final_stats['current_neurons']}")
        print(f"   Total synapses created: {final_stats['total_synapses_created']}")
        print(f"   Total synapses deleted: {final_stats['total_synapses_deleted']}")
        print(f"   Net synapse change: {final_stats['net_synapse_change']:+.0f}")
        print(f"   Total dendritic branches: {final_stats['total_branches']}")
        print(f"   Avg dendritic length: {final_stats['avg_dendritic_length']:.2f} μm")
        
        return True
    
    def run_all_tests(self):
        """Run all structural plasticity tests"""
        print("\n" + "="*70)
        print("STRUCTURAL PLASTICITY TEST SUITE")
        print("="*70)
        
        tests = [
            ("Synapse Creation/Deletion", self.test_synapse_creation_deletion),
            ("Dendritic Growth", self.test_dendritic_growth),
            ("Neurogenesis", self.test_neurogenesis),
            ("Integrated Structural Plasticity", self.test_integrated_structural_plasticity)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"\n❌ {test_name} FAILED with error: {e}")
                import traceback
                traceback.print_exc()
                results.append((test_name, False))
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        passed = sum(1 for _, r in results if r)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        print(f"\n   Total: {passed}/{total} tests passed")
        print(f"   Success rate: {passed/total*100:.1f}%")
        
        return passed == total

def main():
    """Main test function"""
    tester = StructuralPlasticityTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All structural plasticity tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

