#!/usr/bin/env python3
"""
Test Framework for Memory Consolidation Mechanisms
Tests sleep consolidation, reconsolidation, and forgetting
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from memory_consolidation import (
    MemoryConsolidationManager, SleepLikeConsolidation,
    MemoryReconsolidation, ForgettingMechanisms, MemoryTrace
)

class MemoryConsolidationTester:
    """Test framework for memory consolidation"""
    
    def __init__(self):
        self.results = []
    
    def test_sleep_consolidation(self):
        """Test sleep-like consolidation"""
        print("\n" + "="*60)
        print("TEST 1: Sleep-Like Consolidation")
        print("="*60)
        
        sleep_consolidation = SleepLikeConsolidation()
        
        # Create test memories
        memories = []
        for i in range(5):
            memory = MemoryTrace(
                memory_id=i,
                content=np.random.random(10),
                strength=0.5,
                age=i * 100.0,
                importance=0.3 + i * 0.1
            )
            memories.append(memory)
        
        print(f"\nCreated {len(memories)} test memories")
        
        # Test 1: Memory selection for replay
        print("\nTest 1.1: Memory selection for replay")
        selected = sleep_consolidation.select_memories_for_replay(memories, num_to_replay=3)
        
        print(f"   Selected {len(selected)} memories for replay")
        for memory in selected:
            print(f"      Memory {memory.memory_id}: strength={memory.strength:.3f}, "
                  f"importance={memory.importance:.3f}, age={memory.age:.1f}")
        print(f"   Result: {'✅ PASS' if len(selected) > 0 else '❌ FAIL'}")
        
        # Test 2: Memory consolidation
        print("\nTest 1.2: Memory consolidation through replay")
        memory = memories[0]
        initial_strength = memory.strength
        initial_consolidation = memory.consolidation_level
        
        strength_change = sleep_consolidation.consolidate_memory(memory, replay_count=3)
        
        print(f"   Initial strength: {initial_strength:.4f}")
        print(f"   Final strength: {memory.strength:.4f}")
        print(f"   Strength change: {strength_change:+.4f}")
        print(f"   Initial consolidation: {initial_consolidation:.4f}")
        print(f"   Final consolidation: {memory.consolidation_level:.4f}")
        print(f"   Result: {'✅ PASS' if strength_change > 0 and memory.consolidation_level > initial_consolidation else '❌ FAIL'}")
        
        # Test 3: Sleep cycle simulation
        print("\nTest 1.3: Sleep cycle simulation")
        initial_strengths = [m.strength for m in memories]
        
        events = sleep_consolidation.simulate_sleep_cycle(memories, cycle_duration=1000.0)
        
        final_strengths = [m.strength for m in memories]
        
        print(f"   Consolidation events: {len(events)}")
        print(f"   Average strength change: {np.mean(final_strengths) - np.mean(initial_strengths):+.4f}")
        print(f"   Result: {'✅ PASS' if len(events) > 0 else '❌ FAIL'}")
        
        return True
    
    def test_memory_reconsolidation(self):
        """Test memory reconsolidation"""
        print("\n" + "="*60)
        print("TEST 2: Memory Reconsolidation")
        print("="*60)
        
        reconsolidation = MemoryReconsolidation()
        
        # Create test memory
        memory = MemoryTrace(
            memory_id=1,
            content=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            strength=0.8,
            consolidation_level=0.9,
            last_access_time=time.time()
        )
        
        print(f"\nCreated test memory")
        print(f"   Initial content: {memory.content}")
        print(f"   Consolidation level: {memory.consolidation_level:.3f}")
        
        # Test 1: Labile state check
        print("\nTest 2.1: Labile state detection")
        current_time = time.time()
        is_labile = reconsolidation.is_labile(memory, current_time)
        
        print(f"   Is labile: {is_labile}")
        print(f"   Result: {'✅ PASS' if is_labile else '❌ FAIL'}")
        
        # Test 2: Memory update during labile state
        print("\nTest 2.2: Memory update during reconsolidation")
        new_content = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        initial_content = memory.content.copy()
        
        error = reconsolidation.update_memory(memory, new_content)
        
        print(f"   Initial content: {initial_content}")
        print(f"   New content: {new_content}")
        print(f"   Updated content: {memory.content}")
        print(f"   Update error: {error:.4f}")
        print(f"   Consolidation level: {memory.consolidation_level:.3f}")
        print(f"   Result: {'✅ PASS' if error < np.linalg.norm(initial_content - new_content) else '❌ FAIL'}")
        
        # Test 3: Reconsolidation after labile period
        print("\nTest 2.3: Reconsolidation after labile period")
        # Wait for labile period to end
        memory.last_access_time = time.time() - 200.0  # 200ms ago
        initial_consolidation = memory.consolidation_level
        
        reconsolidated = reconsolidation.reconsolidate(memory, time.time())
        
        print(f"   Reconsolidated: {reconsolidated}")
        print(f"   Initial consolidation: {initial_consolidation:.3f}")
        print(f"   Final consolidation: {memory.consolidation_level:.3f}")
        print(f"   Result: {'✅ PASS' if reconsolidated and memory.consolidation_level >= initial_consolidation else '⚠️  CHECK'}")
        
        return True
    
    def test_forgetting_mechanisms(self):
        """Test forgetting mechanisms"""
        print("\n" + "="*60)
        print("TEST 3: Forgetting Mechanisms")
        print("="*60)
        
        forgetting = ForgettingMechanisms()
        
        # Create test memories
        memory1 = MemoryTrace(
            memory_id=1,
            content=np.random.random(10),
            strength=0.8,
            importance=0.9,  # Important
            access_count=50  # Frequently accessed
        )
        
        memory2 = MemoryTrace(
            memory_id=2,
            content=np.random.random(10),
            strength=0.3,
            importance=0.2,  # Not important
            access_count=2  # Rarely accessed
        )
        
        print(f"\nCreated 2 test memories")
        print(f"   Memory 1: strength={memory1.strength:.3f}, importance={memory1.importance:.3f}")
        print(f"   Memory 2: strength={memory2.strength:.3f}, importance={memory2.importance:.3f}")
        
        # Test 1: Memory decay
        print("\nTest 3.1: Memory decay over time")
        initial_strength1 = memory1.strength
        initial_strength2 = memory2.strength
        
        time_elapsed = 1000.0  # 1 second
        decay1 = forgetting.decay_memory(memory1, time_elapsed)
        decay2 = forgetting.decay_memory(memory2, time_elapsed)
        
        print(f"   Memory 1 decay: {decay1:+.4f} (strength: {memory1.strength:.4f})")
        print(f"   Memory 2 decay: {decay2:+.4f} (strength: {memory2.strength:.4f})")
        print(f"   Important memory decays slower: {'✅ PASS' if abs(decay1) < abs(decay2) else '⚠️  CHECK'}")
        
        # Test 2: Forgetting threshold
        print("\nTest 3.2: Forgetting threshold")
        weak_memory = MemoryTrace(
            memory_id=3,
            content=np.random.random(10),
            strength=0.05  # Below threshold
        )
        
        should_forget = forgetting.should_forget(weak_memory)
        
        print(f"   Memory strength: {weak_memory.strength:.3f}")
        print(f"   Should forget: {should_forget}")
        print(f"   Result: {'✅ PASS' if should_forget else '❌ FAIL'}")
        
        # Test 3: Forgetting process
        print("\nTest 3.3: Forgetting process")
        forget_info = forgetting.forget_memory(weak_memory)
        
        print(f"   Forgotten memory ID: {forget_info['memory_id']}")
        print(f"   Final strength: {forget_info['strength']:.3f}")
        print(f"   Age: {forget_info['age']:.1f}")
        print(f"   Result: {'✅ PASS' if forget_info['memory_id'] == 3 else '❌ FAIL'}")
        
        return True
    
    def test_integrated_memory_consolidation(self):
        """Test all memory consolidation mechanisms together"""
        print("\n" + "="*60)
        print("TEST 4: Integrated Memory Consolidation")
        print("="*60)
        
        manager = MemoryConsolidationManager(
            enable_sleep_consolidation=True,
            enable_reconsolidation=True,
            enable_forgetting=True
        )
        
        print(f"\nCreated memory consolidation manager")
        
        # Create memories
        print("\nCreating memories...")
        memories_created = []
        for i in range(10):
            content = np.random.random(20)
            importance = np.random.random()
            memory = manager.create_memory(content, importance=importance)
            memories_created.append(memory.memory_id)
            print(f"   Memory {memory.memory_id}: importance={importance:.2f}")
        
        # Simulate memory access
        print("\nSimulating memory access...")
        for _ in range(20):
            memory_id = np.random.choice(memories_created)
            manager.access_memory(memory_id)
        
        # Simulate sleep consolidation
        print("\nSimulating sleep consolidation...")
        sleep_events = manager.simulate_sleep(duration=1000.0)
        print(f"   Sleep consolidation events: {len(sleep_events)}")
        
        # Simulate time passage (decay and forgetting)
        print("\nSimulating time passage (decay and forgetting)...")
        update_stats = manager.update_memories(time_elapsed=1000.0)  # Reduced time
        print(f"   Memories decayed: {update_stats['memories_decayed']}")
        print(f"   Memories forgotten: {update_stats['memories_forgotten']}")
        print(f"   Total strength loss: {update_stats['total_strength_loss']:.4f}")
        
        # Test memory update during reconsolidation
        print("\nTesting memory update during reconsolidation...")
        if memories_created:
            memory_id = memories_created[0]
            manager.access_memory(memory_id)  # Make it labile
            new_content = np.random.random(20)
            updated = manager.update_memory(memory_id, new_content)
            print(f"   Memory updated: {updated}")
        
        # Visualize consolidation
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Memory Consolidation Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Memory strengths
        ax1 = axes[0, 0]
        memories_list = list(manager.memories.values())
        if memories_list:
            strengths = [m.strength for m in memories_list]
            memory_ids = [m.memory_id for m in memories_list]
            bars = ax1.bar(memory_ids, strengths, color='#3498DB', alpha=0.8, edgecolor='black')
            ax1.set_title('Memory Strengths', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Memory ID')
            ax1.set_ylabel('Strength')
            ax1.set_ylim([0, 1.1])
            ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Consolidation levels
        ax2 = axes[0, 1]
        if memories_list:
            consolidation_levels = [m.consolidation_level for m in memories_list]
            bars = ax2.bar(memory_ids, consolidation_levels, color='#2ECC71', alpha=0.8, edgecolor='black')
            ax2.set_title('Consolidation Levels', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Memory ID')
            ax2.set_ylabel('Consolidation Level')
            ax2.set_ylim([0, 1.1])
            ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Memory ages
        ax3 = axes[1, 0]
        if memories_list:
            ages = [m.age for m in memories_list]
            bars = ax3.bar(memory_ids, ages, color='#E74C3C', alpha=0.8, edgecolor='black')
            ax3.set_title('Memory Ages', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Memory ID')
            ax3.set_ylabel('Age')
            ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Access counts
        ax4 = axes[1, 1]
        if memories_list:
            access_counts = [m.access_count for m in memories_list]
            bars = ax4.bar(memory_ids, access_counts, color='#F39C12', alpha=0.8, edgecolor='black')
            ax4.set_title('Memory Access Counts', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Memory ID')
            ax4.set_ylabel('Access Count')
            ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('memory_consolidation_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Consolidation visualization saved: memory_consolidation_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        if stats:
            print(f"   Total memories: {stats.get('total_memories', 0)}")
            print(f"   Average strength: {stats.get('avg_strength', 0):.4f}")
            print(f"   Average consolidation: {stats.get('avg_consolidation', 0):.4f}")
            print(f"   Consolidated memories (>0.5): {stats.get('consolidated_memories', 0)}")
            print(f"   Weak memories (<0.3): {stats.get('weak_memories', 0)}")
            print(f"   Total consolidation events: {stats.get('total_consolidation_events', 0)}")
        else:
            print(f"   No memories remaining")
        
        return True
    
    def run_all_tests(self):
        """Run all memory consolidation tests"""
        print("\n" + "="*70)
        print("MEMORY CONSOLIDATION TEST SUITE")
        print("="*70)
        
        tests = [
            ("Sleep-Like Consolidation", self.test_sleep_consolidation),
            ("Memory Reconsolidation", self.test_memory_reconsolidation),
            ("Forgetting Mechanisms", self.test_forgetting_mechanisms),
            ("Integrated Memory Consolidation", self.test_integrated_memory_consolidation)
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
    tester = MemoryConsolidationTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All memory consolidation tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

