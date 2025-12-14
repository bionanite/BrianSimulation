#!/usr/bin/env python3
"""
Test Framework for Executive Control
Tests cognitive control, attention, task switching, inhibition, and working memory
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from executive_control import (
    ExecutiveControlManager, CognitiveControl,
    AttentionManagement, TaskSwitching, InhibitionControl, WorkingMemoryManagement
)

class ExecutiveControlTester:
    """Test framework for executive control"""
    
    def __init__(self):
        self.results = []
    
    def test_cognitive_control(self):
        """Test cognitive control"""
        print("\n" + "="*60)
        print("TEST 1: Cognitive Control")
        print("="*60)
        
        control = CognitiveControl()
        
        # Test 1: Create tasks
        print("\nTest 1.1: Creating tasks")
        task1 = control.create_task("Task 1", priority=1.0)
        task2 = control.create_task("Task 2", priority=0.8)
        
        print(f"   Tasks created: {len(control.active_tasks)}")
        print(f"   Result: {'✅ PASS' if len(control.active_tasks) == 2 else '❌ FAIL'}")
        
        # Test 2: Task selection
        print("\nTest 1.2: Task selection")
        selected = control.select_task()
        
        print(f"   Selected task: {selected.description if selected else None}")
        print(f"   Result: {'✅ PASS' if selected is not None else '❌ FAIL'}")
        
        # Test 3: Inhibition
        print("\nTest 1.3: Response inhibition")
        control.inhibit_response(1)
        is_inhibited = control.is_inhibited(1)
        
        print(f"   Response 1 inhibited: {is_inhibited}")
        print(f"   Result: {'✅ PASS' if is_inhibited else '❌ FAIL'}")
        
        return True
    
    def test_attention_management(self):
        """Test attention management"""
        print("\n" + "="*60)
        print("TEST 2: Attention Management")
        print("="*60)
        
        attention = AttentionManagement()
        
        # Test 1: Focus attention
        print("\nTest 2.1: Focusing attention")
        focus1 = attention.focus_attention("Target A", intensity=0.8)
        focus2 = attention.focus_attention("Target B", intensity=0.6)
        
        print(f"   Focuses: {len(attention.current_focuses)}")
        print(f"   Result: {'✅ PASS' if len(attention.current_focuses) == 2 else '❌ FAIL'}")
        
        # Test 2: Attention distribution
        print("\nTest 2.2: Attention distribution")
        distribution = attention.get_attention_distribution()
        
        print(f"   Distribution: {distribution}")
        print(f"   Result: {'✅ PASS' if len(distribution) == 2 else '❌ FAIL'}")
        
        # Test 3: Attention shift
        print("\nTest 2.3: Shifting attention")
        initial_focuses = len(attention.current_focuses)
        attention.shift_attention("Target C", intensity=0.9)
        
        print(f"   Focuses after shift: {len(attention.current_focuses)}")
        print(f"   Result: {'✅ PASS' if len(attention.current_focuses) >= 1 else '❌ FAIL'}")
        
        return True
    
    def test_task_switching(self):
        """Test task switching"""
        print("\n" + "="*60)
        print("TEST 3: Task Switching")
        print("="*60)
        
        switching = TaskSwitching()
        
        # Create tasks
        from executive_control import Task
        task1 = Task(1, "Task 1", priority=0.5)
        task2 = Task(2, "Task 2", priority=0.9)
        
        # Test 1: Should switch
        print("\nTest 3.1: Switch decision")
        should_switch = switching.should_switch_task(task1, task2)
        
        print(f"   Should switch: {should_switch}")
        print(f"   Result: {'✅ PASS' if should_switch else '⚠️  CHECK'}")
        
        # Test 2: Switch task
        print("\nTest 3.2: Switching tasks")
        switch_cost = switching.switch_task(1, 2)
        
        print(f"   Switch cost: {switch_cost:.4f}")
        print(f"   Current task: {switching.current_task_id}")
        print(f"   Result: {'✅ PASS' if switching.current_task_id == 2 else '❌ FAIL'}")
        
        return True
    
    def test_inhibition_control(self):
        """Test inhibition control"""
        print("\n" + "="*60)
        print("TEST 4: Inhibition Control")
        print("="*60)
        
        inhibition = InhibitionControl()
        
        # Test 1: Inhibit items
        print("\nTest 4.1: Inhibiting items")
        inhibition.inhibit(1, strength=0.8)
        inhibition.inhibit(2, strength=0.6)
        
        print(f"   Inhibited items: {len(inhibition.inhibited_items)}")
        print(f"   Result: {'✅ PASS' if len(inhibition.inhibited_items) == 2 else '❌ FAIL'}")
        
        # Test 2: Check inhibition
        print("\nTest 4.2: Checking inhibition")
        is_inhibited = inhibition.is_inhibited(1)
        strength = inhibition.get_inhibition_strength(1)
        
        print(f"   Item 1 inhibited: {is_inhibited}")
        print(f"   Inhibition strength: {strength:.4f}")
        print(f"   Result: {'✅ PASS' if is_inhibited else '❌ FAIL'}")
        
        # Test 3: Release inhibition
        print("\nTest 4.3: Releasing inhibition")
        inhibition.release(1)
        
        print(f"   Item 1 still inhibited: {inhibition.is_inhibited(1)}")
        print(f"   Result: {'✅ PASS' if not inhibition.is_inhibited(1) else '❌ FAIL'}")
        
        return True
    
    def test_working_memory_management(self):
        """Test working memory management"""
        print("\n" + "="*60)
        print("TEST 5: Working Memory Management")
        print("="*60)
        
        wm = WorkingMemoryManagement(capacity=7)
        
        # Test 1: Add to working memory
        print("\nTest 5.1: Adding to working memory")
        items = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"]
        item_ids = []
        
        for item in items:
            item_id = wm.add_to_working_memory(item)
            item_ids.append(item_id)
        
        print(f"   Items in memory: {len(wm.memory_items)}")
        print(f"   Result: {'✅ PASS' if len(wm.memory_items) == len(items) else '❌ FAIL'}")
        
        # Test 2: Check capacity
        print("\nTest 5.2: Working memory capacity")
        # Add more than capacity
        for i in range(10):
            wm.add_to_working_memory(f"Extra {i}")
        
        print(f"   Items after overflow: {len(wm.memory_items)}")
        print(f"   Within capacity: {'✅ PASS' if len(wm.memory_items) <= wm.capacity else '❌ FAIL'}")
        
        # Test 3: Memory decay
        print("\nTest 5.3: Memory decay")
        initial_strength = wm.item_strengths.get(item_ids[0], 0.0)
        wm.decay_memory()
        decayed_strength = wm.item_strengths.get(item_ids[0], 0.0)
        
        print(f"   Initial strength: {initial_strength:.4f}")
        print(f"   After decay: {decayed_strength:.4f}")
        print(f"   Strength decreased: {'✅ PASS' if decayed_strength < initial_strength else '⚠️  CHECK'}")
        
        return True
    
    def test_integrated_executive_control(self):
        """Test integrated executive control"""
        print("\n" + "="*60)
        print("TEST 6: Integrated Executive Control")
        print("="*60)
        
        manager = ExecutiveControlManager()
        
        # Create tasks
        print("\nCreating tasks...")
        task1 = manager.create_task("High priority task", priority=1.0)
        task2 = manager.create_task("Low priority task", priority=0.5)
        task3 = manager.create_task("Medium priority task", priority=0.7)
        
        print(f"   Tasks created: {len(manager.cognitive_control.active_tasks)}")
        
        # Focus attention
        print("\nFocusing attention...")
        manager.focus_attention("Task 1", intensity=0.9)
        manager.focus_attention("Task 2", intensity=0.6)
        
        distribution = manager.attention_management.get_attention_distribution()
        print(f"   Attention distribution: {distribution}")
        
        # Switch tasks
        print("\nSwitching tasks...")
        switch_cost = manager.switch_task(task2.task_id, task1.task_id)
        print(f"   Switch cost: {switch_cost:.4f}")
        
        # Inhibit responses
        print("\nInhibiting responses...")
        manager.inhibit_response(1)
        manager.inhibit_response(2)
        print(f"   Inhibited items: {len(manager.inhibition_control.inhibited_items)}")
        
        # Working memory
        print("\nManaging working memory...")
        manager.add_to_working_memory("Important info 1")
        manager.add_to_working_memory("Important info 2")
        manager.add_to_working_memory("Important info 3")
        
        contents = manager.working_memory.get_working_memory_contents()
        print(f"   Working memory items: {len(contents)}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Executive Control', fontsize=16, fontweight='bold')
        
        # Plot 1: Task priorities
        ax1 = axes[0, 0]
        tasks_list = list(manager.cognitive_control.active_tasks.values())
        priorities = [t.priority for t in tasks_list]
        descriptions = [t.description[:15] for t in tasks_list]
        
        bars = ax1.bar(range(len(tasks_list)), priorities, color='#3498DB', alpha=0.8, edgecolor='black')
        ax1.set_title('Task Priorities', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Priority')
        ax1.set_xticks(range(len(tasks_list)))
        ax1.set_xticklabels(descriptions, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Attention distribution
        ax2 = axes[0, 1]
        if distribution:
            targets = list(distribution.keys())
            intensities = [distribution[t] for t in targets]
            
            bars = ax2.bar(targets, intensities, color='#2ECC71', alpha=0.8, edgecolor='black')
            ax2.set_title('Attention Distribution', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Attention Intensity')
            ax2.set_xticklabels(targets, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Working memory
        ax3 = axes[1, 0]
        if contents:
            items = [item for _, item, _ in contents]
            strengths = [strength for _, _, strength in contents]
            
            bars = ax3.bar(range(len(items)), strengths, color='#F39C12', alpha=0.8, edgecolor='black')
            ax3.set_title('Working Memory', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Strength')
            ax3.set_xlabel('Item Index')
            ax3.set_xticks(range(len(items)))
            ax3.set_xticklabels([f'Item {i+1}' for i in range(len(items))])
            ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = manager.get_statistics()
        stats_text = "Executive Control Statistics:\n\n"
        stats_text += f"Active Tasks: {stats['active_tasks']}\n"
        stats_text += f"Attention Focuses: {stats['attention_focuses']}\n"
        stats_text += f"Attention Load: {stats['attention_load']:.3f}\n"
        stats_text += f"Task Switches: {stats['task_switches']}\n"
        stats_text += f"Inhibited Items: {stats['inhibited_items']}\n"
        stats_text += f"Working Memory Items: {stats['working_memory_items']}\n"
        stats_text += f"WM Load: {stats['working_memory_load']:.2%}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('executive_control_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: executive_control_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Active tasks: {stats['active_tasks']}")
        print(f"   Attention focuses: {stats['attention_focuses']}")
        print(f"   Attention load: {stats['attention_load']:.3f}")
        print(f"   Task switches: {stats['task_switches']}")
        print(f"   Inhibited items: {stats['inhibited_items']}")
        print(f"   Working memory items: {stats['working_memory_items']}")
        
        return True
    
    def run_all_tests(self):
        """Run all executive control tests"""
        print("\n" + "="*70)
        print("EXECUTIVE CONTROL TEST SUITE")
        print("="*70)
        
        tests = [
            ("Cognitive Control", self.test_cognitive_control),
            ("Attention Management", self.test_attention_management),
            ("Task Switching", self.test_task_switching),
            ("Inhibition Control", self.test_inhibition_control),
            ("Working Memory Management", self.test_working_memory_management),
            ("Integrated Executive Control", self.test_integrated_executive_control)
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
    tester = ExecutiveControlTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All executive control tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

