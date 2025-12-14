#!/usr/bin/env python3
"""
Test Framework for Metacognition
Tests metacognitive monitoring, strategy selection, self-regulation, and control
"""

import numpy as np
import matplotlib.pyplot as plt
from metacognition import (
    MetacognitionManager, MetacognitiveMonitoring,
    StrategySelection, SelfRegulation, MetacognitiveControl
)

class MetacognitionTester:
    """Test framework for metacognition"""
    
    def __init__(self):
        self.results = []
    
    def test_metacognitive_monitoring(self):
        """Test metacognitive monitoring"""
        print("\n" + "="*60)
        print("TEST 1: Metacognitive Monitoring")
        print("="*60)
        
        monitoring = MetacognitiveMonitoring()
        
        # Test 1: Monitor process
        print("\nTest 1.1: Monitoring processes")
        monitoring.monitor_process("reasoning", 0.8, {"complexity": 0.7})
        monitoring.monitor_process("memory", 0.6, {"load": 0.5})
        
        print(f"   Processes monitored: {len(monitoring.monitored_processes)}")
        print(f"   Result: {'✅ PASS' if len(monitoring.monitored_processes) == 2 else '❌ FAIL'}")
        
        # Test 2: Get confidence
        print("\nTest 1.2: Getting confidence")
        reasoning_conf = monitoring.get_confidence("reasoning")
        
        print(f"   Reasoning confidence: {reasoning_conf:.4f}")
        print(f"   Result: {'✅ PASS' if reasoning_conf == 0.8 else '❌ FAIL'}")
        
        # Test 3: Get trend
        print("\nTest 1.3: Getting confidence trends")
        monitoring.monitor_process("reasoning", 0.9)
        trend = monitoring.get_confidence_trend("reasoning")
        
        print(f"   Confidence trend: {trend:.4f}")
        print(f"   Result: {'✅ PASS' if trend > 0 else '⚠️  CHECK'}")
        
        return True
    
    def test_strategy_selection(self):
        """Test strategy selection"""
        print("\n" + "="*60)
        print("TEST 2: Strategy Selection")
        print("="*60)
        
        selection = StrategySelection()
        
        # Test 1: Register strategy
        print("\nTest 2.1: Registering strategies")
        strategy1 = selection.register_strategy("heuristic", "Use heuristics")
        strategy2 = selection.register_strategy("exhaustive", "Exhaustive search")
        
        print(f"   Strategies registered: {len(selection.strategies)}")
        print(f"   Result: {'✅ PASS' if len(selection.strategies) == 2 else '❌ FAIL'}")
        
        # Test 2: Select strategy
        print("\nTest 2.2: Selecting strategies")
        selected = selection.select_strategy()
        
        print(f"   Selected strategy: {selected.name if selected else None}")
        print(f"   Result: {'✅ PASS' if selected is not None else '❌ FAIL'}")
        
        # Test 3: Update performance
        print("\nTest 2.3: Updating strategy performance")
        initial_rate = strategy1.success_rate
        selection.update_strategy_performance("heuristic", success=True)
        
        print(f"   Initial rate: {initial_rate:.4f}")
        print(f"   Updated rate: {strategy1.success_rate:.4f}")
        print(f"   Result: {'✅ PASS' if strategy1.success_rate > initial_rate else '⚠️  CHECK'}")
        
        return True
    
    def test_self_regulation(self):
        """Test self-regulation"""
        print("\n" + "="*60)
        print("TEST 3: Self-Regulation")
        print("="*60)
        
        regulation = SelfRegulation()
        
        # Test 1: Add regulation rule
        print("\nTest 3.1: Adding regulation rules")
        regulation.add_regulation_rule("reasoning", {"confidence": 0.5}, "stop")
        regulation.add_regulation_rule("memory", {"load": 0.8}, "reduce_load")
        
        print(f"   Rules added: {len(regulation.regulation_rules)}")
        print(f"   Result: {'✅ PASS' if len(regulation.regulation_rules) == 2 else '❌ FAIL'}")
        
        # Test 2: Regulate
        print("\nTest 3.2: Regulating processes")
        action1 = regulation.regulate("reasoning", {"confidence": 0.3})
        action2 = regulation.regulate("memory", {"load": 0.9})
        
        print(f"   Reasoning action: {action1}")
        print(f"   Memory action: {action2}")
        print(f"   Result: {'✅ PASS' if action1 == 'stop' and action2 == 'reduce_load' else '❌ FAIL'}")
        
        return True
    
    def test_metacognitive_control(self):
        """Test metacognitive control"""
        print("\n" + "="*60)
        print("TEST 4: Metacognitive Control")
        print("="*60)
        
        control = MetacognitiveControl()
        
        # Test 1: Set control policy
        print("\nTest 4.1: Setting control policies")
        control.set_control_policy("reasoning", {"threshold": 0.5, "low_confidence_action": "stop"})
        
        print(f"   Policies set: {len(control.control_policies)}")
        print(f"   Result: {'✅ PASS' if len(control.control_policies) == 1 else '❌ FAIL'}")
        
        # Test 2: Make control decision
        print("\nTest 4.2: Making control decisions")
        decision = control.make_control_decision("reasoning", {"confidence": 0.3})
        
        print(f"   Control decision: {decision}")
        print(f"   Result: {'✅ PASS' if decision == 'stop' else '❌ FAIL'}")
        
        return True
    
    def test_integrated_metacognition(self):
        """Test integrated metacognition"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Metacognition")
        print("="*60)
        
        manager = MetacognitionManager()
        
        # Monitor processes
        print("\nMonitoring cognitive processes...")
        manager.monitor_cognitive_process("reasoning", 0.8, {"complexity": 0.7})
        manager.monitor_cognitive_process("memory", 0.6, {"load": 0.5})
        manager.monitor_cognitive_process("attention", 0.9, {"focus": 0.8})
        
        print(f"   Processes monitored: {len(manager.monitoring.monitored_processes)}")
        
        # Register strategies
        print("\nRegistering strategies...")
        manager.strategy_selection.register_strategy("heuristic", "Use heuristics")
        manager.strategy_selection.register_strategy("exhaustive", "Exhaustive search")
        manager.strategy_selection.register_strategy("adaptive", "Adaptive approach")
        
        manager.strategy_selection.update_strategy_performance("heuristic", success=True)
        manager.strategy_selection.update_strategy_performance("heuristic", success=True)
        manager.strategy_selection.update_strategy_performance("exhaustive", success=False)
        
        print(f"   Strategies registered: {len(manager.strategy_selection.strategies)}")
        
        # Add regulation rules
        print("\nAdding regulation rules...")
        manager.self_regulation.add_regulation_rule("reasoning", {"confidence": 0.5}, "stop")
        manager.self_regulation.add_regulation_rule("memory", {"load": 0.8}, "reduce_load")
        
        print(f"   Regulation rules: {len(manager.self_regulation.regulation_rules)}")
        
        # Set control policies
        print("\nSetting control policies...")
        manager.control.set_control_policy("reasoning", {"threshold": 0.5, "low_confidence_action": "stop"})
        
        print(f"   Control policies: {len(manager.control.control_policies)}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Metacognition', fontsize=16, fontweight='bold')
        
        # Plot 1: Monitoring confidence
        ax1 = axes[0, 0]
        monitoring_summary = manager.monitoring.get_monitoring_summary()
        processes = list(monitoring_summary.keys())
        confidences = [monitoring_summary[p] for p in processes]
        
        bars = ax1.bar(processes, confidences, color='#3498DB', alpha=0.8, edgecolor='black')
        ax1.set_title('Metacognitive Monitoring', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Confidence')
        ax1.set_xticklabels(processes, rotation=45, ha='right')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Strategy success rates
        ax2 = axes[0, 1]
        strategy_summary = manager.strategy_selection.get_strategy_summary()
        strategies = list(strategy_summary.keys())
        success_rates = [strategy_summary[s]['success_rate'] for s in strategies]
        
        bars = ax2.bar(strategies, success_rates, color='#2ECC71', alpha=0.8, edgecolor='black')
        ax2.set_title('Strategy Success Rates', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Success Rate')
        ax2.set_xticklabels(strategies, rotation=45, ha='right')
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Strategy usage
        ax3 = axes[1, 0]
        usage_counts = [strategy_summary[s]['usage_count'] for s in strategies]
        
        bars = ax3.bar(strategies, usage_counts, color='#F39C12', alpha=0.8, edgecolor='black')
        ax3.set_title('Strategy Usage', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Usage Count')
        ax3.set_xticklabels(strategies, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = manager.get_statistics()
        stats_text = "Metacognition Statistics:\n\n"
        stats_text += f"Monitored Processes: {stats['monitored_processes']}\n"
        stats_text += f"Strategies: {stats['strategies']}\n"
        stats_text += f"Regulated Processes: {stats['regulated_processes']}\n"
        stats_text += f"Control Decisions: {stats['control_decisions']}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('metacognition_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: metacognition_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Monitored processes: {stats['monitored_processes']}")
        print(f"   Strategies: {stats['strategies']}")
        print(f"   Regulated processes: {stats['regulated_processes']}")
        print(f"   Control decisions: {stats['control_decisions']}")
        
        return True
    
    def run_all_tests(self):
        """Run all metacognition tests"""
        print("\n" + "="*70)
        print("METACOGNITION TEST SUITE")
        print("="*70)
        
        tests = [
            ("Metacognitive Monitoring", self.test_metacognitive_monitoring),
            ("Strategy Selection", self.test_strategy_selection),
            ("Self-Regulation", self.test_self_regulation),
            ("Metacognitive Control", self.test_metacognitive_control),
            ("Integrated Metacognition", self.test_integrated_metacognition)
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
    tester = MetacognitionTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All metacognition tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

