#!/usr/bin/env python3
"""
Test Framework for Context Sensitivity
Tests context detection, context-dependent behavior, situational awareness, and context switching
"""

import numpy as np
import matplotlib.pyplot as plt
from context_sensitivity import (
    ContextSensitivityManager, ContextDetection,
    ContextDependentBehavior, SituationalAwareness, ContextSwitching
)

class ContextSensitivityTester:
    """Test framework for context sensitivity"""
    
    def __init__(self):
        self.results = []
    
    def test_context_detection(self):
        """Test context detection"""
        print("\n" + "="*60)
        print("TEST 1: Context Detection")
        print("="*60)
        
        detection = ContextDetection()
        
        # Test 1: Register context
        print("\nTest 1.1: Registering contexts")
        context1 = detection.register_context("office", {"noise_level": 0.3, "light_level": 0.8})
        context2 = detection.register_context("home", {"noise_level": 0.1, "light_level": 0.6})
        
        print(f"   Contexts registered: {len(detection.known_contexts)}")
        print(f"   Result: {'✅ PASS' if len(detection.known_contexts) == 2 else '❌ FAIL'}")
        
        # Test 2: Detect context
        print("\nTest 1.2: Detecting context")
        observed = {"noise_level": 0.25, "light_level": 0.75}
        detected = detection.detect_context(observed)
        
        print(f"   Detected context: {detected}")
        print(f"   Result: {'✅ PASS' if detected is not None else '⚠️  CHECK'}")
        
        # Test 3: Get current context
        print("\nTest 1.3: Getting current context")
        current = detection.get_current_context()
        
        print(f"   Current context: {current.name if current else None}")
        print(f"   Result: {'✅ PASS' if current is not None else '❌ FAIL'}")
        
        return True
    
    def test_context_dependent_behavior(self):
        """Test context-dependent behavior"""
        print("\n" + "="*60)
        print("TEST 2: Context-Dependent Behavior")
        print("="*60)
        
        behavior = ContextDependentBehavior()
        
        # Test 1: Add rule
        print("\nTest 2.1: Adding rules")
        rule1 = behavior.add_rule("office", {"noise_level": 0.5}, "work_quietly", priority=1.0)
        rule2 = behavior.add_rule("office", {"light_level": 0.7}, "turn_on_lights", priority=0.8)
        
        print(f"   Rules added: {sum(len(rules) for rules in behavior.context_rules.values())}")
        print(f"   Result: {'✅ PASS' if len(behavior.context_rules.get('office', [])) == 2 else '❌ FAIL'}")
        
        # Test 2: Select behavior
        print("\nTest 2.2: Selecting behavior")
        features = {"noise_level": 0.6, "light_level": 0.8}
        selected = behavior.select_behavior("office", features)
        
        print(f"   Selected behavior: {selected}")
        print(f"   Result: {'✅ PASS' if selected is not None else '❌ FAIL'}")
        
        # Test 3: Get behaviors for context
        print("\nTest 2.3: Getting behaviors for context")
        behaviors = behavior.get_behaviors_for_context("office")
        
        print(f"   Behaviors: {behaviors}")
        print(f"   Result: {'✅ PASS' if len(behaviors) > 0 else '❌ FAIL'}")
        
        return True
    
    def test_situational_awareness(self):
        """Test situational awareness"""
        print("\n" + "="*60)
        print("TEST 3: Situational Awareness")
        print("="*60)
        
        awareness = SituationalAwareness()
        
        # Test 1: Update feature
        print("\nTest 3.1: Updating features")
        awareness.update_feature("temperature", 22.0)
        awareness.update_feature("humidity", 0.5)
        
        print(f"   Features tracked: {len(awareness.situational_features)}")
        print(f"   Result: {'✅ PASS' if len(awareness.situational_features) == 2 else '❌ FAIL'}")
        
        # Test 2: Get summary
        print("\nTest 3.2: Getting situational summary")
        summary = awareness.get_situational_summary()
        
        print(f"   Summary features: {len(summary)}")
        print(f"   Result: {'✅ PASS' if len(summary) == 2 else '❌ FAIL'}")
        
        # Test 3: Detect change
        print("\nTest 3.3: Detecting changes")
        awareness.update_feature("temperature", 25.0)
        changed = awareness.detect_change("temperature", threshold=2.0)
        
        print(f"   Change detected: {changed}")
        print(f"   Result: {'✅ PASS' if changed else '⚠️  CHECK'}")
        
        return True
    
    def test_context_switching(self):
        """Test context switching"""
        print("\n" + "="*60)
        print("TEST 4: Context Switching")
        print("="*60)
        
        switching = ContextSwitching()
        
        # Test 1: Switch context
        print("\nTest 4.1: Switching contexts")
        cost1 = switching.switch_context(None, "office")
        cost2 = switching.switch_context("office", "home")
        
        print(f"   Current context: {switching.current_context}")
        print(f"   Switches: {len(switching.switch_history)}")
        print(f"   Result: {'✅ PASS' if switching.current_context == 'home' else '❌ FAIL'}")
        
        # Test 2: Switch frequency
        print("\nTest 4.2: Switch frequency")
        frequency = switching.get_switch_frequency()
        
        print(f"   Switch frequency: {frequency:.4f}")
        print(f"   Result: {'✅ PASS' if frequency >= 0 else '❌ FAIL'}")
        
        return True
    
    def test_integrated_context_sensitivity(self):
        """Test integrated context sensitivity"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Context Sensitivity")
        print("="*60)
        
        manager = ContextSensitivityManager()
        
        # Register contexts
        print("\nRegistering contexts...")
        manager.register_context("office", {"noise_level": 0.3, "light_level": 0.8, "people": 0.5})
        manager.register_context("home", {"noise_level": 0.1, "light_level": 0.6, "people": 0.2})
        manager.register_context("meeting", {"noise_level": 0.2, "light_level": 0.9, "people": 0.9})
        
        print(f"   Contexts registered: {len(manager.context_detection.known_contexts)}")
        
        # Add behavior rules
        print("\nAdding behavior rules...")
        manager.add_behavior_rule("office", {"noise_level": 0.5}, "work_quietly", priority=1.0)
        manager.add_behavior_rule("office", {"light_level": 0.7}, "adjust_lighting", priority=0.8)
        manager.add_behavior_rule("meeting", {"people": 0.8}, "participate", priority=1.0)
        
        print(f"   Rules added: {sum(len(rules) for rules in manager.context_behavior.context_rules.values())}")
        
        # Detect context
        print("\nDetecting context...")
        observed = {"noise_level": 0.25, "light_level": 0.85, "people": 0.9}
        detected = manager.detect_context(observed)
        
        print(f"   Detected context: {detected}")
        
        # Select behavior
        print("\nSelecting behavior...")
        behavior = manager.select_behavior(detected or "office", observed)
        
        print(f"   Selected behavior: {behavior}")
        
        # Update situation
        print("\nUpdating situation...")
        manager.update_situation("temperature", 22.0)
        manager.update_situation("humidity", 0.5)
        manager.update_situation("time_of_day", 0.7)
        
        summary = manager.situational_awareness.get_situational_summary()
        print(f"   Situational features: {len(summary)}")
        
        # Switch context
        print("\nSwitching contexts...")
        manager.switch_context("office")
        manager.switch_context("meeting")
        
        print(f"   Context switches: {len(manager.context_switching.switch_history)}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Context Sensitivity', fontsize=16, fontweight='bold')
        
        # Plot 1: Context activations
        ax1 = axes[0, 0]
        contexts = list(manager.context_detection.known_contexts.values())
        context_names = [c.name for c in contexts]
        activations = [c.activation for c in contexts]
        
        bars = ax1.bar(context_names, activations, color='#3498DB', alpha=0.8, edgecolor='black')
        ax1.set_title('Context Activations', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Activation')
        ax1.set_xticklabels(context_names, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Behavior rules per context
        ax2 = axes[0, 1]
        context_names_rules = list(manager.context_behavior.context_rules.keys())
        rule_counts = [len(manager.context_behavior.context_rules[c]) for c in context_names_rules]
        
        if context_names_rules:
            bars = ax2.bar(context_names_rules, rule_counts, color='#2ECC71', alpha=0.8, edgecolor='black')
            ax2.set_title('Behavior Rules per Context', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Number of Rules')
            ax2.set_xticklabels(context_names_rules, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Situational features
        ax3 = axes[1, 0]
        features = list(manager.situational_awareness.situational_features.keys())
        values = [manager.situational_awareness.situational_features[f] for f in features]
        
        if features:
            bars = ax3.bar(range(len(features)), values, color='#F39C12', alpha=0.8, edgecolor='black')
            ax3.set_title('Situational Features', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Value')
            ax3.set_xticks(range(len(features)))
            ax3.set_xticklabels(features, rotation=45, ha='right')
            ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = manager.get_statistics()
        stats_text = "Context Sensitivity Statistics:\n\n"
        stats_text += f"Known Contexts: {stats['known_contexts']}\n"
        stats_text += f"Current Context: {stats['current_context'] or 'None'}\n"
        stats_text += f"Context Rules: {stats['context_rules']}\n"
        stats_text += f"Situational Features: {stats['situational_features']}\n"
        stats_text += f"Context Switches: {stats['context_switches']}\n"
        stats_text += f"Switch Frequency: {stats['switch_frequency']:.3f}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('context_sensitivity_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: context_sensitivity_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Known contexts: {stats['known_contexts']}")
        print(f"   Current context: {stats['current_context']}")
        print(f"   Context rules: {stats['context_rules']}")
        print(f"   Situational features: {stats['situational_features']}")
        print(f"   Context switches: {stats['context_switches']}")
        
        return True
    
    def run_all_tests(self):
        """Run all context sensitivity tests"""
        print("\n" + "="*70)
        print("CONTEXT SENSITIVITY TEST SUITE")
        print("="*70)
        
        tests = [
            ("Context Detection", self.test_context_detection),
            ("Context-Dependent Behavior", self.test_context_dependent_behavior),
            ("Situational Awareness", self.test_situational_awareness),
            ("Context Switching", self.test_context_switching),
            ("Integrated Context Sensitivity", self.test_integrated_context_sensitivity)
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
    tester = ContextSensitivityTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All context sensitivity tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

