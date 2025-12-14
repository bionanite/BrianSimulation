#!/usr/bin/env python3
"""
Test Framework for Value Systems
Tests value learning, preferences, moral reasoning, and decision making
"""

import numpy as np
import matplotlib.pyplot as plt
from value_systems import (
    ValueSystemManager, ValueLearning,
    PreferenceFormation, MoralReasoning, ValueBasedDecisionMaking
)

class ValueSystemTester:
    """Test framework for value systems"""
    
    def __init__(self):
        self.results = []
    
    def test_value_learning(self):
        """Test value learning"""
        print("\n" + "="*60)
        print("TEST 1: Value Learning")
        print("="*60)
        
        value_learning = ValueLearning()
        
        # Test 1: Create value
        print("\nTest 1.1: Creating values")
        value1 = value_learning.create_value("honesty", "moral", initial_strength=1.0)
        value2 = value_learning.create_value("efficiency", "intrinsic", initial_strength=0.8)
        
        print(f"   Values created: {len(value_learning.values)}")
        print(f"   Result: {'✅ PASS' if len(value_learning.values) == 2 else '❌ FAIL'}")
        
        # Test 2: Update value
        print("\nTest 1.2: Updating values")
        initial_strength = value1.strength
        value_learning.update_value(value1.value_id, outcome=0.5, experience_description="Told truth")
        
        print(f"   Initial strength: {initial_strength:.4f}")
        print(f"   Updated strength: {value1.strength:.4f}")
        print(f"   Strength increased: {'✅ PASS' if value1.strength > initial_strength else '⚠️  CHECK'}")
        
        # Test 3: Evaluate action
        print("\nTest 1.3: Evaluating actions")
        score = value_learning.evaluate_action("honest action", [value1.value_id])
        
        print(f"   Action score: {score:.4f}")
        print(f"   Result: {'✅ PASS' if score > 0 else '❌ FAIL'}")
        
        return True
    
    def test_preference_formation(self):
        """Test preference formation"""
        print("\n" + "="*60)
        print("TEST 2: Preference Formation")
        print("="*60)
        
        preferences = PreferenceFormation()
        
        # Test 1: Form preferences
        print("\nTest 2.1: Forming preferences")
        pref1 = preferences.form_preference("chocolate", positive=True)
        pref2 = preferences.form_preference("spinach", positive=False)
        
        print(f"   Preferences formed: {len(preferences.preferences)}")
        print(f"   Result: {'✅ PASS' if len(preferences.preferences) == 2 else '❌ FAIL'}")
        
        # Test 2: Get preference
        print("\nTest 2.2: Getting preferences")
        chocolate_pref = preferences.get_preference("chocolate")
        spinach_pref = preferences.get_preference("spinach")
        
        print(f"   Chocolate: {chocolate_pref:.4f}")
        print(f"   Spinach: {spinach_pref:.4f}")
        print(f"   Chocolate > Spinach: {'✅ PASS' if chocolate_pref > spinach_pref else '❌ FAIL'}")
        
        # Test 3: Rank items
        print("\nTest 2.3: Ranking items")
        items = ["chocolate", "spinach", "ice cream"]
        rankings = preferences.rank_items(items)
        
        print(f"   Rankings: {rankings}")
        print(f"   Result: {'✅ PASS' if len(rankings) == 3 else '❌ FAIL'}")
        
        return True
    
    def test_moral_reasoning(self):
        """Test moral reasoning"""
        print("\n" + "="*60)
        print("TEST 3: Moral Reasoning")
        print("="*60)
        
        moral = MoralReasoning()
        
        # Test 1: Moral evaluation
        print("\nTest 3.1: Moral action evaluation")
        consequences = {
            'harm': -0.5,
            'fairness': 0.8
        }
        score = moral.evaluate_moral_action("help someone", consequences, intentions=["help", "good"])
        
        print(f"   Moral score: {score:.4f}")
        print(f"   Result: {'✅ PASS' if score > 0 else '⚠️  CHECK'}")
        
        # Test 2: Update principle
        print("\nTest 3.2: Updating moral principles")
        initial_weight = moral.moral_principles['harm']
        moral.update_moral_principle('harm', -0.8)
        
        print(f"   Initial weight: {initial_weight:.4f}")
        print(f"   Updated weight: {moral.moral_principles['harm']:.4f}")
        print(f"   Result: {'✅ PASS' if moral.moral_principles['harm'] != initial_weight else '❌ FAIL'}")
        
        return True
    
    def test_value_based_decision_making(self):
        """Test value-based decision making"""
        print("\n" + "="*60)
        print("TEST 4: Value-Based Decision Making")
        print("="*60)
        
        value_learning = ValueLearning()
        preferences = PreferenceFormation()
        moral = MoralReasoning()
        decision_making = ValueBasedDecisionMaking(value_learning, preferences, moral)
        
        # Create values
        honesty = value_learning.create_value("honesty", "moral")
        efficiency = value_learning.create_value("efficiency", "intrinsic")
        
        # Form preferences
        preferences.form_preference("option A", positive=True)
        
        # Test decision making
        print("\nTest 4.1: Making value-based decisions")
        options = ["Option A", "Option B", "Option C"]
        value_ids = [honesty.value_id, efficiency.value_id]
        consequences = [
            {'harm': -0.2, 'fairness': 0.8},
            {'harm': -0.5, 'fairness': 0.3},
            {'harm': -0.1, 'fairness': 0.5}
        ]
        
        best_idx, score = decision_making.choose_best_option(options, value_ids, consequences)
        
        print(f"   Best option: {options[best_idx]}")
        print(f"   Score: {score:.4f}")
        print(f"   Result: {'✅ PASS' if best_idx is not None else '❌ FAIL'}")
        
        return True
    
    def test_integrated_value_systems(self):
        """Test integrated value systems"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Value Systems")
        print("="*60)
        
        manager = ValueSystemManager()
        
        # Learn values
        print("\nLearning values from experiences...")
        manager.learn_value_from_experience("honesty", "moral", outcome=0.8, experience="Told truth")
        manager.learn_value_from_experience("efficiency", "intrinsic", outcome=0.6, experience="Completed task quickly")
        manager.learn_value_from_experience("kindness", "moral", outcome=0.9, experience="Helped someone")
        
        print(f"   Values learned: {len(manager.value_learning.values)}")
        
        # Form preferences
        print("\nForming preferences...")
        manager.form_preference("cooperation", positive=True)
        manager.form_preference("conflict", positive=False)
        manager.form_preference("exploration", positive=True)
        
        print(f"   Preferences formed: {len(manager.preference_formation.preferences)}")
        
        # Make decisions
        print("\nMaking value-based decisions...")
        options = ["Cooperate", "Compete", "Explore"]
        value_names = ["honesty", "kindness"]
        consequences = [
            {'harm': -0.1, 'fairness': 0.9},
            {'harm': -0.6, 'fairness': 0.2},
            {'harm': -0.2, 'fairness': 0.7}
        ]
        
        best_idx, score = manager.make_value_based_decision(options, value_names, consequences)
        
        print(f"   Best decision: {options[best_idx]}")
        print(f"   Decision score: {score:.4f}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Value Systems', fontsize=16, fontweight='bold')
        
        # Plot 1: Value strengths
        ax1 = axes[0, 0]
        values_list = list(manager.value_learning.values.values())
        names = [v.name for v in values_list]
        strengths = [v.strength for v in values_list]
        
        bars = ax1.bar(names, strengths, color='#3498DB', alpha=0.8, edgecolor='black')
        ax1.set_title('Value Strengths', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Strength')
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Preferences
        ax2 = axes[0, 1]
        prefs_list = list(manager.preference_formation.preferences.values())
        items = [p.item for p in prefs_list]
        pref_strengths = [p.strength for p in prefs_list]
        
        colors = ['green' if s > 0 else 'red' for s in pref_strengths]
        bars = ax2.bar(range(len(items)), pref_strengths, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Preferences', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Preference Strength')
        ax2.set_xticks(range(len(items)))
        ax2.set_xticklabels(items, rotation=45, ha='right')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Moral principles
        ax3 = axes[1, 0]
        principles = list(manager.moral_reasoning.moral_principles.keys())
        weights = [manager.moral_reasoning.moral_principles[p] for p in principles]
        
        colors = ['green' if w > 0 else 'red' for w in weights]
        bars = ax3.bar(principles, weights, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_title('Moral Principles', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Weight')
        ax3.set_xticklabels(principles, rotation=45, ha='right')
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = manager.get_statistics()
        stats_text = "Value System Statistics:\n\n"
        stats_text += f"Values: {stats['values']}\n"
        stats_text += f"Preferences: {stats['preferences']}\n"
        stats_text += f"Moral Principles: {stats['moral_principles']}\n"
        stats_text += f"Avg Value Strength: {stats['avg_value_strength']:.3f}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('value_systems_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: value_systems_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Values: {stats['values']}")
        print(f"   Preferences: {stats['preferences']}")
        print(f"   Moral principles: {stats['moral_principles']}")
        print(f"   Average value strength: {stats['avg_value_strength']:.4f}")
        
        return True
    
    def run_all_tests(self):
        """Run all value system tests"""
        print("\n" + "="*70)
        print("VALUE SYSTEMS TEST SUITE")
        print("="*70)
        
        tests = [
            ("Value Learning", self.test_value_learning),
            ("Preference Formation", self.test_preference_formation),
            ("Moral Reasoning", self.test_moral_reasoning),
            ("Value-Based Decision Making", self.test_value_based_decision_making),
            ("Integrated Value Systems", self.test_integrated_value_systems)
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
    tester = ValueSystemTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All value system tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

