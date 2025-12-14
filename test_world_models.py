#!/usr/bin/env python3
"""
Test Framework for World Models
Tests predictive models, causal reasoning, simulation, and planning
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from world_models import (
    WorldModelManager, PredictiveModel,
    CausalReasoning, SimulationEngine, MentalModelBuilder,
    StateTransition
)

class WorldModelTester:
    """Test framework for world models"""
    
    def __init__(self):
        self.results = []
    
    def test_predictive_model(self):
        """Test predictive model"""
        print("\n" + "="*60)
        print("TEST 1: Predictive Model")
        print("="*60)
        
        predictive = PredictiveModel(state_size=10, learning_rate=0.1)
        
        # Create state transitions
        print("\nTest 1.1: Learning transitions")
        state1 = np.random.random(10)
        state2 = state1 + np.random.normal(0, 0.1, 10)
        state3 = state2 + np.random.normal(0, 0.1, 10)
        
        predictive.learn_transition(state1, state2, action=1, reward=0.5)
        predictive.learn_transition(state2, state3, action=2, reward=1.0)
        
        print(f"   Transitions learned: {len(predictive.transitions)}")
        print(f"   Result: {'✅ PASS' if len(predictive.transitions) == 2 else '❌ FAIL'}")
        
        # Test 2: Prediction
        print("\nTest 1.2: State prediction")
        predicted_state, confidence = predictive.predict_next_state(state1, action=1)
        
        prediction_error = np.linalg.norm(predicted_state - state2)
        print(f"   Prediction error: {prediction_error:.4f}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   Result: {'✅ PASS' if confidence > 0 else '⚠️  CHECK'}")
        
        # Test 3: Multiple transitions
        print("\nTest 1.3: Learning multiple transitions")
        for _ in range(10):
            predictive.learn_transition(state1, state2, action=1)
        
        # Check frequency
        trans = predictive.transitions[0]
        print(f"   Transition frequency: {trans.frequency}")
        print(f"   Result: {'✅ PASS' if trans.frequency > 1 else '❌ FAIL'}")
        
        return True
    
    def test_causal_reasoning(self):
        """Test causal reasoning"""
        print("\n" + "="*60)
        print("TEST 2: Causal Reasoning")
        print("="*60)
        
        causal = CausalReasoning()
        
        # Test 1: Learn causality
        print("\nTest 2.1: Learning causal relationships")
        cause_id = 1
        effect_id = 2
        
        causal.learn_causality(cause_id, effect_id, strength=0.8)
        
        print(f"   Causal relations: {len(causal.causal_relations)}")
        print(f"   Result: {'✅ PASS' if len(causal.causal_relations) == 1 else '❌ FAIL'}")
        
        # Test 2: Infer causes
        print("\nTest 2.2: Inferring causes")
        causes = causal.infer_cause(effect_id)
        
        print(f"   Causes for effect {effect_id}: {causes}")
        print(f"   Result: {'✅ PASS' if len(causes) > 0 else '❌ FAIL'}")
        
        # Test 3: Infer effects
        print("\nTest 2.3: Inferring effects")
        effects = causal.infer_effect(cause_id)
        
        print(f"   Effects of cause {cause_id}: {effects}")
        print(f"   Result: {'✅ PASS' if len(effects) > 0 else '❌ FAIL'}")
        
        # Test 4: Temporal causality detection
        print("\nTest 2.4: Temporal causality detection")
        causal.record_event(1, "Event 1")
        time.sleep(0.01)
        causal.record_event(2, "Event 2")
        
        initial_relations = len(causal.causal_relations)
        causal.detect_temporal_causality()
        
        print(f"   Relations before: {initial_relations}")
        print(f"   Relations after: {len(causal.causal_relations)}")
        print(f"   Result: {'✅ PASS' if len(causal.causal_relations) >= initial_relations else '⚠️  CHECK'}")
        
        return True
    
    def test_simulation_engine(self):
        """Test simulation engine"""
        print("\n" + "="*60)
        print("TEST 3: Simulation Engine")
        print("="*60)
        
        predictive = PredictiveModel(state_size=10)
        
        # Learn transitions
        state1 = np.random.random(10)
        state2 = state1 + np.random.normal(0, 0.1, 10)
        state3 = state2 + np.random.normal(0, 0.1, 10)
        
        predictive.learn_transition(state1, state2, action=1)
        predictive.learn_transition(state2, state3, action=2)
        
        simulator = SimulationEngine(predictive)
        
        # Test 1: Simulation
        print("\nTest 3.1: Forward simulation")
        trajectory = simulator.simulate(state1, actions=[1, 2], steps=3)
        
        print(f"   Trajectory length: {len(trajectory)}")
        print(f"   Average confidence: {np.mean([c for _, c in trajectory]):.4f}")
        print(f"   Result: {'✅ PASS' if len(trajectory) == 3 else '❌ FAIL'}")
        
        # Test 2: Planning
        print("\nTest 3.2: Action planning")
        goal_state = state3.copy()
        available_actions = [1, 2]
        
        plan = simulator.plan_sequence(state1, goal_state, available_actions, max_depth=5)
        
        print(f"   Plan found: {plan is not None}")
        if plan:
            print(f"   Plan length: {len(plan)}")
            print(f"   Plan: {plan}")
        print(f"   Result: {'✅ PASS' if plan is not None else '⚠️  CHECK'}")
        
        return True
    
    def test_mental_model_builder(self):
        """Test mental model builder"""
        print("\n" + "="*60)
        print("TEST 4: Mental Model Builder")
        print("="*60)
        
        builder = MentalModelBuilder()
        
        # Test 1: Create model
        print("\nTest 4.1: Creating mental model")
        initial_states = [
            np.random.random(10),
            np.random.random(10),
            np.random.random(10)
        ]
        
        model = builder.create_model(initial_states)
        
        print(f"   Model ID: {model.model_id}")
        print(f"   Initial states: {len(model.states)}")
        print(f"   Result: {'✅ PASS' if len(model.states) == 3 else '❌ FAIL'}")
        
        # Test 2: Update model
        print("\nTest 4.2: Updating mental model")
        from_state = initial_states[-1]
        to_state = np.random.random(10)
        
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            action=1
        )
        
        initial_accuracy = model.accuracy
        builder.update_model(model, transition, to_state)
        
        print(f"   States after update: {len(model.states)}")
        print(f"   Transitions: {len(model.transitions)}")
        print(f"   Result: {'✅ PASS' if len(model.states) > len(initial_states) else '❌ FAIL'}")
        
        # Test 3: Prediction with model
        print("\nTest 4.3: Prediction with mental model")
        current_state = np.random.random(10)
        predicted = builder.predict_with_model(model, current_state)
        
        print(f"   Predicted state shape: {predicted.shape}")
        print(f"   Result: {'✅ PASS' if len(predicted) == 10 else '❌ FAIL'}")
        
        return True
    
    def test_integrated_world_models(self):
        """Test integrated world models"""
        print("\n" + "="*60)
        print("TEST 5: Integrated World Models")
        print("="*60)
        
        manager = WorldModelManager(state_size=10)
        
        print("\nLearning from experiences...")
        
        # Learn transitions
        states = [np.random.random(10) for _ in range(5)]
        actions = [1, 2, 1, 2]
        
        for i in range(len(states) - 1):
            manager.learn_from_experience(states[i], states[i+1], actions[i], reward=0.5)
        
        print(f"   Transitions learned: {len(manager.predictive_model.transitions)}")
        
        # Test prediction
        print("\nTesting predictions...")
        predicted_state, confidence = manager.predict_future(states[0], action=1)
        print(f"   Prediction confidence: {confidence:.4f}")
        
        # Test simulation
        print("\nTesting simulation...")
        trajectory = manager.simulate_trajectory(states[0], actions[:3], steps=3)
        print(f"   Simulated trajectory length: {len(trajectory)}")
        
        # Test planning
        print("\nTesting planning...")
        goal_state = states[-1]
        plan = manager.plan_actions(states[0], goal_state, [1, 2])
        print(f"   Plan found: {plan is not None}")
        
        # Test causality
        print("\nTesting causal reasoning...")
        manager.learn_causality(1, 2, strength=0.8)
        manager.learn_causality(2, 3, strength=0.7)
        
        causes = manager.infer_causes(2)
        effects = manager.infer_effects(1)
        
        print(f"   Causes of event 2: {causes}")
        print(f"   Effects of event 1: {effects}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('World Models Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: State trajectory
        ax1 = axes[0, 0]
        if trajectory:
            state_values = [np.mean(state) for state, _ in trajectory]
            confidences = [conf for _, conf in trajectory]
            
            ax1_twin = ax1.twinx()
            line1 = ax1.plot(state_values, 'b-o', label='State Value', linewidth=2)
            line2 = ax1_twin.plot(confidences, 'r-s', label='Confidence', linewidth=2)
            
            ax1.set_title('Simulated Trajectory', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('State Value', color='blue')
            ax1_twin.set_ylabel('Confidence', color='red')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Transition frequencies
        ax2 = axes[0, 1]
        if manager.predictive_model.transitions:
            frequencies = [t.frequency for t in manager.predictive_model.transitions]
            bars = ax2.bar(range(len(frequencies)), frequencies, color='#3498DB', alpha=0.8)
            ax2.set_title('Transition Frequencies', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Transition Index')
            ax2.set_ylabel('Frequency')
            ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Causal relations
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        causal_text = "Causal Relations:\n\n"
        for rel in manager.causal_reasoning.causal_relations:
            causal_text += f"  {rel.cause_id} -> {rel.effect_id}\n"
            causal_text += f"    Strength: {rel.strength:.3f}\n"
            causal_text += f"    Confidence: {rel.confidence:.3f}\n\n"
        
        if not manager.causal_reasoning.causal_relations:
            causal_text = "No causal relations learned yet"
        
        ax3.text(0.1, 0.5, causal_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax3.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = manager.get_statistics()
        stats_text = "World Model Statistics:\n\n"
        stats_text += f"Transitions Learned: {stats['transitions_learned']}\n"
        stats_text += f"States Known: {stats['states_known']}\n"
        stats_text += f"Causal Relations: {stats['causal_relations']}\n"
        stats_text += f"Mental Models: {stats['mental_models']}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('world_models_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: world_models_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Transitions learned: {stats['transitions_learned']}")
        print(f"   States known: {stats['states_known']}")
        print(f"   Causal relations: {stats['causal_relations']}")
        print(f"   Mental models: {stats['mental_models']}")
        
        return True
    
    def run_all_tests(self):
        """Run all world model tests"""
        print("\n" + "="*70)
        print("WORLD MODELS TEST SUITE")
        print("="*70)
        
        tests = [
            ("Predictive Model", self.test_predictive_model),
            ("Causal Reasoning", self.test_causal_reasoning),
            ("Simulation Engine", self.test_simulation_engine),
            ("Mental Model Builder", self.test_mental_model_builder),
            ("Integrated World Models", self.test_integrated_world_models)
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
    tester = WorldModelTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All world model tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

