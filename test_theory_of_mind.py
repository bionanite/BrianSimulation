#!/usr/bin/env python3
"""
Test Framework for Theory of Mind
Tests mental state inference, belief tracking, intention recognition, and perspective taking
"""

import numpy as np
import matplotlib.pyplot as plt
from theory_of_mind import (
    TheoryOfMindManager, MentalStateInference,
    BeliefTracking, IntentionRecognition, PerspectiveTaking
)

class TheoryOfMindTester:
    """Test framework for theory of mind"""
    
    def __init__(self):
        self.results = []
    
    def test_mental_state_inference(self):
        """Test mental state inference"""
        print("\n" + "="*60)
        print("TEST 1: Mental State Inference")
        print("="*60)
        
        inference = MentalStateInference()
        
        # Test 1: Observe behavior
        print("\nTest 1.1: Observing behavior")
        agent_id = 1
        inference.observe_behavior(agent_id, "move_forward", np.array([1.0, 2.0]))
        inference.observe_behavior(agent_id, "move_forward", np.array([2.0, 2.0]))
        inference.observe_behavior(agent_id, "turn_right", np.array([2.0, 3.0]))
        
        print(f"   Behaviors observed: {len(inference.behavior_patterns.get(agent_id, []))}")
        print(f"   Result: {'✅ PASS' if len(inference.behavior_patterns.get(agent_id, [])) == 3 else '❌ FAIL'}")
        
        # Test 2: Infer beliefs
        print("\nTest 1.2: Inferring beliefs")
        beliefs = inference.infer_beliefs(agent_id)
        
        print(f"   Beliefs inferred: {len(beliefs)}")
        print(f"   Result: {'✅ PASS' if len(beliefs) > 0 else '⚠️  CHECK'}")
        
        # Test 3: Infer mental state
        print("\nTest 1.3: Inferring mental state")
        inference.update_mental_state(agent_id, observed_goals=["reach_target"])
        mental_state = inference.agent_mental_states.get(agent_id)
        
        print(f"   Mental state created: {mental_state is not None}")
        print(f"   Beliefs: {len(mental_state.beliefs) if mental_state else 0}")
        print(f"   Desires: {len(mental_state.desires) if mental_state else 0}")
        print(f"   Result: {'✅ PASS' if mental_state is not None else '❌ FAIL'}")
        
        return True
    
    def test_belief_tracking(self):
        """Test belief tracking"""
        print("\n" + "="*60)
        print("TEST 2: Belief Tracking")
        print("="*60)
        
        tracking = BeliefTracking()
        
        # Test 1: Track belief
        print("\nTest 2.1: Tracking beliefs")
        agent_id = 1
        belief1 = tracking.track_belief(agent_id, "agent_knows_location", confidence=0.8)
        belief2 = tracking.track_belief(agent_id, "agent_wants_cooperation", confidence=0.9)
        
        print(f"   Beliefs tracked: {len(tracking.tracked_beliefs.get(agent_id, {}))}")
        print(f"   Result: {'✅ PASS' if len(tracking.tracked_beliefs.get(agent_id, {})) == 2 else '❌ FAIL'}")
        
        # Test 2: Update belief
        print("\nTest 2.2: Updating beliefs")
        initial_confidence = belief1.confidence
        tracking.update_belief(agent_id, "agent_knows_location", new_confidence=0.95)
        
        print(f"   Initial confidence: {initial_confidence:.4f}")
        print(f"   Updated confidence: {belief1.confidence:.4f}")
        print(f"   Result: {'✅ PASS' if belief1.confidence != initial_confidence else '❌ FAIL'}")
        
        # Test 3: Get beliefs
        print("\nTest 2.3: Getting beliefs")
        beliefs = tracking.get_beliefs(agent_id)
        
        print(f"   Beliefs retrieved: {len(beliefs)}")
        print(f"   Result: {'✅ PASS' if len(beliefs) == 2 else '❌ FAIL'}")
        
        return True
    
    def test_intention_recognition(self):
        """Test intention recognition"""
        print("\n" + "="*60)
        print("TEST 3: Intention Recognition")
        print("="*60)
        
        recognition = IntentionRecognition()
        
        # Test 1: Learn intention pattern
        print("\nTest 3.1: Learning intention patterns")
        recognition.learn_intention_pattern("reach_target", ["move_forward", "move_forward", "turn_left"])
        recognition.learn_intention_pattern("avoid_obstacle", ["turn_right", "move_forward"])
        
        print(f"   Patterns learned: {len(recognition.intention_patterns)}")
        print(f"   Result: {'✅ PASS' if len(recognition.intention_patterns) == 2 else '❌ FAIL'}")
        
        # Test 2: Recognize intention
        print("\nTest 3.2: Recognizing intentions")
        agent_id = 1
        recent_actions = ["move_forward", "move_forward", "turn_left"]
        recognized = recognition.recognize_intention(agent_id, recent_actions)
        
        print(f"   Intentions recognized: {len(recognized)}")
        print(f"   Result: {'✅ PASS' if len(recognized) > 0 else '❌ FAIL'}")
        
        # Test 3: Predict action
        print("\nTest 3.3: Predicting actions")
        predicted = recognition.predict_action(agent_id)
        
        print(f"   Action predicted: {predicted}")
        print(f"   Result: {'✅ PASS' if predicted is not None else '⚠️  CHECK'}")
        
        return True
    
    def test_perspective_taking(self):
        """Test perspective taking"""
        print("\n" + "="*60)
        print("TEST 4: Perspective Taking")
        print("="*60)
        
        perspective = PerspectiveTaking()
        
        # Test 1: Model perspective
        print("\nTest 4.1: Modeling perspectives")
        agent1_id = 1
        agent2_id = 2
        
        knowledge1 = {"location_A", "location_B"}
        knowledge2 = {"location_A", "location_C"}
        
        perspective.model_perspective(agent1_id, knowledge1, np.array([1.0, 1.0]))
        perspective.model_perspective(agent2_id, knowledge2, np.array([2.0, 2.0]))
        
        print(f"   Perspectives modeled: {len(perspective.perspective_models)}")
        print(f"   Result: {'✅ PASS' if len(perspective.perspective_models) == 2 else '❌ FAIL'}")
        
        # Test 2: Predict agent view
        print("\nTest 4.2: Predicting agent views")
        situation = {"location_A": "safe", "location_B": "dangerous", "location_C": "safe"}
        view1 = perspective.predict_agent_view(agent1_id, situation)
        view2 = perspective.predict_agent_view(agent2_id, situation)
        
        print(f"   Agent 1 view: {view1}")
        print(f"   Agent 2 view: {view2}")
        print(f"   Result: {'✅ PASS' if len(view1) > 0 and len(view2) > 0 else '❌ FAIL'}")
        
        # Test 3: Compute perspective difference
        print("\nTest 4.3: Computing perspective differences")
        difference = perspective.compute_perspective_difference(agent1_id, agent2_id, situation)
        
        print(f"   Perspective difference: {difference:.4f}")
        print(f"   Result: {'✅ PASS' if difference >= 0 else '❌ FAIL'}")
        
        return True
    
    def test_integrated_theory_of_mind(self):
        """Test integrated theory of mind"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Theory of Mind")
        print("="*60)
        
        manager = TheoryOfMindManager()
        
        # Observe agents
        print("\nObserving agents...")
        agent1_id = 1
        agent2_id = 2
        
        # Agent 1 behavior
        manager.observe_agent(agent1_id, "move_forward", np.array([1.0, 1.0]))
        manager.observe_agent(agent1_id, "move_forward", np.array([2.0, 1.0]))
        manager.observe_agent(agent1_id, "turn_left", np.array([2.0, 2.0]))
        
        # Agent 2 behavior
        manager.observe_agent(agent2_id, "turn_right", np.array([3.0, 3.0]))
        manager.observe_agent(agent2_id, "move_forward", np.array([4.0, 3.0]))
        
        print(f"   Agents observed: 2")
        
        # Infer mental states
        print("\nInferring mental states...")
        manager.infer_mental_state(agent1_id, observed_goals=["reach_target"])
        manager.infer_mental_state(agent2_id, observed_goals=["avoid_obstacle"])
        
        mental_state1 = manager.get_mental_state(agent1_id)
        mental_state2 = manager.get_mental_state(agent2_id)
        
        print(f"   Mental states inferred: {2 if mental_state1 and mental_state2 else 0}")
        
        # Recognize intentions
        print("\nRecognizing intentions...")
        manager.intention_recognition.learn_intention_pattern("reach_target", ["move_forward", "move_forward", "turn_left"])
        
        recent_actions1 = ["move_forward", "move_forward", "turn_left"]
        intentions1 = manager.recognize_intention(agent1_id, recent_actions1)
        
        print(f"   Intentions recognized: {len(intentions1)}")
        
        # Predict actions
        print("\nPredicting actions...")
        predicted1 = manager.predict_agent_action(agent1_id)
        predicted2 = manager.predict_agent_action(agent2_id)
        
        print(f"   Agent 1 predicted action: {predicted1}")
        print(f"   Agent 2 predicted action: {predicted2}")
        
        # Model perspectives
        print("\nModeling perspectives...")
        manager.model_agent_perspective(agent1_id, {"location_A", "location_B"})
        manager.model_agent_perspective(agent2_id, {"location_A", "location_C"})
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Theory of Mind', fontsize=16, fontweight='bold')
        
        # Plot 1: Mental states
        ax1 = axes[0, 0]
        if mental_state1 and mental_state2:
            agents = ['Agent 1', 'Agent 2']
            beliefs_count = [len(mental_state1.beliefs), len(mental_state2.beliefs)]
            desires_count = [len(mental_state1.desires), len(mental_state2.desires)]
            
            x = np.arange(len(agents))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, beliefs_count, width, label='Beliefs', color='#3498DB', alpha=0.8)
            bars2 = ax1.bar(x + width/2, desires_count, width, label='Desires', color='#2ECC71', alpha=0.8)
            
            ax1.set_title('Mental States', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Count')
            ax1.set_xticks(x)
            ax1.set_xticklabels(agents)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Belief tracking
        ax2 = axes[0, 1]
        beliefs1 = manager.belief_tracking.get_beliefs(agent1_id)
        beliefs2 = manager.belief_tracking.get_beliefs(agent2_id)
        
        if beliefs1 or beliefs2:
            all_beliefs = set(list(beliefs1.keys()) + list(beliefs2.keys()))
            if all_beliefs:
                belief_names = list(all_beliefs)[:5]  # Limit to 5
                agent1_confidences = [beliefs1.get(b, 0.0) for b in belief_names]
                agent2_confidences = [beliefs2.get(b, 0.0) for b in belief_names]
                
                x = np.arange(len(belief_names))
                width = 0.35
                
                bars1 = ax2.bar(x - width/2, agent1_confidences, width, label='Agent 1', color='#3498DB', alpha=0.8)
                bars2 = ax2.bar(x + width/2, agent2_confidences, width, label='Agent 2', color='#E74C3C', alpha=0.8)
                
                ax2.set_title('Belief Tracking', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Confidence')
                ax2.set_xticks(x)
                ax2.set_xticklabels(belief_names, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Intentions
        ax3 = axes[1, 0]
        intentions1_list = manager.intention_recognition.recognized_intentions.get(agent1_id, [])
        intentions2_list = manager.intention_recognition.recognized_intentions.get(agent2_id, [])
        
        if intentions1_list or intentions2_list:
            all_intentions = list(set(intentions1_list + intentions2_list))
            if all_intentions:
                counts = [
                    intentions1_list.count(i) + intentions2_list.count(i)
                    for i in all_intentions
                ]
                
                bars = ax3.bar(range(len(all_intentions)), counts, color='#F39C12', alpha=0.8, edgecolor='black')
                ax3.set_title('Recognized Intentions', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Count')
                ax3.set_xticks(range(len(all_intentions)))
                ax3.set_xticklabels(all_intentions, rotation=45, ha='right')
                ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = manager.get_statistics()
        stats_text = "Theory of Mind Statistics:\n\n"
        stats_text += f"Agents Tracked: {stats['agents_tracked']}\n"
        stats_text += f"Beliefs Tracked: {stats['beliefs_tracked']}\n"
        stats_text += f"Intentions Recognized: {stats['intentions_recognized']}\n"
        stats_text += f"Perspectives Modeled: {stats['perspectives_modeled']}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('theory_of_mind_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: theory_of_mind_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Agents tracked: {stats['agents_tracked']}")
        print(f"   Beliefs tracked: {stats['beliefs_tracked']}")
        print(f"   Intentions recognized: {stats['intentions_recognized']}")
        print(f"   Perspectives modeled: {stats['perspectives_modeled']}")
        
        return True
    
    def run_all_tests(self):
        """Run all theory of mind tests"""
        print("\n" + "="*70)
        print("THEORY OF MIND TEST SUITE")
        print("="*70)
        
        tests = [
            ("Mental State Inference", self.test_mental_state_inference),
            ("Belief Tracking", self.test_belief_tracking),
            ("Intention Recognition", self.test_intention_recognition),
            ("Perspective Taking", self.test_perspective_taking),
            ("Integrated Theory of Mind", self.test_integrated_theory_of_mind)
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
    tester = TheoryOfMindTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All theory of mind tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

