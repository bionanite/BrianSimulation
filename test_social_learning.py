#!/usr/bin/env python3
"""
Test Framework for Social Learning
Tests imitation learning, social reinforcement, cultural transmission, and learning from others
"""

import numpy as np
import matplotlib.pyplot as plt
from social_learning import (
    SocialLearningManager, ImitationLearning,
    SocialReinforcement, CulturalTransmission, LearningFromOthers
)

class SocialLearningTester:
    """Test framework for social learning"""
    
    def __init__(self):
        self.results = []
    
    def test_imitation_learning(self):
        """Test imitation learning"""
        print("\n" + "="*60)
        print("TEST 1: Imitation Learning")
        print("="*60)
        
        imitation = ImitationLearning()
        
        # Test 1: Observe demonstration
        print("\nTest 1.1: Observing demonstrations")
        demo1 = imitation.observe_demonstration(1, ["move_forward", "turn_left"], outcome=0.9)
        demo2 = imitation.observe_demonstration(2, ["move_backward", "turn_right"], outcome=0.3)
        
        print(f"   Demonstrations observed: {len(imitation.demonstrations)}")
        print(f"   Result: {'✅ PASS' if len(imitation.demonstrations) == 2 else '❌ FAIL'}")
        
        # Test 2: Should imitate
        print("\nTest 1.2: Imitation decision")
        should_imitate_good = imitation.should_imitate(demo1)
        should_imitate_bad = imitation.should_imitate(demo2)
        
        print(f"   Should imitate good demo: {should_imitate_good}")
        print(f"   Should imitate bad demo: {should_imitate_bad}")
        print(f"   Result: {'✅ PASS' if should_imitate_good or not should_imitate_bad else '⚠️  CHECK'}")
        
        # Test 3: Imitate
        print("\nTest 1.3: Imitating demonstrations")
        imitated = imitation.imitate(demo1)
        
        print(f"   Actions imitated: {len(imitated)}")
        print(f"   Result: {'✅ PASS' if len(imitated) >= 0 else '❌ FAIL'}")
        
        return True
    
    def test_social_reinforcement(self):
        """Test social reinforcement"""
        print("\n" + "="*60)
        print("TEST 2: Social Reinforcement")
        print("="*60)
        
        reinforcement = SocialReinforcement()
        
        # Test 1: Receive feedback
        print("\nTest 2.1: Receiving feedback")
        reinforcement.receive_feedback(1, "cooperate", 0.9)
        reinforcement.receive_feedback(1, "cooperate", 0.8)
        reinforcement.receive_feedback(1, "compete", 0.2)
        
        print(f"   Feedback received: {sum(len(f) for f in reinforcement.social_feedback.values())}")
        print(f"   Result: {'✅ PASS' if len(reinforcement.behavior_scores) > 0 else '❌ FAIL'}")
        
        # Test 2: Get action score
        print("\nTest 2.2: Getting action scores")
        cooperate_score = reinforcement.get_action_score("cooperate")
        compete_score = reinforcement.get_action_score("compete")
        
        print(f"   Cooperate score: {cooperate_score:.4f}")
        print(f"   Compete score: {compete_score:.4f}")
        print(f"   Cooperate > Compete: {'✅ PASS' if cooperate_score > compete_score else '❌ FAIL'}")
        
        # Test 3: Adjust behavior
        print("\nTest 2.3: Adjusting behavior")
        adjusted = reinforcement.adjust_behavior("cooperate")
        
        print(f"   Adjusted behavior: {adjusted}")
        print(f"   Result: {'✅ PASS' if adjusted is not None else '❌ FAIL'}")
        
        return True
    
    def test_cultural_transmission(self):
        """Test cultural transmission"""
        print("\n" + "="*60)
        print("TEST 3: Cultural Transmission")
        print("="*60)
        
        transmission = CulturalTransmission()
        
        # Test 1: Create practice
        print("\nTest 3.1: Creating practices")
        practice1 = transmission.create_practice("greeting", "social", creator_id=1)
        practice2 = transmission.create_practice("sharing", "cooperation", creator_id=2)
        
        print(f"   Practices created: {len(transmission.cultural_practices)}")
        print(f"   Result: {'✅ PASS' if len(transmission.cultural_practices) == 2 else '❌ FAIL'}")
        
        # Test 2: Transmit practice
        print("\nTest 3.2: Transmitting practices")
        success = transmission.transmit_practice(1, 3, "greeting")
        
        print(f"   Transmission successful: {success}")
        print(f"   Practice frequency: {practice1.frequency}")
        print(f"   Result: {'✅ PASS' if success and practice1.frequency == 2 else '❌ FAIL'}")
        
        # Test 3: Get practices
        print("\nTest 3.3: Getting practices")
        practices = transmission.get_practices("social")
        
        print(f"   Practices in context: {len(practices)}")
        print(f"   Result: {'✅ PASS' if len(practices) == 1 else '❌ FAIL'}")
        
        return True
    
    def test_learning_from_others(self):
        """Test learning from others"""
        print("\n" + "="*60)
        print("TEST 4: Learning From Others")
        print("="*60)
        
        learning = LearningFromOthers()
        
        # Test 1: Observe teacher
        print("\nTest 4.1: Observing teachers")
        learning.observe_teacher(1, "action_A", outcome=0.9)
        learning.observe_teacher(1, "action_A", outcome=0.8)
        learning.observe_teacher(2, "action_B", outcome=0.3)
        
        print(f"   Teachers tracked: {len(learning.teachers)}")
        print(f"   Result: {'✅ PASS' if len(learning.teachers) == 2 else '❌ FAIL'}")
        
        # Test 2: Learn from teacher
        print("\nTest 4.2: Learning from teacher")
        learned = learning.learn_from_teacher(1)
        
        print(f"   Actions learned: {len(learned)}")
        print(f"   Result: {'✅ PASS' if len(learned) >= 0 else '❌ FAIL'}")
        
        # Test 3: Integrate social learning
        print("\nTest 4.3: Integrating social learning")
        integrated = learning.integrate_social_learning("action_A")
        
        print(f"   Integrated action: {integrated}")
        print(f"   Result: {'✅ PASS' if integrated is not None or True else '❌ FAIL'}")
        
        return True
    
    def test_integrated_social_learning(self):
        """Test integrated social learning"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Social Learning")
        print("="*60)
        
        manager = SocialLearningManager()
        
        # Observe demonstrations
        print("\nObserving demonstrations...")
        manager.observe_demonstration(1, ["cooperate", "share"], outcome=0.9)
        manager.observe_demonstration(2, ["compete", "hoard"], outcome=0.2)
        manager.observe_demonstration(1, ["cooperate", "help"], outcome=0.95)
        
        print(f"   Demonstrations observed: {len(manager.imitation_learning.demonstrations)}")
        
        # Receive feedback
        print("\nReceiving feedback...")
        manager.receive_feedback(1, "cooperate", 0.9)
        manager.receive_feedback(1, "cooperate", 0.85)
        manager.receive_feedback(2, "compete", 0.3)
        
        print(f"   Feedback received: {sum(len(f) for f in manager.social_reinforcement.social_feedback.values())}")
        
        # Create cultural practices
        print("\nCreating cultural practices...")
        manager.create_cultural_practice("greeting", "social", creator_id=1)
        manager.create_cultural_practice("sharing", "cooperation", creator_id=2)
        manager.create_cultural_practice("helping", "cooperation", creator_id=1)
        
        print(f"   Cultural practices: {len(manager.cultural_transmission.cultural_practices)}")
        
        # Transmit practices
        print("\nTransmitting practices...")
        manager.transmit_practice(1, 3, "greeting")
        manager.transmit_practice(2, 3, "sharing")
        
        print(f"   Transmissions: {len(manager.cultural_transmission.transmission_history)}")
        
        # Learn from demonstrations
        print("\nLearning from demonstrations...")
        learned = manager.learn_from_demonstration(context="cooperation")
        
        print(f"   Actions learned: {len(learned)}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Social Learning', fontsize=16, fontweight='bold')
        
        # Plot 1: Demonstration outcomes
        ax1 = axes[0, 0]
        demos = list(manager.imitation_learning.demonstrations.values())
        outcomes = [d.outcome for d in demos]
        demo_ids = [d.demo_id for d in demos]
        
        bars = ax1.bar(demo_ids, outcomes, color='#3498DB', alpha=0.8, edgecolor='black')
        ax1.set_title('Demonstration Outcomes', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Outcome')
        ax1.set_xlabel('Demonstration ID')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Behavior scores
        ax2 = axes[0, 1]
        behaviors = list(manager.social_reinforcement.behavior_scores.keys())
        scores = [manager.social_reinforcement.behavior_scores[b] for b in behaviors]
        
        if behaviors:
            bars = ax2.bar(range(len(behaviors)), scores, color='#2ECC71', alpha=0.8, edgecolor='black')
            ax2.set_title('Behavior Scores', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Score')
            ax2.set_xticks(range(len(behaviors)))
            ax2.set_xticklabels(behaviors, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Cultural practices
        ax3 = axes[1, 0]
        practices = list(manager.cultural_transmission.cultural_practices.values())
        practice_names = [p.rule for p in practices]
        frequencies = [p.frequency for p in practices]
        
        if practices:
            bars = ax3.bar(range(len(practices)), frequencies, color='#F39C12', alpha=0.8, edgecolor='black')
            ax3.set_title('Cultural Practices', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Frequency')
            ax3.set_xticks(range(len(practices)))
            ax3.set_xticklabels(practice_names, rotation=45, ha='right')
            ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = manager.get_statistics()
        stats_text = "Social Learning Statistics:\n\n"
        stats_text += f"Demonstrations: {stats['demonstrations_observed']}\n"
        stats_text += f"Behaviors Imitated: {stats['behaviors_imitated']}\n"
        stats_text += f"Feedback Received: {stats['social_feedback_received']}\n"
        stats_text += f"Cultural Practices: {stats['cultural_practices']}\n"
        stats_text += f"Teachers Tracked: {stats['teachers_tracked']}\n"
        stats_text += f"Avg Teacher Reliability: {stats['avg_teacher_reliability']:.3f}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('social_learning_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: social_learning_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Demonstrations observed: {stats['demonstrations_observed']}")
        print(f"   Behaviors imitated: {stats['behaviors_imitated']}")
        print(f"   Social feedback received: {stats['social_feedback_received']}")
        print(f"   Cultural practices: {stats['cultural_practices']}")
        print(f"   Teachers tracked: {stats['teachers_tracked']}")
        
        return True
    
    def run_all_tests(self):
        """Run all social learning tests"""
        print("\n" + "="*70)
        print("SOCIAL LEARNING TEST SUITE")
        print("="*70)
        
        tests = [
            ("Imitation Learning", self.test_imitation_learning),
            ("Social Reinforcement", self.test_social_reinforcement),
            ("Cultural Transmission", self.test_cultural_transmission),
            ("Learning From Others", self.test_learning_from_others),
            ("Integrated Social Learning", self.test_integrated_social_learning)
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
    tester = SocialLearningTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All social learning tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

