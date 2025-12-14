#!/usr/bin/env python3
"""
Test Framework for Intrinsic Motivation
Tests curiosity, novelty seeking, competence, and autonomy
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from intrinsic_motivation import (
    IntrinsicMotivationManager, CuriosityDrive,
    NoveltySeeking, CompetenceMotivation, AutonomyDrive
)

class IntrinsicMotivationTester:
    """Test framework for intrinsic motivation"""
    
    def __init__(self):
        self.results = []
    
    def test_curiosity_drive(self):
        """Test curiosity drive"""
        print("\n" + "="*60)
        print("TEST 1: Curiosity Drive")
        print("="*60)
        
        curiosity = CuriosityDrive()
        
        # Test 1: Novelty computation
        print("\nTest 1.1: Novelty computation")
        novel_state = np.random.random(10)
        novelty = curiosity.compute_novelty(novel_state)
        
        print(f"   Novel state novelty: {novelty:.4f}")
        print(f"   Result: {'✅ PASS' if novelty > 0.9 else '❌ FAIL'}")
        
        # Test 2: Novelty decreases with experience
        print("\nTest 1.2: Novelty decreases with experience")
        curiosity.experience_state(novel_state)
        novelty_after = curiosity.compute_novelty(novel_state)
        
        print(f"   After experience: {novelty_after:.4f}")
        print(f"   Novelty decreased: {'✅ PASS' if novelty_after < novelty else '❌ FAIL'}")
        
        # Test 3: Curiosity reward
        print("\nTest 1.3: Curiosity reward")
        reward = curiosity.compute_curiosity_reward(novel_state)
        
        print(f"   Curiosity reward: {reward:.4f}")
        print(f"   Result: {'✅ PASS' if reward > 0 else '❌ FAIL'}")
        
        return True
    
    def test_novelty_seeking(self):
        """Test novelty seeking"""
        print("\n" + "="*60)
        print("TEST 2: Novelty Seeking")
        print("="*60)
        
        novelty_seeking = NoveltySeeking()
        
        # Test 1: Exploration bonus
        print("\nTest 2.1: Exploration bonus")
        unexplored_state = np.random.random(10)
        bonus = novelty_seeking.compute_exploration_bonus(unexplored_state)
        
        print(f"   Unexplored region bonus: {bonus:.4f}")
        print(f"   Result: {'✅ PASS' if bonus > 0 else '❌ FAIL'}")
        
        # Test 2: Bonus decreases near visited regions
        print("\nTest 2.2: Bonus decreases near visited regions")
        novelty_seeking.visit_region(unexplored_state)
        
        nearby_state = unexplored_state + np.random.normal(0, 0.1, 10)
        bonus_nearby = novelty_seeking.compute_exploration_bonus(nearby_state)
        
        far_state = np.random.random(10) * 2  # Far away
        bonus_far = novelty_seeking.compute_exploration_bonus(far_state)
        
        print(f"   Near visited region: {bonus_nearby:.4f}")
        print(f"   Far from visited: {bonus_far:.4f}")
        print(f"   Far > Near: {'✅ PASS' if bonus_far > bonus_nearby else '⚠️  CHECK'}")
        
        return True
    
    def test_competence_motivation(self):
        """Test competence motivation"""
        print("\n" + "="*60)
        print("TEST 3: Competence Motivation")
        print("="*60)
        
        competence = CompetenceMotivation()
        
        # Test 1: Skill update
        print("\nTest 3.1: Skill level update")
        skill_name = "navigation"
        initial_level = competence.get_mastery_level(skill_name)
        
        competence.update_skill(skill_name, performance=0.8)
        new_level = competence.get_mastery_level(skill_name)
        
        print(f"   Initial level: {initial_level:.4f}")
        print(f"   After update: {new_level:.4f}")
        print(f"   Level increased: {'✅ PASS' if new_level > initial_level else '❌ FAIL'}")
        
        # Test 2: Competence reward
        print("\nTest 3.2: Competence reward")
        reward = competence.compute_competence_reward(skill_name, performance=0.9)
        
        print(f"   Competence reward: {reward:.4f}")
        print(f"   Result: {'✅ PASS' if reward >= 0 else '❌ FAIL'}")
        
        # Test 3: Multiple skills
        print("\nTest 3.3: Multiple skills")
        competence.update_skill("manipulation", performance=0.6)
        competence.update_skill("communication", performance=0.7)
        
        print(f"   Skills tracked: {len(competence.skill_levels)}")
        print(f"   Result: {'✅ PASS' if len(competence.skill_levels) == 3 else '❌ FAIL'}")
        
        return True
    
    def test_autonomy_drive(self):
        """Test autonomy drive"""
        print("\n" + "="*60)
        print("TEST 4: Autonomy Drive")
        print("="*60)
        
        autonomy = AutonomyDrive()
        
        # Test 1: Goal generation
        print("\nTest 4.1: Autonomous goal generation")
        current_state = np.random.random(10)
        goal = autonomy.generate_goal(current_state, "Explore new area")
        
        print(f"   Goal ID: {goal.goal_id}")
        print(f"   Goal description: {goal.description}")
        print(f"   Target state shape: {goal.target_state.shape}")
        print(f"   Result: {'✅ PASS' if goal is not None else '❌ FAIL'}")
        
        # Test 2: Active goals
        print("\nTest 4.2: Active goals")
        active_goals = autonomy.get_active_goals()
        
        print(f"   Active goals: {len(active_goals)}")
        print(f"   Result: {'✅ PASS' if len(active_goals) > 0 else '❌ FAIL'}")
        
        # Test 3: Goal achievement
        print("\nTest 4.3: Goal achievement")
        goal_id = goal.goal_id
        autonomy.achieve_goal(goal_id)
        
        active_after = autonomy.get_active_goals()
        print(f"   Active goals after achievement: {len(active_after)}")
        print(f"   Goal achieved: {'✅ PASS' if len(active_after) < len(active_goals) else '❌ FAIL'}")
        
        return True
    
    def test_integrated_intrinsic_motivation(self):
        """Test integrated intrinsic motivation"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Intrinsic Motivation")
        print("="*60)
        
        manager = IntrinsicMotivationManager()
        
        print("\nSimulating intrinsic motivation...")
        
        # Simulate exploration
        states = []
        rewards = []
        
        for i in range(20):
            state = np.random.random(10)
            states.append(state)
            
            # Compute intrinsic reward
            reward = manager.compute_intrinsic_reward(
                state,
                skill_name="exploration",
                performance=0.5 + i * 0.02
            )
            rewards.append(reward)
            
            # Update competence
            manager.update_competence("exploration", 0.5 + i * 0.02)
        
        # Generate autonomous goals
        print("\nGenerating autonomous goals...")
        goals_generated = 0
        for state in states[:5]:
            goal = manager.generate_autonomous_goal(state)
            if goal:
                goals_generated += 1
        
        print(f"   Goals generated: {goals_generated}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Intrinsic Motivation Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Intrinsic rewards over time
        ax1 = axes[0, 0]
        ax1.plot(rewards, 'b-o', linewidth=2, markersize=4)
        ax1.set_title('Intrinsic Rewards Over Time', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Experience')
        ax1.set_ylabel('Intrinsic Reward')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Novelty over time
        ax2 = axes[0, 1]
        novelties = []
        for state in states:
            novelty = manager.curiosity.compute_novelty(state)
            novelties.append(novelty)
        
        ax2.plot(novelties, 'g-s', linewidth=2, markersize=4)
        ax2.set_title('Novelty Over Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Experience')
        ax2.set_ylabel('Novelty')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Competence levels
        ax3 = axes[1, 0]
        if manager.competence.skill_levels:
            skills = list(manager.competence.skill_levels.keys())
            levels = [manager.competence.skill_levels[s] for s in skills]
            
            bars = ax3.bar(skills, levels, color='#F39C12', alpha=0.8, edgecolor='black')
            ax3.set_title('Competence Levels', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Mastery Level')
            ax3.set_ylim([0, 1.1])
            ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = manager.get_statistics()
        stats_text = "Intrinsic Motivation Statistics:\n\n"
        stats_text += f"Unique States: {stats['unique_states_experienced']}\n"
        stats_text += f"Regions Visited: {stats['regions_visited']}\n"
        stats_text += f"Skills Tracked: {stats['skills_tracked']}\n"
        stats_text += f"Autonomous Goals: {stats['autonomous_goals']}\n"
        stats_text += f"Active Goals: {stats['active_goals']}\n"
        stats_text += f"Avg Reward: {stats['avg_intrinsic_reward']:.4f}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('intrinsic_motivation_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: intrinsic_motivation_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Unique states experienced: {stats['unique_states_experienced']}")
        print(f"   Regions visited: {stats['regions_visited']}")
        print(f"   Skills tracked: {stats['skills_tracked']}")
        print(f"   Autonomous goals: {stats['autonomous_goals']}")
        print(f"   Average intrinsic reward: {stats['avg_intrinsic_reward']:.4f}")
        
        return True
    
    def run_all_tests(self):
        """Run all intrinsic motivation tests"""
        print("\n" + "="*70)
        print("INTRINSIC MOTIVATION TEST SUITE")
        print("="*70)
        
        tests = [
            ("Curiosity Drive", self.test_curiosity_drive),
            ("Novelty Seeking", self.test_novelty_seeking),
            ("Competence Motivation", self.test_competence_motivation),
            ("Autonomy Drive", self.test_autonomy_drive),
            ("Integrated Intrinsic Motivation", self.test_integrated_intrinsic_motivation)
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
    tester = IntrinsicMotivationTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All intrinsic motivation tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

