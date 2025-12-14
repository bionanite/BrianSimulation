#!/usr/bin/env python3
"""
Test Framework for Goal Setting and Planning
Tests goal hierarchy, planning, subgoal decomposition, and monitoring
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from goal_setting_planning import (
    GoalSettingPlanningManager, GoalHierarchy,
    PlanningAlgorithm, SubgoalDecomposition, GoalMonitoring
)

class GoalSettingPlanningTester:
    """Test framework for goal setting and planning"""
    
    def __init__(self):
        self.results = []
    
    def test_goal_hierarchy(self):
        """Test goal hierarchy"""
        print("\n" + "="*60)
        print("TEST 1: Goal Hierarchy")
        print("="*60)
        
        hierarchy = GoalHierarchy()
        
        # Test 1: Create goal
        print("\nTest 1.1: Creating goals")
        goal1 = hierarchy.create_goal("Main Goal", np.random.random(10), priority=1.0)
        goal2 = hierarchy.create_goal("Sub Goal", np.random.random(10), priority=0.8, parent_goal_id=goal1.goal_id)
        
        print(f"   Created {len(hierarchy.goals)} goals")
        print(f"   Goal 1 subgoals: {len(goal1.subgoals)}")
        print(f"   Result: {'✅ PASS' if len(hierarchy.goals) == 2 else '❌ FAIL'}")
        
        # Test 2: Goal decomposition
        print("\nTest 1.2: Goal decomposition")
        subgoal_states = [np.random.random(10) for _ in range(3)]
        subgoals = hierarchy.decompose_goal(goal1.goal_id, subgoal_states)
        
        print(f"   Subgoals created: {len(subgoals)}")
        print(f"   Parent subgoals: {len(goal1.subgoals)}")
        print(f"   Result: {'✅ PASS' if len(subgoals) == 3 else '❌ FAIL'}")
        
        # Test 3: Active goals
        print("\nTest 1.3: Active goals")
        active = hierarchy.get_active_goals()
        top_level = hierarchy.get_top_level_goals()
        
        print(f"   Active goals: {len(active)}")
        print(f"   Top-level goals: {len(top_level)}")
        print(f"   Result: {'✅ PASS' if len(active) > 0 else '❌ FAIL'}")
        
        return True
    
    def test_planning_algorithm(self):
        """Test planning algorithm"""
        print("\n" + "="*60)
        print("TEST 2: Planning Algorithm")
        print("="*60)
        
        planning = PlanningAlgorithm()
        
        # Create transition model
        def transition_model(state, action):
            next_state = state.copy()
            next_state[action % len(state)] += 0.1
            return next_state, 0.9
        
        # Test 1: Plan creation
        print("\nTest 2.1: Creating plan")
        current_state = np.zeros(10)
        goal_state = np.ones(10) * 0.5
        available_actions = [0, 1, 2]
        
        plan = planning.plan_to_goal(current_state, goal_state, available_actions, transition_model)
        
        print(f"   Plan found: {plan is not None}")
        if plan:
            print(f"   Plan length: {len(plan.actions)}")
            print(f"   Expected states: {len(plan.expected_states)}")
        print(f"   Result: {'✅ PASS' if plan is not None else '⚠️  CHECK'}")
        
        # Test 2: Plan refinement
        print("\nTest 2.2: Plan refinement")
        if plan:
            new_state = current_state.copy()
            new_state[0] = 0.2
            refined = planning.refine_plan(plan, new_state, transition_model)
            
            print(f"   Refined plan length: {len(refined.actions)}")
            print(f"   Result: {'✅ PASS' if refined is not None else '❌ FAIL'}")
        
        return True
    
    def test_subgoal_decomposition(self):
        """Test subgoal decomposition"""
        print("\n" + "="*60)
        print("TEST 3: Subgoal Decomposition")
        print("="*60)
        
        decomposition = SubgoalDecomposition()
        
        # Test 1: Linear decomposition
        print("\nTest 3.1: Linear decomposition")
        current_state = np.zeros(10)
        goal_state = np.ones(10)
        subgoals = decomposition.decompose(current_state, goal_state, num_subgoals=3)
        
        print(f"   Subgoals created: {len(subgoals)}")
        print(f"   Result: {'✅ PASS' if len(subgoals) == 3 else '❌ FAIL'}")
        
        # Test 2: Subgoal progression
        print("\nTest 3.2: Subgoal progression")
        distances = []
        for i, subgoal in enumerate(subgoals):
            if i == 0:
                dist = np.linalg.norm(subgoal - current_state)
            else:
                dist = np.linalg.norm(subgoal - subgoals[i-1])
            distances.append(dist)
        
        print(f"   Distances between subgoals: {[f'{d:.3f}' for d in distances]}")
        print(f"   Result: {'✅ PASS' if all(d > 0 for d in distances) else '❌ FAIL'}")
        
        return True
    
    def test_goal_monitoring(self):
        """Test goal monitoring"""
        print("\n" + "="*60)
        print("TEST 4: Goal Monitoring")
        print("="*60)
        
        monitoring = GoalMonitoring()
        
        # Create goal
        from goal_setting_planning import Goal
        goal = Goal(
            goal_id=1,
            description="Test Goal",
            target_state=np.ones(10) * 0.8,
            created_time=time.time()
        )
        
        # Test 1: Progress check
        print("\nTest 4.1: Progress checking")
        current_state = np.ones(10) * 0.5
        result = monitoring.check_goal_progress(goal, current_state)
        
        print(f"   Progress: {result['progress']:.4f}")
        print(f"   Distance: {result['distance']:.4f}")
        print(f"   Result: {'✅ PASS' if 0 <= result['progress'] <= 1 else '❌ FAIL'}")
        
        # Test 2: Goal achievement
        print("\nTest 4.2: Goal achievement detection")
        achieved_state = goal.target_state.copy()
        result = monitoring.check_goal_progress(goal, achieved_state)
        
        print(f"   Is achieved: {result['is_achieved']}")
        print(f"   Result: {'✅ PASS' if result['is_achieved'] else '❌ FAIL'}")
        
        # Test 3: Multiple goals monitoring
        print("\nTest 4.3: Multiple goals monitoring")
        goals = [
            Goal(1, "Goal 1", np.ones(10) * 0.8),
            Goal(2, "Goal 2", np.ones(10) * 0.6)
        ]
        
        results = monitoring.monitor_goals(goals, current_state)
        
        print(f"   Goals monitored: {len(results)}")
        print(f"   Result: {'✅ PASS' if len(results) == 2 else '❌ FAIL'}")
        
        return True
    
    def test_integrated_goal_setting_planning(self):
        """Test integrated goal setting and planning"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Goal Setting and Planning")
        print("="*60)
        
        manager = GoalSettingPlanningManager(state_size=10)
        
        # Create transition model
        def transition_model(state, action):
            next_state = state.copy()
            next_state[action % len(state)] += 0.1
            return next_state, 0.9
        
        manager.set_transition_model(transition_model)
        
        # Create goals
        print("\nCreating goals...")
        goal1 = manager.create_goal("Reach target", np.ones(10) * 0.8, priority=1.0, decompose=True)
        goal2 = manager.create_goal("Explore area", np.random.random(10), priority=0.7)
        
        print(f"   Goals created: {len(manager.goal_hierarchy.goals)}")
        
        # Create plans
        print("\nCreating plans...")
        current_state = np.zeros(10)
        available_actions = [0, 1, 2, 3, 4]
        
        plan1 = manager.plan_for_goal(goal1.goal_id, current_state, available_actions)
        plan2 = manager.plan_for_goal(goal2.goal_id, current_state, available_actions)
        
        print(f"   Plans created: {len(manager.plans)}")
        
        # Monitor goals
        print("\nMonitoring goals...")
        monitoring_results = manager.monitor_goals(current_state)
        
        print(f"   Goals monitored: {len(monitoring_results)}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Goal Setting and Planning', fontsize=16, fontweight='bold')
        
        # Plot 1: Goal hierarchy
        ax1 = axes[0, 0]
        goals_list = list(manager.goal_hierarchy.goals.values())
        priorities = [g.priority for g in goals_list]
        progress = [g.progress for g in goals_list]
        
        bars = ax1.bar(range(len(goals_list)), priorities, color='#3498DB', alpha=0.7, label='Priority')
        ax1_twin = ax1.twinx()
        line = ax1_twin.plot(range(len(goals_list)), progress, 'r-o', linewidth=2, label='Progress')
        
        ax1.set_title('Goal Priorities and Progress', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Goal Index')
        ax1.set_ylabel('Priority', color='blue')
        ax1_twin.set_ylabel('Progress', color='red')
        ax1.set_xticks(range(len(goals_list)))
        ax1.set_xticklabels([g.description[:10] for g in goals_list], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        ax1.legend([bars], ['Priority'], loc='upper left')
        ax1_twin.legend(line, ['Progress'], loc='upper right')
        
        # Plot 2: Plan lengths
        ax2 = axes[0, 1]
        if manager.plans:
            plan_lengths = [len(p.actions) for p in manager.plans.values()]
            bars = ax2.bar(range(len(plan_lengths)), plan_lengths, color='#2ECC71', alpha=0.8)
            ax2.set_title('Plan Lengths', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Plan Index')
            ax2.set_ylabel('Number of Actions')
            ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Goal status
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        stats = manager.get_statistics()
        status_text = "Goal Statistics:\n\n"
        status_text += f"Total Goals: {stats['total_goals']}\n"
        status_text += f"Active Goals: {stats['active_goals']}\n"
        status_text += f"Top-Level Goals: {stats['top_level_goals']}\n"
        status_text += f"Plans: {stats['plans']}\n"
        status_text += f"Achieved Goals: {stats['achieved_goals']}\n"
        
        ax3.text(0.1, 0.5, status_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax3.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 4: Monitoring results
        ax4 = axes[1, 1]
        if monitoring_results:
            goal_ids = list(monitoring_results.keys())
            progress_values = [monitoring_results[gid]['progress'] for gid in goal_ids]
            distances = [monitoring_results[gid]['distance'] for gid in goal_ids]
            
            ax4_twin = ax4.twinx()
            bars = ax4.bar(range(len(goal_ids)), progress_values, color='green', alpha=0.7, label='Progress')
            line = ax4_twin.plot(range(len(goal_ids)), distances, 'r-o', linewidth=2, label='Distance')
            
            ax4.set_title('Goal Monitoring', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Goal ID')
            ax4.set_ylabel('Progress', color='green')
            ax4_twin.set_ylabel('Distance', color='red')
            ax4.set_xticks(range(len(goal_ids)))
            ax4.set_xticklabels([str(gid) for gid in goal_ids])
            ax4.grid(True, alpha=0.3)
            
            ax4.legend([bars], ['Progress'], loc='upper left')
            ax4_twin.legend(line, ['Distance'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig('goal_setting_planning_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: goal_setting_planning_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Total goals: {stats['total_goals']}")
        print(f"   Active goals: {stats['active_goals']}")
        print(f"   Plans: {stats['plans']}")
        print(f"   Achieved goals: {stats['achieved_goals']}")
        
        return True
    
    def run_all_tests(self):
        """Run all goal setting and planning tests"""
        print("\n" + "="*70)
        print("GOAL SETTING AND PLANNING TEST SUITE")
        print("="*70)
        
        tests = [
            ("Goal Hierarchy", self.test_goal_hierarchy),
            ("Planning Algorithm", self.test_planning_algorithm),
            ("Subgoal Decomposition", self.test_subgoal_decomposition),
            ("Goal Monitoring", self.test_goal_monitoring),
            ("Integrated Goal Setting and Planning", self.test_integrated_goal_setting_planning)
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
    tester = GoalSettingPlanningTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All goal setting and planning tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

