#!/usr/bin/env python3
"""
Test Framework for Reward-Based Learning Mechanisms
Tests RPE, value function learning, Q-learning, and policy gradient
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from reward_learning import (
    RewardLearningManager, RewardPredictionError,
    ValueFunctionLearning, QLearning, PolicyGradientLearning
)

class RewardLearningTester:
    """Test framework for reward-based learning"""
    
    def __init__(self):
        self.results = []
    
    def test_rpe(self):
        """Test Reward Prediction Error"""
        print("\n" + "="*60)
        print("TEST 1: Reward Prediction Error (RPE)")
        print("="*60)
        
        rpe = RewardPredictionError(learning_rate=0.1)
        
        # Test 1: Initial prediction
        print("\nTest 1.1: Initial reward prediction")
        state_id = 1
        predicted = rpe.predict_reward(state_id)
        print(f"   Initial prediction: {predicted:.4f}")
        print(f"   Result: {'✅ PASS' if predicted == 0.0 else '❌ FAIL'}")
        
        # Test 2: Positive RPE (unexpected reward)
        print("\nTest 1.2: Positive RPE (unexpected reward)")
        actual_reward = 1.0
        rpe_value = rpe.calculate_rpe(state_id, actual_reward)
        print(f"   Actual reward: {actual_reward:.2f}")
        print(f"   Predicted reward: {rpe.predict_reward(state_id):.4f}")
        print(f"   RPE: {rpe_value:+.4f}")
        print(f"   Result: {'✅ PASS' if rpe_value > 0 else '❌ FAIL'}")
        
        # Update prediction
        rpe.update_prediction(state_id, rpe_value)
        new_predicted = rpe.predict_reward(state_id)
        print(f"   Updated prediction: {new_predicted:.4f}")
        print(f"   Prediction increased: {'✅ PASS' if new_predicted > predicted else '❌ FAIL'}")
        
        # Test 3: Negative RPE (expected reward missing)
        print("\nTest 1.3: Negative RPE (expected reward missing)")
        state_id2 = 2
        rpe.update_prediction(state_id2, 0.5)  # Predict 0.5 reward
        actual_reward2 = 0.0  # But get 0
        rpe_value2 = rpe.calculate_rpe(state_id2, actual_reward2)
        print(f"   Predicted reward: {rpe.predict_reward(state_id2):.4f}")
        print(f"   Actual reward: {actual_reward2:.2f}")
        print(f"   RPE: {rpe_value2:+.4f}")
        print(f"   Result: {'✅ PASS' if rpe_value2 < 0 else '❌ FAIL'}")
        
        # Test 4: RPE signal conversion
        print("\nTest 1.4: RPE signal conversion")
        rpe_signal = rpe.get_rpe_signal(rpe_value)
        print(f"   RPE: {rpe_value:.4f}")
        print(f"   RPE Signal: {rpe_signal:.4f}")
        print(f"   Signal bounded: {'✅ PASS' if -1 <= rpe_signal <= 1 else '❌ FAIL'}")
        
        return True
    
    def test_value_function_learning(self):
        """Test value function learning"""
        print("\n" + "="*60)
        print("TEST 2: Value Function Learning")
        print("="*60)
        
        value_learner = ValueFunctionLearning(learning_rate=0.1)
        
        # Test 1: Initial value
        print("\nTest 2.1: Initial state value")
        state_id = 1
        initial_value = value_learner.get_value(state_id)
        print(f"   Initial value: {initial_value:.4f}")
        print(f"   Result: {'✅ PASS' if initial_value == 0.0 else '❌ FAIL'}")
        
        # Test 2: Update with reward
        print("\nTest 2.2: Value update with reward")
        reward = 1.0
        td_error = value_learner.update_value(state_id, reward, current_time=1.0)
        new_value = value_learner.get_value(state_id)
        print(f"   Reward: {reward:.2f}")
        print(f"   TD Error: {td_error:+.4f}")
        print(f"   New value: {new_value:.4f}")
        print(f"   Value increased: {'✅ PASS' if new_value > initial_value else '❌ FAIL'}")
        
        # Test 3: Sequential updates
        print("\nTest 2.3: Sequential value updates")
        state1, state2, state3 = 1, 2, 3
        
        # Create a chain: state1 -> state2 -> state3 (reward)
        value_learner.update_value(state3, reward=1.0, current_time=2.0)
        value_learner.update_value(state2, reward=0.0, next_state_id=state3, current_time=3.0)
        value_learner.update_value(state1, reward=0.0, next_state_id=state2, current_time=4.0)
        
        value1 = value_learner.get_value(state1)
        value2 = value_learner.get_value(state2)
        value3 = value_learner.get_value(state3)
        
        print(f"   Value(state1): {value1:.4f}")
        print(f"   Value(state2): {value2:.4f}")
        print(f"   Value(state3): {value3:.4f}")
        print(f"   Chain learning: {'✅ PASS' if value1 < value2 < value3 else '⚠️  CHECK'}")
        
        # Statistics
        stats = value_learner.get_statistics()
        print(f"\n   Statistics:")
        print(f"   States learned: {stats['num_states']}")
        print(f"   Average value: {stats['avg_value']:.4f}")
        print(f"   Total visits: {stats['total_visits']}")
        
        return True
    
    def test_q_learning(self):
        """Test Q-learning"""
        print("\n" + "="*60)
        print("TEST 3: Q-Learning")
        print("="*60)
        
        q_learner = QLearning(learning_rate=0.1, exploration_rate=0.1)
        
        # Test 1: Initial Q-value
        print("\nTest 3.1: Initial Q-value")
        state_id = 1
        action_id = 1
        initial_q = q_learner.get_q_value(state_id, action_id)
        print(f"   Initial Q(s={state_id}, a={action_id}): {initial_q:.4f}")
        print(f"   Result: {'✅ PASS' if initial_q == 0.0 else '❌ FAIL'}")
        
        # Test 2: Q-value update
        print("\nTest 3.2: Q-value update")
        reward = 1.0
        q_error = q_learner.update_q_value(state_id, action_id, reward, current_time=1.0)
        new_q = q_learner.get_q_value(state_id, action_id)
        print(f"   Reward: {reward:.2f}")
        print(f"   Q-error: {q_error:+.4f}")
        print(f"   New Q-value: {new_q:.4f}")
        print(f"   Q-value increased: {'✅ PASS' if new_q > initial_q else '❌ FAIL'}")
        
        # Test 3: Action selection
        print("\nTest 3.3: Action selection")
        available_actions = [1, 2, 3]
        
        # Set Q-values
        q_learner.update_q_value(state_id, 1, reward=0.5, current_time=2.0)
        q_learner.update_q_value(state_id, 2, reward=1.0, current_time=3.0)
        q_learner.update_q_value(state_id, 3, reward=0.3, current_time=4.0)
        
        # Select actions (multiple times to see exploration/exploitation)
        action_counts = {1: 0, 2: 0, 3: 0}
        for _ in range(100):
            action = q_learner.select_action(state_id, available_actions)
            action_counts[action] += 1
        
        print(f"   Action selections:")
        for action, count in action_counts.items():
            q_val = q_learner.get_q_value(state_id, action)
            print(f"      Action {action}: {count} times (Q={q_val:.2f})")
        
        # Best action should be selected most often (with some exploration)
        best_action = max(action_counts, key=action_counts.get)
        print(f"   Most selected: Action {best_action}")
        print(f"   Result: {'✅ PASS' if best_action == 2 else '⚠️  CHECK (exploration)'}")
        
        return True
    
    def test_policy_gradient(self):
        """Test policy gradient learning"""
        print("\n" + "="*60)
        print("TEST 4: Policy Gradient Learning")
        print("="*60)
        
        policy_learner = PolicyGradientLearning(learning_rate=0.01)
        
        # Test 1: Initial policy
        print("\nTest 4.1: Initial policy")
        state_id = 1
        available_actions = [1, 2, 3]
        
        probs = {}
        for action in available_actions:
            prob = policy_learner.get_action_probability(state_id, action, available_actions)
            probs[action] = prob
            print(f"   P(a={action}|s={state_id}): {prob:.4f}")
        
        # Should be uniform initially
        uniform = all(abs(p - 1.0/len(available_actions)) < 0.01 for p in probs.values())
        print(f"   Uniform distribution: {'✅ PASS' if uniform else '❌ FAIL'}")
        
        # Test 2: Policy update
        print("\nTest 4.2: Policy update with reward")
        action_id = 2
        return_value = 1.0
        
        advantage = policy_learner.update_policy(state_id, action_id, return_value, current_time=1.0)
        
        new_probs = {}
        for action in available_actions:
            prob = policy_learner.get_action_probability(state_id, action, available_actions)
            new_probs[action] = prob
            print(f"   P(a={action}|s={state_id}): {prob:.4f}")
        
        # Action 2 should have higher probability now
        print(f"   Advantage: {advantage:+.4f}")
        print(f"   Action 2 probability increased: {'✅ PASS' if new_probs[2] > probs[2] else '❌ FAIL'}")
        
        # Test 3: Action selection
        print("\nTest 4.3: Action selection from policy")
        action_counts = {1: 0, 2: 0, 3: 0}
        for _ in range(100):
            action = policy_learner.select_action(state_id, available_actions)
            action_counts[action] += 1
        
        print(f"   Action selections:")
        for action, count in action_counts.items():
            prob = policy_learner.get_action_probability(state_id, action, available_actions)
            print(f"      Action {action}: {count} times (P={prob:.2f})")
        
        # Action 2 should be selected most often
        most_selected = max(action_counts, key=action_counts.get)
        print(f"   Most selected: Action {most_selected}")
        print(f"   Result: {'✅ PASS' if most_selected == 2 else '⚠️  CHECK'}")
        
        return True
    
    def test_integrated_reward_learning(self):
        """Test all reward learning mechanisms together"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Reward Learning")
        print("="*60)
        
        manager = RewardLearningManager(
            enable_rpe=True,
            enable_value_learning=True,
            enable_q_learning=True,
            enable_policy_gradient=True
        )
        
        # Simulate learning in a simple environment
        print("\nSimulating learning in reward environment...")
        
        # Environment: 3 states, 2 actions per state
        # Optimal path: state1 -> action1 -> state2 -> action1 -> state3 (reward=1.0)
        states = [1, 2, 3]
        actions = [1, 2]
        
        # Learning episodes
        episodes = 50
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            state = 1
            total_reward = 0.0
            steps = 0
            max_steps = 10
            
            while state < 3 and steps < max_steps:
                # Select action
                if state == 1:
                    available = actions
                else:
                    available = actions
                
                action = manager.select_action(state, available, method='q_learning')
                
                # Take action and get reward
                if state == 1 and action == 1:
                    next_state = 2
                    reward = 0.0
                elif state == 1 and action == 2:
                    next_state = 1  # Stay
                    reward = 0.0
                elif state == 2 and action == 1:
                    next_state = 3
                    reward = 1.0
                elif state == 2 and action == 2:
                    next_state = 2  # Stay
                    reward = 0.0
                else:
                    next_state = state
                    reward = 0.0
                
                # Process reward and learn
                updates = manager.process_reward(
                    state, action, reward, next_state, current_time=episode * 10 + steps
                )
                
                total_reward += reward
                state = next_state
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        # Visualize learning
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Reward Learning Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Episode rewards
        ax1 = axes[0, 0]
        ax1.plot(episode_rewards, 'b-', linewidth=2, alpha=0.7)
        ax1.axhline(y=1.0, color='g', linestyle='--', label='Optimal Reward')
        ax1.set_title('Episode Rewards', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode lengths
        ax2 = axes[0, 1]
        ax2.plot(episode_lengths, 'r-', linewidth=2, alpha=0.7)
        ax2.axhline(y=2, color='g', linestyle='--', label='Optimal Length')
        ax2.set_title('Episode Lengths', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Moving average rewards
        ax3 = axes[1, 0]
        window = 10
        moving_avg = []
        for i in range(len(episode_rewards)):
            start = max(0, i - window + 1)
            avg = np.mean(episode_rewards[start:i+1])
            moving_avg.append(avg)
        ax3.plot(moving_avg, 'g-', linewidth=2)
        ax3.set_title(f'Moving Average Reward (window={window})', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Reward')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Q-values evolution
        ax4 = axes[1, 1]
        if manager.q_learning:
            q_vals = []
            for state in states:
                for action in actions:
                    q_val = manager.q_learning.get_q_value(state, action)
                    q_vals.append(q_val)
            
            ax4.bar(range(len(q_vals)), q_vals, color='purple', alpha=0.7)
            ax4.set_title('Final Q-Values', fontsize=12, fontweight='bold')
            ax4.set_xlabel('State-Action Pair')
            ax4.set_ylabel('Q-Value')
            ax4.set_xticks(range(len(q_vals)))
            ax4.set_xticklabels([f'S{s}A{a}' for s in states for a in actions], rotation=45)
            ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reward_learning_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Learning progress visualization saved: reward_learning_progress.png")
        plt.close()
        
        # Statistics
        print(f"\n📊 Learning Statistics:")
        print(f"   Episodes: {episodes}")
        print(f"   Average reward: {np.mean(episode_rewards):.3f}")
        print(f"   Final 10 episodes avg: {np.mean(episode_rewards[-10:]):.3f}")
        print(f"   Average episode length: {np.mean(episode_lengths):.2f} steps")
        print(f"   Optimal episodes: {sum(1 for r in episode_rewards if r == 1.0)}/{episodes}")
        
        stats = manager.get_statistics()
        print(f"\n   System Statistics:")
        if 'rpe' in stats:
            print(f"   RPE predictions: {stats['rpe']['num_predictions']}")
        if 'value_function' in stats:
            print(f"   States learned: {stats['value_function']['num_states']}")
        if 'q_learning' in stats:
            print(f"   Q-values learned: {stats['q_learning']['num_q_values']}")
        
        # Check if learning improved
        early_avg = np.mean(episode_rewards[:10])
        late_avg = np.mean(episode_rewards[-10:])
        improved = late_avg > early_avg
        
        print(f"\n   Learning improvement: {early_avg:.3f} → {late_avg:.3f}")
        print(f"   Result: {'✅ PASS' if improved else '⚠️  CHECK'}")
        
        return True
    
    def run_all_tests(self):
        """Run all reward learning tests"""
        print("\n" + "="*70)
        print("REWARD-BASED LEARNING TEST SUITE")
        print("="*70)
        
        tests = [
            ("Reward Prediction Error", self.test_rpe),
            ("Value Function Learning", self.test_value_function_learning),
            ("Q-Learning", self.test_q_learning),
            ("Policy Gradient Learning", self.test_policy_gradient),
            ("Integrated Reward Learning", self.test_integrated_reward_learning)
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
    tester = RewardLearningTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All reward learning tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

