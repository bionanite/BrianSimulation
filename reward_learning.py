#!/usr/bin/env python3
"""
Reward-Based Learning Mechanisms
Implements RPE, value function learning, and policy gradient learning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

@dataclass
class StateValue:
    """Represents value of a state"""
    state_id: int
    value: float = 0.0
    visit_count: int = 0
    last_update_time: float = 0.0

@dataclass
class ActionValue:
    """Represents Q-value (state-action value)"""
    state_id: int
    action_id: int
    q_value: float = 0.0
    visit_count: int = 0
    last_update_time: float = 0.0

@dataclass
class Policy:
    """Represents action selection policy"""
    state_id: int
    action_probabilities: Dict[int, float] = field(default_factory=dict)
    action_preferences: Dict[int, float] = field(default_factory=dict)
    last_update_time: float = 0.0

class RewardPredictionError:
    """
    Reward Prediction Error (RPE) - Dopamine-like signal
    
    RPE = actual_reward - predicted_reward
    Positive RPE → unexpected reward → learning
    Negative RPE → expected reward missing → learning
    """
    
    def __init__(self,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 baseline_decay: float = 0.99):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.baseline_decay = baseline_decay
        
        # Track predictions
        self.predicted_rewards: Dict[int, float] = {}  # state_id -> predicted reward
        self.baseline_reward: float = 0.0  # Average reward baseline
    
    def predict_reward(self, state_id: int) -> float:
        """Get predicted reward for a state"""
        return self.predicted_rewards.get(state_id, 0.0)
    
    def calculate_rpe(self,
                     state_id: int,
                     actual_reward: float,
                     next_state_id: Optional[int] = None) -> float:
        """
        Calculate reward prediction error
        
        RPE = actual_reward + γ * V(next_state) - V(current_state)
        """
        predicted = self.predict_reward(state_id)
        
        # If next state provided, use its value
        if next_state_id is not None:
            next_value = self.predict_reward(next_state_id)
            target = actual_reward + self.discount_factor * next_value
        else:
            target = actual_reward
        
        rpe = target - predicted
        
        # Update baseline
        self.baseline_reward = (self.baseline_decay * self.baseline_reward + 
                               (1 - self.baseline_decay) * actual_reward)
        
        return rpe
    
    def update_prediction(self,
                        state_id: int,
                        rpe: float):
        """Update reward prediction based on RPE"""
        if state_id not in self.predicted_rewards:
            self.predicted_rewards[state_id] = 0.0
        
        # Update: prediction += learning_rate * RPE
        self.predicted_rewards[state_id] += self.learning_rate * rpe
    
    def get_rpe_signal(self, rpe: float) -> float:
        """
        Convert RPE to dopamine-like signal
        
        Positive RPE → positive signal (reward)
        Negative RPE → negative signal (punishment)
        """
        # Normalize RPE signal
        return np.tanh(rpe)  # Bounded between -1 and 1

class ValueFunctionLearning:
    """
    Value Function Learning
    
    Learns V(s) - value of states
    Uses Temporal Difference (TD) learning
    """
    
    def __init__(self,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.state_values: Dict[int, StateValue] = {}
    
    def get_value(self, state_id: int) -> float:
        """Get value of a state"""
        if state_id not in self.state_values:
            self.state_values[state_id] = StateValue(state_id=state_id)
        return self.state_values[state_id].value
    
    def update_value(self,
                    state_id: int,
                    reward: float,
                    next_state_id: Optional[int] = None,
                    current_time: float = 0.0):
        """
        Update state value using TD learning
        
        TD error = reward + γ * V(next_state) - V(current_state)
        V(current_state) += learning_rate * TD_error
        """
        if state_id not in self.state_values:
            self.state_values[state_id] = StateValue(state_id=state_id)
        
        state_value = self.state_values[state_id]
        current_value = state_value.value
        
        # Calculate TD target
        if next_state_id is not None:
            next_value = self.get_value(next_state_id)
            td_target = reward + self.discount_factor * next_value
        else:
            td_target = reward
        
        # Calculate TD error
        td_error = td_target - current_value
        
        # Update value
        state_value.value += self.learning_rate * td_error
        state_value.visit_count += 1
        state_value.last_update_time = current_time
        
        return td_error
    
    def get_statistics(self) -> Dict:
        """Get statistics about value function"""
        if not self.state_values:
            return {}
        
        values = [sv.value for sv in self.state_values.values()]
        visit_counts = [sv.visit_count for sv in self.state_values.values()]
        
        return {
            'num_states': len(self.state_values),
            'avg_value': np.mean(values),
            'value_std': np.std(values),
            'value_range': (min(values), max(values)),
            'avg_visits': np.mean(visit_counts),
            'total_visits': sum(visit_counts)
        }

class QLearning:
    """
    Q-Learning: Learn Q(s,a) - state-action values
    
    Q(s,a) = expected future reward for taking action a in state s
    """
    
    def __init__(self,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 exploration_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        self.q_values: Dict[Tuple[int, int], ActionValue] = {}  # (state, action) -> Q-value
    
    def get_q_value(self, state_id: int, action_id: int) -> float:
        """Get Q-value for state-action pair"""
        key = (state_id, action_id)
        if key not in self.q_values:
            self.q_values[key] = ActionValue(state_id=state_id, action_id=action_id)
        return self.q_values[key].q_value
    
    def update_q_value(self,
                      state_id: int,
                      action_id: int,
                      reward: float,
                      next_state_id: Optional[int] = None,
                      current_time: float = 0.0):
        """
        Update Q-value using Q-learning
        
        Q(s,a) += learning_rate * [reward + γ * max_a' Q(s',a') - Q(s,a)]
        """
        key = (state_id, action_id)
        if key not in self.q_values:
            self.q_values[key] = ActionValue(state_id=state_id, action_id=action_id)
        
        action_value = self.q_values[key]
        current_q = action_value.q_value
        
        # Calculate target Q-value
        if next_state_id is not None:
            # Find max Q-value for next state
            max_next_q = self.get_max_q_value(next_state_id)
            target_q = reward + self.discount_factor * max_next_q
        else:
            target_q = reward
        
        # Calculate TD error
        td_error = target_q - current_q
        
        # Update Q-value
        action_value.q_value += self.learning_rate * td_error
        action_value.visit_count += 1
        action_value.last_update_time = current_time
        
        return td_error
    
    def get_max_q_value(self, state_id: int) -> float:
        """Get maximum Q-value for a state (across all actions)"""
        q_vals = []
        for (s, a), av in self.q_values.items():
            if s == state_id:
                q_vals.append(av.q_value)
        
        return max(q_vals) if q_vals else 0.0
    
    def select_action(self, state_id: int, available_actions: List[int]) -> int:
        """
        Select action using epsilon-greedy policy
        
        With probability exploration_rate: random action
        Otherwise: action with highest Q-value
        """
        if np.random.random() < self.exploration_rate:
            # Explore: random action
            return np.random.choice(available_actions)
        else:
            # Exploit: best action
            best_action = None
            best_q = float('-inf')
            
            for action_id in available_actions:
                q_val = self.get_q_value(state_id, action_id)
                if q_val > best_q:
                    best_q = q_val
                    best_action = action_id
            
            return best_action if best_action is not None else np.random.choice(available_actions)

class PolicyGradientLearning:
    """
    Policy Gradient Learning
    
    Learns policy π(a|s) - probability of taking action a in state s
    Uses REINFORCE algorithm
    """
    
    def __init__(self,
                 learning_rate: float = 0.01,
                 baseline_decay: float = 0.99):
        self.learning_rate = learning_rate
        self.baseline_decay = baseline_decay
        
        self.policies: Dict[int, Policy] = {}
        self.baseline_return: float = 0.0
    
    def get_policy(self, state_id: int) -> Policy:
        """Get or create policy for a state"""
        if state_id not in self.policies:
            self.policies[state_id] = Policy(state_id=state_id)
        return self.policies[state_id]
    
    def get_action_probability(self,
                              state_id: int,
                              action_id: int,
                              available_actions: List[int]) -> float:
        """Get probability of taking action in state"""
        policy = self.get_policy(state_id)
        
        # Initialize if needed
        if action_id not in policy.action_preferences:
            policy.action_preferences[action_id] = 0.0
        
        # Convert preferences to probabilities using softmax
        preferences = [policy.action_preferences.get(a, 0.0) for a in available_actions]
        exp_prefs = np.exp(preferences - np.max(preferences))  # Numerical stability
        probabilities = exp_prefs / np.sum(exp_prefs)
        
        action_idx = available_actions.index(action_id)
        return probabilities[action_idx]
    
    def select_action(self, state_id: int, available_actions: List[int]) -> int:
        """Select action according to policy"""
        policy = self.get_policy(state_id)
        
        # Initialize preferences if needed
        for action_id in available_actions:
            if action_id not in policy.action_preferences:
                policy.action_preferences[action_id] = 0.0
        
        # Get probabilities
        preferences = [policy.action_preferences.get(a, 0.0) for a in available_actions]
        exp_prefs = np.exp(preferences - np.max(preferences))
        probabilities = exp_prefs / np.sum(exp_prefs)
        
        # Sample action
        return np.random.choice(available_actions, p=probabilities)
    
    def update_policy(self,
                     state_id: int,
                     action_id: int,
                     return_value: float,
                     current_time: float = 0.0):
        """
        Update policy using REINFORCE
        
        preference += learning_rate * (return - baseline) * (1 - probability)
        """
        policy = self.get_policy(state_id)
        
        # Calculate advantage (return - baseline)
        advantage = return_value - self.baseline_return
        
        # Update baseline
        self.baseline_return = (self.baseline_decay * self.baseline_return +
                               (1 - self.baseline_decay) * return_value)
        
        # Get current probability
        available_actions = list(policy.action_preferences.keys())
        if not available_actions:
            available_actions = [action_id]
        
        probability = self.get_action_probability(state_id, action_id, available_actions)
        
        # Update preference
        if action_id not in policy.action_preferences:
            policy.action_preferences[action_id] = 0.0
        
        # REINFORCE update
        policy.action_preferences[action_id] += (
            self.learning_rate * advantage * (1 - probability)
        )
        
        policy.last_update_time = current_time
        
        return advantage

class RewardLearningManager:
    """
    Manages all reward-based learning mechanisms
    """
    
    def __init__(self,
                 enable_rpe: bool = True,
                 enable_value_learning: bool = True,
                 enable_q_learning: bool = True,
                 enable_policy_gradient: bool = True):
        self.enable_rpe = enable_rpe
        self.enable_value_learning = enable_value_learning
        self.enable_q_learning = enable_q_learning
        self.enable_policy_gradient = enable_policy_gradient
        
        self.rpe = RewardPredictionError() if enable_rpe else None
        self.value_learning = ValueFunctionLearning() if enable_value_learning else None
        self.q_learning = QLearning() if enable_q_learning else None
        self.policy_gradient = PolicyGradientLearning() if enable_policy_gradient else None
        
        self.learning_history = []
    
    def process_reward(self,
                      state_id: int,
                      action_id: int,
                      reward: float,
                      next_state_id: Optional[int] = None,
                      current_time: float = 0.0) -> Dict:
        """
        Process reward and update all learning systems
        
        Returns:
            Dictionary with learning updates
        """
        updates = {
            'rpe': 0.0,
            'td_error': 0.0,
            'q_error': 0.0,
            'policy_advantage': 0.0
        }
        
        # RPE
        if self.enable_rpe and self.rpe:
            rpe = self.rpe.calculate_rpe(state_id, reward, next_state_id)
            self.rpe.update_prediction(state_id, rpe)
            updates['rpe'] = rpe
        
        # Value function learning
        if self.enable_value_learning and self.value_learning:
            td_error = self.value_learning.update_value(
                state_id, reward, next_state_id, current_time
            )
            updates['td_error'] = td_error
        
        # Q-learning
        if self.enable_q_learning and self.q_learning:
            q_error = self.q_learning.update_q_value(
                state_id, action_id, reward, next_state_id, current_time
            )
            updates['q_error'] = q_error
        
        return updates
    
    def select_action(self,
                     state_id: int,
                     available_actions: List[int],
                     method: str = 'q_learning') -> int:
        """
        Select action using specified method
        
        Methods: 'q_learning', 'policy_gradient', 'random'
        """
        if method == 'q_learning' and self.enable_q_learning and self.q_learning:
            return self.q_learning.select_action(state_id, available_actions)
        elif method == 'policy_gradient' and self.enable_policy_gradient and self.policy_gradient:
            return self.policy_gradient.select_action(state_id, available_actions)
        else:
            return np.random.choice(available_actions)
    
    def get_statistics(self) -> Dict:
        """Get statistics about all learning systems"""
        stats = {}
        
        if self.enable_rpe and self.rpe:
            stats['rpe'] = {
                'num_predictions': len(self.rpe.predicted_rewards),
                'baseline_reward': self.rpe.baseline_reward,
                'avg_prediction': np.mean(list(self.rpe.predicted_rewards.values())) if self.rpe.predicted_rewards else 0.0
            }
        
        if self.enable_value_learning and self.value_learning:
            stats['value_function'] = self.value_learning.get_statistics()
        
        if self.enable_q_learning and self.q_learning:
            q_vals = [av.q_value for av in self.q_learning.q_values.values()]
            stats['q_learning'] = {
                'num_q_values': len(self.q_learning.q_values),
                'avg_q_value': np.mean(q_vals) if q_vals else 0.0,
                'q_value_std': np.std(q_vals) if q_vals else 0.0,
                'exploration_rate': self.q_learning.exploration_rate
            }
        
        if self.enable_policy_gradient and self.policy_gradient:
            stats['policy_gradient'] = {
                'num_policies': len(self.policy_gradient.policies),
                'baseline_return': self.policy_gradient.baseline_return
            }
        
        return stats

