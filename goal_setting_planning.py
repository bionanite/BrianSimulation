#!/usr/bin/env python3
"""
Goal Setting and Planning
Implements goal hierarchy, planning algorithms, subgoal decomposition, and goal monitoring
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
from collections import deque

@dataclass
class Goal:
    """Represents a goal"""
    goal_id: int
    description: str
    target_state: np.ndarray
    priority: float = 1.0
    deadline: Optional[float] = None
    parent_goal_id: Optional[int] = None
    subgoals: List[int] = field(default_factory=list)
    status: str = 'active'  # 'active', 'achieved', 'failed', 'suspended'
    created_time: float = 0.0
    progress: float = 0.0

@dataclass
class Plan:
    """Represents a plan to achieve a goal"""
    plan_id: int
    goal_id: int
    actions: List[int]
    expected_states: List[np.ndarray]
    confidence: float = 1.0
    created_time: float = 0.0

class GoalHierarchy:
    """
    Goal Hierarchy
    
    Organizes goals in hierarchical structure
    Supports subgoal decomposition
    """
    
    def __init__(self):
        self.goals: Dict[int, Goal] = {}
        self.next_goal_id = 0
    
    def create_goal(self,
                   description: str,
                   target_state: np.ndarray,
                   priority: float = 1.0,
                   parent_goal_id: Optional[int] = None,
                   deadline: Optional[float] = None) -> Goal:
        """Create a new goal"""
        goal = Goal(
            goal_id=self.next_goal_id,
            description=description,
            target_state=target_state.copy(),
            priority=priority,
            parent_goal_id=parent_goal_id,
            deadline=deadline,
            created_time=time.time()
        )
        
        self.goals[self.next_goal_id] = goal
        
        # Add to parent's subgoals
        if parent_goal_id is not None and parent_goal_id in self.goals:
            self.goals[parent_goal_id].subgoals.append(self.next_goal_id)
        
        self.next_goal_id += 1
        return goal
    
    def decompose_goal(self,
                      goal_id: int,
                      subgoal_states: List[np.ndarray],
                      subgoal_descriptions: Optional[List[str]] = None) -> List[Goal]:
        """Decompose a goal into subgoals"""
        if goal_id not in self.goals:
            return []
        
        goal = self.goals[goal_id]
        subgoals = []
        
        for i, subgoal_state in enumerate(subgoal_states):
            description = subgoal_descriptions[i] if subgoal_descriptions and i < len(subgoal_descriptions) else f"Subgoal {i+1}"
            subgoal = self.create_goal(
                description=description,
                target_state=subgoal_state,
                priority=goal.priority * 0.8,  # Slightly lower priority
                parent_goal_id=goal_id
            )
            subgoals.append(subgoal)
        
        return subgoals
    
    def get_active_goals(self) -> List[Goal]:
        """Get all active goals"""
        return [g for g in self.goals.values() if g.status == 'active']
    
    def get_top_level_goals(self) -> List[Goal]:
        """Get top-level goals (no parent)"""
        return [g for g in self.goals.values() if g.parent_goal_id is None and g.status == 'active']
    
    def update_goal_progress(self, goal_id: int, current_state: np.ndarray):
        """Update progress toward a goal"""
        if goal_id not in self.goals:
            return
        
        goal = self.goals[goal_id]
        
        # Compute progress (distance to target)
        distance = np.linalg.norm(current_state - goal.target_state)
        max_distance = np.linalg.norm(goal.target_state) + 1e-10
        
        # Progress: 1 - normalized_distance
        goal.progress = max(0.0, min(1.0, 1.0 - distance / max_distance))
        
        # Check if achieved
        if goal.progress > 0.95:
            goal.status = 'achieved'
            goal.progress = 1.0

class PlanningAlgorithm:
    """
    Planning Algorithm
    
    Generates plans to achieve goals
    Uses forward search and heuristics
    """
    
    def __init__(self,
                 max_plan_length: int = 20,
                 planning_horizon: int = 10):
        self.max_plan_length = max_plan_length
        self.planning_horizon = planning_horizon
    
    def plan_to_goal(self,
                    current_state: np.ndarray,
                    goal_state: np.ndarray,
                    available_actions: List[int],
                    transition_model) -> Optional[Plan]:
        """
        Plan sequence of actions to reach goal
        
        Uses forward search with heuristics
        """
        visited = set()
        
        # Simple forward search with visited tracking
        def search(state, goal, actions_taken, depth):
            if depth >= self.max_plan_length:
                return None
            
            # Check if goal reached (more lenient threshold)
            distance = np.linalg.norm(state - goal)
            if distance < 0.5:  # More lenient
                return actions_taken
            
            # Check if visited (prevent loops)
            state_hash = hash(tuple(np.round(state, 2)))
            if state_hash in visited:
                return None
            visited.add(state_hash)
            
            # Limit search breadth
            if depth > 5:
                # Only try best action (heuristic: action that moves toward goal)
                best_action = None
                best_score = float('-inf')
                for action in available_actions:
                    next_state, confidence = transition_model(state, action)
                    if confidence > 0.5:
                        # Score: how much closer to goal
                        new_distance = np.linalg.norm(next_state - goal)
                        score = distance - new_distance
                        if score > best_score:
                            best_score = score
                            best_action = action
                
                if best_action is not None:
                    next_state, confidence = transition_model(state, best_action)
                    if confidence > 0.5:
                        return search(next_state, goal, actions_taken + [best_action], depth + 1)
                return None
            
            # Try each action (limited depth)
            for action in available_actions:
                # Predict next state
                next_state, confidence = transition_model(state, action)
                
                if confidence > 0.5:
                    result = search(next_state, goal, actions_taken + [action], depth + 1)
                    if result is not None:
                        return result
            
            return None
        
        actions = search(current_state, goal_state, [], 0)
        
        if actions is None:
            return None
        
        # Build expected states
        expected_states = []
        current = current_state.copy()
        for action in actions:
            next_state, _ = transition_model(current, action)
            expected_states.append(next_state.copy())
            current = next_state
        
        plan = Plan(
            plan_id=hash(tuple(actions)) % 1000000,
            goal_id=-1,  # Will be set by caller
            actions=actions,
            expected_states=expected_states,
            confidence=0.8,
            created_time=time.time()
        )
        
        return plan
    
    def refine_plan(self, plan: Plan, current_state: np.ndarray, transition_model) -> Plan:
        """Refine plan based on current state"""
        # Replan from current state
        if not plan.expected_states:
            return plan
        
        goal_state = plan.expected_states[-1]
        available_actions = list(set(plan.actions))
        
        new_plan = self.plan_to_goal(
            current_state, goal_state, available_actions, transition_model
        )
        
        if new_plan:
            new_plan.goal_id = plan.goal_id
            return new_plan
        
        return plan

class SubgoalDecomposition:
    """
    Subgoal Decomposition
    
    Breaks complex goals into simpler subgoals
    """
    
    def __init__(self,
                 decomposition_strategy: str = 'linear'):  # 'linear', 'hierarchical'
        self.decomposition_strategy = decomposition_strategy
    
    def decompose(self,
                  current_state: np.ndarray,
                  goal_state: np.ndarray,
                  num_subgoals: int = 3) -> List[np.ndarray]:
        """
        Decompose goal into subgoals
        
        Returns:
            List of intermediate states (subgoals)
        """
        if self.decomposition_strategy == 'linear':
            # Linear interpolation
            subgoals = []
            for i in range(1, num_subgoals + 1):
                alpha = i / (num_subgoals + 1)
                subgoal_state = (1 - alpha) * current_state + alpha * goal_state
                subgoals.append(subgoal_state)
            return subgoals
        
        elif self.decomposition_strategy == 'hierarchical':
            # Hierarchical decomposition (divide space)
            subgoals = []
            diff = goal_state - current_state
            
            for i in range(num_subgoals):
                # Create subgoal at different points
                alpha = (i + 1) / (num_subgoals + 1)
                subgoal_state = current_state + alpha * diff
                subgoals.append(subgoal_state)
            
            return subgoals
        
        return []

class GoalMonitoring:
    """
    Goal Monitoring
    
    Monitors progress toward goals
    Detects failures and adjusts plans
    """
    
    def __init__(self,
                 failure_threshold: float = 0.1,
                 progress_threshold: float = 0.01):
        self.failure_threshold = failure_threshold
        self.progress_threshold = progress_threshold
    
    def check_goal_progress(self,
                          goal: Goal,
                          current_state: np.ndarray,
                          previous_state: Optional[np.ndarray] = None) -> Dict:
        """
        Check progress toward goal
        
        Returns:
            Dictionary with progress information
        """
        distance = np.linalg.norm(current_state - goal.target_state)
        progress = 1.0 - min(1.0, distance / (np.linalg.norm(goal.target_state) + 1e-10))
        
        result = {
            'distance': distance,
            'progress': progress,
            'is_achieved': progress > 0.95,
            'is_failed': False,
            'needs_replanning': False
        }
        
        # Check for failure (no progress)
        if previous_state is not None:
            prev_distance = np.linalg.norm(previous_state - goal.target_state)
            progress_made = prev_distance - distance
            
            if progress_made < self.progress_threshold and distance > self.failure_threshold:
                result['is_failed'] = True
                result['needs_replanning'] = True
        
        # Check deadline
        if goal.deadline is not None:
            time_remaining = goal.deadline - time.time()
            if time_remaining < 0:
                result['is_failed'] = True
        
        return result
    
    def monitor_goals(self,
                     goals: List[Goal],
                     current_state: np.ndarray,
                     previous_state: Optional[np.ndarray] = None) -> Dict:
        """Monitor multiple goals"""
        monitoring_results = {}
        
        for goal in goals:
            result = self.check_goal_progress(goal, current_state, previous_state)
            monitoring_results[goal.goal_id] = result
            
            # Update goal status
            if result['is_achieved']:
                goal.status = 'achieved'
                goal.progress = 1.0
            elif result['is_failed']:
                goal.status = 'failed'
        
        return monitoring_results

class GoalSettingPlanningManager:
    """
    Manages goal setting and planning
    """
    
    def __init__(self,
                 state_size: int = 20):
        self.state_size = state_size
        
        self.goal_hierarchy = GoalHierarchy()
        self.planning = PlanningAlgorithm()
        self.decomposition = SubgoalDecomposition()
        self.monitoring = GoalMonitoring()
        
        self.plans: Dict[int, Plan] = {}  # goal_id -> plan
        self.transition_model_cache = None
    
    def set_transition_model(self, transition_model):
        """Set transition model for planning"""
        self.transition_model_cache = transition_model
    
    def create_goal(self,
                   description: str,
                   target_state: np.ndarray,
                   priority: float = 1.0,
                   decompose: bool = False) -> Goal:
        """Create a goal and optionally decompose it"""
        goal = self.goal_hierarchy.create_goal(description, target_state, priority)
        
        if decompose:
            current_state = np.zeros(self.state_size)  # Default
            subgoals = self.decomposition.decompose(current_state, target_state)
            self.goal_hierarchy.decompose_goal(goal.goal_id, subgoals)
        
        return goal
    
    def plan_for_goal(self,
                     goal_id: int,
                     current_state: np.ndarray,
                     available_actions: List[int]) -> Optional[Plan]:
        """Create a plan for a goal"""
        if goal_id not in self.goal_hierarchy.goals:
            return None
        
        goal = self.goal_hierarchy.goals[goal_id]
        
        if self.transition_model_cache is None:
            # Default transition model
            def default_model(state, action):
                next_state = state + np.random.normal(0, 0.1, len(state))
                return next_state, 0.8
            transition_model = default_model
        else:
            transition_model = self.transition_model_cache
        
        plan = self.planning.plan_to_goal(
            current_state, goal.target_state, available_actions, transition_model
        )
        
        if plan:
            plan.goal_id = goal_id
            self.plans[goal_id] = plan
        
        return plan
    
    def monitor_goals(self,
                     current_state: np.ndarray,
                     previous_state: Optional[np.ndarray] = None) -> Dict:
        """Monitor all active goals"""
        active_goals = self.goal_hierarchy.get_active_goals()
        return self.monitoring.monitor_goals(active_goals, current_state, previous_state)
    
    def get_statistics(self) -> Dict:
        """Get statistics about goals and plans"""
        return {
            'total_goals': len(self.goal_hierarchy.goals),
            'active_goals': len(self.goal_hierarchy.get_active_goals()),
            'top_level_goals': len(self.goal_hierarchy.get_top_level_goals()),
            'plans': len(self.plans),
            'achieved_goals': sum(1 for g in self.goal_hierarchy.goals.values() if g.status == 'achieved')
        }

