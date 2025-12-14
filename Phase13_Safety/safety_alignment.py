#!/usr/bin/env python3
"""
Safety & Alignment - Phase 13.2
Implements value alignment, safe exploration, constraint satisfaction,
harm prevention, and value learning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

# Import dependencies
try:
    from value_systems import ValueSystemsManager
    from goal_setting_planning import GoalSettingPlanning
    from reward_learning import RewardLearningManager
except ImportError:
    ValueSystemsManager = None
    GoalSettingPlanning = None
    RewardLearningManager = None


@dataclass
class SafetyConstraint:
    """Represents a safety constraint"""
    constraint_id: int
    description: str
    constraint_type: str  # 'hard', 'soft'
    violation_penalty: float = 1.0
    enabled: bool = True


@dataclass
class Value:
    """Represents a value"""
    value_id: int
    name: str
    description: str
    importance: float = 1.0
    learned_from_feedback: bool = False


class ValueAlignment:
    """
    Value Alignment
    
    Aligns behavior with human values
    Ensures value consistency
    """
    
    def __init__(self, value_systems: Optional[ValueSystemsManager] = None):
        self.value_systems = value_systems
        self.values: Dict[int, Value] = {}
        self.next_value_id = 0
        self.alignment_history: List[Dict] = []
    
    def register_value(self,
                      name: str,
                      description: str,
                      importance: float = 1.0) -> Value:
        """Register a value"""
        value = Value(
            value_id=self.next_value_id,
            name=name,
            description=description,
            importance=importance
        )
        
        self.values[self.next_value_id] = value
        self.next_value_id += 1
        
        return value
    
    def check_value_alignment(self,
                            action: Dict,
                            values: List[Value] = None) -> Dict:
        """
        Check if action aligns with values
        
        Returns:
            Alignment result
        """
        if values is None:
            values = list(self.values.values())
        
        alignment_scores = []
        
        for value in values:
            # Simplified alignment check
            action_description = action.get('description', '').lower()
            value_name = value.name.lower()
            
            # Check if action aligns with value
            if value_name in action_description:
                score = value.importance
            else:
                score = 0.5 * value.importance  # Neutral alignment
            
            alignment_scores.append(score)
        
        overall_alignment = np.mean(alignment_scores) if alignment_scores else 0.5
        
        result = {
            'action': action,
            'alignment_score': overall_alignment,
            'is_aligned': overall_alignment > 0.6,
            'value_scores': dict(zip([v.name for v in values], alignment_scores))
        }
        
        self.alignment_history.append(result)
        return result
    
    def align_action(self,
                    action: Dict,
                    values: List[Value] = None) -> Dict:
        """
        Align action with values
        
        Returns:
            Aligned action
        """
        alignment = self.check_value_alignment(action, values)
        
        if not alignment['is_aligned']:
            # Modify action to better align
            aligned_action = action.copy()
            aligned_action['description'] = f"{action.get('description', '')} (aligned with values)"
            aligned_action['aligned'] = True
            return aligned_action
        
        return action


class SafeExploration:
    """
    Safe Exploration
    
    Explores safely without causing harm
    Constrains exploration
    """
    
    def __init__(self):
        self.exploration_history: List[Dict] = []
        self.safety_bounds: Dict[str, Tuple[float, float]] = {}
    
    def set_safety_bounds(self,
                         parameter_name: str,
                         min_value: float,
                         max_value: float):
        """Set safety bounds for parameter"""
        self.safety_bounds[parameter_name] = (min_value, max_value)
    
    def explore_safely(self,
                      action_space: Dict[str, np.ndarray],
                      constraints: List[SafetyConstraint] = None) -> Dict:
        """
        Explore action space safely
        
        Returns:
            Safe action
        """
        if constraints is None:
            constraints = []
        
        # Generate action within safety bounds
        safe_action = {}
        
        for param_name, param_space in action_space.items():
            if param_name in self.safety_bounds:
                min_val, max_val = self.safety_bounds[param_name]
                # Sample within bounds
                if isinstance(param_space, np.ndarray):
                    safe_value = np.clip(
                        np.random.choice(param_space),
                        min_val, max_val
                    )
                else:
                    safe_value = np.clip(
                        np.random.uniform(0, 1),
                        min_val, max_val
                    )
            else:
                # No bounds, use original space
                if isinstance(param_space, np.ndarray):
                    safe_value = np.random.choice(param_space)
                else:
                    safe_value = np.random.uniform(0, 1)
            
            safe_action[param_name] = safe_value
        
        # Check constraints
        for constraint in constraints:
            if constraint.enabled:
                if not self._satisfies_constraint(safe_action, constraint):
                    # Adjust action to satisfy constraint
                    safe_action = self._adjust_for_constraint(safe_action, constraint)
        
        self.exploration_history.append({
            'action': safe_action,
            'timestamp': time.time()
        })
        
        return safe_action
    
    def _satisfies_constraint(self, action: Dict, constraint: SafetyConstraint) -> bool:
        """Check if action satisfies constraint"""
        # Simplified: check if action description matches constraint
        action_desc = str(action).lower()
        constraint_desc = constraint.description.lower()
        
        # Check for violations
        violation_keywords = ['harm', 'danger', 'unsafe']
        for keyword in violation_keywords:
            if keyword in action_desc and keyword not in constraint_desc:
                return False
        
        return True
    
    def _adjust_for_constraint(self,
                              action: Dict,
                              constraint: SafetyConstraint) -> Dict:
        """Adjust action to satisfy constraint"""
        adjusted = action.copy()
        adjusted['safety_adjusted'] = True
        adjusted['constraint_applied'] = constraint.constraint_id
        return adjusted


class ConstraintSatisfaction:
    """
    Constraint Satisfaction
    
    Satisfies safety constraints
    Ensures constraint compliance
    """
    
    def __init__(self):
        self.constraints: Dict[int, SafetyConstraint] = {}
        self.next_constraint_id = 0
        self.violations: List[Dict] = []
    
    def add_constraint(self,
                      description: str,
                      constraint_type: str = 'hard',
                      violation_penalty: float = 1.0) -> SafetyConstraint:
        """Add safety constraint"""
        constraint = SafetyConstraint(
            constraint_id=self.next_constraint_id,
            description=description,
            constraint_type=constraint_type,
            violation_penalty=violation_penalty
        )
        
        self.constraints[self.next_constraint_id] = constraint
        self.next_constraint_id += 1
        
        return constraint
    
    def check_constraints(self,
                         action: Dict,
                         constraints: List[SafetyConstraint] = None) -> Dict:
        """
        Check if action satisfies constraints
        
        Returns:
            Constraint check result
        """
        if constraints is None:
            constraints = list(self.constraints.values())
        
        violations = []
        total_penalty = 0.0
        
        for constraint in constraints:
            if constraint.enabled:
                if not self._satisfies_constraint(action, constraint):
                    violations.append(constraint)
                    total_penalty += constraint.violation_penalty
        
        result = {
            'satisfies_all': len(violations) == 0,
            'violations': [v.constraint_id for v in violations],
            'total_penalty': total_penalty,
            'action': action
        }
        
        if violations:
            self.violations.append({
                'action': action,
                'violations': [v.constraint_id for v in violations],
                'timestamp': time.time()
            })
        
        return result
    
    def _satisfies_constraint(self, action: Dict, constraint: SafetyConstraint) -> bool:
        """Check constraint satisfaction"""
        action_desc = str(action).lower()
        constraint_desc = constraint.description.lower()
        
        # Check for constraint violations
        violation_keywords = ['harm', 'danger', 'unsafe', 'illegal']
        for keyword in violation_keywords:
            if keyword in action_desc and 'not' not in action_desc:
                return False
        
        return True


class HarmPrevention:
    """
    Harm Prevention
    
    Prevents harmful actions
    Blocks dangerous behaviors
    """
    
    def __init__(self):
        self.harmful_patterns: List[str] = ['harm', 'danger', 'violence', 'destruction']
        self.blocked_actions: List[Dict] = []
    
    def check_for_harm(self, action: Dict) -> bool:
        """
        Check if action could cause harm
        
        Returns:
            True if harmful
        """
        action_desc = str(action).lower()
        
        for pattern in self.harmful_patterns:
            if pattern in action_desc:
                return True
        
        return False
    
    def prevent_harm(self, action: Dict) -> Dict:
        """
        Prevent harmful action
        
        Returns:
            Prevention result
        """
        is_harmful = self.check_for_harm(action)
        
        if is_harmful:
            self.blocked_actions.append({
                'action': action,
                'reason': 'harmful_pattern_detected',
                'timestamp': time.time()
            })
            
            return {
                'action_blocked': True,
                'reason': 'harmful_action_detected',
                'original_action': action
            }
        
        return {
            'action_blocked': False,
            'action': action
        }


class ValueLearning:
    """
    Value Learning
    
    Learns human values from feedback
    Updates value system
    """
    
    def __init__(self,
                 reward_learning: Optional[RewardLearningManager] = None,
                 value_alignment: Optional[ValueAlignment] = None):
        self.reward_learning = reward_learning
        self.value_alignment = value_alignment
        self.learned_values: List[Value] = []
        self.feedback_history: List[Dict] = []
    
    def learn_from_feedback(self,
                           action: Dict,
                           feedback: Dict) -> Optional[Value]:
        """
        Learn value from feedback
        
        Returns:
            Learned value or None
        """
        # Extract value from feedback
        feedback_text = feedback.get('text', '').lower()
        
        # Look for value indicators
        value_indicators = {
            'helpful': 'helpfulness',
            'honest': 'honesty',
            'respectful': 'respect',
            'fair': 'fairness',
            'safe': 'safety'
        }
        
        learned_value = None
        
        for indicator, value_name in value_indicators.items():
            if indicator in feedback_text:
                # Create or update value
                learned_value = Value(
                    value_id=-1,
                    name=value_name,
                    description=f"Learned from feedback: {feedback_text}",
                    importance=1.0 if 'important' in feedback_text else 0.7,
                    learned_from_feedback=True
                )
                
                self.learned_values.append(learned_value)
                
                # Register with value alignment if available
                if self.value_alignment:
                    self.value_alignment.register_value(
                        learned_value.name,
                        learned_value.description,
                        learned_value.importance
                    )
                
                break
        
        self.feedback_history.append({
            'action': action,
            'feedback': feedback,
            'learned_value': learned_value.name if learned_value else None,
            'timestamp': time.time()
        })
        
        return learned_value
    
    def update_value_importance(self,
                               value_name: str,
                               importance_change: float):
        """Update value importance based on feedback"""
        for value in self.learned_values:
            if value.name == value_name:
                value.importance = max(0.0, min(1.0, value.importance + importance_change))
                break


class SafetyAlignmentSystem:
    """
    Safety & Alignment System Manager
    
    Integrates all safety and alignment components
    """
    
    def __init__(self,
                 brain_system=None,
                 value_systems: Optional[ValueSystemsManager] = None,
                 goal_setting: Optional[GoalSettingPlanning] = None,
                 reward_learning: Optional[RewardLearningManager] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.value_alignment = ValueAlignment(value_systems)
        self.safe_exploration = SafeExploration()
        self.constraint_satisfaction = ConstraintSatisfaction()
        self.harm_prevention = HarmPrevention()
        self.value_learning = ValueLearning(reward_learning, None)
        
        # Set up value learning reference
        self.value_learning.value_alignment = self.value_alignment
        
        # Integration with existing systems
        self.value_systems = value_systems
        self.goal_setting = goal_setting
        self.reward_learning = reward_learning
        
        # Statistics
        self.stats = {
            'values_registered': 0,
            'actions_aligned': 0,
            'safe_explorations': 0,
            'constraints_satisfied': 0,
            'harmful_actions_blocked': 0,
            'values_learned': 0
        }
    
    def register_value(self, name: str, description: str, importance: float = 1.0) -> Value:
        """Register a value"""
        value = self.value_alignment.register_value(name, description, importance)
        self.stats['values_registered'] += 1
        return value
    
    def check_action_safety(self, action: Dict) -> Dict:
        """Check if action is safe and aligned"""
        # Check for harm
        harm_check = self.harm_prevention.prevent_harm(action)
        if harm_check['action_blocked']:
            self.stats['harmful_actions_blocked'] += 1
            return harm_check
        
        # Check constraints
        constraint_check = self.constraint_satisfaction.check_constraints(action)
        if constraint_check['satisfies_all']:
            self.stats['constraints_satisfied'] += 1
        
        # Check value alignment
        alignment_check = self.value_alignment.check_value_alignment(action)
        if alignment_check['is_aligned']:
            self.stats['actions_aligned'] += 1
        
        return {
            'safe': not harm_check['action_blocked'] and constraint_check['satisfies_all'],
            'aligned': alignment_check['is_aligned'],
            'harm_check': harm_check,
            'constraint_check': constraint_check,
            'alignment_check': alignment_check
        }
    
    def explore_safely(self, action_space: Dict[str, np.ndarray]) -> Dict:
        """Explore action space safely"""
        safe_action = self.safe_exploration.explore_safely(
            action_space,
            list(self.constraint_satisfaction.constraints.values())
        )
        self.stats['safe_explorations'] += 1
        return safe_action
    
    def learn_from_feedback(self, action: Dict, feedback: Dict) -> Optional[Value]:
        """Learn value from feedback"""
        learned = self.value_learning.learn_from_feedback(action, feedback)
        if learned:
            self.stats['values_learned'] += 1
        return learned
    
    def get_statistics(self) -> Dict:
        """Get safety and alignment statistics"""
        return self.stats.copy()

