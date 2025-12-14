#!/usr/bin/env python3
"""
Intrinsic Motivation
Implements curiosity, novelty seeking, competence, and autonomous goal generation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from collections import deque

@dataclass
class IntrinsicReward:
    """Represents an intrinsic reward signal"""
    reward_type: str  # 'novelty', 'curiosity', 'competence', 'autonomy'
    magnitude: float
    source: str
    timestamp: float = 0.0

@dataclass
class Goal:
    """Represents an autonomously generated goal"""
    goal_id: int
    description: str
    target_state: np.ndarray
    priority: float = 1.0
    created_time: float = 0.0
    achieved: bool = False

class CuriosityDrive:
    """
    Curiosity Drive
    
    Seeks novel and informative experiences
    Rewards exploration of unknown states
    """
    
    def __init__(self,
                 novelty_threshold: float = 0.3,
                 curiosity_strength: float = 1.0):
        self.novelty_threshold = novelty_threshold
        self.curiosity_strength = curiosity_strength
        
        self.experienced_states: deque = deque(maxlen=1000)
        self.state_counts: Dict[int, int] = {}
    
    def _hash_state(self, state: np.ndarray) -> int:
        """Hash state to integer"""
        return hash(tuple(np.round(state, 2)))
    
    def compute_novelty(self, state: np.ndarray) -> float:
        """Compute novelty of a state"""
        state_hash = self._hash_state(state)
        
        # Check if state has been experienced
        if state_hash in self.state_counts:
            count = self.state_counts[state_hash]
            # Novelty decreases with experience
            novelty = 1.0 / (1.0 + count)
        else:
            # Completely novel
            novelty = 1.0
        
        return novelty
    
    def compute_curiosity_reward(self, state: np.ndarray) -> float:
        """Compute curiosity-based intrinsic reward"""
        novelty = self.compute_novelty(state)
        
        # Reward is proportional to novelty
        reward = self.curiosity_strength * novelty
        
        return reward
    
    def experience_state(self, state: np.ndarray):
        """Record experience of a state"""
        state_hash = self._hash_state(state)
        
        if state_hash not in self.state_counts:
            self.state_counts[state_hash] = 0
        
        self.state_counts[state_hash] += 1
        self.experienced_states.append(state.copy())

class NoveltySeeking:
    """
    Novelty Seeking
    
    Actively seeks out novel experiences
    Prefers unexplored regions
    """
    
    def __init__(self,
                 exploration_bonus: float = 0.5):
        self.exploration_bonus = exploration_bonus
        self.visited_regions: List[np.ndarray] = []
    
    def compute_exploration_bonus(self, state: np.ndarray) -> float:
        """Compute exploration bonus for a state"""
        if not self.visited_regions:
            return self.exploration_bonus
        
        # Find minimum distance to visited regions
        distances = [np.linalg.norm(state - region) for region in self.visited_regions]
        min_distance = min(distances) if distances else float('inf')
        
        # Bonus decreases with proximity to visited regions
        bonus = self.exploration_bonus * np.exp(-min_distance)
        
        return bonus
    
    def visit_region(self, state: np.ndarray):
        """Record visit to a region"""
        self.visited_regions.append(state.copy())
        
        # Limit memory
        if len(self.visited_regions) > 1000:
            self.visited_regions = self.visited_regions[-1000:]

class CompetenceMotivation:
    """
    Competence Motivation
    
    Seeks to improve skills and capabilities
    Rewards progress and mastery
    """
    
    def __init__(self):
        self.skill_levels: Dict[str, float] = {}  # skill_name -> level
        self.skill_history: Dict[str, List[float]] = {}
    
    def update_skill(self, skill_name: str, performance: float):
        """Update skill level based on performance"""
        if skill_name not in self.skill_levels:
            self.skill_levels[skill_name] = 0.0
            self.skill_history[skill_name] = []
        
        old_level = self.skill_levels[skill_name]
        self.skill_levels[skill_name] = 0.9 * old_level + 0.1 * performance
        self.skill_history[skill_name].append(self.skill_levels[skill_name])
    
    def compute_competence_reward(self, skill_name: str, performance: float) -> float:
        """Compute reward based on competence improvement"""
        if skill_name not in self.skill_levels:
            # New skill - high reward for learning
            return 1.0
        
        old_level = self.skill_levels[skill_name]
        improvement = performance - old_level
        
        # Reward improvement
        reward = max(0.0, improvement)
        
        return reward
    
    def get_mastery_level(self, skill_name: str) -> float:
        """Get current mastery level of a skill"""
        return self.skill_levels.get(skill_name, 0.0)

class AutonomyDrive:
    """
    Autonomy Drive
    
    Generates goals autonomously
    Seeks self-directed behavior
    """
    
    def __init__(self,
                 goal_generation_rate: float = 0.1):
        self.goal_generation_rate = goal_generation_rate
        self.autonomous_goals: List[Goal] = []
        self.next_goal_id = 0
    
    def generate_goal(self,
                     current_state: np.ndarray,
                     target_description: str = "") -> Goal:
        """Generate an autonomous goal"""
        # Create goal that differs from current state
        target_state = current_state.copy()
        
        # Modify target state (explore different direction)
        modification = np.random.normal(0, 0.2, len(target_state))
        target_state = target_state + modification
        target_state = np.clip(target_state, 0, 1)
        
        goal = Goal(
            goal_id=self.next_goal_id,
            description=target_description or f"Autonomous Goal {self.next_goal_id}",
            target_state=target_state,
            priority=np.random.random(),
            created_time=time.time()
        )
        
        self.autonomous_goals.append(goal)
        self.next_goal_id += 1
        
        return goal
    
    def should_generate_goal(self, current_time: float) -> bool:
        """Determine if a new goal should be generated"""
        if not self.autonomous_goals:
            return True
        
        # Check time since last goal
        last_goal_time = max(g.created_time for g in self.autonomous_goals)
        time_since_last = current_time - last_goal_time
        
        # Generate goal with some probability
        return np.random.random() < self.goal_generation_rate * time_since_last
    
    def get_active_goals(self) -> List[Goal]:
        """Get active (non-achieved) goals"""
        return [g for g in self.autonomous_goals if not g.achieved]
    
    def achieve_goal(self, goal_id: int):
        """Mark a goal as achieved"""
        for goal in self.autonomous_goals:
            if goal.goal_id == goal_id:
                goal.achieved = True
                break

class IntrinsicMotivationManager:
    """
    Manages all intrinsic motivation mechanisms
    """
    
    def __init__(self):
        self.curiosity = CuriosityDrive()
        self.novelty_seeking = NoveltySeeking()
        self.competence = CompetenceMotivation()
        self.autonomy = AutonomyDrive()
        
        self.intrinsic_rewards: List[IntrinsicReward] = []
    
    def compute_intrinsic_reward(self,
                                state: np.ndarray,
                                action: Optional[int] = None,
                                performance: Optional[float] = None,
                                skill_name: Optional[str] = None) -> float:
        """
        Compute total intrinsic reward
        
        Combines curiosity, novelty, competence, and autonomy
        """
        total_reward = 0.0
        
        # Curiosity reward
        curiosity_reward = self.curiosity.compute_curiosity_reward(state)
        total_reward += curiosity_reward
        
        # Novelty bonus
        novelty_bonus = self.novelty_seeking.compute_exploration_bonus(state)
        total_reward += novelty_bonus
        
        # Competence reward
        if skill_name and performance is not None:
            competence_reward = self.competence.compute_competence_reward(skill_name, performance)
            total_reward += competence_reward
        
        # Record experience
        self.curiosity.experience_state(state)
        self.novelty_seeking.visit_region(state)
        
        # Create reward record
        reward = IntrinsicReward(
            reward_type='combined',
            magnitude=total_reward,
            source='intrinsic_motivation',
            timestamp=time.time()
        )
        self.intrinsic_rewards.append(reward)
        
        return total_reward
    
    def generate_autonomous_goal(self,
                                current_state: np.ndarray,
                                description: str = "") -> Optional[Goal]:
        """Generate an autonomous goal"""
        if self.autonomy.should_generate_goal(time.time()):
            return self.autonomy.generate_goal(current_state, description)
        return None
    
    def update_competence(self, skill_name: str, performance: float):
        """Update competence in a skill"""
        self.competence.update_skill(skill_name, performance)
    
    def get_statistics(self) -> Dict:
        """Get statistics about intrinsic motivation"""
        return {
            'unique_states_experienced': len(self.curiosity.state_counts),
            'regions_visited': len(self.novelty_seeking.visited_regions),
            'skills_tracked': len(self.competence.skill_levels),
            'autonomous_goals': len(self.autonomy.autonomous_goals),
            'active_goals': len(self.autonomy.get_active_goals()),
            'total_intrinsic_rewards': len(self.intrinsic_rewards),
            'avg_intrinsic_reward': np.mean([r.magnitude for r in self.intrinsic_rewards]) if self.intrinsic_rewards else 0.0
        }

