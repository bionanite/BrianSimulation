#!/usr/bin/env python3
"""
Social Learning
Implements imitation learning, social reinforcement, cultural transmission, and learning from others
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from collections import defaultdict

@dataclass
class Demonstration:
    """Represents a demonstration"""
    demo_id: int
    demonstrator_id: int
    action_sequence: List[str]
    outcome: float
    context: Optional[str] = None
    timestamp: float = 0.0

@dataclass
class SocialRule:
    """Represents a learned social rule"""
    rule_id: int
    rule: str
    context: str
    confidence: float = 1.0
    learned_from: List[int] = field(default_factory=list)  # agent_ids
    frequency: int = 1

class ImitationLearning:
    """
    Imitation Learning
    
    Learns by imitating others
    Copies successful behaviors
    """
    
    def __init__(self,
                 imitation_rate: float = 0.8,
                 selectivity: float = 0.7):
        self.imitation_rate = imitation_rate
        self.selectivity = selectivity
        
        self.demonstrations: Dict[int, Demonstration] = {}
        self.imitated_behaviors: Dict[str, int] = {}  # behavior -> count
        self.next_demo_id = 0
    
    def observe_demonstration(self,
                            demonstrator_id: int,
                            action_sequence: List[str],
                            outcome: float,
                            context: Optional[str] = None) -> Demonstration:
        """Observe a demonstration"""
        demo = Demonstration(
            demo_id=self.next_demo_id,
            demonstrator_id=demonstrator_id,
            action_sequence=action_sequence.copy(),
            outcome=outcome,
            context=context,
            timestamp=time.time()
        )
        
        self.demonstrations[self.next_demo_id] = demo
        self.next_demo_id += 1
        
        return demo
    
    def should_imitate(self, demonstration: Demonstration) -> bool:
        """Decide if should imitate a demonstration"""
        # Imitate if outcome is good
        if demonstration.outcome < self.selectivity:
            return False
        
        # Imitate based on imitation rate
        return np.random.random() < self.imitation_rate
    
    def imitate(self, demonstration: Demonstration) -> List[str]:
        """Imitate a demonstration"""
        if self.should_imitate(demonstration):
            behavior_key = "_".join(demonstration.action_sequence)
            self.imitated_behaviors[behavior_key] = self.imitated_behaviors.get(behavior_key, 0) + 1
            return demonstration.action_sequence.copy()
        return []
    
    def get_best_demonstration(self, context: Optional[str] = None) -> Optional[Demonstration]:
        """Get best demonstration for a context"""
        candidates = []
        
        for demo in self.demonstrations.values():
            if context is None or demo.context == context:
                candidates.append(demo)
        
        if not candidates:
            return None
        
        # Return demonstration with best outcome
        return max(candidates, key=lambda d: d.outcome)
    
    def learn_from_demonstrations(self, context: Optional[str] = None) -> List[str]:
        """Learn action sequence from demonstrations"""
        best_demo = self.get_best_demonstration(context)
        
        if best_demo:
            return self.imitate(best_demo)
        return []

class SocialReinforcement:
    """
    Social Reinforcement
    
    Learns from social feedback
    Adjusts behavior based on reactions
    """
    
    def __init__(self,
                 reinforcement_sensitivity: float = 0.8):
        self.reinforcement_sensitivity = reinforcement_sensitivity
        
        self.social_feedback: Dict[int, List[Tuple[str, float]]] = {}  # agent_id -> [(action, feedback)]
        self.behavior_scores: Dict[str, float] = {}  # behavior -> average_score
    
    def receive_feedback(self,
                        agent_id: int,
                        action: str,
                        feedback: float):
        """Receive social feedback"""
        if agent_id not in self.social_feedback:
            self.social_feedback[agent_id] = []
        
        self.social_feedback[agent_id].append((action, feedback))
        
        # Update behavior score
        if action not in self.behavior_scores:
            self.behavior_scores[action] = feedback
        else:
            # Moving average
            self.behavior_scores[action] = (
                0.7 * self.behavior_scores[action] + 0.3 * feedback
            )
    
    def get_action_score(self, action: str) -> float:
        """Get social score for an action"""
        return self.behavior_scores.get(action, 0.5)  # Neutral default
    
    def should_repeat_action(self, action: str) -> bool:
        """Decide if should repeat an action based on feedback"""
        score = self.get_action_score(action)
        return score > self.reinforcement_sensitivity
    
    def adjust_behavior(self, action: str) -> Optional[str]:
        """Adjust behavior based on social feedback"""
        score = self.get_action_score(action)
        
        if score < 0.3:
            # Negative feedback - avoid this action
            return None
        elif score > 0.7:
            # Positive feedback - continue this action
            return action
        else:
            # Neutral - might try variation
            return action

class CulturalTransmission:
    """
    Cultural Transmission
    
    Transmits knowledge across generations
    Maintains cultural practices
    """
    
    def __init__(self):
        self.cultural_practices: Dict[str, SocialRule] = {}
        self.transmission_history: List[Tuple[int, int, str]] = []  # (from, to, practice)
        self.next_rule_id = 0
    
    def create_practice(self,
                       practice: str,
                       context: str,
                       creator_id: int) -> SocialRule:
        """Create a new cultural practice"""
        rule = SocialRule(
            rule_id=self.next_rule_id,
            rule=practice,
            context=context,
            confidence=1.0,
            learned_from=[creator_id],
            frequency=1
        )
        
        self.cultural_practices[practice] = rule
        self.next_rule_id += 1
        
        return rule
    
    def transmit_practice(self,
                         from_agent_id: int,
                         to_agent_id: int,
                         practice: str):
        """Transmit a practice from one agent to another"""
        if practice not in self.cultural_practices:
            return False
        
        rule = self.cultural_practices[practice]
        
        # Add to transmission history
        self.transmission_history.append((from_agent_id, to_agent_id, practice))
        
        # Update rule frequency
        rule.frequency += 1
        
        # Add to learned_from if not already there
        if to_agent_id not in rule.learned_from:
            rule.learned_from.append(to_agent_id)
        
        return True
    
    def learn_practice(self,
                      agent_id: int,
                      practice: str,
                      context: str):
        """Learn a practice from observation"""
        if practice not in self.cultural_practices:
            # Create new practice
            self.create_practice(practice, context, agent_id)
        else:
            # Update existing practice
            rule = self.cultural_practices[practice]
            rule.frequency += 1
            if agent_id not in rule.learned_from:
                rule.learned_from.append(agent_id)
    
    def get_practices(self, context: Optional[str] = None) -> List[SocialRule]:
        """Get cultural practices"""
        if context is None:
            return list(self.cultural_practices.values())
        
        return [
            rule for rule in self.cultural_practices.values()
            if rule.context == context
        ]
    
    def get_most_common_practices(self, n: int = 5) -> List[SocialRule]:
        """Get most common practices"""
        practices = list(self.cultural_practices.values())
        practices.sort(key=lambda r: r.frequency, reverse=True)
        return practices[:n]

class LearningFromOthers:
    """
    Learning From Others
    
    General framework for learning from other agents
    Combines multiple social learning mechanisms
    """
    
    def __init__(self):
        self.imitation_learning = ImitationLearning()
        self.social_reinforcement = SocialReinforcement()
        self.cultural_transmission = CulturalTransmission()
        
        self.learned_behaviors: Dict[str, float] = {}  # behavior -> success_rate
        self.teachers: Dict[int, float] = {}  # teacher_id -> reliability_score
    
    def observe_teacher(self,
                       teacher_id: int,
                       action: str,
                       outcome: float):
        """Observe a teacher's behavior"""
        # Update teacher reliability
        if teacher_id not in self.teachers:
            self.teachers[teacher_id] = 0.5
        
        # Update reliability based on outcome
        self.teachers[teacher_id] = (
            0.9 * self.teachers[teacher_id] + 0.1 * outcome
        )
        
        # Record demonstration
        self.imitation_learning.observe_demonstration(
            teacher_id, [action], outcome
        )
    
    def learn_from_teacher(self,
                          teacher_id: int,
                          context: Optional[str] = None) -> List[str]:
        """Learn from a specific teacher"""
        # Check teacher reliability
        reliability = self.teachers.get(teacher_id, 0.5)
        
        if reliability < 0.5:
            return []  # Don't learn from unreliable teacher
        
        # Get best demonstration from this teacher
        demonstrations = [
            d for d in self.imitation_learning.demonstrations.values()
            if d.demonstrator_id == teacher_id
        ]
        
        if not demonstrations:
            return []
        
        best_demo = max(demonstrations, key=lambda d: d.outcome)
        return self.imitation_learning.imitate(best_demo)
    
    def integrate_social_learning(self,
                                action: str,
                                context: Optional[str] = None) -> Optional[str]:
        """Integrate multiple social learning mechanisms"""
        # Check social reinforcement
        if self.social_reinforcement.should_repeat_action(action):
            return action
        
        # Check cultural practices
        practices = self.cultural_transmission.get_practices(context)
        if practices:
            # Use most common practice
            most_common = practices[0]
            return most_common.rule
        
        # Use imitation learning
        learned = self.imitation_learning.learn_from_demonstrations(context)
        if learned:
            return learned[0] if learned else None
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get statistics about social learning"""
        return {
            'demonstrations_observed': len(self.imitation_learning.demonstrations),
            'behaviors_imitated': len(self.imitation_learning.imitated_behaviors),
            'social_feedback_received': sum(len(feedback) for feedback in self.social_reinforcement.social_feedback.values()),
            'cultural_practices': len(self.cultural_transmission.cultural_practices),
            'teachers_tracked': len(self.teachers),
            'avg_teacher_reliability': np.mean(list(self.teachers.values())) if self.teachers else 0.0
        }

class SocialLearningManager:
    """
    Manages all social learning mechanisms
    """
    
    def __init__(self):
        self.imitation_learning = ImitationLearning()
        self.social_reinforcement = SocialReinforcement()
        self.cultural_transmission = CulturalTransmission()
        self.learning_from_others = LearningFromOthers()
    
    def observe_demonstration(self,
                            demonstrator_id: int,
                            action_sequence: List[str],
                            outcome: float,
                            context: Optional[str] = None):
        """Observe a demonstration"""
        demo = self.imitation_learning.observe_demonstration(
            demonstrator_id, action_sequence, outcome, context
        )
        self.learning_from_others.observe_teacher(
            demonstrator_id, action_sequence[0] if action_sequence else "", outcome
        )
        return demo
    
    def receive_feedback(self, agent_id: int, action: str, feedback: float):
        """Receive social feedback"""
        self.social_reinforcement.receive_feedback(agent_id, action, feedback)
    
    def learn_from_demonstration(self, context: Optional[str] = None) -> List[str]:
        """Learn from demonstrations"""
        return self.imitation_learning.learn_from_demonstrations(context)
    
    def create_cultural_practice(self,
                                practice: str,
                                context: str,
                                creator_id: int):
        """Create a cultural practice"""
        return self.cultural_transmission.create_practice(practice, context, creator_id)
    
    def transmit_practice(self,
                         from_agent_id: int,
                         to_agent_id: int,
                         practice: str):
        """Transmit a practice"""
        return self.cultural_transmission.transmit_practice(from_agent_id, to_agent_id, practice)
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        stats = self.learning_from_others.get_statistics()
        stats.update({
            'imitated_behaviors': len(self.imitation_learning.imitated_behaviors),
            'cultural_practices': len(self.cultural_transmission.cultural_practices)
        })
        return stats

