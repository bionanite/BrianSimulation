#!/usr/bin/env python3
"""
Theory of Mind
Implements mental state inference, belief tracking, intention recognition, and perspective taking
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
from collections import defaultdict

@dataclass
class MentalState:
    """Represents a mental state of another agent"""
    agent_id: int
    beliefs: Dict[str, float]  # belief_name -> confidence
    desires: Dict[str, float]  # desire_name -> strength
    intentions: List[str] = field(default_factory=list)
    knowledge: Set[str] = field(default_factory=set)
    last_updated: float = 0.0

@dataclass
class Belief:
    """Represents a belief"""
    belief_id: int
    content: str
    confidence: float = 1.0
    source: str = "direct_observation"
    last_updated: float = 0.0

class MentalStateInference:
    """
    Mental State Inference
    
    Infers mental states of other agents
    Based on observed behavior
    """
    
    def __init__(self,
                 inference_confidence: float = 0.7):
        self.inference_confidence = inference_confidence
        
        self.agent_mental_states: Dict[int, MentalState] = {}
        self.behavior_patterns: Dict[int, List[Tuple[str, np.ndarray]]] = {}  # agent_id -> [(action, state)]
    
    def observe_behavior(self,
                       agent_id: int,
                       action: str,
                       state: np.ndarray,
                       outcome: Optional[float] = None):
        """Observe behavior of an agent"""
        if agent_id not in self.behavior_patterns:
            self.behavior_patterns[agent_id] = []
        
        self.behavior_patterns[agent_id].append((action, state.copy()))
        
        # Limit history
        if len(self.behavior_patterns[agent_id]) > 100:
            self.behavior_patterns[agent_id] = self.behavior_patterns[agent_id][-100:]
    
    def infer_beliefs(self, agent_id: int) -> Dict[str, float]:
        """Infer beliefs of an agent from behavior"""
        if agent_id not in self.behavior_patterns:
            return {}
        
        beliefs = {}
        behavior_history = self.behavior_patterns[agent_id]
        
        # Analyze behavior patterns
        if len(behavior_history) >= 2:
            # Infer beliefs based on consistent actions
            action_counts = defaultdict(int)
            for action, _ in behavior_history:
                action_counts[action] += 1
            
            # Belief: agent prefers actions they do frequently
            total_actions = len(behavior_history)
            for action, count in action_counts.items():
                belief_name = f"prefers_{action}"
                confidence = count / total_actions
                beliefs[belief_name] = confidence
        
        return beliefs
    
    def infer_desires(self, agent_id: int, observed_goals: List[str]) -> Dict[str, float]:
        """Infer desires/goals of an agent"""
        desires = {}
        
        # If agent pursues certain goals, infer desire for them
        for goal in observed_goals:
            desire_strength = 0.8  # Default strength
            desires[goal] = desire_strength
        
        return desires
    
    def infer_intentions(self, agent_id: int) -> List[str]:
        """Infer intentions from recent behavior"""
        if agent_id not in self.behavior_patterns:
            return []
        
        behavior_history = self.behavior_patterns[agent_id]
        if not behavior_history:
            return []
        
        # Recent actions suggest intentions
        recent_actions = [action for action, _ in behavior_history[-5:]]
        
        # Infer intention from action pattern
        intentions = []
        if len(recent_actions) >= 2:
            # If repeating same action, infer intention to achieve something
            if len(set(recent_actions)) == 1:
                intentions.append(f"intends_to_{recent_actions[0]}")
        
        return intentions
    
    def update_mental_state(self, agent_id: int, observed_goals: Optional[List[str]] = None):
        """Update mental state model of an agent"""
        if agent_id not in self.agent_mental_states:
            self.agent_mental_states[agent_id] = MentalState(
                agent_id=agent_id,
                beliefs={},
                desires={}
            )
        
        mental_state = self.agent_mental_states[agent_id]
        
        # Infer beliefs
        beliefs = self.infer_beliefs(agent_id)
        mental_state.beliefs.update(beliefs)
        
        # Infer desires
        if observed_goals:
            desires = self.infer_desires(agent_id, observed_goals)
            mental_state.desires.update(desires)
        
        # Infer intentions
        intentions = self.infer_intentions(agent_id)
        mental_state.intentions = intentions
        
        mental_state.last_updated = time.time()

class BeliefTracking:
    """
    Belief Tracking
    
    Tracks beliefs of other agents
    Updates based on new information
    """
    
    def __init__(self):
        self.tracked_beliefs: Dict[int, Dict[str, Belief]] = {}  # agent_id -> {belief_name -> Belief}
        self.next_belief_id = 0
    
    def track_belief(self,
                    agent_id: int,
                    belief_content: str,
                    confidence: float = 1.0,
                    source: str = "inference") -> Belief:
        """Track a belief of an agent"""
        if agent_id not in self.tracked_beliefs:
            self.tracked_beliefs[agent_id] = {}
        
        if belief_content not in self.tracked_beliefs[agent_id]:
            belief = Belief(
                belief_id=self.next_belief_id,
                content=belief_content,
                confidence=confidence,
                source=source
            )
            self.tracked_beliefs[agent_id][belief_content] = belief
            self.next_belief_id += 1
        else:
            belief = self.tracked_beliefs[agent_id][belief_content]
            # Update confidence
            belief.confidence = 0.9 * belief.confidence + 0.1 * confidence
        
        belief.last_updated = time.time()
        return belief
    
    def update_belief(self,
                     agent_id: int,
                     belief_content: str,
                     new_confidence: float,
                     evidence: str = ""):
        """Update tracked belief"""
        if agent_id in self.tracked_beliefs and belief_content in self.tracked_beliefs[agent_id]:
            belief = self.tracked_beliefs[agent_id][belief_content]
            belief.confidence = new_confidence
            belief.last_updated = time.time()
    
    def get_beliefs(self, agent_id: int) -> Dict[str, float]:
        """Get all tracked beliefs for an agent"""
        if agent_id not in self.tracked_beliefs:
            return {}
        
        return {name: belief.confidence for name, belief in self.tracked_beliefs[agent_id].items()}

class IntentionRecognition:
    """
    Intention Recognition
    
    Recognizes intentions from behavior
    Predicts future actions
    """
    
    def __init__(self):
        self.intention_patterns: Dict[str, List[str]] = {}  # intention -> [action_sequence]
        self.recognized_intentions: Dict[int, List[str]] = {}  # agent_id -> [intentions]
    
    def learn_intention_pattern(self,
                               intention: str,
                               action_sequence: List[str]):
        """Learn pattern for recognizing intention"""
        if intention not in self.intention_patterns:
            self.intention_patterns[intention] = []
        
        self.intention_patterns[intention].append(action_sequence.copy())
    
    def recognize_intention(self,
                          agent_id: int,
                          recent_actions: List[str]) -> List[Tuple[str, float]]:
        """
        Recognize intention from action sequence
        
        Returns:
            List of (intention, confidence) tuples
        """
        recognized = []
        
        for intention, patterns in self.intention_patterns.items():
            # Check if recent actions match any pattern
            for pattern in patterns:
                if len(recent_actions) >= len(pattern):
                    # Check if pattern matches end of recent actions
                    if recent_actions[-len(pattern):] == pattern:
                        confidence = 0.8
                        recognized.append((intention, confidence))
                        break
        
        # Store recognized intentions
        if recognized:
            if agent_id not in self.recognized_intentions:
                self.recognized_intentions[agent_id] = []
            for intention, _ in recognized:
                if intention not in self.recognized_intentions[agent_id]:
                    self.recognized_intentions[agent_id].append(intention)
        
        return recognized
    
    def predict_action(self,
                      agent_id: int,
                      context: Optional[str] = None) -> Optional[str]:
        """Predict next action based on recognized intentions"""
        if agent_id not in self.recognized_intentions:
            return None
        
        intentions = self.recognized_intentions[agent_id]
        if not intentions:
            return None
        
        # Use most recent intention
        intention = intentions[-1]
        
        # Predict action based on intention pattern
        if intention in self.intention_patterns:
            patterns = self.intention_patterns[intention]
            if patterns:
                # Use first pattern
                pattern = patterns[0]
                if len(pattern) > 0:
                    return pattern[0]  # Next action in pattern
        
        return None

class PerspectiveTaking:
    """
    Perspective Taking
    
    Takes perspective of other agents
    Understands different viewpoints
    """
    
    def __init__(self):
        self.perspective_models: Dict[int, Dict] = {}  # agent_id -> perspective_model
    
    def model_perspective(self,
                        agent_id: int,
                        agent_knowledge: Set[str],
                        agent_position: Optional[np.ndarray] = None) -> Dict:
        """Model the perspective of an agent"""
        perspective = {
            'knowledge': agent_knowledge.copy(),
            'position': agent_position.copy() if agent_position is not None else None,
            'viewpoint': {}
        }
        
        self.perspective_models[agent_id] = perspective
        return perspective
    
    def predict_agent_view(self,
                          agent_id: int,
                          situation: Dict) -> Dict:
        """
        Predict how agent views a situation
        
        Based on their knowledge and perspective
        """
        if agent_id not in self.perspective_models:
            return {}
        
        perspective = self.perspective_models[agent_id]
        agent_view = {}
        
        # Agent's view depends on their knowledge
        agent_knowledge = perspective.get('knowledge', set())
        
        # Predict what agent knows about situation
        for key, value in situation.items():
            if key in agent_knowledge:
                agent_view[key] = value
            else:
                agent_view[key] = None  # Unknown to agent
        
        return agent_view
    
    def compute_perspective_difference(self,
                                     agent1_id: int,
                                     agent2_id: int,
                                     situation: Dict) -> float:
        """Compute difference in perspectives between two agents"""
        view1 = self.predict_agent_view(agent1_id, situation)
        view2 = self.predict_agent_view(agent2_id, situation)
        
        # Count differences
        differences = 0
        total = len(situation)
        
        for key in situation.keys():
            if view1.get(key) != view2.get(key):
                differences += 1
        
        return differences / total if total > 0 else 0.0

class TheoryOfMindManager:
    """
    Manages all theory of mind mechanisms
    """
    
    def __init__(self):
        self.mental_state_inference = MentalStateInference()
        self.belief_tracking = BeliefTracking()
        self.intention_recognition = IntentionRecognition()
        self.perspective_taking = PerspectiveTaking()
    
    def observe_agent(self,
                     agent_id: int,
                     action: str,
                     state: np.ndarray,
                     outcome: Optional[float] = None):
        """Observe an agent's behavior"""
        self.mental_state_inference.observe_behavior(agent_id, action, state, outcome)
    
    def infer_mental_state(self,
                          agent_id: int,
                          observed_goals: Optional[List[str]] = None):
        """Infer mental state of an agent"""
        self.mental_state_inference.update_mental_state(agent_id, observed_goals)
    
    def get_mental_state(self, agent_id: int) -> Optional[MentalState]:
        """Get mental state model of an agent"""
        return self.mental_state_inference.agent_mental_states.get(agent_id)
    
    def recognize_intention(self,
                          agent_id: int,
                          recent_actions: List[str]) -> List[Tuple[str, float]]:
        """Recognize intention of an agent"""
        return self.intention_recognition.recognize_intention(agent_id, recent_actions)
    
    def predict_agent_action(self, agent_id: int) -> Optional[str]:
        """Predict next action of an agent"""
        return self.intention_recognition.predict_action(agent_id)
    
    def model_agent_perspective(self,
                               agent_id: int,
                               knowledge: Set[str],
                               position: Optional[np.ndarray] = None):
        """Model perspective of an agent"""
        return self.perspective_taking.model_perspective(agent_id, knowledge, position)
    
    def get_statistics(self) -> Dict:
        """Get statistics about theory of mind"""
        return {
            'agents_tracked': len(self.mental_state_inference.agent_mental_states),
            'beliefs_tracked': sum(len(beliefs) for beliefs in self.belief_tracking.tracked_beliefs.values()),
            'intentions_recognized': sum(len(ints) for ints in self.intention_recognition.recognized_intentions.values()),
            'perspectives_modeled': len(self.perspective_taking.perspective_models)
        }

