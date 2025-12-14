#!/usr/bin/env python3
"""
World Models
Implements predictive models, causal reasoning, simulation, and planning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
from collections import defaultdict, deque

@dataclass
class StateTransition:
    """Represents a state transition"""
    from_state: np.ndarray
    to_state: np.ndarray
    action: Optional[int] = None
    reward: float = 0.0
    probability: float = 1.0
    frequency: int = 1

@dataclass
class CausalRelation:
    """Represents a causal relationship"""
    cause_id: int
    effect_id: int
    strength: float = 1.0
    confidence: float = 1.0
    frequency: int = 1

@dataclass
class MentalModel:
    """Represents a mental model of the world"""
    model_id: int
    states: List[np.ndarray]
    transitions: List[StateTransition]
    predictions: Dict[int, np.ndarray]  # state_id -> predicted_next_state
    accuracy: float = 0.0
    last_update: float = 0.0

class PredictiveModel:
    """
    Predictive Model
    
    Learns to predict future states from current states
    Uses transition probabilities
    """
    
    def __init__(self,
                 state_size: int,
                 learning_rate: float = 0.1,
                 prediction_horizon: int = 1):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.prediction_horizon = prediction_horizon
        
        self.transitions: List[StateTransition] = []
        self.transition_matrix: Dict[Tuple[int, int], float] = {}  # (from_idx, to_idx) -> prob
        self.state_index: Dict[int, np.ndarray] = {}  # state_hash -> state
        self.next_state_id = 0
    
    def _hash_state(self, state: np.ndarray) -> int:
        """Hash state to integer ID"""
        # Simple hash based on state values
        return hash(tuple(np.round(state, 2)))
    
    def _get_state_id(self, state: np.ndarray) -> int:
        """Get or create state ID"""
        state_hash = self._hash_state(state)
        if state_hash not in self.state_index:
            self.state_index[state_hash] = state.copy()
            return self.next_state_id
        return state_hash
    
    def learn_transition(self,
                        from_state: np.ndarray,
                        to_state: np.ndarray,
                        action: Optional[int] = None,
                        reward: float = 0.0):
        """Learn a state transition"""
        from_id = self._get_state_id(from_state)
        to_id = self._get_state_id(to_state)
        
        # Check if transition exists
        existing = None
        for trans in self.transitions:
            if (np.allclose(trans.from_state, from_state) and 
                np.allclose(trans.to_state, to_state) and
                trans.action == action):
                existing = trans
                break
        
        if existing:
            # Update existing transition
            existing.frequency += 1
            existing.reward = (existing.reward * (existing.frequency - 1) + reward) / existing.frequency
        else:
            # Create new transition
            transition = StateTransition(
                from_state=from_state.copy(),
                to_state=to_state.copy(),
                action=action,
                reward=reward
            )
            self.transitions.append(transition)
        
        # Update transition matrix
        key = (from_id, to_id)
        if key not in self.transition_matrix:
            self.transition_matrix[key] = 0.0
        self.transition_matrix[key] += self.learning_rate * (1.0 - self.transition_matrix[key])
    
    def predict_next_state(self,
                          current_state: np.ndarray,
                          action: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Predict next state from current state
        
        Returns:
            (predicted_state, confidence)
        """
        # Find similar transitions
        candidates = []
        for trans in self.transitions:
            if action is None or trans.action == action:
                similarity = self._compute_state_similarity(current_state, trans.from_state)
                if similarity > 0.5:
                    candidates.append((trans, similarity))
        
        if not candidates:
            # No prediction available
            return current_state.copy(), 0.0
        
        # Weighted average of candidate transitions
        total_weight = sum(sim for _, sim in candidates)
        if total_weight == 0:
            return current_state.copy(), 0.0
        
        predicted_state = np.zeros(self.state_size)
        for trans, similarity in candidates:
            weight = similarity / total_weight
            predicted_state += weight * trans.to_state
        
        confidence = min(1.0, total_weight / len(candidates))
        
        return predicted_state, confidence
    
    def _compute_state_similarity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Compute similarity between states"""
        diff = state1 - state2
        distance = np.linalg.norm(diff)
        similarity = np.exp(-distance)
        return similarity

class CausalReasoning:
    """
    Causal Reasoning
    
    Learns causal relationships between events
    Infers causes and effects
    """
    
    def __init__(self,
                 correlation_threshold: float = 0.6,
                 temporal_window: int = 5):
        self.correlation_threshold = correlation_threshold
        self.temporal_window = temporal_window
        
        self.causal_relations: List[CausalRelation] = []
        self.event_history: deque = deque(maxlen=100)
        self.event_index: Dict[int, str] = {}  # event_id -> description
        self.next_event_id = 0
    
    def record_event(self, event_id: int, description: str = ""):
        """Record an event occurrence"""
        self.event_history.append((time.time(), event_id))
        if event_id not in self.event_index:
            self.event_index[event_id] = description
    
    def learn_causality(self,
                       cause_id: int,
                       effect_id: int,
                       strength: float = 1.0):
        """Learn a causal relationship"""
        # Check if relation exists
        existing = None
        for rel in self.causal_relations:
            if rel.cause_id == cause_id and rel.effect_id == effect_id:
                existing = rel
                break
        
        if existing:
            # Update existing relation
            existing.frequency += 1
            existing.strength = (existing.strength * (existing.frequency - 1) + strength) / existing.frequency
            existing.confidence = min(1.0, existing.confidence + 0.1)
        else:
            # Create new relation
            relation = CausalRelation(
                cause_id=cause_id,
                effect_id=effect_id,
                strength=strength
            )
            self.causal_relations.append(relation)
    
    def infer_cause(self, effect_id: int) -> List[Tuple[int, float]]:
        """Infer possible causes for an effect"""
        causes = []
        for rel in self.causal_relations:
            if rel.effect_id == effect_id:
                causes.append((rel.cause_id, rel.strength * rel.confidence))
        
        # Sort by strength
        causes.sort(key=lambda x: x[1], reverse=True)
        return causes
    
    def infer_effect(self, cause_id: int) -> List[Tuple[int, float]]:
        """Infer possible effects of a cause"""
        effects = []
        for rel in self.causal_relations:
            if rel.cause_id == cause_id:
                effects.append((rel.effect_id, rel.strength * rel.confidence))
        
        # Sort by strength
        effects.sort(key=lambda x: x[1], reverse=True)
        return effects
    
    def detect_temporal_causality(self):
        """Detect causality from temporal patterns"""
        if len(self.event_history) < 2:
            return
        
        # Look for events that frequently occur before other events
        for i in range(len(self.event_history) - 1):
            cause_time, cause_id = self.event_history[i]
            effect_time, effect_id = self.event_history[i+1]
            
            time_diff = effect_time - cause_time
            if 0 < time_diff < self.temporal_window:
                # Potential causal relationship
                strength = 1.0 / (1.0 + time_diff)
                self.learn_causality(cause_id, effect_id, strength)

class SimulationEngine:
    """
    Simulation Engine
    
    Simulates world dynamics forward in time
    Uses predictive models
    """
    
    def __init__(self,
                 predictive_model: PredictiveModel,
                 max_steps: int = 100):
        self.predictive_model = predictive_model
        self.max_steps = max_steps
    
    def simulate(self,
                initial_state: np.ndarray,
                actions: Optional[List[int]] = None,
                steps: int = 10) -> List[Tuple[np.ndarray, float]]:
        """
        Simulate world forward
        
        Returns:
            List of (state, confidence) tuples
        """
        trajectory = []
        current_state = initial_state.copy()
        
        for step in range(min(steps, self.max_steps)):
            action = actions[step] if actions and step < len(actions) else None
            next_state, confidence = self.predictive_model.predict_next_state(current_state, action)
            
            trajectory.append((next_state.copy(), confidence))
            current_state = next_state.copy()
        
        return trajectory
    
    def plan_sequence(self,
                     initial_state: np.ndarray,
                     goal_state: np.ndarray,
                     available_actions: List[int],
                     max_depth: int = 5) -> Optional[List[int]]:
        """
        Plan sequence of actions to reach goal
        
        Returns:
            Sequence of actions or None if no plan found
        """
        # Simple forward search
        def search(current_state, goal_state, actions_taken, depth):
            if depth >= max_depth:
                return None
            
            # Check if goal reached
            if np.linalg.norm(current_state - goal_state) < 0.1:
                return actions_taken
            
            # Try each action
            for action in available_actions:
                next_state, confidence = self.predictive_model.predict_next_state(current_state, action)
                
                if confidence > 0.5:
                    result = search(next_state, goal_state, actions_taken + [action], depth + 1)
                    if result is not None:
                        return result
            
            return None
        
        return search(initial_state, goal_state, [], 0)

class MentalModelBuilder:
    """
    Mental Model Builder
    
    Builds and maintains mental models of the world
    """
    
    def __init__(self):
        self.mental_models: Dict[int, MentalModel] = {}
        self.next_model_id = 0
    
    def create_model(self, initial_states: List[np.ndarray]) -> MentalModel:
        """Create a new mental model"""
        model = MentalModel(
            model_id=self.next_model_id,
            states=initial_states.copy(),
            transitions=[],
            predictions={}
        )
        
        self.mental_models[self.next_model_id] = model
        self.next_model_id += 1
        
        return model
    
    def update_model(self,
                    model: MentalModel,
                    transition: StateTransition,
                    actual_next_state: np.ndarray):
        """Update mental model with new transition"""
        # Add transition
        model.transitions.append(transition)
        
        # Update prediction accuracy
        predicted_state = model.predictions.get(len(model.states) - 1)
        if predicted_state is not None:
            error = np.linalg.norm(predicted_state - actual_next_state)
            accuracy = 1.0 / (1.0 + error)
            model.accuracy = 0.9 * model.accuracy + 0.1 * accuracy
        
        # Add new state
        model.states.append(actual_next_state.copy())
        model.last_update = time.time()
    
    def predict_with_model(self,
                          model: MentalModel,
                          current_state: np.ndarray) -> np.ndarray:
        """Use mental model to predict next state"""
        # Find similar state in model
        best_match_idx = 0
        best_similarity = -1.0
        
        for i, state in enumerate(model.states):
            similarity = np.exp(-np.linalg.norm(state - current_state))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = i
        
        # Predict based on model transitions
        if best_match_idx < len(model.states) - 1:
            # Use next state in model
            predicted = model.states[best_match_idx + 1].copy()
        else:
            # Use last state
            predicted = model.states[-1].copy()
        
        model.predictions[len(model.states)] = predicted.copy()
        return predicted

class WorldModelManager:
    """
    Manages all world modeling mechanisms
    """
    
    def __init__(self,
                 state_size: int = 20):
        self.state_size = state_size
        
        self.predictive_model = PredictiveModel(state_size=state_size)
        self.causal_reasoning = CausalReasoning()
        self.simulation_engine = SimulationEngine(self.predictive_model)
        self.mental_model_builder = MentalModelBuilder()
    
    def learn_from_experience(self,
                             from_state: np.ndarray,
                             to_state: np.ndarray,
                             action: Optional[int] = None,
                             reward: float = 0.0):
        """Learn from experience"""
        self.predictive_model.learn_transition(from_state, to_state, action, reward)
    
    def predict_future(self,
                      current_state: np.ndarray,
                      action: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """Predict future state"""
        return self.predictive_model.predict_next_state(current_state, action)
    
    def simulate_trajectory(self,
                           initial_state: np.ndarray,
                           actions: Optional[List[int]] = None,
                           steps: int = 10) -> List[Tuple[np.ndarray, float]]:
        """Simulate future trajectory"""
        return self.simulation_engine.simulate(initial_state, actions, steps)
    
    def plan_actions(self,
                    initial_state: np.ndarray,
                    goal_state: np.ndarray,
                    available_actions: List[int]) -> Optional[List[int]]:
        """Plan sequence of actions"""
        return self.simulation_engine.plan_sequence(initial_state, goal_state, available_actions)
    
    def learn_causality(self,
                       cause_id: int,
                       effect_id: int,
                       strength: float = 1.0):
        """Learn causal relationship"""
        self.causal_reasoning.learn_causality(cause_id, effect_id, strength)
    
    def infer_causes(self, effect_id: int) -> List[Tuple[int, float]]:
        """Infer causes of effect"""
        return self.causal_reasoning.infer_cause(effect_id)
    
    def infer_effects(self, cause_id: int) -> List[Tuple[int, float]]:
        """Infer effects of cause"""
        return self.causal_reasoning.infer_effect(cause_id)
    
    def get_statistics(self) -> Dict:
        """Get statistics about world models"""
        return {
            'transitions_learned': len(self.predictive_model.transitions),
            'states_known': len(self.predictive_model.state_index),
            'causal_relations': len(self.causal_reasoning.causal_relations),
            'mental_models': len(self.mental_model_builder.mental_models)
        }

