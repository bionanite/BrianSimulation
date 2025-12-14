#!/usr/bin/env python3
"""
Value Systems
Implements value learning, preference formation, moral reasoning, and value-based decision making
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

@dataclass
class Value:
    """Represents a learned value"""
    value_id: int
    name: str
    value_type: str  # 'intrinsic', 'extrinsic', 'moral', 'aesthetic'
    strength: float = 1.0
    learned_from: List[str] = field(default_factory=list)
    last_updated: float = 0.0

@dataclass
class Preference:
    """Represents a preference"""
    preference_id: int
    item: str  # What is preferred
    context: Optional[str] = None
    strength: float = 1.0
    frequency: int = 1

class ValueLearning:
    """
    Value Learning
    
    Learns values from experiences
    Updates values based on outcomes
    """
    
    def __init__(self,
                 learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        
        self.values: Dict[int, Value] = {}
        self.next_value_id = 0
    
    def create_value(self,
                    name: str,
                    value_type: str,
                    initial_strength: float = 1.0) -> Value:
        """Create a new value"""
        value = Value(
            value_id=self.next_value_id,
            name=name,
            value_type=value_type,
            strength=initial_strength,
            last_updated=time.time()
        )
        
        self.values[self.next_value_id] = value
        self.next_value_id += 1
        
        return value
    
    def update_value(self,
                    value_id: int,
                    outcome: float,
                    experience_description: str = ""):
        """Update value based on outcome"""
        if value_id not in self.values:
            return
        
        value = self.values[value_id]
        
        # Update strength based on outcome
        # Positive outcome strengthens, negative weakens
        value.strength += self.learning_rate * outcome
        value.strength = np.clip(value.strength, 0.0, 2.0)  # Bounded
        
        if experience_description:
            value.learned_from.append(experience_description)
        
        value.last_updated = time.time()
    
    def evaluate_action(self,
                       action_description: str,
                       relevant_values: List[int]) -> float:
        """Evaluate an action based on values"""
        if not relevant_values:
            return 0.0
        
        total_value = 0.0
        total_weight = 0.0
        
        for value_id in relevant_values:
            if value_id in self.values:
                value = self.values[value_id]
                total_value += value.strength
                total_weight += 1.0
        
        return total_value / total_weight if total_weight > 0 else 0.0

class PreferenceFormation:
    """
    Preference Formation
    
    Forms preferences based on experiences
    Learns what is liked/disliked
    """
    
    def __init__(self):
        self.preferences: Dict[str, Preference] = {}
        self.next_preference_id = 0
    
    def form_preference(self,
                       item: str,
                       context: Optional[str] = None,
                       positive: bool = True) -> Preference:
        """Form or update a preference"""
        key = f"{item}_{context}" if context else item
        
        if key not in self.preferences:
            preference = Preference(
                preference_id=self.next_preference_id,
                item=item,
                context=context,
                strength=1.0 if positive else -1.0
            )
            self.preferences[key] = preference
            self.next_preference_id += 1
        else:
            preference = self.preferences[key]
            # Update strength
            if positive:
                preference.strength = min(1.0, preference.strength + 0.1)
            else:
                preference.strength = max(-1.0, preference.strength - 0.1)
            preference.frequency += 1
        
        return preference
    
    def get_preference(self, item: str, context: Optional[str] = None) -> float:
        """Get preference strength for an item"""
        key = f"{item}_{context}" if context else item
        
        if key in self.preferences:
            return self.preferences[key].strength
        return 0.0
    
    def rank_items(self, items: List[str]) -> List[Tuple[str, float]]:
        """Rank items by preference"""
        rankings = []
        for item in items:
            preference = self.get_preference(item)
            rankings.append((item, preference))
        
        # Sort by preference strength (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

class MoralReasoning:
    """
    Moral Reasoning
    
    Evaluates actions based on moral principles
    Considers consequences and intentions
    """
    
    def __init__(self):
        self.moral_principles: Dict[str, float] = {
            'harm': -1.0,      # Avoid harm
            'fairness': 1.0,    # Promote fairness
            'loyalty': 0.5,     # Value loyalty
            'authority': 0.3,   # Respect authority
            'purity': 0.2       # Maintain purity
        }
    
    def evaluate_moral_action(self,
                            action_description: str,
                            consequences: Dict[str, float],
                            intentions: Optional[List[str]] = None) -> float:
        """
        Evaluate moral value of an action
        
        Returns:
            Moral score (higher = more moral)
        """
        moral_score = 0.0
        
        # Evaluate based on consequences
        for principle, weight in self.moral_principles.items():
            if principle in consequences:
                moral_score += weight * consequences[principle]
        
        # Intentions matter
        if intentions:
            positive_intentions = sum(1 for i in intentions if 'good' in i.lower() or 'help' in i.lower())
            negative_intentions = sum(1 for i in intentions if 'harm' in i.lower() or 'hurt' in i.lower())
            intention_score = (positive_intentions - negative_intentions) / max(1, len(intentions))
            moral_score += 0.3 * intention_score
        
        return moral_score
    
    def update_moral_principle(self, principle: str, new_weight: float):
        """Update moral principle weight"""
        if principle in self.moral_principles:
            self.moral_principles[principle] = np.clip(new_weight, -1.0, 1.0)

class ValueBasedDecisionMaking:
    """
    Value-Based Decision Making
    
    Makes decisions based on values and preferences
    """
    
    def __init__(self,
                 value_learning: ValueLearning,
                 preference_formation: PreferenceFormation,
                 moral_reasoning: MoralReasoning):
        self.value_learning = value_learning
        self.preference_formation = preference_formation
        self.moral_reasoning = moral_reasoning
    
    def evaluate_option(self,
                       option_description: str,
                       value_ids: List[int],
                       consequences: Optional[Dict[str, float]] = None,
                       preferences: Optional[List[str]] = None) -> float:
        """
        Evaluate an option based on values, preferences, and morality
        
        Returns:
            Overall evaluation score
        """
        score = 0.0
        
        # Value-based evaluation
        value_score = self.value_learning.evaluate_action(option_description, value_ids)
        score += 0.4 * value_score
        
        # Preference-based evaluation
        if preferences:
            pref_scores = [self.preference_formation.get_preference(p) for p in preferences]
            pref_score = np.mean(pref_scores) if pref_scores else 0.0
            score += 0.3 * pref_score
        
        # Moral evaluation
        if consequences:
            moral_score = self.moral_reasoning.evaluate_moral_action(
                option_description, consequences
            )
            score += 0.3 * moral_score
        
        return score
    
    def choose_best_option(self,
                          options: List[str],
                          value_ids: List[int],
                          consequences_list: Optional[List[Dict[str, float]]] = None,
                          preferences: Optional[List[str]] = None) -> Tuple[int, float]:
        """
        Choose best option from alternatives
        
        Returns:
            (option_index, score)
        """
        scores = []
        
        for i, option in enumerate(options):
            consequences = consequences_list[i] if consequences_list and i < len(consequences_list) else None
            score = self.evaluate_option(option, value_ids, consequences, preferences)
            scores.append(score)
        
        best_idx = np.argmax(scores)
        return best_idx, scores[best_idx]

class ValueSystemManager:
    """
    Manages all value system mechanisms
    """
    
    def __init__(self):
        self.value_learning = ValueLearning()
        self.preference_formation = PreferenceFormation()
        self.moral_reasoning = MoralReasoning()
        self.decision_making = ValueBasedDecisionMaking(
            self.value_learning,
            self.preference_formation,
            self.moral_reasoning
        )
    
    def learn_value_from_experience(self,
                                   value_name: str,
                                   value_type: str,
                                   outcome: float,
                                   experience: str = ""):
        """Learn a value from experience"""
        # Find or create value
        value = None
        for v in self.value_learning.values.values():
            if v.name == value_name:
                value = v
                break
        
        if value is None:
            value = self.value_learning.create_value(value_name, value_type)
        
        self.value_learning.update_value(value.value_id, outcome, experience)
    
    def form_preference(self, item: str, positive: bool = True):
        """Form a preference"""
        return self.preference_formation.form_preference(item, positive=positive)
    
    def make_value_based_decision(self,
                                 options: List[str],
                                 value_names: List[str],
                                 consequences_list: Optional[List[Dict[str, float]]] = None) -> Tuple[int, float]:
        """Make a decision based on values"""
        # Get value IDs
        value_ids = []
        for name in value_names:
            for v in self.value_learning.values.values():
                if v.name == name:
                    value_ids.append(v.value_id)
                    break
        
        return self.decision_making.choose_best_option(options, value_ids, consequences_list)
    
    def get_statistics(self) -> Dict:
        """Get statistics about value systems"""
        return {
            'values': len(self.value_learning.values),
            'preferences': len(self.preference_formation.preferences),
            'moral_principles': len(self.moral_reasoning.moral_principles),
            'avg_value_strength': np.mean([v.strength for v in self.value_learning.values.values()]) if self.value_learning.values else 0.0
        }

