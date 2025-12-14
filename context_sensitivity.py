#!/usr/bin/env python3
"""
Context Sensitivity
Implements context detection, context-dependent behavior, situational awareness, and context switching
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from collections import defaultdict

@dataclass
class Context:
    """Represents a context"""
    context_id: int
    name: str
    features: Dict[str, float]  # feature_name -> value
    activation: float = 1.0
    last_activated: float = 0.0

@dataclass
class ContextRule:
    """Represents a context-dependent rule"""
    rule_id: int
    context_name: str
    condition: Dict[str, float]  # feature -> threshold
    behavior: str
    priority: float = 1.0

class ContextDetection:
    """
    Context Detection
    
    Detects current context
    Identifies context features
    """
    
    def __init__(self):
        self.known_contexts: Dict[str, Context] = {}
        self.current_context: Optional[Context] = None
        self.context_history: List[Tuple[str, float]] = []  # (context_name, timestamp)
        self.next_context_id = 0
    
    def register_context(self,
                        name: str,
                        features: Dict[str, float]) -> Context:
        """Register a known context"""
        if name not in self.known_contexts:
            context = Context(
                context_id=self.next_context_id,
                name=name,
                features=features.copy(),
                activation=1.0
            )
            self.known_contexts[name] = context
            self.next_context_id += 1
        else:
            context = self.known_contexts[name]
            # Update features
            context.features.update(features)
        
        return self.known_contexts[name]
    
    def detect_context(self,
                      observed_features: Dict[str, float],
                      threshold: float = 0.7) -> Optional[str]:
        """
        Detect current context from observed features
        
        Returns:
            Context name if detected, None otherwise
        """
        best_match = None
        best_score = 0.0
        
        for context_name, context in self.known_contexts.items():
            # Compute similarity score
            score = self._compute_similarity(observed_features, context.features)
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = context_name
        
        if best_match:
            self.current_context = self.known_contexts[best_match]
            self.current_context.activation = best_score
            self.current_context.last_activated = time.time()
            self.context_history.append((best_match, time.time()))
        
        return best_match
    
    def _compute_similarity(self,
                           features1: Dict[str, float],
                           features2: Dict[str, float]) -> float:
        """Compute similarity between feature sets"""
        if not features1 or not features2:
            return 0.0
        
        # Get common features
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return 0.0
        
        # Compute cosine similarity
        values1 = np.array([features1[f] for f in common_features])
        values2 = np.array([features2[f] for f in common_features])
        
        dot_product = np.dot(values1, values2)
        norm1 = np.linalg.norm(values1)
        norm2 = np.linalg.norm(values2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_current_context(self) -> Optional[Context]:
        """Get current context"""
        return self.current_context
    
    def update_context_features(self,
                               context_name: str,
                               new_features: Dict[str, float]):
        """Update features of a context"""
        if context_name in self.known_contexts:
            context = self.known_contexts[context_name]
            context.features.update(new_features)

class ContextDependentBehavior:
    """
    Context-Dependent Behavior
    
    Adjusts behavior based on context
    Applies context-specific rules
    """
    
    def __init__(self):
        self.context_rules: Dict[str, List[ContextRule]] = {}  # context_name -> [rules]
        self.behavior_history: List[Tuple[str, str, float]] = []  # (context, behavior, timestamp)
        self.next_rule_id = 0
    
    def add_rule(self,
                context_name: str,
                condition: Dict[str, float],
                behavior: str,
                priority: float = 1.0) -> ContextRule:
        """Add a context-dependent rule"""
        rule = ContextRule(
            rule_id=self.next_rule_id,
            context_name=context_name,
            condition=condition.copy(),
            behavior=behavior,
            priority=priority
        )
        
        if context_name not in self.context_rules:
            self.context_rules[context_name] = []
        
        self.context_rules[context_name].append(rule)
        self.next_rule_id += 1
        
        return rule
    
    def select_behavior(self,
                      context_name: str,
                      current_features: Dict[str, float]) -> Optional[str]:
        """
        Select behavior based on context and features
        
        Returns:
            Selected behavior or None
        """
        if context_name not in self.context_rules:
            return None
        
        applicable_rules = []
        
        for rule in self.context_rules[context_name]:
            # Check if condition is met
            if self._check_condition(rule.condition, current_features):
                applicable_rules.append(rule)
        
        if not applicable_rules:
            return None
        
        # Select rule with highest priority
        best_rule = max(applicable_rules, key=lambda r: r.priority)
        
        self.behavior_history.append((context_name, best_rule.behavior, time.time()))
        
        return best_rule.behavior
    
    def _check_condition(self,
                        condition: Dict[str, float],
                        features: Dict[str, float]) -> bool:
        """Check if condition is met"""
        for feature, threshold in condition.items():
            if feature not in features:
                return False
            if features[feature] < threshold:
                return False
        return True
    
    def get_behaviors_for_context(self, context_name: str) -> List[str]:
        """Get all behaviors for a context"""
        if context_name not in self.context_rules:
            return []
        
        behaviors = [rule.behavior for rule in self.context_rules[context_name]]
        return list(set(behaviors))  # Unique behaviors

class SituationalAwareness:
    """
    Situational Awareness
    
    Maintains awareness of situation
    Tracks multiple context dimensions
    """
    
    def __init__(self):
        self.situational_features: Dict[str, float] = {}
        self.feature_history: Dict[str, List[Tuple[float, float]]] = {}  # feature -> [(value, timestamp)]
        self.attention_weights: Dict[str, float] = {}  # feature -> weight
    
    def update_feature(self,
                      feature_name: str,
                      value: float):
        """Update a situational feature"""
        self.situational_features[feature_name] = value
        
        # Update history
        if feature_name not in self.feature_history:
            self.feature_history[feature_name] = []
        
        self.feature_history[feature_name].append((value, time.time()))
        
        # Limit history
        if len(self.feature_history[feature_name]) > 100:
            self.feature_history[feature_name] = self.feature_history[feature_name][-100:]
    
    def set_attention_weight(self,
                           feature_name: str,
                           weight: float):
        """Set attention weight for a feature"""
        self.attention_weights[feature_name] = weight
    
    def get_situational_summary(self) -> Dict[str, float]:
        """Get summary of current situation"""
        summary = {}
        
        for feature, value in self.situational_features.items():
            weight = self.attention_weights.get(feature, 1.0)
            summary[feature] = value * weight
        
        return summary
    
    def detect_change(self,
                     feature_name: str,
                     threshold: float = 0.2) -> bool:
        """Detect significant change in a feature"""
        if feature_name not in self.feature_history or len(self.feature_history[feature_name]) < 2:
            return False
        
        history = self.feature_history[feature_name]
        recent_value = history[-1][0]
        previous_value = history[-2][0]
        
        change = abs(recent_value - previous_value)
        return change >= threshold

class ContextSwitching:
    """
    Context Switching
    
    Manages switching between contexts
    Handles context transitions
    """
    
    def __init__(self,
                 switch_threshold: float = 0.3,
                 switch_cost: float = 0.1):
        self.switch_threshold = switch_threshold
        self.switch_cost = switch_cost
        
        self.current_context: Optional[str] = None
        self.switch_history: List[Tuple[str, str, float]] = []  # (from, to, timestamp)
    
    def should_switch(self,
                     current_context: str,
                     new_context: str) -> bool:
        """Decide if should switch context"""
        if current_context == new_context:
            return False
        
        # Always switch if no current context
        if self.current_context is None:
            return True
        
        # Switch if new context is significantly different
        return True  # Simplified: always allow switch
    
    def switch_context(self,
                      from_context: Optional[str],
                      to_context: str) -> float:
        """
        Switch to a new context
        
        Returns:
            Switch cost
        """
        self.switch_history.append((from_context or "none", to_context, time.time()))
        self.current_context = to_context
        
        return self.switch_cost
    
    def get_switch_frequency(self, time_window: float = 1000.0) -> float:
        """Get context switching frequency"""
        if not self.switch_history:
            return 0.0
        
        current_time = time.time()
        recent_switches = [
            s for s in self.switch_history
            if current_time - s[2] < time_window
        ]
        
        return len(recent_switches) / (time_window / 1000.0) if time_window > 0 else 0.0

class ContextSensitivityManager:
    """
    Manages all context sensitivity mechanisms
    """
    
    def __init__(self):
        self.context_detection = ContextDetection()
        self.context_behavior = ContextDependentBehavior()
        self.situational_awareness = SituationalAwareness()
        self.context_switching = ContextSwitching()
    
    def register_context(self,
                        name: str,
                        features: Dict[str, float]):
        """Register a context"""
        return self.context_detection.register_context(name, features)
    
    def detect_context(self,
                      observed_features: Dict[str, float]) -> Optional[str]:
        """Detect current context"""
        return self.context_detection.detect_context(observed_features)
    
    def add_behavior_rule(self,
                         context_name: str,
                         condition: Dict[str, float],
                         behavior: str,
                         priority: float = 1.0):
        """Add a context-dependent behavior rule"""
        return self.context_behavior.add_rule(context_name, condition, behavior, priority)
    
    def select_behavior(self,
                       context_name: str,
                       current_features: Dict[str, float]) -> Optional[str]:
        """Select behavior for context"""
        return self.context_behavior.select_behavior(context_name, current_features)
    
    def update_situation(self,
                        feature_name: str,
                        value: float):
        """Update situational awareness"""
        self.situational_awareness.update_feature(feature_name, value)
    
    def switch_context(self, new_context: str) -> float:
        """Switch to a new context"""
        return self.context_switching.switch_context(
            self.context_switching.current_context,
            new_context
        )
    
    def get_statistics(self) -> Dict:
        """Get statistics about context sensitivity"""
        return {
            'known_contexts': len(self.context_detection.known_contexts),
            'current_context': self.context_detection.current_context.name if self.context_detection.current_context else None,
            'context_rules': sum(len(rules) for rules in self.context_behavior.context_rules.values()),
            'situational_features': len(self.situational_awareness.situational_features),
            'context_switches': len(self.context_switching.switch_history),
            'switch_frequency': self.context_switching.get_switch_frequency()
        }

