#!/usr/bin/env python3
"""
Explainability & Interpretability - Phase 13.3
Implements self-explanation, decision explanation, feature attribution,
counterfactual explanations, and interpretable representations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

# Import dependencies
try:
    from metacognition import MetacognitiveMonitoring
    from advanced_reasoning import AdvancedReasoningSystem
    from global_workspace import GlobalWorkspace
except ImportError:
    MetacognitiveMonitoring = None
    AdvancedReasoningSystem = None
    GlobalWorkspace = None


@dataclass
class Explanation:
    """Represents an explanation"""
    explanation_id: int
    target: str  # What is being explained
    explanation_text: str
    explanation_type: str  # 'self', 'decision', 'counterfactual'
    confidence: float = 0.5
    created_time: float = 0.0


@dataclass
class FeatureAttribution:
    """Represents feature attribution"""
    feature_name: str
    attribution_score: float
    contribution: float  # Positive or negative


class SelfExplanation:
    """
    Self-Explanation
    
    Explains own reasoning processes
    Makes reasoning transparent
    """
    
    def __init__(self, metacognition: Optional[MetacognitiveMonitoring] = None):
        self.metacognition = metacognition
        self.explanations: Dict[int, Explanation] = {}
        self.next_explanation_id = 0
    
    def explain_reasoning_process(self,
                                 reasoning_steps: List[str],
                                 conclusion: str) -> Explanation:
        """
        Explain reasoning process
        
        Returns:
            Explanation
        """
        explanation_parts = ["I reached this conclusion through the following reasoning steps:"]
        
        for i, step in enumerate(reasoning_steps, 1):
            explanation_parts.append(f"{i}. {step}")
        
        explanation_parts.append(f"Therefore, I concluded: {conclusion}")
        
        explanation_text = "\n".join(explanation_parts)
        
        explanation = Explanation(
            explanation_id=self.next_explanation_id,
            target='reasoning_process',
            explanation_text=explanation_text,
            explanation_type='self',
            confidence=0.8,
            created_time=time.time()
        )
        
        self.explanations[self.next_explanation_id] = explanation
        self.next_explanation_id += 1
        
        return explanation
    
    def explain_decision_process(self,
                                decision: str,
                                factors: List[str],
                                weights: List[float] = None) -> Explanation:
        """
        Explain decision process
        
        Returns:
            Explanation
        """
        if weights is None:
            weights = [1.0 / len(factors)] * len(factors)
        
        explanation_parts = [f"I decided to {decision} based on the following factors:"]
        
        # Sort by weight
        sorted_factors = sorted(zip(factors, weights), key=lambda x: x[1], reverse=True)
        
        for factor, weight in sorted_factors:
            explanation_parts.append(f"- {factor} (weight: {weight:.2f})")
        
        explanation_text = "\n".join(explanation_parts)
        
        explanation = Explanation(
            explanation_id=self.next_explanation_id,
            target=decision,
            explanation_text=explanation_text,
            explanation_type='self',
            confidence=0.7,
            created_time=time.time()
        )
        
        self.explanations[self.next_explanation_id] = explanation
        self.next_explanation_id += 1
        
        return explanation


class DecisionExplanation:
    """
    Decision Explanation
    
    Explains specific decisions
    Provides decision rationale
    """
    
    def __init__(self):
        self.decision_explanations: List[Explanation] = []
    
    def explain_decision(self,
                        decision: str,
                        input_features: Dict[str, float],
                        decision_rule: str = None) -> Explanation:
        """
        Explain a decision
        
        Returns:
            Explanation
        """
        explanation_parts = [f"I made the decision: {decision}"]
        
        # Explain based on features
        if input_features:
            explanation_parts.append("Key factors influencing this decision:")
            sorted_features = sorted(
                input_features.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            for feature_name, value in sorted_features[:5]:  # Top 5 features
                explanation_parts.append(f"- {feature_name}: {value:.2f}")
        
        # Explain decision rule if available
        if decision_rule:
            explanation_parts.append(f"Decision rule applied: {decision_rule}")
        
        explanation_text = "\n".join(explanation_parts)
        
        explanation = Explanation(
            explanation_id=-1,
            target=decision,
            explanation_text=explanation_text,
            explanation_type='decision',
            confidence=0.7,
            created_time=time.time()
        )
        
        self.decision_explanations.append(explanation)
        return explanation


class FeatureAttributionSystem:
    """
    Feature Attribution
    
    Attributes decisions to features
    Identifies important features
    """
    
    def __init__(self):
        self.attributions: List[FeatureAttribution] = []
    
    def attribute_features(self,
                          input_features: Dict[str, float],
                          output: float,
                          baseline_output: float = 0.0) -> List[FeatureAttribution]:
        """
        Attribute output to input features
        
        Returns:
            List of feature attributions
        """
        attributions = []
        
        # Simple attribution: proportional to feature value
        total_contribution = output - baseline_output
        
        if total_contribution != 0:
            for feature_name, feature_value in input_features.items():
                # Compute contribution (simplified)
                contribution = feature_value * (total_contribution / sum(abs(v) for v in input_features.values()))
                attribution_score = abs(contribution) / (abs(total_contribution) + 1e-10)
                
                attribution = FeatureAttribution(
                    feature_name=feature_name,
                    attribution_score=attribution_score,
                    contribution=contribution
                )
                
                attributions.append(attribution)
        
        # Sort by attribution score
        attributions.sort(key=lambda x: x.attribution_score, reverse=True)
        
        self.attributions.extend(attributions)
        return attributions
    
    def get_top_features(self,
                        attributions: List[FeatureAttribution],
                        top_k: int = 5) -> List[FeatureAttribution]:
        """Get top K features"""
        return attributions[:top_k]


class CounterfactualExplanations:
    """
    Counterfactual Explanations
    
    Explains "what if" scenarios
    Shows alternative outcomes
    """
    
    def __init__(self, reasoning_system: Optional[AdvancedReasoningSystem] = None):
        self.reasoning_system = reasoning_system
        self.counterfactuals: List[Explanation] = []
    
    def generate_counterfactual(self,
                              original_decision: str,
                              original_features: Dict[str, float],
                              changed_features: Dict[str, float],
                              alternative_outcome: str) -> Explanation:
        """
        Generate counterfactual explanation
        
        Returns:
            Counterfactual explanation
        """
        explanation_parts = [
            f"Original decision: {original_decision}",
            f"Original features: {original_features}",
            "",
            "If the following features had been different:"
        ]
        
        for feature_name, new_value in changed_features.items():
            old_value = original_features.get(feature_name, 0.0)
            explanation_parts.append(
                f"- {feature_name}: {old_value:.2f} → {new_value:.2f}"
            )
        
        explanation_parts.append(f"Then the outcome would have been: {alternative_outcome}")
        
        explanation_text = "\n".join(explanation_parts)
        
        explanation = Explanation(
            explanation_id=-1,
            target='counterfactual',
            explanation_text=explanation_text,
            explanation_type='counterfactual',
            confidence=0.6,
            created_time=time.time()
        )
        
        self.counterfactuals.append(explanation)
        return explanation
    
    def explain_why_not(self,
                       decision: str,
                       alternative_decision: str,
                       reason: str) -> Explanation:
        """
        Explain why a different decision wasn't made
        
        Returns:
            Explanation
        """
        explanation_text = (
            f"I chose {decision} instead of {alternative_decision} "
            f"because: {reason}"
        )
        
        explanation = Explanation(
            explanation_id=-1,
            target=decision,
            explanation_text=explanation_text,
            explanation_type='counterfactual',
            confidence=0.7,
            created_time=time.time()
        )
        
        self.counterfactuals.append(explanation)
        return explanation


class InterpretableRepresentations:
    """
    Interpretable Representations
    
    Uses interpretable internal representations
    Makes representations human-readable
    """
    
    def __init__(self):
        self.interpretable_reps: Dict[str, Dict] = {}
    
    def create_interpretable_representation(self,
                                          representation_id: str,
                                          features: Dict[str, float],
                                          meaning: str) -> Dict:
        """
        Create interpretable representation
        
        Returns:
            Interpretable representation
        """
        rep = {
            'id': representation_id,
            'features': features,
            'meaning': meaning,
            'human_readable': self._make_human_readable(features, meaning),
            'created_time': time.time()
        }
        
        self.interpretable_reps[representation_id] = rep
        return rep
    
    def _make_human_readable(self,
                            features: Dict[str, float],
                            meaning: str) -> str:
        """Make representation human-readable"""
        feature_descriptions = []
        
        for feature_name, value in features.items():
            if abs(value) > 0.1:
                direction = "high" if value > 0 else "low"
                feature_descriptions.append(f"{feature_name} is {direction}")
        
        if feature_descriptions:
            return f"{meaning}: {', '.join(feature_descriptions)}"
        else:
            return meaning
    
    def interpret_representation(self, representation_id: str) -> Optional[str]:
        """Interpret a representation"""
        if representation_id in self.interpretable_reps:
            return self.interpretable_reps[representation_id]['human_readable']
        return None


class ExplainabilitySystem:
    """
    Explainability System Manager
    
    Integrates all explainability components
    """
    
    def __init__(self,
                 brain_system=None,
                 metacognition: Optional[MetacognitiveMonitoring] = None,
                 reasoning_system: Optional[AdvancedReasoningSystem] = None,
                 global_workspace: Optional[GlobalWorkspace] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.self_explanation = SelfExplanation(metacognition)
        self.decision_explanation = DecisionExplanation()
        self.feature_attribution = FeatureAttributionSystem()
        self.counterfactual_explanations = CounterfactualExplanations(reasoning_system)
        self.interpretable_reps = InterpretableRepresentations()
        
        # Integration with existing systems
        self.metacognition = metacognition
        self.reasoning_system = reasoning_system
        self.global_workspace = global_workspace
        
        # Statistics
        self.stats = {
            'self_explanations_generated': 0,
            'decision_explanations_generated': 0,
            'feature_attributions_computed': 0,
            'counterfactuals_generated': 0,
            'interpretable_reps_created': 0
        }
    
    def explain_reasoning(self,
                         reasoning_steps: List[str],
                         conclusion: str) -> Explanation:
        """Explain reasoning process"""
        explanation = self.self_explanation.explain_reasoning_process(
            reasoning_steps, conclusion
        )
        self.stats['self_explanations_generated'] += 1
        return explanation
    
    def explain_decision(self,
                        decision: str,
                        input_features: Dict[str, float]) -> Explanation:
        """Explain a decision"""
        explanation = self.decision_explanation.explain_decision(
            decision, input_features
        )
        self.stats['decision_explanations_generated'] += 1
        return explanation
    
    def attribute_features(self,
                          input_features: Dict[str, float],
                          output: float) -> List[FeatureAttribution]:
        """Attribute output to features"""
        attributions = self.feature_attribution.attribute_features(
            input_features, output
        )
        self.stats['feature_attributions_computed'] += len(attributions)
        return attributions
    
    def generate_counterfactual(self,
                              original_decision: str,
                              changed_features: Dict[str, float],
                              alternative_outcome: str) -> Explanation:
        """Generate counterfactual explanation"""
        explanation = self.counterfactual_explanations.generate_counterfactual(
            original_decision, {}, changed_features, alternative_outcome
        )
        self.stats['counterfactuals_generated'] += 1
        return explanation
    
    def create_interpretable_representation(self,
                                          representation_id: str,
                                          features: Dict[str, float],
                                          meaning: str) -> Dict:
        """Create interpretable representation"""
        rep = self.interpretable_reps.create_interpretable_representation(
            representation_id, features, meaning
        )
        self.stats['interpretable_reps_created'] += 1
        return rep
    
    def get_statistics(self) -> Dict:
        """Get explainability statistics"""
        return self.stats.copy()

