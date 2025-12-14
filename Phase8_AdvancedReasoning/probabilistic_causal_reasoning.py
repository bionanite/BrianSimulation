#!/usr/bin/env python3
"""
Probabilistic & Causal Reasoning - Phase 8.3
Implements Bayesian inference, uncertainty quantification, causal structure learning,
counterfactual reasoning, and confidence calibration
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from collections import defaultdict

# Import dependencies
try:
    from world_models import WorldModelManager
    from semantic_representations import SemanticNetwork
    from metacognition import MetacognitiveMonitoring
except ImportError:
    WorldModelManager = None
    SemanticNetwork = None
    MetacognitiveMonitoring = None


@dataclass
class Belief:
    """Represents a probabilistic belief"""
    belief_id: int
    proposition: str
    probability: float
    confidence: float = 0.5
    evidence: List[Dict] = field(default_factory=list)


@dataclass
class CausalGraph:
    """Represents a causal graph"""
    nodes: List[str]
    edges: List[Tuple[str, str, float]]  # (from, to, strength)
    probabilities: Dict[str, Dict[str, float]]  # node -> {value: probability}


class BayesianInference:
    """
    Bayesian Inference
    
    Updates beliefs with evidence
    Performs probabilistic reasoning
    """
    
    def __init__(self):
        self.beliefs: Dict[int, Belief] = {}
        self.next_belief_id = 0
    
    def update_belief(self,
                     belief: Belief,
                     evidence: Dict[str, float]) -> Belief:
        """
        Update belief using Bayesian inference
        
        Returns:
            Updated belief
        """
        # Simplified Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)
        prior = belief.probability
        
        # Compute likelihood (simplified)
        likelihood = evidence.get('likelihood', 0.5)
        evidence_prob = evidence.get('probability', 0.5)
        
        # Bayesian update
        if evidence_prob > 0:
            posterior = (likelihood * prior) / evidence_prob
        else:
            posterior = prior
        
        # Ensure valid probability
        posterior = max(0.0, min(1.0, posterior))
        
        belief.probability = posterior
        belief.evidence.append(evidence)
        
        # Update confidence based on amount of evidence
        belief.confidence = min(1.0, 0.5 + len(belief.evidence) * 0.1)
        
        return belief
    
    def create_belief(self,
                     proposition: str,
                     prior_probability: float = 0.5) -> Belief:
        """Create a new belief"""
        belief = Belief(
            belief_id=self.next_belief_id,
            proposition=proposition,
            probability=prior_probability,
            confidence=0.5
        )
        
        self.next_belief_id += 1
        self.beliefs[belief.belief_id] = belief
        
        return belief


class UncertaintyQuantification:
    """
    Uncertainty Quantification
    
    Quantifies uncertainty in predictions
    Tracks prediction confidence
    """
    
    def __init__(self):
        self.uncertainty_history: List[Dict] = []
    
    def quantify_uncertainty(self,
                          prediction: float,
                          confidence: float,
                          variance: Optional[float] = None) -> Dict:
        """
        Quantify uncertainty in a prediction
        
        Returns:
            Uncertainty metrics
        """
        if variance is None:
            # Estimate variance from confidence
            variance = (1.0 - confidence) * 0.25
        
        std_dev = np.sqrt(variance)
        
        # Compute uncertainty intervals
        lower_bound = prediction - 1.96 * std_dev  # 95% interval
        upper_bound = prediction + 1.96 * std_dev
        
        uncertainty = {
            'prediction': prediction,
            'confidence': confidence,
            'variance': variance,
            'std_dev': std_dev,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'uncertainty_level': 1.0 - confidence
        }
        
        self.uncertainty_history.append(uncertainty)
        return uncertainty
    
    def compute_epistemic_uncertainty(self,
                                     model_confidence: float,
                                     data_quality: float) -> float:
        """
        Compute epistemic uncertainty (uncertainty about the model)
        
        Returns:
            Epistemic uncertainty score
        """
        # Epistemic uncertainty increases with low model confidence and low data quality
        epistemic = (1.0 - model_confidence) * 0.6 + (1.0 - data_quality) * 0.4
        return epistemic
    
    def compute_aleatoric_uncertainty(self,
                                     prediction_variance: float) -> float:
        """
        Compute aleatoric uncertainty (inherent randomness)
        
        Returns:
            Aleatoric uncertainty score
        """
        # Aleatoric uncertainty is related to prediction variance
        aleatoric = min(1.0, prediction_variance * 2.0)
        return aleatoric


class CausalStructureLearning:
    """
    Causal Structure Learning
    
    Learns causal graphs from data
    Discovers causal relationships
    """
    
    def __init__(self,
                 correlation_threshold: float = 0.6,
                 temporal_threshold: float = 0.5):
        self.correlation_threshold = correlation_threshold
        self.temporal_threshold = temporal_threshold
        self.causal_graphs: List[CausalGraph] = []
    
    def learn_causal_structure(self,
                              data: Dict[str, np.ndarray],
                              temporal_order: Optional[List[str]] = None) -> CausalGraph:
        """
        Learn causal structure from data
        
        Returns:
            Causal graph
        """
        variables = list(data.keys())
        edges = []
        probabilities = {}
        
        # Learn causal relationships
        for i, var1 in enumerate(variables):
            var1_probs = {}
            var1_values = data[var1]
            
            # Compute probability distribution
            unique_values, counts = np.unique(var1_values, return_counts=True)
            probs = counts / len(var1_values)
            var1_probs = {str(val): prob for val, prob in zip(unique_values, probs)}
            probabilities[var1] = var1_probs
            
            # Find causes (variables that precede var1)
            for var2 in variables:
                if var1 == var2:
                    continue
                
                # Check temporal order
                if temporal_order:
                    var1_idx = temporal_order.index(var1) if var1 in temporal_order else len(temporal_order)
                    var2_idx = temporal_order.index(var2) if var2 in temporal_order else len(temporal_order)
                    if var2_idx >= var1_idx:
                        continue  # var2 doesn't precede var1
                
                # Compute correlation
                if len(data[var1]) == len(data[var2]):
                    correlation = np.corrcoef(data[var1], data[var2])[0, 1]
                    
                    if abs(correlation) > self.correlation_threshold:
                        # Determine direction (simplified)
                        # In practice would use more sophisticated methods
                        if temporal_order or correlation > 0:
                            edges.append((var2, var1, abs(correlation)))
        
        graph = CausalGraph(
            nodes=variables,
            edges=edges,
            probabilities=probabilities
        )
        
        self.causal_graphs.append(graph)
        return graph
    
    def infer_cause(self,
                   effect: str,
                   graph: CausalGraph) -> List[Tuple[str, float]]:
        """
        Infer causes of an effect
        
        Returns:
            List of (cause, strength) tuples
        """
        causes = []
        for from_node, to_node, strength in graph.edges:
            if to_node == effect:
                causes.append((from_node, strength))
        
        # Sort by strength
        causes.sort(key=lambda x: x[1], reverse=True)
        return causes


class CounterfactualReasoning:
    """
    Counterfactual Reasoning
    
    Reasons about "what if" scenarios
    Explores alternative possibilities
    """
    
    def __init__(self):
        self.counterfactuals: List[Dict] = []
    
    def reason_counterfactual(self,
                            actual_state: Dict[str, float],
                            changed_variable: str,
                            new_value: float,
                            causal_graph: CausalGraph) -> Dict[str, float]:
        """
        Reason about counterfactual scenario
        
        Returns:
            Counterfactual state
        """
        counterfactual_state = actual_state.copy()
        counterfactual_state[changed_variable] = new_value
        
        # Propagate effects through causal graph
        affected_variables = {changed_variable}
        queue = [changed_variable]
        
        while queue:
            current_var = queue.pop(0)
            
            # Find variables affected by current_var
            for from_node, to_node, strength in causal_graph.edges:
                if from_node == current_var and to_node not in affected_variables:
                    affected_variables.add(to_node)
                    queue.append(to_node)
                    
                    # Update counterfactual value (simplified)
                    if to_node in counterfactual_state:
                        # Simple linear propagation
                        change = (new_value - actual_state[changed_variable]) * strength
                        counterfactual_state[to_node] = actual_state[to_node] + change
        
        counterfactual = {
            'actual_state': actual_state,
            'changed_variable': changed_variable,
            'new_value': new_value,
            'counterfactual_state': counterfactual_state,
            'affected_variables': list(affected_variables),
            'timestamp': time.time()
        }
        
        self.counterfactuals.append(counterfactual)
        return counterfactual_state
    
    def compute_counterfactual_probability(self,
                                         counterfactual_state: Dict[str, float],
                                         causal_graph: CausalGraph) -> float:
        """
        Compute probability of counterfactual scenario
        
        Returns:
            Probability score
        """
        # Compute joint probability (simplified)
        probability = 1.0
        
        for variable, value in counterfactual_state.items():
            if variable in causal_graph.probabilities:
                probs = causal_graph.probabilities[variable]
                value_str = str(value)
                
                # Find closest probability
                if value_str in probs:
                    prob = probs[value_str]
                else:
                    # Use average probability
                    prob = np.mean(list(probs.values())) if probs else 0.5
                
                probability *= prob
        
        return probability


class ConfidenceCalibration:
    """
    Confidence Calibration
    
    Calibrates confidence with accuracy
    Ensures confidence matches actual performance
    """
    
    def __init__(self):
        self.calibration_history: List[Tuple[float, bool]] = []  # (confidence, correct)
        self.calibration_curve: Dict[float, float] = {}
    
    def update_calibration(self,
                         confidence: float,
                         correct: bool):
        """Update calibration with prediction result"""
        self.calibration_history.append((confidence, correct))
        
        # Limit history
        if len(self.calibration_history) > 1000:
            self.calibration_history = self.calibration_history[-1000:]
    
    def compute_calibration_error(self) -> float:
        """
        Compute calibration error
        
        Returns:
            Expected Calibration Error (ECE)
        """
        if len(self.calibration_history) < 10:
            return 0.0
        
        # Bin confidences
        bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            bin_low = bins[i]
            bin_high = bins[i + 1]
            
            bin_items = [(conf, corr) for conf, corr in self.calibration_history
                        if bin_low <= conf < bin_high]
            
            if bin_items:
                bin_conf = np.mean([conf for conf, _ in bin_items])
                bin_acc = np.mean([1.0 if corr else 0.0 for _, corr in bin_items])
                bin_count = len(bin_items)
                
                bin_confidences.append(bin_conf)
                bin_accuracies.append(bin_acc)
                bin_counts.append(bin_count)
        
        if not bin_confidences:
            return 0.0
        
        # Compute ECE
        total = sum(bin_counts)
        ece = sum(abs(acc - conf) * count / total
                 for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts))
        
        return ece
    
    def calibrate_confidence(self,
                            raw_confidence: float) -> float:
        """
        Calibrate raw confidence score
        
        Returns:
            Calibrated confidence
        """
        # Simple calibration based on history
        if len(self.calibration_history) < 10:
            return raw_confidence
        
        # Find similar confidence levels
        similar = [(conf, corr) for conf, corr in self.calibration_history
                  if abs(conf - raw_confidence) < 0.1]
        
        if similar:
            actual_accuracy = np.mean([1.0 if corr else 0.0 for _, corr in similar])
            # Adjust confidence towards actual accuracy
            calibrated = 0.7 * raw_confidence + 0.3 * actual_accuracy
        else:
            calibrated = raw_confidence
        
        return max(0.0, min(1.0, calibrated))


class ProbabilisticCausalReasoningSystem:
    """
    Probabilistic & Causal Reasoning System Manager
    
    Integrates all probabilistic and causal reasoning components
    """
    
    def __init__(self,
                 brain_system=None,
                 world_model: Optional[WorldModelManager] = None,
                 semantic_network: Optional[SemanticNetwork] = None,
                 metacognition: Optional[MetacognitiveMonitoring] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.bayesian_inference = BayesianInference()
        self.uncertainty_quantification = UncertaintyQuantification()
        self.causal_structure_learning = CausalStructureLearning()
        self.counterfactual_reasoning = CounterfactualReasoning()
        self.confidence_calibration = ConfidenceCalibration()
        
        # Integration with existing systems
        self.world_model = world_model
        self.semantic_network = semantic_network
        self.metacognition = metacognition
        
        # Statistics
        self.stats = {
            'beliefs_updated': 0,
            'uncertainties_quantified': 0,
            'causal_graphs_learned': 0,
            'counterfactuals_reasoned': 0,
            'average_calibration_error': 0.0
        }
    
    def update_belief_with_evidence(self,
                                   proposition: str,
                                   evidence: Dict[str, float]) -> Belief:
        """Update belief with new evidence"""
        # Find or create belief
        belief = None
        for b in self.bayesian_inference.beliefs.values():
            if b.proposition == proposition:
                belief = b
                break
        
        if belief is None:
            belief = self.bayesian_inference.create_belief(proposition)
        
        # Update with evidence
        updated = self.bayesian_inference.update_belief(belief, evidence)
        self.stats['beliefs_updated'] += 1
        
        return updated
    
    def quantify_prediction_uncertainty(self,
                                      prediction: float,
                                      confidence: float) -> Dict:
        """Quantify uncertainty in a prediction"""
        uncertainty = self.uncertainty_quantification.quantify_uncertainty(
            prediction, confidence
        )
        self.stats['uncertainties_quantified'] += 1
        return uncertainty
    
    def learn_causal_structure_from_data(self,
                                       data: Dict[str, np.ndarray]) -> CausalGraph:
        """Learn causal structure from data"""
        graph = self.causal_structure_learning.learn_causal_structure(data)
        self.stats['causal_graphs_learned'] += 1
        return graph
    
    def reason_about_counterfactual(self,
                                  actual_state: Dict[str, float],
                                  changed_variable: str,
                                  new_value: float,
                                  causal_graph: CausalGraph) -> Dict[str, float]:
        """Reason about counterfactual scenario"""
        counterfactual = self.counterfactual_reasoning.reason_counterfactual(
            actual_state, changed_variable, new_value, causal_graph
        )
        self.stats['counterfactuals_reasoned'] += 1
        return counterfactual
    
    def calibrate_prediction_confidence(self,
                                      raw_confidence: float,
                                      was_correct: bool) -> float:
        """Calibrate prediction confidence"""
        self.confidence_calibration.update_calibration(raw_confidence, was_correct)
        calibrated = self.confidence_calibration.calibrate_confidence(raw_confidence)
        
        # Update calibration error
        ece = self.confidence_calibration.compute_calibration_error()
        self.stats['average_calibration_error'] = ece
        
        return calibrated
    
    def get_statistics(self) -> Dict:
        """Get probabilistic reasoning statistics"""
        return self.stats.copy()

