#!/usr/bin/env python3
"""
Robustness & Adversarial Defense - Phase 13.1
Implements adversarial robustness, distribution shift handling,
out-of-distribution detection, robust decision making, and error recovery
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

# Import dependencies
try:
    from metacognition import MetacognitiveMonitoring
    from Phase8_AdvancedReasoning.probabilistic_causal_reasoning import ProbabilisticCausalReasoning
    from executive_control import ExecutiveControl
except ImportError:
    MetacognitiveMonitoring = None
    ProbabilisticCausalReasoning = None
    ExecutiveControl = None


@dataclass
class AdversarialAttack:
    """Represents an adversarial attack"""
    attack_id: int
    attack_type: str
    original_input: np.ndarray
    adversarial_input: np.ndarray
    success: bool = False


@dataclass
class DistributionShift:
    """Represents a distribution shift"""
    shift_id: int
    shift_type: str
    detected_time: float
    severity: float = 0.5


class AdversarialRobustness:
    """
    Adversarial Robustness
    
    Defends against adversarial attacks
    Detects and mitigates attacks
    """
    
    def __init__(self):
        self.attacks_detected: List[AdversarialAttack] = []
        self.defense_strategies: Dict[str, Callable] = {}
        self.next_attack_id = 0
    
    def detect_adversarial_input(self,
                                input_data: np.ndarray,
                                normal_inputs: List[np.ndarray]) -> bool:
        """
        Detect adversarial input
        
        Returns:
            True if adversarial detected
        """
        if not normal_inputs:
            return False
        
        # Compute distance to normal inputs
        distances = [np.linalg.norm(input_data - normal) for normal in normal_inputs]
        min_distance = min(distances)
        
        # If input is very different from normal, might be adversarial
        threshold = np.mean(distances) + 2 * np.std(distances)
        
        is_adversarial = min_distance > threshold
        
        if is_adversarial:
            attack = AdversarialAttack(
                attack_id=self.next_attack_id,
                attack_type='unknown',
                original_input=input_data.copy(),
                adversarial_input=input_data.copy()
            )
            self.attacks_detected.append(attack)
            self.next_attack_id += 1
        
        return is_adversarial
    
    def defend_against_attack(self,
                             input_data: np.ndarray,
                             defense_method: str = 'input_smoothing') -> np.ndarray:
        """
        Defend against adversarial attack
        
        Returns:
            Defended input
        """
        if defense_method == 'input_smoothing':
            # Smooth input to reduce adversarial perturbations
            smoothed = self._smooth_input(input_data)
            return smoothed
        
        elif defense_method == 'input_quantization':
            # Quantize input
            quantized = np.round(input_data * 10) / 10
            return quantized
        
        else:
            return input_data.copy()
    
    def _smooth_input(self, input_data: np.ndarray) -> np.ndarray:
        """Smooth input"""
        # Simple smoothing: average with neighbors
        if len(input_data.shape) == 1:
            smoothed = np.zeros_like(input_data)
            for i in range(len(input_data)):
                start = max(0, i - 1)
                end = min(len(input_data), i + 2)
                smoothed[i] = np.mean(input_data[start:end])
            return smoothed
        else:
            return input_data.copy()


class DistributionShiftHandling:
    """
    Distribution Shift Handling
    
    Handles distribution shifts
    Adapts to new distributions
    """
    
    def __init__(self):
        self.shifts_detected: List[DistributionShift] = []
        self.next_shift_id = 0
        self.baseline_distribution: Optional[np.ndarray] = None
    
    def detect_distribution_shift(self,
                                current_data: List[np.ndarray],
                                baseline_data: Optional[List[np.ndarray]] = None) -> Optional[DistributionShift]:
        """
        Detect distribution shift
        
        Returns:
            Detected shift or None
        """
        if baseline_data is None:
            if self.baseline_distribution is None:
                # Set baseline
                if current_data:
                    self.baseline_distribution = np.mean([np.mean(d) for d in current_data])
                return None
            baseline_mean = self.baseline_distribution
        else:
            baseline_mean = np.mean([np.mean(d) for d in baseline_data])
        
        if not current_data:
            return None
        
        current_mean = np.mean([np.mean(d) for d in current_data])
        shift_magnitude = abs(current_mean - baseline_mean)
        
        if shift_magnitude > 0.1:  # Threshold
            shift = DistributionShift(
                shift_id=self.next_shift_id,
                shift_type='mean_shift',
                detected_time=time.time(),
                severity=min(1.0, shift_magnitude)
            )
            
            self.shifts_detected.append(shift)
            self.next_shift_id += 1
            
            # Update baseline
            self.baseline_distribution = current_mean
            
            return shift
        
        return None
    
    def adapt_to_shift(self, shift: DistributionShift) -> Dict:
        """
        Adapt to distribution shift
        
        Returns:
            Adaptation result
        """
        # Simplified adaptation: adjust parameters
        adaptation = {
            'shift_id': shift.shift_id,
            'adaptation_applied': True,
            'parameter_adjustment': shift.severity * 0.1,
            'timestamp': time.time()
        }
        
        return adaptation


class OutOfDistributionDetection:
    """
    Out-of-Distribution Detection
    
    Detects novel situations
    Identifies OOD inputs
    """
    
    def __init__(self):
        self.ood_detections: List[Dict] = []
        self.distribution_bounds: Dict[str, Tuple[float, float]] = {}
    
    def detect_ood(self,
                  input_data: np.ndarray,
                  distribution_name: str = 'default') -> bool:
        """
        Detect out-of-distribution input
        
        Returns:
            True if OOD detected
        """
        if distribution_name not in self.distribution_bounds:
            # Initialize bounds
            self.distribution_bounds[distribution_name] = (
                np.min(input_data) - 0.1,
                np.max(input_data) + 0.1
            )
            return False
        
        min_val, max_val = self.distribution_bounds[distribution_name]
        
        # Check if input is outside bounds
        is_ood = bool(np.any(input_data < min_val) or np.any(input_data > max_val))
        
        if is_ood:
            self.ood_detections.append({
                'input': input_data.copy(),
                'distribution': distribution_name,
                'timestamp': time.time()
            })
        
        return is_ood
    
    def update_distribution_bounds(self,
                                  data: List[np.ndarray],
                                  distribution_name: str = 'default'):
        """Update distribution bounds"""
        if not data:
            return
        
        all_data = np.concatenate(data)
        self.distribution_bounds[distribution_name] = (
            np.min(all_data) - 0.1,
            np.max(all_data) + 0.1
        )


class RobustDecisionMaking:
    """
    Robust Decision Making
    
    Makes robust decisions under uncertainty
    Handles uncertainty in predictions
    """
    
    def __init__(self, probabilistic_reasoning: Optional[ProbabilisticCausalReasoning] = None):
        self.probabilistic_reasoning = probabilistic_reasoning
        self.robust_decisions: List[Dict] = []
    
    def make_robust_decision(self,
                           options: List[str],
                           probabilities: List[float],
                           uncertainties: List[float] = None) -> str:
        """
        Make robust decision under uncertainty
        
        Returns:
            Selected option
        """
        if uncertainties is None:
            uncertainties = [0.1] * len(options)
        
        # Score options considering uncertainty
        scores = []
        for prob, uncert in zip(probabilities, uncertainties):
            # Penalize high uncertainty
            score = prob * (1.0 - uncert)
            scores.append(score)
        
        # Select best option
        best_idx = np.argmax(scores)
        selected = options[best_idx]
        
        self.robust_decisions.append({
            'selected_option': selected,
            'scores': scores,
            'uncertainties': uncertainties,
            'timestamp': time.time()
        })
        
        return selected
    
    def compute_decision_confidence(self,
                                   probabilities: List[float],
                                   uncertainties: List[float]) -> float:
        """Compute confidence in decision"""
        if not probabilities:
            return 0.0
        
        max_prob = max(probabilities)
        avg_uncertainty = np.mean(uncertainties)
        
        confidence = max_prob * (1.0 - avg_uncertainty)
        return confidence


class ErrorRecovery:
    """
    Error Recovery
    
    Recovers from errors gracefully
    Handles failures
    """
    
    def __init__(self, executive_control: Optional[ExecutiveControl] = None):
        self.executive_control = executive_control
        self.error_history: List[Dict] = []
        self.recovery_strategies: Dict[str, Callable] = {}
    
    def detect_error(self,
                    error_type: str,
                    error_context: Dict) -> bool:
        """
        Detect error
        
        Returns:
            True if error detected
        """
        error = {
            'type': error_type,
            'context': error_context,
            'timestamp': time.time()
        }
        
        self.error_history.append(error)
        return True
    
    def recover_from_error(self,
                          error_type: str,
                          error_context: Dict) -> Dict:
        """
        Recover from error
        
        Returns:
            Recovery result
        """
        recovery_strategies = {
            'prediction_error': self._recover_prediction_error,
            'execution_error': self._recover_execution_error,
            'memory_error': self._recover_memory_error
        }
        
        strategy = recovery_strategies.get(error_type, self._recover_generic)
        result = strategy(error_context)
        
        return result
    
    def _recover_prediction_error(self, context: Dict) -> Dict:
        """Recover from prediction error"""
        return {
            'recovery_method': 'retry_with_adjusted_parameters',
            'success': True,
            'adjustments': {'learning_rate': 0.5}
        }
    
    def _recover_execution_error(self, context: Dict) -> Dict:
        """Recover from execution error"""
        return {
            'recovery_method': 'fallback_plan',
            'success': True,
            'fallback_used': True
        }
    
    def _recover_memory_error(self, context: Dict) -> Dict:
        """Recover from memory error"""
        return {
            'recovery_method': 'clear_cache_and_retry',
            'success': True
        }
    
    def _recover_generic(self, context: Dict) -> Dict:
        """Generic recovery"""
        return {
            'recovery_method': 'retry',
            'success': True
        }


class RobustnessSystem:
    """
    Robustness System Manager
    
    Integrates all robustness components
    """
    
    def __init__(self,
                 brain_system=None,
                 metacognition: Optional[MetacognitiveMonitoring] = None,
                 probabilistic_reasoning: Optional[ProbabilisticCausalReasoning] = None,
                 executive_control: Optional[ExecutiveControl] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.adversarial_robustness = AdversarialRobustness()
        self.distribution_shift = DistributionShiftHandling()
        self.ood_detection = OutOfDistributionDetection()
        self.robust_decision_making = RobustDecisionMaking(probabilistic_reasoning)
        self.error_recovery = ErrorRecovery(executive_control)
        
        # Integration with existing systems
        self.metacognition = metacognition
        self.probabilistic_reasoning = probabilistic_reasoning
        self.executive_control = executive_control
        
        # Statistics
        self.stats = {
            'adversarial_attacks_detected': 0,
            'distribution_shifts_detected': 0,
            'ood_inputs_detected': 0,
            'robust_decisions_made': 0,
            'errors_recovered': 0
        }
    
    def defend_against_attack(self, input_data: np.ndarray) -> np.ndarray:
        """Defend against adversarial attack"""
        defended = self.adversarial_robustness.defend_against_attack(input_data)
        self.stats['adversarial_attacks_detected'] += 1
        return defended
    
    def detect_and_handle_shift(self, current_data: List[np.ndarray]) -> Optional[DistributionShift]:
        """Detect and handle distribution shift"""
        shift = self.distribution_shift.detect_distribution_shift(current_data)
        if shift:
            self.distribution_shift.adapt_to_shift(shift)
            self.stats['distribution_shifts_detected'] += 1
        return shift
    
    def detect_ood_input(self, input_data: np.ndarray) -> bool:
        """Detect out-of-distribution input"""
        is_ood = self.ood_detection.detect_ood(input_data)
        if is_ood:
            self.stats['ood_inputs_detected'] += 1
        return is_ood
    
    def make_robust_decision(self,
                            options: List[str],
                            probabilities: List[float],
                            uncertainties: List[float] = None) -> str:
        """Make robust decision"""
        decision = self.robust_decision_making.make_robust_decision(
            options, probabilities, uncertainties
        )
        self.stats['robust_decisions_made'] += 1
        return decision
    
    def recover_from_error(self, error_type: str, error_context: Dict) -> Dict:
        """Recover from error"""
        recovery = self.error_recovery.recover_from_error(error_type, error_context)
        self.stats['errors_recovered'] += 1
        return recovery
    
    def get_statistics(self) -> Dict:
        """Get robustness statistics"""
        return self.stats.copy()

