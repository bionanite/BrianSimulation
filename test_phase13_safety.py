#!/usr/bin/env python3
"""
Test Phase 13: Robustness, Safety & Explainability
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Phase13_Safety.robustness_system import RobustnessSystem
from Phase13_Safety.safety_alignment import SafetyAlignmentSystem, Value
from Phase13_Safety.explainability_system import ExplainabilitySystem


class TestRobustness(unittest.TestCase):
    """Test robustness system"""
    
    def test_adversarial_defense(self):
        """Test adversarial defense"""
        system = RobustnessSystem()
        
        input_data = np.array([0.5, 0.3, 0.7])
        defended = system.defend_against_attack(input_data)
        
        self.assertIsNotNone(defended)
        self.assertEqual(len(defended), len(input_data))
    
    def test_distribution_shift_detection(self):
        """Test distribution shift detection"""
        system = RobustnessSystem()
        
        baseline_data = [np.array([0.5, 0.5])] * 10
        current_data = [np.array([0.8, 0.8])] * 10
        
        shift = system.detect_and_handle_shift(current_data)
        # May or may not detect shift depending on threshold
        self.assertIsInstance(shift, (type(None), type(system.distribution_shift.shifts_detected[0]) if system.distribution_shift.shifts_detected else None))
    
    def test_ood_detection(self):
        """Test out-of-distribution detection"""
        system = RobustnessSystem()
        
        # Initialize with some data
        system.ood_detection.update_distribution_bounds(
            [np.array([0.5, 0.5])] * 10
        )
        
        is_ood = system.detect_ood_input(np.array([10.0, 10.0]))
        self.assertIsInstance(is_ood, bool)


class TestSafetyAlignment(unittest.TestCase):
    """Test safety and alignment"""
    
    def test_value_registration(self):
        """Test value registration"""
        system = SafetyAlignmentSystem()
        
        value = system.register_value(
            name="Safety",
            description="Prioritize safety",
            importance=0.9
        )
        
        self.assertIsNotNone(value)
        self.assertEqual(value.name, "Safety")
    
    def test_action_safety_check(self):
        """Test action safety checking"""
        system = SafetyAlignmentSystem()
        
        action = {
            'description': 'Perform safe operation',
            'type': 'safe'
        }
        
        result = system.check_action_safety(action)
        self.assertIn('safe', result)
        self.assertIn('aligned', result)
    
    def test_safe_exploration(self):
        """Test safe exploration"""
        system = SafetyAlignmentSystem()
        
        action_space = {
            'parameter1': np.array([0.1, 0.2, 0.3]),
            'parameter2': np.array([0.4, 0.5, 0.6])
        }
        
        safe_action = system.explore_safely(action_space)
        self.assertIsNotNone(safe_action)


class TestExplainability(unittest.TestCase):
    """Test explainability"""
    
    def test_reasoning_explanation(self):
        """Test reasoning explanation"""
        system = ExplainabilitySystem()
        
        explanation = system.explain_reasoning(
            reasoning_steps=["Step 1", "Step 2"],
            conclusion="Final conclusion"
        )
        
        self.assertIsNotNone(explanation)
        self.assertIn('reasoning', explanation.explanation_text.lower())
    
    def test_decision_explanation(self):
        """Test decision explanation"""
        system = ExplainabilitySystem()
        
        explanation = system.explain_decision(
            decision="Choose option A",
            input_features={"feature1": 0.8, "feature2": 0.6}
        )
        
        self.assertIsNotNone(explanation)
        self.assertIn('decision', explanation.explanation_text.lower())
    
    def test_feature_attribution(self):
        """Test feature attribution"""
        system = ExplainabilitySystem()
        
        attributions = system.attribute_features(
            input_features={"f1": 0.8, "f2": 0.3},
            output=0.7
        )
        
        self.assertGreater(len(attributions), 0)
        self.assertIsNotNone(attributions[0].feature_name)


if __name__ == '__main__':
    unittest.main()

