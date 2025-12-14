#!/usr/bin/env python3
"""
Test suite for Phase 8: Advanced Reasoning
"""

import numpy as np
import sys
import os

# Add Phase8_AdvancedReasoning to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Phase8_AdvancedReasoning'))

from mathematical_reasoning import MathematicalReasoningSystem
from scientific_discovery import ScientificDiscoverySystem
from probabilistic_causal_reasoning import ProbabilisticCausalReasoningSystem


def test_mathematical_reasoning():
    """Test mathematical reasoning"""
    print("Testing Mathematical Reasoning...")
    
    math_system = MathematicalReasoningSystem()
    
    # Test expression processing
    expr = math_system.process_expression("x + y = z")
    assert expr is not None, "Should process expression"
    
    # Test theorem proving
    result = math_system.prove_theorem(
        statement="If A then B",
        premises=["A -> B", "A"],
        conclusion="B"
    )
    assert result['success'], "Should prove simple theorem"
    
    print("  ✓ Mathematical reasoning works")
    return True


def test_scientific_discovery():
    """Test scientific discovery"""
    print("Testing Scientific Discovery...")
    
    discovery_system = ScientificDiscoverySystem()
    
    # Test hypothesis generation
    observations = [
        {'variable1': 1.0, 'variable2': 2.0},
        {'variable1': 2.0, 'variable2': 4.0}
    ]
    result = discovery_system.discover_scientific_relationship(
        observations, ['variable1', 'variable2']
    )
    assert result['success'], "Should discover relationship"
    assert result['hypothesis'] is not None, "Should generate hypothesis"
    
    print("  ✓ Scientific discovery works")
    return True


def test_probabilistic_reasoning():
    """Test probabilistic reasoning"""
    print("Testing Probabilistic & Causal Reasoning...")
    
    prob_system = ProbabilisticCausalReasoningSystem()
    
    # Test belief update
    belief = prob_system.update_belief_with_evidence(
        "It will rain",
        {'likelihood': 0.8, 'probability': 0.6}
    )
    assert belief is not None, "Should update belief"
    
    # Test uncertainty quantification
    uncertainty = prob_system.quantify_prediction_uncertainty(0.7, 0.8)
    assert 'uncertainty_level' in uncertainty, "Should quantify uncertainty"
    
    # Test causal structure learning
    data = {
        'cause': np.array([1, 2, 3, 4, 5]),
        'effect': np.array([2, 4, 6, 8, 10])
    }
    graph = prob_system.learn_causal_structure_from_data(data)
    assert len(graph.nodes) > 0, "Should learn causal structure"
    
    print("  ✓ Probabilistic reasoning works")
    return True


def run_all_tests():
    """Run all Phase 8 tests"""
    print("=" * 60)
    print("Testing Phase 8: Advanced Reasoning")
    print("=" * 60)
    
    tests = [
        test_mathematical_reasoning,
        test_scientific_discovery,
        test_probabilistic_reasoning
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

