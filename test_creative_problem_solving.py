#!/usr/bin/env python3
"""
Test suite for Creative Problem Solving (Phase 6.2)
"""

import numpy as np
import sys
import os

# Add Phase6_Creativity to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Phase6_Creativity'))

from creative_problem_solving import (
    CreativeProblemSolving, Problem, Solution,
    AnalogicalReasoning, ConstraintRelaxation, ProblemReframing,
    SolutionSynthesis, CreativeEvaluation
)


def test_analogical_reasoning():
    """Test analogical reasoning"""
    print("Testing Analogical Reasoning...")
    
    reasoning = AnalogicalReasoning()
    
    # Create test problems
    problem1 = Problem(
        problem_id=0,
        description="Problem 1",
        initial_state=np.array([1.0, 0.0, 0.5]),
        goal_state=np.array([0.0, 1.0, 0.5]),
        constraints=[],
        domain="test"
    )
    
    problem2 = Problem(
        problem_id=1,
        description="Problem 2",
        initial_state=np.array([0.9, 0.1, 0.5]),
        goal_state=np.array([0.1, 0.9, 0.5]),
        constraints=[],
        domain="test"
    )
    
    problem_database = {0: problem1, 1: problem2}
    
    # Find analogous problems
    analogous = reasoning.find_analogous_problems(problem1, problem_database)
    assert len(analogous) > 0, "Should find analogous problems"
    
    print("  ✓ Analogical reasoning works")
    return True


def test_constraint_relaxation():
    """Test constraint relaxation"""
    print("Testing Constraint Relaxation...")
    
    relaxation = ConstraintRelaxation()
    
    problem = Problem(
        problem_id=0,
        description="Test problem",
        initial_state=np.array([1.0, 0.0]),
        goal_state=np.array([0.0, 1.0]),
        constraints=[{'threshold': 0.8, 'type': 'max'}],
        domain="test"
    )
    
    # Relax constraints
    relaxed = relaxation.relax_constraints(problem)
    assert len(relaxed) > 0, "Should create relaxed problems"
    
    print("  ✓ Constraint relaxation works")
    return True


def test_problem_reframing():
    """Test problem reframing"""
    print("Testing Problem Reframing...")
    
    reframing = ProblemReframing()
    
    problem = Problem(
        problem_id=0,
        description="Test problem",
        initial_state=np.array([1.0, 0.0]),
        goal_state=np.array([0.0, 1.0]),
        constraints=[],
        domain="test"
    )
    
    # Reframe problem
    reframed = reframing.reframe_problem(problem)
    assert len(reframed) > 0, "Should create reframed problems"
    
    print("  ✓ Problem reframing works")
    return True


def test_creative_problem_solving():
    """Test integrated creative problem solving"""
    print("Testing Creative Problem Solving Integration...")
    
    solver = CreativeProblemSolving()
    
    # Create a problem
    problem = Problem(
        problem_id=0,
        description="Test problem",
        initial_state=np.array([1.0, 0.0, 0.5]),
        goal_state=np.array([0.0, 1.0, 0.5]),
        constraints=[],
        domain="test"
    )
    
    # Solve problem
    solutions = solver.solve_problem(problem, method='creative')
    assert len(solutions) > 0, "Should generate solutions"
    
    # Check statistics
    stats = solver.get_statistics()
    assert stats['problems_solved'] > 0, "Should track solved problems"
    
    print("  ✓ Creative problem solving integration works")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing Creative Problem Solving (Phase 6.2)")
    print("=" * 60)
    
    tests = [
        test_analogical_reasoning,
        test_constraint_relaxation,
        test_problem_reframing,
        test_creative_problem_solving
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

