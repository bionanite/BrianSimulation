#!/usr/bin/env python3
"""
Test suite for Artistic Creation (Phase 6.3)
"""

import numpy as np
import sys
import os

# Add Phase6_Creativity to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Phase6_Creativity'))

from artistic_creation import (
    ArtisticCreation, Artwork, ArtisticStyle,
    AestheticEvaluation, StyleLearning, CompositionGeneration,
    EmotionalExpression, StyleTransfer
)


def test_aesthetic_evaluation():
    """Test aesthetic evaluation"""
    print("Testing Aesthetic Evaluation...")
    
    evaluator = AestheticEvaluation()
    
    artwork = Artwork(
        artwork_id=0,
        description="Test artwork",
        features=np.array([0.5, 0.6, 0.7, 0.5, 0.6]),
        style="test",
        emotion="neutral",
        aesthetic_score=0.5,
        creativity_score=0.6
    )
    
    evaluation = evaluator.evaluate_aesthetics(artwork)
    assert 'aesthetic_score' in evaluation, "Should compute aesthetic score"
    assert 0 <= evaluation['aesthetic_score'] <= 1, "Aesthetic score should be in [0, 1]"
    
    print("  ✓ Aesthetic evaluation works")
    return True


def test_style_learning():
    """Test style learning"""
    print("Testing Style Learning...")
    
    style_learner = StyleLearning()
    
    # Create example artworks
    examples = [
        Artwork(0, "Art 1", np.array([1.0, 0.0, 0.5]), "style1", "neutral", 0.5, 0.6),
        Artwork(1, "Art 2", np.array([0.9, 0.1, 0.6]), "style1", "neutral", 0.5, 0.6),
        Artwork(2, "Art 3", np.array([0.8, 0.2, 0.5]), "style1", "neutral", 0.5, 0.6),
    ]
    
    # Learn style
    style = style_learner.learn_style("style1", examples)
    assert style is not None, "Should learn style"
    assert style.name == "style1", "Style should have correct name"
    
    print("  ✓ Style learning works")
    return True


def test_emotional_expression():
    """Test emotional expression"""
    print("Testing Emotional Expression...")
    
    expression = EmotionalExpression()
    
    base_features = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    
    # Express emotion
    expressive = expression.express_emotion(base_features, "joy", intensity=0.7)
    assert len(expressive) == len(base_features), "Should maintain dimension"
    
    # Detect emotion
    artwork = Artwork(0, "Test", expressive, "test", "neutral", 0.5, 0.6)
    detected = expression.detect_emotion(artwork)
    assert detected in expression.emotion_mappings, "Should detect valid emotion"
    
    print("  ✓ Emotional expression works")
    return True


def test_artistic_creation():
    """Test integrated artistic creation"""
    print("Testing Artistic Creation Integration...")
    
    creator = ArtisticCreation()
    
    # Create artwork
    artwork = creator.create_artwork(
        base_theme=np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        emotion="joy",
        medium="visual"
    )
    
    assert artwork is not None, "Should create artwork"
    assert artwork.aesthetic_score > 0, "Should have aesthetic score"
    
    # Check statistics
    stats = creator.get_statistics()
    assert stats['artworks_created'] > 0, "Should track created artworks"
    
    print("  ✓ Artistic creation integration works")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing Artistic Creation (Phase 6.3)")
    print("=" * 60)
    
    tests = [
        test_aesthetic_evaluation,
        test_style_learning,
        test_emotional_expression,
        test_artistic_creation
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

