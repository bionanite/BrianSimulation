#!/usr/bin/env python3
"""
Test suite for Creativity System (Phase 6.1)
"""

import numpy as np
import sys
import os

# Add Phase6_Creativity to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Phase6_Creativity'))

from creativity_system import (
    CreativitySystem, ConceptualBlending, DivergentThinking,
    AssociativeNetworks, RandomnessInjection, NoveltyDetection, CreativeIdea
)

# Import dependencies
from semantic_representations import ConceptFormation, SemanticNetwork, Concept
from intrinsic_motivation import CuriosityDrive


def test_conceptual_blending():
    """Test conceptual blending"""
    print("Testing Conceptual Blending...")
    
    blending = ConceptualBlending()
    
    # Create test concepts
    concept1 = Concept(
        concept_id=0,
        name="Fire",
        features=np.array([1.0, 0.0, 0.5, 0.2]),
        instances=[],
        strength=1.0
    )
    
    concept2 = Concept(
        concept_id=1,
        name="Water",
        features=np.array([0.0, 1.0, 0.3, 0.8]),
        instances=[],
        strength=1.0
    )
    
    # Test blending
    blended = blending.blend_concepts(concept1, concept2, blend_type='average')
    assert len(blended) == len(concept1.features), "Blended vector should have same dimension"
    assert np.allclose(np.linalg.norm(blended), 1.0, atol=0.1), "Blended vector should be normalized"
    
    print("  ✓ Conceptual blending works")
    return True


def test_divergent_thinking():
    """Test divergent thinking"""
    print("Testing Divergent Thinking...")
    
    divergent = DivergentThinking(num_alternatives=5)
    
    base_concept = np.array([1.0, 0.0, 0.5, 0.2])
    alternatives = divergent.generate_alternatives(base_concept, num_alternatives=5)
    
    assert len(alternatives) == 5, "Should generate 5 alternatives"
    assert all(len(alt) == len(base_concept) for alt in alternatives), "All alternatives should have same dimension"
    
    # Test diversity
    diversity = divergent.compute_diversity(alternatives)
    assert diversity > 0, "Alternatives should have some diversity"
    
    print("  ✓ Divergent thinking works")
    return True


def test_randomness_injection():
    """Test randomness injection"""
    print("Testing Randomness Injection...")
    
    randomness = RandomnessInjection(randomness_strength=0.2)
    
    base_vector = np.array([1.0, 0.0, 0.5, 0.2])
    randomized = randomness.inject_randomness(base_vector, randomness_type='gaussian')
    
    assert len(randomized) == len(base_vector), "Randomized vector should have same dimension"
    assert np.allclose(np.linalg.norm(randomized), 1.0, atol=0.1), "Randomized vector should be normalized"
    
    # Test adaptive randomness
    low_novelty = 0.1
    high_novelty = 0.9
    strength_low = randomness.adaptive_randomness_strength(low_novelty)
    strength_high = randomness.adaptive_randomness_strength(high_novelty)
    assert strength_low > strength_high, "Should increase randomness for low novelty"
    
    print("  ✓ Randomness injection works")
    return True


def test_novelty_detection():
    """Test novelty detection"""
    print("Testing Novelty Detection...")
    
    novelty = NoveltyDetection(novelty_threshold=0.3)
    
    # First idea should be maximally novel
    idea1 = np.array([1.0, 0.0, 0.5, 0.2])
    novelty1 = novelty.compute_novelty(idea1)
    assert novelty1 == 1.0, "First idea should be maximally novel"
    
    # Similar idea should have low novelty
    idea2 = np.array([0.95, 0.05, 0.5, 0.2])
    novelty2 = novelty.compute_novelty(idea2)
    assert novelty2 < novelty1, "Similar idea should have lower novelty"
    
    # Very different idea should have high novelty
    idea3 = np.array([0.0, 1.0, 0.0, 1.0])
    novelty3 = novelty.compute_novelty(idea3)
    assert novelty3 > novelty2, "Different idea should have higher novelty"
    
    # Test creativity score
    creativity = novelty.compute_creativity_score(idea3)
    assert 0 <= creativity <= 1, "Creativity score should be in [0, 1]"
    
    print("  ✓ Novelty detection works")
    return True


def test_creativity_system():
    """Test integrated creativity system"""
    print("Testing Creativity System Integration...")
    
    creativity = CreativitySystem()
    
    # Create concept formation system
    concept_formation = ConceptFormation()
    
    # Add some concepts
    concept1 = concept_formation.create_concept(
        np.array([1.0, 0.0, 0.5, 0.2]), name="Fire"
    )
    concept2 = concept_formation.create_concept(
        np.array([0.0, 1.0, 0.3, 0.8]), name="Water"
    )
    concept3 = concept_formation.create_concept(
        np.array([0.5, 0.5, 0.0, 0.5]), name="Earth"
    )
    
    # Create semantic network
    semantic_network = SemanticNetwork()
    semantic_network.add_relation(concept1.concept_id, concept2.concept_id, 'opposite_of')
    semantic_network.add_relation(concept2.concept_id, concept3.concept_id, 'related_to')
    
    # Initialize integrations
    creativity.initialize_integrations(
        semantic_network=semantic_network,
        concept_formation=concept_formation
    )
    
    # Generate ideas
    ideas = creativity.generate_ideas(num_ideas=5, method='blending')
    assert len(ideas) > 0, "Should generate ideas"
    assert all(isinstance(idea, CreativeIdea) for idea in ideas), "All should be CreativeIdea objects"
    
    # Test statistics
    stats = creativity.get_statistics()
    assert stats['ideas_generated'] > 0, "Should track generated ideas"
    
    # Test best ideas
    best_ideas = creativity.get_best_ideas(top_k=3, metric='creativity')
    assert len(best_ideas) <= 3, "Should return at most top_k ideas"
    
    print("  ✓ Creativity system integration works")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing Creativity System (Phase 6.1)")
    print("=" * 60)
    
    tests = [
        test_conceptual_blending,
        test_divergent_thinking,
        test_randomness_injection,
        test_novelty_detection,
        test_creativity_system
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

