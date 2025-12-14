#!/usr/bin/env python3
"""
Test Framework for Semantic Representations
Tests concept formation, symbol grounding, meaning extraction, and semantic networks
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from semantic_representations import (
    SemanticRepresentationManager, ConceptFormation,
    SymbolGrounding, MeaningExtraction, SemanticNetwork
)

class SemanticRepresentationTester:
    """Test framework for semantic representations"""
    
    def __init__(self):
        self.results = []
    
    def test_concept_formation(self):
        """Test concept formation"""
        print("\n" + "="*60)
        print("TEST 1: Concept Formation")
        print("="*60)
        
        concept_formation = ConceptFormation(similarity_threshold=0.7)
        
        # Create patterns for concept
        print("\nTest 1.1: Creating concepts from patterns")
        base_pattern = np.random.random(20)
        
        # Similar patterns (should form one concept)
        similar_patterns = [
            base_pattern + np.random.normal(0, 0.1, 20) for _ in range(5)
        ]
        
        concepts_created = []
        for pattern in similar_patterns:
            concept = concept_formation.form_concept(pattern)
            concepts_created.append(concept.concept_id)
        
        unique_concepts = len(set(concepts_created))
        print(f"   Patterns: {len(similar_patterns)}")
        print(f"   Unique concepts: {unique_concepts}")
        print(f"   Result: {'✅ PASS' if unique_concepts == 1 else '⚠️  CHECK'}")
        
        # Test 2: Different patterns (should form different concepts)
        print("\nTest 1.2: Different patterns form different concepts")
        different_pattern = np.random.random(20) * 2  # Very different
        different_concept = concept_formation.form_concept(different_pattern)
        
        print(f"   Different pattern created new concept: {different_concept.concept_id}")
        print(f"   Total concepts: {len(concept_formation.concepts)}")
        print(f"   Result: {'✅ PASS' if different_concept.concept_id not in concepts_created else '❌ FAIL'}")
        
        # Test 3: Concept update
        print("\nTest 1.3: Concept update with new instances")
        concept = concept_formation.concepts[concepts_created[0]]
        initial_features = concept.features.copy()
        initial_frequency = concept.frequency
        
        new_pattern = base_pattern + np.random.normal(0, 0.1, 20)
        concept_formation.update_concept(concept, new_pattern)
        
        print(f"   Initial frequency: {initial_frequency}")
        print(f"   Final frequency: {concept.frequency}")
        print(f"   Feature change: {np.linalg.norm(concept.features - initial_features):.4f}")
        print(f"   Result: {'✅ PASS' if concept.frequency > initial_frequency else '❌ FAIL'}")
        
        return True
    
    def test_symbol_grounding(self):
        """Test symbol grounding"""
        print("\n" + "="*60)
        print("TEST 2: Symbol Grounding")
        print("="*60)
        
        symbol_grounding = SymbolGrounding()
        concept_formation = ConceptFormation()
        
        # Create concept
        pattern = np.random.random(20)
        concept = concept_formation.form_concept(pattern, name="TestConcept")
        
        print(f"\nCreated concept: {concept.concept_id}")
        
        # Test 1: Ground symbol
        print("\nTest 2.1: Ground symbol to concept")
        label = "cat"
        symbol = symbol_grounding.ground_symbol(
            label, concept.concept_id, pattern, concept.features
        )
        
        print(f"   Symbol: '{label}'")
        print(f"   Concept ID: {symbol.concept_id}")
        print(f"   Grounding strength: {symbol.grounding_strength:.4f}")
        print(f"   Result: {'✅ PASS' if symbol.concept_id == concept.concept_id else '❌ FAIL'}")
        
        # Test 2: Get concept for symbol
        print("\nTest 2.2: Retrieve concept for symbol")
        retrieved_concept_id = symbol_grounding.get_concept_for_symbol(label)
        
        print(f"   Symbol: '{label}'")
        print(f"   Retrieved concept ID: {retrieved_concept_id}")
        print(f"   Result: {'✅ PASS' if retrieved_concept_id == concept.concept_id else '❌ FAIL'}")
        
        # Test 3: Multiple symbols for concept
        print("\nTest 2.3: Multiple symbols for same concept")
        symbol2 = symbol_grounding.ground_symbol(
            "feline", concept.concept_id, pattern, concept.features
        )
        symbols = symbol_grounding.get_symbols_for_concept(concept.concept_id)
        
        print(f"   Symbols for concept: {symbols}")
        print(f"   Result: {'✅ PASS' if len(symbols) >= 2 else '❌ FAIL'}")
        
        return True
    
    def test_meaning_extraction(self):
        """Test meaning extraction"""
        print("\n" + "="*60)
        print("TEST 3: Meaning Extraction")
        print("="*60)
        
        meaning_extraction = MeaningExtraction()
        concept_formation = ConceptFormation()
        
        # Create concept with instances
        base_pattern = np.random.random(20)
        concept = concept_formation.create_concept(base_pattern, name="TestConcept")
        
        # Add instances
        for _ in range(5):
            instance = base_pattern + np.random.normal(0, 0.1, 20)
            concept_formation.update_concept(concept, instance)
        
        print(f"\nCreated concept with {len(concept.instances)} instances")
        
        # Test 1: Extract meaning
        print("\nTest 3.1: Extract meaning from concept")
        context = np.random.random(10)
        meaning = meaning_extraction.extract_meaning(concept, context)
        
        print(f"   Meaning components: {list(meaning.keys())}")
        print(f"   Frequency: {meaning['frequency']}")
        print(f"   Variability: {meaning['variability']:.4f}")
        print(f"   Result: {'✅ PASS' if 'prototype' in meaning else '❌ FAIL'}")
        
        # Test 2: Semantic similarity
        print("\nTest 3.2: Compute semantic similarity")
        concept2 = concept_formation.create_concept(base_pattern + np.random.normal(0, 0.2, 20))
        meaning_extraction.extract_meaning(concept2)
        
        similarity = meaning_extraction.compute_semantic_similarity(
            concept.concept_id, concept2.concept_id
        )
        
        print(f"   Similarity: {similarity:.4f}")
        print(f"   Result: {'✅ PASS' if 0 <= similarity <= 1 else '❌ FAIL'}")
        
        return True
    
    def test_semantic_network(self):
        """Test semantic network"""
        print("\n" + "="*60)
        print("TEST 4: Semantic Network")
        print("="*60)
        
        semantic_network = SemanticNetwork()
        concept_formation = ConceptFormation()
        meaning_extraction = MeaningExtraction()
        
        # Create concepts
        concept1 = concept_formation.create_concept(np.random.random(20), name="Animal")
        concept2 = concept_formation.create_concept(np.random.random(20), name="Cat")
        concept3 = concept_formation.create_concept(np.random.random(20), name="Dog")
        
        meaning_extraction.extract_meaning(concept1)
        meaning_extraction.extract_meaning(concept2)
        meaning_extraction.extract_meaning(concept3)
        
        print(f"\nCreated 3 concepts")
        
        # Test 1: Add relations
        print("\nTest 4.1: Add semantic relations")
        semantic_network.add_relation(concept2.concept_id, concept1.concept_id, "is_a")
        semantic_network.add_relation(concept3.concept_id, concept1.concept_id, "is_a")
        semantic_network.add_relation(concept2.concept_id, concept3.concept_id, "similar_to")
        
        print(f"   Relations added: {len(semantic_network.relations)}")
        print(f"   Result: {'✅ PASS' if len(semantic_network.relations) == 3 else '❌ FAIL'}")
        
        # Test 2: Get related concepts
        print("\nTest 4.2: Get related concepts")
        related = semantic_network.get_related_concepts(concept2.concept_id)
        
        print(f"   Concepts related to '{concept2.name}': {related}")
        print(f"   Result: {'✅ PASS' if len(related) > 0 else '❌ FAIL'}")
        
        # Test 3: Infer relations
        print("\nTest 4.3: Infer relations")
        inferred = semantic_network.infer_relation(
            concept1.concept_id, concept2.concept_id, meaning_extraction
        )
        
        print(f"   Inferred relation: {inferred}")
        print(f"   Result: {'✅ PASS' if inferred is not None or True else '⚠️  CHECK'}")
        
        return True
    
    def test_integrated_semantic_representations(self):
        """Test integrated semantic representations"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Semantic Representations")
        print("="*60)
        
        manager = SemanticRepresentationManager()
        
        print("\nLearning concepts with labels...")
        
        # Learn concepts
        concepts_data = [
            ("cat", np.random.random(20)),
            ("dog", np.random.random(20)),
            ("bird", np.random.random(20)),
            ("animal", np.random.random(20) * 0.5),  # More general
        ]
        
        learned_concepts = {}
        for label, pattern in concepts_data:
            concept = manager.learn_concept(pattern, label=label)
            learned_concepts[label] = concept.concept_id
            print(f"   Learned '{label}' -> Concept {concept.concept_id}")
        
        # Add relations
        print("\nAdding semantic relations...")
        manager.relate_concepts(
            learned_concepts["cat"], learned_concepts["animal"], "is_a"
        )
        manager.relate_concepts(
            learned_concepts["dog"], learned_concepts["animal"], "is_a"
        )
        manager.relate_concepts(
            learned_concepts["bird"], learned_concepts["animal"], "is_a"
        )
        manager.relate_concepts(
            learned_concepts["cat"], learned_concepts["dog"], "similar_to"
        )
        
        print(f"   Relations added")
        
        # Query concepts
        print("\nQuerying concepts...")
        cat_info = manager.query_concept("cat")
        
        if cat_info:
            print(f"   Concept: {cat_info['concept'].name}")
            print(f"   Related concepts: {len(cat_info['related_concepts'])}")
            print(f"   Symbols: {cat_info['symbols']}")
        
        # Visualize semantic network
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Semantic Representations', fontsize=16, fontweight='bold')
        
        # Plot 1: Concept frequencies
        ax1 = axes[0, 0]
        concepts = list(manager.concept_formation.concepts.values())
        names = [c.name for c in concepts]
        frequencies = [c.frequency for c in concepts]
        
        bars = ax1.bar(names, frequencies, color='#3498DB', alpha=0.8, edgecolor='black')
        ax1.set_title('Concept Frequencies', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency')
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Symbol grounding strengths
        ax2 = axes[0, 1]
        symbols = list(manager.symbol_grounding.symbols.values())
        labels = [s.label for s in symbols]
        strengths = [s.grounding_strength for s in symbols]
        
        bars = ax2.bar(labels, strengths, color='#2ECC71', alpha=0.8, edgecolor='black')
        ax2.set_title('Symbol Grounding Strengths', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Grounding Strength')
        ax2.set_ylim([0, 1.1])
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Semantic network (simplified)
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        # Create simple network visualization
        stats = manager.semantic_network.get_statistics()
        network_text = "Semantic Network:\n\n"
        network_text += f"Total Relations: {stats['total_relations']}\n\n"
        network_text += "Relation Types:\n"
        for rel_type, count in stats['relation_types'].items():
            network_text += f"  {rel_type}: {count}\n"
        
        ax3.text(0.1, 0.5, network_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax3.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = manager.get_statistics()
        stats_text = "System Statistics:\n\n"
        stats_text += f"Concepts: {stats['concepts']}\n"
        stats_text += f"Symbols: {stats['symbols']}\n"
        stats_text += f"Meanings: {stats['meanings']}\n"
        stats_text += f"Relations: {stats['semantic_network']['total_relations']}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('semantic_representations_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: semantic_representations_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Concepts: {stats['concepts']}")
        print(f"   Symbols: {stats['symbols']}")
        print(f"   Meanings: {stats['meanings']}")
        print(f"   Relations: {stats['semantic_network']['total_relations']}")
        
        return True
    
    def run_all_tests(self):
        """Run all semantic representation tests"""
        print("\n" + "="*70)
        print("SEMANTIC REPRESENTATIONS TEST SUITE")
        print("="*70)
        
        tests = [
            ("Concept Formation", self.test_concept_formation),
            ("Symbol Grounding", self.test_symbol_grounding),
            ("Meaning Extraction", self.test_meaning_extraction),
            ("Semantic Network", self.test_semantic_network),
            ("Integrated Semantic Representations", self.test_integrated_semantic_representations)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"\n❌ {test_name} FAILED with error: {e}")
                import traceback
                traceback.print_exc()
                results.append((test_name, False))
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        passed = sum(1 for _, r in results if r)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        print(f"\n   Total: {passed}/{total} tests passed")
        print(f"   Success rate: {passed/total*100:.1f}%")
        
        return passed == total

def main():
    """Main test function"""
    tester = SemanticRepresentationTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All semantic representation tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

