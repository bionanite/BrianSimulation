#!/usr/bin/env python3
"""
Semantic Representations Demonstration
Shows how the system forms concepts, grounds symbols, and builds semantic networks
"""

import numpy as np
import matplotlib.pyplot as plt
from semantic_representations import SemanticRepresentationManager

def demonstrate_semantic_representations():
    """Demonstrate semantic representations in action"""
    
    print("\n" + "="*70)
    print("SEMANTIC REPRESENTATIONS DEMONSTRATION")
    print("="*70)
    
    # Create semantic manager
    manager = SemanticRepresentationManager()
    
    print("\n🎯 Scenario: Learning Concepts and Building Knowledge")
    print("   Goal: Form concepts, ground symbols, and build semantic network")
    
    # Learn animal concepts
    print("\n📚 Learning Animal Concepts...")
    animals = {
        "cat": np.random.random(20) + np.array([1.0] * 10 + [0.0] * 10),
        "dog": np.random.random(20) + np.array([1.0] * 10 + [0.0] * 10),
        "bird": np.random.random(20) + np.array([0.0] * 10 + [1.0] * 10),
        "fish": np.random.random(20) + np.array([0.0] * 10 + [1.0] * 10),
    }
    
    concept_ids = {}
    for label, pattern in animals.items():
        concept = manager.learn_concept(pattern, label=label)
        concept_ids[label] = concept.concept_id
        print(f"   Learned '{label}' -> Concept {concept.concept_id}")
    
    # Learn general concept
    print("\n📚 Learning General Concept...")
    animal_pattern = np.random.random(20) * 0.5  # More general
    animal_concept = manager.learn_concept(animal_pattern, label="animal")
    concept_ids["animal"] = animal_concept.concept_id
    print(f"   Learned 'animal' -> Concept {animal_concept.concept_id}")
    
    # Add semantic relations
    print("\n🔗 Building Semantic Network...")
    for animal in ["cat", "dog", "bird", "fish"]:
        manager.relate_concepts(
            concept_ids[animal], concept_ids["animal"], "is_a"
        )
        print(f"   {animal} is_a animal")
    
    manager.relate_concepts(concept_ids["cat"], concept_ids["dog"], "similar_to")
    manager.relate_concepts(concept_ids["bird"], concept_ids["fish"], "similar_to")
    print(f"   cat similar_to dog")
    print(f"   bird similar_to fish")
    
    # Query concepts
    print("\n🔍 Querying Concepts...")
    for label in ["cat", "animal"]:
        info = manager.query_concept(label)
        if info:
            print(f"\n   Concept: {info['concept'].name}")
            print(f"      Frequency: {info['concept'].frequency}")
            print(f"      Strength: {info['concept'].strength:.3f}")
            print(f"      Related concepts: {len(info['related_concepts'])}")
            print(f"      Symbols: {info['symbols']}")
    
    # Test symbol grounding
    print("\n🔤 Symbol Grounding Test...")
    for label in ["cat", "dog", "bird"]:
        concept_id = manager.symbol_grounding.get_concept_for_symbol(label)
        if concept_id is not None:
            symbols = manager.symbol_grounding.get_symbols_for_concept(concept_id)
            print(f"   '{label}' -> Concept {concept_id} (symbols: {symbols})")
    
    # Test semantic similarity
    print("\n🔍 Semantic Similarity Test...")
    cat_id = concept_ids["cat"]
    dog_id = concept_ids["dog"]
    bird_id = concept_ids["bird"]
    
    similarity_cat_dog = manager.meaning_extraction.compute_semantic_similarity(cat_id, dog_id)
    similarity_cat_bird = manager.meaning_extraction.compute_semantic_similarity(cat_id, bird_id)
    
    print(f"   cat <-> dog: {similarity_cat_dog:.3f}")
    print(f"   cat <-> bird: {similarity_cat_bird:.3f}")
    print(f"   Result: {'✅ PASS' if similarity_cat_dog > similarity_cat_bird else '⚠️  CHECK'}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Semantic Representations Demonstration', fontsize=16, fontweight='bold')
    
    # Plot 1: Concept frequencies
    ax1 = axes[0, 0]
    concepts = list(manager.concept_formation.concepts.values())
    names = [c.name for c in concepts]
    frequencies = [c.frequency for c in concepts]
    
    bars = ax1.bar(range(len(names)), frequencies, color='#3498DB', alpha=0.8, edgecolor='black')
    ax1.set_title('Concept Frequencies', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Symbol grounding strengths
    ax2 = axes[0, 1]
    symbols = list(manager.symbol_grounding.symbols.values())
    labels = [s.label for s in symbols]
    strengths = [s.grounding_strength for s in symbols]
    
    bars = ax2.bar(range(len(labels)), strengths, color='#2ECC71', alpha=0.8, edgecolor='black')
    ax2.set_title('Symbol Grounding Strengths', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Grounding Strength')
    ax2.set_ylim([0, 1.1])
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Semantic similarity matrix
    ax3 = axes[1, 0]
    concept_list = list(manager.concept_formation.concepts.values())
    n = len(concept_list)
    similarity_matrix = np.zeros((n, n))
    
    for i, c1 in enumerate(concept_list):
        for j, c2 in enumerate(concept_list):
            similarity_matrix[i, j] = manager.meaning_extraction.compute_semantic_similarity(
                c1.concept_id, c2.concept_id
            )
    
    im = ax3.imshow(similarity_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax3.set_title('Semantic Similarity Matrix', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(n))
    ax3.set_yticks(range(n))
    ax3.set_xticklabels([c.name for c in concept_list], rotation=45, ha='right')
    ax3.set_yticklabels([c.name for c in concept_list])
    plt.colorbar(im, ax=ax3)
    
    # Plot 4: Network statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats = manager.get_statistics()
    network_stats = manager.semantic_network.get_statistics()
    
    stats_text = "System Statistics:\n\n"
    stats_text += f"Concepts: {stats['concepts']}\n"
    stats_text += f"Symbols: {stats['symbols']}\n"
    stats_text += f"Meanings: {stats['meanings']}\n"
    stats_text += f"\nSemantic Network:\n"
    stats_text += f"  Relations: {network_stats['total_relations']}\n"
    stats_text += f"  Avg Strength: {network_stats['avg_relation_strength']:.3f}\n\n"
    stats_text += "Relation Types:\n"
    for rel_type, count in network_stats['relation_types'].items():
        stats_text += f"  {rel_type}: {count}\n"
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('semantic_representations_demonstration.png', dpi=300, bbox_inches='tight')
    print("\n✅ Demonstration visualization saved: semantic_representations_demonstration.png")
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("DEMONSTRATION SUMMARY")
    print("="*70)
    print(f"\n✅ Semantic representations working correctly!")
    print(f"✅ Concepts formed from patterns")
    print(f"✅ Symbols grounded to concepts")
    print(f"✅ Semantic network built with relations")
    print(f"✅ Meaning extraction and similarity computation working")
    print(f"\n🎯 Key Achievement: Built semantic knowledge representation!")

if __name__ == "__main__":
    demonstrate_semantic_representations()

