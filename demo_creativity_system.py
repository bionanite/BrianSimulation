#!/usr/bin/env python3
"""
Demonstration of Creativity System (Phase 6.1)
Shows creative idea generation capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add Phase6_Creativity to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Phase6_Creativity'))

from creativity_system import CreativitySystem
from semantic_representations import ConceptFormation, SemanticNetwork
from intrinsic_motivation import CuriosityDrive


def create_demo_concepts():
    """Create a set of demo concepts"""
    concept_formation = ConceptFormation()
    semantic_network = SemanticNetwork()
    
    # Create diverse concepts
    concepts_data = [
        ("Fire", [1.0, 0.0, 0.5, 0.2, 0.8]),
        ("Water", [0.0, 1.0, 0.3, 0.8, 0.1]),
        ("Earth", [0.5, 0.5, 0.0, 0.5, 0.6]),
        ("Air", [0.2, 0.3, 1.0, 0.1, 0.4]),
        ("Light", [1.0, 1.0, 0.8, 0.2, 0.9]),
        ("Dark", [0.0, 0.0, 0.2, 0.8, 0.1]),
        ("Fast", [0.9, 0.1, 0.7, 0.3, 0.8]),
        ("Slow", [0.1, 0.9, 0.3, 0.7, 0.2]),
        ("Hot", [1.0, 0.2, 0.6, 0.4, 0.7]),
        ("Cold", [0.1, 1.0, 0.2, 0.6, 0.3]),
    ]
    
    concepts = {}
    for name, features in concepts_data:
        concept = concept_formation.create_concept(
            np.array(features, dtype=np.float32), name=name
        )
        concepts[name] = concept
        
        # Normalize features
        norm = np.linalg.norm(concept.features)
        if norm > 0:
            concept.features = concept.features / norm
    
    # Add semantic relations
    relations = [
        ("Fire", "Water", "opposite_of"),
        ("Hot", "Cold", "opposite_of"),
        ("Light", "Dark", "opposite_of"),
        ("Fast", "Slow", "opposite_of"),
        ("Fire", "Hot", "causes"),
        ("Water", "Cold", "related_to"),
        ("Air", "Light", "related_to"),
        ("Earth", "Slow", "related_to"),
    ]
    
    for from_name, to_name, rel_type in relations:
        if from_name in concepts and to_name in concepts:
            semantic_network.add_relation(
                concepts[from_name].concept_id,
                concepts[to_name].concept_id,
                rel_type
            )
    
    return concept_formation, semantic_network, concepts


def demonstrate_creativity():
    """Demonstrate creativity system capabilities"""
    print("=" * 70)
    print("Creativity System Demonstration (Phase 6.1)")
    print("=" * 70)
    
    # Create creativity system
    creativity = CreativitySystem(
        blend_strength=0.5,
        num_alternatives=10,
        randomness_strength=0.2
    )
    
    # Create demo concepts
    print("\n1. Creating concept base...")
    concept_formation, semantic_network, concepts = create_demo_concepts()
    print(f"   Created {len(concepts)} concepts")
    
    # Initialize integrations
    creativity.initialize_integrations(
        semantic_network=semantic_network,
        concept_formation=concept_formation
    )
    
    # Generate ideas using different methods
    print("\n2. Generating creative ideas...")
    
    methods = ['blending', 'divergent', 'associative', 'random']
    all_ideas = []
    
    for method in methods:
        print(f"\n   Method: {method}")
        ideas = creativity.generate_ideas(num_ideas=3, method=method)
        all_ideas.extend(ideas)
        
        for i, idea in enumerate(ideas, 1):
            print(f"     Idea {i}:")
            print(f"       Description: {idea.description}")
            print(f"       Novelty: {idea.novelty_score:.3f}")
            print(f"       Creativity: {idea.creativity_score:.3f}")
            print(f"       Feasibility: {idea.feasibility_score:.3f}")
    
    # Show statistics
    print("\n3. Creativity Statistics:")
    stats = creativity.get_statistics()
    print(f"   Total ideas generated: {stats['ideas_generated']}")
    print(f"   Novel ideas: {stats['novel_ideas']}")
    print(f"   Blends created: {stats['blends_created']}")
    print(f"   Average novelty: {stats['average_novelty']:.3f}")
    print(f"   Average creativity: {stats['average_creativity']:.3f}")
    
    # Show best ideas
    print("\n4. Top Creative Ideas:")
    best_ideas = creativity.get_best_ideas(top_k=5, metric='creativity')
    for i, idea in enumerate(best_ideas, 1):
        print(f"   {i}. {idea.description}")
        print(f"      Creativity: {idea.creativity_score:.3f}, Novelty: {idea.novelty_score:.3f}")
    
    # Visualize creativity metrics
    print("\n5. Creating visualization...")
    visualize_creativity_metrics(creativity, all_ideas)
    
    print("\n" + "=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


def visualize_creativity_metrics(creativity, ideas):
    """Visualize creativity metrics"""
    if not ideas:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Novelty vs Creativity scatter
    novelties = [idea.novelty_score for idea in ideas]
    creativities = [idea.creativity_score for idea in ideas]
    
    axes[0, 0].scatter(novelties, creativities, alpha=0.6, s=50)
    axes[0, 0].set_xlabel('Novelty Score')
    axes[0, 0].set_ylabel('Creativity Score')
    axes[0, 0].set_title('Novelty vs Creativity')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Novelty distribution
    axes[0, 1].hist(novelties, bins=10, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].set_xlabel('Novelty Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Novelty Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Creativity distribution
    axes[1, 0].hist(creativities, bins=10, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('Creativity Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Creativity Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Feasibility vs Novelty trade-off
    feasibilities = [idea.feasibility_score for idea in ideas]
    axes[1, 1].scatter(novelties, feasibilities, alpha=0.6, s=50, color='red')
    axes[1, 1].set_xlabel('Novelty Score')
    axes[1, 1].set_ylabel('Feasibility Score')
    axes[1, 1].set_title('Novelty vs Feasibility Trade-off')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('creativity_system_progress.png', dpi=150, bbox_inches='tight')
    print("   Saved visualization to creativity_system_progress.png")
    plt.close()


if __name__ == "__main__":
    demonstrate_creativity()

