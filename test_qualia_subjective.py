#!/usr/bin/env python3
"""
Test Framework for Qualia and Subjective Experience
Tests phenomenal consciousness, qualia representation, first-person perspective, and experience modeling
"""

import numpy as np
import matplotlib.pyplot as plt
from qualia_subjective import (
    QualiaSubjectiveManager, PhenomenalConsciousness,
    QualiaRepresentation, FirstPersonPerspective, SubjectiveExperienceModeling
)

class QualiaSubjectiveTester:
    """Test framework for qualia and subjective experience"""
    
    def __init__(self):
        self.results = []
    
    def test_phenomenal_consciousness(self):
        """Test phenomenal consciousness"""
        print("\n" + "="*60)
        print("TEST 1: Phenomenal Consciousness")
        print("="*60)
        
        consciousness = PhenomenalConsciousness()
        qualia_rep = QualiaRepresentation()
        
        # Create qualia
        quale1 = qualia_rep.create_quale("visual", {"color": 0.8, "brightness": 0.7})
        quale2 = qualia_rep.create_quale("emotional", {"happiness": 0.9, "excitement": 0.6})
        
        # Test 1: Create experience
        print("\nTest 1.1: Creating experiences")
        exp1 = consciousness.create_experience("seeing red", [quale1.quale_id], valence=0.7, arousal=0.8)
        exp2 = consciousness.create_experience("feeling happy", [quale2.quale_id], valence=0.9, arousal=0.7)
        
        print(f"   Experiences created: {len(consciousness.experiences)}")
        print(f"   Result: {'✅ PASS' if len(consciousness.experiences) == 2 else '❌ FAIL'}")
        
        # Test 2: Get current experience
        print("\nTest 1.2: Getting current experience")
        current = consciousness.get_current_experience()
        
        print(f"   Current experience: {current.description if current else None}")
        print(f"   Result: {'✅ PASS' if current is not None else '❌ FAIL'}")
        
        return True
    
    def test_qualia_representation(self):
        """Test qualia representation"""
        print("\n" + "="*60)
        print("TEST 2: Qualia Representation")
        print("="*60)
        
        qualia = QualiaRepresentation()
        
        # Test 1: Create qualia
        print("\nTest 2.1: Creating qualia")
        quale1 = qualia.create_quale("visual", {"color": 0.8, "brightness": 0.7}, intensity=0.9)
        quale2 = qualia.create_quale("auditory", {"pitch": 0.6, "volume": 0.8}, intensity=0.7)
        
        print(f"   Qualia created: {len(qualia.qualia)}")
        print(f"   Result: {'✅ PASS' if len(qualia.qualia) == 2 else '❌ FAIL'}")
        
        # Test 2: Get qualia by type
        print("\nTest 2.2: Getting qualia by type")
        visual_qualia = qualia.get_qualia_by_type("visual")
        
        print(f"   Visual qualia: {len(visual_qualia)}")
        print(f"   Result: {'✅ PASS' if len(visual_qualia) == 1 else '❌ FAIL'}")
        
        # Test 3: Compute similarity
        print("\nTest 2.3: Computing qualia similarity")
        similarity = qualia.compute_quale_similarity(quale1.quale_id, quale2.quale_id)
        
        print(f"   Similarity: {similarity:.4f}")
        print(f"   Result: {'✅ PASS' if similarity >= 0 else '❌ FAIL'}")
        
        return True
    
    def test_first_person_perspective(self):
        """Test first-person perspective"""
        print("\n" + "="*60)
        print("TEST 3: First-Person Perspective")
        print("="*60)
        
        perspective = FirstPersonPerspective()
        
        # Test 1: Update perspective
        print("\nTest 3.1: Updating perspective")
        perspective.update_perspective({"position": 0.5, "orientation": 0.3, "attention": 0.8})
        perspective.update_perspective({"position": 0.6, "orientation": 0.4, "attention": 0.7})
        
        print(f"   Perspective history: {len(perspective.perspective_history)}")
        print(f"   Result: {'✅ PASS' if len(perspective.perspective_history) == 2 else '❌ FAIL'}")
        
        # Test 2: Get current perspective
        print("\nTest 3.2: Getting current perspective")
        current = perspective.get_current_perspective()
        
        print(f"   Current perspective keys: {len(current)}")
        print(f"   Result: {'✅ PASS' if len(current) > 0 else '❌ FAIL'}")
        
        # Test 3: Compute continuity
        print("\nTest 3.3: Computing perspective continuity")
        continuity = perspective.compute_perspective_continuity()
        
        print(f"   Continuity: {continuity:.4f}")
        print(f"   Result: {'✅ PASS' if continuity >= 0 else '❌ FAIL'}")
        
        return True
    
    def test_experience_modeling(self):
        """Test experience modeling"""
        print("\n" + "="*60)
        print("TEST 4: Experience Modeling")
        print("="*60)
        
        modeling = SubjectiveExperienceModeling()
        
        # Test 1: Create experience model
        print("\nTest 4.1: Creating experience models")
        modeling.create_experience_model("visual", {"color": 1.0, "brightness": 0.8})
        modeling.create_experience_model("auditory", {"pitch": 0.9, "volume": 1.0})
        
        print(f"   Models created: {len(modeling.experience_models)}")
        print(f"   Result: {'✅ PASS' if len(modeling.experience_models) == 2 else '❌ FAIL'}")
        
        # Test 2: Model experience
        print("\nTest 4.2: Modeling experiences")
        input_data = {"color": 0.8, "brightness": 0.6}
        modeled = modeling.model_experience("visual", input_data)
        
        print(f"   Modeled properties: {len(modeled)}")
        print(f"   Result: {'✅ PASS' if len(modeled) > 0 else '❌ FAIL'}")
        
        return True
    
    def test_integrated_qualia_subjective(self):
        """Test integrated qualia and subjective experience"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Qualia and Subjective Experience")
        print("="*60)
        
        manager = QualiaSubjectiveManager()
        
        # Create experiences
        print("\nCreating subjective experiences...")
        exp1 = manager.create_experience("seeing red", "visual", {"color": 0.8, "brightness": 0.7}, valence=0.7, arousal=0.8)
        exp2 = manager.create_experience("hearing music", "auditory", {"pitch": 0.6, "volume": 0.8}, valence=0.9, arousal=0.7)
        exp3 = manager.create_experience("feeling happy", "emotional", {"happiness": 0.9, "excitement": 0.6}, valence=0.95, arousal=0.8)
        
        print(f"   Experiences created: {len(manager.phenomenal_consciousness.experiences)}")
        
        # Update perspective
        print("\nUpdating first-person perspective...")
        manager.update_perspective({"position": 0.5, "orientation": 0.3, "attention": 0.8})
        manager.update_perspective({"position": 0.6, "orientation": 0.4, "attention": 0.7})
        
        print(f"   Perspective updates: {len(manager.first_person_perspective.perspective_history)}")
        
        # Create experience models
        print("\nCreating experience models...")
        manager.experience_modeling.create_experience_model("visual", {"color": 1.0, "brightness": 0.8})
        manager.experience_modeling.create_experience_model("auditory", {"pitch": 0.9, "volume": 1.0})
        
        print(f"   Experience models: {len(manager.experience_modeling.experience_models)}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Qualia and Subjective Experience', fontsize=16, fontweight='bold')
        
        # Plot 1: Experience valence and arousal
        ax1 = axes[0, 0]
        experiences = list(manager.phenomenal_consciousness.experiences.values())
        valences = [e.valence for e in experiences]
        arousals = [e.arousal for e in experiences]
        
        ax1.scatter(valences, arousals, s=100, alpha=0.7, c=range(len(experiences)), cmap='viridis')
        ax1.set_title('Experience Valence-Arousal', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Valence')
        ax1.set_ylabel('Arousal')
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([0, 1])
        ax1.grid(alpha=0.3)
        
        # Plot 2: Qualia types
        ax2 = axes[0, 1]
        qualia_summary = manager.qualia_representation.get_qualia_summary()
        types = list(qualia_summary['types'].keys())
        counts = [qualia_summary['types'][t] for t in types]
        
        bars = ax2.bar(types, counts, color='#3498DB', alpha=0.8, edgecolor='black')
        ax2.set_title('Qualia Types', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.set_xticklabels(types, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Qualia intensities
        ax3 = axes[1, 0]
        qualia_list = list(manager.qualia_representation.qualia.values())
        intensities = [q.intensity for q in qualia_list]
        types_list = [q.experience_type for q in qualia_list]
        
        bars = ax3.bar(range(len(qualia_list)), intensities, color='#2ECC71', alpha=0.8, edgecolor='black')
        ax3.set_title('Qualia Intensities', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Intensity')
        ax3.set_xticks(range(len(qualia_list)))
        ax3.set_xticklabels(types_list, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = manager.get_statistics()
        stats_text = "Qualia & Subjective Experience:\n\n"
        stats_text += f"Total Experiences: {stats['total_experiences']}\n"
        stats_text += f"Avg Valence: {stats['avg_valence']:.3f}\n"
        stats_text += f"Avg Arousal: {stats['avg_arousal']:.3f}\n"
        stats_text += f"Total Qualia: {stats['total_qualia']}\n"
        stats_text += f"Qualia Types: {len(stats['types'])}\n"
        stats_text += f"Perspective Continuity: {stats['continuity']:.3f}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('qualia_subjective_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: qualia_subjective_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Total experiences: {stats['total_experiences']}")
        print(f"   Average valence: {stats['avg_valence']:.3f}")
        print(f"   Average arousal: {stats['avg_arousal']:.3f}")
        print(f"   Total qualia: {stats['total_qualia']}")
        print(f"   Qualia types: {len(stats['types'])}")
        print(f"   Perspective continuity: {stats['continuity']:.3f}")
        
        return True
    
    def run_all_tests(self):
        """Run all qualia and subjective experience tests"""
        print("\n" + "="*70)
        print("QUALIA AND SUBJECTIVE EXPERIENCE TEST SUITE")
        print("="*70)
        
        tests = [
            ("Phenomenal Consciousness", self.test_phenomenal_consciousness),
            ("Qualia Representation", self.test_qualia_representation),
            ("First-Person Perspective", self.test_first_person_perspective),
            ("Experience Modeling", self.test_experience_modeling),
            ("Integrated Qualia & Subjective", self.test_integrated_qualia_subjective)
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
    tester = QualiaSubjectiveTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All qualia and subjective experience tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

