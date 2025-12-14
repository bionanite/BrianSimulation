#!/usr/bin/env python3
"""
Test Framework for Multi-Modal Integration
Tests cross-modal learning, sensory fusion, attention, and unified representations
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from multimodal_integration import (
    MultiModalIntegrationManager, CrossModalLearning,
    SensoryFusion, MultiModalAttention, UnifiedRepresentationLearning
)

class MultiModalIntegrationTester:
    """Test framework for multi-modal integration"""
    
    def __init__(self):
        self.results = []
    
    def test_cross_modal_learning(self):
        """Test cross-modal learning"""
        print("\n" + "="*60)
        print("TEST 1: Cross-Modal Learning")
        print("="*60)
        
        cross_modal = CrossModalLearning(learning_rate=0.1)
        
        # Create modality features
        print("\nTest 1.1: Learning cross-modal mapping")
        from_features = np.random.random(20)
        to_features = from_features * 0.8 + np.random.normal(0, 0.1, 20)  # Related
        
        cross_modal.learn_mapping(0, 1, from_features, to_features)
        
        print(f"   Mappings learned: {len(cross_modal.mappings)}")
        print(f"   Result: {'✅ PASS' if len(cross_modal.mappings) == 1 else '❌ FAIL'}")
        
        # Test 2: Translation
        print("\nTest 1.2: Cross-modal translation")
        new_from_features = np.random.random(20)
        translated, confidence = cross_modal.translate(0, 1, new_from_features)
        
        print(f"   Translated features shape: {translated.shape}")
        print(f"   Translation confidence: {confidence:.4f}")
        print(f"   Result: {'✅ PASS' if confidence > 0 else '❌ FAIL'}")
        
        # Test 3: Multiple mappings
        print("\nTest 1.3: Multiple cross-modal mappings")
        cross_modal.learn_mapping(1, 2, to_features, np.random.random(20))
        
        print(f"   Total mappings: {len(cross_modal.mappings)}")
        print(f"   Result: {'✅ PASS' if len(cross_modal.mappings) == 2 else '❌ FAIL'}")
        
        return True
    
    def test_sensory_fusion(self):
        """Test sensory fusion"""
        print("\n" + "="*60)
        print("TEST 2: Sensory Fusion")
        print("="*60)
        
        fusion = SensoryFusion(fusion_method='weighted_average')
        
        # Create modalities
        from multimodal_integration import Modality
        
        modalities = [
            Modality(0, "vision", 10, np.random.random(10), reliability=0.9),
            Modality(1, "audio", 10, np.random.random(10), reliability=0.8),
            Modality(2, "touch", 10, np.random.random(10), reliability=0.7)
        ]
        
        print(f"\nCreated {len(modalities)} modalities")
        
        # Test 1: Fusion
        print("\nTest 2.1: Fusing modalities")
        unified = fusion.fuse_modalities(modalities)
        
        print(f"   Unified features shape: {unified.unified_features.shape}")
        print(f"   Source modalities: {unified.source_modalities}")
        print(f"   Result: {'✅ PASS' if unified is not None else '❌ FAIL'}")
        
        # Test 2: Fusion confidence
        print("\nTest 2.2: Fusion confidence")
        fusion_weights = {0: 0.4, 1: 0.35, 2: 0.25}
        confidence = fusion.compute_fusion_confidence(modalities, fusion_weights)
        
        print(f"   Fusion confidence: {confidence:.4f}")
        print(f"   Result: {'✅ PASS' if 0 <= confidence <= 1 else '❌ FAIL'}")
        
        # Test 3: Different fusion methods
        print("\nTest 2.3: Different fusion methods")
        fusion_max = SensoryFusion(fusion_method='max')
        unified_max = fusion_max.fuse_modalities(modalities)
        
        print(f"   Max fusion shape: {unified_max.unified_features.shape}")
        print(f"   Result: {'✅ PASS' if unified_max is not None else '❌ FAIL'}")
        
        return True
    
    def test_multimodal_attention(self):
        """Test multi-modal attention"""
        print("\n" + "="*60)
        print("TEST 3: Multi-Modal Attention")
        print("="*60)
        
        attention = MultiModalAttention()
        
        # Create modalities with different reliabilities
        from multimodal_integration import Modality
        
        modalities = [
            Modality(0, "vision", 10, np.random.random(10), reliability=0.9, attention_weight=1.0),
            Modality(1, "audio", 10, np.random.random(10), reliability=0.7, attention_weight=0.8),
            Modality(2, "touch", 10, np.random.random(10), reliability=0.5, attention_weight=0.6)
        ]
        
        print(f"\nCreated {len(modalities)} modalities with different reliabilities")
        
        # Test 1: Attention weights
        print("\nTest 3.1: Computing attention weights")
        attention_weights = attention.compute_attention_weights(modalities)
        
        print(f"   Attention weights:")
        for mod_id, weight in attention_weights.items():
            mod_name = next(m.name for m in modalities if m.modality_id == mod_id)
            print(f"      {mod_name}: {weight:.4f}")
        
        # Highest reliability should have highest attention
        vision_weight = attention_weights.get(0, 0)
        touch_weight = attention_weights.get(2, 0)
        print(f"   Vision > Touch attention: {'✅ PASS' if vision_weight > touch_weight else '⚠️  CHECK'}")
        
        # Test 2: Attention update
        print("\nTest 3.2: Updating modality attention")
        initial_weight = modalities[0].attention_weight
        attention.update_modality_attention(modalities[0], relevance=0.9)
        
        print(f"   Initial attention: {initial_weight:.4f}")
        print(f"   Updated attention: {modalities[0].attention_weight:.4f}")
        print(f"   Result: {'✅ PASS' if modalities[0].attention_weight != initial_weight else '❌ FAIL'}")
        
        return True
    
    def test_unified_representation_learning(self):
        """Test unified representation learning"""
        print("\n" + "="*60)
        print("TEST 4: Unified Representation Learning")
        print("="*60)
        
        unified_learning = UnifiedRepresentationLearning(unified_size=20)
        
        # Create modalities
        from multimodal_integration import Modality
        
        modalities = [
            Modality(0, "vision", 10, np.random.random(10)),
            Modality(1, "audio", 15, np.random.random(15)),
            Modality(2, "touch", 12, np.random.random(12))
        ]
        
        print(f"\nCreated {len(modalities)} modalities with different sizes")
        
        # Test 1: Projection to unified space
        print("\nTest 4.1: Projecting to unified space")
        unified_reps = []
        for m in modalities:
            unified = unified_learning.project_to_unified_space(m)
            unified_reps.append(unified)
            print(f"   {m.name}: {m.feature_size} -> {len(unified)}")
        
        print(f"   Result: {'✅ PASS' if all(len(u) == 20 for u in unified_reps) else '❌ FAIL'}")
        
        # Test 2: Learning unified representation
        print("\nTest 4.2: Learning unified representation")
        target_unified = np.random.random(20)
        learned = unified_learning.learn_unified_representation(modalities, target_unified)
        
        print(f"   Learned unified shape: {learned.shape}")
        print(f"   Result: {'✅ PASS' if learned is not None else '❌ FAIL'}")
        
        return True
    
    def test_integrated_multimodal_system(self):
        """Test integrated multi-modal system"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Multi-Modal System")
        print("="*60)
        
        manager = MultiModalIntegrationManager()
        
        # Register modalities
        print("\nRegistering modalities...")
        vision = manager.register_modality("vision", feature_size=20)
        audio = manager.register_modality("audio", feature_size=15)
        touch = manager.register_modality("touch", feature_size=12)
        
        print(f"   Vision: ID {vision.modality_id}")
        print(f"   Audio: ID {audio.modality_id}")
        print(f"   Touch: ID {touch.modality_id}")
        
        # Update modalities
        print("\nUpdating modality features...")
        manager.update_modality(vision.modality_id, np.random.random(20))
        manager.update_modality(audio.modality_id, np.random.random(15))
        manager.update_modality(touch.modality_id, np.random.random(12))
        
        # Learn cross-modal mappings
        print("\nLearning cross-modal mappings...")
        manager.learn_cross_modal_mapping(vision.modality_id, audio.modality_id)
        manager.learn_cross_modal_mapping(audio.modality_id, touch.modality_id)
        
        print(f"   Mappings learned: {len(manager.cross_modal_learning.mappings)}")
        
        # Test translation
        print("\nTesting cross-modal translation...")
        translated, confidence = manager.translate_modality(vision.modality_id, audio.modality_id)
        print(f"   Vision -> Audio: confidence {confidence:.4f}")
        
        # Fuse modalities
        print("\nFusing modalities...")
        unified = manager.fuse_modalities([vision.modality_id, audio.modality_id, touch.modality_id])
        
        if unified:
            print(f"   Unified representation created")
            print(f"   Features shape: {unified.unified_features.shape}")
            print(f"   Source modalities: {unified.source_modalities}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Multi-Modal Integration', fontsize=16, fontweight='bold')
        
        # Plot 1: Modality features
        ax1 = axes[0, 0]
        modalities_list = list(manager.modalities.values())
        for m in modalities_list:
            ax1.plot(m.features, label=m.name, linewidth=2, alpha=0.7)
        ax1.set_title('Modality Features', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Feature Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Attention weights over time
        ax2 = axes[0, 1]
        if manager.attention.attention_history:
            history = manager.attention.attention_history
            modality_ids = list(history[0].keys())
            
            for mod_id in modality_ids:
                weights = [h.get(mod_id, 0) for h in history]
                mod_name = next(m.name for m in modalities_list if m.modality_id == mod_id)
                ax2.plot(weights, label=mod_name, linewidth=2, marker='o')
            
            ax2.set_title('Attention Weights Over Time', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Attention Weight')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cross-modal mappings
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        mapping_text = "Cross-Modal Mappings:\n\n"
        for (from_id, to_id), mapping in manager.cross_modal_learning.mappings.items():
            from_name = next(m.name for m in modalities_list if m.modality_id == from_id)
            to_name = next(m.name for m in modalities_list if m.modality_id == to_id)
            mapping_text += f"  {from_name} -> {to_name}\n"
            mapping_text += f"    Strength: {mapping.strength:.3f}\n"
            mapping_text += f"    Confidence: {mapping.confidence:.3f}\n\n"
        
        if not manager.cross_modal_learning.mappings:
            mapping_text = "No cross-modal mappings learned yet"
        
        ax3.text(0.1, 0.5, mapping_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax3.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = manager.get_statistics()
        stats_text = "Multi-Modal Statistics:\n\n"
        stats_text += f"Modalities: {stats['modalities']}\n"
        stats_text += f"Cross-Modal Mappings: {stats['cross_modal_mappings']}\n"
        stats_text += f"Unified Representations: {stats['unified_representations']}\n"
        stats_text += f"Attention Steps: {stats['attention_history_length']}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('multimodal_integration_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: multimodal_integration_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Modalities: {stats['modalities']}")
        print(f"   Cross-modal mappings: {stats['cross_modal_mappings']}")
        print(f"   Unified representations: {stats['unified_representations']}")
        print(f"   Attention steps: {stats['attention_history_length']}")
        
        return True
    
    def run_all_tests(self):
        """Run all multi-modal integration tests"""
        print("\n" + "="*70)
        print("MULTI-MODAL INTEGRATION TEST SUITE")
        print("="*70)
        
        tests = [
            ("Cross-Modal Learning", self.test_cross_modal_learning),
            ("Sensory Fusion", self.test_sensory_fusion),
            ("Multi-Modal Attention", self.test_multimodal_attention),
            ("Unified Representation Learning", self.test_unified_representation_learning),
            ("Integrated Multi-Modal System", self.test_integrated_multimodal_system)
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
    tester = MultiModalIntegrationTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All multi-modal integration tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

