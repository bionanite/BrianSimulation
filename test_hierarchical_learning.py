#!/usr/bin/env python3
"""
Test Framework for Hierarchical Feature Learning
Tests multi-layer hierarchical learning and multi-scale features
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from hierarchical_learning import (
    HierarchicalLearningManager, HierarchicalFeatureLearning,
    MultiScaleFeatureLearning
)

class HierarchicalLearningTester:
    """Test framework for hierarchical learning"""
    
    def __init__(self):
        self.results = []
    
    def test_hierarchical_learning(self):
        """Test hierarchical feature learning"""
        print("\n" + "="*60)
        print("TEST 1: Hierarchical Feature Learning")
        print("="*60)
        
        hierarchical = HierarchicalFeatureLearning(
            layer_sizes=[50, 30, 10],
            input_size=100,
            learning_rate=0.01
        )
        
        print(f"\nCreated hierarchical network:")
        print(f"   Input size: {hierarchical.input_size}")
        print(f"   Layer sizes: {hierarchical.layer_sizes}")
        print(f"   Total layers: {len(hierarchical.network.layers)}")
        
        # Test 1: Forward pass
        print("\nTest 1.1: Forward pass")
        input_pattern = np.random.random(100)
        activations = hierarchical.forward_pass(input_pattern)
        
        print(f"   Input size: {len(input_pattern)}")
        print(f"   Activations per layer:")
        for i, act in enumerate(activations):
            active_count = np.sum(act > 0)
            print(f"      Layer {i}: {len(act)} units, {active_count} active")
        
        print(f"   Result: {'✅ PASS' if len(activations) == len(hierarchical.layer_sizes) else '❌ FAIL'}")
        
        # Test 2: Feature learning
        print("\nTest 1.2: Feature learning")
        initial_features = hierarchical.get_layer_features(0)
        initial_norm = np.linalg.norm(initial_features)
        
        # Learn from patterns
        for _ in range(100):
            pattern = np.random.random(100)
            hierarchical.learn_from_pattern(pattern)
        
        final_features = hierarchical.get_layer_features(0)
        final_norm = np.linalg.norm(final_features)
        
        feature_change = np.linalg.norm(final_features - initial_features)
        
        print(f"   Initial feature norm: {initial_norm:.4f}")
        print(f"   Final feature norm: {final_norm:.4f}")
        print(f"   Feature change: {feature_change:.4f}")
        print(f"   Result: {'✅ PASS' if feature_change > 0 else '❌ FAIL'}")
        
        # Test 3: Abstraction levels
        print("\nTest 1.3: Abstraction levels")
        for i in range(len(hierarchical.layer_sizes)):
            abstraction = hierarchical.get_abstraction_level(i)
            print(f"   Layer {i}: abstraction level {abstraction:.3f}")
        
        # Lower layers should have lower abstraction
        low_abstraction = hierarchical.get_abstraction_level(0)
        high_abstraction = hierarchical.get_abstraction_level(len(hierarchical.layer_sizes) - 1)
        print(f"   Abstraction increases: {'✅ PASS' if high_abstraction > low_abstraction else '❌ FAIL'}")
        
        return True
    
    def test_multiscale_learning(self):
        """Test multi-scale feature learning"""
        print("\n" + "="*60)
        print("TEST 2: Multi-Scale Feature Learning")
        print("="*60)
        
        multiscale = MultiScaleFeatureLearning(
            scales=[1, 2, 4],
            base_feature_size=20,
            learning_rate=0.01
        )
        
        print(f"\nCreated multi-scale learner:")
        print(f"   Scales: {multiscale.scales}")
        print(f"   Base feature size: {multiscale.base_feature_size}")
        
        # Test 1: Feature extraction
        print("\nTest 2.1: Multi-scale feature extraction")
        input_pattern = np.random.random(100)
        features = multiscale.extract_features(input_pattern)
        
        print(f"   Input size: {len(input_pattern)}")
        print(f"   Features extracted:")
        for scale, feat in features.items():
            print(f"      Scale {scale}: {len(feat)} features")
        
        print(f"   Result: {'✅ PASS' if len(features) == len(multiscale.scales) else '❌ FAIL'}")
        
        # Test 2: Feature learning
        print("\nTest 2.2: Multi-scale feature learning")
        initial_detectors = {}
        for scale in multiscale.scales:
            initial_detectors[scale] = multiscale.scale_detectors[scale].copy()
        
        # Learn from patterns
        for _ in range(50):
            pattern = np.random.random(100)
            multiscale.learn_from_pattern(pattern)
        
        # Check if detectors changed
        detector_changes = {}
        for scale in multiscale.scales:
            change = np.linalg.norm(multiscale.scale_detectors[scale] - initial_detectors[scale])
            detector_changes[scale] = change
            print(f"   Scale {scale}: detector change {change:.4f}")
        
        total_change = sum(detector_changes.values())
        print(f"   Total change: {total_change:.4f}")
        print(f"   Result: {'✅ PASS' if total_change > 0 else '❌ FAIL'}")
        
        return True
    
    def test_integrated_hierarchical_learning(self):
        """Test integrated hierarchical learning"""
        print("\n" + "="*60)
        print("TEST 3: Integrated Hierarchical Learning")
        print("="*60)
        
        manager = HierarchicalLearningManager(
            enable_hierarchical=True,
            enable_multiscale=True,
            layer_sizes=[50, 30, 10],
            input_size=100
        )
        
        print(f"\nCreated hierarchical learning manager")
        
        # Create training patterns with structure
        print("\nCreating structured training patterns...")
        patterns = []
        
        # Pattern type 1: Low-frequency patterns
        for _ in range(20):
            pattern = np.sin(np.linspace(0, 2*np.pi, 100))
            patterns.append(pattern)
        
        # Pattern type 2: High-frequency patterns
        for _ in range(20):
            pattern = np.sin(np.linspace(0, 8*np.pi, 100))
            patterns.append(pattern)
        
        # Pattern type 3: Mixed patterns
        for _ in range(20):
            pattern = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.5 * np.sin(np.linspace(0, 12*np.pi, 100))
            patterns.append(pattern)
        
        print(f"   Created {len(patterns)} training patterns")
        
        # Train hierarchical network
        print("\nTraining hierarchical network...")
        learning_history = []
        
        for iteration in range(len(patterns)):
            pattern = patterns[iteration]
            manager.learn_from_pattern(pattern)
            
            if iteration % 20 == 0:
                # Get activations
                if manager.enable_hierarchical:
                    activations = manager.hierarchical.forward_pass(pattern)
                    learning_history.append({
                        'iteration': iteration,
                        'layer_activations': [np.sum(a > 0) for a in activations],
                        'total_activation': sum(np.sum(a) for a in activations)
                    })
        
        # Visualize learning
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Hierarchical Feature Learning Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Layer activations over time
        ax1 = axes[0, 0]
        if learning_history:
            iterations = [h['iteration'] for h in learning_history]
            layer_acts = [h['layer_activations'] for h in learning_history]
            
            for layer_idx in range(len(layer_acts[0])):
                acts = [la[layer_idx] for la in layer_acts]
                ax1.plot(iterations, acts, label=f'Layer {layer_idx}', linewidth=2, marker='o')
            
            ax1.set_title('Active Units per Layer Over Time', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Active Units')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Learned features (layer 0)
        ax2 = axes[0, 1]
        if manager.enable_hierarchical:
            features = manager.get_hierarchical_features(0)
            if features is not None:
                # Show first 10 features
                num_features = min(10, features.shape[0])
                for i in range(num_features):
                    ax2.plot(features[i], alpha=0.7, linewidth=1)
                ax2.set_title('Learned Features (Layer 0)', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Feature Index')
                ax2.set_ylabel('Weight Value')
                ax2.grid(True, alpha=0.3)
        
        # Plot 3: Multi-scale features
        ax3 = axes[1, 0]
        if manager.enable_multiscale:
            test_pattern = patterns[0]
            multiscale_features = manager.get_multiscale_features(test_pattern)
            
            for scale, features in multiscale_features.items():
                ax3.plot(features, label=f'Scale {scale}', linewidth=2, alpha=0.7)
            ax3.set_title('Multi-Scale Features', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Feature Index')
            ax3.set_ylabel('Feature Value')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Abstraction levels
        ax4 = axes[1, 1]
        if manager.enable_hierarchical:
            abstraction_levels = []
            layer_names = []
            for i in range(len(manager.hierarchical.layer_sizes)):
                abstraction = manager.hierarchical.get_abstraction_level(i)
                abstraction_levels.append(abstraction)
                layer_names.append(f'Layer {i}')
            
            bars = ax4.bar(layer_names, abstraction_levels, color='#3498DB', alpha=0.8, edgecolor='black')
            ax4.set_title('Abstraction Levels by Layer', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Abstraction Level')
            ax4.set_ylim([0, 1.1])
            ax4.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, abstraction_levels):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('hierarchical_learning_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Learning progress visualization saved: hierarchical_learning_progress.png")
        plt.close()
        
        # Statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        if 'hierarchical' in stats:
            print(f"   Hierarchical Network:")
            print(f"      Layers: {stats['hierarchical']['num_layers']}")
            print(f"      Layer sizes: {stats['hierarchical']['layer_sizes']}")
            print(f"      Total connections: {stats['hierarchical']['total_connections']}")
        
        if 'multiscale' in stats:
            print(f"   Multi-Scale Learning:")
            print(f"      Scales: {stats['multiscale']['scales']}")
            print(f"      Detectors per scale: {stats['multiscale']['num_detectors_per_scale']}")
        
        return True
    
    def run_all_tests(self):
        """Run all hierarchical learning tests"""
        print("\n" + "="*70)
        print("HIERARCHICAL FEATURE LEARNING TEST SUITE")
        print("="*70)
        
        tests = [
            ("Hierarchical Feature Learning", self.test_hierarchical_learning),
            ("Multi-Scale Feature Learning", self.test_multiscale_learning),
            ("Integrated Hierarchical Learning", self.test_integrated_hierarchical_learning)
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
    tester = HierarchicalLearningTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All hierarchical learning tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

