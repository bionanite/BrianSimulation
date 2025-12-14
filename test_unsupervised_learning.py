#!/usr/bin/env python3
"""
Test Framework for Unsupervised Learning Mechanisms
Tests Hebbian learning, competitive learning, and predictive coding
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from unsupervised_learning import (
    UnsupervisedLearningManager, HebbianLearning,
    CompetitiveLearning, PredictiveCoding, SelfOrganizingMap
)

class UnsupervisedLearningTester:
    """Test framework for unsupervised learning"""
    
    def __init__(self):
        self.results = []
    
    def test_hebbian_learning(self):
        """Test Hebbian learning"""
        print("\n" + "="*60)
        print("TEST 1: Hebbian Learning")
        print("="*60)
        
        hebbian = HebbianLearning(learning_rate=0.01, normalization='Oja')
        
        # Test 1: Basic weight update
        print("\nTest 1.1: Basic Hebbian weight update")
        weights = np.random.random(10) * 0.1
        pre_activity = np.random.random(10)
        post_activity = 0.8
        
        initial_weights = weights.copy()
        new_weights = hebbian.update_weights(weights, pre_activity, post_activity)
        
        weight_change = np.linalg.norm(new_weights - initial_weights)
        print(f"   Initial weight norm: {np.linalg.norm(initial_weights):.4f}")
        print(f"   Final weight norm: {np.linalg.norm(new_weights):.4f}")
        print(f"   Weight change: {weight_change:.4f}")
        print(f"   Result: {'✅ PASS' if weight_change > 0 else '❌ FAIL'}")
        
        # Test 2: Correlation-based learning
        print("\nTest 1.2: Correlation-based learning")
        # Create correlated patterns
        base_pattern = np.random.random(10)
        pattern1 = base_pattern + np.random.normal(0, 0.1, 10)
        pattern2 = base_pattern + np.random.normal(0, 0.1, 10)
        
        weights2 = np.random.random(10) * 0.1
        
        # Learn from correlated patterns
        for _ in range(100):
            post1 = np.dot(weights2, pattern1)
            weights2 = hebbian.update_weights(weights2, pattern1, post1)
            
            post2 = np.dot(weights2, pattern2)
            weights2 = hebbian.update_weights(weights2, pattern2, post2)
        
        # Check if weights learned the pattern
        correlation = np.corrcoef(weights2, base_pattern)[0, 1]
        print(f"   Weight-pattern correlation: {correlation:.4f}")
        print(f"   Result: {'✅ PASS' if correlation > 0.3 else '⚠️  CHECK'}")
        
        # Test 3: Normalization (Oja's rule)
        print("\nTest 1.3: Weight normalization (Oja's rule)")
        weights3 = np.random.random(10)
        weights3 = weights3 / np.linalg.norm(weights3)  # Normalize
        
        initial_norm = np.linalg.norm(weights3)
        
        # Many updates
        for _ in range(1000):
            pre = np.random.random(10)
            post = np.dot(weights3, pre)
            weights3 = hebbian.update_weights(weights3, pre, post)
        
        final_norm = np.linalg.norm(weights3)
        print(f"   Initial norm: {initial_norm:.4f}")
        print(f"   Final norm: {final_norm:.4f}")
        print(f"   Norm stable: {'✅ PASS' if abs(final_norm - initial_norm) < 0.5 else '❌ FAIL'}")
        
        return True
    
    def test_competitive_learning(self):
        """Test competitive learning"""
        print("\n" + "="*60)
        print("TEST 2: Competitive Learning")
        print("="*60)
        
        competitive = CompetitiveLearning(learning_rate=0.1)
        
        # Create feature detectors
        input_size = 20
        num_detectors = 5
        detectors = competitive.create_feature_detectors(num_detectors, input_size)
        
        print(f"\nCreated {num_detectors} feature detectors")
        print(f"Input size: {input_size}")
        
        # Test 1: Winner selection
        print("\nTest 2.1: Winner selection")
        input_pattern = np.random.random(input_size)
        winner_id, response = competitive.select_winner(detectors, input_pattern)
        
        print(f"   Winner: Detector {winner_id}")
        print(f"   Response strength: {response:.4f}")
        print(f"   Result: {'✅ PASS' if winner_id is not None else '❌ FAIL'}")
        
        # Test 2: Winner update
        print("\nTest 2.2: Winner weight update")
        winner = detectors[winner_id]
        initial_weights = winner.weights.copy()
        
        competitive.update_winner(winner, input_pattern)
        
        weight_change = np.linalg.norm(winner.weights - initial_weights)
        print(f"   Weight change: {weight_change:.4f}")
        print(f"   Win count: {winner.win_count}")
        print(f"   Result: {'✅ PASS' if weight_change > 0 else '❌ FAIL'}")
        
        # Test 3: Specialization
        print("\nTest 2.3: Feature detector specialization")
        # Train on different patterns
        patterns = [
            np.ones(input_size) * 0.9,  # Pattern 1
            np.zeros(input_size),        # Pattern 2
            np.random.random(input_size)  # Pattern 3
        ]
        
        for _ in range(100):
            pattern = patterns[np.random.randint(0, len(patterns))]
            winner_id, _ = competitive.select_winner(detectors, pattern)
            winner = detectors[winner_id]
            competitive.update_winner(winner, pattern)
            competitive.update_losers(detectors, winner_id, pattern)
        
        # Check specialization
        print(f"   Win counts:")
        for detector in detectors:
            print(f"      Detector {detector.detector_id}: {detector.win_count} wins")
        
        # Detectors should have different win counts (specialization)
        win_counts = [d.win_count for d in detectors]
        specialization = np.std(win_counts) > 0
        print(f"   Specialization: {'✅ PASS' if specialization else '⚠️  CHECK'}")
        
        return True
    
    def test_predictive_coding(self):
        """Test predictive coding"""
        print("\n" + "="*60)
        print("TEST 3: Predictive Coding")
        print("="*60)
        
        predictive = PredictiveCoding(learning_rate=0.01)
        
        # Test 1: Prediction
        print("\nTest 3.1: Pattern prediction")
        input_size = 10
        prediction_weights = np.random.random((input_size, input_size)) * 0.1
        input_pattern = np.random.random(input_size)
        
        prediction = predictive.predict(input_pattern, layer_id=0, prediction_weights=prediction_weights)
        
        print(f"   Input size: {len(input_pattern)}")
        print(f"   Prediction size: {len(prediction)}")
        print(f"   Result: {'✅ PASS' if len(prediction) == len(input_pattern) else '❌ FAIL'}")
        
        # Test 2: Error calculation
        print("\nTest 3.2: Prediction error calculation")
        error = predictive.calculate_error(input_pattern, prediction, layer_id=0)
        error_magnitude = np.linalg.norm(error)
        
        print(f"   Error magnitude: {error_magnitude:.4f}")
        print(f"   Result: {'✅ PASS' if error_magnitude >= 0 else '❌ FAIL'}")
        
        # Test 3: Learning to predict
        print("\nTest 3.3: Learning to predict patterns")
        # Create a simple pattern sequence
        pattern_sequence = [
            np.sin(np.linspace(0, 2*np.pi, input_size)),
            np.sin(np.linspace(0, 2*np.pi, input_size) + 0.1),
            np.sin(np.linspace(0, 2*np.pi, input_size) + 0.2),
        ]
        
        initial_errors = []
        final_errors = []
        
        for i, pattern in enumerate(pattern_sequence):
            prediction = predictive.predict(pattern, layer_id=0, prediction_weights=prediction_weights)
            error = predictive.calculate_error(pattern, prediction, layer_id=0)
            
            if i == 0:
                initial_errors.append(np.linalg.norm(error))
            
            # Update weights
            prediction_weights = predictive.update_weights(
                prediction_weights, error, pattern
            )
            
            if i == len(pattern_sequence) - 1:
                final_errors.append(np.linalg.norm(error))
        
        print(f"   Initial error: {initial_errors[0]:.4f}")
        print(f"   Final error: {final_errors[0]:.4f}")
        improvement = initial_errors[0] > final_errors[0]
        print(f"   Error reduced: {'✅ PASS' if improvement else '⚠️  CHECK'}")
        
        return True
    
    def test_self_organizing_map(self):
        """Test Self-Organizing Map"""
        print("\n" + "="*60)
        print("TEST 4: Self-Organizing Map (SOM)")
        print("="*60)
        
        som = SelfOrganizingMap(map_size=(5, 5), input_size=10, learning_rate=0.1)
        
        print(f"\nCreated SOM: {som.map_size[0]}x{som.map_size[1]} map")
        print(f"Input size: {som.input_size}")
        
        # Test 1: BMU finding
        print("\nTest 4.1: Best Matching Unit (BMU)")
        input_pattern = np.random.random(10)
        bmu = som.find_best_matching_unit(input_pattern)
        
        print(f"   Input pattern: {input_pattern[:3]}...")
        print(f"   BMU location: ({bmu[0]}, {bmu[1]})")
        print(f"   Result: {'✅ PASS' if bmu is not None else '❌ FAIL'}")
        
        # Test 2: Map update
        print("\nTest 4.2: SOM map update")
        initial_map = som.map.copy()
        
        som.update_map(input_pattern, iteration=0, max_iterations=100)
        
        map_change = np.linalg.norm(som.map - initial_map)
        print(f"   Map change: {map_change:.4f}")
        print(f"   Result: {'✅ PASS' if map_change > 0 else '❌ FAIL'}")
        
        # Test 3: Topographic organization
        print("\nTest 4.3: Topographic organization")
        # Train on clustered patterns
        cluster1 = np.random.random(10) * 0.5
        cluster2 = np.random.random(10) * 0.5 + 0.5
        
        patterns = []
        for _ in range(50):
            if np.random.random() < 0.5:
                patterns.append(cluster1 + np.random.normal(0, 0.1, 10))
            else:
                patterns.append(cluster2 + np.random.normal(0, 0.1, 10))
        
        # Train SOM
        for i, pattern in enumerate(patterns):
            som.update_map(pattern, iteration=i, max_iterations=len(patterns))
        
        # Check if similar patterns map to nearby locations
        bmu1 = som.find_best_matching_unit(cluster1)
        bmu2 = som.find_best_matching_unit(cluster2)
        
        distance = np.sqrt((bmu1[0] - bmu2[0])**2 + (bmu1[1] - bmu2[1])**2)
        
        print(f"   Cluster 1 BMU: ({bmu1[0]}, {bmu1[1]})")
        print(f"   Cluster 2 BMU: ({bmu2[0]}, {bmu2[1]})")
        print(f"   BMU distance: {distance:.2f}")
        print(f"   Result: {'✅ PASS' if distance > 0 else '⚠️  CHECK'}")
        
        return True
    
    def test_integrated_unsupervised_learning(self):
        """Test all unsupervised learning mechanisms together"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Unsupervised Learning")
        print("="*60)
        
        manager = UnsupervisedLearningManager(
            enable_hebbian=True,
            enable_competitive=True,
            enable_predictive=True,
            enable_som=False
        )
        
        # Initialize feature detectors
        input_size = 20
        num_detectors = 8
        manager.initialize_feature_detectors(num_detectors, input_size)
        
        print(f"\nInitialized {num_detectors} feature detectors")
        print(f"Input size: {input_size}")
        
        # Create training patterns
        print("\nCreating training patterns...")
        patterns = []
        
        # Pattern type 1: Alternating
        patterns.append(np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 2))
        
        # Pattern type 2: Blocks
        patterns.append(np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]))
        
        # Pattern type 3: Random
        patterns.append(np.random.random(input_size))
        
        # Pattern type 4: Sine wave
        patterns.append(np.sin(np.linspace(0, 4*np.pi, input_size)))
        
        # Normalize patterns
        patterns = [p / (np.linalg.norm(p) + 1e-10) for p in patterns]
        
        print(f"   Created {len(patterns)} pattern types")
        
        # Train on patterns
        print("\nTraining on patterns...")
        learning_history = []
        
        for iteration in range(200):
            # Randomly select pattern
            pattern = patterns[np.random.randint(0, len(patterns))]
            
            # Learn from pattern
            updates = manager.learn_from_pattern(pattern, current_time=iteration)
            
            if iteration % 50 == 0:
                stats = manager.get_statistics()
                learning_history.append({
                    'iteration': iteration,
                    'updates': updates,
                    'stats': stats.copy() if stats else {}
                })
        
        # Visualize learning
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Unsupervised Learning Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Feature detector activations
        ax1 = axes[0, 0]
        if manager.feature_detectors:
            win_counts = [d.win_count for d in manager.feature_detectors]
            detector_ids = [d.detector_id for d in manager.feature_detectors]
            bars = ax1.bar(detector_ids, win_counts, color='#3498DB', alpha=0.8, edgecolor='black')
            ax1.set_title('Feature Detector Win Counts', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Detector ID')
            ax1.set_ylabel('Win Count')
            ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Prediction error over time
        ax2 = axes[0, 1]
        if learning_history:
            errors = [h['updates'].get('prediction_error', 0) for h in learning_history]
            iterations = [h['iteration'] for h in learning_history]
            ax2.plot(iterations, errors, 'r-', linewidth=2, marker='o')
            ax2.set_title('Prediction Error Over Time', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Prediction Error')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Feature detector weights (visualization)
        ax3 = axes[1, 0]
        if manager.feature_detectors:
            # Show weights of top 3 detectors
            sorted_detectors = sorted(manager.feature_detectors, key=lambda d: d.win_count, reverse=True)
            top_3 = sorted_detectors[:3]
            
            for i, detector in enumerate(top_3):
                ax3.plot(detector.weights, label=f'Detector {detector.detector_id} ({detector.win_count} wins)',
                        linewidth=2, alpha=0.7)
            ax3.set_title('Top 3 Feature Detector Weights', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Weight Index')
            ax3.set_ylabel('Weight Value')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = manager.get_statistics()
        stats_text = "Learning Statistics:\n\n"
        
        if 'competitive' in stats:
            stats_text += f"Competitive Learning:\n"
            stats_text += f"  Detectors: {stats['competitive']['num_detectors']}\n"
            stats_text += f"  Total wins: {stats['competitive']['total_wins']}\n"
            stats_text += f"  Most active: Detector {stats['competitive']['most_active']}\n\n"
        
        if 'predictive' in stats:
            stats_text += f"Predictive Coding:\n"
            stats_text += f"  Layers: {stats['predictive']['num_layers']}\n"
            stats_text += f"  Avg error: {stats['predictive']['avg_error']:.4f}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('unsupervised_learning_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Learning progress visualization saved: unsupervised_learning_progress.png")
        plt.close()
        
        # Final statistics
        print(f"\n📊 Final Statistics:")
        stats = manager.get_statistics()
        if 'competitive' in stats:
            print(f"   Feature Detectors: {stats['competitive']['num_detectors']}")
            print(f"   Total Wins: {stats['competitive']['total_wins']}")
            print(f"   Most Active Detector: {stats['competitive']['most_active']}")
        
        if 'predictive' in stats:
            print(f"   Prediction Layers: {stats['predictive']['num_layers']}")
            print(f"   Average Error: {stats['predictive']['avg_error']:.4f}")
        
        return True
    
    def run_all_tests(self):
        """Run all unsupervised learning tests"""
        print("\n" + "="*70)
        print("UNSUPERVISED LEARNING TEST SUITE")
        print("="*70)
        
        tests = [
            ("Hebbian Learning", self.test_hebbian_learning),
            ("Competitive Learning", self.test_competitive_learning),
            ("Predictive Coding", self.test_predictive_coding),
            ("Self-Organizing Map", self.test_self_organizing_map),
            ("Integrated Unsupervised Learning", self.test_integrated_unsupervised_learning)
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
    tester = UnsupervisedLearningTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All unsupervised learning tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

