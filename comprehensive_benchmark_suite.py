#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite
Runs all benchmarks and generates comprehensive performance report
"""

import sys
import os
import json
from datetime import datetime
from benchmark_framework import BenchmarkFramework
from benchmark_adapters import (
    MMLUAdapter, HellaSwagAdapter, GSM8KAdapter,
    ARCAdapter, HumanEvalAdapter
)
from performance_tracking import PerformanceTracker
from benchmark_learning import BenchmarkLearner
from final_enhanced_brain import FinalEnhancedBrain
from language_integration import LLMIntegration
from advanced_reasoning import AdvancedReasoning


def run_comprehensive_suite(neuron_count: int = 10000,
                            use_llm: bool = False,
                            max_questions_per_benchmark: int = 20,
                            enable_learning: bool = True):
    """
    Run comprehensive benchmark suite
    
    Args:
        neuron_count: Number of neurons to use
        use_llm: Whether to use LLM integration
        max_questions_per_benchmark: Max questions per benchmark
        enable_learning: Whether to enable learning from feedback
    """
    print("=" * 70)
    print("COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 70)
    print(f"Neuron Count: {neuron_count:,}")
    print(f"LLM Integration: {'Enabled' if use_llm else 'Disabled'}")
    print(f"Learning Enabled: {'Yes' if enable_learning else 'No'}")
    print("=" * 70)
    print()
    
    # Initialize brain system
    print("Initializing brain system...")
    brain = FinalEnhancedBrain(total_neurons=neuron_count, debug=False)
    print("✅ Brain initialized")
    
    # Initialize advanced reasoning
    advanced_reasoning = AdvancedReasoning(brain_system=brain)
    brain.advanced_reasoning = advanced_reasoning  # Attach for use
    
    # Initialize LLM integration if requested
    llm_integration = None
    if use_llm:
        print("Initializing LLM integration...")
        llm_integration = LLMIntegration(provider="openai")
        if llm_integration.provider != "none":
            print("✅ LLM integration enabled")
        else:
            print("⚠️  LLM integration not available, using brain-only mode")
            llm_integration = None
    
    # Initialize benchmark framework
    print("Initializing benchmark framework...")
    framework = BenchmarkFramework(brain_system=brain)
    
    # Register all benchmarks
    benchmarks = {
        "MMLU": MMLUAdapter(),
        "HellaSwag": HellaSwagAdapter(),
        "GSM8K": GSM8KAdapter(),
        "ARC": ARCAdapter(),
        "HumanEval": HumanEvalAdapter()
    }
    
    for name, adapter in benchmarks.items():
        framework.register_adapter(name, adapter)
    
    print(f"✅ Registered {len(benchmarks)} benchmarks")
    
    # Initialize performance tracker
    tracker = PerformanceTracker()
    
    # Initialize benchmark learner
    learner = None
    if enable_learning:
        learner = BenchmarkLearner(brain_system=brain)
        print("✅ Benchmark learning enabled")
    
    print()
    
    # Run benchmarks
    all_results = {}
    all_detailed_results = []
    
    for benchmark_name in benchmarks.keys():
        print(f"\n{'='*70}")
        print(f"Running: {benchmark_name}")
        print(f"{'='*70}")
        
        try:
            # Run benchmark
            summary = framework.run_benchmark(
                benchmark_name=benchmark_name,
                max_questions=max_questions_per_benchmark,
                verbose=True
            )
            
            all_results[benchmark_name] = summary
            
            # Get detailed results
            detailed = [r for r in framework.results if r.benchmark_name == benchmark_name]
            all_detailed_results.extend(detailed)
            
            # Track performance
            tracker.add_metrics(summary)
            if learner:
                learner.track_performance(benchmark_name, summary)
            
            # Learn from results if enabled
            if learner and detailed:
                learning_result = learner.learn_from_feedback(detailed, update_weights=True)
                print(f"\nLearning: {learning_result['updates_applied']} updates applied")
            
            # Show comparison
            comparison = tracker.compare_with_baselines(benchmark_name, summary.accuracy)
            if comparison:
                print(f"\nComparison:")
                print(f"  Our Accuracy: {comparison.our_accuracy:.2%}")
                print(f"  Rank: {comparison.rank}/{len(comparison.baselines) + 1}")
                if comparison.improvement_over_baseline is not None:
                    if comparison.improvement_over_baseline > 0:
                        print(f"  Improvement: +{comparison.improvement_over_baseline:.2%}")
                    else:
                        print(f"  Gap: {comparison.improvement_over_baseline:.2%}")
            
        except Exception as e:
            print(f"❌ Error running {benchmark_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comprehensive report
    print("\n" + "=" * 70)
    print("COMPREHENSIVE PERFORMANCE REPORT")
    print("=" * 70)
    
    report = tracker.generate_report()
    print(report)
    
    # Learning report if enabled
    if learner:
        print("\n" + "=" * 70)
        print("LEARNING REPORT")
        print("=" * 70)
        learning_report = learner.generate_learning_report()
        print(learning_report)
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    if all_results:
        total_questions = sum(r.total_questions for r in all_results.values())
        total_correct = sum(r.correct_answers for r in all_results.values())
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        print(f"Total Questions: {total_questions}")
        print(f"Total Correct: {total_correct}")
        print(f"Overall Accuracy: {overall_accuracy:.2%}")
        print()
        
        print("Per-Benchmark Results:")
        for name, summary in sorted(all_results.items()):
            print(f"  {name:15s}: {summary.accuracy:6.2%} ({summary.correct_answers:3d}/{summary.total_questions:3d})")
        
        # Check superhuman thresholds
        print("\n" + "=" * 70)
        print("SUPERHUMAN INTELLIGENCE ASSESSMENT")
        print("=" * 70)
        
        superhuman_thresholds = {
            'MMLU': 0.90,      # Human: ~89%
            'HellaSwag': 0.95,  # Human: ~96%
            'GSM8K': 0.90,     # Human: ~92%
            'ARC': 0.85,       # Human: ~85%
            'HumanEval': 0.85  # Human: ~100%
        }
        
        superhuman_count = 0
        for name, threshold in superhuman_thresholds.items():
            if name in all_results:
                accuracy = all_results[name].accuracy
                is_superhuman = accuracy >= threshold
                status = "✅ SUPERHUMAN" if is_superhuman else "❌ Below threshold"
                print(f"  {name:15s}: {accuracy:6.2%} (threshold: {threshold:.0%}) {status}")
                if is_superhuman:
                    superhuman_count += 1
        
        print(f"\nSuperhuman Benchmarks: {superhuman_count}/{len(superhuman_thresholds)}")
        
        if superhuman_count == len(superhuman_thresholds):
            print("🎉 ACHIEVED SUPERHUMAN INTELLIGENCE ON ALL BENCHMARKS!")
        elif superhuman_count > 0:
            print(f"✅ Achieved superhuman performance on {superhuman_count} benchmark(s)")
        else:
            print("📈 Continue improving to reach superhuman thresholds")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join("benchmark_results", f"comprehensive_results_{timestamp}.json")
    
    comprehensive_data = {
        'timestamp': timestamp,
        'neuron_count': neuron_count,
        'use_llm': use_llm,
        'enable_learning': enable_learning,
        'results': {name: {
            'accuracy': r.accuracy,
            'total_questions': r.total_questions,
            'correct_answers': r.correct_answers,
            'average_confidence': r.average_confidence,
            'average_response_time': r.average_response_time
        } for name, r in all_results.items()},
        'overall_accuracy': overall_accuracy if all_results else 0.0
    }
    
    os.makedirs("benchmark_results", exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(comprehensive_data, f, indent=2)
    
    print(f"\n💾 Comprehensive results saved to: {results_file}")
    print("\n✅ Comprehensive benchmark suite complete!")


def main():
    """Main function"""
    neuron_count = 10000
    use_llm = False
    max_questions = 20
    enable_learning = True
    
    # Parse arguments
    if len(sys.argv) > 1:
        try:
            neuron_count = int(sys.argv[1])
        except ValueError:
            pass
    
    if len(sys.argv) > 2:
        use_llm = sys.argv[2].lower() in ['true', '1', 'yes', 'on']
    
    if len(sys.argv) > 3:
        try:
            max_questions = int(sys.argv[3])
        except ValueError:
            pass
    
    run_comprehensive_suite(
        neuron_count=neuron_count,
        use_llm=use_llm,
        max_questions_per_benchmark=max_questions,
        enable_learning=enable_learning
    )


if __name__ == "__main__":
    main()

