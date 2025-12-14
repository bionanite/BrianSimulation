#!/usr/bin/env python3
"""
Run Benchmarks - Execute benchmark tests and generate reports
"""

import sys
import os
from benchmark_framework import BenchmarkFramework
from benchmark_adapters import (
    MMLUAdapter, HellaSwagAdapter, GSM8KAdapter, 
    ARCAdapter, HumanEvalAdapter
)
from performance_tracking import PerformanceTracker
from final_enhanced_brain import FinalEnhancedBrain


def main():
    """Main function to run benchmarks"""
    print("=" * 70)
    print("BENCHMARK VALIDATION SYSTEM")
    print("=" * 70)
    print()
    
    # Initialize brain system
    print("Initializing brain system...")
    neuron_count = 10000  # Start with 10K neurons
    if len(sys.argv) > 1:
        try:
            neuron_count = int(sys.argv[1])
        except ValueError:
            print(f"⚠️  Invalid neuron count, using default: {neuron_count}")
    
    brain = FinalEnhancedBrain(total_neurons=neuron_count, debug=False)
    print(f"✅ Brain initialized with {neuron_count:,} neurons")
    print()
    
    # Initialize benchmark framework
    framework = BenchmarkFramework(brain_system=brain)
    
    # Initialize performance tracker
    tracker = PerformanceTracker()
    
    # Register benchmark adapters
    print("Registering benchmark adapters...")
    framework.register_adapter("MMLU", MMLUAdapter())
    framework.register_adapter("HellaSwag", HellaSwagAdapter())
    framework.register_adapter("GSM8K", GSM8KAdapter())
    framework.register_adapter("ARC", ARCAdapter())
    framework.register_adapter("HumanEval", HumanEvalAdapter())
    print()
    
    # Run benchmarks
    benchmarks_to_run = []
    
    if len(sys.argv) > 2:
        # Run specific benchmark
        benchmark_name = sys.argv[2]
        if benchmark_name in framework.adapters:
            benchmarks_to_run = [benchmark_name]
        else:
            print(f"❌ Unknown benchmark: {benchmark_name}")
            print(f"Available: {list(framework.adapters.keys())}")
            return
    else:
        # Run all benchmarks
        benchmarks_to_run = list(framework.adapters.keys())
    
    # Limit questions for initial testing
    max_questions = 10  # Can be increased later
    
    print(f"Running {len(benchmarks_to_run)} benchmark(s)...")
    print(f"Questions per benchmark: {max_questions}")
    print()
    
    results = {}
    
    for benchmark_name in benchmarks_to_run:
        try:
            print(f"\n{'='*70}")
            print(f"Running: {benchmark_name}")
            print(f"{'='*70}")
            
            summary = framework.run_benchmark(
                benchmark_name=benchmark_name,
                max_questions=max_questions,
                verbose=True
            )
            
            results[benchmark_name] = summary
            
            # Add to performance tracker
            tracker.add_metrics(summary)
            
            # Show comparison
            comparison = tracker.compare_with_baselines(benchmark_name, summary.accuracy)
            if comparison:
                print(f"\nComparison with Baselines:")
                print(f"  Our Accuracy: {comparison.our_accuracy:.2%}")
                print(f"  Rank: {comparison.rank}/{len(comparison.baselines) + 1}")
                print(f"  Percentile: {comparison.percentile:.1f}%")
                if comparison.improvement_over_baseline is not None:
                    if comparison.improvement_over_baseline > 0:
                        print(f"  Improvement over best AI: +{comparison.improvement_over_baseline:.2%}")
                    else:
                        print(f"  Gap to best AI: {comparison.improvement_over_baseline:.2%}")
                
                print(f"\n  Baseline Scores:")
                for model, score in sorted(comparison.baselines.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {model}: {score:.2%}")
            
        except Exception as e:
            print(f"❌ Error running {benchmark_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comprehensive report
    print("\n" + "=" * 70)
    print("PERFORMANCE REPORT")
    print("=" * 70)
    report = tracker.generate_report()
    print(report)
    
    # Save report
    report_file = os.path.join("benchmark_results", "performance_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\n💾 Full report saved to: {report_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if results:
        avg_accuracy = sum(r.accuracy for r in results.values()) / len(results)
        print(f"Average Accuracy Across Benchmarks: {avg_accuracy:.2%}")
        print(f"Benchmarks Completed: {len(results)}")
        
        print("\nPer-Benchmark Results:")
        for name, summary in results.items():
            print(f"  {name}: {summary.accuracy:.2%} ({summary.correct_answers}/{summary.total_questions})")
    else:
        print("No benchmarks completed successfully")
    
    print("\n✅ Benchmark testing complete!")


if __name__ == "__main__":
    main()

