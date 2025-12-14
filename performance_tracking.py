#!/usr/bin/env python3
"""
Performance Tracking - Track benchmark scores over time and compare with baselines
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

from benchmark_framework import BenchmarkSummary


@dataclass
class PerformanceMetrics:
    """Performance metrics for tracking"""
    timestamp: str
    benchmark_name: str
    accuracy: float
    average_confidence: float
    average_response_time: float
    total_questions: int
    correct_answers: int
    per_category_accuracy: Optional[Dict[str, float]] = None
    metadata: Optional[Dict] = None


@dataclass
class BaselineComparison:
    """Comparison with baseline models"""
    benchmark_name: str
    our_accuracy: float
    baselines: Dict[str, float]
    rank: int
    percentile: float
    improvement_over_baseline: Optional[float] = None


class PerformanceTracker:
    """Track performance over time and compare with baselines"""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        """
        Initialize performance tracker
        
        Args:
            results_dir: Directory containing benchmark results
        """
        self.results_dir = results_dir
        self.metrics_history: List[PerformanceMetrics] = []
        self.comparisons: List[BaselineComparison] = []
        
        # Known baselines (approximate values - update with real data)
        self.baselines = {
            'MMLU': {
                'human': 0.897,
                'gpt4': 0.863,
                'claude3_opus': 0.850,
                'claude3_sonnet': 0.840,
                'gemini_pro': 0.830,
                'gpt3.5': 0.700
            },
            'HellaSwag': {
                'human': 0.956,
                'gpt4': 0.950,
                'claude3_opus': 0.945,
                'claude3_sonnet': 0.940,
                'gemini_pro': 0.940,
                'gpt3.5': 0.850
            },
            'GSM8K': {
                'human': 0.920,
                'gpt4': 0.920,
                'claude3_opus': 0.915,
                'claude3_sonnet': 0.900,
                'gemini_pro': 0.900,
                'gpt3.5': 0.570
            },
            'ARC': {
                'human': 0.850,
                'gpt4': 0.850,
                'claude3_opus': 0.840,
                'claude3_sonnet': 0.830,
                'gemini_pro': 0.830,
                'gpt3.5': 0.510
            },
            'HumanEval': {
                'human': 1.000,
                'gpt4': 0.670,
                'claude3_opus': 0.650,
                'claude3_sonnet': 0.600,
                'gemini_pro': 0.550,
                'gpt3.5': 0.480
            }
        }
        
        # Load existing metrics if available
        self.load_metrics()
    
    def add_metrics(self, summary: BenchmarkSummary):
        """
        Add performance metrics from a benchmark summary
        
        Args:
            summary: BenchmarkSummary object
        """
        metrics = PerformanceMetrics(
            timestamp=summary.metadata.get('timestamp', datetime.now().isoformat()) if summary.metadata else datetime.now().isoformat(),
            benchmark_name=summary.benchmark_name,
            accuracy=summary.accuracy,
            average_confidence=summary.average_confidence,
            average_response_time=summary.average_response_time,
            total_questions=summary.total_questions,
            correct_answers=summary.correct_answers,
            per_category_accuracy=summary.per_category_accuracy,
            metadata=summary.metadata
        )
        
        self.metrics_history.append(metrics)
        self.save_metrics()
        
        # Generate comparison
        comparison = self.compare_with_baselines(summary.benchmark_name, summary.accuracy)
        if comparison:
            self.comparisons.append(comparison)
    
    def compare_with_baselines(self, benchmark_name: str, our_accuracy: float) -> Optional[BaselineComparison]:
        """
        Compare our performance with baseline models
        
        Args:
            benchmark_name: Name of benchmark
            our_accuracy: Our accuracy score
            
        Returns:
            BaselineComparison object
        """
        if benchmark_name not in self.baselines:
            return None
        
        baselines = self.baselines[benchmark_name]
        
        # Calculate rank
        all_scores = [(name, score) for name, score in baselines.items()]
        all_scores.append(('ours', our_accuracy))
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        rank = next((i + 1 for i, (name, _) in enumerate(all_scores) if name == 'ours'), len(all_scores))
        
        # Calculate percentile
        max_score = max(baselines.values())
        min_score = min(baselines.values())
        if max_score > min_score:
            percentile = (our_accuracy - min_score) / (max_score - min_score) * 100
        else:
            percentile = 50.0
        
        # Calculate improvement over best baseline (excluding human)
        ai_baselines = {k: v for k, v in baselines.items() if k != 'human'}
        if ai_baselines:
            best_ai = max(ai_baselines.values())
            improvement = our_accuracy - best_ai
        else:
            improvement = None
        
        return BaselineComparison(
            benchmark_name=benchmark_name,
            our_accuracy=our_accuracy,
            baselines=baselines,
            rank=rank,
            percentile=percentile,
            improvement_over_baseline=improvement
        )
    
    def get_latest_metrics(self, benchmark_name: Optional[str] = None) -> List[PerformanceMetrics]:
        """
        Get latest metrics for a benchmark
        
        Args:
            benchmark_name: Name of benchmark (None for all)
            
        Returns:
            List of PerformanceMetrics
        """
        if benchmark_name:
            return [m for m in self.metrics_history if m.benchmark_name == benchmark_name]
        return self.metrics_history
    
    def get_trend(self, benchmark_name: str) -> Dict[str, Any]:
        """
        Get performance trend over time
        
        Args:
            benchmark_name: Name of benchmark
            
        Returns:
            Dictionary with trend data
        """
        metrics = [m for m in self.metrics_history if m.benchmark_name == benchmark_name]
        
        if len(metrics) < 2:
            return {'trend': 'insufficient_data', 'metrics': len(metrics)}
        
        # Sort by timestamp
        metrics.sort(key=lambda x: x.timestamp)
        
        # Calculate trend
        accuracies = [m.accuracy for m in metrics]
        first_acc = accuracies[0]
        last_acc = accuracies[-1]
        
        if last_acc > first_acc:
            trend = 'improving'
            improvement = last_acc - first_acc
        elif last_acc < first_acc:
            trend = 'declining'
            improvement = last_acc - first_acc
        else:
            trend = 'stable'
            improvement = 0.0
        
        # Calculate average improvement rate
        if len(accuracies) > 1:
            improvements = [accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))]
            avg_improvement = np.mean(improvements) if improvements else 0.0
        else:
            avg_improvement = 0.0
        
        return {
            'trend': trend,
            'first_accuracy': first_acc,
            'last_accuracy': last_acc,
            'total_improvement': improvement,
            'average_improvement_per_run': avg_improvement,
            'num_runs': len(metrics),
            'accuracies': accuracies
        }
    
    def identify_capability_gaps(self) -> Dict[str, List[str]]:
        """
        Identify capability gaps based on performance
        
        Returns:
            Dictionary mapping benchmark names to list of gaps
        """
        gaps = {}
        
        for benchmark_name in self.baselines.keys():
            latest_metrics = self.get_latest_metrics(benchmark_name)
            if not latest_metrics:
                continue
            
            latest = latest_metrics[-1]
            baseline_scores = self.baselines[benchmark_name]
            human_score = baseline_scores.get('human', 1.0)
            best_ai_score = max([v for k, v in baseline_scores.items() if k != 'human'], default=0.0)
            
            gap_list = []
            
            # Check if below human performance
            if latest.accuracy < human_score * 0.9:
                gap_list.append(f"Below human performance (need {human_score:.1%}, have {latest.accuracy:.1%})")
            
            # Check if below best AI
            if latest.accuracy < best_ai_score * 0.9:
                gap_list.append(f"Below best AI performance (need {best_ai_score:.1%}, have {latest.accuracy:.1%})")
            
            # Check confidence vs accuracy
            if latest.average_confidence > latest.accuracy + 0.2:
                gap_list.append("Overconfident - confidence higher than accuracy")
            elif latest.average_confidence < latest.accuracy - 0.2:
                gap_list.append("Underconfident - accuracy higher than confidence")
            
            # Check response time
            if latest.average_response_time > 10.0:
                gap_list.append(f"Slow response time ({latest.average_response_time:.2f}s)")
            
            if gap_list:
                gaps[benchmark_name] = gap_list
        
        return gaps
    
    def generate_report(self) -> str:
        """
        Generate comprehensive performance report
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("PERFORMANCE TRACKING REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall summary
        if self.metrics_history:
            report.append("OVERALL SUMMARY")
            report.append("-" * 70)
            total_runs = len(self.metrics_history)
            benchmarks_tested = len(set(m.benchmark_name for m in self.metrics_history))
            avg_accuracy = np.mean([m.accuracy for m in self.metrics_history])
            
            report.append(f"Total Benchmark Runs: {total_runs}")
            report.append(f"Benchmarks Tested: {benchmarks_tested}")
            report.append(f"Average Accuracy: {avg_accuracy:.2%}")
            report.append("")
        
        # Per-benchmark performance
        if self.metrics_history:
            report.append("PER-BENCHMARK PERFORMANCE")
            report.append("-" * 70)
            
            benchmarks = set(m.benchmark_name for m in self.metrics_history)
            for bench_name in sorted(benchmarks):
                metrics = self.get_latest_metrics(bench_name)
                if not metrics:
                    continue
                
                latest = metrics[-1]
                report.append(f"\n{bench_name}:")
                report.append(f"  Latest Accuracy: {latest.accuracy:.2%}")
                report.append(f"  Questions Tested: {latest.total_questions}")
                report.append(f"  Average Confidence: {latest.average_confidence:.3f}")
                report.append(f"  Average Response Time: {latest.average_response_time:.3f}s")
                
                # Show comparison if available
                comparison = next((c for c in self.comparisons if c.benchmark_name == bench_name), None)
                if comparison:
                    report.append(f"  Rank: {comparison.rank}/{len(comparison.baselines) + 1}")
                    report.append(f"  Percentile: {comparison.percentile:.1f}%")
                    if comparison.improvement_over_baseline is not None:
                        if comparison.improvement_over_baseline > 0:
                            report.append(f"  Improvement over best AI: +{comparison.improvement_over_baseline:.2%}")
                        else:
                            report.append(f"  Gap to best AI: {comparison.improvement_over_baseline:.2%}")
                
                # Show trend
                trend = self.get_trend(bench_name)
                if trend.get('num_runs', 0) > 1:
                    report.append(f"  Trend: {trend['trend']} ({trend['total_improvement']:+.2%})")
        
        # Capability gaps
        gaps = self.identify_capability_gaps()
        if gaps:
            report.append("")
            report.append("CAPABILITY GAPS")
            report.append("-" * 70)
            for bench_name, gap_list in gaps.items():
                report.append(f"\n{bench_name}:")
                for gap in gap_list:
                    report.append(f"  - {gap}")
        
        # Recommendations
        report.append("")
        report.append("RECOMMENDATIONS")
        report.append("-" * 70)
        
        if not self.metrics_history:
            report.append("  - Run initial benchmarks to establish baseline")
        else:
            if gaps:
                report.append("  - Focus on improving performance in identified gap areas")
            else:
                report.append("  - Performance looks good! Consider scaling to larger test sets")
            
            avg_acc = np.mean([m.accuracy for m in self.metrics_history])
            if avg_acc < 0.5:
                report.append("  - Current performance is below 50% - focus on fundamental improvements")
            elif avg_acc < 0.8:
                report.append("  - Performance is good but can be improved - continue optimization")
            else:
                report.append("  - Excellent performance! Consider advanced capabilities")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_metrics(self):
        """Save metrics to file"""
        filename = os.path.join(self.results_dir, "performance_metrics.json")
        
        data = {
            'metrics': [asdict(m) for m in self.metrics_history],
            'comparisons': [asdict(c) for c in self.comparisons],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_metrics(self):
        """Load metrics from file"""
        filename = os.path.join(self.results_dir, "performance_metrics.json")
        
        if not os.path.exists(filename):
            return
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Load metrics
            self.metrics_history = [
                PerformanceMetrics(**m) for m in data.get('metrics', [])
            ]
            
            # Load comparisons
            self.comparisons = [
                BaselineComparison(**c) for c in data.get('comparisons', [])
            ]
            
        except Exception as e:
            print(f"⚠️  Error loading metrics: {e}")

