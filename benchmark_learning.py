#!/usr/bin/env python3
"""
Benchmark Learning - Learn from benchmark feedback using plasticity mechanisms
Uses existing plasticity mechanisms to improve performance based on benchmark results
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from benchmark_framework import BenchmarkResult, BenchmarkSummary
import json
import os


class BenchmarkLearner:
    """Learn from benchmark feedback to improve performance"""
    
    def __init__(self, brain_system, learning_rate: float = 0.1):
        """
        Initialize benchmark learner
        
        Args:
            brain_system: FinalEnhancedBrain instance
            learning_rate: Learning rate for weight updates
        """
        self.brain_system = brain_system
        self.learning_rate = learning_rate
        
        # Learning history
        self.learning_history: List[Dict] = []
        
        # Performance tracking
        self.baseline_performance: Dict[str, float] = {}
        self.current_performance: Dict[str, float] = {}
        
        # Error patterns for learning
        self.error_patterns: List[Dict] = []
    
    def analyze_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """
        Analyze benchmark results to identify learning opportunities
        
        Args:
            results: List of benchmark results
            
        Returns:
            Analysis dictionary
        """
        analysis = {
            'total_questions': len(results),
            'correct': sum(1 for r in results if r.is_correct),
            'incorrect': sum(1 for r in results if not r.is_correct),
            'accuracy': sum(1 for r in results if r.is_correct) / len(results) if results else 0.0,
            'error_analysis': {},
            'learning_opportunities': []
        }
        
        # Analyze errors
        incorrect_results = [r for r in results if not r.is_correct]
        
        if incorrect_results:
            # Confidence analysis
            incorrect_confidences = [r.confidence for r in incorrect_results]
            analysis['error_analysis']['avg_confidence_on_errors'] = np.mean(incorrect_confidences)
            analysis['error_analysis']['overconfidence'] = np.mean(incorrect_confidences) > 0.7
            
            # Response time analysis
            incorrect_times = [r.response_time for r in incorrect_results]
            analysis['error_analysis']['avg_response_time_on_errors'] = np.mean(incorrect_times)
            
            # Pattern analysis
            analysis['error_analysis']['common_error_patterns'] = self._identify_error_patterns(incorrect_results)
        
        # Identify learning opportunities
        if incorrect_results:
            analysis['learning_opportunities'] = self._identify_learning_opportunities(incorrect_results)
        
        return analysis
    
    def _identify_error_patterns(self, incorrect_results: List[BenchmarkResult]) -> List[Dict]:
        """Identify common patterns in errors"""
        patterns = []
        
        # Group by benchmark name
        by_benchmark = {}
        for result in incorrect_results:
            bench = result.benchmark_name
            if bench not in by_benchmark:
                by_benchmark[bench] = []
            by_benchmark[bench].append(result)
        
        for bench, results in by_benchmark.items():
            if len(results) >= 2:  # Need at least 2 errors to identify pattern
                patterns.append({
                    'benchmark': bench,
                    'error_count': len(results),
                    'avg_confidence': np.mean([r.confidence for r in results]),
                    'common_categories': self._get_common_categories(results)
                })
        
        return patterns
    
    def _get_common_categories(self, results: List[BenchmarkResult]) -> List[str]:
        """Get common categories from results"""
        categories = []
        for result in results:
            if result.metadata and 'category' in result.metadata:
                categories.append(result.metadata['category'])
        
        # Count frequencies
        from collections import Counter
        category_counts = Counter(categories)
        return [cat for cat, count in category_counts.most_common(3)]
    
    def _identify_learning_opportunities(self, incorrect_results: List[BenchmarkResult]) -> List[str]:
        """Identify specific learning opportunities"""
        opportunities = []
        
        # High confidence errors (overconfidence)
        high_conf_errors = [r for r in incorrect_results if r.confidence > 0.7]
        if high_conf_errors:
            opportunities.append(f"Overconfidence: {len(high_conf_errors)} high-confidence errors")
        
        # Low confidence correct (underconfidence - though these are incorrect)
        low_conf_errors = [r for r in incorrect_results if r.confidence < 0.3]
        if low_conf_errors:
            opportunities.append(f"Low confidence errors: {len(low_conf_errors)} cases")
        
        # Slow responses
        slow_errors = [r for r in incorrect_results if r.response_time > 5.0]
        if slow_errors:
            opportunities.append(f"Slow responses: {len(slow_errors)} cases taking >5s")
        
        # Category-specific issues
        by_category = {}
        for result in incorrect_results:
            if result.metadata and 'category' in result.metadata:
                cat = result.metadata['category']
                if cat not in by_category:
                    by_category[cat] = 0
                by_category[cat] += 1
        
        for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True)[:3]:
            opportunities.append(f"Category '{cat}': {count} errors")
        
        return opportunities
    
    def learn_from_feedback(self, 
                           results: List[BenchmarkResult],
                           update_weights: bool = True) -> Dict[str, Any]:
        """
        Learn from benchmark feedback
        
        Args:
            results: List of benchmark results
            update_weights: Whether to update brain weights
            
        Returns:
            Learning results dictionary
        """
        learning_result = {
            'results_analyzed': len(results),
            'updates_applied': 0,
            'improvements': {}
        }
        
        # Analyze results
        analysis = self.analyze_results(results)
        learning_result['analysis'] = analysis
        
        if not update_weights:
            return learning_result
        
        # Apply learning updates
        correct_results = [r for r in results if r.is_correct]
        incorrect_results = [r for r in results if not r.is_correct]
        
        # Strengthen connections for correct answers
        if correct_results:
            updates = self._strengthen_correct_patterns(correct_results)
            learning_result['updates_applied'] += updates
        
        # Weaken or adjust connections for incorrect answers
        if incorrect_results:
            updates = self._adjust_incorrect_patterns(incorrect_results)
            learning_result['updates_applied'] += updates
        
        # Store learning event
        learning_event = {
            'timestamp': np.datetime64('now').astype(str),
            'results_analyzed': len(results),
            'accuracy': analysis['accuracy'],
            'updates_applied': learning_result['updates_applied']
        }
        self.learning_history.append(learning_event)
        
        return learning_result
    
    def _strengthen_correct_patterns(self, correct_results: List[BenchmarkResult]) -> int:
        """Strengthen patterns associated with correct answers"""
        updates = 0
        
        for result in correct_results:
            try:
                # Convert question to pattern
                question_pattern = self._text_to_pattern(result.question)
                
                # Use brain's plasticity mechanisms if available
                if hasattr(self.brain_system, 'pattern_system'):
                    # Strengthen pattern recognition
                    pattern_system = self.brain_system.pattern_system
                    
                    # Add to pattern memory with high strength
                    if 'pattern_memory' in pattern_system:
                        pattern_memory = pattern_system['pattern_memory']
                        
                        # Store successful pattern
                        pattern_entry = {
                            'pattern': question_pattern.tolist(),
                            'strength': 1.0,
                            'success_count': 1,
                            'metadata': {
                                'benchmark': result.benchmark_name,
                                'correct': True
                            }
                        }
                        
                        # Add or update in memory
                        pattern_memory.append(pattern_entry)
                        
                        # Limit memory size
                        if len(pattern_memory) > 100:
                            # Remove weakest patterns
                            pattern_memory.sort(key=lambda x: x.get('strength', 0))
                            pattern_memory = pattern_memory[-100:]
                            pattern_system['pattern_memory'] = pattern_memory
                        
                        updates += 1
                
            except Exception as e:
                # Skip if error
                continue
        
        return updates
    
    def _adjust_incorrect_patterns(self, incorrect_results: List[BenchmarkResult]) -> int:
        """Adjust patterns associated with incorrect answers"""
        updates = 0
        
        for result in incorrect_results:
            try:
                # Convert question to pattern
                question_pattern = self._text_to_pattern(result.question)
                
                # Store error pattern for analysis
                error_pattern = {
                    'pattern': question_pattern.tolist(),
                    'question': result.question,
                    'prediction': result.prediction,
                    'ground_truth': result.ground_truth,
                    'confidence': result.confidence,
                    'benchmark': result.benchmark_name
                }
                self.error_patterns.append(error_pattern)
                
                # If overconfident, reduce confidence calibration
                if result.confidence > 0.7:
                    # This would adjust confidence calibration in the brain
                    # For now, we just track it
                    pass
                
                updates += 1
                
            except Exception as e:
                # Skip if error
                continue
        
        return updates
    
    def _text_to_pattern(self, text: str) -> np.ndarray:
        """Convert text to pattern"""
        # Use language processor if available
        if hasattr(self.brain_system, 'lang_processor'):
            return self.brain_system.lang_processor.text_to_pattern(text)
        else:
            # Fallback: simple encoding
            from language_processor import LanguageProcessor
            processor = LanguageProcessor()
            return processor.text_to_pattern(text)
    
    def track_performance(self, benchmark_name: str, summary: BenchmarkSummary):
        """
        Track performance over time
        
        Args:
            benchmark_name: Name of benchmark
            summary: Benchmark summary
        """
        if benchmark_name not in self.baseline_performance:
            self.baseline_performance[benchmark_name] = summary.accuracy
        
        self.current_performance[benchmark_name] = summary.accuracy
    
    def get_learning_curve(self, benchmark_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get learning curve data
        
        Args:
            benchmark_name: Name of benchmark (None for all)
            
        Returns:
            Learning curve data
        """
        if benchmark_name:
            events = [e for e in self.learning_history if benchmark_name in str(e)]
        else:
            events = self.learning_history
        
        if len(events) < 2:
            return {'insufficient_data': True, 'events': len(events)}
        
        accuracies = [e.get('accuracy', 0.0) for e in events]
        
        return {
            'num_events': len(events),
            'accuracies': accuracies,
            'improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0.0,
            'trend': 'improving' if len(accuracies) > 1 and accuracies[-1] > accuracies[0] else 'stable'
        }
    
    def generate_learning_report(self) -> str:
        """Generate learning report"""
        report = []
        report.append("=" * 70)
        report.append("BENCHMARK LEARNING REPORT")
        report.append("=" * 70)
        report.append("")
        
        if not self.learning_history:
            report.append("No learning events recorded yet.")
            return "\n".join(report)
        
        report.append(f"Total Learning Events: {len(self.learning_history)}")
        report.append("")
        
        # Performance tracking
        if self.current_performance:
            report.append("Performance Tracking:")
            for bench, accuracy in self.current_performance.items():
                baseline = self.baseline_performance.get(bench, accuracy)
                improvement = accuracy - baseline
                report.append(f"  {bench}: {accuracy:.2%} (baseline: {baseline:.2%}, improvement: {improvement:+.2%})")
            report.append("")
        
        # Learning curves
        report.append("Learning Curves:")
        for bench in set(self.current_performance.keys()):
            curve = self.get_learning_curve(bench)
            if not curve.get('insufficient_data'):
                report.append(f"  {bench}: {curve['trend']} ({curve['improvement']:+.2%})")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_learning_data(self, filename: str = "benchmark_learning_data.json"):
        """Save learning data to file"""
        data = {
            'learning_history': self.learning_history,
            'baseline_performance': self.baseline_performance,
            'current_performance': self.current_performance,
            'error_patterns': self.error_patterns[-100:]  # Keep last 100
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_learning_data(self, filename: str = "benchmark_learning_data.json"):
        """Load learning data from file"""
        if not os.path.exists(filename):
            return
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.learning_history = data.get('learning_history', [])
            self.baseline_performance = data.get('baseline_performance', {})
            self.current_performance = data.get('current_performance', {})
            self.error_patterns = data.get('error_patterns', [])
            
        except Exception as e:
            print(f"⚠️  Error loading learning data: {e}")

