#!/usr/bin/env python3
"""
Benchmark Framework - Core benchmark testing infrastructure
Provides unified interface for running AI benchmarks and validating performance
"""

import numpy as np
import time
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Structure for benchmark test results"""
    benchmark_name: str
    task_name: str
    question: str
    ground_truth: Any
    prediction: Any
    is_correct: bool
    confidence: float
    response_time: float
    reasoning_steps: Optional[List[str]] = None
    metadata: Optional[Dict] = None


@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark run"""
    benchmark_name: str
    total_questions: int
    correct_answers: int
    accuracy: float
    average_confidence: float
    average_response_time: float
    total_time: float
    per_category_accuracy: Optional[Dict[str, float]] = None
    metadata: Optional[Dict] = None


class BenchmarkAdapter(ABC):
    """Abstract base class for benchmark adapters"""
    
    @abstractmethod
    def load_dataset(self) -> List[Dict]:
        """Load the benchmark dataset"""
        pass
    
    @abstractmethod
    def format_question(self, item: Dict) -> str:
        """Format a benchmark item as a question"""
        pass
    
    @abstractmethod
    def extract_answer(self, item: Dict) -> Any:
        """Extract the ground truth answer from benchmark item"""
        pass
    
    @abstractmethod
    def evaluate_answer(self, prediction: Any, ground_truth: Any) -> bool:
        """Evaluate if prediction matches ground truth"""
        pass
    
    @abstractmethod
    def format_response(self, prediction: Any) -> str:
        """Format the brain's response for evaluation"""
        pass


class BenchmarkFramework:
    """Core framework for running benchmarks"""
    
    def __init__(self, brain_system, results_dir: str = "benchmark_results"):
        """
        Initialize benchmark framework
        
        Args:
            brain_system: Instance of FinalEnhancedBrain or compatible system
            results_dir: Directory to save benchmark results
        """
        self.brain_system = brain_system
        self.results_dir = results_dir
        self.adapters: Dict[str, BenchmarkAdapter] = {}
        self.results: List[BenchmarkResult] = []
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Performance tracking
        self.performance_history: List[BenchmarkSummary] = []
    
    def register_adapter(self, name: str, adapter: BenchmarkAdapter):
        """Register a benchmark adapter"""
        self.adapters[name] = adapter
        print(f"✅ Registered benchmark adapter: {name}")
    
    def text_to_pattern(self, text: str, max_length: int = 1000) -> np.ndarray:
        """
        Convert text input to neural pattern
        
        Args:
            text: Input text
            max_length: Maximum pattern length
            
        Returns:
            Neural pattern array
        """
        # Simple text encoding: character frequencies + word features
        # This is a basic implementation - can be enhanced with proper embeddings
        
        # Character-level encoding
        chars = list(text.lower())
        char_freq = {}
        for char in chars:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        # Normalize frequencies
        if len(char_freq) > 0:
            max_freq = max(char_freq.values())
            char_features = [char_freq.get(chr(i), 0) / max_freq if max_freq > 0 else 0 
                           for i in range(ord('a'), ord('z') + 1)]
        else:
            char_features = [0.0] * 26
        
        # Word-level features (simple)
        words = text.lower().split()
        word_count = len(words)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Combine features
        pattern = np.array(char_features + [
            min(word_count / 100.0, 1.0),  # Normalized word count
            min(avg_word_length / 20.0, 1.0),  # Normalized avg word length
            len(text) / max_length  # Normalized text length
        ], dtype=np.float32)
        
        # Pad or truncate to max_length
        if len(pattern) < max_length:
            pattern = np.pad(pattern, (0, max_length - len(pattern)), mode='constant')
        else:
            pattern = pattern[:max_length]
        
        return pattern
    
    def pattern_to_text(self, pattern: np.ndarray) -> str:
        """
        Convert neural pattern back to text (simplified)
        
        Args:
            pattern: Neural pattern array
            
        Returns:
            Text representation
        """
        # This is a simplified conversion - in practice, would use proper decoder
        # For now, return a description of the pattern
        
        # Extract key features
        if len(pattern) > 0:
            max_val = np.max(pattern)
            mean_val = np.mean(pattern)
            std_val = np.std(pattern)
            
            # Simple heuristic: high values indicate positive response
            if max_val > 0.7:
                return "positive"
            elif mean_val > 0.5:
                return "moderate"
            else:
                return "negative"
        return "unknown"
    
    def run_benchmark(self, 
                     benchmark_name: str,
                     max_questions: Optional[int] = None,
                     verbose: bool = True) -> BenchmarkSummary:
        """
        Run a benchmark test
        
        Args:
            benchmark_name: Name of registered benchmark adapter
            max_questions: Maximum number of questions to test (None = all)
            verbose: Print progress information
            
        Returns:
            BenchmarkSummary with results
        """
        if benchmark_name not in self.adapters:
            raise ValueError(f"Benchmark '{benchmark_name}' not registered")
        
        adapter = self.adapters[benchmark_name]
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Running Benchmark: {benchmark_name}")
            print(f"{'='*70}")
        
        # Load dataset
        if verbose:
            print("Loading dataset...")
        
        try:
            dataset = adapter.load_dataset()
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
        
        if max_questions:
            dataset = dataset[:max_questions]
        
        total_questions = len(dataset)
        if verbose:
            print(f"Testing on {total_questions} questions")
        
        # Run tests
        correct_answers = 0
        total_confidence = 0.0
        total_response_time = 0.0
        per_category = {}
        
        start_time = time.time()
        
        for i, item in enumerate(dataset):
            if verbose and (i + 1) % max(1, total_questions // 10) == 0:
                progress = (i + 1) / total_questions * 100
                print(f"Progress: {progress:.1f}% ({i+1}/{total_questions})")
            
            # Format question
            question = adapter.format_question(item)
            ground_truth = adapter.extract_answer(item)
            
            # Convert question to pattern
            question_pattern = self.text_to_pattern(question)
            
            # Get brain response
            question_start = time.time()
            
            # Use brain's pattern recognition and reasoning
            pattern_result = self.brain_system.enhanced_pattern_recognition(question_pattern)
            
            # Use reasoning if available
            reasoning_result = None
            if hasattr(self.brain_system, 'reasoning'):
                context = {
                    'sensory_input': question_pattern,
                    'pattern_result': pattern_result
                }
                reasoning_result = self.brain_system.reasoning(context)
            
            # Extract prediction
            if reasoning_result and 'logical_conclusion' in reasoning_result:
                prediction = reasoning_result['logical_conclusion']
            elif pattern_result and 'pattern_recognized' in pattern_result:
                prediction = pattern_result['pattern_recognized']
            else:
                # Fallback: use pattern features
                prediction = self.pattern_to_text(question_pattern)
            
            response_time = time.time() - question_start
            
            # Format prediction for evaluation
            formatted_prediction = adapter.format_response(prediction)
            
            # Evaluate
            is_correct = adapter.evaluate_answer(formatted_prediction, ground_truth)
            
            # Calculate confidence
            if reasoning_result and 'confidence' in reasoning_result:
                confidence = reasoning_result['confidence']
            elif pattern_result and 'confidence' in pattern_result:
                confidence = pattern_result['confidence']
            else:
                confidence = 0.5
            
            if is_correct:
                correct_answers += 1
            
            total_confidence += confidence
            total_response_time += response_time
            
            # Track by category if available
            category = item.get('category', 'unknown')
            if category not in per_category:
                per_category[category] = {'correct': 0, 'total': 0}
            per_category[category]['total'] += 1
            if is_correct:
                per_category[category]['correct'] += 1
            
            # Store result
            result = BenchmarkResult(
                benchmark_name=benchmark_name,
                task_name=item.get('task', 'unknown'),
                question=question,
                ground_truth=str(ground_truth),
                prediction=str(formatted_prediction),
                is_correct=is_correct,
                confidence=confidence,
                response_time=response_time,
                reasoning_steps=reasoning_result.get('reasoning_steps') if reasoning_result else None,
                metadata={'category': category}
            )
            self.results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate summary
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        avg_confidence = total_confidence / total_questions if total_questions > 0 else 0.0
        avg_response_time = total_response_time / total_questions if total_questions > 0 else 0.0
        
        # Per-category accuracy
        per_category_accuracy = {}
        for cat, stats in per_category.items():
            if stats['total'] > 0:
                per_category_accuracy[cat] = stats['correct'] / stats['total']
        
        summary = BenchmarkSummary(
            benchmark_name=benchmark_name,
            total_questions=total_questions,
            correct_answers=correct_answers,
            accuracy=accuracy,
            average_confidence=avg_confidence,
            average_response_time=avg_response_time,
            total_time=total_time,
            per_category_accuracy=per_category_accuracy if per_category_accuracy else None,
            metadata={'timestamp': datetime.now().isoformat()}
        )
        
        self.performance_history.append(summary)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Benchmark Results: {benchmark_name}")
            print(f"{'='*70}")
            print(f"Accuracy: {accuracy:.2%} ({correct_answers}/{total_questions})")
            print(f"Average Confidence: {avg_confidence:.3f}")
            print(f"Average Response Time: {avg_response_time:.3f}s")
            print(f"Total Time: {total_time:.2f}s")
            if per_category_accuracy:
                print(f"\nPer-Category Accuracy:")
                for cat, acc in sorted(per_category_accuracy.items()):
                    print(f"  {cat}: {acc:.2%}")
        
        # Save results
        self.save_results(benchmark_name, summary)
        
        return summary
    
    def save_results(self, benchmark_name: str, summary: BenchmarkSummary):
        """Save benchmark results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.results_dir, f"{benchmark_name}_{timestamp}.json")
        
        data = {
            'summary': asdict(summary),
            'detailed_results': [asdict(r) for r in self.results if r.benchmark_name == benchmark_name]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")
    
    def compare_with_baselines(self, benchmark_name: str) -> Dict:
        """
        Compare results with known baselines (GPT-4, Claude, etc.)
        
        Args:
            benchmark_name: Name of benchmark
            
        Returns:
            Comparison dictionary
        """
        # Get latest summary for this benchmark
        summaries = [s for s in self.performance_history if s.benchmark_name == benchmark_name]
        if not summaries:
            return {}
        
        latest = summaries[-1]
        
        # Known baselines (approximate - would be updated with real data)
        baselines = {
            'MMLU': {
                'human': 0.897,
                'gpt4': 0.863,
                'claude3': 0.850,
                'gemini': 0.830
            },
            'HellaSwag': {
                'human': 0.956,
                'gpt4': 0.950,
                'claude3': 0.945,
                'gemini': 0.940
            },
            'GSM8K': {
                'human': 0.920,
                'gpt4': 0.920,
                'claude3': 0.915,
                'gemini': 0.900
            },
            'ARC': {
                'human': 0.850,
                'gpt4': 0.850,
                'claude3': 0.840,
                'gemini': 0.830
            }
        }
        
        comparison = {
            'our_accuracy': latest.accuracy,
            'baselines': baselines.get(benchmark_name, {}),
            'comparison': {}
        }
        
        if benchmark_name in baselines:
            for model, baseline_acc in baselines[benchmark_name].items():
                diff = latest.accuracy - baseline_acc
                comparison['comparison'][model] = {
                    'baseline': baseline_acc,
                    'difference': diff,
                    'relative': diff / baseline_acc if baseline_acc > 0 else 0
                }
        
        return comparison
    
    def get_performance_report(self) -> str:
        """Generate a performance report"""
        report = []
        report.append("=" * 70)
        report.append("BENCHMARK PERFORMANCE REPORT")
        report.append("=" * 70)
        report.append("")
        
        for summary in self.performance_history:
            report.append(f"Benchmark: {summary.benchmark_name}")
            report.append(f"  Accuracy: {summary.accuracy:.2%}")
            report.append(f"  Questions: {summary.total_questions}")
            report.append(f"  Average Response Time: {summary.average_response_time:.3f}s")
            report.append("")
        
        return "\n".join(report)

