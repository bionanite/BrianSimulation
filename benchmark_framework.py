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
        
        # Initialize advanced reasoning if not already attached
        if not hasattr(brain_system, 'advanced_reasoning'):
            try:
                from advanced_reasoning import AdvancedReasoning
                brain_system.advanced_reasoning = AdvancedReasoning(brain_system=brain_system)
            except ImportError:
                brain_system.advanced_reasoning = None
        
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
    
    def _extract_answer_from_brain(self,
                                   question: str,
                                   question_pattern: np.ndarray,
                                   pattern_result: Dict,
                                   reasoning_result: Optional[Dict],
                                   item: Dict,
                                   adapter: BenchmarkAdapter) -> str:
        """
        Extract answer from brain output, handling multiple-choice questions
        
        Args:
            question: Formatted question text
            question_pattern: Neural pattern of question
            pattern_result: Pattern recognition result
            reasoning_result: Reasoning result (if available)
            item: Original benchmark item
            adapter: Benchmark adapter
            
        Returns:
            Predicted answer string
        """
        # CRITICAL: Check if multiple-choice FIRST before extracting numbers
        import re
        has_multiple_choice = bool(re.search(r'[A-D]\)', question))
        
        # Try to use advanced reasoning if available
        if hasattr(self.brain_system, 'advanced_reasoning') and self.brain_system.advanced_reasoning:
            try:
                reasoning_chain = self.brain_system.advanced_reasoning.chain_of_thought(
                    question, max_steps=3, verbose=False
                )
                conclusion = reasoning_chain.get('conclusion', '')
                
                # Try to extract answer from conclusion
                if conclusion:
                    # For multiple-choice, ONLY extract letters, NEVER numbers
                    if has_multiple_choice:
                        choice_match = re.search(r'\b([A-D])\b', conclusion.upper())
                        if choice_match:
                            return choice_match.group(1)
                        # Don't extract numbers for multiple-choice!
                    else:
                        # For math questions, extract numbers
                        num_match = re.search(r'\b(\d+)\b', conclusion)
                        if num_match:
                            return num_match.group(1)
                    
                    # Use conclusion as-is if it seems like an answer (non-multiple-choice only)
                    if len(conclusion) < 100 and not has_multiple_choice:
                        return conclusion
            except Exception as e:
                pass  # Fall back to other methods
        
        # Try to extract choices from question for multiple-choice
        choices = []
        # Check multiple possible fields for choices
        if 'choices' in item:
            if isinstance(item['choices'], list):
                choices = item['choices']
            elif isinstance(item['choices'], dict) and 'text' in item['choices']:
                choices = item['choices']['text']
        elif 'endings' in item:
            if isinstance(item['endings'], list):
                choices = item['endings']
        
        # Also try to extract from formatted question text (as fallback)
        if not choices:
            import re
            # Look for A), B), C), D) patterns in question
            choice_pattern = r'[A-D]\)\s+([^\n]+)'
            matches = re.findall(choice_pattern, question)
            if matches:
                choices = matches
        
        # Debug: ensure we have choices for multiple-choice questions
        # If question contains "A)", "B)", etc., we should have choices
        if not choices and ('A)' in question or 'B)' in question):
            import re
            # More aggressive extraction
            lines = question.split('\n')
            for line in lines:
                if re.match(r'^[A-D]\)', line.strip()):
                    choice_text = line.split(')', 1)[1].strip() if ')' in line else line.strip()
                    if choice_text:
                        if not choices:
                            choices = []
                        choices.append(choice_text)
        
        # If we have choices, try to select best match
        if choices:
            # Use pattern similarity to select best choice
            best_choice_idx = 0
            best_similarity = -1.0
            similarities = []
            
            for idx, choice_text in enumerate(choices):
                if isinstance(choice_text, str):
                    # Convert choice to pattern
                    choice_pattern = self.text_to_pattern(choice_text)
                    
                    # Calculate similarity with question pattern
                    dot_product = np.dot(question_pattern, choice_pattern)
                    norm_q = np.linalg.norm(question_pattern)
                    norm_c = np.linalg.norm(choice_pattern)
                    
                    if norm_q > 0 and norm_c > 0:
                        similarity = dot_product / (norm_q * norm_c)
                    else:
                        similarity = 0.0
                    
                    # Boost similarity if reasoning result mentions this choice
                    if reasoning_result:
                        conclusion = str(reasoning_result.get('logical_conclusion', '')).lower()
                        choice_lower = choice_text.lower()
                        # Check for keyword matches
                        conclusion_words = set(conclusion.split())
                        choice_words = set(choice_lower.split()[:5])  # First 5 words
                        if len(conclusion_words & choice_words) > 0:
                            similarity += 0.3  # Boost for keyword matches
                    
                    similarities.append((idx, similarity, choice_text))
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_choice_idx = idx
            
            # If all similarities are very low, use random selection as fallback
            # But prefer the one with highest similarity
            if best_similarity < 0.1 and len(similarities) > 0:
                # Sort by similarity and pick top one
                similarities.sort(key=lambda x: x[1], reverse=True)
                best_choice_idx = similarities[0][0]
                best_similarity = similarities[0][1]
            
            # Return choice letter (A, B, C, D)
            # Always return a letter, even if similarity is low (better than returning "3")
            # Ensure index is valid
            if best_choice_idx >= len(choices):
                best_choice_idx = len(choices) - 1
            if best_choice_idx < 0:
                best_choice_idx = 0
            return chr(65 + best_choice_idx)  # A, B, C, D
        
        # Fallback: use reasoning conclusion or pattern recognition
        # has_multiple_choice already checked above at line 193
        if reasoning_result and 'logical_conclusion' in reasoning_result:
            conclusion = str(reasoning_result['logical_conclusion'])
            
            # For multiple-choice questions, prioritize letter extraction
            if has_multiple_choice:
                # Look for answer patterns - prioritize letters
                letter_match = re.search(r'\b([A-D])\b', conclusion.upper())
                if letter_match:
                    return letter_match.group(1)
                
                # Try to match conclusion to choices if we have them
                if choices:
                    conclusion_lower = conclusion.lower()
                    for idx, choice in enumerate(choices):
                        if isinstance(choice, str):
                            choice_lower = choice.lower()
                            # Check if conclusion contains key words from choice
                            choice_words = set(choice_lower.split()[:3])
                            conclusion_words = set(conclusion_lower.split())
                            if len(choice_words & conclusion_words) >= 1:
                                return chr(65 + idx)  # Return letter
                    # If no match, return first choice letter as fallback
                    return chr(65 + 0)  # Return "A"
                # If no choices and multiple-choice, return "A" as safe default
                return chr(65 + 0)  # Return "A"
            
            # For math questions (non-multiple-choice), extract numbers
            if not has_multiple_choice:
                num_match = re.search(r'\b(\d+)\b', conclusion)
                if num_match:
                    return num_match.group(1)
            
            # If conclusion is short and seems like an answer (non-multiple-choice)
            if len(conclusion) < 100 and len(conclusion) > 0 and not has_multiple_choice:
                return conclusion[:50]  # Limit length
        
        # Final fallback: has_multiple_choice already checked above
        
        if has_multiple_choice:
            # Multiple-choice question - MUST return letter, not number
            # Use similarity-based selection if we have any choices extracted
            if len(choices) > 0:
                # We already tried similarity above, but if we got here, use first choice
                return chr(65 + 0)  # Return "A" as safe default
            else:
                # No choices extracted but question has A), B), etc.
                # Extract choices directly from question text
                lines = question.split('\n')
                extracted_choices = []
                for line in lines:
                    line = line.strip()
                    if re.match(r'^[A-D]\)', line):
                        # Extract text after "A) " or "A)"
                        match = re.match(r'^[A-D]\)\s*(.+)', line)
                        if match:
                            extracted_choices.append(match.group(1))
                
                if len(extracted_choices) > 0:
                    # Use similarity to select best choice
                    best_idx = 0
                    best_sim = -1.0
                    for idx, choice_text in enumerate(extracted_choices):
                        choice_pattern = self.text_to_pattern(choice_text)
                        dot = np.dot(question_pattern, choice_pattern)
                        norm_q = np.linalg.norm(question_pattern)
                        norm_c = np.linalg.norm(choice_pattern)
                        if norm_q > 0 and norm_c > 0:
                            sim = dot / (norm_q * norm_c)
                            if sim > best_sim:
                                best_sim = sim
                                best_idx = idx
                    return chr(65 + best_idx)  # Return letter
                else:
                    # Last resort: random letter
                    import random
                    return chr(65 + random.randint(0, 3))  # Random A-D
        
        # For non-multiple-choice (like GSM8K math), try to extract number
        # Only extract numbers if NOT multiple-choice
        if not has_multiple_choice and reasoning_result and 'logical_conclusion' in reasoning_result:
            conclusion = str(reasoning_result['logical_conclusion'])
            num_match = re.search(r'\b(\d+)\b', conclusion)
            if num_match:
                return num_match.group(1)
        
        # Final fallback
        # If multiple-choice and we got here, return "A" as safe default
        if has_multiple_choice:
            return chr(65 + 0)  # Return "A"
        
        return self.pattern_to_text(question_pattern)
    
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
            
            # Extract prediction with improved answer selection
            # Pass both formatted question AND original item for choice extraction
            prediction = self._extract_answer_from_brain(
                question, 
                question_pattern,
                pattern_result,
                reasoning_result,
                item,  # Original item with choices/endings
                adapter
            )
            
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

