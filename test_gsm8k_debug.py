#!/usr/bin/env python3
"""
Test GSM8K Math Reasoning with Debug Output
Verifies that math reasoning fixes work correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_framework import BenchmarkFramework
from benchmark_adapters import GSM8KAdapter
from final_enhanced_brain import FinalEnhancedBrain

def test_gsm8k_math_reasoning():
    """Test GSM8K math reasoning with debug output"""
    print("=" * 70)
    print("GSM8K MATH REASONING DEBUG TEST")
    print("=" * 70)
    print()
    
    # Initialize brain system
    print("Initializing brain system...")
    brain = FinalEnhancedBrain(total_neurons=10000, debug=False)
    print("✅ Brain initialized")
    print()
    
    # Initialize benchmark framework with debug enabled
    framework = BenchmarkFramework(brain_system=brain)
    framework.debug = True  # Enable debug mode
    
    # Register GSM8K adapter
    framework.register_adapter("GSM8K", GSM8KAdapter())
    
    # Test with simple problems first
    print("Testing with simple math problems...")
    print()
    
    # Create test questions
    test_questions = [
        {
            'question': 'Janet has 5 apples. She gives away 2 apples. How many apples does she have left?',
            'answer': '3',
            'expected': '3'
        },
        {
            'question': 'There are 3 boxes. Each box has 4 apples. How many apples are there in total?',
            'answer': '12',
            'expected': '12'
        },
        {
            'question': 'Tom has 10 dollars. He spends 3 dollars. How much money does he have left?',
            'answer': '7',
            'expected': '7'
        }
    ]
    
    # Test each question
    correct = 0
    total = len(test_questions)
    
    for i, test in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{total}: {test['question']}")
        print(f"{'='*70}")
        
        # Convert question to pattern
        question_pattern = framework.text_to_pattern(test['question'])
        
        # Get brain response
        pattern_result = brain.enhanced_pattern_recognition(question_pattern)
        
        # Get reasoning result
        reasoning_result = None
        if hasattr(brain, 'reasoning'):
            context = {
                'sensory_input': question_pattern,
                'pattern_result': pattern_result,
                'question_text': test['question']
            }
            reasoning_result = brain.reasoning(context)
        
        # Extract answer - need to register adapter first for benchmark_name detection
        item = {'question': test['question'], 'answer': test['answer']}
        adapter = framework.adapters.get('GSM8K', GSM8KAdapter())
        
        prediction = framework._extract_answer_from_brain(
            test['question'],
            question_pattern,
            pattern_result,
            reasoning_result,
            item,
            adapter
        )
        
        print(f"\nExpected Answer: {test['expected']}")
        print(f"Predicted Answer: {prediction}")
        print(f"Match: {'✅' if prediction == test['expected'] else '❌'}")
        
        if prediction == test['expected']:
            correct += 1
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"{'='*70}")
    
    # Now test with actual GSM8K benchmark (limited questions)
    print("\n\nTesting with GSM8K benchmark dataset (5 questions)...")
    print("=" * 70)
    
    summary = framework.run_benchmark(
        benchmark_name='GSM8K',
        max_questions=5,
        verbose=True
    )
    
    print(f"\n{'='*70}")
    print(f"GSM8K Benchmark Results:")
    print(f"  Accuracy: {summary.accuracy:.2%} ({summary.correct_answers}/{summary.total_questions})")
    print(f"  Average Confidence: {summary.average_confidence:.3f}")
    print(f"{'='*70}")
    
    return summary.accuracy > 0.0  # Success if we get any correct answers

if __name__ == '__main__':
    success = test_gsm8k_math_reasoning()
    sys.exit(0 if success else 1)

