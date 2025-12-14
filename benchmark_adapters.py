#!/usr/bin/env python3
"""
Benchmark Adapters - Adapters for different benchmark formats
Converts various benchmark datasets to unified format for testing
"""

import numpy as np
from typing import Dict, List, Any, Optional
from benchmark_framework import BenchmarkAdapter

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️  Warning: datasets library not available. Install with: pip install datasets")


class MMLUAdapter(BenchmarkAdapter):
    """Adapter for MMLU (Massive Multitask Language Understanding) benchmark"""
    
    def __init__(self, subset: Optional[str] = None):
        """
        Initialize MMLU adapter
        
        Args:
            subset: Specific subset to load (e.g., 'high_school_biology')
                     If None, loads a sample from multiple subjects
        """
        self.subset = subset
        self.dataset = None
    
    def load_dataset(self) -> List[Dict]:
        """Load MMLU dataset"""
        if not DATASETS_AVAILABLE:
            # Return mock data for testing
            return self._mock_data()
        
        try:
            if self.subset:
                # Load specific subset
                dataset = load_dataset("cais/mmlu", self.subset, split="test")
            else:
                # Load a sample from multiple subjects for testing
                # MMLU has 57 subjects, we'll sample a few
                subjects = [
                    "high_school_biology",
                    "high_school_chemistry", 
                    "high_school_mathematics",
                    "high_school_physics",
                    "abstract_algebra",
                    "college_biology",
                    "college_mathematics",
                    "college_physics"
                ]
                
                all_items = []
                for subject in subjects:
                    try:
                        ds = load_dataset("cais/mmlu", subject, split="test")
                        # Sample 10 items per subject for testing
                        sample_size = min(10, len(ds))
                        indices = np.random.choice(len(ds), sample_size, replace=False)
                        for idx in indices:
                            item = ds[int(idx)]
                            item['category'] = subject
                            all_items.append(item)
                    except Exception as e:
                        print(f"⚠️  Could not load {subject}: {e}")
                        continue
                
                return all_items[:50]  # Limit to 50 items for initial testing
            
            # Convert to list format
            items = []
            for item in dataset:
                item_dict = {
                    'question': item.get('question', ''),
                    'choices': item.get('choices', []),
                    'answer': item.get('answer', 0),
                    'category': self.subset or 'unknown'
                }
                items.append(item_dict)
            
            return items[:50]  # Limit for testing
            
        except Exception as e:
            print(f"⚠️  Error loading MMLU dataset: {e}")
            print("   Using mock data instead")
            return self._mock_data()
    
    def _mock_data(self) -> List[Dict]:
        """Generate mock MMLU data for testing"""
        return [
            {
                'question': 'What is the primary function of mitochondria?',
                'choices': ['A) Protein synthesis', 'B) Energy production', 'C) DNA replication', 'D) Waste removal'],
                'answer': 1,  # B
                'category': 'biology'
            },
            {
                'question': 'What is 2 + 2?',
                'choices': ['A) 3', 'B) 4', 'C) 5', 'D) 6'],
                'answer': 1,  # B
                'category': 'mathematics'
            },
            {
                'question': 'What is the chemical symbol for water?',
                'choices': ['A) H2O', 'B) CO2', 'C) NaCl', 'D) O2'],
                'answer': 0,  # A
                'category': 'chemistry'
            }
        ]
    
    def format_question(self, item: Dict) -> str:
        """Format MMLU item as question"""
        question = item.get('question', '')
        choices = item.get('choices', [])
        
        formatted = f"Question: {question}\n\n"
        for i, choice in enumerate(choices):
            formatted += f"{chr(65 + i)}) {choice}\n"
        
        return formatted
    
    def extract_answer(self, item: Dict) -> Any:
        """Extract ground truth answer"""
        answer_idx = item.get('answer', 0)
        choices = item.get('choices', [])
        
        if answer_idx < len(choices):
            return choices[answer_idx]
        return chr(65 + answer_idx)  # Return letter (A, B, C, D)
    
    def evaluate_answer(self, prediction: Any, ground_truth: Any) -> bool:
        """Evaluate if prediction matches ground truth"""
        # Normalize both to strings
        pred_str = str(prediction).strip().upper()
        truth_str = str(ground_truth).strip().upper()
        
        # Check exact match
        if pred_str == truth_str:
            return True
        
        # Check if prediction contains ground truth
        if truth_str in pred_str:
            return True
        
        # Check if ground truth contains prediction
        if pred_str in truth_str:
            return True
        
        # Check for letter answers (A, B, C, D)
        if len(pred_str) == 1 and len(truth_str) == 1:
            return pred_str == truth_str
        
        return False
    
    def format_response(self, prediction: Any) -> str:
        """Format brain's response for evaluation"""
        return str(prediction).strip()


class HellaSwagAdapter(BenchmarkAdapter):
    """Adapter for HellaSwag commonsense reasoning benchmark"""
    
    def load_dataset(self) -> List[Dict]:
        """Load HellaSwag dataset"""
        if not DATASETS_AVAILABLE:
            return self._mock_data()
        
        try:
            dataset = load_dataset("Rowan/hellaswag", split="validation")
            # Sample 50 items for testing
            sample_size = min(50, len(dataset))
            indices = np.random.choice(len(dataset), sample_size, replace=False)
            
            items = []
            for idx in indices:
                item = dataset[int(idx)]
                items.append({
                    'context': item.get('ctx', ''),
                    'endings': item.get('endings', []),
                    'label': item.get('label', 0),
                    'activity_label': item.get('activity_label', 'unknown')
                })
            
            return items
            
        except Exception as e:
            print(f"⚠️  Error loading HellaSwag dataset: {e}")
            return self._mock_data()
    
    def _mock_data(self) -> List[Dict]:
        """Generate mock HellaSwag data"""
        return [
            {
                'context': 'A person is walking down the street',
                'endings': [
                    'and then stops at a crosswalk',
                    'and then starts flying',
                    'and then turns into a car',
                    'and then disappears'
                ],
                'label': 0,
                'activity_label': 'walking'
            }
        ]
    
    def format_question(self, item: Dict) -> str:
        """Format HellaSwag item as question"""
        context = item.get('context', '')
        endings = item.get('endings', [])
        
        formatted = f"Context: {context}\n\n"
        formatted += "What happens next? Choose the most likely ending:\n\n"
        for i, ending in enumerate(endings):
            formatted += f"{chr(65 + i)}) {ending}\n"
        
        return formatted
    
    def extract_answer(self, item: Dict) -> Any:
        """Extract ground truth answer"""
        label = item.get('label', 0)
        endings = item.get('endings', [])
        
        # Handle both string and integer labels
        if isinstance(label, str):
            try:
                label = int(label)
            except ValueError:
                # If label is a letter (A, B, C, D), convert to index
                if label.isalpha() and len(label) == 1:
                    label = ord(label.upper()) - ord('A')
                else:
                    label = 0
        
        if isinstance(label, int) and label < len(endings):
            return endings[label]
        return chr(65 + (label if isinstance(label, int) else 0))
    
    def evaluate_answer(self, prediction: Any, ground_truth: Any) -> bool:
        """Evaluate if prediction matches ground truth"""
        pred_str = str(prediction).strip()
        truth_str = str(ground_truth).strip()
        
        # Exact match
        if pred_str == truth_str:
            return True
        
        # Check if prediction contains ground truth or vice versa
        if truth_str.lower() in pred_str.lower() or pred_str.lower() in truth_str.lower():
            return True
        
        return False
    
    def format_response(self, prediction: Any) -> str:
        """Format brain's response"""
        return str(prediction).strip()


class GSM8KAdapter(BenchmarkAdapter):
    """Adapter for GSM8K mathematical reasoning benchmark"""
    
    def load_dataset(self) -> List[Dict]:
        """Load GSM8K dataset"""
        if not DATASETS_AVAILABLE:
            return self._mock_data()
        
        try:
            dataset = load_dataset("gsm8k", "main", split="test")
            # Sample 50 items for testing
            sample_size = min(50, len(dataset))
            indices = np.random.choice(len(dataset), sample_size, replace=False)
            
            items = []
            for idx in indices:
                item = dataset[int(idx)]
                items.append({
                    'question': item.get('question', ''),
                    'answer': item.get('answer', '')
                })
            
            return items
            
        except Exception as e:
            print(f"⚠️  Error loading GSM8K dataset: {e}")
            return self._mock_data()
    
    def _mock_data(self) -> List[Dict]:
        """Generate mock GSM8K data"""
        return [
            {
                'question': 'Janet has 5 apples. She gives away 2 apples. How many apples does she have left?',
                'answer': '3'
            },
            {
                'question': 'A store has 20 books. They sell 8 books. How many books are left?',
                'answer': '12'
            }
        ]
    
    def format_question(self, item: Dict) -> str:
        """Format GSM8K item as question"""
        return item.get('question', '')
    
    def extract_answer(self, item: Dict) -> Any:
        """Extract ground truth answer (numeric)"""
        answer = item.get('answer', '')
        # Extract numeric answer
        import re
        numbers = re.findall(r'\d+', answer)
        if numbers:
            return numbers[-1]  # Return last number (usually the final answer)
        return answer
    
    def evaluate_answer(self, prediction: Any, ground_truth: Any) -> bool:
        """Evaluate if prediction matches ground truth"""
        # Extract numbers from both
        import re
        pred_nums = re.findall(r'\d+', str(prediction))
        truth_nums = re.findall(r'\d+', str(ground_truth))
        
        if pred_nums and truth_nums:
            return pred_nums[-1] == truth_nums[-1]
        
        # Fallback to string comparison
        return str(prediction).strip() == str(ground_truth).strip()
    
    def format_response(self, prediction: Any) -> str:
        """Format brain's response"""
        return str(prediction).strip()


class ARCAdapter(BenchmarkAdapter):
    """Adapter for ARC (AI Reasoning Challenge) benchmark"""
    
    def load_dataset(self) -> List[Dict]:
        """Load ARC dataset"""
        if not DATASETS_AVAILABLE:
            return self._mock_data()
        
        try:
            # ARC is available via HuggingFace
            dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            # Sample 30 items (ARC questions are harder)
            sample_size = min(30, len(dataset))
            indices = np.random.choice(len(dataset), sample_size, replace=False)
            
            items = []
            for idx in indices:
                item = dataset[int(idx)]
                items.append({
                    'question': item.get('question', ''),
                    'choices': item.get('choices', {}).get('text', []),
                    'answerKey': item.get('answerKey', 'A')
                })
            
            return items
            
        except Exception as e:
            print(f"⚠️  Error loading ARC dataset: {e}")
            return self._mock_data()
    
    def _mock_data(self) -> List[Dict]:
        """Generate mock ARC data"""
        return [
            {
                'question': 'What happens when water freezes?',
                'choices': ['A) It expands', 'B) It contracts', 'C) It stays the same', 'D) It evaporates'],
                'answerKey': 'A'
            }
        ]
    
    def format_question(self, item: Dict) -> str:
        """Format ARC item as question"""
        question = item.get('question', '')
        choices = item.get('choices', [])
        
        formatted = f"Question: {question}\n\n"
        if isinstance(choices, list):
            for i, choice in enumerate(choices):
                formatted += f"{chr(65 + i)}) {choice}\n"
        elif isinstance(choices, dict):
            texts = choices.get('text', [])
            for i, text in enumerate(texts):
                formatted += f"{chr(65 + i)}) {text}\n"
        
        return formatted
    
    def extract_answer(self, item: Dict) -> Any:
        """Extract ground truth answer"""
        answer_key = item.get('answerKey', 'A')
        choices = item.get('choices', [])
        
        if isinstance(choices, list):
            idx = ord(answer_key.upper()) - ord('A')
            if 0 <= idx < len(choices):
                return choices[idx]
        elif isinstance(choices, dict):
            texts = choices.get('text', [])
            idx = ord(answer_key.upper()) - ord('A')
            if 0 <= idx < len(texts):
                return texts[idx]
        
        return answer_key
    
    def evaluate_answer(self, prediction: Any, ground_truth: Any) -> bool:
        """Evaluate if prediction matches ground truth"""
        pred_str = str(prediction).strip().upper()
        truth_str = str(ground_truth).strip().upper()
        
        if pred_str == truth_str:
            return True
        
        if truth_str in pred_str or pred_str in truth_str:
            return True
        
        return False
    
    def format_response(self, prediction: Any) -> str:
        """Format brain's response"""
        return str(prediction).strip()


class HumanEvalAdapter(BenchmarkAdapter):
    """Adapter for HumanEval code generation benchmark"""
    
    def load_dataset(self) -> List[Dict]:
        """Load HumanEval dataset"""
        if not DATASETS_AVAILABLE:
            print("⚠️  datasets library not available. Using mock data.")
            print("   💡 Install datasets library: pip install datasets")
            return self._mock_data()
        
        # Try multiple dataset paths
        dataset_paths = [
            "openai/humaneval",
            "bigcode/humaneval-python",
            "THUDM/humaneval-x"  # Alternative path
        ]
        
        dataset = None
        last_error = None
        
        for path in dataset_paths:
            try:
                print(f"   Attempting to load: {path}")
                dataset = load_dataset(path, split="test")
                print(f"   ✅ Successfully loaded: {path}")
                break
            except Exception as e:
                last_error = e
                print(f"   ❌ Failed to load {path}: {e}")
                continue
        
        if dataset is None:
            print(f"⚠️  Error loading HumanEval dataset from all paths")
            print(f"   Attempted paths: {', '.join(dataset_paths)}")
            print(f"   DATASETS_AVAILABLE: {DATASETS_AVAILABLE}")
            print(f"   Last error: {last_error}")
            print(f"   💡 Install datasets library: pip install datasets")
            print(f"   💡 Using mock data instead")
            return self._mock_data()
        
        # Sample 20 items (code generation is slower)
        sample_size = min(20, len(dataset))
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        
        items = []
        for idx in indices:
            item = dataset[int(idx)]
            items.append({
                'task_id': item.get('task_id', ''),
                'prompt': item.get('prompt', ''),
                'canonical_solution': item.get('canonical_solution', ''),
                'test': item.get('test', '')
            })
        
        return items
    
    def _mock_data(self) -> List[Dict]:
        """Generate mock HumanEval data"""
        return [
            {
                'task_id': 'test_1',
                'prompt': 'def add(a, b):\n    """Add two numbers"""\n    return',
                'canonical_solution': '    return a + b',
                'test': 'assert add(2, 3) == 5'
            },
            {
                'task_id': 'test_2',
                'prompt': 'def multiply(x, y):\n    """Multiply two numbers"""\n    return',
                'canonical_solution': '    return x * y',
                'test': 'assert multiply(3, 4) == 12'
            },
            {
                'task_id': 'test_3',
                'prompt': 'def subtract(a, b):\n    """Subtract b from a"""\n    return',
                'canonical_solution': '    return a - b',
                'test': 'assert subtract(10, 3) == 7'
            },
            {
                'task_id': 'test_4',
                'prompt': 'def divide(a, b):\n    """Divide a by b"""\n    return',
                'canonical_solution': '    return a / b',
                'test': 'assert divide(12, 3) == 4'
            },
            {
                'task_id': 'test_5',
                'prompt': 'def power(base, exp):\n    """Raise base to exp"""\n    return',
                'canonical_solution': '    return base ** exp',
                'test': 'assert power(2, 3) == 8'
            },
            {
                'task_id': 'test_6',
                'prompt': 'def max_value(lst):\n    """Return maximum value in list"""\n    return',
                'canonical_solution': '    return max(lst)',
                'test': 'assert max_value([1, 5, 3, 9, 2]) == 9'
            },
            {
                'task_id': 'test_7',
                'prompt': 'def min_value(lst):\n    """Return minimum value in list"""\n    return',
                'canonical_solution': '    return min(lst)',
                'test': 'assert min_value([5, 2, 8, 1, 9]) == 1'
            },
            {
                'task_id': 'test_8',
                'prompt': 'def sum_list(lst):\n    """Sum all values in list"""\n    return',
                'canonical_solution': '    return sum(lst)',
                'test': 'assert sum_list([1, 2, 3, 4]) == 10'
            },
            {
                'task_id': 'test_9',
                'prompt': 'def is_even(n):\n    """Check if number is even"""\n    return',
                'canonical_solution': '    return n % 2 == 0',
                'test': 'assert is_even(4) == True'
            },
            {
                'task_id': 'test_10',
                'prompt': 'def factorial(n):\n    """Calculate factorial"""\n    return',
                'canonical_solution': '    return 1 if n <= 1 else n * factorial(n - 1)',
                'test': 'assert factorial(5) == 120'
            }
        ]
    
    def format_question(self, item: Dict) -> str:
        """Format HumanEval item as question"""
        prompt = item.get('prompt', '')
        return f"Complete this Python function:\n\n{prompt}"
    
    def extract_answer(self, item: Dict) -> Any:
        """Extract ground truth solution"""
        return item.get('canonical_solution', '')
    
    def evaluate_answer(self, prediction: Any, ground_truth: Any) -> bool:
        """Evaluate code generation (simplified - would need actual execution)"""
        # This is a simplified check - real evaluation would execute code
        pred_str = str(prediction).strip()
        truth_str = str(ground_truth).strip()
        
        # Check if key components match
        # Extract function body (remove def line if present)
        if 'return' in pred_str and 'return' in truth_str:
            # Check if return statements are similar
            pred_return = pred_str.split('return')[-1].strip()
            truth_return = truth_str.split('return')[-1].strip()
            if pred_return == truth_return:
                return True
        
        # Check for exact match
        if pred_str == truth_str:
            return True
        
        return False
    
    def format_response(self, prediction: Any) -> str:
        """Format brain's response"""
        return str(prediction).strip()

