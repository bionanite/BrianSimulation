#!/usr/bin/env python3
"""
Advanced Reasoning - Multi-step reasoning capabilities
Implements chain-of-thought reasoning, multi-hop inference, and causal reasoning chains
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from language_processor import LanguageProcessor


class AdvancedReasoning:
    """Advanced multi-step reasoning system"""
    
    def __init__(self, brain_system=None):
        """
        Initialize advanced reasoning system
        
        Args:
            brain_system: FinalEnhancedBrain instance (optional)
        """
        self.brain_system = brain_system
        self.lang_processor = LanguageProcessor()
        
        # Reasoning history
        self.reasoning_chains: List[Dict] = []
        
        # Causal knowledge base
        self.causal_knowledge: Dict[str, List[str]] = {}
    
    def chain_of_thought(self, 
                        question: str,
                        max_steps: int = 5,
                        verbose: bool = False) -> Dict[str, Any]:
        """
        Chain-of-thought reasoning: break down problem into steps
        
        Args:
            question: Question to reason about
            max_steps: Maximum reasoning steps
            verbose: Print reasoning steps
            
        Returns:
            Dictionary with reasoning chain and conclusion
        """
        reasoning_chain = {
            'question': question,
            'steps': [],
            'conclusion': '',
            'confidence': 0.0
        }
        
        # Step 1: Understand the question
        question_pattern = self.lang_processor.text_to_pattern(question)
        keywords = self.lang_processor.extract_keywords(question, top_k=5)
        
        step1 = {
            'step': 1,
            'action': 'understand_question',
            'content': f"Question: {question}",
            'keywords': [word for word, _ in keywords],
            'pattern': question_pattern.tolist()
        }
        reasoning_chain['steps'].append(step1)
        
        if verbose:
            print(f"Step 1: Understanding question")
            print(f"  Keywords: {', '.join(step1['keywords'])}")
        
        # Step 2: Identify reasoning type
        reasoning_type = self._identify_reasoning_type(question)
        step2 = {
            'step': 2,
            'action': 'identify_reasoning_type',
            'content': f"Reasoning type: {reasoning_type}",
            'reasoning_type': reasoning_type
        }
        reasoning_chain['steps'].append(step2)
        
        if verbose:
            print(f"Step 2: Reasoning type identified: {reasoning_type}")
        
        # Step 3-N: Apply reasoning steps
        current_state = question_pattern
        for i in range(3, max_steps + 1):
            if reasoning_type == 'mathematical':
                step_result = self._mathematical_reasoning_step(current_state, question, i)
            elif reasoning_type in ['biology', 'chemistry', 'physics', 'humanities', 'social_science', 'stem']:
                # Use domain-specific reasoning
                step_result = self._domain_specific_reasoning_step(current_state, question, i, reasoning_type)
            elif reasoning_type == 'logical':
                step_result = self._logical_reasoning_step(current_state, question, i)
            elif reasoning_type == 'causal':
                step_result = self._causal_reasoning_step(current_state, question, i)
            elif reasoning_type == 'comparison':
                step_result = self._comparison_reasoning_step(current_state, question, i)
            else:
                step_result = self._general_reasoning_step(current_state, question, i)
            
            reasoning_chain['steps'].append(step_result)
            current_state = np.array(step_result.get('pattern', current_state))
            
            if verbose:
                print(f"Step {i}: {step_result['action']}")
                print(f"  {step_result['content']}")
            
            # Check if we can conclude
            if step_result.get('can_conclude', False):
                break
        
        # Final step: Generate conclusion
        conclusion = self._generate_conclusion(reasoning_chain['steps'], question)
        reasoning_chain['conclusion'] = conclusion
        
        # Calculate confidence
        confidence = self._calculate_confidence(reasoning_chain)
        reasoning_chain['confidence'] = confidence
        
        # Store reasoning chain
        self.reasoning_chains.append(reasoning_chain)
        
        return reasoning_chain
    
    def _identify_reasoning_type(self, question: str) -> str:
        """Identify the type of reasoning needed"""
        question_lower = question.lower()
        
        # STEM domain detection (for MMLU)
        stem_keywords = [
            'biology', 'chemistry', 'physics', 'mathematics', 'science',
            'cell', 'molecule', 'atom', 'electron', 'protein', 'dna', 'rna',
            'equation', 'formula', 'reaction', 'compound', 'element',
            'force', 'energy', 'mass', 'velocity', 'acceleration'
        ]
        if any(keyword in question_lower for keyword in stem_keywords):
            # Further classify STEM
            if any(kw in question_lower for kw in ['biology', 'cell', 'protein', 'dna', 'rna', 'organism']):
                return 'biology'
            elif any(kw in question_lower for kw in ['chemistry', 'molecule', 'atom', 'reaction', 'compound', 'element']):
                return 'chemistry'
            elif any(kw in question_lower for kw in ['physics', 'force', 'energy', 'mass', 'velocity', 'acceleration']):
                return 'physics'
            elif any(kw in question_lower for kw in ['mathematics', 'equation', 'formula', 'calculate', 'compute']):
                return 'mathematical'
            else:
                return 'stem'
        
        # Humanities domain detection
        humanities_keywords = [
            'history', 'literature', 'philosophy', 'art', 'music',
            'author', 'poet', 'novel', 'poem', 'painting', 'sculpture',
            'ancient', 'medieval', 'renaissance', 'classical'
        ]
        if any(keyword in question_lower for keyword in humanities_keywords):
            return 'humanities'
        
        # Social sciences domain detection
        social_science_keywords = [
            'psychology', 'sociology', 'economics', 'politics', 'government',
            'behavior', 'society', 'culture', 'market', 'demand', 'supply',
            'cognitive', 'social', 'political', 'economic'
        ]
        if any(keyword in question_lower for keyword in social_science_keywords):
            return 'social_science'
        
        # Mathematical reasoning
        math_keywords = ['calculate', 'compute', 'solve', 'how many', 'what is', '+', '-', '*', '/', 'equals']
        if any(keyword in question_lower for keyword in math_keywords):
            return 'mathematical'
        
        # Causal reasoning
        causal_keywords = ['why', 'because', 'causes', 'leads to', 'results in', 'effect']
        if any(keyword in question_lower for keyword in causal_keywords):
            return 'causal'
        
        # Comparison reasoning
        comparison_keywords = ['compare', 'difference', 'similar', 'better', 'worse', 'versus']
        if any(keyword in question_lower for keyword in comparison_keywords):
            return 'comparison'
        
        # Logical reasoning
        logical_keywords = ['if', 'then', 'all', 'some', 'none', 'logical', 'infer']
        if any(keyword in question_lower for keyword in logical_keywords):
            return 'logical'
        
        return 'general'
    
    def _mathematical_reasoning_step(self, state: np.ndarray, question: str, step_num: int) -> Dict:
        """Perform a mathematical reasoning step"""
        # Extract numbers from question
        import re
        numbers = re.findall(r'\d+', question)
        
        # Extract operations
        operations = []
        if '+' in question or 'plus' in question:
            operations.append('add')
        if '-' in question or 'minus' in question:
            operations.append('subtract')
        if '*' in question or 'times' in question or 'multiply' in question:
            operations.append('multiply')
        if '/' in question or 'divide' in question:
            operations.append('divide')
        
        # Perform calculation if possible
        result = None
        if len(numbers) >= 2 and operations:
            try:
                num1 = float(numbers[0])
                num2 = float(numbers[1])
                op = operations[0]
                
                if op == 'add':
                    result = num1 + num2
                elif op == 'subtract':
                    result = num1 - num2
                elif op == 'multiply':
                    result = num1 * num2
                elif op == 'divide' and num2 != 0:
                    result = num1 / num2
            except:
                pass
        
        content = f"Mathematical reasoning step {step_num}"
        if result is not None:
            content += f": {numbers[0]} {operations[0]} {numbers[1]} = {result}"
        
        # Update state pattern
        new_pattern = state.copy()
        if result is not None:
            # Encode result into pattern
            result_pattern = self.lang_processor.text_to_pattern(str(result))
            new_pattern = (new_pattern + result_pattern) / 2.0
        
        return {
            'step': step_num,
            'action': 'mathematical_reasoning',
            'content': content,
            'pattern': new_pattern.tolist(),
            'result': result,
            'can_conclude': result is not None
        }
    
    def _domain_specific_reasoning_step(self, state: np.ndarray, question: str, step_num: int, domain: str) -> Dict:
        """Perform domain-specific reasoning step"""
        question_lower = question.lower()
        content = f"{domain.capitalize()} reasoning step {step_num}"
        
        # Domain-specific keyword matching
        domain_keywords = {
            'biology': ['cell', 'organism', 'protein', 'dna', 'gene', 'evolution', 'species'],
            'chemistry': ['molecule', 'atom', 'reaction', 'compound', 'element', 'bond'],
            'physics': ['force', 'energy', 'mass', 'velocity', 'acceleration', 'momentum'],
            'humanities': ['author', 'literature', 'philosophy', 'art', 'culture', 'historical'],
            'social_science': ['behavior', 'society', 'cognitive', 'economic', 'political']
        }
        
        keywords = domain_keywords.get(domain, [])
        found_keywords = [kw for kw in keywords if kw in question_lower]
        
        if found_keywords:
            content += f": Analyzing {', '.join(found_keywords[:3])}"
        
        # Update state with domain-specific features
        new_pattern = state.copy()
        domain_features = np.array([0.7, 0.8, 0.6], dtype=np.float32)
        if len(new_pattern) >= len(domain_features):
            new_pattern[:len(domain_features)] = (new_pattern[:len(domain_features)] + domain_features) / 2.0
        
        return {
            'step': step_num,
            'action': f'{domain}_reasoning',
            'content': content,
            'pattern': new_pattern.tolist(),
            'domain': domain,
            'can_conclude': len(found_keywords) > 0
        }
    
    def _logical_reasoning_step(self, state: np.ndarray, question: str, step_num: int) -> Dict:
        """Perform a logical reasoning step"""
        # Extract logical structure
        keywords = self.lang_processor.extract_keywords(question, top_k=5)
        
        content = f"Logical reasoning step {step_num}: Analyzing logical structure"
        
        # Simple logical inference
        question_lower = question.lower()
        if 'if' in question_lower and 'then' in question_lower:
            content += " (conditional reasoning)"
        
        # Update state
        new_pattern = state.copy()
        # Add logical reasoning features
        logical_features = np.array([0.8, 0.6, 0.7], dtype=np.float32)  # Logical reasoning indicators
        if len(new_pattern) >= len(logical_features):
            new_pattern[:len(logical_features)] = (
                new_pattern[:len(logical_features)] * 0.7 + logical_features * 0.3
            )
        
        return {
            'step': step_num,
            'action': 'logical_reasoning',
            'content': content,
            'pattern': new_pattern.tolist(),
            'can_conclude': step_num >= 3  # Can conclude after a few steps
        }
    
    def _causal_reasoning_step(self, state: np.ndarray, question: str, step_num: int) -> Dict:
        """Perform a causal reasoning step"""
        keywords = self.lang_processor.extract_keywords(question, top_k=3)
        main_concept = keywords[0][0] if keywords else ''
        
        # Look up causal relations
        causes = self.causal_knowledge.get(main_concept, [])
        
        content = f"Causal reasoning step {step_num}: Analyzing causes and effects"
        if causes:
            content += f"\n  Known causes of '{main_concept}': {', '.join(causes[:3])}"
        
        # Update state with causal information
        new_pattern = state.copy()
        causal_features = np.array([0.7, 0.8, 0.6], dtype=np.float32)
        if len(new_pattern) >= len(causal_features):
            new_pattern[:len(causal_features)] = (
                new_pattern[:len(causal_features)] * 0.7 + causal_features * 0.3
            )
        
        return {
            'step': step_num,
            'action': 'causal_reasoning',
            'content': content,
            'pattern': new_pattern.tolist(),
            'causes': causes,
            'can_conclude': len(causes) > 0 or step_num >= 4
        }
    
    def _comparison_reasoning_step(self, state: np.ndarray, question: str, step_num: int) -> Dict:
        """Perform a comparison reasoning step"""
        # Extract entities to compare
        keywords = self.lang_processor.extract_keywords(question, top_k=5)
        
        content = f"Comparison reasoning step {step_num}: Comparing entities"
        if len(keywords) >= 2:
            content += f"\n  Comparing: {keywords[0][0]} vs {keywords[1][0]}"
        
        # Update state
        new_pattern = state.copy()
        comparison_features = np.array([0.6, 0.7, 0.8], dtype=np.float32)
        if len(new_pattern) >= len(comparison_features):
            new_pattern[:len(comparison_features)] = (
                new_pattern[:len(comparison_features)] * 0.7 + comparison_features * 0.3
            )
        
        return {
            'step': step_num,
            'action': 'comparison_reasoning',
            'content': content,
            'pattern': new_pattern.tolist(),
            'can_conclude': step_num >= 3
        }
    
    def _general_reasoning_step(self, state: np.ndarray, question: str, step_num: int) -> Dict:
        """Perform a general reasoning step"""
        content = f"General reasoning step {step_num}: Analyzing question"
        
        # Update state
        new_pattern = state.copy()
        # Add general reasoning features
        general_features = np.array([0.5, 0.6, 0.5], dtype=np.float32)
        if len(new_pattern) >= len(general_features):
            new_pattern[:len(general_features)] = (
                new_pattern[:len(general_features)] * 0.8 + general_features * 0.2
            )
        
        return {
            'step': step_num,
            'action': 'general_reasoning',
            'content': content,
            'pattern': new_pattern.tolist(),
            'can_conclude': step_num >= 4
        }
    
    def _generate_conclusion(self, steps: List[Dict], question: str) -> str:
        """Generate final conclusion from reasoning steps"""
        # Extract key information from steps
        conclusions = []
        
        for step in steps:
            if step.get('result') is not None:
                conclusions.append(str(step['result']))
            elif 'causes' in step and step['causes']:
                conclusions.append(f"Causes: {', '.join(step['causes'][:2])}")
        
        if conclusions:
            return ". ".join(conclusions)
        
        # Fallback: use last step's content
        if steps:
            last_step = steps[-1]
            return last_step.get('content', 'No conclusion reached')
        
        return "Unable to reach conclusion"
    
    def _calculate_confidence(self, reasoning_chain: Dict) -> float:
        """Calculate confidence in reasoning chain"""
        steps = reasoning_chain.get('steps', [])
        
        if not steps:
            return 0.0
        
        # Base confidence on number of steps and their quality
        num_steps = len(steps)
        step_quality = 0.0
        
        for step in steps:
            # Higher quality if step has results or specific content
            if step.get('result') is not None:
                step_quality += 0.3
            elif step.get('causes'):
                step_quality += 0.2
            elif len(step.get('content', '')) > 50:
                step_quality += 0.1
        
        # Normalize
        step_quality = min(step_quality / num_steps, 1.0) if num_steps > 0 else 0.0
        
        # Confidence increases with more steps (up to a point)
        step_bonus = min(num_steps / 5.0, 0.3)
        
        confidence = 0.4 + step_quality * 0.3 + step_bonus
        return min(confidence, 1.0)
    
    def multi_hop_inference(self, 
                           premise: str,
                           max_hops: int = 3) -> Dict[str, Any]:
        """
        Multi-hop inference: chain multiple inferences
        
        Args:
            premise: Starting premise
            max_hops: Maximum inference hops
            
        Returns:
            Dictionary with inference chain and conclusion
        """
        inference_chain = {
            'premise': premise,
            'hops': [],
            'conclusion': ''
        }
        
        current_state = self.lang_processor.text_to_pattern(premise)
        
        for hop in range(1, max_hops + 1):
            # Perform inference step
            inference_result = self._inference_step(current_state, hop)
            inference_chain['hops'].append(inference_result)
            
            # Update state
            current_state = np.array(inference_result.get('pattern', current_state))
            
            # Check if we can stop
            if inference_result.get('can_stop', False):
                break
        
        # Generate conclusion
        conclusion = self._generate_inference_conclusion(inference_chain['hops'])
        inference_chain['conclusion'] = conclusion
        
        return inference_chain
    
    def _inference_step(self, state: np.ndarray, hop_num: int) -> Dict:
        """Perform a single inference step"""
        # Simple inference: extract patterns and make connections
        content = f"Inference hop {hop_num}: Making logical connections"
        
        # Update state with inference features
        new_pattern = state.copy()
        inference_features = np.array([0.7, 0.6, 0.8], dtype=np.float32)
        if len(new_pattern) >= len(inference_features):
            new_pattern[:len(inference_features)] = (
                new_pattern[:len(inference_features)] * 0.7 + inference_features * 0.3
            )
        
        return {
            'hop': hop_num,
            'content': content,
            'pattern': new_pattern.tolist(),
            'can_stop': hop_num >= 3
        }
    
    def _generate_inference_conclusion(self, hops: List[Dict]) -> str:
        """Generate conclusion from inference hops"""
        if not hops:
            return "No inference made"
        
        conclusions = [hop.get('content', '') for hop in hops]
        return " → ".join(conclusions)
    
    def add_causal_knowledge(self, cause: str, effect: str):
        """Add causal knowledge"""
        if cause not in self.causal_knowledge:
            self.causal_knowledge[cause] = []
        if effect not in self.causal_knowledge[cause]:
            self.causal_knowledge[cause].append(effect)
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get reasoning statistics"""
        return {
            'total_chains': len(self.reasoning_chains),
            'causal_relations': sum(len(effects) for effects in self.causal_knowledge.values()),
            'avg_chain_length': np.mean([len(c['steps']) for c in self.reasoning_chains]) if self.reasoning_chains else 0.0
        }

