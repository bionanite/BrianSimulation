#!/usr/bin/env python3
"""
Abstract Reasoning System
Implements symbolic reasoning, logical inference, and abstract concept manipulation
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
import re

try:
    from advanced_reasoning import AdvancedReasoningSystem
except ImportError:
    AdvancedReasoningSystem = None

try:
    from Phase8_AdvancedReasoning.probabilistic_causal_reasoning import ProbabilisticCausalReasoningSystem
except ImportError:
    ProbabilisticCausalReasoningSystem = None


@dataclass
class Symbol:
    """Represents a symbolic entity"""
    name: str
    type: str  # 'variable', 'constant', 'function', 'predicate'
    value: Optional[Any] = None
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class LogicalExpression:
    """Represents a logical expression"""
    expression_id: int
    expression_type: str  # 'proposition', 'implication', 'conjunction', 'disjunction', 'negation', 'quantifier'
    components: List[Any]  # Sub-expressions or symbols
    truth_value: Optional[bool] = None
    confidence: float = 0.5


class SymbolicReasoningEngine:
    """
    Symbolic Reasoning Engine
    Performs logical inference and symbolic manipulation
    """
    
    def __init__(self):
        self.symbols: Dict[str, Symbol] = {}
        self.expressions: List[LogicalExpression] = []
        self.inference_rules: List[Dict] = []
        self.next_expression_id = 0
        
        # Initialize basic inference rules
        self._init_inference_rules()
    
    def _init_inference_rules(self):
        """Initialize basic logical inference rules"""
        self.inference_rules = [
            {
                'name': 'modus_ponens',
                'pattern': ('implication', 'proposition'),
                'result': 'proposition'
            },
            {
                'name': 'modus_tollens',
                'pattern': ('implication', 'negation'),
                'result': 'negation'
            },
            {
                'name': 'conjunction_introduction',
                'pattern': ('proposition', 'proposition'),
                'result': 'conjunction'
            },
            {
                'name': 'disjunction_introduction',
                'pattern': ('proposition',),
                'result': 'disjunction'
            }
        ]
    
    def create_symbol(self, name: str, symbol_type: str = 'variable', value: Any = None) -> Symbol:
        """Create a new symbolic entity"""
        symbol = Symbol(name=name, type=symbol_type, value=value)
        self.symbols[name] = symbol
        return symbol
    
    def create_expression(self, expression_type: str, components: List[Any], 
                         truth_value: Optional[bool] = None) -> LogicalExpression:
        """Create a new logical expression"""
        expr = LogicalExpression(
            expression_id=self.next_expression_id,
            expression_type=expression_type,
            components=components,
            truth_value=truth_value
        )
        self.next_expression_id += 1
        self.expressions.append(expr)
        return expr
    
    def infer(self, premises: List[LogicalExpression], goal: Optional[LogicalExpression] = None) -> Dict:
        """
        Perform logical inference from premises
        
        Args:
            premises: List of logical expressions (premises)
            goal: Optional goal expression to prove
            
        Returns:
            Dict with inferred conclusions and confidence
        """
        conclusions = []
        confidence = 0.5
        
        # Apply inference rules
        for rule in self.inference_rules:
            # Try to match rule pattern to premises
            if self._match_rule(rule, premises):
                conclusion = self._apply_rule(rule, premises)
                if conclusion:
                    conclusions.append(conclusion)
                    confidence = max(confidence, 0.6)  # Inference increases confidence
        
        # If goal provided, check if we can prove it
        goal_proven = False
        if goal:
            for conclusion in conclusions:
                if self._expressions_match(conclusion, goal):
                    goal_proven = True
                    confidence = 0.8
                    break
        
        return {
            'conclusions': conclusions,
            'goal_proven': goal_proven,
            'confidence': confidence,
            'num_inferences': len(conclusions)
        }
    
    def _match_rule(self, rule: Dict, premises: List[LogicalExpression]) -> bool:
        """Check if inference rule matches premises"""
        pattern = rule.get('pattern', [])
        if len(pattern) != len(premises):
            return False
        
        # Simple matching: check expression types
        for i, expr_type in enumerate(pattern):
            if i < len(premises) and premises[i].expression_type != expr_type:
                return False
        return True
    
    def _apply_rule(self, rule: Dict, premises: List[LogicalExpression]) -> Optional[LogicalExpression]:
        """Apply inference rule to premises"""
        result_type = rule.get('result', 'proposition')
        
        # Create conclusion based on rule
        if result_type == 'proposition':
            # Modus ponens: If P -> Q and P, then Q
            if len(premises) >= 2:
                return self.create_expression('proposition', premises[1].components, True)
        elif result_type == 'conjunction':
            # Conjunction: P and Q
            return self.create_expression('conjunction', premises, True)
        elif result_type == 'disjunction':
            # Disjunction: P or Q
            return self.create_expression('disjunction', premises, True)
        
        return None
    
    def _expressions_match(self, expr1: LogicalExpression, expr2: LogicalExpression) -> bool:
        """Check if two expressions match"""
        if expr1.expression_type != expr2.expression_type:
            return False
        if len(expr1.components) != len(expr2.components):
            return False
        return True
    
    def evaluate_expression(self, expression: LogicalExpression) -> bool:
        """Evaluate truth value of logical expression"""
        if expression.truth_value is not None:
            return expression.truth_value
        
        # Evaluate based on expression type
        if expression.expression_type == 'conjunction':
            # All components must be true
            return all(self.evaluate_expression(c) if isinstance(c, LogicalExpression) else True 
                      for c in expression.components)
        elif expression.expression_type == 'disjunction':
            # At least one component must be true
            return any(self.evaluate_expression(c) if isinstance(c, LogicalExpression) else True 
                      for c in expression.components)
        elif expression.expression_type == 'negation':
            # Negate the component
            if expression.components:
                return not self.evaluate_expression(expression.components[0]) if isinstance(expression.components[0], LogicalExpression) else False
        elif expression.expression_type == 'implication':
            # P -> Q: false only if P true and Q false
            if len(expression.components) >= 2:
                p = self.evaluate_expression(expression.components[0]) if isinstance(expression.components[0], LogicalExpression) else True
                q = self.evaluate_expression(expression.components[1]) if isinstance(expression[1], LogicalExpression) else True
                return not (p and not q)
        
        return True  # Default to true


class AbstractConceptManipulator:
    """
    Abstract Concept Manipulator
    Manipulates abstract concepts and relationships
    """
    
    def __init__(self):
        self.concepts: Dict[str, Dict] = {}
        self.relationships: List[Dict] = []
        self.concept_hierarchy: Dict[str, List[str]] = {}
    
    def create_concept(self, name: str, properties: Dict[str, Any] = None) -> Dict:
        """Create a new abstract concept"""
        concept = {
            'name': name,
            'properties': properties or {},
            'instances': [],
            'relationships': []
        }
        self.concepts[name] = concept
        return concept
    
    def add_relationship(self, concept1: str, relationship_type: str, concept2: str, strength: float = 1.0):
        """Add relationship between concepts"""
        relationship = {
            'concept1': concept1,
            'relationship_type': relationship_type,  # 'is_a', 'part_of', 'similar_to', 'opposite_of', etc.
            'concept2': concept2,
            'strength': strength
        }
        self.relationships.append(relationship)
        
        # Update concept relationships
        if concept1 in self.concepts:
            self.concepts[concept1]['relationships'].append(relationship)
        if concept2 in self.concepts:
            self.concepts[concept2]['relationships'].append(relationship)
    
    def find_analogies(self, source_concept: str, target_domain: str) -> List[Dict]:
        """Find analogies between concepts"""
        analogies = []
        
        if source_concept not in self.concepts:
            return analogies
        
        source_relationships = self.concepts[source_concept].get('relationships', [])
        
        # Find concepts in target domain with similar relationship patterns
        for concept_name, concept in self.concepts.items():
            if concept_name == source_concept:
                continue
            
            # Check if concept is in target domain
            concept_domain = concept.get('properties', {}).get('domain', '')
            if target_domain.lower() not in concept_domain.lower():
                continue
            
            # Compare relationship patterns
            target_relationships = concept.get('relationships', [])
            similarity = self._compare_relationship_patterns(source_relationships, target_relationships)
            
            if similarity > 0.5:
                analogies.append({
                    'source': source_concept,
                    'target': concept_name,
                    'similarity': similarity,
                    'relationship_matches': self._find_matching_relationships(source_relationships, target_relationships)
                })
        
        # Sort by similarity
        analogies.sort(key=lambda x: x['similarity'], reverse=True)
        return analogies
    
    def _compare_relationship_patterns(self, rels1: List[Dict], rels2: List[Dict]) -> float:
        """Compare relationship patterns between concepts"""
        if not rels1 or not rels2:
            return 0.0
        
        # Extract relationship types
        types1 = set(r.get('relationship_type', '') for r in rels1)
        types2 = set(r.get('relationship_type', '') for r in rels2)
        
        # Calculate similarity
        intersection = len(types1 & types2)
        union = len(types1 | types2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _find_matching_relationships(self, rels1: List[Dict], rels2: List[Dict]) -> List[Dict]:
        """Find matching relationships between two concept sets"""
        matches = []
        types1 = {r.get('relationship_type', ''): r for r in rels1}
        
        for rel2 in rels2:
            rel_type = rel2.get('relationship_type', '')
            if rel_type in types1:
                matches.append({
                    'type': rel_type,
                    'source': types1[rel_type],
                    'target': rel2
                })
        
        return matches
    
    def abstract_generalization(self, instances: List[Dict]) -> Dict:
        """Generalize from specific instances to abstract concept"""
        if not instances:
            return {'concept': None, 'generalization_score': 0.0}
        
        # Extract common properties
        common_properties = {}
        if instances:
            # Get properties from first instance
            first_props = instances[0].get('properties', {})
            for prop_name, prop_value in first_props.items():
                # Check if property is common across instances
                if all(inst.get('properties', {}).get(prop_name) == prop_value for inst in instances):
                    common_properties[prop_name] = prop_value
        
        # Calculate generalization score
        generalization_score = len(common_properties) / max(len(first_props), 1) if first_props else 0.5
        
        abstract_concept = {
            'name': f"Abstract_{len(self.concepts)}",
            'properties': common_properties,
            'instances': instances,
            'generalization_score': generalization_score
        }
        
        return abstract_concept


class AbstractReasoningSystem:
    """
    Abstract Reasoning System
    Integrates symbolic reasoning and abstract concept manipulation
    """
    
    def __init__(self,
                 brain_system=None,
                 advanced_reasoning: Optional[AdvancedReasoningSystem] = None,
                 probabilistic_reasoning: Optional[ProbabilisticCausalReasoningSystem] = None):
        self.brain_system = brain_system
        self.advanced_reasoning = advanced_reasoning
        self.probabilistic_reasoning = probabilistic_reasoning
        
        self.symbolic_engine = SymbolicReasoningEngine()
        self.concept_manipulator = AbstractConceptManipulator()
        
        # Statistics
        self.stats = {
            'inferences_performed': 0,
            'concepts_created': 0,
            'analogies_found': 0,
            'generalizations_made': 0
        }
    
    def reason_abstractly(self, problem: str, context: Optional[Dict] = None) -> Dict:
        """
        Perform abstract reasoning on a problem
        
        Args:
            problem: Problem description
            context: Optional context information
            
        Returns:
            Dict with reasoning results and abstract conclusions
        """
        if context is None:
            context = {}
        
        # Extract concepts from problem
        concepts = self._extract_concepts(problem)
        
        # Create symbolic representations
        symbols = []
        for concept in concepts:
            symbol = self.symbolic_engine.create_symbol(concept, 'variable')
            symbols.append(symbol)
            self.stats['concepts_created'] += 1
        
        # Perform logical inference
        premises = []
        for symbol in symbols[:3]:  # Use first 3 symbols as premises
            expr = self.symbolic_engine.create_expression('proposition', [symbol], True)
            premises.append(expr)
        
        inference_result = self.symbolic_engine.infer(premises)
        
        # Find analogies if applicable
        analogies = []
        if len(concepts) >= 2:
            for i, concept in enumerate(concepts[:2]):
                concept_analogies = self.concept_manipulator.find_analogies(concept, 'general')
                analogies.extend(concept_analogies)
        
        self.stats['inferences_performed'] += inference_result.get('num_inferences', 0)
        self.stats['analogies_found'] += len(analogies)
        
        # Calculate abstract reasoning score
        reasoning_score = self._calculate_reasoning_score(inference_result, analogies, len(concepts))
        
        return {
            'concepts_identified': concepts,
            'symbols_created': len(symbols),
            'inference_result': inference_result,
            'analogies': analogies[:3],  # Top 3 analogies
            'abstract_reasoning_score': reasoning_score,
            'confidence': inference_result.get('confidence', 0.5)
        }
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract abstract concepts from text"""
        concepts = []
        text_lower = text.lower()
        
        # Common abstract concepts
        abstract_keywords = {
            'cause': ['cause', 'causes', 'caused', 'causing', 'reason', 'reasons'],
            'effect': ['effect', 'effects', 'result', 'results', 'consequence', 'consequences'],
            'similarity': ['similar', 'similarity', 'like', 'alike', 'same'],
            'difference': ['different', 'difference', 'distinct', 'unlike'],
            'relationship': ['relationship', 'relation', 'connection', 'link', 'association'],
            'pattern': ['pattern', 'structure', 'organization', 'arrangement'],
            'principle': ['principle', 'rule', 'law', 'theory', 'concept'],
            'abstraction': ['abstract', 'general', 'universal', 'conceptual']
        }
        
        for concept_name, keywords in abstract_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                concepts.append(concept_name)
        
        # Also extract capitalized words (often concepts)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        concepts.extend(capitalized_words[:5])  # Limit to 5
        
        # Remove duplicates
        return list(set(concepts))
    
    def _calculate_reasoning_score(self, inference_result: Dict, analogies: List[Dict], num_concepts: int) -> float:
        """Calculate abstract reasoning score"""
        # Base score from inference
        inference_score = min(1.0, inference_result.get('num_inferences', 0) / 3.0)
        
        # Analogy score
        analogy_score = min(1.0, len(analogies) / 2.0)
        
        # Concept diversity score
        concept_score = min(1.0, num_concepts / 5.0)
        
        # Combined score
        reasoning_score = (
            inference_score * 0.4 +
            analogy_score * 0.3 +
            concept_score * 0.3
        )
        
        # Boost if goal was proven
        if inference_result.get('goal_proven', False):
            reasoning_score = min(1.0, reasoning_score + 0.2)
        
        return reasoning_score
    
    def solve_abstract_problem(self, problem_description: str) -> Dict:
        """
        Solve an abstract reasoning problem
        
        Args:
            problem_description: Description of abstract problem
            
        Returns:
            Dict with solution and reasoning steps
        """
        # Perform abstract reasoning
        reasoning_result = self.reason_abstractly(problem_description)
        
        # Generate solution
        solution = {
            'problem': problem_description,
            'reasoning_steps': reasoning_result.get('inference_result', {}).get('conclusions', []),
            'concepts_used': reasoning_result.get('concepts_identified', []),
            'analogies_applied': reasoning_result.get('analogies', []),
            'solution_confidence': reasoning_result.get('confidence', 0.5),
            'abstract_reasoning_score': reasoning_result.get('abstract_reasoning_score', 0.5)
        }
        
        return solution
    
    def get_statistics(self) -> Dict:
        """Get abstract reasoning statistics"""
        return self.stats.copy()

