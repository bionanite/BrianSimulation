#!/usr/bin/env python3
"""
Code Generation Module
Implements code generation capabilities for HumanEval benchmark
"""

import re
import ast
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import dependencies
try:
    from Phase11_Language.language_generation import LanguageGenerationSystem
    from Phase6_Creativity.creative_problem_solving import CreativeProblemSolving
    from advanced_reasoning import AdvancedReasoningSystem
except ImportError:
    LanguageGenerationSystem = None
    CreativeProblemSolving = None
    AdvancedReasoningSystem = None


class CodeGenerationSystem:
    """
    Code Generation System
    
    Generates Python code from function signatures and docstrings
    Uses language generation and creative problem solving
    """
    
    def __init__(self,
                 language_generator: Optional[LanguageGenerationSystem] = None,
                 problem_solver: Optional[CreativeProblemSolving] = None,
                 reasoning_system: Optional[AdvancedReasoningSystem] = None):
        self.language_generator = language_generator
        self.problem_solver = problem_solver
        self.reasoning_system = reasoning_system
        
        # Code templates for common patterns
        self.code_templates = {
            'return_sum': 'return {args}',
            'return_product': 'return {args}',
            'return_difference': 'return {args}',
            'return_quotient': 'return {args}',
            'return_list': 'return [{args}]',
            'return_dict': 'return {{{args}}}',
            'return_string': 'return "{args}"',
            'return_bool': 'return {args}',
            'return_none': 'return None'
        }
    
    def parse_function_signature(self, prompt: str) -> Dict:
        """
        Parse function signature from prompt
        
        Returns:
            Parsed function info (name, params, docstring, return_hint)
        """
        func_info = {
            'name': None,
            'parameters': [],
            'docstring': '',
            'return_hint': None
        }
        
        # Extract function definition
        func_match = re.search(r'def\s+(\w+)\s*\((.*?)\):', prompt)
        if func_match:
            func_info['name'] = func_match.group(1)
            params_str = func_match.group(2)
            
            # Parse parameters
            if params_str.strip():
                params = [p.strip() for p in params_str.split(',')]
                func_info['parameters'] = params
        
        # Extract docstring
        docstring_match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
        if docstring_match:
            func_info['docstring'] = docstring_match.group(1).strip()
        
        # Try to infer return type from docstring
        docstring = func_info['docstring'].lower()
        if 'return' in docstring:
            # Extract return description
            return_match = re.search(r'return[s]?\s+(.+?)(?:\.|$)', docstring)
            if return_match:
                return_desc = return_match.group(1).lower()
                if 'sum' in return_desc or 'total' in return_desc:
                    func_info['return_hint'] = 'sum'
                elif 'product' in return_desc or 'multiply' in return_desc:
                    func_info['return_hint'] = 'product'
                elif 'list' in return_desc:
                    func_info['return_hint'] = 'list'
                elif 'dict' in return_desc or 'dictionary' in return_desc:
                    func_info['return_hint'] = 'dict'
                elif 'string' in return_desc or 'str' in return_desc:
                    func_info['return_hint'] = 'string'
                elif 'bool' in return_desc or 'true' in return_desc or 'false' in return_desc:
                    func_info['return_hint'] = 'bool'
        
        return func_info
    
    def generate_code_completion(self, prompt: str) -> str:
        """
        Generate code completion for a function
        
        Returns:
            Generated code (function body)
        """
        func_info = self.parse_function_signature(prompt)
        
        if not func_info['name']:
            return "    pass"
        
        # Extract the incomplete function body
        lines = prompt.split('\n')
        incomplete_body = []
        in_function = False
        
        for line in lines:
            if line.strip().startswith('def '):
                in_function = True
                continue
            if in_function:
                if line.strip() and not line.strip().startswith('"""'):
                    incomplete_body.append(line)
        
        # Generate code based on function info
        params = func_info['parameters']
        return_hint = func_info['return_hint']
        docstring = func_info['docstring']
        
        # Simple code generation based on patterns
        if return_hint == 'sum':
            if len(params) >= 2:
                return f"    return {params[0]} + {params[1]}"
            elif len(params) == 1:
                return f"    return sum({params[0]})"
            else:
                return "    return 0"
        
        elif return_hint == 'product':
            if len(params) >= 2:
                return f"    return {params[0]} * {params[1]}"
            elif len(params) == 1:
                # Try to multiply elements
                return f"    result = 1\n    for x in {params[0]}:\n        result *= x\n    return result"
            else:
                return "    return 1"
        
        elif return_hint == 'list':
            return f"    return list({params[0] if params else '[]'})"
        
        elif return_hint == 'dict':
            return f"    return dict({params[0] if params else '{{}}'})"
        
        elif return_hint == 'string':
            if params:
                return f"    return str({params[0]})"
            else:
                return '    return ""'
        
        elif return_hint == 'bool':
            if params:
                return f"    return bool({params[0]})"
            else:
                return "    return True"
        
        # Default: try to infer from docstring keywords
        if 'add' in docstring.lower() or 'sum' in docstring.lower():
            if len(params) >= 2:
                return f"    return {params[0]} + {params[1]}"
        
        if 'multiply' in docstring.lower() or 'product' in docstring.lower():
            if len(params) >= 2:
                return f"    return {params[0]} * {params[1]}"
        
        # Use creative problem solving if available
        if self.problem_solver:
            try:
                # Create problem representation
                from Phase6_Creativity.creative_problem_solving import Problem
                problem = Problem(
                    problem_id=0,
                    description=docstring,
                    initial_state=np.array([0.0]),
                    goal_state=np.array([1.0]),
                    constraints=[],
                    domain="programming"
                )
                
                # Generate solution
                solution = self.problem_solver.solve_creatively(problem)
                if solution and solution.solution_description:
                    # Extract code from solution
                    code_lines = solution.solution_description.split('\n')
                    # Format as function body
                    formatted_code = '\n'.join(['    ' + line if line.strip() else line for line in code_lines])
                    return formatted_code
            except Exception:
                pass
        
        # Fallback: simple return statement
        if params:
            return f"    return {params[0]}"
        else:
            return "    pass"
    
    def validate_code_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python code syntax
        
        Returns:
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def extract_function_body(self, prompt: str) -> str:
        """
        Extract incomplete function body from prompt
        
        Returns:
            Function body text (without def line)
        """
        lines = prompt.split('\n')
        body_lines = []
        in_function = False
        indent_level = None
        
        for line in lines:
            if line.strip().startswith('def '):
                in_function = True
                # Detect indentation
                if ':' in line:
                    continue
            elif in_function:
                # Skip docstring lines
                if '"""' in line:
                    continue
                # Collect body lines
                if line.strip():
                    # Remove def-level indentation
                    if indent_level is None:
                        # Find indentation of first non-empty line
                        indent_level = len(line) - len(line.lstrip())
                    if indent_level > 0:
                        body_lines.append(line[indent_level:])
                    else:
                        body_lines.append(line)
        
        return '\n'.join(body_lines)
    
    def complete_function(self, prompt: str) -> str:
        """
        Complete a function from its prompt
        
        Returns:
            Complete function code
        """
        # Extract function signature
        func_match = re.search(r'(def\s+\w+\s*\([^)]*\):.*?)(?=\n\ndef|\Z)', prompt, re.DOTALL)
        if not func_match:
            return prompt
        
        signature_part = func_match.group(1)
        
        # Generate completion
        completion = self.generate_code_completion(prompt)
        
        # Combine signature with completion
        # Remove any existing incomplete return statement
        signature_lines = signature_part.split('\n')
        cleaned_signature = []
        for line in signature_lines:
            if line.strip().startswith('return') and not line.strip().endswith(':'):
                continue  # Skip incomplete return
            cleaned_signature.append(line)
        
        # Add completion
        if completion:
            cleaned_signature.append(completion)
        
        return '\n'.join(cleaned_signature)

