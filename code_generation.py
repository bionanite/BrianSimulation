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
        docstring_lower = docstring.lower()
        
        # Enhanced code generation with better pattern matching
        # Check for common patterns in docstring
        
        # Pattern 1: Sum/Add operations
        if return_hint == 'sum' or 'add' in docstring_lower or 'sum' in docstring_lower:
            if len(params) >= 2:
                # Check if params are lists/iterables
                if 'list' in docstring_lower or 'array' in docstring_lower:
                    return f"    return {params[0]} + {params[1]}"
                else:
                    return f"    return {params[0]} + {params[1]}"
            elif len(params) == 1:
                # Check if it's a list sum
                if 'list' in docstring_lower or 'elements' in docstring_lower:
                    return f"    return sum({params[0]})"
                else:
                    return f"    return {params[0]}"
            else:
                return "    return 0"
        
        # Pattern 2: Product/Multiply operations
        elif return_hint == 'product' or 'multiply' in docstring_lower or 'product' in docstring_lower:
            if len(params) >= 2:
                return f"    return {params[0]} * {params[1]}"
            elif len(params) == 1:
                # Check if it's a list product
                if 'list' in docstring_lower or 'elements' in docstring_lower:
                    return f"    result = 1\n    for x in {params[0]}:\n        result *= x\n    return result"
                else:
                    return f"    return {params[0]}"
            else:
                return "    return 1"
        
        # Pattern 3: List operations
        elif return_hint == 'list' or ('list' in docstring_lower and 'return' in docstring_lower):
            if params:
                # Check for list comprehension patterns
                if 'even' in docstring_lower or 'odd' in docstring_lower:
                    return f"    return [x for x in {params[0]} if x % 2 == 0]"
                elif 'filter' in docstring_lower:
                    return f"    return [x for x in {params[0]} if x]"
                else:
                    return f"    return list({params[0]})"
            else:
                return "    return []"
        
        # Pattern 4: Dictionary operations
        elif return_hint == 'dict' or ('dict' in docstring_lower and 'return' in docstring_lower):
            if params:
                return f"    return dict({params[0]})"
            else:
                return "    return {}"
        
        # Pattern 5: String operations
        elif return_hint == 'string' or ('string' in docstring_lower and 'return' in docstring_lower):
            if params:
                if 'reverse' in docstring_lower or 'backward' in docstring_lower:
                    return f"    return {params[0]}[::-1]"
                elif 'upper' in docstring_lower:
                    return f"    return {params[0]}.upper()"
                elif 'lower' in docstring_lower:
                    return f"    return {params[0]}.lower()"
                else:
                    return f"    return str({params[0]})"
            else:
                return '    return ""'
        
        # Pattern 6: Boolean operations
        elif return_hint == 'bool' or ('bool' in docstring_lower and 'return' in docstring_lower):
            if params:
                if 'even' in docstring_lower:
                    return f"    return {params[0]} % 2 == 0"
                elif 'odd' in docstring_lower:
                    return f"    return {params[0]} % 2 == 1"
                elif 'positive' in docstring_lower:
                    return f"    return {params[0]} > 0"
                elif 'negative' in docstring_lower:
                    return f"    return {params[0]} < 0"
                else:
                    return f"    return bool({params[0]})"
            else:
                return "    return True"
        
        # Pattern 7: Maximum/Minimum operations
        elif 'maximum' in docstring_lower or 'max' in docstring_lower:
            if params:
                if 'list' in docstring_lower or len(params) == 1:
                    return f"    return max({params[0]})"
                else:
                    return f"    return max({params[0]}, {params[1] if len(params) > 1 else ''})"
            else:
                return "    return 0"
        
        elif 'minimum' in docstring_lower or 'min' in docstring_lower:
            if params:
                if 'list' in docstring_lower or len(params) == 1:
                    return f"    return min({params[0]})"
                else:
                    return f"    return min({params[0]}, {params[1] if len(params) > 1 else ''})"
            else:
                return "    return 0"
        
        # Pattern 8: Count operations
        elif 'count' in docstring_lower:
            if params:
                if len(params) >= 2:
                    return f"    return {params[0]}.count({params[1]})"
                else:
                    return f"    return len({params[0]})"
            else:
                return "    return 0"
        
        # Pattern 9: Sort operations
        elif 'sort' in docstring_lower or 'sorted' in docstring_lower:
            if params:
                if 'reverse' in docstring_lower or 'descending' in docstring_lower:
                    return f"    return sorted({params[0]}, reverse=True)"
                else:
                    return f"    return sorted({params[0]})"
            else:
                return "    return []"
        
        # Use language generation system if available
        if self.language_generator:
            try:
                # Generate code using language generation
                code_description = f"Python function that {docstring}"
                generated_text = self.language_generator.generate_text(
                    prompt=code_description,
                    max_length=200,
                    temperature=0.7
                )
                # Extract code from generated text
                code_match = re.search(r'return\s+[^\n]+', generated_text)
                if code_match:
                    return f"    {code_match.group(0)}"
            except Exception:
                pass
        
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
        
        # Fallback: simple return statement based on parameters
        if params:
            # If only one param, return it
            if len(params) == 1:
                return f"    return {params[0]}"
            # If multiple params, try addition
            elif len(params) == 2:
                return f"    return {params[0]} + {params[1]}"
            else:
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
    
    def fix_code_syntax(self, code: str) -> str:
        """
        Attempt to fix common syntax errors in code
        
        Returns:
            Fixed code (or original if can't fix)
        """
        fixed = code
        
        # Fix common issues
        # 1. Missing colon after if/for/while
        fixed = re.sub(r'(if|for|while|elif|else)\s+([^:]+)$', r'\1 \2:', fixed, flags=re.MULTILINE)
        
        # 2. Missing return statement
        if 'return' not in fixed and fixed.strip():
            # Add return if code looks like an expression
            lines = fixed.strip().split('\n')
            last_line = lines[-1].strip()
            if last_line and not last_line.startswith('return') and not last_line.startswith('if') and not last_line.startswith('for'):
                # Check if it's an expression (not a statement)
                if not last_line.endswith(':') and '=' not in last_line:
                    lines[-1] = f"    return {last_line}"
                    fixed = '\n'.join(lines)
        
        # 3. Fix indentation issues
        lines = fixed.split('\n')
        fixed_lines = []
        for line in lines:
            if line.strip():
                # Ensure proper indentation (4 spaces)
                if not line.startswith(' '):
                    fixed_lines.append('    ' + line.strip())
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        fixed = '\n'.join(fixed_lines)
        
        return fixed
    
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

