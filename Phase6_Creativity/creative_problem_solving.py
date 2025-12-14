#!/usr/bin/env python3
"""
Creative Problem Solving - Phase 6.2
Implements analogical reasoning, constraint relaxation, reframing,
solution synthesis, and creative evaluation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
import random
from collections import defaultdict

# Import dependencies
try:
    from world_models import WorldModelManager
    from goal_setting_planning import GoalSettingPlanning
    from executive_control import ExecutiveControl
except ImportError:
    WorldModelManager = None
    GoalSettingPlanning = None
    ExecutiveControl = None


@dataclass
class Problem:
    """Represents a problem to solve"""
    problem_id: int
    description: str
    initial_state: np.ndarray
    goal_state: np.ndarray
    constraints: List[Dict]  # List of constraint dictionaries
    domain: str  # Problem domain
    created_time: float = 0.0


@dataclass
class Solution:
    """Represents a solution to a problem"""
    solution_id: int
    problem_id: int
    description: str
    solution_steps: List[np.ndarray]  # Sequence of states
    creativity_score: float
    feasibility_score: float
    novelty_score: float
    created_time: float = 0.0
    method: str = ''  # Method used to generate solution


class AnalogicalReasoning:
    """
    Analogical Reasoning
    
    Transfers solutions from similar problems
    Finds analogies between problem domains
    """
    
    def __init__(self,
                 similarity_threshold: float = 0.6,
                 transfer_strength: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.transfer_strength = transfer_strength
        self.problem_solutions: Dict[int, List[Solution]] = defaultdict(list)
        self.analogies_found: List[Tuple[int, int, float]] = []  # (problem1_id, problem2_id, similarity)
    
    def find_analogous_problems(self,
                               problem: Problem,
                               problem_database: Dict[int, Problem]) -> List[Tuple[int, float]]:
        """
        Find problems analogous to the given problem
        
        Returns:
            List of (problem_id, similarity) tuples
        """
        analogous = []
        
        for other_id, other_problem in problem_database.items():
            if other_id == problem.problem_id:
                continue
            
            # Compute similarity
            similarity = self._compute_problem_similarity(problem, other_problem)
            
            if similarity >= self.similarity_threshold:
                analogous.append((other_id, similarity))
        
        # Sort by similarity
        analogous.sort(key=lambda x: x[1], reverse=True)
        return analogous
    
    def _compute_problem_similarity(self, problem1: Problem, problem2: Problem) -> float:
        """Compute similarity between two problems"""
        # Compare initial states
        state_sim = np.dot(problem1.initial_state, problem2.initial_state) / (
            np.linalg.norm(problem1.initial_state) * np.linalg.norm(problem2.initial_state) + 1e-10
        )
        
        # Compare goal states
        goal_sim = np.dot(problem1.goal_state, problem2.goal_state) / (
            np.linalg.norm(problem1.goal_state) * np.linalg.norm(problem2.goal_state) + 1e-10
        )
        
        # Compare domains
        domain_sim = 1.0 if problem1.domain == problem2.domain else 0.3
        
        # Weighted combination
        similarity = 0.4 * state_sim + 0.4 * goal_sim + 0.2 * domain_sim
        
        return similarity
    
    def transfer_solution(self,
                         source_solution: Solution,
                         target_problem: Problem) -> Optional[Solution]:
        """
        Transfer solution from source problem to target problem
        
        Returns:
            Transferred solution or None
        """
        # Adapt solution steps to target problem
        adapted_steps = []
        
        for step in source_solution.solution_steps:
            # Map step from source space to target space
            # Simple linear mapping based on problem similarity
            adapted_step = self._adapt_state(step, source_solution.problem_id, target_problem)
            adapted_steps.append(adapted_step)
        
        # Create new solution
        transferred_solution = Solution(
            solution_id=-1,  # Will be assigned later
            problem_id=target_problem.problem_id,
            description=f"Transferred from problem {source_solution.problem_id}",
            solution_steps=adapted_steps,
            creativity_score=source_solution.creativity_score * self.transfer_strength,
            feasibility_score=source_solution.feasibility_score * 0.8,  # Lower feasibility after transfer
            novelty_score=source_solution.novelty_score * 0.6,  # Less novel after transfer
            method='analogical_transfer'
        )
        
        return transferred_solution
    
    def _adapt_state(self, state: np.ndarray, source_problem_id: int, target_problem: Problem) -> np.ndarray:
        """Adapt state from source problem to target problem"""
        # Simple linear transformation
        # In practice, this would be more sophisticated
        adapted = state.copy()
        
        # Scale to match target problem's state space
        target_norm = np.linalg.norm(target_problem.initial_state)
        source_norm = np.linalg.norm(state)
        
        if source_norm > 0 and target_norm > 0:
            scale = target_norm / source_norm
            adapted = adapted * scale
        
        # Add some noise for adaptation
        noise = np.random.normal(0, 0.1, len(adapted))
        adapted = adapted + noise
        
        # Normalize
        norm = np.linalg.norm(adapted)
        if norm > 0:
            adapted = adapted / norm
        
        return adapted


class ConstraintRelaxation:
    """
    Constraint Relaxation
    
    Temporarily relaxes constraints to explore solution space
    Finds solutions that violate constraints, then adapts them
    """
    
    def __init__(self,
                 relaxation_strength: float = 0.5,
                 num_relaxations: int = 3):
        self.relaxation_strength = relaxation_strength
        self.num_relaxations = num_relaxations
        self.relaxation_history: List[Dict] = []
    
    def relax_constraints(self, problem: Problem) -> List[Problem]:
        """
        Create relaxed versions of the problem
        
        Returns:
            List of problems with relaxed constraints
        """
        relaxed_problems = []
        
        if not problem.constraints:
            return relaxed_problems
        
        for i in range(self.num_relaxations):
            relaxed_problem = Problem(
                problem_id=problem.problem_id * 1000 + i,  # Unique ID
                description=f"{problem.description} (relaxed {i+1})",
                initial_state=problem.initial_state.copy(),
                goal_state=problem.goal_state.copy(),
                constraints=self._relax_constraint_set(problem.constraints, i),
                domain=problem.domain,
                created_time=time.time()
            )
            relaxed_problems.append(relaxed_problem)
        
        return relaxed_problems
    
    def _relax_constraint_set(self, constraints: List[Dict], relaxation_index: int) -> List[Dict]:
        """Relax a set of constraints"""
        relaxed = []
        
        for i, constraint in enumerate(constraints):
            relaxed_constraint = constraint.copy()
            
            # Relax constraint based on type
            if 'threshold' in constraint:
                # Increase threshold (make constraint easier)
                relaxed_constraint['threshold'] = constraint['threshold'] * (
                    1.0 + self.relaxation_strength * (relaxation_index + 1)
                )
            elif 'weight' in constraint:
                # Reduce weight (make constraint less important)
                relaxed_constraint['weight'] = constraint['weight'] * (
                    1.0 - self.relaxation_strength * 0.3 * (relaxation_index + 1)
                )
            
            relaxed.append(relaxed_constraint)
        
        return relaxed
    
    def adapt_solution_to_constraints(self,
                                     solution: Solution,
                                     original_problem: Problem) -> Solution:
        """
        Adapt a solution from relaxed problem back to original constraints
        
        Returns:
            Adapted solution
        """
        # Simple adaptation: adjust solution steps to satisfy constraints
        adapted_steps = []
        
        for step in solution.solution_steps:
            adapted_step = self._satisfy_constraints(step, original_problem.constraints)
            adapted_steps.append(adapted_step)
        
        adapted_solution = Solution(
            solution_id=solution.solution_id,
            problem_id=original_problem.problem_id,
            description=f"{solution.description} (adapted to constraints)",
            solution_steps=adapted_steps,
            creativity_score=solution.creativity_score * 0.9,
            feasibility_score=solution.feasibility_score * 1.1,  # Higher feasibility after adaptation
            novelty_score=solution.novelty_score,
            method=solution.method + '_adapted'
        )
        
        return adapted_solution
    
    def _satisfy_constraints(self, state: np.ndarray, constraints: List[Dict]) -> np.ndarray:
        """Modify state to satisfy constraints"""
        satisfied_state = state.copy()
        
        for constraint in constraints:
            if 'threshold' in constraint:
                # Ensure state doesn't exceed threshold
                if np.max(satisfied_state) > constraint['threshold']:
                    satisfied_state = satisfied_state * (
                        constraint['threshold'] / (np.max(satisfied_state) + 1e-10)
                    )
        
        return satisfied_state


class ProblemReframing:
    """
    Problem Reframing
    
    Views problems from different perspectives
    Reformulates problems to enable new solutions
    """
    
    def __init__(self,
                 num_reframes: int = 3):
        self.num_reframes = num_reframes
        self.reframing_history: List[Dict] = []
    
    def reframe_problem(self, problem: Problem) -> List[Problem]:
        """
        Generate reframed versions of the problem
        
        Returns:
            List of reframed problems
        """
        reframed = []
        
        reframing_strategies = [
            'inverse',  # Invert goal and initial state
            'decompose',  # Break into subproblems
            'generalize',  # Generalize to broader problem
            'specialize',  # Specialize to narrower problem
        ]
        
        for i, strategy in enumerate(reframing_strategies[:self.num_reframes]):
            reframed_problem = self._apply_reframing_strategy(problem, strategy, i)
            if reframed_problem:
                reframed.append(reframed_problem)
        
        return reframed
    
    def _apply_reframing_strategy(self,
                                 problem: Problem,
                                 strategy: str,
                                 index: int) -> Optional[Problem]:
        """Apply a specific reframing strategy"""
        if strategy == 'inverse':
            # Swap initial and goal states
            return Problem(
                problem_id=problem.problem_id * 10000 + index,
                description=f"{problem.description} (inverted)",
                initial_state=problem.goal_state.copy(),
                goal_state=problem.initial_state.copy(),
                constraints=problem.constraints.copy(),
                domain=problem.domain,
                created_time=time.time()
            )
        
        elif strategy == 'decompose':
            # Create subproblem (simplified version)
            mid_state = (problem.initial_state + problem.goal_state) / 2.0
            return Problem(
                problem_id=problem.problem_id * 10000 + index,
                description=f"{problem.description} (subproblem)",
                initial_state=problem.initial_state.copy(),
                goal_state=mid_state,
                constraints=problem.constraints.copy(),
                domain=problem.domain,
                created_time=time.time()
            )
        
        elif strategy == 'generalize':
            # Generalize: reduce specificity
            generalized_initial = problem.initial_state * 0.8 + np.random.normal(0, 0.1, len(problem.initial_state))
            generalized_goal = problem.goal_state * 0.8 + np.random.normal(0, 0.1, len(problem.goal_state))
            
            # Normalize
            for vec in [generalized_initial, generalized_goal]:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            
            return Problem(
                problem_id=problem.problem_id * 10000 + index,
                description=f"{problem.description} (generalized)",
                initial_state=generalized_initial,
                goal_state=generalized_goal,
                constraints=problem.constraints.copy(),
                domain=problem.domain,
                created_time=time.time()
            )
        
        elif strategy == 'specialize':
            # Specialize: increase specificity
            specialized_initial = problem.initial_state * 1.2
            specialized_goal = problem.goal_state * 1.2
            
            # Normalize
            for vec in [specialized_initial, specialized_goal]:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            
            return Problem(
                problem_id=problem.problem_id * 10000 + index,
                description=f"{problem.description} (specialized)",
                initial_state=specialized_initial,
                goal_state=specialized_goal,
                constraints=problem.constraints.copy(),
                domain=problem.domain,
                created_time=time.time()
            )
        
        return None


class SolutionSynthesis:
    """
    Solution Synthesis
    
    Combines partial solutions into complete solutions
    Merges solutions from different approaches
    """
    
    def __init__(self,
                 synthesis_strength: float = 0.5):
        self.synthesis_strength = synthesis_strength
        self.synthesized_solutions: List[Solution] = []
    
    def synthesize_solutions(self,
                            partial_solutions: List[Solution],
                            problem: Problem) -> Optional[Solution]:
        """
        Synthesize multiple partial solutions into one complete solution
        
        Returns:
            Synthesized solution or None
        """
        if not partial_solutions:
            return None
        
        if len(partial_solutions) == 1:
            return partial_solutions[0]
        
        # Combine solution steps
        synthesized_steps = []
        
        # Interleave steps from different solutions
        max_steps = max(len(sol.solution_steps) for sol in partial_solutions)
        
        for step_idx in range(max_steps):
            step_contributions = []
            
            for sol in partial_solutions:
                if step_idx < len(sol.solution_steps):
                    step_contributions.append(sol.solution_steps[step_idx])
            
            if step_contributions:
                # Average or blend steps
                synthesized_step = np.mean(step_contributions, axis=0)
                
                # Normalize
                norm = np.linalg.norm(synthesized_step)
                if norm > 0:
                    synthesized_step = synthesized_step / norm
                
                synthesized_steps.append(synthesized_step)
        
        # Compute aggregated scores
        avg_creativity = np.mean([sol.creativity_score for sol in partial_solutions])
        avg_feasibility = np.mean([sol.feasibility_score for sol in partial_solutions])
        avg_novelty = np.mean([sol.novelty_score for sol in partial_solutions])
        
        synthesized_solution = Solution(
            solution_id=-1,  # Will be assigned later
            problem_id=problem.problem_id,
            description=f"Synthesized from {len(partial_solutions)} solutions",
            solution_steps=synthesized_steps,
            creativity_score=avg_creativity * (1.0 + self.synthesis_strength * 0.2),  # Slight boost
            feasibility_score=avg_feasibility,
            novelty_score=avg_novelty * 0.9,  # Slight reduction
            method='synthesis'
        )
        
        self.synthesized_solutions.append(synthesized_solution)
        return synthesized_solution


class CreativeEvaluation:
    """
    Creative Evaluation
    
    Assesses creativity and novelty of solutions
    Evaluates solution quality from multiple perspectives
    """
    
    def __init__(self,
                 novelty_weight: float = 0.4,
                 feasibility_weight: float = 0.3,
                 originality_weight: float = 0.3):
        self.novelty_weight = novelty_weight
        self.feasibility_weight = feasibility_weight
        self.originality_weight = originality_weight
        self.evaluation_history: List[Dict] = []
    
    def evaluate_solution(self,
                         solution: Solution,
                         problem: Problem,
                         solution_database: List[Solution]) -> Dict:
        """
        Evaluate a solution's creativity
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Compute novelty relative to other solutions
        novelty = self._compute_novelty(solution, solution_database)
        
        # Compute feasibility
        feasibility = solution.feasibility_score
        
        # Compute originality (how different from standard approaches)
        originality = self._compute_originality(solution, problem)
        
        # Overall creativity score
        creativity = (
            self.novelty_weight * novelty +
            self.feasibility_weight * feasibility +
            self.originality_weight * originality
        )
        
        evaluation = {
            'solution_id': solution.solution_id,
            'creativity_score': creativity,
            'novelty_score': novelty,
            'feasibility_score': feasibility,
            'originality_score': originality,
            'overall_quality': (creativity + feasibility) / 2.0
        }
        
        self.evaluation_history.append(evaluation)
        return evaluation
    
    def _compute_novelty(self, solution: Solution, solution_database: List[Solution]) -> float:
        """Compute novelty relative to other solutions"""
        if not solution_database:
            return 1.0
        
        similarities = []
        for other_sol in solution_database:
            if other_sol.solution_id == solution.solution_id:
                continue
            
            # Compare solution steps
            if solution.solution_steps and other_sol.solution_steps:
                # Compare first step as proxy
                step1 = solution.solution_steps[0]
                step2 = other_sol.solution_steps[0]
                
                similarity = np.dot(step1, step2) / (
                    np.linalg.norm(step1) * np.linalg.norm(step2) + 1e-10
                )
                similarities.append(similarity)
        
        if not similarities:
            return 1.0
        
        max_similarity = max(similarities)
        novelty = 1.0 - max_similarity
        
        return max(0.0, novelty)
    
    def _compute_originality(self, solution: Solution, problem: Problem) -> float:
        """Compute originality of solution approach"""
        # Check if solution method is standard
        standard_methods = ['direct', 'greedy', 'brute_force']
        
        if solution.method in standard_methods:
            originality = 0.3
        elif 'creative' in solution.method.lower() or 'novel' in solution.method.lower():
            originality = 0.9
        else:
            originality = 0.6
        
        # Boost if solution steps are non-linear
        if len(solution.solution_steps) > 2:
            # Check for non-monotonic progression
            progressions = []
            for i in range(len(solution.solution_steps) - 1):
                step1 = solution.solution_steps[i]
                step2 = solution.solution_steps[i + 1]
                
                # Distance to goal
                dist1 = np.linalg.norm(step1 - problem.goal_state)
                dist2 = np.linalg.norm(step2 - problem.goal_state)
                
                progressions.append(dist2 - dist1)
            
            # Non-monotonic progressions indicate creative approach
            if any(p > 0 for p in progressions):  # Some steps move away from goal
                originality = min(1.0, originality + 0.2)
        
        return originality


class CreativeProblemSolving:
    """
    Creative Problem Solving Manager
    
    Integrates all creative problem-solving components
    """
    
    def __init__(self,
                 brain_system=None,
                 world_model: Optional[WorldModelManager] = None,
                 goal_setting: Optional[GoalSettingPlanning] = None,
                 executive_control: Optional[ExecutiveControl] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.analogical_reasoning = AnalogicalReasoning()
        self.constraint_relaxation = ConstraintRelaxation()
        self.problem_reframing = ProblemReframing()
        self.solution_synthesis = SolutionSynthesis()
        self.creative_evaluation = CreativeEvaluation()
        
        # Integration with existing systems
        self.world_model = world_model
        self.goal_setting = goal_setting
        self.executive_control = executive_control
        
        # Problem and solution tracking
        self.problems: Dict[int, Problem] = {}
        self.solutions: Dict[int, List[Solution]] = defaultdict(list)
        self.next_problem_id = 0
        self.next_solution_id = 0
        
        # Statistics
        self.stats = {
            'problems_solved': 0,
            'creative_solutions': 0,
            'analogies_used': 0,
            'reframings_applied': 0,
            'average_creativity': 0.0
        }
    
    def solve_problem(self,
                     problem: Problem,
                     method: str = 'creative') -> List[Solution]:
        """
        Solve a problem using creative methods
        
        Args:
            problem: Problem to solve
            method: Solving method ('analogical', 'relaxation', 'reframing', 'creative')
        
        Returns:
            List of solutions
        """
        # Store problem
        problem.problem_id = self.next_problem_id
        self.next_problem_id += 1
        self.problems[problem.problem_id] = problem
        
        solutions = []
        
        if method == 'analogical':
            solutions = self._solve_with_analogy(problem)
        elif method == 'relaxation':
            solutions = self._solve_with_relaxation(problem)
        elif method == 'reframing':
            solutions = self._solve_with_reframing(problem)
        else:  # 'creative' - try all methods
            solutions.extend(self._solve_with_analogy(problem))
            solutions.extend(self._solve_with_relaxation(problem))
            solutions.extend(self._solve_with_reframing(problem))
        
        # Evaluate solutions
        evaluated_solutions = []
        all_solutions = []
        for task_solutions in self.solutions.values():
            all_solutions.extend(task_solutions)
        
        for solution in solutions:
            solution.solution_id = self.next_solution_id
            self.next_solution_id += 1
            
            evaluation = self.creative_evaluation.evaluate_solution(
                solution, problem, all_solutions
            )
            
            # Update solution scores
            solution.creativity_score = evaluation['creativity_score']
            solution.novelty_score = evaluation['novelty_score']
            solution.feasibility_score = evaluation['feasibility_score']
            
            evaluated_solutions.append(solution)
            self.solutions[problem.problem_id].append(solution)
        
        # Update statistics
        self._update_stats(evaluated_solutions)
        
        return evaluated_solutions
    
    def _solve_with_analogy(self, problem: Problem) -> List[Solution]:
        """Solve using analogical reasoning"""
        solutions = []
        
        # Find analogous problems
        analogous = self.analogical_reasoning.find_analogous_problems(
            problem, self.problems
        )
        
        for other_id, similarity in analogous[:3]:  # Top 3 analogies
            # Get solutions from analogous problem
            if other_id in self.solutions:
                for source_solution in self.solutions[other_id]:
                    # Transfer solution
                    transferred = self.analogical_reasoning.transfer_solution(
                        source_solution, problem
                    )
                    if transferred:
                        solutions.append(transferred)
                        self.stats['analogies_used'] += 1
        
        return solutions
    
    def _solve_with_relaxation(self, problem: Problem) -> List[Solution]:
        """Solve using constraint relaxation"""
        solutions = []
        
        # Relax constraints
        relaxed_problems = self.constraint_relaxation.relax_constraints(problem)
        
        for relaxed_problem in relaxed_problems:
            # Solve relaxed problem (simplified - in practice would use world model)
            relaxed_solution = self._solve_simple(relaxed_problem)
            
            if relaxed_solution:
                # Adapt back to original constraints
                adapted = self.constraint_relaxation.adapt_solution_to_constraints(
                    relaxed_solution, problem
                )
                solutions.append(adapted)
        
        return solutions
    
    def _solve_with_reframing(self, problem: Problem) -> List[Solution]:
        """Solve using problem reframing"""
        solutions = []
        
        # Reframe problem
        reframed_problems = self.problem_reframing.reframe_problem(problem)
        self.stats['reframings_applied'] += len(reframed_problems)
        
        for reframed_problem in reframed_problems:
            # Solve reframed problem
            reframed_solution = self._solve_simple(reframed_problem)
            
            if reframed_solution:
                # Map solution back to original problem space
                solutions.append(reframed_solution)
        
        return solutions
    
    def _solve_simple(self, problem: Problem) -> Optional[Solution]:
        """Simple problem solver (placeholder - would use world model in practice)"""
        # Generate a simple solution path
        num_steps = 5
        steps = []
        
        current_state = problem.initial_state.copy()
        goal_state = problem.goal_state.copy()
        
        for i in range(num_steps):
            # Interpolate towards goal
            alpha = (i + 1) / num_steps
            step = (1 - alpha) * current_state + alpha * goal_state
            
            # Normalize
            norm = np.linalg.norm(step)
            if norm > 0:
                step = step / norm
            
            steps.append(step)
        
        solution = Solution(
            solution_id=-1,
            problem_id=problem.problem_id,
            description="Simple solution",
            solution_steps=steps,
            creativity_score=0.5,
            feasibility_score=0.8,
            novelty_score=0.3,
            method='simple'
        )
        
        return solution
    
    def synthesize_best_solutions(self, problem_id: int, top_k: int = 3) -> Optional[Solution]:
        """Synthesize top solutions for a problem"""
        if problem_id not in self.solutions:
            return None
        
        solutions = self.solutions[problem_id]
        if not solutions:
            return None
        
        # Get top solutions by creativity
        sorted_solutions = sorted(solutions, key=lambda x: x.creativity_score, reverse=True)
        top_solutions = sorted_solutions[:top_k]
        
        # Synthesize
        synthesized = self.solution_synthesis.synthesize_solutions(top_solutions, self.problems[problem_id])
        
        return synthesized
    
    def _update_stats(self, solutions: List[Solution]):
        """Update statistics"""
        if not solutions:
            return
        
        self.stats['problems_solved'] += 1
        
        creative_count = sum(1 for sol in solutions if sol.creativity_score > 0.6)
        self.stats['creative_solutions'] += creative_count
        
        avg_creativity = np.mean([sol.creativity_score for sol in solutions])
        total_problems = self.stats['problems_solved']
        self.stats['average_creativity'] = (
            (self.stats['average_creativity'] * (total_problems - 1) + avg_creativity) / total_problems
        )
    
    def get_statistics(self) -> Dict:
        """Get problem-solving statistics"""
        return self.stats.copy()

