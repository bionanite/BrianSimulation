#!/usr/bin/env python3
"""
Long-Term Strategic Planning - Phase 9.1
Implements multi-horizon planning, scenario planning, resource management,
strategic goal decomposition, and plan monitoring & adaptation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
from collections import defaultdict, deque

# Import dependencies
try:
    from goal_setting_planning import GoalSettingPlanning, Goal, Plan
    from world_models import WorldModelManager
    from executive_control import ExecutiveControl
except ImportError:
    GoalSettingPlanning = None
    Goal = None
    Plan = None
    WorldModelManager = None
    ExecutiveControl = None

# Fallback Plan dataclass if import fails
if Plan is None:
    @dataclass
    class Plan:
        """Represents a plan to achieve a goal"""
        plan_id: int
        goal_id: int
        actions: List[int]
        expected_states: List[np.ndarray]
        confidence: float = 1.0
        created_time: float = 0.0


@dataclass
class StrategicGoal:
    """Represents a strategic goal"""
    goal_id: int
    description: str
    target_state: np.ndarray
    time_horizon: float  # Days/months/years
    priority: float = 1.0
    strategic_importance: float = 1.0
    tactical_steps: List[int] = field(default_factory=list)
    scenarios: List[int] = field(default_factory=list)
    created_time: float = 0.0
    status: str = 'active'


@dataclass
class Scenario:
    """Represents a future scenario"""
    scenario_id: int
    name: str
    probability: float
    conditions: Dict[str, float]
    outcomes: Dict[str, np.ndarray]
    created_time: float = 0.0


@dataclass
class Resource:
    """Represents a resource"""
    resource_id: int
    name: str
    current_amount: float
    max_amount: float
    replenishment_rate: float = 0.0
    usage_history: List[Tuple[float, float]] = field(default_factory=list)  # (time, amount)


class MultiHorizonPlanning:
    """
    Multi-Horizon Planning
    
    Plans across multiple time horizons
    Short-term, medium-term, long-term planning
    """
    
    def __init__(self,
                 horizons: List[float] = None):  # [short, medium, long] in days
        if horizons is None:
            horizons = [7.0, 30.0, 365.0]  # 1 week, 1 month, 1 year
        self.horizons = horizons
        self.plans_by_horizon: Dict[float, List[Plan]] = defaultdict(list)
    
    def create_multi_horizon_plan(self,
                                 goal: StrategicGoal,
                                 world_model: Optional[WorldModelManager] = None) -> Dict[float, Plan]:
        """
        Create plans for multiple time horizons
        
        Returns:
            Dictionary mapping horizon -> plan
        """
        plans = {}
        
        for horizon in self.horizons:
            # Create plan for this horizon
            plan = self._create_horizon_plan(goal, horizon, world_model)
            plans[horizon] = plan
            self.plans_by_horizon[horizon].append(plan)
        
        return plans
    
    def _create_horizon_plan(self,
                            goal: StrategicGoal,
                            horizon: float,
                            world_model: Optional[WorldModelManager]) -> Plan:
        """Create plan for a specific horizon"""
        # Interpolate target state based on horizon
        if goal.time_horizon > 0:
            progress = min(1.0, horizon / goal.time_horizon)
            target_state = goal.target_state * progress
        else:
            target_state = goal.target_state.copy()
        
        # Generate actions (simplified)
        num_actions = max(1, int(horizon / 7.0))  # Roughly one action per week
        actions = list(range(num_actions))
        
        # Generate expected states
        expected_states = []
        current_state = np.zeros_like(goal.target_state)
        for i in range(num_actions):
            # Interpolate towards target
            alpha = (i + 1) / num_actions
            state = current_state + alpha * (target_state - current_state)
            expected_states.append(state)
        
        plan = Plan(
            plan_id=-1,  # Will be assigned later
            goal_id=goal.goal_id,
            actions=actions,
            expected_states=expected_states,
            confidence=1.0 - (horizon / goal.time_horizon) if goal.time_horizon > 0 else 0.5,
            created_time=time.time()
        )
        
        return plan
    
    def align_horizons(self, plans: Dict[float, Plan]) -> Dict[float, Plan]:
        """Align plans across horizons to ensure consistency"""
        # Ensure later horizons build on earlier ones
        sorted_horizons = sorted(plans.keys())
        
        for i in range(1, len(sorted_horizons)):
            prev_horizon = sorted_horizons[i-1]
            curr_horizon = sorted_horizons[i]
            
            prev_plan = plans[prev_horizon]
            curr_plan = plans[curr_horizon]
            
            # Ensure current plan starts where previous ended
            if prev_plan.expected_states:
                final_state = prev_plan.expected_states[-1]
                if curr_plan.expected_states:
                    curr_plan.expected_states[0] = final_state.copy()
        
        return plans


class ScenarioPlanning:
    """
    Scenario Planning
    
    Plans for multiple possible futures
    Considers different scenarios
    """
    
    def __init__(self):
        self.scenarios: Dict[int, Scenario] = {}
        self.next_scenario_id = 0
        self.scenario_plans: Dict[int, Dict[int, Plan]] = {}  # goal_id -> scenario_id -> plan
    
    def create_scenarios(self,
                        goal: StrategicGoal,
                        num_scenarios: int = 3) -> List[Scenario]:
        """
        Create multiple scenarios for a goal
        
        Returns:
            List of scenarios
        """
        scenarios = []
        
        scenario_types = ['optimistic', 'realistic', 'pessimistic']
        
        for i, scenario_type in enumerate(scenario_types[:num_scenarios]):
            if scenario_type == 'optimistic':
                probability = 0.3
                conditions = {'success_rate': 0.9, 'resource_availability': 0.8}
                outcomes = {'target_state': goal.target_state * 1.1}
            elif scenario_type == 'realistic':
                probability = 0.5
                conditions = {'success_rate': 0.7, 'resource_availability': 0.6}
                outcomes = {'target_state': goal.target_state}
            else:  # pessimistic
                probability = 0.2
                conditions = {'success_rate': 0.5, 'resource_availability': 0.4}
                outcomes = {'target_state': goal.target_state * 0.8}
            
            scenario = Scenario(
                scenario_id=self.next_scenario_id,
                name=f"{goal.description} - {scenario_type}",
                probability=probability,
                conditions=conditions,
                outcomes=outcomes,
                created_time=time.time()
            )
            
            self.scenarios[self.next_scenario_id] = scenario
            goal.scenarios.append(self.next_scenario_id)
            self.next_scenario_id += 1
            scenarios.append(scenario)
        
        return scenarios
    
    def plan_for_scenario(self,
                         goal: StrategicGoal,
                         scenario: Scenario,
                         world_model: Optional[WorldModelManager] = None) -> Plan:
        """
        Create plan for a specific scenario
        
        Returns:
            Plan for scenario
        """
        # Adjust goal based on scenario
        scenario_target = scenario.outcomes.get('target_state', goal.target_state)
        
        # Create plan
        num_actions = 10
        actions = list(range(num_actions))
        
        expected_states = []
        current_state = np.zeros_like(scenario_target)
        for i in range(num_actions):
            alpha = (i + 1) / num_actions
            state = current_state + alpha * (scenario_target - current_state)
            expected_states.append(state)
        
        plan = Plan(
            plan_id=-1,
            goal_id=goal.goal_id,
            actions=actions,
            expected_states=expected_states,
            confidence=scenario.probability,
            created_time=time.time()
        )
        
        # Store scenario plan
        if goal.goal_id not in self.scenario_plans:
            self.scenario_plans[goal.goal_id] = {}
        self.scenario_plans[goal.goal_id][scenario.scenario_id] = plan
        
        return plan
    
    def select_best_scenario_plan(self, goal: StrategicGoal) -> Optional[Plan]:
        """Select best plan based on scenario probabilities"""
        if goal.goal_id not in self.scenario_plans:
            return None
        
        scenario_plans = self.scenario_plans[goal.goal_id]
        if not scenario_plans:
            return None
        
        # Weight plans by scenario probability
        best_plan = None
        best_score = -1.0
        
        for scenario_id, plan in scenario_plans.items():
            if scenario_id in self.scenarios:
                scenario = self.scenarios[scenario_id]
                score = scenario.probability * plan.confidence
                if score > best_score:
                    best_score = score
                    best_plan = plan
        
        return best_plan


class ResourceManagement:
    """
    Resource Management
    
    Manages limited resources over time
    Allocates resources to plans
    """
    
    def __init__(self):
        self.resources: Dict[int, Resource] = {}
        self.next_resource_id = 0
        self.allocations: Dict[int, Dict[int, float]] = {}  # plan_id -> resource_id -> amount
    
    def create_resource(self,
                       name: str,
                       initial_amount: float,
                       max_amount: float,
                       replenishment_rate: float = 0.0) -> Resource:
        """Create a new resource"""
        resource = Resource(
            resource_id=self.next_resource_id,
            name=name,
            current_amount=initial_amount,
            max_amount=max_amount,
            replenishment_rate=replenishment_rate
        )
        
        self.resources[self.next_resource_id] = resource
        self.next_resource_id += 1
        return resource
    
    def allocate_resource(self,
                         plan_id: int,
                         resource_id: int,
                         amount: float) -> bool:
        """
        Allocate resource to a plan
        
        Returns:
            Success status
        """
        if resource_id not in self.resources:
            return False
        
        resource = self.resources[resource_id]
        
        # Check availability
        if resource.current_amount < amount:
            return False
        
        # Allocate
        resource.current_amount -= amount
        
        if plan_id not in self.allocations:
            self.allocations[plan_id] = {}
        self.allocations[plan_id][resource_id] = amount
        
        # Record usage
        resource.usage_history.append((time.time(), amount))
        
        return True
    
    def update_resources(self, delta_time: float):
        """Update resources over time (replenishment)"""
        for resource in self.resources.values():
            # Replenish
            new_amount = resource.current_amount + resource.replenishment_rate * delta_time
            resource.current_amount = min(new_amount, resource.max_amount)
    
    def check_resource_constraints(self, plan: Plan) -> Tuple[bool, List[str]]:
        """
        Check if plan satisfies resource constraints
        
        Returns:
            (is_feasible, missing_resources)
        """
        if plan.plan_id not in self.allocations:
            return True, []
        
        missing = []
        for resource_id, required_amount in self.allocations[plan.plan_id].items():
            if resource_id in self.resources:
                resource = self.resources[resource_id]
                if resource.current_amount < required_amount:
                    missing.append(resource.name)
        
        return len(missing) == 0, missing


class StrategicGoalDecomposition:
    """
    Strategic Goal Decomposition
    
    Breaks strategic goals into tactical steps
    Creates hierarchical goal structure
    """
    
    def __init__(self):
        self.decompositions: Dict[int, List[int]] = {}  # goal_id -> subgoal_ids
    
    def decompose_strategic_goal(self,
                                strategic_goal: StrategicGoal,
                                num_tactical_steps: int = 5,
                                goal_hierarchy: Optional[GoalSettingPlanning] = None) -> List[Goal]:
        """
        Decompose strategic goal into tactical steps
        
        Returns:
            List of tactical goals
        """
        tactical_goals = []
        
        # Create intermediate states
        initial_state = np.zeros_like(strategic_goal.target_state)
        target_state = strategic_goal.target_state
        
        for i in range(num_tactical_steps):
            # Interpolate intermediate state
            alpha = (i + 1) / num_tactical_steps
            intermediate_state = initial_state + alpha * (target_state - initial_state)
            
            # Create tactical goal
            if goal_hierarchy:
                tactical_goal = goal_hierarchy.goal_hierarchy.create_goal(
                    description=f"Tactical step {i+1} for {strategic_goal.description}",
                    target_state=intermediate_state,
                    priority=strategic_goal.priority * (1.0 - i * 0.1),
                    parent_goal_id=strategic_goal.goal_id
                )
            else:
                # Create simplified goal
                from goal_setting_planning import Goal
                tactical_goal = Goal(
                    goal_id=-1,
                    description=f"Tactical step {i+1}",
                    target_state=intermediate_state,
                    priority=strategic_goal.priority * (1.0 - i * 0.1)
                )
            
            tactical_goals.append(tactical_goal)
            strategic_goal.tactical_steps.append(tactical_goal.goal_id)
        
        self.decompositions[strategic_goal.goal_id] = [g.goal_id for g in tactical_goals]
        
        return tactical_goals


class PlanMonitoring:
    """
    Plan Monitoring & Adaptation
    
    Monitors plan execution and adapts dynamically
    Detects failures and adjusts plans
    """
    
    def __init__(self,
                 monitoring_frequency: float = 1.0,  # Check every N time units
                 adaptation_threshold: float = 0.3):  # Adapt if deviation > threshold
        self.monitoring_frequency = monitoring_frequency
        self.adaptation_threshold = adaptation_threshold
        self.monitoring_history: Dict[int, List[Dict]] = {}  # plan_id -> monitoring records
        self.adaptations: List[Dict] = []
    
    def monitor_plan(self,
                    plan: Plan,
                    current_state: np.ndarray,
                    expected_state: Optional[np.ndarray] = None) -> Dict:
        """
        Monitor plan execution
        
        Returns:
            Monitoring results
        """
        if expected_state is None:
            # Use next expected state from plan
            if plan.expected_states:
                expected_state = plan.expected_states[0]
            else:
                expected_state = current_state
        
        # Compute deviation
        deviation = np.linalg.norm(current_state - expected_state)
        max_deviation = np.linalg.norm(expected_state) + 1e-10
        relative_deviation = deviation / max_deviation
        
        # Check if adaptation needed
        needs_adaptation = relative_deviation > self.adaptation_threshold
        
        monitoring_result = {
            'plan_id': plan.plan_id,
            'current_state': current_state.copy(),
            'expected_state': expected_state.copy(),
            'deviation': deviation,
            'relative_deviation': relative_deviation,
            'needs_adaptation': needs_adaptation,
            'timestamp': time.time()
        }
        
        # Store history
        if plan.plan_id not in self.monitoring_history:
            self.monitoring_history[plan.plan_id] = []
        self.monitoring_history[plan.plan_id].append(monitoring_result)
        
        return monitoring_result
    
    def adapt_plan(self,
                  plan: Plan,
                  current_state: np.ndarray,
                  goal: StrategicGoal) -> Plan:
        """
        Adapt plan based on current state
        
        Returns:
            Adapted plan
        """
        # Compute remaining path
        remaining_path = goal.target_state - current_state
        
        # Recompute actions and states
        num_remaining_actions = len(plan.actions)
        adapted_actions = list(range(num_remaining_actions))
        adapted_states = []
        
        for i in range(num_remaining_actions):
            alpha = (i + 1) / num_remaining_actions
            state = current_state + alpha * remaining_path
            adapted_states.append(state)
        
        # Create adapted plan
        adapted_plan = Plan(
            plan_id=plan.plan_id,
            goal_id=plan.goal_id,
            actions=adapted_actions,
            expected_states=adapted_states,
            confidence=plan.confidence * 0.9,  # Slightly lower confidence after adaptation
            created_time=time.time()
        )
        
        # Record adaptation
        self.adaptations.append({
            'original_plan_id': plan.plan_id,
            'adapted_plan_id': adapted_plan.plan_id,
            'reason': 'deviation_detected',
            'timestamp': time.time()
        })
        
        return adapted_plan
    
    def detect_failure(self, plan: Plan, monitoring_results: List[Dict]) -> bool:
        """Detect if plan has failed"""
        if not monitoring_results:
            return False
        
        # Check for consistent large deviations
        large_deviations = sum(1 for r in monitoring_results 
                              if r['relative_deviation'] > 0.5)
        
        # Plan fails if >50% of steps have large deviations
        failure_threshold = len(monitoring_results) * 0.5
        return large_deviations > failure_threshold


class StrategicPlanningSystem:
    """
    Strategic Planning System Manager
    
    Integrates all strategic planning components
    """
    
    def __init__(self,
                 brain_system=None,
                 goal_setting: Optional[GoalSettingPlanning] = None,
                 world_model: Optional[WorldModelManager] = None,
                 executive_control: Optional[ExecutiveControl] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.multi_horizon_planning = MultiHorizonPlanning()
        self.scenario_planning = ScenarioPlanning()
        self.resource_management = ResourceManagement()
        self.goal_decomposition = StrategicGoalDecomposition()
        self.plan_monitoring = PlanMonitoring()
        
        # Integration with existing systems
        self.goal_setting = goal_setting
        self.world_model = world_model
        self.executive_control = executive_control
        
        # Strategic goals tracking
        self.strategic_goals: Dict[int, StrategicGoal] = {}
        self.next_goal_id = 0
        
        # Statistics
        self.stats = {
            'strategic_goals_created': 0,
            'scenarios_planned': 0,
            'plans_adapted': 0,
            'average_plan_horizon': 0.0
        }
    
    def create_strategic_goal(self,
                             description: str,
                             target_state: np.ndarray,
                             time_horizon: float,
                             priority: float = 1.0) -> StrategicGoal:
        """Create a strategic goal"""
        goal = StrategicGoal(
            goal_id=self.next_goal_id,
            description=description,
            target_state=target_state,
            time_horizon=time_horizon,
            priority=priority,
            strategic_importance=priority,
            created_time=time.time()
        )
        
        self.strategic_goals[self.next_goal_id] = goal
        self.next_goal_id += 1
        self.stats['strategic_goals_created'] += 1
        
        return goal
    
    def plan_for_goal(self, goal: StrategicGoal) -> Dict:
        """
        Create comprehensive plan for strategic goal
        
        Returns:
            Planning results
        """
        # Decompose into tactical steps
        tactical_goals = self.goal_decomposition.decompose_strategic_goal(
            goal, num_tactical_steps=5, goal_hierarchy=self.goal_setting
        )
        
        # Create scenarios
        scenarios = self.scenario_planning.create_scenarios(goal, num_scenarios=3)
        self.stats['scenarios_planned'] += len(scenarios)
        
        # Create multi-horizon plans
        horizon_plans = self.multi_horizon_planning.create_multi_horizon_plan(
            goal, self.world_model
        )
        
        # Select best scenario plan
        best_scenario_plan = self.scenario_planning.select_best_scenario_plan(goal)
        
        # Update statistics
        if horizon_plans:
            avg_horizon = np.mean(list(horizon_plans.keys()))
            total = self.stats['strategic_goals_created']
            self.stats['average_plan_horizon'] = (
                (self.stats['average_plan_horizon'] * (total - 1) + avg_horizon) / total
            )
        
        return {
            'goal_id': goal.goal_id,
            'tactical_goals': tactical_goals,
            'scenarios': scenarios,
            'horizon_plans': horizon_plans,
            'best_scenario_plan': best_scenario_plan
        }
    
    def monitor_and_adapt(self,
                         goal: StrategicGoal,
                         plan: Plan,
                         current_state: np.ndarray) -> Optional[Plan]:
        """Monitor plan and adapt if needed"""
        # Monitor
        monitoring_result = self.plan_monitoring.monitor_plan(plan, current_state)
        
        if monitoring_result['needs_adaptation']:
            # Adapt plan
            adapted_plan = self.plan_monitoring.adapt_plan(plan, current_state, goal)
            self.stats['plans_adapted'] += 1
            return adapted_plan
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get strategic planning statistics"""
        return self.stats.copy()

