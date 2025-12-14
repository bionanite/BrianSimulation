#!/usr/bin/env python3
"""
Test Phase 9: Strategic Planning & Multi-Agent Systems
"""

import unittest
import numpy as np
import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Phase9_StrategicPlanning.strategic_planning import (
    StrategicPlanningSystem, StrategicGoal, MultiHorizonPlanning,
    ScenarioPlanning, ResourceManagement
)
from Phase9_StrategicPlanning.multi_agent_coordination import (
    MultiAgentCoordinationSystem, Agent, Task
)
from Phase9_StrategicPlanning.agent_strategies import (
    AgentStrategiesSystem, GameTheory, CooperationMechanisms
)


class TestStrategicPlanning(unittest.TestCase):
    """Test strategic planning components"""
    
    def test_strategic_goal_creation(self):
        """Test creating strategic goals"""
        system = StrategicPlanningSystem()
        goal = system.create_strategic_goal(
            description="Achieve market leadership",
            target_state=np.array([1.0, 0.8, 0.9]),
            time_horizon=365.0,
            priority=0.9
        )
        self.assertIsNotNone(goal)
        self.assertEqual(goal.description, "Achieve market leadership")
        self.assertEqual(goal.time_horizon, 365.0)
    
    def test_multi_horizon_planning(self):
        """Test multi-horizon planning"""
        planning = MultiHorizonPlanning()
        goal = StrategicGoal(
            goal_id=0,
            description="Test goal",
            target_state=np.array([1.0]),
            time_horizon=365.0
        )
        plans = planning.create_multi_horizon_plan(goal)
        self.assertGreater(len(plans), 0)
        self.assertIn(7.0, plans)  # Short-term horizon
    
    def test_scenario_planning(self):
        """Test scenario planning"""
        scenario_planning = ScenarioPlanning()
        goal = StrategicGoal(
            goal_id=0,
            description="Test goal",
            target_state=np.array([1.0]),
            time_horizon=365.0
        )
        scenarios = scenario_planning.create_scenarios(goal, num_scenarios=3)
        self.assertEqual(len(scenarios), 3)
        self.assertTrue(any('optimistic' in s.name.lower() for s in scenarios))
    
    def test_resource_management(self):
        """Test resource management"""
        rm = ResourceManagement()
        resource = rm.create_resource(
            name="Budget",
            initial_amount=1000.0,
            max_amount=10000.0,
            replenishment_rate=100.0
        )
        self.assertIsNotNone(resource)
        self.assertEqual(resource.current_amount, 1000.0)
        
        # Test allocation
        plan_id = 1
        success = rm.allocate_resource(plan_id, resource.resource_id, 500.0)
        self.assertTrue(success)
        self.assertEqual(resource.current_amount, 500.0)


class TestMultiAgentCoordination(unittest.TestCase):
    """Test multi-agent coordination"""
    
    def test_agent_creation(self):
        """Test creating agents"""
        system = MultiAgentCoordinationSystem()
        agent = system.create_agent(
            name="Agent1",
            capabilities={"planning": 0.8, "execution": 0.7}
        )
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "Agent1")
        self.assertIn("planning", agent.capabilities)
    
    def test_task_allocation(self):
        """Test task allocation"""
        system = MultiAgentCoordinationSystem()
        
        # Create agents
        agent1 = system.create_agent("Agent1", {"planning": 0.9})
        agent2 = system.create_agent("Agent2", {"execution": 0.9})
        
        # Create task
        task = system.create_task(
            description="Plan project",
            requirements={"planning": 0.8}
        )
        
        # Allocate
        success = system.coordinate_task_execution(task.task_id)
        self.assertTrue(success)
        self.assertIsNotNone(task.assigned_agent)
    
    def test_consensus_building(self):
        """Test consensus building"""
        system = MultiAgentCoordinationSystem()
        
        agent1 = system.create_agent("Agent1", {})
        agent2 = system.create_agent("Agent2", {})
        
        consensus, ratio = system.build_consensus(
            "proposal1",
            "Implement feature X",
            [agent1.agent_id, agent2.agent_id]
        )
        self.assertIsInstance(consensus, bool)
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)


class TestAgentStrategies(unittest.TestCase):
    """Test agent strategies"""
    
    def test_game_theory(self):
        """Test game theory"""
        system = AgentStrategiesSystem()
        
        players = [1, 2]
        actions = {1: ["cooperate", "defect"], 2: ["cooperate", "defect"]}
        payoffs = {
            ("cooperate", "cooperate"): {1: 3.0, 2: 3.0},
            ("cooperate", "defect"): {1: 0.0, 2: 5.0},
            ("defect", "cooperate"): {1: 5.0, 2: 0.0},
            ("defect", "defect"): {1: 1.0, 2: 1.0}
        }
        
        result = system.analyze_game(players, actions, payoffs)
        self.assertIsNotNone(result)
        self.assertIn('game_id', result)
    
    def test_cooperation(self):
        """Test cooperation mechanisms"""
        system = AgentStrategiesSystem()
        
        decision = system.decide_cooperation(
            agent1_id=1,
            agent2_id=2,
            context={'cooperation_benefit': 0.7}
        )
        self.assertIsInstance(decision, bool)
    
    def test_trust_modeling(self):
        """Test trust modeling"""
        system = AgentStrategiesSystem()
        
        system.update_trust(
            from_agent_id=1,
            to_agent_id=2,
            interaction_outcome={'success': True, 'cooperation': True}
        )
        
        trust = system.trust_modeling.get_trust(1, 2)
        self.assertGreater(trust, 0.0)


if __name__ == '__main__':
    unittest.main()

