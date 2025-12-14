#!/usr/bin/env python3
"""
Test Phase 10: Tool Use & Self-Improvement
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Phase10_ToolUse.tool_use import ToolUseSystem, Tool
from Phase10_ToolUse.self_improvement import SelfImprovementSystem, Architecture
from Phase10_ToolUse.knowledge_synthesis import KnowledgeSynthesisSystem, KnowledgeItem


class TestToolUse(unittest.TestCase):
    """Test tool use components"""
    
    def test_tool_registration(self):
        """Test tool registration"""
        system = ToolUseSystem()
        
        def add_function(x):
            return x + 1
        
        tool = system.register_tool(
            name="AddOne",
            description="Adds one to input",
            function=add_function,
            capabilities={"arithmetic": 0.8}
        )
        
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "AddOne")
    
    def test_tool_selection(self):
        """Test tool selection"""
        system = ToolUseSystem()
        
        def multiply(x):
            return x * 2
        
        system.register_tool("Multiply", "Multiplies by 2", multiply, {"arithmetic": 0.9})
        
        tool, output = system.use_tool_for_task(
            task_requirements={"arithmetic": 0.8},
            input_data=5
        )
        
        self.assertIsNotNone(tool)
        self.assertEqual(output, 10)
    
    def test_tool_composition(self):
        """Test tool composition"""
        system = ToolUseSystem()
        
        def add_one(x):
            return x + 1
        
        def multiply_two(x):
            return x * 2
        
        tool1 = system.register_tool("AddOne", "Adds one", add_one, {"arithmetic": 0.8})
        tool2 = system.register_tool("MultiplyTwo", "Multiplies by two", multiply_two, {"arithmetic": 0.8})
        
        composition = system.create_tool_composition([tool1, tool2])
        self.assertIsNotNone(composition)
        self.assertEqual(len(composition.tools), 2)


class TestSelfImprovement(unittest.TestCase):
    """Test self-improvement components"""
    
    def test_architecture_search(self):
        """Test architecture search"""
        system = SelfImprovementSystem()
        
        def performance_func(arch):
            return 0.5 + np.random.rand() * 0.3
        
        improved = system.improve_architecture(performance_func)
        self.assertIsNotNone(system.current_architecture)
    
    def test_capability_expansion(self):
        """Test capability expansion"""
        system = SelfImprovementSystem()
        
        capability = system.add_new_capability(
            name="NewCapability",
            description="A new capability"
        )
        
        self.assertIsNotNone(capability)
        self.assertEqual(capability.name, "NewCapability")
    
    def test_performance_monitoring(self):
        """Test performance monitoring"""
        system = SelfImprovementSystem()
        
        system.monitor_and_improve("accuracy", 0.85)
        stats = system.get_statistics()
        
        self.assertIn('performance_trends', stats)


class TestKnowledgeSynthesis(unittest.TestCase):
    """Test knowledge synthesis"""
    
    def test_knowledge_addition(self):
        """Test adding knowledge"""
        system = KnowledgeSynthesisSystem()
        
        knowledge = system.add_knowledge(
            content="Python is a programming language",
            domain="computer_science",
            confidence=0.9
        )
        
        self.assertIsNotNone(knowledge)
        self.assertEqual(knowledge.domain, "computer_science")
    
    def test_knowledge_synthesis(self):
        """Test knowledge synthesis"""
        system = KnowledgeSynthesisSystem()
        
        k1 = system.add_knowledge("Knowledge 1", "domain1", 0.8)
        k2 = system.add_knowledge("Knowledge 2", "domain1", 0.7)
        
        synthesized = system.synthesize_knowledge([k1.knowledge_id, k2.knowledge_id])
        self.assertIsNotNone(synthesized)
    
    def test_contradiction_resolution(self):
        """Test contradiction resolution"""
        system = KnowledgeSynthesisSystem()
        
        k1 = system.add_knowledge("X is true", "domain1", 0.9)
        k2 = system.add_knowledge("X is false", "domain1", 0.8)
        
        resolutions = system.resolve_all_contradictions()
        self.assertGreaterEqual(len(resolutions), 0)


if __name__ == '__main__':
    unittest.main()

