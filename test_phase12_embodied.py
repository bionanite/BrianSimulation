#!/usr/bin/env python3
"""
Test Phase 12: Embodied Intelligence & Temporal Reasoning
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Phase12_Embodied.embodied_intelligence import EmbodiedIntelligenceSystem
from Phase12_Embodied.temporal_reasoning import TemporalReasoningSystem
from Phase12_Embodied.long_term_memory import LongTermMemorySystem


class TestEmbodiedIntelligence(unittest.TestCase):
    """Test embodied intelligence"""
    
    def test_sensorimotor_learning(self):
        """Test sensorimotor learning"""
        system = EmbodiedIntelligenceSystem()
        
        sensor_input = np.array([0.5, 0.3, 0.7])
        motor_output = np.array([0.2, 0.4, 0.6])
        
        mapping = system.learn_sensorimotor_mapping(
            sensor_input, motor_output, success=True
        )
        
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping.success_rate, 1.0)
    
    def test_action_planning(self):
        """Test action planning"""
        system = EmbodiedIntelligenceSystem()
        
        goal_state = np.array([1.0, 1.0, 1.0])
        current_state = np.array([0.0, 0.0, 0.0])
        
        actions = system.plan_actions(goal_state, current_state)
        self.assertGreater(len(actions), 0)
    
    def test_spatial_reasoning(self):
        """Test spatial reasoning"""
        system = EmbodiedIntelligenceSystem()
        
        system.spatial_reasoning.represent_space("obj1", np.array([0.0, 0.0, 0.0]))
        system.spatial_reasoning.represent_space("obj2", np.array([1.0, 1.0, 1.0]))
        
        relationship = system.compute_spatial_relationship("obj1", "obj2")
        self.assertIn('relationship', relationship)
        self.assertIn('distance', relationship)


class TestTemporalReasoning(unittest.TestCase):
    """Test temporal reasoning"""
    
    def test_episodic_memory(self):
        """Test episodic memory"""
        system = TemporalReasoningSystem()
        
        memory = system.store_episode(
            event_description="Attended conference",
            location=np.array([10.0, 20.0, 0.0]),
            participants=["Alice", "Bob"]
        )
        
        self.assertIsNotNone(memory)
        self.assertEqual(memory.event_description, "Attended conference")
    
    def test_temporal_sequences(self):
        """Test temporal sequences"""
        system = TemporalReasoningSystem()
        
        events = [1, 2, 3]
        timestamps = [100.0, 200.0, 300.0]
        
        sequence = system.learn_sequence(events, timestamps)
        self.assertIsNotNone(sequence)
        self.assertEqual(len(sequence.events), 3)
    
    def test_time_perception(self):
        """Test time perception"""
        system = TemporalReasoningSystem()
        
        perception = system.perceive_time(
            perceived_duration=10.0,
            actual_duration=8.0
        )
        
        self.assertIsNotNone(perception)
        self.assertGreater(perception.time_dilation_factor, 1.0)


class TestLongTermMemory(unittest.TestCase):
    """Test long-term memory"""
    
    def test_life_event_storage(self):
        """Test storing life events"""
        system = LongTermMemorySystem()
        
        memory = system.store_life_event(
            event_description="Graduated from university",
            location="Boston",
            importance=0.9
        )
        
        self.assertIsNotNone(memory)
        self.assertEqual(memory.location, "Boston")
    
    def test_semantic_knowledge(self):
        """Test semantic knowledge storage"""
        system = LongTermMemorySystem()
        
        entry = system.store_semantic_knowledge(
            concept="Python",
            definition="A programming language",
            properties={"type": "language", "paradigm": "multi-paradigm"}
        )
        
        self.assertIsNotNone(entry)
        self.assertEqual(entry.concept, "Python")
    
    def test_memory_retrieval(self):
        """Test memory retrieval"""
        system = LongTermMemorySystem()
        
        system.store_life_event("Event 1", "Location1", 0.8)
        system.store_semantic_knowledge("Concept1", "Definition1")
        
        results = system.retrieve_memories("Event")
        self.assertGreater(results['total_results'], 0)


if __name__ == '__main__':
    unittest.main()

