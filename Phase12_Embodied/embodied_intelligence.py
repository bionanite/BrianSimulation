#!/usr/bin/env python3
"""
Embodied Intelligence - Phase 12.1
Implements sensorimotor learning, action planning, proprioception,
tool manipulation, and spatial reasoning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

# Import dependencies
try:
    from multimodal_integration import MultimodalIntegrationSystem
    from world_models import WorldModelManager
    from goal_setting_planning import GoalSettingPlanning
except ImportError:
    MultimodalIntegrationSystem = None
    WorldModelManager = None
    GoalSettingPlanning = None


@dataclass
class SensorimotorMapping:
    """Represents a sensorimotor mapping"""
    mapping_id: int
    sensor_input: np.ndarray
    motor_output: np.ndarray
    success_rate: float = 0.5
    usage_count: int = 0


@dataclass
class Action:
    """Represents a physical action"""
    action_id: int
    name: str
    parameters: Dict[str, float]
    expected_outcome: np.ndarray
    execution_time: float = 0.0


@dataclass
class BodyState:
    """Represents body position and movement"""
    position: np.ndarray
    orientation: np.ndarray
    velocity: np.ndarray
    timestamp: float = 0.0


class SensorimotorLearning:
    """
    Sensorimotor Learning
    
    Learns sensorimotor mappings
    Connects perception to action
    """
    
    def __init__(self):
        self.mappings: Dict[int, SensorimotorMapping] = {}
        self.next_mapping_id = 0
    
    def learn_mapping(self,
                     sensor_input: np.ndarray,
                     motor_output: np.ndarray,
                     success: bool = True) -> SensorimotorMapping:
        """
        Learn a sensorimotor mapping
        
        Returns:
            Learned mapping
        """
        # Check if similar mapping exists
        existing_mapping = None
        for mapping in self.mappings.values():
            if np.allclose(mapping.sensor_input, sensor_input, atol=0.1):
                existing_mapping = mapping
                break
        
        if existing_mapping:
            # Update existing mapping
            if success:
                existing_mapping.success_rate = (
                    existing_mapping.success_rate * existing_mapping.usage_count + 1.0
                ) / (existing_mapping.usage_count + 1)
            else:
                existing_mapping.success_rate = (
                    existing_mapping.success_rate * existing_mapping.usage_count
                ) / (existing_mapping.usage_count + 1)
            
            existing_mapping.usage_count += 1
            return existing_mapping
        
        # Create new mapping
        mapping = SensorimotorMapping(
            mapping_id=self.next_mapping_id,
            sensor_input=sensor_input.copy(),
            motor_output=motor_output.copy(),
            success_rate=1.0 if success else 0.0,
            usage_count=1
        )
        
        self.mappings[self.next_mapping_id] = mapping
        self.next_mapping_id += 1
        
        return mapping
    
    def get_motor_command(self, sensor_input: np.ndarray) -> Optional[np.ndarray]:
        """
        Get motor command for sensor input
        
        Returns:
            Motor command or None
        """
        # Find best matching mapping
        best_mapping = None
        best_similarity = -1.0
        
        for mapping in self.mappings.values():
            similarity = 1.0 - np.linalg.norm(sensor_input - mapping.sensor_input)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_mapping = mapping
        
        if best_mapping and best_similarity > 0.5:
            return best_mapping.motor_output.copy()
        
        return None


class ActionPlanning:
    """
    Action Planning
    
    Plans physical actions
    Sequences actions to achieve goals
    """
    
    def __init__(self, world_model: Optional[WorldModelManager] = None):
        self.world_model = world_model
        self.actions: Dict[int, Action] = {}
        self.next_action_id = 0
    
    def create_action(self,
                     name: str,
                     parameters: Dict[str, float],
                     expected_outcome: np.ndarray) -> Action:
        """Create a new action"""
        action = Action(
            action_id=self.next_action_id,
            name=name,
            parameters=parameters,
            expected_outcome=expected_outcome,
            execution_time=time.time()
        )
        
        self.actions[self.next_action_id] = action
        self.next_action_id += 1
        
        return action
    
    def plan_action_sequence(self,
                            goal_state: np.ndarray,
                            current_state: np.ndarray) -> List[Action]:
        """
        Plan sequence of actions to reach goal
        
        Returns:
            List of actions
        """
        actions = []
        
        # Compute path to goal
        path = goal_state - current_state
        num_steps = max(1, int(np.linalg.norm(path) / 0.1))
        
        for i in range(num_steps):
            # Interpolate intermediate state
            alpha = (i + 1) / num_steps
            intermediate_state = current_state + alpha * path
            
            # Create action to reach intermediate state
            action = self.create_action(
                name=f"Move to step {i+1}",
                parameters={'direction': alpha},
                expected_outcome=intermediate_state
            )
            actions.append(action)
        
        return actions


class Proprioception:
    """
    Proprioception
    
    Senses body position and movement
    Maintains body state
    """
    
    def __init__(self):
        self.body_state: Optional[BodyState] = None
        self.state_history: List[BodyState] = []
    
    def update_body_state(self,
                         position: np.ndarray,
                         orientation: np.ndarray = None,
                         velocity: np.ndarray = None):
        """Update body state"""
        if orientation is None:
            orientation = np.zeros(3)
        if velocity is None:
            velocity = np.zeros(3)
        
        self.body_state = BodyState(
            position=position.copy(),
            orientation=orientation.copy(),
            velocity=velocity.copy(),
            timestamp=time.time()
        )
        
        self.state_history.append(self.body_state)
        if len(self.state_history) > 100:
            self.state_history.pop(0)
    
    def get_body_state(self) -> Optional[BodyState]:
        """Get current body state"""
        return self.body_state
    
    def sense_movement(self) -> np.ndarray:
        """Sense current movement"""
        if not self.body_state:
            return np.zeros(3)
        
        return self.body_state.velocity.copy()


class ToolManipulation:
    """
    Tool Manipulation
    
    Manipulates physical objects
    Uses tools effectively
    """
    
    def __init__(self):
        self.tool_usage_history: List[Dict] = []
    
    def manipulate_object(self,
                         object_position: np.ndarray,
                         target_position: np.ndarray,
                         tool_type: str = 'gripper') -> Dict:
        """
        Manipulate object with tool
        
        Returns:
            Manipulation result
        """
        # Compute manipulation path
        path = target_position - object_position
        distance = np.linalg.norm(path)
        
        # Simulate manipulation
        success = distance < 1.0  # Simplified success condition
        
        result = {
            'success': success,
            'distance': distance,
            'tool_type': tool_type,
            'timestamp': time.time()
        }
        
        self.tool_usage_history.append(result)
        return result
    
    def learn_tool_usage(self,
                        tool_type: str,
                        success: bool):
        """Learn from tool usage"""
        # Update tool usage patterns (simplified)
        pass


class SpatialReasoning:
    """
    Spatial Reasoning
    
    Reasons about spatial relationships
    Understands spatial configurations
    """
    
    def __init__(self):
        self.spatial_representations: Dict[str, np.ndarray] = {}
    
    def represent_space(self,
                       object_id: str,
                       position: np.ndarray,
                       size: np.ndarray = None):
        """Represent object in space"""
        if size is None:
            size = np.ones(3) * 0.1
        
        # Store spatial representation
        representation = np.concatenate([position, size])
        self.spatial_representations[object_id] = representation
    
    def compute_spatial_relationship(self,
                                   object1_id: str,
                                   object2_id: str) -> Dict:
        """
        Compute spatial relationship between objects
        
        Returns:
            Relationship description
        """
        if object1_id not in self.spatial_representations:
            return {'relationship': 'unknown'}
        
        if object2_id not in self.spatial_representations:
            return {'relationship': 'unknown'}
        
        pos1 = self.spatial_representations[object1_id][:3]
        pos2 = self.spatial_representations[object2_id][:3]
        
        # Compute distance
        distance = np.linalg.norm(pos1 - pos2)
        
        # Determine relationship
        if distance < 0.5:
            relationship = 'near'
        elif distance < 2.0:
            relationship = 'moderate_distance'
        else:
            relationship = 'far'
        
        # Compute direction
        direction = pos2 - pos1
        direction_normalized = direction / (np.linalg.norm(direction) + 1e-10)
        
        return {
            'relationship': relationship,
            'distance': distance,
            'direction': direction_normalized
        }
    
    def plan_spatial_path(self,
                        start_position: np.ndarray,
                        goal_position: np.ndarray,
                        obstacles: List[np.ndarray] = None) -> List[np.ndarray]:
        """
        Plan path through space
        
        Returns:
            List of waypoints
        """
        if obstacles is None:
            obstacles = []
        
        # Simple path planning: straight line if no obstacles
        waypoints = []
        
        if not obstacles:
            # Direct path
            num_waypoints = max(2, int(np.linalg.norm(goal_position - start_position) / 0.5))
            for i in range(num_waypoints + 1):
                alpha = i / num_waypoints
                waypoint = start_position + alpha * (goal_position - start_position)
                waypoints.append(waypoint)
        else:
            # Avoid obstacles (simplified)
            waypoints = [start_position, goal_position]
        
        return waypoints


class EmbodiedIntelligenceSystem:
    """
    Embodied Intelligence System Manager
    
    Integrates all embodied intelligence components
    """
    
    def __init__(self,
                 brain_system=None,
                 multimodal_integration: Optional[MultimodalIntegrationSystem] = None,
                 world_model: Optional[WorldModelManager] = None,
                 goal_setting: Optional[GoalSettingPlanning] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.sensorimotor_learning = SensorimotorLearning()
        self.action_planning = ActionPlanning(world_model)
        self.proprioception = Proprioception()
        self.tool_manipulation = ToolManipulation()
        self.spatial_reasoning = SpatialReasoning()
        
        # Integration with existing systems
        self.multimodal_integration = multimodal_integration
        self.world_model = world_model
        self.goal_setting = goal_setting
        
        # Statistics
        self.stats = {
            'sensorimotor_mappings_learned': 0,
            'actions_planned': 0,
            'body_states_updated': 0,
            'objects_manipulated': 0,
            'spatial_relationships_computed': 0
        }
    
    def learn_sensorimotor_mapping(self,
                                  sensor_input: np.ndarray,
                                  motor_output: np.ndarray,
                                  success: bool = True) -> SensorimotorMapping:
        """Learn sensorimotor mapping"""
        mapping = self.sensorimotor_learning.learn_mapping(
            sensor_input, motor_output, success
        )
        self.stats['sensorimotor_mappings_learned'] += 1
        return mapping
    
    def plan_actions(self,
                    goal_state: np.ndarray,
                    current_state: np.ndarray) -> List[Action]:
        """Plan actions to reach goal"""
        actions = self.action_planning.plan_action_sequence(goal_state, current_state)
        self.stats['actions_planned'] += len(actions)
        return actions
    
    def update_body_state(self,
                         position: np.ndarray,
                         orientation: np.ndarray = None,
                         velocity: np.ndarray = None):
        """Update body state"""
        self.proprioception.update_body_state(position, orientation, velocity)
        self.stats['body_states_updated'] += 1
    
    def manipulate_object(self,
                         object_position: np.ndarray,
                         target_position: np.ndarray) -> Dict:
        """Manipulate object"""
        result = self.tool_manipulation.manipulate_object(
            object_position, target_position
        )
        if result['success']:
            self.stats['objects_manipulated'] += 1
        return result
    
    def compute_spatial_relationship(self,
                                    object1_id: str,
                                    object2_id: str) -> Dict:
        """Compute spatial relationship"""
        relationship = self.spatial_reasoning.compute_spatial_relationship(
            object1_id, object2_id
        )
        self.stats['spatial_relationships_computed'] += 1
        return relationship
    
    def get_statistics(self) -> Dict:
        """Get embodied intelligence statistics"""
        return self.stats.copy()

