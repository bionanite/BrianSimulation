#!/usr/bin/env python3
"""
Self-Model
Implements self-representation, self-awareness, body schema, and identity maintenance
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from collections import defaultdict

@dataclass
class SelfRepresentation:
    """Represents self-knowledge"""
    representation_id: int
    aspect: str  # 'capabilities', 'preferences', 'goals', 'history'
    content: Dict[str, float]  # attribute -> value
    confidence: float = 1.0
    last_updated: float = 0.0

@dataclass
class BodySchema:
    """Represents body schema"""
    schema_id: int
    body_part: str
    position: np.ndarray
    capabilities: List[str]
    constraints: Dict[str, float]  # constraint_name -> limit

class SelfRepresentationSystem:
    """
    Self-Representation System
    
    Maintains representation of self
    Tracks capabilities, preferences, goals
    """
    
    def __init__(self):
        self.self_representations: Dict[str, SelfRepresentation] = {}
        self.next_representation_id = 0
    
    def create_representation(self,
                            aspect: str,
                            content: Dict[str, float]) -> SelfRepresentation:
        """Create or update self-representation"""
        if aspect not in self.self_representations:
            representation = SelfRepresentation(
                representation_id=self.next_representation_id,
                aspect=aspect,
                content=content.copy(),
                confidence=1.0,
                last_updated=time.time()
            )
            self.self_representations[aspect] = representation
            self.next_representation_id += 1
        else:
            representation = self.self_representations[aspect]
            # Update content
            representation.content.update(content)
            representation.last_updated = time.time()
        
        return self.self_representations[aspect]
    
    def get_capabilities(self) -> Dict[str, float]:
        """Get self-perceived capabilities"""
        if 'capabilities' in self.self_representations:
            return self.self_representations['capabilities'].content
        return {}
    
    def get_preferences(self) -> Dict[str, float]:
        """Get self-perceived preferences"""
        if 'preferences' in self.self_representations:
            return self.self_representations['preferences'].content
        return {}
    
    def update_capability(self,
                        capability: str,
                        level: float):
        """Update perceived capability level"""
        if 'capabilities' not in self.self_representations:
            self.create_representation('capabilities', {})
        
        self.self_representations['capabilities'].content[capability] = level
        self.self_representations['capabilities'].last_updated = time.time()
    
    def update_preference(self,
                         preference: str,
                         strength: float):
        """Update preference strength"""
        if 'preferences' not in self.self_representations:
            self.create_representation('preferences', {})
        
        self.self_representations['preferences'].content[preference] = strength
        self.self_representations['preferences'].last_updated = time.time()
    
    def get_self_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of self-representations"""
        summary = {}
        for aspect, representation in self.self_representations.items():
            summary[aspect] = representation.content.copy()
        return summary

class SelfAwareness:
    """
    Self-Awareness
    
    Maintains awareness of self
    Monitors internal states
    """
    
    def __init__(self):
        self.internal_states: Dict[str, float] = {}
        self.state_history: Dict[str, List[Tuple[float, float]]] = {}  # state -> [(value, timestamp)]
        self.awareness_level: float = 1.0
    
    def update_internal_state(self,
                            state_name: str,
                            value: float):
        """Update internal state"""
        self.internal_states[state_name] = value
        
        # Update history
        if state_name not in self.state_history:
            self.state_history[state_name] = []
        
        self.state_history[state_name].append((value, time.time()))
        
        # Limit history
        if len(self.state_history[state_name]) > 100:
            self.state_history[state_name] = self.state_history[state_name][-100:]
    
    def get_internal_state(self, state_name: str) -> Optional[float]:
        """Get current internal state"""
        return self.internal_states.get(state_name)
    
    def get_state_trend(self, state_name: str) -> float:
        """Get trend of a state (positive = increasing, negative = decreasing)"""
        if state_name not in self.state_history or len(self.state_history[state_name]) < 2:
            return 0.0
        
        history = self.state_history[state_name]
        recent_value = history[-1][0]
        previous_value = history[-2][0]
        
        return recent_value - previous_value
    
    def compute_awareness_level(self) -> float:
        """Compute current awareness level"""
        # Awareness increases with number of tracked states
        num_states = len(self.internal_states)
        self.awareness_level = min(1.0, num_states / 10.0)
        return self.awareness_level
    
    def get_self_awareness_summary(self) -> Dict[str, float]:
        """Get summary of self-awareness"""
        return {
            'awareness_level': self.compute_awareness_level(),
            'tracked_states': len(self.internal_states),
            'state_history_length': sum(len(h) for h in self.state_history.values())
        }

class BodySchemaSystem:
    """
    Body Schema System
    
    Maintains body schema
    Tracks body parts and capabilities
    """
    
    def __init__(self):
        self.body_schemas: Dict[str, BodySchema] = {}
        self.next_schema_id = 0
    
    def create_body_schema(self,
                          body_part: str,
                          position: np.ndarray,
                          capabilities: List[str],
                          constraints: Optional[Dict[str, float]] = None) -> BodySchema:
        """Create or update body schema"""
        if body_part not in self.body_schemas:
            schema = BodySchema(
                schema_id=self.next_schema_id,
                body_part=body_part,
                position=position.copy(),
                capabilities=capabilities.copy(),
                constraints=constraints.copy() if constraints else {}
            )
            self.body_schemas[body_part] = schema
            self.next_schema_id += 1
        else:
            schema = self.body_schemas[body_part]
            schema.position = position.copy()
            schema.capabilities = capabilities.copy()
            if constraints:
                schema.constraints.update(constraints)
        
        return self.body_schemas[body_part]
    
    def update_position(self,
                       body_part: str,
                       new_position: np.ndarray):
        """Update position of body part"""
        if body_part in self.body_schemas:
            self.body_schemas[body_part].position = new_position.copy()
    
    def get_capabilities(self, body_part: str) -> List[str]:
        """Get capabilities of a body part"""
        if body_part in self.body_schemas:
            return self.body_schemas[body_part].capabilities.copy()
        return []
    
    def check_constraint(self,
                        body_part: str,
                        action: str) -> bool:
        """Check if action is within constraints"""
        if body_part not in self.body_schemas:
            return False
        
        schema = self.body_schemas[body_part]
        
        # Check if action is in capabilities
        if action not in schema.capabilities:
            return False
        
        # Check constraints (simplified)
        return True
    
    def get_body_summary(self) -> Dict[str, Dict]:
        """Get summary of body schema"""
        summary = {}
        for body_part, schema in self.body_schemas.items():
            summary[body_part] = {
                'position': schema.position.tolist(),
                'capabilities': schema.capabilities.copy(),
                'constraints': schema.constraints.copy()
            }
        return summary

class IdentityMaintenance:
    """
    Identity Maintenance
    
    Maintains sense of identity
    Tracks continuity over time
    """
    
    def __init__(self):
        self.identity_markers: Dict[str, float] = {}  # marker -> strength
        self.identity_history: List[Tuple[Dict[str, float], float]] = []  # (markers, timestamp)
        self.continuity_score: float = 1.0
    
    def add_identity_marker(self,
                          marker: str,
                          strength: float = 1.0):
        """Add an identity marker"""
        self.identity_markers[marker] = strength
        
        # Record in history
        self.identity_history.append((self.identity_markers.copy(), time.time()))
        
        # Limit history
        if len(self.identity_history) > 100:
            self.identity_history = self.identity_history[-100:]
    
    def update_identity_marker(self,
                              marker: str,
                              new_strength: float):
        """Update identity marker"""
        if marker in self.identity_markers:
            self.identity_markers[marker] = new_strength
            self.identity_history.append((self.identity_markers.copy(), time.time()))
    
    def compute_continuity(self) -> float:
        """Compute identity continuity score"""
        if len(self.identity_history) < 2:
            return 1.0
        
        # Compare recent identity with past identity
        recent_markers = self.identity_history[-1][0]
        previous_markers = self.identity_history[-2][0]
        
        # Compute overlap
        common_markers = set(recent_markers.keys()) & set(previous_markers.keys())
        if not common_markers:
            return 0.0
        
        # Compute similarity
        similarities = []
        for marker in common_markers:
            recent_val = recent_markers[marker]
            previous_val = previous_markers[marker]
            similarity = 1.0 - abs(recent_val - previous_val)
            similarities.append(similarity)
        
        self.continuity_score = np.mean(similarities) if similarities else 0.0
        return self.continuity_score
    
    def get_identity_summary(self) -> Dict:
        """Get identity summary"""
        return {
            'identity_markers': len(self.identity_markers),
            'continuity_score': self.compute_continuity(),
            'history_length': len(self.identity_history)
        }

class SelfModelManager:
    """
    Manages all self-model mechanisms
    """
    
    def __init__(self):
        self.self_representation = SelfRepresentationSystem()
        self.self_awareness = SelfAwareness()
        self.body_schema = BodySchemaSystem()
        self.identity_maintenance = IdentityMaintenance()
    
    def update_self_representation(self,
                                  aspect: str,
                                  content: Dict[str, float]):
        """Update self-representation"""
        return self.self_representation.create_representation(aspect, content)
    
    def update_internal_state(self,
                            state_name: str,
                            value: float):
        """Update internal state"""
        self.self_awareness.update_internal_state(state_name, value)
    
    def create_body_part(self,
                        body_part: str,
                        position: np.ndarray,
                        capabilities: List[str],
                        constraints: Optional[Dict[str, float]] = None):
        """Create body part schema"""
        return self.body_schema.create_body_schema(body_part, position, capabilities, constraints)
    
    def add_identity_marker(self,
                          marker: str,
                          strength: float = 1.0):
        """Add identity marker"""
        self.identity_maintenance.add_identity_marker(marker, strength)
    
    def get_statistics(self) -> Dict:
        """Get statistics about self-model"""
        return {
            'self_aspects': len(self.self_representation.self_representations),
            'internal_states': len(self.self_awareness.internal_states),
            'awareness_level': self.self_awareness.compute_awareness_level(),
            'body_parts': len(self.body_schema.body_schemas),
            'identity_markers': len(self.identity_maintenance.identity_markers),
            'continuity_score': self.identity_maintenance.compute_continuity()
        }

