#!/usr/bin/env python3
"""
Global Workspace
Implements global workspace theory, information integration, consciousness access, and broadcast mechanisms
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from collections import deque

@dataclass
class WorkspaceContent:
    """Represents content in global workspace"""
    content_id: int
    source: str  # 'sensory', 'memory', 'reasoning', etc.
    content: Dict[str, float]  # information -> value
    activation: float = 1.0
    timestamp: float = 0.0
    broadcast_count: int = 0

class GlobalWorkspace:
    """
    Global Workspace
    
    Central information integration system
    Broadcasts information globally
    """
    
    def __init__(self,
                 capacity: int = 10,
                 activation_threshold: float = 0.5):
        self.capacity = capacity
        self.activation_threshold = activation_threshold
        
        self.workspace_contents: deque = deque(maxlen=capacity)
        self.broadcast_history: List[Tuple[int, float]] = []  # (content_id, timestamp)
        self.next_content_id = 0
    
    def add_content(self,
                   source: str,
                   content: Dict[str, float],
                   activation: float = 1.0) -> Optional[WorkspaceContent]:
        """Add content to global workspace"""
        if activation < self.activation_threshold:
            return None
        
        workspace_content = WorkspaceContent(
            content_id=self.next_content_id,
            source=source,
            content=content.copy(),
            activation=activation,
            timestamp=time.time()
        )
        
        self.workspace_contents.append(workspace_content)
        self.next_content_id += 1
        
        return workspace_content
    
    def get_active_contents(self) -> List[WorkspaceContent]:
        """Get all active contents above threshold"""
        return [
            content for content in self.workspace_contents
            if content.activation >= self.activation_threshold
        ]
    
    def broadcast_content(self, content_id: int) -> bool:
        """Broadcast content globally"""
        content = None
        for c in self.workspace_contents:
            if c.content_id == content_id:
                content = c
                break
        
        if content is None:
            return False
        
        content.broadcast_count += 1
        self.broadcast_history.append((content_id, time.time()))
        return True
    
    def integrate_contents(self) -> Dict[str, float]:
        """Integrate all active contents"""
        active_contents = self.get_active_contents()
        
        if not active_contents:
            return {}
        
        integrated = {}
        
        for content in active_contents:
            # Weight by activation
            weight = content.activation
            for key, value in content.content.items():
                if key not in integrated:
                    integrated[key] = 0.0
                integrated[key] += weight * value
        
        # Normalize
        total_weight = sum(c.activation for c in active_contents)
        if total_weight > 0:
            for key in integrated:
                integrated[key] /= total_weight
        
        return integrated
    
    def get_workspace_summary(self) -> Dict:
        """Get summary of workspace"""
        return {
            'total_contents': len(self.workspace_contents),
            'active_contents': len(self.get_active_contents()),
            'broadcasts': len(self.broadcast_history),
            'capacity_usage': len(self.workspace_contents) / self.capacity
        }

class InformationIntegration:
    """
    Information Integration
    
    Integrates information from multiple sources
    Creates unified representation
    """
    
    def __init__(self):
        self.integration_history: List[Tuple[Dict[str, float], float]] = []  # (integrated, timestamp)
        self.source_weights: Dict[str, float] = {}  # source -> weight
    
    def set_source_weight(self,
                         source: str,
                         weight: float):
        """Set weight for information source"""
        self.source_weights[source] = weight
    
    def integrate(self,
                 sources: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Integrate information from multiple sources
        
        Args:
            sources: {source_name -> {info -> value}}
        
        Returns:
            Integrated information
        """
        integrated = {}
        total_weight = 0.0
        
        for source, content in sources.items():
            weight = self.source_weights.get(source, 1.0)
            total_weight += weight
            
            for key, value in content.items():
                if key not in integrated:
                    integrated[key] = 0.0
                integrated[key] += weight * value
        
        # Normalize
        if total_weight > 0:
            for key in integrated:
                integrated[key] /= total_weight
        
        # Record integration
        self.integration_history.append((integrated.copy(), time.time()))
        
        # Limit history
        if len(self.integration_history) > 100:
            self.integration_history = self.integration_history[-100:]
        
        return integrated
    
    def get_integration_summary(self) -> Dict:
        """Get integration summary"""
        return {
            'integrations': len(self.integration_history),
            'source_weights': len(self.source_weights)
        }

class ConsciousnessAccess:
    """
    Consciousness Access
    
    Manages access to conscious awareness
    Controls what becomes conscious
    """
    
    def __init__(self,
                 access_threshold: float = 0.6):
        self.access_threshold = access_threshold
        
        self.conscious_contents: List[WorkspaceContent] = []
        self.access_history: List[Tuple[int, float]] = []  # (content_id, timestamp)
    
    def grant_access(self,
                    content: WorkspaceContent) -> bool:
        """Grant conscious access to content"""
        if content.activation < self.access_threshold:
            return False
        
        # Add to conscious contents
        if content not in self.conscious_contents:
            self.conscious_contents.append(content)
        
        # Record access
        self.access_history.append((content.content_id, time.time()))
        
        return True
    
    def revoke_access(self, content_id: int):
        """Revoke conscious access"""
        self.conscious_contents = [
            c for c in self.conscious_contents
            if c.content_id != content_id
        ]
    
    def get_conscious_contents(self) -> List[WorkspaceContent]:
        """Get all conscious contents"""
        return self.conscious_contents.copy()
    
    def is_conscious(self, content_id: int) -> bool:
        """Check if content is conscious"""
        return any(c.content_id == content_id for c in self.conscious_contents)
    
    def get_access_summary(self) -> Dict:
        """Get access summary"""
        return {
            'conscious_contents': len(self.conscious_contents),
            'access_events': len(self.access_history)
        }

class BroadcastMechanism:
    """
    Broadcast Mechanism
    
    Broadcasts information to all systems
    Ensures global availability
    """
    
    def __init__(self):
        self.broadcast_history: List[Tuple[int, List[str], float]] = []  # (content_id, recipients, timestamp)
        self.subscribers: Dict[str, List] = {}  # system_name -> [callbacks]
    
    def subscribe(self,
                 system_name: str,
                 callback):
        """Subscribe to broadcasts"""
        if system_name not in self.subscribers:
            self.subscribers[system_name] = []
        self.subscribers[system_name].append(callback)
    
    def broadcast(self,
                 content_id: int,
                 content: Dict[str, float],
                 recipients: Optional[List[str]] = None):
        """Broadcast content to subscribers"""
        if recipients is None:
            recipients = list(self.subscribers.keys())
        
        for recipient in recipients:
            if recipient in self.subscribers:
                for callback in self.subscribers[recipient]:
                    try:
                        callback(content_id, content)
                    except Exception as e:
                        print(f"Error in broadcast callback: {e}")
        
        self.broadcast_history.append((content_id, recipients.copy(), time.time()))
    
    def get_broadcast_summary(self) -> Dict:
        """Get broadcast summary"""
        return {
            'subscribers': len(self.subscribers),
            'total_callbacks': sum(len(callbacks) for callbacks in self.subscribers.values()),
            'broadcasts': len(self.broadcast_history)
        }

class GlobalWorkspaceManager:
    """
    Manages all global workspace mechanisms
    """
    
    def __init__(self):
        self.global_workspace = GlobalWorkspace()
        self.information_integration = InformationIntegration()
        self.consciousness_access = ConsciousnessAccess()
        self.broadcast_mechanism = BroadcastMechanism()
    
    def add_to_workspace(self,
                        source: str,
                        content: Dict[str, float],
                        activation: float = 1.0) -> Optional[WorkspaceContent]:
        """Add content to workspace"""
        workspace_content = self.global_workspace.add_content(source, content, activation)
        
        if workspace_content:
            # Try to grant conscious access
            self.consciousness_access.grant_access(workspace_content)
        
        return workspace_content
    
    def integrate_sources(self,
                         sources: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Integrate information from multiple sources"""
        return self.information_integration.integrate(sources)
    
    def broadcast(self,
                 content_id: int,
                 content: Dict[str, float],
                 recipients: Optional[List[str]] = None):
        """Broadcast content"""
        self.broadcast_mechanism.broadcast(content_id, content, recipients)
        self.global_workspace.broadcast_content(content_id)
    
    def get_statistics(self) -> Dict:
        """Get statistics about global workspace"""
        workspace_stats = self.global_workspace.get_workspace_summary()
        integration_stats = self.information_integration.get_integration_summary()
        access_stats = self.consciousness_access.get_access_summary()
        broadcast_stats = self.broadcast_mechanism.get_broadcast_summary()
        
        return {
            **workspace_stats,
            **integration_stats,
            **access_stats,
            **broadcast_stats
        }

