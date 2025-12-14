#!/usr/bin/env python3
"""
Metacognition
Implements thinking about thinking, confidence monitoring, strategy selection, and self-regulation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from collections import defaultdict

@dataclass
class MetacognitiveState:
    """Represents metacognitive state"""
    state_id: int
    cognitive_process: str
    confidence: float = 0.5
    monitoring: Dict[str, float] = field(default_factory=dict)
    last_updated: float = 0.0

@dataclass
class Strategy:
    """Represents a cognitive strategy"""
    strategy_id: int
    name: str
    description: str
    success_rate: float = 0.5
    usage_count: int = 0
    last_used: float = 0.0

class MetacognitiveMonitoring:
    """
    Metacognitive Monitoring
    
    Monitors cognitive processes
    Tracks confidence and performance
    """
    
    def __init__(self):
        self.monitored_processes: Dict[str, MetacognitiveState] = {}
        self.monitoring_history: Dict[str, List[Tuple[float, float]]] = {}  # process -> [(confidence, timestamp)]
        self.next_state_id = 0
    
    def monitor_process(self,
                       process_name: str,
                       confidence: float,
                       additional_info: Optional[Dict[str, float]] = None):
        """Monitor a cognitive process"""
        if process_name not in self.monitored_processes:
            state = MetacognitiveState(
                state_id=self.next_state_id,
                cognitive_process=process_name,
                confidence=confidence,
                monitoring=additional_info.copy() if additional_info else {},
                last_updated=time.time()
            )
            self.monitored_processes[process_name] = state
            self.next_state_id += 1
        else:
            state = self.monitored_processes[process_name]
            state.confidence = confidence
            if additional_info:
                state.monitoring.update(additional_info)
            state.last_updated = time.time()
        
        # Update history
        if process_name not in self.monitoring_history:
            self.monitoring_history[process_name] = []
        
        self.monitoring_history[process_name].append((confidence, time.time()))
        
        # Limit history
        if len(self.monitoring_history[process_name]) > 100:
            self.monitoring_history[process_name] = self.monitoring_history[process_name][-100:]
    
    def get_confidence(self, process_name: str) -> float:
        """Get confidence for a process"""
        if process_name in self.monitored_processes:
            return self.monitored_processes[process_name].confidence
        return 0.5  # Default
    
    def get_confidence_trend(self, process_name: str) -> float:
        """Get confidence trend"""
        if process_name not in self.monitoring_history or len(self.monitoring_history[process_name]) < 2:
            return 0.0
        
        history = self.monitoring_history[process_name]
        recent = history[-1][0]
        previous = history[-2][0]
        
        return recent - previous
    
    def get_monitoring_summary(self) -> Dict[str, float]:
        """Get summary of monitored processes"""
        summary = {}
        for process_name, state in self.monitored_processes.items():
            summary[process_name] = state.confidence
        return summary

class StrategySelection:
    """
    Strategy Selection
    
    Selects cognitive strategies
    Adapts based on performance
    """
    
    def __init__(self):
        self.strategies: Dict[str, Strategy] = {}
        self.next_strategy_id = 0
    
    def register_strategy(self,
                         name: str,
                         description: str) -> Strategy:
        """Register a cognitive strategy"""
        if name not in self.strategies:
            strategy = Strategy(
                strategy_id=self.next_strategy_id,
                name=name,
                description=description,
                success_rate=0.5
            )
            self.strategies[name] = strategy
            self.next_strategy_id += 1
        
        return self.strategies[name]
    
    def select_strategy(self,
                       context: Optional[Dict[str, float]] = None) -> Optional[Strategy]:
        """Select best strategy for context"""
        if not self.strategies:
            return None
        
        # Select strategy with highest success rate
        best_strategy = max(self.strategies.values(), key=lambda s: s.success_rate)
        return best_strategy
    
    def update_strategy_performance(self,
                                   strategy_name: str,
                                   success: bool):
        """Update strategy performance"""
        if strategy_name not in self.strategies:
            return
        
        strategy = self.strategies[strategy_name]
        strategy.usage_count += 1
        strategy.last_used = time.time()
        
        # Update success rate (moving average)
        alpha = 0.1
        if success:
            strategy.success_rate = (1 - alpha) * strategy.success_rate + alpha * 1.0
        else:
            strategy.success_rate = (1 - alpha) * strategy.success_rate + alpha * 0.0
    
    def get_strategy_summary(self) -> Dict[str, Dict]:
        """Get summary of strategies"""
        summary = {}
        for name, strategy in self.strategies.items():
            summary[name] = {
                'success_rate': strategy.success_rate,
                'usage_count': strategy.usage_count
            }
        return summary

class SelfRegulation:
    """
    Self-Regulation
    
    Regulates cognitive processes
    Adjusts behavior based on monitoring
    """
    
    def __init__(self):
        self.regulation_rules: Dict[str, Dict] = {}  # process -> regulation_rule
        self.regulation_history: List[Tuple[str, str, float]] = []  # (process, action, timestamp)
    
    def add_regulation_rule(self,
                           process_name: str,
                           condition: Dict[str, float],
                           action: str):
        """Add regulation rule"""
        self.regulation_rules[process_name] = {
            'condition': condition.copy(),
            'action': action
        }
    
    def regulate(self,
                process_name: str,
                current_state: Dict[str, float]) -> Optional[str]:
        """
        Regulate a process based on current state
        
        Returns:
            Regulation action or None
        """
        if process_name not in self.regulation_rules:
            return None
        
        rule = self.regulation_rules[process_name]
        condition = rule['condition']
        
        # Check if condition is met
        condition_met = True
        for key, threshold in condition.items():
            if key not in current_state or current_state[key] < threshold:
                condition_met = False
                break
        
        if condition_met:
            action = rule['action']
            self.regulation_history.append((process_name, action, time.time()))
            return action
        
        return None
    
    def get_regulation_summary(self) -> Dict:
        """Get regulation summary"""
        return {
            'regulated_processes': len(self.regulation_rules),
            'regulation_events': len(self.regulation_history)
        }

class MetacognitiveControl:
    """
    Metacognitive Control
    
    Controls cognitive processes
    Makes decisions about thinking
    """
    
    def __init__(self):
        self.control_decisions: List[Tuple[str, str, float]] = []  # (process, decision, timestamp)
        self.control_policies: Dict[str, Dict] = {}  # process -> policy
    
    def set_control_policy(self,
                          process_name: str,
                          policy: Dict):
        """Set control policy for a process"""
        self.control_policies[process_name] = policy.copy()
    
    def make_control_decision(self,
                            process_name: str,
                            context: Dict[str, float]) -> Optional[str]:
        """
        Make control decision for a process
        
        Returns:
            Control action or None
        """
        if process_name not in self.control_policies:
            return None
        
        policy = self.control_policies[process_name]
        
        # Simple policy: check thresholds
        if 'threshold' in policy:
            threshold = policy['threshold']
            if 'confidence' in context and context['confidence'] < threshold:
                action = policy.get('low_confidence_action', 'stop')
                self.control_decisions.append((process_name, action, time.time()))
                return action
        
        return None
    
    def get_control_summary(self) -> Dict:
        """Get control summary"""
        return {
            'controlled_processes': len(self.control_policies),
            'control_decisions': len(self.control_decisions)
        }

class MetacognitionManager:
    """
    Manages all metacognition mechanisms
    """
    
    def __init__(self):
        self.monitoring = MetacognitiveMonitoring()
        self.strategy_selection = StrategySelection()
        self.self_regulation = SelfRegulation()
        self.control = MetacognitiveControl()
    
    def monitor_cognitive_process(self,
                                 process_name: str,
                                 confidence: float,
                                 additional_info: Optional[Dict[str, float]] = None):
        """Monitor a cognitive process"""
        self.monitoring.monitor_process(process_name, confidence, additional_info)
    
    def select_cognitive_strategy(self, context: Optional[Dict[str, float]] = None) -> Optional[Strategy]:
        """Select cognitive strategy"""
        return self.strategy_selection.select_strategy(context)
    
    def regulate_process(self,
                        process_name: str,
                        current_state: Dict[str, float]) -> Optional[str]:
        """Regulate a process"""
        return self.self_regulation.regulate(process_name, current_state)
    
    def make_control_decision(self,
                             process_name: str,
                             context: Dict[str, float]) -> Optional[str]:
        """Make control decision"""
        return self.control.make_control_decision(process_name, context)
    
    def get_statistics(self) -> Dict:
        """Get statistics about metacognition"""
        monitoring_summary = self.monitoring.get_monitoring_summary()
        strategy_summary = self.strategy_selection.get_strategy_summary()
        regulation_summary = self.self_regulation.get_regulation_summary()
        control_summary = self.control.get_control_summary()
        
        return {
            'monitored_processes': len(monitoring_summary),
            'strategies': len(strategy_summary),
            'regulated_processes': regulation_summary['regulated_processes'],
            'control_decisions': control_summary['control_decisions']
        }

