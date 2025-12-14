#!/usr/bin/env python3
"""
Tool Use & Tool Creation - Phase 10.1
Implements tool discovery, tool selection, tool composition,
tool creation, and tool optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import time
from collections import defaultdict

# Import dependencies
try:
    from goal_setting_planning import GoalSettingPlanning
    from hierarchical_learning import HierarchicalFeatureLearning
    from world_models import WorldModelManager
except ImportError:
    GoalSettingPlanning = None
    HierarchicalFeatureLearning = None
    WorldModelManager = None


@dataclass
class Tool:
    """Represents a tool"""
    tool_id: int
    name: str
    description: str
    function: Callable
    capabilities: Dict[str, float]  # capability -> strength
    efficiency: float = 1.0
    usage_count: int = 0
    success_rate: float = 0.5
    created_time: float = 0.0


@dataclass
class ToolComposition:
    """Represents a composition of multiple tools"""
    composition_id: int
    name: str
    tools: List[int]  # Tool IDs
    execution_order: List[int]
    efficiency: float = 1.0
    usage_count: int = 0


class ToolDiscovery:
    """
    Tool Discovery
    
    Discovers available tools
    Maintains tool registry
    """
    
    def __init__(self):
        self.tools: Dict[int, Tool] = {}
        self.tool_index: Dict[str, List[int]] = defaultdict(list)  # capability -> tool_ids
        self.next_tool_id = 0
    
    def register_tool(self,
                     name: str,
                     description: str,
                     function: Callable,
                     capabilities: Dict[str, float]) -> Tool:
        """Register a new tool"""
        tool = Tool(
            tool_id=self.next_tool_id,
            name=name,
            description=description,
            function=function,
            capabilities=capabilities,
            created_time=time.time()
        )
        
        self.tools[self.next_tool_id] = tool
        
        # Index by capabilities
        for capability in capabilities.keys():
            self.tool_index[capability].append(self.next_tool_id)
        
        self.next_tool_id += 1
        return tool
    
    def discover_tools_by_capability(self, capability: str) -> List[Tool]:
        """Discover tools with a specific capability"""
        tool_ids = self.tool_index.get(capability, [])
        return [self.tools[tid] for tid in tool_ids if tid in self.tools]
    
    def search_tools(self, query: str) -> List[Tool]:
        """Search tools by name or description"""
        query_lower = query.lower()
        matching_tools = []
        
        for tool in self.tools.values():
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower()):
                matching_tools.append(tool)
        
        return matching_tools


class ToolSelection:
    """
    Tool Selection
    
    Chooses appropriate tools for tasks
    Matches tools to goals
    """
    
    def __init__(self,
                 selection_strategy: str = 'capability_match'):
        self.selection_strategy = selection_strategy
    
    def select_tool(self,
                   task_requirements: Dict[str, float],
                   available_tools: List[Tool]) -> Optional[Tool]:
        """
        Select best tool for task
        
        Returns:
            Selected tool or None
        """
        if not available_tools:
            return None
        
        if self.selection_strategy == 'capability_match':
            return self._select_by_capability(task_requirements, available_tools)
        elif self.selection_strategy == 'efficiency':
            return self._select_by_efficiency(available_tools)
        else:
            return self._select_by_capability(task_requirements, available_tools)
    
    def _select_by_capability(self,
                             requirements: Dict[str, float],
                             tools: List[Tool]) -> Optional[Tool]:
        """Select tool based on capability match"""
        best_tool = None
        best_score = -1.0
        
        for tool in tools:
            score = 0.0
            total_requirements = 0.0
            
            for req_name, req_level in requirements.items():
                tool_capability = tool.capabilities.get(req_name, 0.0)
                match = min(1.0, tool_capability / (req_level + 1e-10))
                score += match * req_level
                total_requirements += req_level
            
            if total_requirements > 0:
                score = score / total_requirements
            
            # Factor in efficiency and success rate
            score = score * tool.efficiency * tool.success_rate
            
            if score > best_score:
                best_score = score
                best_tool = tool
        
        return best_tool if best_score > 0.3 else None
    
    def _select_by_efficiency(self, tools: List[Tool]) -> Optional[Tool]:
        """Select tool based on efficiency"""
        if not tools:
            return None
        
        return max(tools, key=lambda t: t.efficiency * t.success_rate)


class ToolCompositionManager:
    """
    Tool Composition Manager
    
    Combines multiple tools
    Creates tool pipelines
    """
    
    def __init__(self):
        self.compositions: Dict[int, ToolComposition] = {}
        self.next_composition_id = 0
    
    def compose_tools(self,
                      tools: List[Tool],
                      execution_order: Optional[List[int]] = None) -> ToolComposition:
        """
        Compose multiple tools into a pipeline
        
        Returns:
            Tool composition
        """
        if execution_order is None:
            execution_order = list(range(len(tools)))
        
        composition = ToolComposition(
            composition_id=self.next_composition_id,
            name=f"Composition of {len(tools)} tools",
            tools=[t.tool_id for t in tools],
            execution_order=execution_order,
            efficiency=self._compute_composition_efficiency(tools)
        )
        
        self.compositions[self.next_composition_id] = composition
        self.next_composition_id += 1
        
        return composition
    
    def _compute_composition_efficiency(self, tools: List[Tool]) -> float:
        """Compute efficiency of tool composition"""
        if not tools:
            return 0.0
        
        # Efficiency decreases with each tool (simplified)
        base_efficiency = np.mean([t.efficiency for t in tools])
        composition_penalty = 0.9 ** (len(tools) - 1)
        
        return base_efficiency * composition_penalty
    
    def execute_composition(self,
                           composition: ToolComposition,
                           tools: Dict[int, Tool],
                           input_data: Any) -> Any:
        """
        Execute a tool composition
        
        Returns:
            Final output
        """
        current_data = input_data
        
        for tool_idx in composition.execution_order:
            if tool_idx < len(composition.tools):
                tool_id = composition.tools[tool_idx]
                if tool_id in tools:
                    tool = tools[tool_id]
                    try:
                        current_data = tool.function(current_data)
                        tool.usage_count += 1
                    except Exception as e:
                        # Tool failed
                        break
        
        composition.usage_count += 1
        return current_data


class ToolCreation:
    """
    Tool Creation
    
    Creates new tools from existing components
    Synthesizes tool functionality
    """
    
    def __init__(self):
        self.created_tools: List[Tool] = []
    
    def create_tool_from_components(self,
                                   name: str,
                                   component_tools: List[Tool],
                                   new_capabilities: Dict[str, float]) -> Tool:
        """
        Create new tool from existing components
        
        Returns:
            Created tool
        """
        # Combine capabilities
        combined_capabilities = new_capabilities.copy()
        for tool in component_tools:
            for cap_name, cap_value in tool.capabilities.items():
                if cap_name in combined_capabilities:
                    combined_capabilities[cap_name] = max(
                        combined_capabilities[cap_name], cap_value
                    )
                else:
                    combined_capabilities[cap_name] = cap_value
        
        # Create composite function
        def composite_function(input_data):
            result = input_data
            for tool in component_tools:
                result = tool.function(result)
            return result
        
        # Create tool
        tool = Tool(
            tool_id=-1,  # Will be assigned by registry
            name=name,
            description=f"Created from {len(component_tools)} components",
            function=composite_function,
            capabilities=combined_capabilities,
            efficiency=np.mean([t.efficiency for t in component_tools]) * 0.9,
            created_time=time.time()
        )
        
        self.created_tools.append(tool)
        return tool
    
    def synthesize_tool(self,
                       requirements: Dict[str, float],
                       available_tools: List[Tool]) -> Optional[Tool]:
        """
        Synthesize a tool to meet requirements
        
        Returns:
            Synthesized tool or None
        """
        # Find tools that partially meet requirements
        partial_tools = []
        for tool in available_tools:
            coverage = 0.0
            for req_name, req_level in requirements.items():
                if req_name in tool.capabilities:
                    coverage += min(1.0, tool.capabilities[req_name] / req_level)
            
            coverage = coverage / len(requirements) if requirements else 0.0
            if coverage > 0.3:
                partial_tools.append((tool, coverage))
        
        if not partial_tools:
            return None
        
        # Select best tools to combine
        partial_tools.sort(key=lambda x: x[1], reverse=True)
        selected_tools = [t[0] for t in partial_tools[:3]]  # Top 3
        
        # Create synthesized tool
        synthesized = self.create_tool_from_components(
            name="Synthesized tool",
            component_tools=selected_tools,
            new_capabilities=requirements
        )
        
        return synthesized


class ToolOptimization:
    """
    Tool Optimization
    
    Improves tool efficiency
    Optimizes tool performance
    """
    
    def __init__(self):
        self.optimization_history: List[Dict] = []
    
    def optimize_tool(self, tool: Tool, performance_data: List[float]) -> Tool:
        """
        Optimize a tool based on performance data
        
        Returns:
            Optimized tool
        """
        if not performance_data:
            return tool
        
        # Compute average performance
        avg_performance = np.mean(performance_data)
        
        # Adjust efficiency based on performance
        if avg_performance > 0.8:
            tool.efficiency = min(1.0, tool.efficiency * 1.1)
        elif avg_performance < 0.5:
            tool.efficiency = max(0.1, tool.efficiency * 0.9)
        
        # Update success rate
        tool.success_rate = avg_performance
        
        self.optimization_history.append({
            'tool_id': tool.tool_id,
            'old_efficiency': tool.efficiency / 1.1 if avg_performance > 0.8 else tool.efficiency / 0.9,
            'new_efficiency': tool.efficiency,
            'timestamp': time.time()
        })
        
        return tool
    
    def optimize_composition(self,
                           composition: ToolComposition,
                           performance_data: List[float]) -> ToolComposition:
        """Optimize tool composition"""
        if not performance_data:
            return composition
        
        avg_performance = np.mean(performance_data)
        composition.efficiency = avg_performance
        
        return composition


class ToolUseSystem:
    """
    Tool Use System Manager
    
    Integrates all tool use components
    """
    
    def __init__(self,
                 brain_system=None,
                 goal_setting: Optional[GoalSettingPlanning] = None,
                 hierarchical_learner: Optional[HierarchicalFeatureLearning] = None,
                 world_model: Optional[WorldModelManager] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.tool_discovery = ToolDiscovery()
        self.tool_selection = ToolSelection()
        self.tool_composition = ToolCompositionManager()
        self.tool_creation = ToolCreation()
        self.tool_optimization = ToolOptimization()
        
        # Integration with existing systems
        self.goal_setting = goal_setting
        self.hierarchical_learner = hierarchical_learner
        self.world_model = world_model
        
        # Statistics
        self.stats = {
            'tools_registered': 0,
            'tools_selected': 0,
            'compositions_created': 0,
            'tools_created': 0,
            'tools_optimized': 0
        }
    
    def register_tool(self,
                     name: str,
                     description: str,
                     function: Callable,
                     capabilities: Dict[str, float]) -> Tool:
        """Register a new tool"""
        tool = self.tool_discovery.register_tool(
            name, description, function, capabilities
        )
        self.stats['tools_registered'] += 1
        return tool
    
    def use_tool_for_task(self,
                         task_requirements: Dict[str, float],
                         input_data: Any) -> Tuple[Optional[Tool], Any]:
        """
        Select and use tool for a task
        
        Returns:
            (selected_tool, output)
        """
        # Discover tools
        available_tools = []
        for req_name in task_requirements.keys():
            tools = self.tool_discovery.discover_tools_by_capability(req_name)
            available_tools.extend(tools)
        
        # Remove duplicates
        available_tools = list({t.tool_id: t for t in available_tools}.values())
        
        # Select tool
        selected_tool = self.tool_selection.select_tool(task_requirements, available_tools)
        
        if selected_tool:
            # Use tool
            try:
                output = selected_tool.function(input_data)
                selected_tool.usage_count += 1
                self.stats['tools_selected'] += 1
                return selected_tool, output
            except Exception as e:
                return None, input_data
        
        return None, input_data
    
    def create_tool_composition(self,
                               tools: List[Tool],
                               execution_order: Optional[List[int]] = None) -> ToolComposition:
        """Create a tool composition"""
        composition = self.tool_composition.compose_tools(tools, execution_order)
        self.stats['compositions_created'] += 1
        return composition
    
    def create_new_tool(self,
                        name: str,
                        requirements: Dict[str, float]) -> Optional[Tool]:
        """Create a new tool to meet requirements"""
        # Find available tools
        available_tools = list(self.tool_discovery.tools.values())
        
        # Synthesize tool
        new_tool = self.tool_creation.synthesize_tool(requirements, available_tools)
        
        if new_tool:
            # Register it
            registered = self.tool_discovery.register_tool(
                new_tool.name, new_tool.description,
                new_tool.function, new_tool.capabilities
            )
            registered.tool_id = new_tool.tool_id
            self.stats['tools_created'] += 1
            return registered
        
        return None
    
    def optimize_tool_performance(self,
                                  tool: Tool,
                                  performance_data: List[float]) -> Tool:
        """Optimize tool performance"""
        optimized = self.tool_optimization.optimize_tool(tool, performance_data)
        self.stats['tools_optimized'] += 1
        return optimized
    
    def get_statistics(self) -> Dict:
        """Get tool use statistics"""
        return self.stats.copy()

