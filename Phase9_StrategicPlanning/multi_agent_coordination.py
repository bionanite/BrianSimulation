#!/usr/bin/env python3
"""
Multi-Agent Coordination - Phase 9.2
Implements agent communication protocols, task allocation, consensus building,
emergent behaviors, and collective intelligence
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
import random
from collections import defaultdict, deque

# Import dependencies
try:
    from communication import CommunicationSystem
    from theory_of_mind import TheoryOfMindSystem
    from social_learning import SocialLearningSystem
except ImportError:
    CommunicationSystem = None
    TheoryOfMindSystem = None
    SocialLearningSystem = None


@dataclass
class Agent:
    """Represents an agent in the multi-agent system"""
    agent_id: int
    name: str
    capabilities: Dict[str, float]  # capability_name -> strength
    current_task: Optional[int] = None
    status: str = 'idle'  # 'idle', 'busy', 'waiting'
    beliefs: Dict[str, any] = field(default_factory=dict)
    created_time: float = 0.0


@dataclass
class Task:
    """Represents a task for agents"""
    task_id: int
    description: str
    requirements: Dict[str, float]  # capability_name -> required_level
    assigned_agent: Optional[int] = None
    status: str = 'pending'  # 'pending', 'assigned', 'in_progress', 'completed'
    created_time: float = 0.0


@dataclass
class Message:
    """Represents a message between agents"""
    message_id: int
    from_agent_id: int
    to_agent_id: int
    content: str
    message_type: str  # 'request', 'response', 'inform', 'propose'
    timestamp: float = 0.0


class AgentCommunicationProtocols:
    """
    Agent Communication Protocols
    
    Structured communication between agents
    Message passing and protocols
    """
    
    def __init__(self):
        self.protocols: Dict[str, Dict] = {}
        self.messages: Dict[int, Message] = {}
        self.message_queue: deque = deque()
        self.next_message_id = 0
        self._initialize_protocols()
    
    def _initialize_protocols(self):
        """Initialize communication protocols"""
        self.protocols = {
            'request_response': {
                'steps': ['request', 'acknowledge', 'response'],
                'timeout': 10.0
            },
            'broadcast': {
                'steps': ['broadcast', 'receive'],
                'timeout': 5.0
            },
            'consensus': {
                'steps': ['propose', 'vote', 'decide'],
                'timeout': 30.0
            }
        }
    
    def send_message(self,
                    from_agent_id: int,
                    to_agent_id: int,
                    content: str,
                    message_type: str = 'inform') -> Message:
        """
        Send a message between agents
        
        Returns:
            Created message
        """
        message = Message(
            message_id=self.next_message_id,
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            content=content,
            message_type=message_type,
            timestamp=time.time()
        )
        
        self.messages[self.next_message_id] = message
        self.message_queue.append(message)
        self.next_message_id += 1
        
        return message
    
    def broadcast_message(self,
                         from_agent_id: int,
                         content: str,
                         agent_ids: List[int]) -> List[Message]:
        """
        Broadcast message to multiple agents
        
        Returns:
            List of messages created
        """
        messages = []
        for agent_id in agent_ids:
            if agent_id != from_agent_id:
                message = self.send_message(
                    from_agent_id, agent_id, content, message_type='broadcast'
                )
                messages.append(message)
        
        return messages
    
    def receive_messages(self, agent_id: int) -> List[Message]:
        """Get messages for an agent"""
        return [msg for msg in self.messages.values() 
                if msg.to_agent_id == agent_id]
    
    def follow_protocol(self,
                       protocol_name: str,
                       from_agent_id: int,
                       to_agent_id: int,
                       content: str) -> List[Message]:
        """
        Follow a communication protocol
        
        Returns:
            List of messages in protocol
        """
        if protocol_name not in self.protocols:
            return []
        
        protocol = self.protocols[protocol_name]
        messages = []
        
        for step in protocol['steps']:
            if step == 'request':
                msg = self.send_message(from_agent_id, to_agent_id, content, 'request')
            elif step == 'propose':
                msg = self.send_message(from_agent_id, to_agent_id, content, 'propose')
            elif step == 'broadcast':
                msg = self.send_message(from_agent_id, to_agent_id, content, 'broadcast')
            else:
                msg = self.send_message(from_agent_id, to_agent_id, content, 'inform')
            
            messages.append(msg)
        
        return messages


class TaskAllocation:
    """
    Task Allocation
    
    Distributes tasks among agents
    Matches tasks to agent capabilities
    """
    
    def __init__(self,
                 allocation_strategy: str = 'capability_match'):
        self.allocation_strategy = allocation_strategy
        self.allocations: Dict[int, int] = {}  # task_id -> agent_id
        self.agent_loads: Dict[int, int] = defaultdict(int)  # agent_id -> task_count
    
    def allocate_task(self,
                     task: Task,
                     agents: Dict[int, Agent]) -> Optional[int]:
        """
        Allocate task to best agent
        
        Returns:
            Assigned agent ID or None
        """
        if self.allocation_strategy == 'capability_match':
            return self._allocate_by_capability(task, agents)
        elif self.allocation_strategy == 'load_balance':
            return self._allocate_by_load(task, agents)
        else:
            return self._allocate_by_capability(task, agents)
    
    def _allocate_by_capability(self,
                               task: Task,
                               agents: Dict[int, Agent]) -> Optional[int]:
        """Allocate based on capability match"""
        best_agent_id = None
        best_score = -1.0
        
        for agent_id, agent in agents.items():
            if agent.status == 'busy':
                continue
            
            # Compute capability match score
            score = 0.0
            total_requirements = 0.0
            
            for req_name, req_level in task.requirements.items():
                agent_capability = agent.capabilities.get(req_name, 0.0)
                match = min(1.0, agent_capability / (req_level + 1e-10))
                score += match * req_level
                total_requirements += req_level
            
            if total_requirements > 0:
                score = score / total_requirements
            
            # Prefer agents with lower current load
            load_penalty = self.agent_loads[agent_id] * 0.1
            score = score - load_penalty
            
            if score > best_score:
                best_score = score
                best_agent_id = agent_id
        
        if best_agent_id is not None and best_score > 0.3:
            self.allocations[task.task_id] = best_agent_id
            self.agent_loads[best_agent_id] += 1
            task.assigned_agent = best_agent_id
            task.status = 'assigned'
            agents[best_agent_id].current_task = task.task_id
            agents[best_agent_id].status = 'busy'
            
            return best_agent_id
        
        return None
    
    def _allocate_by_load(self,
                         task: Task,
                         agents: Dict[int, Agent]) -> Optional[int]:
        """Allocate based on load balancing"""
        # Find agent with lowest load
        available_agents = [(aid, agent) for aid, agent in agents.items() 
                           if agent.status != 'busy']
        
        if not available_agents:
            return None
        
        # Sort by load
        available_agents.sort(key=lambda x: self.agent_loads[x[0]])
        
        # Select agent with lowest load
        best_agent_id = available_agents[0][0]
        self.allocations[task.task_id] = best_agent_id
        self.agent_loads[best_agent_id] += 1
        task.assigned_agent = best_agent_id
        task.status = 'assigned'
        agents[best_agent_id].current_task = task.task_id
        agents[best_agent_id].status = 'busy'
        
        return best_agent_id
    
    def release_task(self, task_id: int, agents: Dict[int, Agent]):
        """Release task and free agent"""
        if task_id in self.allocations:
            agent_id = self.allocations[task_id]
            if agent_id in agents:
                agents[agent_id].current_task = None
                agents[agent_id].status = 'idle'
                self.agent_loads[agent_id] = max(0, self.agent_loads[agent_id] - 1)
            del self.allocations[task_id]


class ConsensusBuilding:
    """
    Consensus Building
    
    Reaches agreement among agents
    Voting and negotiation mechanisms
    """
    
    def __init__(self,
                 voting_threshold: float = 0.5,
                 negotiation_rounds: int = 3):
        self.voting_threshold = voting_threshold
        self.negotiation_rounds = negotiation_rounds
        self.votes: Dict[str, Dict[int, bool]] = {}  # proposal_id -> agent_id -> vote
        self.consensus_history: List[Dict] = []
    
    def vote(self,
            proposal_id: str,
            agent_id: int,
            vote: bool):
        """Record a vote"""
        if proposal_id not in self.votes:
            self.votes[proposal_id] = {}
        self.votes[proposal_id][agent_id] = vote
    
    def reach_consensus(self,
                      proposal_id: str,
                      agent_ids: List[int]) -> Tuple[bool, float]:
        """
        Attempt to reach consensus
        
        Returns:
            (consensus_reached, agreement_ratio)
        """
        if proposal_id not in self.votes:
            return False, 0.0
        
        votes = self.votes[proposal_id]
        total_votes = len([v for v in votes.values()])
        positive_votes = sum(1 for v in votes.values() if v)
        
        if total_votes == 0:
            return False, 0.0
        
        agreement_ratio = positive_votes / total_votes
        consensus_reached = agreement_ratio >= self.voting_threshold
        
        if consensus_reached:
            self.consensus_history.append({
                'proposal_id': proposal_id,
                'agreement_ratio': agreement_ratio,
                'timestamp': time.time()
            })
        
        return consensus_reached, agreement_ratio
    
    def negotiate(self,
                 agents: List[Agent],
                 proposal: str,
                 rounds: Optional[int] = None) -> Optional[str]:
        """
        Negotiate to reach agreement
        
        Returns:
            Final agreement or None
        """
        if rounds is None:
            rounds = self.negotiation_rounds
        
        current_proposal = proposal
        
        for round_num in range(rounds):
            # Each agent votes
            votes = []
            for agent in agents:
                # Simplified voting (in practice would use agent beliefs)
                vote = random.random() > 0.3  # 70% chance of yes
                votes.append(vote)
            
            agreement_ratio = sum(votes) / len(votes) if votes else 0.0
            
            if agreement_ratio >= self.voting_threshold:
                return current_proposal
            
            # Modify proposal if no consensus
            # Simplified: add compromise
            current_proposal = f"{current_proposal} (compromise round {round_num + 1})"
        
        return None


class EmergentBehaviors:
    """
    Emergent Behaviors
    
    Observes emergent behaviors from agent interactions
    Detects patterns in multi-agent systems
    """
    
    def __init__(self):
        self.interaction_history: List[Dict] = []
        self.emergent_patterns: List[Dict] = []
    
    def observe_interaction(self,
                           agent1_id: int,
                           agent2_id: int,
                           interaction_type: str,
                           outcome: Dict):
        """Observe an interaction between agents"""
        interaction = {
            'agent1_id': agent1_id,
            'agent2_id': agent2_id,
            'interaction_type': interaction_type,
            'outcome': outcome,
            'timestamp': time.time()
        }
        
        self.interaction_history.append(interaction)
        
        # Check for emergent patterns
        self._detect_patterns()
    
    def _detect_patterns(self):
        """Detect emergent patterns from interactions"""
        if len(self.interaction_history) < 5:
            return
        
        # Detect cooperation patterns
        cooperation_count = sum(1 for i in self.interaction_history[-10:]
                               if i['interaction_type'] == 'cooperation')
        
        if cooperation_count >= 3:
            pattern = {
                'type': 'cooperation_emergence',
                'description': 'Agents showing increased cooperation',
                'strength': cooperation_count / 10.0,
                'timestamp': time.time()
            }
            self.emergent_patterns.append(pattern)
        
        # Detect competition patterns
        competition_count = sum(1 for i in self.interaction_history[-10:]
                                if i['interaction_type'] == 'competition')
        
        if competition_count >= 3:
            pattern = {
                'type': 'competition_emergence',
                'description': 'Agents showing competitive behavior',
                'strength': competition_count / 10.0,
                'timestamp': time.time()
            }
            self.emergent_patterns.append(pattern)
    
    def get_emergent_behaviors(self) -> List[Dict]:
        """Get detected emergent behaviors"""
        return self.emergent_patterns.copy()


class CollectiveIntelligence:
    """
    Collective Intelligence
    
    Harnesses group intelligence
    Aggregates agent knowledge and decisions
    """
    
    def __init__(self,
                 aggregation_method: str = 'weighted_average'):
        self.aggregation_method = aggregation_method
        self.collective_decisions: List[Dict] = []
    
    def aggregate_knowledge(self,
                          agents: List[Agent],
                          knowledge_field: str) -> Dict:
        """
        Aggregate knowledge from multiple agents
        
        Returns:
            Aggregated knowledge
        """
        knowledge_items = []
        weights = []
        
        for agent in agents:
            if knowledge_field in agent.beliefs:
                knowledge_items.append(agent.beliefs[knowledge_field])
                # Weight by agent capability (simplified)
                weight = sum(agent.capabilities.values()) / max(1, len(agent.capabilities))
                weights.append(weight)
        
        if not knowledge_items:
            return {}
        
        if self.aggregation_method == 'weighted_average':
            if isinstance(knowledge_items[0], (int, float)):
                # Numeric aggregation
                total_weight = sum(weights)
                if total_weight > 0:
                    aggregated = sum(k * w for k, w in zip(knowledge_items, weights)) / total_weight
                else:
                    aggregated = np.mean(knowledge_items)
                return {'value': aggregated, 'confidence': min(1.0, total_weight / len(agents))}
            else:
                # Use majority vote for non-numeric
                from collections import Counter
                counter = Counter(knowledge_items)
                most_common = counter.most_common(1)[0]
                return {'value': most_common[0], 'confidence': most_common[1] / len(knowledge_items)}
        
        return {'value': knowledge_items[0], 'confidence': 0.5}
    
    def make_collective_decision(self,
                                agents: List[Agent],
                                decision_options: List[str]) -> Optional[str]:
        """
        Make collective decision
        
        Returns:
            Selected option or None
        """
        # Each agent votes for an option
        votes = defaultdict(int)
        
        for agent in agents:
            # Simplified: agent selects based on beliefs
            if decision_options:
                # Use agent's first capability as preference indicator
                if agent.capabilities:
                    preference_idx = int(sum(agent.capabilities.values()) * 10) % len(decision_options)
                    selected = decision_options[preference_idx]
                else:
                    selected = random.choice(decision_options)
                
                votes[selected] += 1
        
        if votes:
            # Select option with most votes
            selected_option = max(votes.items(), key=lambda x: x[1])[0]
            
            decision = {
                'selected_option': selected_option,
                'vote_distribution': dict(votes),
                'timestamp': time.time()
            }
            self.collective_decisions.append(decision)
            
            return selected_option
        
        return None


class MultiAgentCoordinationSystem:
    """
    Multi-Agent Coordination System Manager
    
    Integrates all multi-agent coordination components
    """
    
    def __init__(self,
                 brain_system=None,
                 communication: Optional[CommunicationSystem] = None,
                 theory_of_mind: Optional[TheoryOfMindSystem] = None,
                 social_learning: Optional[SocialLearningSystem] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.communication = AgentCommunicationProtocols()
        self.task_allocation = TaskAllocation()
        self.consensus_building = ConsensusBuilding()
        self.emergent_behaviors = EmergentBehaviors()
        self.collective_intelligence = CollectiveIntelligence()
        
        # Integration with existing systems
        self.communication_system = communication
        self.theory_of_mind = theory_of_mind
        self.social_learning = social_learning
        
        # Agent and task tracking
        self.agents: Dict[int, Agent] = {}
        self.tasks: Dict[int, Task] = {}
        self.next_agent_id = 0
        self.next_task_id = 0
        
        # Statistics
        self.stats = {
            'agents_created': 0,
            'tasks_allocated': 0,
            'messages_sent': 0,
            'consensus_reached': 0,
            'emergent_patterns_detected': 0
        }
    
    def create_agent(self,
                    name: str,
                    capabilities: Dict[str, float]) -> Agent:
        """Create a new agent"""
        agent = Agent(
            agent_id=self.next_agent_id,
            name=name,
            capabilities=capabilities,
            created_time=time.time()
        )
        
        self.agents[self.next_agent_id] = agent
        self.next_agent_id += 1
        self.stats['agents_created'] += 1
        
        return agent
    
    def create_task(self,
                   description: str,
                   requirements: Dict[str, float]) -> Task:
        """Create a new task"""
        task = Task(
            task_id=self.next_task_id,
            description=description,
            requirements=requirements,
            created_time=time.time()
        )
        
        self.tasks[self.next_task_id] = task
        self.next_task_id += 1
        
        return task
    
    def coordinate_task_execution(self, task_id: int) -> bool:
        """
        Coordinate execution of a task
        
        Returns:
            Success status
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # Allocate task
        assigned_agent_id = self.task_allocation.allocate_task(task, self.agents)
        
        if assigned_agent_id is not None:
            self.stats['tasks_allocated'] += 1
            
            # Notify agent via communication
            message = self.communication.send_message(
                from_agent_id=-1,  # System
                to_agent_id=assigned_agent_id,
                content=f"Task assigned: {task.description}",
                message_type='inform'
            )
            self.stats['messages_sent'] += 1
            
            return True
        
        return False
    
    def build_consensus(self,
                      proposal_id: str,
                      proposal: str,
                      agent_ids: List[int]) -> Tuple[bool, float]:
        """
        Build consensus on a proposal
        
        Returns:
            (consensus_reached, agreement_ratio)
        """
        # Broadcast proposal
        for agent_id in agent_ids:
            self.communication.broadcast_message(
                from_agent_id=-1,  # System
                content=proposal,
                agent_ids=agent_ids
            )
            self.stats['messages_sent'] += len(agent_ids) - 1
        
        # Agents vote (simplified)
        for agent_id in agent_ids:
            # Simplified voting logic
            vote = random.random() > 0.3  # 70% chance of yes
            self.consensus_building.vote(proposal_id, agent_id, vote)
        
        # Check consensus
        consensus_reached, agreement_ratio = self.consensus_building.reach_consensus(
            proposal_id, agent_ids
        )
        
        if consensus_reached:
            self.stats['consensus_reached'] += 1
        
        return consensus_reached, agreement_ratio
    
    def observe_agent_interaction(self,
                                agent1_id: int,
                                agent2_id: int,
                                interaction_type: str):
        """Observe and record agent interaction"""
        outcome = {'success': True, 'type': interaction_type}
        self.emergent_behaviors.observe_interaction(
            agent1_id, agent2_id, interaction_type, outcome
        )
        
        # Update statistics
        patterns = self.emergent_behaviors.get_emergent_behaviors()
        self.stats['emergent_patterns_detected'] = len(patterns)
    
    def aggregate_agent_knowledge(self, knowledge_field: str) -> Dict:
        """Aggregate knowledge from all agents"""
        agents_list = list(self.agents.values())
        return self.collective_intelligence.aggregate_knowledge(agents_list, knowledge_field)
    
    def get_statistics(self) -> Dict:
        """Get multi-agent coordination statistics"""
        return self.stats.copy()

