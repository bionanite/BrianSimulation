#!/usr/bin/env python3
"""
Competitive & Cooperative Strategies - Phase 9.3
Implements game theory, Nash equilibrium, cooperation mechanisms,
trust modeling, and strategy evolution
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
import random
from collections import defaultdict

# Import dependencies
try:
    from value_systems import ValueSystemsManager
    from social_learning import SocialLearningSystem
    from reward_learning import RewardLearningManager
except ImportError:
    ValueSystemsManager = None
    SocialLearningSystem = None
    RewardLearningManager = None


@dataclass
class Strategy:
    """Represents an agent strategy"""
    strategy_id: int
    name: str
    strategy_type: str  # 'cooperative', 'competitive', 'mixed'
    parameters: Dict[str, float]
    success_rate: float = 0.5
    usage_count: int = 0
    last_used: float = 0.0


@dataclass
class Game:
    """Represents a game-theoretic scenario"""
    game_id: int
    players: List[int]  # Agent IDs
    actions: Dict[int, List[str]]  # agent_id -> available actions
    payoffs: Dict[Tuple[str, ...], Dict[int, float]]  # (action1, action2, ...) -> agent_id -> payoff
    game_type: str  # 'zero_sum', 'cooperative', 'mixed'


@dataclass
class TrustRelation:
    """Represents trust between agents"""
    from_agent_id: int
    to_agent_id: int
    trust_level: float  # 0-1
    interaction_count: int = 0
    positive_interactions: int = 0
    last_updated: float = 0.0


class GameTheory:
    """
    Game Theory
    
    Applies game-theoretic principles
    Analyzes strategic interactions
    """
    
    def __init__(self):
        self.games: Dict[int, Game] = {}
        self.next_game_id = 0
        self.nash_equilibria: Dict[int, List[Tuple[str, ...]]] = {}  # game_id -> equilibria
    
    def create_game(self,
                   players: List[int],
                   actions: Dict[int, List[str]],
                   payoffs: Dict[Tuple[str, ...], Dict[int, float]],
                   game_type: str = 'mixed') -> Game:
        """Create a game-theoretic scenario"""
        game = Game(
            game_id=self.next_game_id,
            players=players,
            actions=actions,
            payoffs=payoffs,
            game_type=game_type
        )
        
        self.games[self.next_game_id] = game
        self.next_game_id += 1
        
        return game
    
    def find_nash_equilibrium(self, game: Game) -> List[Tuple[str, ...]]:
        """
        Find Nash equilibrium strategies
        
        Returns:
            List of equilibrium action profiles
        """
        equilibria = []
        
        # Simplified Nash equilibrium search
        # For 2-player games
        if len(game.players) == 2:
            player1_id = game.players[0]
            player2_id = game.players[1]
            
            actions1 = game.actions.get(player1_id, [])
            actions2 = game.actions.get(player2_id, [])
            
            for a1 in actions1:
                for a2 in actions2:
                    action_profile = (a1, a2)
                    
                    if action_profile in game.payoffs:
                        payoffs = game.payoffs[action_profile]
                        payoff1 = payoffs.get(player1_id, 0.0)
                        payoff2 = payoffs.get(player2_id, 0.0)
                        
                        # Check if this is a Nash equilibrium
                        # (simplified check - in practice would check all deviations)
                        is_equilibrium = True
                        
                        # Check if player1 wants to deviate
                        for alt_a1 in actions1:
                            if alt_a1 != a1:
                                alt_profile = (alt_a1, a2)
                                if alt_profile in game.payoffs:
                                    alt_payoff1 = game.payoffs[alt_profile].get(player1_id, 0.0)
                                    if alt_payoff1 > payoff1:
                                        is_equilibrium = False
                                        break
                        
                        # Check if player2 wants to deviate
                        if is_equilibrium:
                            for alt_a2 in actions2:
                                if alt_a2 != a2:
                                    alt_profile = (a1, alt_a2)
                                    if alt_profile in game.payoffs:
                                        alt_payoff2 = game.payoffs[alt_profile].get(player2_id, 0.0)
                                        if alt_payoff2 > payoff2:
                                            is_equilibrium = False
                                            break
                        
                        if is_equilibrium:
                            equilibria.append(action_profile)
        
        self.nash_equilibria[game.game_id] = equilibria
        return equilibria
    
    def compute_payoff(self,
                      game: Game,
                      action_profile: Tuple[str, ...]) -> Dict[int, float]:
        """Compute payoffs for an action profile"""
        if action_profile in game.payoffs:
            return game.payoffs[action_profile].copy()
        return {player_id: 0.0 for player_id in game.players}


class CooperationMechanisms:
    """
    Cooperation Mechanisms
    
    Enables cooperation in competitive settings
    Implements cooperation strategies
    """
    
    def __init__(self,
                 cooperation_threshold: float = 0.6,
                 reciprocity_strength: float = 0.7):
        self.cooperation_threshold = cooperation_threshold
        self.reciprocity_strength = reciprocity_strength
        self.cooperation_history: Dict[Tuple[int, int], List[bool]] = defaultdict(list)
    
    def decide_cooperation(self,
                          agent1_id: int,
                          agent2_id: int,
                          context: Dict) -> bool:
        """
        Decide whether to cooperate
        
        Returns:
            Cooperation decision
        """
        # Check cooperation history
        history_key = tuple(sorted([agent1_id, agent2_id]))
        history = self.cooperation_history[history_key]
        
        if history:
            # Use reciprocity: cooperate if other agent cooperated
            recent_cooperation = sum(history[-5:]) / max(1, len(history[-5:]))
            
            if recent_cooperation >= self.cooperation_threshold:
                return True  # Reciprocate cooperation
            elif recent_cooperation < (1.0 - self.cooperation_threshold):
                return False  # Reciprocate defection
            else:
                # Mixed history - use context
                return context.get('cooperation_benefit', 0.5) > 0.5
        else:
            # No history - initial cooperation based on context
            return context.get('initial_cooperation', True)
    
    def record_cooperation(self,
                          agent1_id: int,
                          agent2_id: int,
                          cooperated: bool):
        """Record cooperation outcome"""
        history_key = tuple(sorted([agent1_id, agent2_id]))
        self.cooperation_history[history_key].append(cooperated)
        
        # Limit history
        if len(self.cooperation_history[history_key]) > 100:
            self.cooperation_history[history_key] = self.cooperation_history[history_key][-100:]
    
    def compute_cooperation_score(self,
                                 agent1_id: int,
                                 agent2_id: int) -> float:
        """Compute cooperation score between agents"""
        history_key = tuple(sorted([agent1_id, agent2_id]))
        history = self.cooperation_history[history_key]
        
        if not history:
            return 0.5  # Neutral
        
        return sum(history) / len(history)


class TrustModeling:
    """
    Trust Modeling
    
    Models and maintains trust between agents
    Updates trust based on interactions
    """
    
    def __init__(self,
                 initial_trust: float = 0.5,
                 trust_decay: float = 0.01):
        self.initial_trust = initial_trust
        self.trust_decay = trust_decay
        self.trust_relations: Dict[Tuple[int, int], TrustRelation] = {}
    
    def get_trust(self,
                 from_agent_id: int,
                 to_agent_id: int) -> float:
        """Get trust level between agents"""
        trust_key = (from_agent_id, to_agent_id)
        
        if trust_key in self.trust_relations:
            trust_rel = self.trust_relations[trust_key]
            
            # Apply decay over time
            time_since_update = time.time() - trust_rel.last_updated
            decay_factor = np.exp(-self.trust_decay * time_since_update)
            trust = trust_rel.trust_level * decay_factor
            
            return max(0.0, min(1.0, trust))
        
        return self.initial_trust
    
    def update_trust(self,
                    from_agent_id: int,
                    to_agent_id: int,
                    interaction_outcome: Dict):
        """
        Update trust based on interaction outcome
        
        Args:
            interaction_outcome: Dict with 'success', 'cooperation', etc.
        """
        trust_key = (from_agent_id, to_agent_id)
        
        if trust_key not in self.trust_relations:
            trust_rel = TrustRelation(
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                trust_level=self.initial_trust
            )
            self.trust_relations[trust_key] = trust_rel
        else:
            trust_rel = self.trust_relations[trust_key]
        
        # Update trust based on outcome
        success = interaction_outcome.get('success', False)
        cooperation = interaction_outcome.get('cooperation', False)
        
        if success and cooperation:
            # Positive interaction
            trust_rel.positive_interactions += 1
            trust_rel.trust_level = min(1.0, trust_rel.trust_level + 0.1)
        elif not success:
            # Negative interaction
            trust_rel.trust_level = max(0.0, trust_rel.trust_level - 0.1)
        
        trust_rel.interaction_count += 1
        trust_rel.last_updated = time.time()
    
    def compute_trust_network(self, agent_ids: List[int]) -> np.ndarray:
        """
        Compute trust network matrix
        
        Returns:
            Trust matrix (agent_id index -> agent_id index -> trust)
        """
        n = len(agent_ids)
        trust_matrix = np.zeros((n, n))
        
        for i, agent1_id in enumerate(agent_ids):
            for j, agent2_id in enumerate(agent_ids):
                if i != j:
                    trust_matrix[i, j] = self.get_trust(agent1_id, agent2_id)
                else:
                    trust_matrix[i, j] = 1.0  # Self-trust
        
        return trust_matrix


class StrategyEvolution:
    """
    Strategy Evolution
    
    Evolves strategies over time
    Learns from experience
    """
    
    def __init__(self,
                 mutation_rate: float = 0.1,
                 selection_pressure: float = 0.7):
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        self.strategies: Dict[int, Strategy] = {}
        self.strategy_performance: Dict[int, List[float]] = defaultdict(list)
        self.next_strategy_id = 0
    
    def create_strategy(self,
                       name: str,
                       strategy_type: str,
                       parameters: Dict[str, float]) -> Strategy:
        """Create a new strategy"""
        strategy = Strategy(
            strategy_id=self.next_strategy_id,
            name=name,
            strategy_type=strategy_type,
            parameters=parameters
        )
        
        self.strategies[self.next_strategy_id] = strategy
        self.next_strategy_id += 1
        
        return strategy
    
    def evolve_strategy(self,
                       strategy: Strategy,
                       performance: float) -> Strategy:
        """
        Evolve strategy based on performance
        
        Returns:
            Evolved strategy
        """
        # Update performance history
        self.strategy_performance[strategy.strategy_id].append(performance)
        
        # Update success rate
        if len(self.strategy_performance[strategy.strategy_id]) > 0:
            strategy.success_rate = np.mean(self.strategy_performance[strategy.strategy_id])
        
        # Evolve parameters
        evolved_params = {}
        for param_name, param_value in strategy.parameters.items():
            # Mutate parameter
            if random.random() < self.mutation_rate:
                mutation = np.random.normal(0, 0.1)
                evolved_value = param_value + mutation
            else:
                evolved_value = param_value
            
            # Keep in reasonable range
            evolved_params[param_name] = max(0.0, min(1.0, evolved_value))
        
        # Create evolved strategy
        evolved_strategy = Strategy(
            strategy_id=self.next_strategy_id,
            name=f"{strategy.name}_evolved",
            strategy_type=strategy.strategy_type,
            parameters=evolved_params,
            success_rate=strategy.success_rate
        )
        
        self.strategies[self.next_strategy_id] = evolved_strategy
        self.next_strategy_id += 1
        
        return evolved_strategy
    
    def select_best_strategy(self,
                            strategies: List[Strategy],
                            context: Dict) -> Strategy:
        """
        Select best strategy for context
        
        Returns:
            Best strategy
        """
        if not strategies:
            return None
        
        # Score strategies
        scores = []
        for strategy in strategies:
            # Base score on success rate
            score = strategy.success_rate
            
            # Boost if strategy type matches context
            if context.get('prefer_cooperation') and strategy.strategy_type == 'cooperative':
                score *= 1.2
            elif context.get('prefer_competition') and strategy.strategy_type == 'competitive':
                score *= 1.2
            
            scores.append(score)
        
        # Select best
        best_idx = np.argmax(scores)
        return strategies[best_idx]


class AgentStrategiesSystem:
    """
    Agent Strategies System Manager
    
    Integrates all strategy components
    """
    
    def __init__(self,
                 brain_system=None,
                 value_systems: Optional[ValueSystemsManager] = None,
                 social_learning: Optional[SocialLearningSystem] = None,
                 reward_learning: Optional[RewardLearningManager] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.game_theory = GameTheory()
        self.cooperation = CooperationMechanisms()
        self.trust_modeling = TrustModeling()
        self.strategy_evolution = StrategyEvolution()
        
        # Integration with existing systems
        self.value_systems = value_systems
        self.social_learning = social_learning
        self.reward_learning = reward_learning
        
        # Statistics
        self.stats = {
            'games_analyzed': 0,
            'nash_equilibria_found': 0,
            'cooperation_decisions': 0,
            'trust_updates': 0,
            'strategies_evolved': 0
        }
    
    def analyze_game(self,
                    players: List[int],
                    actions: Dict[int, List[str]],
                    payoffs: Dict[Tuple[str, ...], Dict[int, float]]) -> Dict:
        """
        Analyze a game-theoretic scenario
        
        Returns:
            Analysis results
        """
        game = self.game_theory.create_game(players, actions, payoffs)
        equilibria = self.game_theory.find_nash_equilibrium(game)
        
        self.stats['games_analyzed'] += 1
        if equilibria:
            self.stats['nash_equilibria_found'] += len(equilibria)
        
        return {
            'game_id': game.game_id,
            'equilibria': equilibria,
            'num_equilibria': len(equilibria)
        }
    
    def decide_cooperation(self,
                          agent1_id: int,
                          agent2_id: int,
                          context: Dict) -> bool:
        """Decide whether agents should cooperate"""
        decision = self.cooperation.decide_cooperation(agent1_id, agent2_id, context)
        self.stats['cooperation_decisions'] += 1
        return decision
    
    def update_trust(self,
                    from_agent_id: int,
                    to_agent_id: int,
                    interaction_outcome: Dict):
        """Update trust between agents"""
        self.trust_modeling.update_trust(from_agent_id, to_agent_id, interaction_outcome)
        self.stats['trust_updates'] += 1
    
    def evolve_agent_strategy(self,
                             strategy: Strategy,
                             performance: float) -> Strategy:
        """Evolve an agent's strategy"""
        evolved = self.strategy_evolution.evolve_strategy(strategy, performance)
        self.stats['strategies_evolved'] += 1
        return evolved
    
    def get_statistics(self) -> Dict:
        """Get strategy system statistics"""
        return self.stats.copy()

