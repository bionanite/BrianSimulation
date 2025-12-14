#!/usr/bin/env python3
"""
Communication
Implements signal generation, signal interpretation, language-like structures, and communication protocols
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from collections import defaultdict

@dataclass
class Signal:
    """Represents a communication signal"""
    signal_id: int
    sender_id: int
    receiver_id: Optional[int]  # None for broadcast
    signal_type: str  # 'request', 'inform', 'command', 'query'
    content: str
    meaning: Dict[str, float]  # semantic components
    timestamp: float = 0.0

@dataclass
class Symbol:
    """Represents a symbol in a language"""
    symbol_id: int
    symbol: str
    meaning: str
    frequency: int = 1
    associations: Dict[str, float] = field(default_factory=dict)  # associated_symbol -> strength

class SignalGeneration:
    """
    Signal Generation
    
    Generates communication signals
    Encodes information into signals
    """
    
    def __init__(self):
        self.generated_signals: List[Signal] = []
        self.signal_vocabulary: Dict[str, Dict[str, float]] = {}  # signal_type -> {content -> meaning}
        self.next_signal_id = 0
    
    def create_signal(self,
                     sender_id: int,
                     signal_type: str,
                     content: str,
                     meaning: Dict[str, float],
                     receiver_id: Optional[int] = None) -> Signal:
        """Create a communication signal"""
        signal = Signal(
            signal_id=self.next_signal_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            signal_type=signal_type,
            content=content,
            meaning=meaning.copy(),
            timestamp=time.time()
        )
        
        self.generated_signals.append(signal)
        self.next_signal_id += 1
        
        # Update vocabulary
        if signal_type not in self.signal_vocabulary:
            self.signal_vocabulary[signal_type] = {}
        self.signal_vocabulary[signal_type][content] = meaning.copy()
        
        return signal
    
    def generate_request(self,
                        sender_id: int,
                        request_content: str,
                        receiver_id: Optional[int] = None) -> Signal:
        """Generate a request signal"""
        meaning = {
            'intent': 'request',
            'urgency': 0.7,
            'politeness': 0.5
        }
        return self.create_signal(sender_id, 'request', request_content, meaning, receiver_id)
    
    def generate_inform(self,
                       sender_id: int,
                       information: str,
                       receiver_id: Optional[int] = None) -> Signal:
        """Generate an inform signal"""
        meaning = {
            'intent': 'inform',
            'certainty': 0.8,
            'importance': 0.6
        }
        return self.create_signal(sender_id, 'inform', information, meaning, receiver_id)
    
    def generate_command(self,
                        sender_id: int,
                        command: str,
                        receiver_id: int) -> Signal:
        """Generate a command signal"""
        meaning = {
            'intent': 'command',
            'authority': 0.9,
            'urgency': 0.8
        }
        return self.create_signal(sender_id, 'command', command, meaning, receiver_id)

class SignalInterpretation:
    """
    Signal Interpretation
    
    Interprets received signals
    Decodes meaning from signals
    """
    
    def __init__(self):
        self.received_signals: List[Signal] = []
        self.interpretation_history: List[Tuple[int, str, Dict]] = []  # (signal_id, interpretation, confidence)
    
    def receive_signal(self, signal: Signal):
        """Receive a signal"""
        self.received_signals.append(signal)
    
    def interpret_signal(self, signal: Signal) -> Tuple[str, float]:
        """
        Interpret a signal
        
        Returns:
            (interpretation, confidence)
        """
        # Basic interpretation based on signal type and content
        interpretation = f"{signal.signal_type}: {signal.content}"
        
        # Confidence based on signal clarity
        confidence = 0.8
        
        # Check meaning components
        if 'certainty' in signal.meaning:
            confidence = signal.meaning['certainty']
        
        self.interpretation_history.append((signal.signal_id, interpretation, {'confidence': confidence}))
        
        return interpretation, confidence
    
    def extract_intent(self, signal: Signal) -> str:
        """Extract intent from signal"""
        return signal.meaning.get('intent', signal.signal_type)
    
    def extract_semantic_content(self, signal: Signal) -> Dict[str, float]:
        """Extract semantic content from signal"""
        return signal.meaning.copy()
    
    def get_recent_signals(self, n: int = 10) -> List[Signal]:
        """Get recent signals"""
        return self.received_signals[-n:] if self.received_signals else []

class LanguageStructures:
    """
    Language Structures
    
    Maintains language-like structures
    Manages symbols and grammar
    """
    
    def __init__(self):
        self.symbols: Dict[str, Symbol] = {}
        self.grammar_rules: List[Tuple[List[str], str]] = []  # (pattern, result)
        self.next_symbol_id = 0
    
    def create_symbol(self,
                    symbol: str,
                    meaning: str) -> Symbol:
        """Create a new symbol"""
        if symbol not in self.symbols:
            sym = Symbol(
                symbol_id=self.next_symbol_id,
                symbol=symbol,
                meaning=meaning
            )
            self.symbols[symbol] = sym
            self.next_symbol_id += 1
        else:
            sym = self.symbols[symbol]
            sym.frequency += 1
        
        return self.symbols[symbol]
    
    def associate_symbols(self,
                        symbol1: str,
                        symbol2: str,
                        strength: float = 0.5):
        """Associate two symbols"""
        if symbol1 in self.symbols and symbol2 in self.symbols:
            self.symbols[symbol1].associations[symbol2] = strength
            self.symbols[symbol2].associations[symbol1] = strength
    
    def add_grammar_rule(self,
                        pattern: List[str],
                        result: str):
        """Add a grammar rule"""
        self.grammar_rules.append((pattern.copy(), result))
    
    def parse_sequence(self, sequence: List[str]) -> Optional[str]:
        """Parse a sequence using grammar rules"""
        for pattern, result in self.grammar_rules:
            if len(sequence) >= len(pattern):
                if sequence[:len(pattern)] == pattern:
                    return result
        return None
    
    def get_symbol_meaning(self, symbol: str) -> Optional[str]:
        """Get meaning of a symbol"""
        if symbol in self.symbols:
            return self.symbols[symbol].meaning
        return None
    
    def get_associated_symbols(self, symbol: str) -> List[Tuple[str, float]]:
        """Get symbols associated with a symbol"""
        if symbol not in self.symbols:
            return []
        
        associations = self.symbols[symbol].associations
        return [(sym, strength) for sym, strength in associations.items()]

class CommunicationProtocols:
    """
    Communication Protocols
    
    Manages communication protocols
    Handles turn-taking and protocols
    """
    
    def __init__(self):
        self.protocols: Dict[str, Dict] = {}  # protocol_name -> protocol_definition
        self.active_conversations: Dict[int, Dict] = {}  # conversation_id -> conversation_state
        self.next_conversation_id = 0
    
    def define_protocol(self,
                       name: str,
                       turn_order: List[int],
                       message_types: List[str]):
        """Define a communication protocol"""
        self.protocols[name] = {
            'turn_order': turn_order.copy(),
            'message_types': message_types.copy(),
            'current_turn': 0
        }
    
    def start_conversation(self,
                          protocol_name: str,
                          participants: List[int]) -> int:
        """Start a conversation using a protocol"""
        if protocol_name not in self.protocols:
            return -1
        
        conversation_id = self.next_conversation_id
        self.next_conversation_id += 1
        
        protocol = self.protocols[protocol_name]
        
        self.active_conversations[conversation_id] = {
            'protocol': protocol_name,
            'participants': participants.copy(),
            'turn_order': protocol['turn_order'].copy(),
            'current_turn': 0,
            'messages': []
        }
        
        return conversation_id
    
    def get_next_speaker(self, conversation_id: int) -> Optional[int]:
        """Get next speaker in conversation"""
        if conversation_id not in self.active_conversations:
            return None
        
        conv = self.active_conversations[conversation_id]
        turn_order = conv['turn_order']
        current_turn = conv['current_turn']
        
        if current_turn >= len(turn_order):
            return None
        
        return turn_order[current_turn]
    
    def advance_turn(self, conversation_id: int):
        """Advance to next turn"""
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id]['current_turn'] += 1
    
    def add_message(self,
                   conversation_id: int,
                   sender_id: int,
                   message: str):
        """Add message to conversation"""
        if conversation_id in self.active_conversations:
            conv = self.active_conversations[conversation_id]
            conv['messages'].append({
                'sender': sender_id,
                'message': message,
                'timestamp': time.time()
            })
            self.advance_turn(conversation_id)

class CommunicationManager:
    """
    Manages all communication mechanisms
    """
    
    def __init__(self):
        self.signal_generation = SignalGeneration()
        self.signal_interpretation = SignalInterpretation()
        self.language_structures = LanguageStructures()
        self.communication_protocols = CommunicationProtocols()
    
    def send_signal(self,
                   sender_id: int,
                   signal_type: str,
                   content: str,
                   meaning: Dict[str, float],
                   receiver_id: Optional[int] = None) -> Signal:
        """Send a signal"""
        signal = self.signal_generation.create_signal(
            sender_id, signal_type, content, meaning, receiver_id
        )
        return signal
    
    def receive_signal(self, signal: Signal) -> Tuple[str, float]:
        """Receive and interpret a signal"""
        self.signal_interpretation.receive_signal(signal)
        return self.signal_interpretation.interpret_signal(signal)
    
    def create_symbol(self, symbol: str, meaning: str):
        """Create a symbol"""
        return self.language_structures.create_symbol(symbol, meaning)
    
    def start_conversation(self,
                          protocol_name: str,
                          participants: List[int]) -> int:
        """Start a conversation"""
        return self.communication_protocols.start_conversation(protocol_name, participants)
    
    def get_statistics(self) -> Dict:
        """Get statistics about communication"""
        return {
            'signals_generated': len(self.signal_generation.generated_signals),
            'signals_received': len(self.signal_interpretation.received_signals),
            'symbols': len(self.language_structures.symbols),
            'grammar_rules': len(self.language_structures.grammar_rules),
            'protocols': len(self.communication_protocols.protocols),
            'active_conversations': len(self.communication_protocols.active_conversations)
        }

