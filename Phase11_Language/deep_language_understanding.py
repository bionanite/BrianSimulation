#!/usr/bin/env python3
"""
Deep Language Understanding - Phase 11.1
Implements semantic parsing, pragmatic inference, contextual understanding,
ambiguity resolution, and metaphor understanding
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
import re
from collections import defaultdict

# Import dependencies
try:
    from language_processor import LanguageProcessor
    from language_integration import LanguageIntegrationSystem
    from semantic_representations import SemanticNetwork
    from context_sensitivity import ContextSensitivitySystem
except ImportError:
    LanguageProcessor = None
    LanguageIntegrationSystem = None
    SemanticNetwork = None
    ContextSensitivitySystem = None


@dataclass
class ParsedMeaning:
    """Represents parsed meaning from language"""
    parse_id: int
    text: str
    semantic_structure: Dict[str, any]
    entities: List[str]
    relations: List[Tuple[str, str, str]]  # (subject, relation, object)
    confidence: float = 0.5


@dataclass
class PragmaticInference:
    """Represents pragmatic inference"""
    inference_id: int
    utterance: str
    inferred_intention: str
    inferred_meaning: str
    confidence: float = 0.5


class SemanticParsing:
    """
    Semantic Parsing
    
    Parses meaning from language
    Extracts semantic structures
    """
    
    def __init__(self):
        self.parses: Dict[int, ParsedMeaning] = {}
        self.next_parse_id = 0
    
    def parse(self, text: str) -> ParsedMeaning:
        """
        Parse meaning from text
        
        Returns:
            Parsed meaning
        """
        # Simplified semantic parsing
        # Extract entities (capitalized words or quoted phrases)
        entities = re.findall(r'\b[A-Z][a-z]+\b|\"[^\"]+\"', text)
        
        # Extract relations (simplified pattern matching)
        relations = []
        words = text.split()
        
        # Look for verb-object patterns
        for i in range(len(words) - 1):
            if words[i].endswith('s') or words[i] in ['is', 'are', 'was', 'were']:
                if i + 1 < len(words):
                    relations.append((words[i-1] if i > 0 else 'subject', words[i], words[i+1]))
        
        # Build semantic structure
        semantic_structure = {
            'entities': entities,
            'relations': relations,
            'main_verb': self._extract_main_verb(text),
            'tense': self._extract_tense(text)
        }
        
        parse = ParsedMeaning(
            parse_id=self.next_parse_id,
            text=text,
            semantic_structure=semantic_structure,
            entities=entities,
            relations=relations,
            confidence=0.7
        )
        
        self.parses[self.next_parse_id] = parse
        self.next_parse_id += 1
        
        return parse
    
    def _extract_main_verb(self, text: str) -> Optional[str]:
        """Extract main verb from text"""
        # Simplified: find first verb-like word
        words = text.split()
        for word in words:
            if word.endswith(('ed', 'ing', 's')) or word in ['is', 'are', 'was', 'were', 'do', 'does', 'did']:
                return word
        return None
    
    def _extract_tense(self, text: str) -> str:
        """Extract tense from text"""
        if any(word in text.lower() for word in ['was', 'were', 'did']):
            return 'past'
        elif any(word in text.lower() for word in ['is', 'are', 'do', 'does']):
            return 'present'
        elif any(word in text.lower() for word in ['will', 'shall']):
            return 'future'
        return 'unknown'


class PragmaticInferenceSystem:
    """
    Pragmatic Inference System
    
    Infers speaker intentions
    Understands implied meaning
    """
    
    def __init__(self):
        self.inferences: Dict[int, PragmaticInference] = {}
        self.next_inference_id = 0
    
    def infer_intention(self, utterance: str, context: Dict = None) -> PragmaticInference:
        """
        Infer speaker intention
        
        Returns:
            Pragmatic inference
        """
        if context is None:
            context = {}
        
        # Simplified intention inference
        utterance_lower = utterance.lower()
        
        # Question detection
        if utterance.endswith('?') or utterance_lower.startswith(('what', 'where', 'when', 'who', 'why', 'how')):
            intention = 'question'
            inferred_meaning = f"Speaker is asking: {utterance}"
        
        # Request detection
        elif any(word in utterance_lower for word in ['please', 'can you', 'could you', 'would you']):
            intention = 'request'
            inferred_meaning = f"Speaker is requesting: {utterance}"
        
        # Statement detection
        elif utterance.endswith('.') or utterance.endswith('!'):
            intention = 'statement'
            inferred_meaning = f"Speaker is stating: {utterance}"
        
        else:
            intention = 'unknown'
            inferred_meaning = utterance
        
        # Adjust based on context
        if context.get('previous_intention') == 'question':
            # Might be a follow-up
            inferred_meaning = f"Follow-up: {inferred_meaning}"
        
        inference = PragmaticInference(
            inference_id=self.next_inference_id,
            utterance=utterance,
            inferred_intention=intention,
            inferred_meaning=inferred_meaning,
            confidence=0.7
        )
        
        self.inferences[self.next_inference_id] = inference
        self.next_inference_id += 1
        
        return inference


class ContextualUnderstanding:
    """
    Contextual Understanding
    
    Understands language in context
    Uses context to disambiguate
    """
    
    def __init__(self):
        self.context_history: List[Dict] = []
        self.context_window_size = 5
    
    def understand_in_context(self,
                              text: str,
                              context: List[str] = None) -> Dict:
        """
        Understand text in context
        
        Returns:
            Contextual understanding
        """
        if context is None:
            context = []
        
        # Use context to disambiguate
        understanding = {
            'text': text,
            'context': context,
            'interpretation': text,
            'confidence': 0.5
        }
        
        # Check for references to previous context
        if context:
            # Look for pronouns and references
            if 'it' in text.lower() or 'this' in text.lower() or 'that' in text.lower():
                # Try to resolve reference
                understanding['interpretation'] = self._resolve_reference(text, context)
                understanding['confidence'] = 0.7
        
        # Add to context history
        self.context_history.append(understanding)
        if len(self.context_history) > self.context_window_size:
            self.context_history.pop(0)
        
        return understanding
    
    def _resolve_reference(self, text: str, context: List[str]) -> str:
        """Resolve pronoun/reference to context"""
        # Simplified: replace pronouns with last mentioned entity
        if not context:
            return text
        
        # Find last entity in context
        last_context = context[-1]
        entities = re.findall(r'\b[A-Z][a-z]+\b', last_context)
        
        if entities:
            # Replace 'it' with last entity
            resolved = text.replace('it', entities[-1])
            resolved = resolved.replace('this', entities[-1])
            resolved = resolved.replace('that', entities[-1])
            return resolved
        
        return text


class AmbiguityResolution:
    """
    Ambiguity Resolution
    
    Resolves linguistic ambiguities
    Selects best interpretation
    """
    
    def __init__(self):
        self.ambiguities_resolved: List[Dict] = []
    
    def resolve_ambiguity(self,
                          ambiguous_text: str,
                          possible_interpretations: List[str],
                          context: Dict = None) -> str:
        """
        Resolve ambiguity
        
        Returns:
            Best interpretation
        """
        if context is None:
            context = {}
        
        if not possible_interpretations:
            return ambiguous_text
        
        # Score interpretations based on context
        scores = []
        for interpretation in possible_interpretations:
            score = 0.5  # Base score
            
            # Boost if interpretation matches context
            if context:
                context_text = ' '.join(context.get('previous_utterances', []))
                if interpretation.lower() in context_text.lower():
                    score += 0.3
            
            # Boost if interpretation is more common (simplified)
            if len(interpretation.split()) < 5:  # Shorter = more common
                score += 0.2
            
            scores.append(score)
        
        # Select best interpretation
        best_idx = np.argmax(scores)
        best_interpretation = possible_interpretations[best_idx]
        
        self.ambiguities_resolved.append({
            'ambiguous_text': ambiguous_text,
            'resolved_interpretation': best_interpretation,
            'confidence': scores[best_idx],
            'timestamp': time.time()
        })
        
        return best_interpretation
    
    def detect_ambiguity(self, text: str) -> List[str]:
        """
        Detect ambiguities in text
        
        Returns:
            List of possible interpretations
        """
        interpretations = []
        
        # Check for common ambiguous words
        ambiguous_words = {
            'bank': ['financial institution', 'river edge'],
            'bark': ['tree covering', 'dog sound'],
            'bat': ['flying mammal', 'sports equipment']
        }
        
        words = text.lower().split()
        for word in words:
            if word in ambiguous_words:
                for meaning in ambiguous_words[word]:
                    interpretation = text.replace(word, meaning)
                    interpretations.append(interpretation)
        
        if not interpretations:
            # Default: return original
            interpretations.append(text)
        
        return interpretations


class MetaphorUnderstanding:
    """
    Metaphor Understanding
    
    Understands metaphors and analogies
    Maps metaphorical meaning
    """
    
    def __init__(self):
        self.metaphors_understood: List[Dict] = []
        self.metaphor_patterns: Dict[str, str] = {
            'time is money': 'time is valuable',
            'love is a journey': 'love involves progress',
            'ideas are food': 'ideas can be consumed'
        }
    
    def understand_metaphor(self, text: str) -> Dict:
        """
        Understand metaphor in text
        
        Returns:
            Metaphor understanding
        """
        understanding = {
            'text': text,
            'is_metaphor': False,
            'literal_meaning': text,
            'metaphorical_meaning': text,
            'mapping': {}
        }
        
        # Check for known metaphor patterns
        text_lower = text.lower()
        for pattern, meaning in self.metaphor_patterns.items():
            if pattern in text_lower:
                understanding['is_metaphor'] = True
                understanding['metaphorical_meaning'] = meaning
                understanding['mapping'] = {'pattern': pattern, 'meaning': meaning}
                break
        
        # Check for common metaphorical structures
        if not understanding['is_metaphor']:
            # Look for "X is Y" patterns
            if ' is ' in text_lower:
                parts = text_lower.split(' is ')
                if len(parts) == 2:
                    # Check if it's metaphorical (abstract concept = concrete thing)
                    understanding['is_metaphor'] = True
                    understanding['mapping'] = {
                        'source': parts[0],
                        'target': parts[1]
                    }
        
        self.metaphors_understood.append(understanding)
        return understanding
    
    def understand_analogy(self,
                          source: str,
                          target: str,
                          relation: str) -> Dict:
        """
        Understand analogy
        
        Returns:
            Analogy understanding
        """
        analogy = {
            'source': source,
            'target': target,
            'relation': relation,
            'mapping': {
                'source_domain': source,
                'target_domain': target,
                'relation': relation
            }
        }
        
        return analogy


class DeepLanguageUnderstandingSystem:
    """
    Deep Language Understanding System Manager
    
    Integrates all language understanding components
    """
    
    def __init__(self,
                 brain_system=None,
                 language_processor: Optional[LanguageProcessor] = None,
                 language_integration: Optional[LanguageIntegrationSystem] = None,
                 semantic_network: Optional[SemanticNetwork] = None,
                 context_sensitivity: Optional[ContextSensitivitySystem] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.semantic_parsing = SemanticParsing()
        self.pragmatic_inference = PragmaticInferenceSystem()
        self.contextual_understanding = ContextualUnderstanding()
        self.ambiguity_resolution = AmbiguityResolution()
        self.metaphor_understanding = MetaphorUnderstanding()
        
        # Integration with existing systems
        self.language_processor = language_processor
        self.language_integration = language_integration
        self.semantic_network = semantic_network
        self.context_sensitivity = context_sensitivity
        
        # Statistics
        self.stats = {
            'texts_parsed': 0,
            'intentions_inferred': 0,
            'contexts_used': 0,
            'ambiguities_resolved': 0,
            'metaphors_understood': 0
        }
    
    def understand_text(self, text: str, context: List[str] = None) -> Dict:
        """
        Comprehensive text understanding
        
        Returns:
            Understanding results
        """
        # Parse semantic structure
        parse = self.semantic_parsing.parse(text)
        self.stats['texts_parsed'] += 1
        
        # Infer intention
        inference = self.pragmatic_inference.infer_intention(text, {'previous_intention': None})
        self.stats['intentions_inferred'] += 1
        
        # Understand in context
        contextual = self.contextual_understanding.understand_in_context(text, context)
        if context:
            self.stats['contexts_used'] += 1
        
        # Resolve ambiguities
        possible_interpretations = self.ambiguity_resolution.detect_ambiguity(text)
        if len(possible_interpretations) > 1:
            resolved = self.ambiguity_resolution.resolve_ambiguity(
                text, possible_interpretations, {'previous_utterances': context or []}
            )
            self.stats['ambiguities_resolved'] += 1
        else:
            resolved = text
        
        # Understand metaphors
        metaphor = self.metaphor_understanding.understand_metaphor(text)
        if metaphor['is_metaphor']:
            self.stats['metaphors_understood'] += 1
        
        return {
            'text': text,
            'semantic_parse': parse,
            'pragmatic_inference': inference,
            'contextual_understanding': contextual,
            'resolved_text': resolved,
            'metaphor_understanding': metaphor
        }
    
    def get_statistics(self) -> Dict:
        """Get language understanding statistics"""
        return self.stats.copy()

