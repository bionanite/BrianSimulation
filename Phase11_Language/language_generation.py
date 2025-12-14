#!/usr/bin/env python3
"""
Language Generation - Phase 11.2
Implements coherent text generation, style adaptation, creative writing,
dialogue generation, and explanation generation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import random

# Import dependencies
try:
    from communication import CommunicationSystem
    from Phase6_Creativity.creativity_system import CreativitySystem
    from advanced_reasoning import AdvancedReasoningSystem
except ImportError:
    CommunicationSystem = None
    CreativitySystem = None
    AdvancedReasoningSystem = None


@dataclass
class GeneratedText:
    """Represents generated text"""
    text_id: int
    content: str
    style: str
    coherence_score: float = 0.5
    creativity_score: float = 0.5
    created_time: float = 0.0


class CoherentTextGeneration:
    """
    Coherent Text Generation
    
    Generates coherent multi-sentence text
    Maintains topic coherence
    """
    
    def __init__(self):
        self.generated_texts: List[GeneratedText] = []
        self.next_text_id = 0
    
    def generate_text(self,
                      topic: str,
                      length: int = 5,
                      context: List[str] = None) -> GeneratedText:
        """
        Generate coherent text
        
        Returns:
            Generated text
        """
        if context is None:
            context = []
        
        sentences = []
        
        # Start with topic sentence
        sentences.append(f"{topic} is an important concept.")
        
        # Generate coherent follow-up sentences
        for i in range(length - 1):
            # Use previous sentence to maintain coherence
            prev_sentence = sentences[-1] if sentences else topic
            
            # Generate next sentence (simplified)
            next_sentence = self._generate_coherent_sentence(prev_sentence, topic)
            sentences.append(next_sentence)
        
        content = " ".join(sentences)
        
        # Compute coherence score
        coherence_score = self._compute_coherence(sentences)
        
        text = GeneratedText(
            text_id=self.next_text_id,
            content=content,
            style='default',
            coherence_score=coherence_score,
            created_time=time.time()
        )
        
        self.generated_texts.append(text)
        self.next_text_id += 1
        
        return text
    
    def _generate_coherent_sentence(self, prev_sentence: str, topic: str) -> str:
        """Generate coherent sentence following previous"""
        # Simplified: use topic words and connectives
        connectives = ['Furthermore', 'Additionally', 'Moreover', 'However', 'Therefore']
        connective = random.choice(connectives)
        
        # Extract key words from topic
        topic_words = topic.split()[:2]
        
        return f"{connective}, {topic} involves {topic_words[0] if topic_words else 'various aspects'}."
    
    def _compute_coherence(self, sentences: List[str]) -> float:
        """Compute coherence score"""
        if len(sentences) < 2:
            return 1.0
        
        # Simple coherence: check for shared words
        shared_words = set(sentences[0].lower().split())
        for sentence in sentences[1:]:
            words = set(sentence.lower().split())
            shared_words = shared_words.intersection(words)
        
        # Coherence based on shared words
        total_words = sum(len(s.split()) for s in sentences)
        coherence = len(shared_words) / max(1, total_words / len(sentences))
        
        return min(1.0, coherence * 2.0)  # Scale up


class StyleAdaptation:
    """
    Style Adaptation
    
    Adapts writing style to context
    Matches style requirements
    """
    
    def __init__(self):
        self.styles: Dict[str, Dict] = {
            'formal': {'tone': 'professional', 'complexity': 'high'},
            'casual': {'tone': 'friendly', 'complexity': 'low'},
            'academic': {'tone': 'scholarly', 'complexity': 'high'},
            'creative': {'tone': 'expressive', 'complexity': 'medium'}
        }
    
    def adapt_style(self,
                   text: str,
                   target_style: str) -> str:
        """
        Adapt text to target style
        
        Returns:
            Style-adapted text
        """
        if target_style not in self.styles:
            return text
        
        style_attrs = self.styles[target_style]
        
        # Apply style transformations (simplified)
        adapted = text
        
        if style_attrs['tone'] == 'professional':
            # Make more formal
            adapted = adapted.replace("don't", "do not")
            adapted = adapted.replace("can't", "cannot")
            adapted = adapted.replace("won't", "will not")
        
        elif style_attrs['tone'] == 'friendly':
            # Make more casual
            adapted = adapted.replace("cannot", "can't")
            adapted = adapted.replace("will not", "won't")
        
        if style_attrs['complexity'] == 'high':
            # Add more complex words (simplified)
            adapted = adapted.replace("important", "significant")
            adapted = adapted.replace("big", "substantial")
        
        return adapted
    
    def detect_style(self, text: str) -> str:
        """Detect style of text"""
        text_lower = text.lower()
        
        # Check for formal markers
        if any(word in text_lower for word in ['cannot', 'will not', 'shall']):
            return 'formal'
        
        # Check for casual markers
        if any(word in text_lower for word in ["can't", "won't", "don't"]):
            return 'casual'
        
        # Check for academic markers
        if any(word in text_lower for word in ['furthermore', 'moreover', 'consequently']):
            return 'academic'
        
        return 'default'


class CreativeWriting:
    """
    Creative Writing
    
    Generates creative and original text
    Uses creativity system
    """
    
    def __init__(self, creativity_system: Optional[CreativitySystem] = None):
        self.creativity_system = creativity_system
        self.creative_texts: List[GeneratedText] = []
    
    def generate_creative_text(self,
                              prompt: str,
                              creativity_level: float = 0.7) -> GeneratedText:
        """
        Generate creative text
        
        Returns:
            Creative text
        """
        # Use creativity system if available
        if self.creativity_system:
            # Generate creative ideas
            ideas = self.creativity_system.generate_ideas(
                np.random.rand(10),  # Random input
                context={'prompt': prompt}
            )
            
            # Convert ideas to text
            creative_content = f"{prompt} {ideas[0]['description'] if ideas else 'unfolds in unexpected ways.'}"
        else:
            # Fallback: simple creative generation
            creative_phrases = [
                "unfolds like a dream",
                "takes unexpected turns",
                "reveals hidden depths",
                "sparkles with possibility"
            ]
            creative_content = f"{prompt} {random.choice(creative_phrases)}."
        
        text = GeneratedText(
            text_id=-1,
            content=creative_content,
            style='creative',
            creativity_score=creativity_level,
            created_time=time.time()
        )
        
        self.creative_texts.append(text)
        return text


class DialogueGeneration:
    """
    Dialogue Generation
    
    Generates natural dialogue
    Maintains conversation flow
    """
    
    def __init__(self):
        self.dialogues: List[List[str]] = []
    
    def generate_response(self,
                        previous_utterances: List[str],
                        context: Dict = None) -> str:
        """
        Generate dialogue response
        
        Returns:
            Generated response
        """
        if context is None:
            context = {}
        
        if not previous_utterances:
            return "Hello! How can I help you?"
        
        # Analyze last utterance
        last_utterance = previous_utterances[-1].lower()
        
        # Generate appropriate response
        if '?' in previous_utterances[-1]:
            # Question detected
            if 'what' in last_utterance:
                response = "That's an interesting question. Let me think about that."
            elif 'how' in last_utterance:
                response = "Here's how that works."
            else:
                response = "I understand your question."
        elif any(word in last_utterance for word in ['thank', 'thanks']):
            response = "You're welcome!"
        else:
            # General response
            response = "I see. Can you tell me more about that?"
        
        return response
    
    def generate_dialogue(self,
                         num_turns: int = 5,
                         initial_prompt: str = "Hello") -> List[str]:
        """
        Generate multi-turn dialogue
        
        Returns:
            List of dialogue turns
        """
        dialogue = [initial_prompt]
        
        for i in range(num_turns - 1):
            response = self.generate_response(dialogue)
            dialogue.append(response)
        
        self.dialogues.append(dialogue)
        return dialogue


class ExplanationGeneration:
    """
    Explanation Generation
    
    Generates explanations of reasoning
    Makes reasoning transparent
    """
    
    def __init__(self, reasoning_system: Optional[AdvancedReasoningSystem] = None):
        self.reasoning_system = reasoning_system
        self.explanations: List[str] = []
    
    def explain_reasoning(self,
                         conclusion: str,
                         reasoning_steps: List[str] = None) -> str:
        """
        Generate explanation of reasoning
        
        Returns:
            Explanation text
        """
        if reasoning_steps is None:
            reasoning_steps = []
        
        explanation_parts = ["Here's how I reached this conclusion:"]
        
        for i, step in enumerate(reasoning_steps, 1):
            explanation_parts.append(f"{i}. {step}")
        
        explanation_parts.append(f"Therefore, {conclusion}")
        
        explanation = " ".join(explanation_parts)
        self.explanations.append(explanation)
        
        return explanation
    
    def explain_decision(self,
                         decision: str,
                         factors: List[str],
                         weights: List[float] = None) -> str:
        """
        Explain a decision
        
        Returns:
            Decision explanation
        """
        if weights is None:
            weights = [1.0 / len(factors)] * len(factors)
        
        explanation_parts = [f"I decided to {decision} based on the following factors:"]
        
        # Sort by weight
        sorted_factors = sorted(zip(factors, weights), key=lambda x: x[1], reverse=True)
        
        for factor, weight in sorted_factors:
            explanation_parts.append(f"- {factor} (weight: {weight:.2f})")
        
        explanation = "\n".join(explanation_parts)
        self.explanations.append(explanation)
        
        return explanation


class LanguageGenerationSystem:
    """
    Language Generation System Manager
    
    Integrates all language generation components
    """
    
    def __init__(self,
                 brain_system=None,
                 communication: Optional[CommunicationSystem] = None,
                 creativity_system: Optional[CreativitySystem] = None,
                 reasoning_system: Optional[AdvancedReasoningSystem] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.coherent_generation = CoherentTextGeneration()
        self.style_adaptation = StyleAdaptation()
        self.creative_writing = CreativeWriting(creativity_system)
        self.dialogue_generation = DialogueGeneration()
        self.explanation_generation = ExplanationGeneration(reasoning_system)
        
        # Integration with existing systems
        self.communication = communication
        self.creativity_system = creativity_system
        self.reasoning_system = reasoning_system
        
        # Statistics
        self.stats = {
            'texts_generated': 0,
            'styles_adapted': 0,
            'creative_texts': 0,
            'dialogues_generated': 0,
            'explanations_generated': 0
        }
    
    def generate_text(self,
                     topic: str,
                     style: str = 'default',
                     length: int = 5,
                     creative: bool = False) -> GeneratedText:
        """Generate text with specified style"""
        if creative:
            text = self.creative_writing.generate_creative_text(topic)
            self.stats['creative_texts'] += 1
        else:
            text = self.coherent_generation.generate_text(topic, length)
        
        # Adapt style
        if style != 'default':
            text.content = self.style_adaptation.adapt_style(text.content, style)
            text.style = style
            self.stats['styles_adapted'] += 1
        
        self.stats['texts_generated'] += 1
        return text
    
    def generate_dialogue(self, num_turns: int = 5) -> List[str]:
        """Generate dialogue"""
        dialogue = self.dialogue_generation.generate_dialogue(num_turns)
        self.stats['dialogues_generated'] += 1
        return dialogue
    
    def generate_explanation(self,
                           conclusion: str,
                           reasoning_steps: List[str] = None) -> str:
        """Generate explanation"""
        explanation = self.explanation_generation.explain_reasoning(conclusion, reasoning_steps)
        self.stats['explanations_generated'] += 1
        return explanation
    
    def get_statistics(self) -> Dict:
        """Get language generation statistics"""
        return self.stats.copy()

