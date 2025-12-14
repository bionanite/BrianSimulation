#!/usr/bin/env python3
"""
Language Integration - Integrate with LLM APIs for hybrid architecture
Uses LLMs as "language cortex" while brain handles reasoning/planning
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any
import numpy as np

# Optional imports for LLM APIs
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None

from language_processor import LanguageProcessor


class LLMIntegration:
    """Integration with LLM APIs for hybrid architecture"""
    
    def __init__(self, 
                 provider: str = "openai",
                 model: Optional[str] = None,
                 use_cache: bool = True):
        """
        Initialize LLM integration
        
        Args:
            provider: LLM provider ("openai", "anthropic", or "none")
            model: Model name (None = use default)
            use_cache: Whether to cache responses
        """
        self.provider = provider.lower()
        self.use_cache = use_cache
        self.cache: Dict[str, str] = {}
        
        # Initialize language processor
        self.lang_processor = LanguageProcessor()
        
        # Initialize provider
        if self.provider == "openai":
            self._init_openai(model)
        elif self.provider == "anthropic":
            self._init_anthropic(model)
        else:
            self.provider = "none"
            self.client = None
            print("⚠️  No LLM provider configured. Using brain-only mode.")
    
    def _init_openai(self, model: Optional[str]):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            print("⚠️  OpenAI library not available. Install with: pip install openai")
            self.provider = "none"
            self.client = None
            return
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OPENAI_API_KEY not set. Set environment variable to use OpenAI.")
            self.provider = "none"
            self.client = None
            return
        
        try:
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model or "gpt-4"
            print(f"✅ OpenAI initialized with model: {self.model}")
        except Exception as e:
            print(f"⚠️  Error initializing OpenAI: {e}")
            self.provider = "none"
            self.client = None
    
    def _init_anthropic(self, model: Optional[str]):
        """Initialize Anthropic client"""
        if not ANTHROPIC_AVAILABLE:
            print("⚠️  Anthropic library not available. Install with: pip install anthropic")
            self.provider = "none"
            self.client = None
            return
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("⚠️  ANTHROPIC_API_KEY not set. Set environment variable to use Anthropic.")
            self.provider = "none"
            self.client = None
            return
        
        try:
            self.client = Anthropic(api_key=api_key)
            self.model = model or "claude-3-opus-20240229"
            print(f"✅ Anthropic initialized with model: {self.model}")
        except Exception as e:
            print(f"⚠️  Error initializing Anthropic: {e}")
            self.provider = "none"
            self.client = None
    
    def _cache_key(self, text: str, context: Optional[str] = None) -> str:
        """Generate cache key"""
        key = text
        if context:
            key = f"{context}|||{text}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def encode_for_llm(self, brain_pattern: np.ndarray, context: Optional[str] = None) -> str:
        """
        Encode brain pattern to text for LLM
        
        Args:
            brain_pattern: Neural pattern from brain
            context: Optional context/question
            
        Returns:
            Text representation for LLM
        """
        # Convert pattern to text
        text = self.lang_processor.pattern_to_text(brain_pattern)
        
        # Enhance with context if available
        if context:
            # Extract keywords from context
            keywords = self.lang_processor.extract_keywords(context, top_k=3)
            keyword_text = ", ".join([word for word, _ in keywords])
            text = f"Context: {keyword_text}\nPattern interpretation: {text}"
        
        return text
    
    def decode_from_llm(self, llm_response: str) -> np.ndarray:
        """
        Decode LLM response to neural pattern
        
        Args:
            llm_response: Text response from LLM
            
        Returns:
            Neural pattern
        """
        return self.lang_processor.text_to_pattern(llm_response)
    
    def call_llm(self, 
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 max_tokens: int = 500) -> str:
        """
        Call LLM API
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (instructions)
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM response text
        """
        # Check cache
        if self.use_cache:
            cache_key = self._cache_key(prompt, system_prompt)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        if self.provider == "none":
            # Fallback: return simple response
            return f"[Brain-only mode] Processed: {prompt[:100]}"
        
        try:
            if self.provider == "openai":
                response = self._call_openai(prompt, system_prompt, max_tokens)
            elif self.provider == "anthropic":
                response = self._call_anthropic(prompt, system_prompt, max_tokens)
            else:
                response = f"[Unknown provider] {prompt[:100]}"
            
            # Cache response
            if self.use_cache:
                cache_key = self._cache_key(prompt, system_prompt)
                self.cache[cache_key] = response
            
            return response
            
        except Exception as e:
            print(f"⚠️  Error calling LLM: {e}")
            return f"[Error] {str(e)}"
    
    def _call_openai(self, prompt: str, system_prompt: Optional[str], max_tokens: int) -> str:
        """Call OpenAI API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str, system_prompt: Optional[str], max_tokens: int) -> str:
        """Call Anthropic API"""
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt or "You are a helpful assistant.",
            messages=messages
        )
        
        return response.content[0].text
    
    def hybrid_process(self,
                      question: str,
                      brain_system,
                      use_brain_reasoning: bool = True) -> Dict[str, Any]:
        """
        Hybrid processing: Brain + LLM
        
        Architecture:
        Question → Brain Pattern Recognition → LLM Language Processing → 
        Brain Reasoning → LLM Response Formatting → Answer
        
        Args:
            question: Input question
            brain_system: FinalEnhancedBrain instance
            use_brain_reasoning: Whether to use brain for reasoning
            
        Returns:
            Dictionary with processing results
        """
        results = {
            'question': question,
            'provider': self.provider,
            'stages': {}
        }
        
        # Stage 1: Brain pattern recognition
        question_pattern = self.lang_processor.text_to_pattern(question)
        pattern_result = brain_system.enhanced_pattern_recognition(question_pattern)
        results['stages']['pattern_recognition'] = {
            'pattern': question_pattern.tolist(),
            'confidence': pattern_result.get('confidence', 0.0),
            'features_detected': pattern_result.get('features_detected', 0)
        }
        
        # Stage 2: LLM language understanding
        if self.provider != "none":
            llm_prompt = f"Question: {question}\n\nAnalyze this question and extract key concepts."
            llm_analysis = self.call_llm(llm_prompt, 
                                        system_prompt="You are a language understanding system.")
            results['stages']['llm_analysis'] = llm_analysis
            
            # Convert LLM analysis to pattern
            llm_pattern = self.decode_from_llm(llm_analysis)
        else:
            llm_pattern = question_pattern
            results['stages']['llm_analysis'] = "[Brain-only mode]"
        
        # Stage 3: Brain reasoning (if enabled)
        if use_brain_reasoning and hasattr(brain_system, 'reasoning'):
            reasoning_context = {
                'sensory_input': question_pattern,
                'llm_analysis': llm_pattern,
                'pattern_result': pattern_result
            }
            reasoning_result = brain_system.reasoning(reasoning_context)
            results['stages']['brain_reasoning'] = {
                'conclusion': reasoning_result.get('logical_conclusion', ''),
                'confidence': reasoning_result.get('confidence', 0.0),
                'plan_quality': reasoning_result.get('plan_quality', 0.0)
            }
            reasoning_text = reasoning_result.get('logical_conclusion', '')
        else:
            reasoning_text = question
        
        # Stage 4: LLM response generation
        if self.provider != "none":
            response_prompt = f"""Question: {question}

Analysis: {results['stages'].get('llm_analysis', '')}

Reasoning: {reasoning_text}

Based on the above, provide a clear, accurate answer."""
            
            llm_response = self.call_llm(response_prompt,
                                        system_prompt="You are a helpful assistant that provides accurate answers.")
            results['stages']['llm_response'] = llm_response
            final_answer = llm_response
        else:
            # Brain-only: use reasoning result
            final_answer = reasoning_text if use_brain_reasoning else self.lang_processor.pattern_to_text(question_pattern)
            results['stages']['llm_response'] = "[Brain-only mode]"
        
        results['answer'] = final_answer
        
        # Convert answer back to pattern for brain storage
        answer_pattern = self.lang_processor.text_to_pattern(final_answer)
        results['answer_pattern'] = answer_pattern.tolist()
        
        return results
    
    def batch_hybrid_process(self,
                            questions: List[str],
                            brain_system,
                            use_brain_reasoning: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch
        
        Args:
            questions: List of questions
            brain_system: FinalEnhancedBrain instance
            use_brain_reasoning: Whether to use brain reasoning
            
        Returns:
            List of processing results
        """
        results = []
        for question in questions:
            result = self.hybrid_process(question, brain_system, use_brain_reasoning)
            results.append(result)
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.cache),
            'provider': self.provider,
            'model': getattr(self, 'model', 'none')
        }

