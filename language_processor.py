#!/usr/bin/env python3
"""
Language Processor - Text understanding and generation interface
Converts between natural language and neural patterns
"""

import numpy as np
import re
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import hashlib


class LanguageProcessor:
    """Process natural language to/from neural patterns"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 100):
        """
        Initialize language processor
        
        Args:
            vocab_size: Maximum vocabulary size
            embedding_dim: Dimension of word embeddings/patterns
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Vocabulary mapping
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        
        # Word embeddings (simple hash-based initially)
        self.word_embeddings: Dict[str, np.ndarray] = {}
        
        # Sentence patterns cache
        self.sentence_patterns: Dict[str, np.ndarray] = {}
    
    def _hash_to_pattern(self, text: str, dim: int) -> np.ndarray:
        """Convert text to pattern using hash-based encoding"""
        # Create deterministic hash-based pattern
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to pattern
        pattern = np.zeros(dim, dtype=np.float32)
        for i, byte in enumerate(hash_bytes[:dim]):
            pattern[i] = byte / 255.0
        
        # Fill remaining if needed
        if len(hash_bytes) < dim:
            extended_hash = hashlib.sha256(text.encode()).digest()
            for i, byte in enumerate(extended_hash[:dim - len(hash_bytes)]):
                pattern[len(hash_bytes) + i] = byte / 255.0
        
        return pattern
    
    def _get_word_embedding(self, word: str) -> np.ndarray:
        """Get or create word embedding"""
        if word not in self.word_embeddings:
            self.word_embeddings[word] = self._hash_to_pattern(word.lower(), self.embedding_dim)
        return self.word_embeddings[word]
    
    def text_to_pattern(self, text: str, max_length: Optional[int] = None) -> np.ndarray:
        """
        Convert text to neural pattern
        
        Args:
            text: Input text
            max_length: Maximum pattern length (None = use embedding_dim)
            
        Returns:
            Neural pattern array
        """
        if max_length is None:
            max_length = self.embedding_dim
        
        # Tokenize
        words = self.tokenize(text)
        
        if not words:
            return np.zeros(max_length, dtype=np.float32)
        
        # Get word embeddings
        word_embeddings = [self._get_word_embedding(word) for word in words]
        
        # Combine embeddings (average pooling)
        if word_embeddings:
            combined = np.mean(word_embeddings, axis=0)
        else:
            combined = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Add sentence-level features
        sentence_features = self._extract_sentence_features(text)
        
        # Combine word-level and sentence-level features
        pattern = np.concatenate([
            combined[:max_length - len(sentence_features)],
            sentence_features
        ])[:max_length]
        
        # Pad if needed
        if len(pattern) < max_length:
            pattern = np.pad(pattern, (0, max_length - len(pattern)), mode='constant')
        
        return pattern.astype(np.float32)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple tokenization: lowercase, split on whitespace and punctuation
        text = text.lower()
        # Remove special characters but keep alphanumeric
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Update vocabulary
        for word in words:
            self.word_freq[word] += 1
            if word not in self.word_to_id:
                word_id = len(self.word_to_id)
                self.word_to_id[word] = word_id
                self.id_to_word[word_id] = word
        
        return words
    
    def _extract_sentence_features(self, text: str) -> np.ndarray:
        """Extract sentence-level features"""
        features = []
        
        # Length features
        features.append(min(len(text) / 500.0, 1.0))  # Normalized length
        features.append(min(len(text.split()) / 100.0, 1.0))  # Word count
        
        # Question features
        is_question = 1.0 if '?' in text else 0.0
        features.append(is_question)
        
        # Exclamation features
        is_exclamation = 1.0 if '!' in text else 0.0
        features.append(is_exclamation)
        
        # Number features
        numbers = re.findall(r'\d+', text)
        has_numbers = 1.0 if numbers else 0.0
        features.append(has_numbers)
        
        # Capitalization (presence of capitals)
        has_capitals = 1.0 if any(c.isupper() for c in text) else 0.0
        features.append(has_capitals)
        
        return np.array(features, dtype=np.float32)
    
    def pattern_to_text(self, pattern: np.ndarray, method: str = "similarity") -> str:
        """
        Convert neural pattern back to text
        
        Args:
            pattern: Neural pattern array
            method: Method to use ("similarity" or "features")
            
        Returns:
            Text representation
        """
        if method == "similarity":
            return self._pattern_to_text_similarity(pattern)
        else:
            return self._pattern_to_text_features(pattern)
    
    def _pattern_to_text_similarity(self, pattern: np.ndarray) -> str:
        """Convert pattern to text using similarity to known patterns"""
        if not self.sentence_patterns:
            return self._pattern_to_text_features(pattern)
        
        # Find most similar known pattern
        best_match = None
        best_similarity = -1.0
        
        for text, known_pattern in self.sentence_patterns.items():
            if len(known_pattern) == len(pattern):
                similarity = np.dot(pattern, known_pattern) / (
                    np.linalg.norm(pattern) * np.linalg.norm(known_pattern) + 1e-8
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = text
        
        if best_match and best_similarity > 0.5:
            return best_match
        
        return self._pattern_to_text_features(pattern)
    
    def _pattern_to_text_features(self, pattern: np.ndarray) -> str:
        """Convert pattern to text using feature extraction"""
        # Extract semantic meaning from pattern features
        
        # Check if it's a question (high value in question feature position)
        # This is a simplified heuristic
        if len(pattern) > 5:
            # Assume question feature is near the end
            question_score = pattern[-5] if len(pattern) > 5 else 0.0
            
            if question_score > 0.5:
                # Generate question-like response
                if np.max(pattern) > 0.7:
                    return "positive response"
                elif np.mean(pattern) > 0.5:
                    return "moderate response"
                else:
                    return "negative response"
        
        # General pattern interpretation
        max_val = np.max(pattern)
        mean_val = np.mean(pattern)
        std_val = np.std(pattern)
        
        if max_val > 0.8:
            return "strong positive"
        elif mean_val > 0.6:
            return "positive"
        elif mean_val > 0.4:
            return "neutral"
        elif mean_val > 0.2:
            return "negative"
        else:
            return "strong negative"
    
    def encode_question(self, question: str) -> Dict[str, Any]:
        """
        Encode a question for processing
        
        Args:
            question: Question text
            
        Returns:
            Dictionary with encoded question and metadata
        """
        pattern = self.text_to_pattern(question)
        
        # Store pattern for later similarity matching
        self.sentence_patterns[question] = pattern
        
        return {
            'pattern': pattern,
            'text': question,
            'tokens': self.tokenize(question),
            'is_question': '?' in question,
            'length': len(question)
        }
    
    def decode_answer(self, pattern: np.ndarray, context: Optional[str] = None) -> str:
        """
        Decode a pattern to answer text
        
        Args:
            pattern: Neural pattern representing answer
            context: Optional context/question for better decoding
            
        Returns:
            Answer text
        """
        if context and context in self.sentence_patterns:
            # Use context-aware decoding
            context_pattern = self.sentence_patterns[context]
            # Combine patterns
            combined = (pattern + context_pattern) / 2.0
            return self.pattern_to_text(combined)
        
        return self.pattern_to_text(pattern)
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Extract keywords from text
        
        Args:
            text: Input text
            top_k: Number of keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        words = self.tokenize(text)
        
        if not words:
            return []
        
        # Calculate TF-IDF-like scores
        word_scores = {}
        total_words = len(words)
        
        for word in words:
            # Term frequency
            tf = words.count(word) / total_words
            
            # Inverse document frequency (simplified)
            # In a real system, would use document corpus
            idf = 1.0 / (1.0 + self.word_freq.get(word, 1))
            
            word_scores[word] = tf * idf
        
        # Sort by score
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_words[:top_k]
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        pattern1 = self.text_to_pattern(text1)
        pattern2 = self.text_to_pattern(text2)
        
        # Cosine similarity
        dot_product = np.dot(pattern1, pattern2)
        norm1 = np.linalg.norm(pattern1)
        norm2 = np.linalg.norm(pattern2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(np.clip(similarity, 0.0, 1.0))
    
    def generate_response(self, 
                         question_pattern: np.ndarray,
                         brain_output: np.ndarray,
                         question_text: Optional[str] = None) -> str:
        """
        Generate natural language response from brain output
        
        Args:
            question_pattern: Pattern representing the question
            brain_output: Brain's output pattern
            question_text: Original question text (for context)
            
        Returns:
            Generated response text
        """
        # Combine question and brain output
        combined = (question_pattern + brain_output) / 2.0
        
        # Decode to text
        response = self.decode_answer(combined, question_text)
        
        return response
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts in batch
        
        Args:
            texts: List of texts
            
        Returns:
            Array of patterns (shape: [len(texts), pattern_dim])
        """
        patterns = [self.text_to_pattern(text) for text in texts]
        return np.array(patterns)
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get vocabulary statistics"""
        return {
            'vocab_size': len(self.word_to_id),
            'total_words': sum(self.word_freq.values()),
            'unique_words': len(self.word_freq),
            'most_common': self.word_freq.most_common(10),
            'cached_patterns': len(self.sentence_patterns)
        }

