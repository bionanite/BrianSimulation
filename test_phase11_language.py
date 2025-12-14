#!/usr/bin/env python3
"""
Test Phase 11: Language & Communication
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Phase11_Language.deep_language_understanding import DeepLanguageUnderstandingSystem
from Phase11_Language.language_generation import LanguageGenerationSystem
from Phase11_Language.multilingual_system import MultilingualSystem


class TestLanguageUnderstanding(unittest.TestCase):
    """Test language understanding"""
    
    def test_text_understanding(self):
        """Test text understanding"""
        system = DeepLanguageUnderstandingSystem()
        
        result = system.understand_text(
            "What is the weather today?",
            context=[]
        )
        
        self.assertIsNotNone(result)
        self.assertIn('semantic_parse', result)
        self.assertIn('pragmatic_inference', result)
    
    def test_semantic_parsing(self):
        """Test semantic parsing"""
        system = DeepLanguageUnderstandingSystem()
        
        parse = system.semantic_parsing.parse("The cat sat on the mat")
        self.assertIsNotNone(parse)
        self.assertIn('entities', parse.semantic_structure)
    
    def test_ambiguity_resolution(self):
        """Test ambiguity resolution"""
        system = DeepLanguageUnderstandingSystem()
        
        interpretations = system.ambiguity_resolution.detect_ambiguity("I saw the bank")
        self.assertGreater(len(interpretations), 0)


class TestLanguageGeneration(unittest.TestCase):
    """Test language generation"""
    
    def test_text_generation(self):
        """Test text generation"""
        system = LanguageGenerationSystem()
        
        text = system.generate_text(
            topic="Artificial Intelligence",
            style="formal",
            length=3
        )
        
        self.assertIsNotNone(text)
        self.assertIsNotNone(text.content)
    
    def test_dialogue_generation(self):
        """Test dialogue generation"""
        system = LanguageGenerationSystem()
        
        dialogue = system.generate_dialogue(num_turns=3)
        self.assertGreaterEqual(len(dialogue), 3)
    
    def test_explanation_generation(self):
        """Test explanation generation"""
        system = LanguageGenerationSystem()
        
        explanation = system.generate_explanation(
            conclusion="The model predicts class A",
            reasoning_steps=["Step 1", "Step 2"]
        )
        
        self.assertIsNotNone(explanation)
        self.assertIn("predicts", explanation)


class TestMultilingual(unittest.TestCase):
    """Test multilingual system"""
    
    def test_language_registration(self):
        """Test language registration"""
        system = MultilingualSystem()
        
        language = system.register_language(
            name="English",
            code="en",
            vocabulary={"hello": "greeting", "world": "planet"}
        )
        
        self.assertIsNotNone(language)
        self.assertEqual(language.code, "en")
    
    def test_translation(self):
        """Test translation"""
        system = MultilingualSystem()
        
        system.register_language("English", "en", {})
        system.register_language("Spanish", "es", {})
        
        translation = system.translate_text(
            "Hello world",
            from_language="en",
            to_language="es"
        )
        
        self.assertIsNotNone(translation)
        self.assertEqual(translation.source_language, "en")
        self.assertEqual(translation.target_language, "es")


if __name__ == '__main__':
    unittest.main()

