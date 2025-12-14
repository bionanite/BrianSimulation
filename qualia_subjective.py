#!/usr/bin/env python3
"""
Qualia and Subjective Experience
Implements phenomenal consciousness, subjective experience modeling, qualia representation, and first-person perspective
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from collections import defaultdict

@dataclass
class Quale:
    """Represents a quale (subjective experience)"""
    quale_id: int
    experience_type: str  # 'visual', 'auditory', 'emotional', etc.
    properties: Dict[str, float]  # property -> intensity
    intensity: float = 1.0
    timestamp: float = 0.0

@dataclass
class SubjectiveExperience:
    """Represents a subjective experience"""
    experience_id: int
    description: str
    qualia: List[int]  # quale_ids
    valence: float = 0.0  # positive/negative
    arousal: float = 0.5  # activation level
    timestamp: float = 0.0

class PhenomenalConsciousness:
    """
    Phenomenal Consciousness
    
    Models phenomenal experiences
    Represents "what it's like" to experience
    """
    
    def __init__(self):
        self.experiences: Dict[int, SubjectiveExperience] = {}
        self.experience_history: List[int] = []
        self.next_experience_id = 0
    
    def create_experience(self,
                         description: str,
                         qualia_ids: List[int],
                         valence: float = 0.0,
                         arousal: float = 0.5) -> SubjectiveExperience:
        """Create a subjective experience"""
        experience = SubjectiveExperience(
            experience_id=self.next_experience_id,
            description=description,
            qualia=qualia_ids.copy(),
            valence=valence,
            arousal=arousal,
            timestamp=time.time()
        )
        
        self.experiences[self.next_experience_id] = experience
        self.experience_history.append(self.next_experience_id)
        self.next_experience_id += 1
        
        # Limit history
        if len(self.experience_history) > 100:
            self.experience_history = self.experience_history[-100:]
        
        return experience
    
    def get_current_experience(self) -> Optional[SubjectiveExperience]:
        """Get most recent experience"""
        if not self.experience_history:
            return None
        
        recent_id = self.experience_history[-1]
        return self.experiences.get(recent_id)
    
    def get_experience_summary(self) -> Dict:
        """Get summary of experiences"""
        return {
            'total_experiences': len(self.experiences),
            'recent_experiences': len(self.experience_history),
            'avg_valence': np.mean([e.valence for e in self.experiences.values()]) if self.experiences else 0.0,
            'avg_arousal': np.mean([e.arousal for e in self.experiences.values()]) if self.experiences else 0.0
        }

class QualiaRepresentation:
    """
    Qualia Representation
    
    Represents qualia (subjective qualities)
    Models "what it's like" properties
    """
    
    def __init__(self):
        self.qualia: Dict[int, Quale] = {}
        self.next_quale_id = 0
    
    def create_quale(self,
                    experience_type: str,
                    properties: Dict[str, float],
                    intensity: float = 1.0) -> Quale:
        """Create a quale"""
        quale = Quale(
            quale_id=self.next_quale_id,
            experience_type=experience_type,
            properties=properties.copy(),
            intensity=intensity,
            timestamp=time.time()
        )
        
        self.qualia[self.next_quale_id] = quale
        self.next_quale_id += 1
        
        return quale
    
    def get_quale(self, quale_id: int) -> Optional[Quale]:
        """Get a quale by ID"""
        return self.qualia.get(quale_id)
    
    def get_qualia_by_type(self, experience_type: str) -> List[Quale]:
        """Get all qualia of a type"""
        return [q for q in self.qualia.values() if q.experience_type == experience_type]
    
    def compute_quale_similarity(self,
                                quale1_id: int,
                                quale2_id: int) -> float:
        """Compute similarity between two qualia"""
        quale1 = self.qualia.get(quale1_id)
        quale2 = self.qualia.get(quale2_id)
        
        if not quale1 or not quale2:
            return 0.0
        
        # Compare properties
        common_properties = set(quale1.properties.keys()) & set(quale2.properties.keys())
        if not common_properties:
            return 0.0
        
        # Compute similarity
        similarities = []
        for prop in common_properties:
            val1 = quale1.properties[prop]
            val2 = quale2.properties[prop]
            similarity = 1.0 - abs(val1 - val2)
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def get_qualia_summary(self) -> Dict:
        """Get summary of qualia"""
        type_counts = defaultdict(int)
        for quale in self.qualia.values():
            type_counts[quale.experience_type] += 1
        
        return {
            'total_qualia': len(self.qualia),
            'types': dict(type_counts),
            'avg_intensity': np.mean([q.intensity for q in self.qualia.values()]) if self.qualia else 0.0
        }

class FirstPersonPerspective:
    """
    First-Person Perspective
    
    Models first-person perspective
    Maintains "I" perspective
    """
    
    def __init__(self):
        self.perspective_history: List[Dict] = []
        self.current_perspective: Dict = {}
    
    def update_perspective(self,
                          viewpoint: Dict[str, float]):
        """Update first-person perspective"""
        self.current_perspective = viewpoint.copy()
        self.perspective_history.append(viewpoint.copy())
        
        # Limit history
        if len(self.perspective_history) > 100:
            self.perspective_history = self.perspective_history[-100:]
    
    def get_current_perspective(self) -> Dict[str, float]:
        """Get current first-person perspective"""
        return self.current_perspective.copy()
    
    def compute_perspective_continuity(self) -> float:
        """Compute continuity of perspective"""
        if len(self.perspective_history) < 2:
            return 1.0
        
        # Compare recent perspectives
        recent = self.perspective_history[-1]
        previous = self.perspective_history[-2]
        
        # Compute overlap
        common_keys = set(recent.keys()) & set(previous.keys())
        if not common_keys:
            return 0.0
        
        # Compute similarity
        similarities = []
        for key in common_keys:
            val1 = recent[key]
            val2 = previous[key]
            similarity = 1.0 - abs(val1 - val2)
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def get_perspective_summary(self) -> Dict:
        """Get perspective summary"""
        return {
            'perspective_history_length': len(self.perspective_history),
            'current_perspective_keys': len(self.current_perspective),
            'continuity': self.compute_perspective_continuity()
        }

class SubjectiveExperienceModeling:
    """
    Subjective Experience Modeling
    
    Models subjective experiences
    Creates rich experiential representations
    """
    
    def __init__(self):
        self.experience_models: Dict[str, Dict] = {}  # experience_type -> model
    
    def create_experience_model(self,
                               experience_type: str,
                               model: Dict):
        """Create a model for an experience type"""
        self.experience_models[experience_type] = model.copy()
    
    def model_experience(self,
                        experience_type: str,
                        input_data: Dict[str, float]) -> Dict[str, float]:
        """
        Model an experience based on input
        
        Returns:
            Modeled experience properties
        """
        if experience_type not in self.experience_models:
            return {}
        
        model = self.experience_models[experience_type]
        
        # Simple modeling: map input to experience properties
        modeled = {}
        for key, value in input_data.items():
            if key in model:
                # Transform according to model
                modeled[key] = value * model[key]
            else:
                modeled[key] = value
        
        return modeled
    
    def get_experience_model_summary(self) -> Dict:
        """Get summary of experience models"""
        return {
            'modeled_types': list(self.experience_models.keys()),
            'num_models': len(self.experience_models)
        }

class QualiaSubjectiveManager:
    """
    Manages all qualia and subjective experience mechanisms
    """
    
    def __init__(self):
        self.phenomenal_consciousness = PhenomenalConsciousness()
        self.qualia_representation = QualiaRepresentation()
        self.first_person_perspective = FirstPersonPerspective()
        self.experience_modeling = SubjectiveExperienceModeling()
    
    def create_experience(self,
                         description: str,
                         experience_type: str,
                         properties: Dict[str, float],
                         valence: float = 0.0,
                         arousal: float = 0.5) -> SubjectiveExperience:
        """Create a subjective experience"""
        # Create qualia
        quale = self.qualia_representation.create_quale(experience_type, properties)
        
        # Create experience
        experience = self.phenomenal_consciousness.create_experience(
            description, [quale.quale_id], valence, arousal
        )
        
        return experience
    
    def update_perspective(self, viewpoint: Dict[str, float]):
        """Update first-person perspective"""
        self.first_person_perspective.update_perspective(viewpoint)
    
    def get_statistics(self) -> Dict:
        """Get statistics about qualia and subjective experience"""
        experience_summary = self.phenomenal_consciousness.get_experience_summary()
        qualia_summary = self.qualia_representation.get_qualia_summary()
        perspective_summary = self.first_person_perspective.get_perspective_summary()
        modeling_summary = self.experience_modeling.get_experience_model_summary()
        
        return {
            **experience_summary,
            **qualia_summary,
            **perspective_summary,
            **modeling_summary
        }

