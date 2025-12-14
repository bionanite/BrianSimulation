#!/usr/bin/env python3
"""
Curriculum Learning - Phase 7.3
Implements difficulty progression, adaptive curriculum, skill prerequisites,
scaffolding, and mastery detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from collections import defaultdict

# Import dependencies
try:
    from goal_setting_planning import GoalSettingPlanning
    from intrinsic_motivation import IntrinsicMotivationManager
    from metacognition import MetacognitiveMonitoring
except ImportError:
    GoalSettingPlanning = None
    IntrinsicMotivationManager = None
    MetacognitiveMonitoring = None


@dataclass
class Skill:
    """Represents a skill to learn"""
    skill_id: int
    name: str
    difficulty: float  # 0-1, higher = more difficult
    prerequisites: List[int]  # List of prerequisite skill IDs
    mastery_threshold: float = 0.8
    current_mastery: float = 0.0
    attempts: int = 0
    successes: int = 0


@dataclass
class CurriculumItem:
    """Represents an item in the curriculum"""
    item_id: int
    skill_id: int
    difficulty: float
    order: int
    completed: bool = False
    performance: float = 0.0


class DifficultyProgression:
    """
    Difficulty Progression
    
    Orders tasks by difficulty
    Creates progressive learning path
    """
    
    def __init__(self,
                 progression_strategy: str = 'linear'):
        self.progression_strategy = progression_strategy
        self.difficulty_history: List[float] = []
    
    def order_by_difficulty(self, skills: List[Skill]) -> List[Skill]:
        """
        Order skills by difficulty
        
        Returns:
            Ordered list of skills
        """
        # Sort by difficulty
        ordered = sorted(skills, key=lambda s: s.difficulty)
        return ordered
    
    def compute_next_difficulty(self, current_difficulty: float, performance: float) -> float:
        """
        Compute next difficulty level based on performance
        
        Returns:
            Next difficulty level
        """
        if self.progression_strategy == 'linear':
            # Linear progression
            if performance > 0.8:
                next_difficulty = min(1.0, current_difficulty + 0.1)
            elif performance > 0.6:
                next_difficulty = current_difficulty
            else:
                next_difficulty = max(0.0, current_difficulty - 0.1)
        
        elif self.progression_strategy == 'exponential':
            # Exponential progression
            if performance > 0.8:
                next_difficulty = min(1.0, current_difficulty * 1.2)
            elif performance > 0.6:
                next_difficulty = current_difficulty
            else:
                next_difficulty = max(0.0, current_difficulty * 0.9)
        
        else:
            next_difficulty = current_difficulty
        
        self.difficulty_history.append(next_difficulty)
        return next_difficulty


class AdaptiveCurriculum:
    """
    Adaptive Curriculum
    
    Adjusts curriculum based on performance
    Personalizes learning path
    """
    
    def __init__(self,
                 adaptation_rate: float = 0.2,
                 performance_window: int = 5):
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window
        self.performance_history: List[float] = []
        self.curriculum_adjustments: List[Dict] = []
    
    def adapt_curriculum(self,
                        curriculum: List[CurriculumItem],
                        recent_performance: List[float]) -> List[CurriculumItem]:
        """
        Adapt curriculum based on performance
        
        Returns:
            Adapted curriculum
        """
        if not recent_performance:
            return curriculum
        
        avg_performance = np.mean(recent_performance[-self.performance_window:])
        
        adapted_curriculum = []
        
        for item in curriculum:
            if item.completed:
                adapted_curriculum.append(item)
                continue
            
            # Adjust difficulty based on performance
            if avg_performance > 0.8:
                # High performance: increase difficulty
                item.difficulty = min(1.0, item.difficulty + self.adaptation_rate)
            elif avg_performance < 0.5:
                # Low performance: decrease difficulty
                item.difficulty = max(0.0, item.difficulty - self.adaptation_rate)
            
            adapted_curriculum.append(item)
        
        # Reorder by new difficulty
        adapted_curriculum.sort(key=lambda x: x.difficulty)
        
        # Update order
        for i, item in enumerate(adapted_curriculum):
            item.order = i
        
        self.curriculum_adjustments.append({
            'avg_performance': avg_performance,
            'timestamp': time.time()
        })
        
        return adapted_curriculum
    
    def update_performance(self, performance: float):
        """Update performance history"""
        self.performance_history.append(performance)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]


class SkillPrerequisites:
    """
    Skill Prerequisites
    
    Ensures prerequisites are learned before advanced skills
    Builds skill dependency graph
    """
    
    def __init__(self):
        self.prerequisite_graph: Dict[int, List[int]] = {}  # skill_id -> prerequisite_ids
        self.mastery_status: Dict[int, bool] = {}  # skill_id -> mastered
    
    def add_prerequisite(self, skill_id: int, prerequisite_id: int):
        """Add a prerequisite relationship"""
        if skill_id not in self.prerequisite_graph:
            self.prerequisite_graph[skill_id] = []
        
        if prerequisite_id not in self.prerequisite_graph[skill_id]:
            self.prerequisite_graph[skill_id].append(prerequisite_id)
    
    def check_prerequisites(self, skill_id: int) -> Tuple[bool, List[int]]:
        """
        Check if prerequisites are met
        
        Returns:
            (all_met, missing_prerequisites)
        """
        if skill_id not in self.prerequisite_graph:
            return True, []
        
        prerequisites = self.prerequisite_graph[skill_id]
        missing = [prereq_id for prereq_id in prerequisites 
                  if not self.mastery_status.get(prereq_id, False)]
        
        return len(missing) == 0, missing
    
    def mark_mastered(self, skill_id: int):
        """Mark a skill as mastered"""
        self.mastery_status[skill_id] = True
    
    def get_available_skills(self, all_skills: List[Skill]) -> List[Skill]:
        """
        Get skills that can be learned (prerequisites met)
        
        Returns:
            List of available skills
        """
        available = []
        
        for skill in all_skills:
            prerequisites_met, _ = self.check_prerequisites(skill.skill_id)
            if prerequisites_met:
                available.append(skill)
        
        return available


class Scaffolding:
    """
    Scaffolding
    
    Provides support structures for learning
    Gradually removes support as mastery increases
    """
    
    def __init__(self,
                 initial_support: float = 0.8,
                 support_decay_rate: float = 0.1):
        self.initial_support = initial_support
        self.support_decay_rate = support_decay_rate
        self.current_support: Dict[int, float] = {}  # skill_id -> support_level
    
    def get_support_level(self, skill_id: int, mastery: float) -> float:
        """
        Get current support level for a skill
        
        Returns:
            Support level (0-1, higher = more support)
        """
        if skill_id not in self.current_support:
            self.current_support[skill_id] = self.initial_support
        
        # Reduce support as mastery increases
        support = self.current_support[skill_id] * (1.0 - mastery)
        support = max(0.0, min(1.0, support))
        
        return support
    
    def reduce_support(self, skill_id: int):
        """Reduce support for a skill"""
        if skill_id in self.current_support:
            self.current_support[skill_id] = max(0.0, 
                self.current_support[skill_id] - self.support_decay_rate)
    
    def provide_hints(self, skill_id: int, support_level: float) -> List[str]:
        """
        Provide hints based on support level
        
        Returns:
            List of hints
        """
        hints = []
        
        if support_level > 0.7:
            hints.append("Strong hint: Try this approach...")
        elif support_level > 0.4:
            hints.append("Moderate hint: Consider this...")
        elif support_level > 0.1:
            hints.append("Light hint: Remember that...")
        
        return hints


class MasteryDetection:
    """
    Mastery Detection
    
    Determines when a skill is mastered
    Tracks learning progress
    """
    
    def __init__(self,
                 mastery_threshold: float = 0.8,
                 consistency_required: int = 3):
        self.mastery_threshold = mastery_threshold
        self.consistency_required = consistency_required
        self.performance_history: Dict[int, List[float]] = defaultdict(list)
    
    def update_performance(self, skill_id: int, performance: float):
        """Update performance for a skill"""
        self.performance_history[skill_id].append(performance)
        
        # Limit history
        if len(self.performance_history[skill_id]) > 20:
            self.performance_history[skill_id] = self.performance_history[skill_id][-20:]
    
    def check_mastery(self, skill_id: int) -> Tuple[bool, float]:
        """
        Check if skill is mastered
        
        Returns:
            (is_mastered, mastery_score)
        """
        if skill_id not in self.performance_history:
            return False, 0.0
        
        performances = self.performance_history[skill_id]
        if not performances:
            return False, 0.0
        
        # Compute mastery score
        recent_performances = performances[-self.consistency_required:]
        avg_performance = np.mean(recent_performances)
        
        # Check consistency
        if len(recent_performances) >= self.consistency_required:
            std_performance = np.std(recent_performances)
            consistency = 1.0 / (1.0 + std_performance)
        else:
            consistency = 0.5
        
        mastery_score = avg_performance * 0.7 + consistency * 0.3
        is_mastered = mastery_score >= self.mastery_threshold
        
        return is_mastered, mastery_score


class CurriculumLearningSystem:
    """
    Curriculum Learning System Manager
    
    Integrates all curriculum learning components
    """
    
    def __init__(self,
                 brain_system=None,
                 goal_setting: Optional[GoalSettingPlanning] = None,
                 intrinsic_motivation: Optional[IntrinsicMotivationManager] = None,
                 metacognition: Optional[MetacognitiveMonitoring] = None):
        self.brain_system = brain_system
        
        # Initialize components
        self.difficulty_progression = DifficultyProgression()
        self.adaptive_curriculum = AdaptiveCurriculum()
        self.skill_prerequisites = SkillPrerequisites()
        self.scaffolding = Scaffolding()
        self.mastery_detection = MasteryDetection()
        
        # Integration with existing systems
        self.goal_setting = goal_setting
        self.intrinsic_motivation = intrinsic_motivation
        self.metacognition = metacognition
        
        # Skills and curriculum tracking
        self.skills: Dict[int, Skill] = {}
        self.curriculum: List[CurriculumItem] = []
        self.next_skill_id = 0
        self.next_item_id = 0
        
        # Statistics
        self.stats = {
            'skills_learned': 0,
            'skills_mastered': 0,
            'curriculum_items_completed': 0,
            'average_mastery': 0.0
        }
    
    def add_skill(self,
                 name: str,
                 difficulty: float,
                 prerequisites: Optional[List[int]] = None) -> Skill:
        """Add a skill to the curriculum"""
        skill = Skill(
            skill_id=self.next_skill_id,
            name=name,
            difficulty=difficulty,
            prerequisites=prerequisites or []
        )
        
        self.skills[self.next_skill_id] = skill
        
        # Add prerequisites
        for prereq_id in skill.prerequisites:
            self.skill_prerequisites.add_prerequisite(skill.skill_id, prereq_id)
        
        self.next_skill_id += 1
        return skill
    
    def build_curriculum(self) -> List[CurriculumItem]:
        """Build curriculum from skills"""
        # Get available skills (prerequisites met)
        available_skills = self.skill_prerequisites.get_available_skills(list(self.skills.values()))
        
        # Order by difficulty
        ordered_skills = self.difficulty_progression.order_by_difficulty(available_skills)
        
        # Create curriculum items
        curriculum = []
        for i, skill in enumerate(ordered_skills):
            item = CurriculumItem(
                item_id=self.next_item_id,
                skill_id=skill.skill_id,
                difficulty=skill.difficulty,
                order=i
            )
            curriculum.append(item)
            self.next_item_id += 1
        
        self.curriculum = curriculum
        return curriculum
    
    def learn_skill(self, skill_id: int, performance: float) -> Dict:
        """
        Learn a skill and update curriculum
        
        Returns:
            Learning results
        """
        if skill_id not in self.skills:
            return {'success': False, 'error': 'Skill not found'}
        
        skill = self.skills[skill_id]
        
        # Update performance
        skill.attempts += 1
        if performance > 0.7:
            skill.successes += 1
        
        skill.current_mastery = skill.successes / max(1, skill.attempts)
        
        # Update mastery detection
        self.mastery_detection.update_performance(skill_id, performance)
        
        # Check mastery
        is_mastered, mastery_score = self.mastery_detection.check_mastery(skill_id)
        
        if is_mastered:
            skill.current_mastery = mastery_score
            self.skill_prerequisites.mark_mastered(skill_id)
            self.stats['skills_mastered'] += 1
        
        # Reduce scaffolding
        self.scaffolding.reduce_support(skill_id)
        
        # Update curriculum performance
        for item in self.curriculum:
            if item.skill_id == skill_id:
                item.performance = performance
                if is_mastered:
                    item.completed = True
                    self.stats['curriculum_items_completed'] += 1
        
        # Adapt curriculum
        recent_performances = [item.performance for item in self.curriculum if item.performance > 0]
        if recent_performances:
            self.adaptive_curriculum.update_performance(np.mean(recent_performances[-5:]))
            self.curriculum = self.adaptive_curriculum.adapt_curriculum(self.curriculum, recent_performances)
        
        # Update statistics
        self.stats['skills_learned'] += 1
        if self.skills:
            self.stats['average_mastery'] = np.mean([s.current_mastery for s in self.skills.values()])
        
        return {
            'success': True,
            'skill_id': skill_id,
            'mastery': mastery_score,
            'is_mastered': is_mastered,
            'support_level': self.scaffolding.get_support_level(skill_id, mastery_score)
        }
    
    def get_next_skill(self) -> Optional[Skill]:
        """Get next skill to learn"""
        # Get available skills
        available_skills = self.skill_prerequisites.get_available_skills(list(self.skills.values()))
        
        if not available_skills:
            return None
        
        # Get skills not yet mastered, ordered by difficulty
        unmastered = [s for s in available_skills if s.current_mastery < s.mastery_threshold]
        if not unmastered:
            return None
        
        ordered = self.difficulty_progression.order_by_difficulty(unmastered)
        return ordered[0] if ordered else None
    
    def get_statistics(self) -> Dict:
        """Get curriculum learning statistics"""
        return self.stats.copy()

