#!/usr/bin/env python3
"""
Comprehensive Demonstration of All Super AGI Phases
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_phase6_creativity():
    """Demonstrate Phase 6: Creativity"""
    print("\n" + "="*60)
    print("PHASE 6: CREATIVITY & INNOVATION")
    print("="*60)
    
    try:
        from Phase6_Creativity.creativity_system import CreativitySystem
        
        system = CreativitySystem()
        ideas = system.generate_ideas(
            np.random.rand(10),
            context={'domain': 'technology'}
        )
        
        print(f"✓ Generated {len(ideas)} creative ideas")
        if ideas:
            print(f"  Example: {ideas[0]['description'][:50]}...")
    except Exception as e:
        print(f"✗ Error: {e}")


def demo_phase7_learning():
    """Demonstrate Phase 7: Advanced Learning"""
    print("\n" + "="*60)
    print("PHASE 7: ADVANCED LEARNING")
    print("="*60)
    
    try:
        from Phase7_AdvancedLearning.meta_learning import MetaLearningSystem
        
        system = MetaLearningSystem()
        examples = [(np.random.rand(5), np.random.rand(3)) for _ in range(5)]
        proficiency = system.few_shot_learn(examples, 'classification')
        
        print(f"✓ Few-shot learning proficiency: {proficiency:.2f}")
    except Exception as e:
        print(f"✗ Error: {e}")


def demo_phase8_reasoning():
    """Demonstrate Phase 8: Advanced Reasoning"""
    print("\n" + "="*60)
    print("PHASE 8: ADVANCED REASONING")
    print("="*60)
    
    try:
        from Phase8_AdvancedReasoning.mathematical_reasoning import MathematicalReasoningSystem
        
        system = MathematicalReasoningSystem()
        # Test equation solving
        result = system.solve_equation("x + 5 = 10")
        print(f"✓ Solved equation: x = {result}")
        
        # Test word problem solving
        word_result = system.solve_word_problem("Janet has 5 apples. She gives away 2 apples. How many apples does she have left?")
        print(f"✓ Solved word problem: {word_result['answer']} (confidence: {word_result['confidence']:.2f})")
    except Exception as e:
        print(f"✗ Error: {e}")


def demo_phase9_strategic():
    """Demonstrate Phase 9: Strategic Planning"""
    print("\n" + "="*60)
    print("PHASE 9: STRATEGIC PLANNING & MULTI-AGENT")
    print("="*60)
    
    try:
        from Phase9_StrategicPlanning.strategic_planning import StrategicPlanningSystem
        
        system = StrategicPlanningSystem()
        goal = system.create_strategic_goal(
            description="Market expansion",
            target_state=np.array([1.0, 0.8]),
            time_horizon=365.0
        )
        
        planning_result = system.plan_for_goal(goal)
        print(f"✓ Created strategic goal: {goal.description}")
        print(f"  Tactical steps: {len(planning_result['tactical_goals'])}")
        print(f"  Scenarios: {len(planning_result['scenarios'])}")
    except Exception as e:
        print(f"✗ Error: {e}")


def demo_phase10_tooluse():
    """Demonstrate Phase 10: Tool Use"""
    print("\n" + "="*60)
    print("PHASE 10: TOOL USE & SELF-IMPROVEMENT")
    print("="*60)
    
    try:
        from Phase10_ToolUse.tool_use import ToolUseSystem
        
        system = ToolUseSystem()
        
        def add_one(x):
            return x + 1
        
        tool = system.register_tool(
            name="AddOne",
            description="Adds one",
            function=add_one,
            capabilities={"arithmetic": 0.9}
        )
        
        tool, result = system.use_tool_for_task(
            {"arithmetic": 0.8},
            5
        )
        
        print(f"✓ Registered tool: {tool.name}")
        print(f"  Tool result: {result}")
    except Exception as e:
        print(f"✗ Error: {e}")


def demo_phase11_language():
    """Demonstrate Phase 11: Language"""
    print("\n" + "="*60)
    print("PHASE 11: LANGUAGE & COMMUNICATION")
    print("="*60)
    
    try:
        from Phase11_Language.language_generation import LanguageGenerationSystem
        
        system = LanguageGenerationSystem()
        text = system.generate_text(
            topic="Artificial Intelligence",
            length=3
        )
        
        print(f"✓ Generated text:")
        print(f"  {text.content[:100]}...")
    except Exception as e:
        print(f"✗ Error: {e}")


def demo_phase12_embodied():
    """Demonstrate Phase 12: Embodied Intelligence"""
    print("\n" + "="*60)
    print("PHASE 12: EMBODIED INTELLIGENCE & TEMPORAL")
    print("="*60)
    
    try:
        from Phase12_Embodied.embodied_intelligence import EmbodiedIntelligenceSystem
        
        system = EmbodiedIntelligenceSystem()
        mapping = system.learn_sensorimotor_mapping(
            np.array([0.5, 0.3]),
            np.array([0.2, 0.4]),
            success=True
        )
        
        print(f"✓ Learned sensorimotor mapping")
        print(f"  Success rate: {mapping.success_rate:.2f}")
    except Exception as e:
        print(f"✗ Error: {e}")


def demo_phase13_safety():
    """Demonstrate Phase 13: Safety"""
    print("\n" + "="*60)
    print("PHASE 13: ROBUSTNESS, SAFETY & EXPLAINABILITY")
    print("="*60)
    
    try:
        from Phase13_Safety.safety_alignment import SafetyAlignmentSystem
        
        system = SafetyAlignmentSystem()
        value = system.register_value(
            name="Safety",
            description="Prioritize safety"
        )
        
        action = {'description': 'Safe operation', 'type': 'safe'}
        result = system.check_action_safety(action)
        
        print(f"✓ Registered value: {value.name}")
        print(f"  Action safe: {result['safe']}")
        print(f"  Action aligned: {result['aligned']}")
    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    """Run all demonstrations"""
    print("\n" + "="*60)
    print("SUPER AGI CAPABILITIES DEMONSTRATION")
    print("="*60)
    
    demos = [
        demo_phase6_creativity,
        demo_phase7_learning,
        demo_phase8_reasoning,
        demo_phase9_strategic,
        demo_phase10_tooluse,
        demo_phase11_language,
        demo_phase12_embodied,
        demo_phase13_safety
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

