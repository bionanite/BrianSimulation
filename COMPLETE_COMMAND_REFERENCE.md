# Complete Command Reference for Super AGI System

## Overview

This document provides a comprehensive list of all commands to train, test, and demonstrate all capabilities of the Super AGI system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Testing Commands](#testing-commands)
3. [Demonstration Commands](#demonstration-commands)
4. [Training Commands](#training-commands)
5. [Benchmark Commands](#benchmark-commands)
6. [Individual Phase Commands](#individual-phase-commands)

---

## Quick Start

### Run All Tests
```bash
# Make script executable
chmod +x run_all_tests_comprehensive.sh

# Run all tests
./run_all_tests_comprehensive.sh
```

### Run All Demonstrations
```bash
python demo_all_phases.py
```

### Run Benchmarks
```bash
python run_benchmarks.py
```

---

## Testing Commands

### Comprehensive Test Suite

```bash
# Run all tests (comprehensive)
./run_all_tests_comprehensive.sh

# Run original test suite (Phases 6 & 8)
./run_all_tests.sh
```

### Individual Phase Tests

#### Phase 1-5: Core Systems
```bash
# Phase 1: Learning Mechanisms
python test_plasticity.py
python test_structural_plasticity.py
python test_reward_learning.py
python test_unsupervised_learning.py
python test_memory_consolidation.py

# Phase 2: Understanding
python test_hierarchical_learning.py
python test_semantic_representations.py
python test_world_models.py
python test_multimodal_integration.py

# Phase 3: Goals
python test_intrinsic_motivation.py
python test_goal_setting_planning.py
python test_value_systems.py
python test_executive_control.py

# Phase 4: Social
python test_theory_of_mind.py
python test_social_learning.py
python test_communication.py
python test_context_sensitivity.py

# Phase 5: Consciousness
python test_self_model.py
python test_global_workspace.py
python test_metacognition.py
python test_qualia_subjective.py
```

#### Phase 6-8: Super AGI Features
```bash
# Phase 6: Creativity
python test_creativity_system.py
python test_creative_problem_solving.py
python test_artistic_creation.py

# Phase 8: Advanced Reasoning
python test_phase8_reasoning.py
```

#### Phase 9-13: Advanced Super AGI Features
```bash
# Phase 9: Strategic Planning
python test_phase9_strategic.py

# Phase 10: Tool Use
python test_phase10_tooluse.py

# Phase 11: Language
python test_phase11_language.py

# Phase 12: Embodied Intelligence
python test_phase12_embodied.py

# Phase 13: Safety
python test_phase13_safety.py
```

### Test with Verbose Output
```bash
# Verbose test output
python -m unittest test_phase9_strategic.py -v
python -m unittest test_phase10_tooluse.py -v
python -m unittest test_phase11_language.py -v
python -m unittest test_phase12_embodied.py -v
python -m unittest test_phase13_safety.py -v
```

---

## Demonstration Commands

### Comprehensive Demonstrations

```bash
# Demonstrate all phases
python demo_all_phases.py
```

### Individual Phase Demonstrations

#### Phase 1-5: Core Systems
```bash
python demo_hierarchical_learning.py
python demo_semantic_representations.py
python demo_world_models.py
python demo_multimodal_integration.py
python demo_memory_consolidation.py
python demo_unsupervised_learning.py
python demo_reward_learning.py
python demo_phase4_social.py
```

#### Phase 6-8: Super AGI Features
```bash
python demo_creativity_system.py
```

### Interactive Demonstrations
```bash
# Interactive brain demo
python interactive_brain_demo.py

# Final enhanced brain demo
python final_demo.py
```

---

## Training Commands

### Basic Training

```bash
# Train basic brain system
python final_enhanced_brain.py

# Train with specific neuron count
python final_enhanced_brain.py --neurons 10000
```

### Scaling Training

```bash
# 10K neuron training
python simple_10k_demo.py

# 50K neuron training
python optimized_50k_v2.py

# Scaling analysis
python sequential_testing.py
```

### Phase-Specific Training

#### Phase 6: Creativity Training
```python
# In Python interactive session
from Phase6_Creativity.creativity_system import CreativitySystem
system = CreativitySystem()
# Train with examples
ideas = system.generate_ideas(input_data, context)
```

#### Phase 7: Meta-Learning Training
```python
from Phase7_AdvancedLearning.meta_learning import MetaLearningSystem
system = MetaLearningSystem()
examples = [(input, output) for _ in range(5)]
proficiency = system.few_shot_learn(examples, 'task_name')
```

#### Phase 8: Reasoning Training
```python
from Phase8_AdvancedReasoning.mathematical_reasoning import MathematicalReasoningSystem
system = MathematicalReasoningSystem()
# Train on mathematical problems
result = system.solve_equation("x + 5 = 10")
```

#### Phase 9: Strategic Planning Training
```python
from Phase9_StrategicPlanning.strategic_planning import StrategicPlanningSystem
system = StrategicPlanningSystem()
goal = system.create_strategic_goal(description, target_state, time_horizon)
planning_result = system.plan_for_goal(goal)
```

#### Phase 10: Tool Use Training
```python
from Phase10_ToolUse.tool_use import ToolUseSystem
system = ToolUseSystem()
# Register and use tools
tool = system.register_tool(name, description, function, capabilities)
result = system.use_tool_for_task(requirements, input_data)
```

#### Phase 11: Language Training
```python
from Phase11_Language.deep_language_understanding import DeepLanguageUnderstandingSystem
system = DeepLanguageUnderstandingSystem()
understanding = system.understand_text(text, context)
```

#### Phase 12: Embodied Intelligence Training
```python
from Phase12_Embodied.embodied_intelligence import EmbodiedIntelligenceSystem
system = EmbodiedIntelligenceSystem()
mapping = system.learn_sensorimotor_mapping(sensor_input, motor_output)
```

#### Phase 13: Safety Training
```python
from Phase13_Safety.safety_alignment import SafetyAlignmentSystem
system = SafetyAlignmentSystem()
value = system.register_value(name, description, importance)
safety_check = system.check_action_safety(action)
```

---

## Benchmark Commands

### Run All Benchmarks
```bash
python run_benchmarks.py
```

### Comprehensive Benchmark Suite
```bash
python comprehensive_benchmark_suite.py
```

### Super AGI Benchmark Integration
```bash
python super_agi_benchmark_integration.py
```

### Benchmark Learning
```bash
python benchmark_learning.py
```

---

## Individual Phase Commands

### Phase 6: Creativity & Innovation

**Test:**
```bash
python test_creativity_system.py
python test_creative_problem_solving.py
python test_artistic_creation.py
```

**Demo:**
```bash
python demo_creativity_system.py
```

**Use:**
```python
from Phase6_Creativity.creativity_system import CreativitySystem
from Phase6_Creativity.creative_problem_solving import CreativeProblemSolving
from Phase6_Creativity.artistic_creation import ArtisticCreation

creativity = CreativitySystem()
problem_solving = CreativeProblemSolving()
artistic = ArtisticCreation()
```

### Phase 7: Advanced Learning

**Use:**
```python
from Phase7_AdvancedLearning.meta_learning import MetaLearningSystem
from Phase7_AdvancedLearning.continual_learning import ContinualLearningSystem
from Phase7_AdvancedLearning.curriculum_learning import CurriculumLearningSystem

meta_learner = MetaLearningSystem()
continual_learner = ContinualLearningSystem()
curriculum_learner = CurriculumLearningSystem()
```

### Phase 8: Advanced Reasoning

**Test:**
```bash
python test_phase8_reasoning.py
```

**Use:**
```python
from Phase8_AdvancedReasoning.mathematical_reasoning import MathematicalReasoningSystem
from Phase8_AdvancedReasoning.scientific_discovery import ScientificDiscoverySystem
from Phase8_AdvancedReasoning.probabilistic_causal_reasoning import ProbabilisticCausalReasoning

math_reasoning = MathematicalReasoningSystem()
scientific = ScientificDiscoverySystem()
causal = ProbabilisticCausalReasoning()
```

### Phase 9: Strategic Planning & Multi-Agent

**Test:**
```bash
python test_phase9_strategic.py
```

**Use:**
```python
from Phase9_StrategicPlanning.strategic_planning import StrategicPlanningSystem
from Phase9_StrategicPlanning.multi_agent_coordination import MultiAgentCoordinationSystem
from Phase9_StrategicPlanning.agent_strategies import AgentStrategiesSystem

strategic = StrategicPlanningSystem()
multi_agent = MultiAgentCoordinationSystem()
strategies = AgentStrategiesSystem()
```

### Phase 10: Tool Use & Self-Improvement

**Test:**
```bash
python test_phase10_tooluse.py
```

**Use:**
```python
from Phase10_ToolUse.tool_use import ToolUseSystem
from Phase10_ToolUse.self_improvement import SelfImprovementSystem
from Phase10_ToolUse.knowledge_synthesis import KnowledgeSynthesisSystem

tools = ToolUseSystem()
self_improve = SelfImprovementSystem()
knowledge = KnowledgeSynthesisSystem()
```

### Phase 11: Language & Communication

**Test:**
```bash
python test_phase11_language.py
```

**Use:**
```python
from Phase11_Language.deep_language_understanding import DeepLanguageUnderstandingSystem
from Phase11_Language.language_generation import LanguageGenerationSystem
from Phase11_Language.multilingual_system import MultilingualSystem

understanding = DeepLanguageUnderstandingSystem()
generation = LanguageGenerationSystem()
multilingual = MultilingualSystem()
```

### Phase 12: Embodied Intelligence & Temporal

**Test:**
```bash
python test_phase12_embodied.py
```

**Use:**
```python
from Phase12_Embodied.embodied_intelligence import EmbodiedIntelligenceSystem
from Phase12_Embodied.temporal_reasoning import TemporalReasoningSystem
from Phase12_Embodied.long_term_memory import LongTermMemorySystem

embodied = EmbodiedIntelligenceSystem()
temporal = TemporalReasoningSystem()
memory = LongTermMemorySystem()
```

### Phase 13: Robustness, Safety & Explainability

**Test:**
```bash
python test_phase13_safety.py
```

**Use:**
```python
from Phase13_Safety.robustness_system import RobustnessSystem
from Phase13_Safety.safety_alignment import SafetyAlignmentSystem
from Phase13_Safety.explainability_system import ExplainabilitySystem

robustness = RobustnessSystem()
safety = SafetyAlignmentSystem()
explainability = ExplainabilitySystem()
```

---

## Analysis and Visualization Commands

### Performance Analysis
```bash
python performance_tracking.py
python analyze_emergent_behavior.py
```

### Visualization
```bash
python visualize_sequential_results.py
python visualize_enhanced_brain_results.py
python create_intelligence_visualization.py
```

### Scaling Analysis
```bash
python sequential_testing.py
python sequential_testing.sh
```

---

## Quick Reference Card

### Most Common Commands

```bash
# Test everything
./run_all_tests_comprehensive.sh

# Demonstrate everything
python demo_all_phases.py

# Run benchmarks
python run_benchmarks.py

# Test specific phase
python test_phase9_strategic.py
python test_phase10_tooluse.py
python test_phase11_language.py
python test_phase12_embodied.py
python test_phase13_safety.py

# Interactive demo
python interactive_brain_demo.py
```

---

## Troubleshooting

### Import Errors
If you encounter import errors, ensure you're running from the project root:
```bash
cd /Users/aichat/Downloads/BrianSimulation
python test_phase9_strategic.py
```

### Permission Errors
Make scripts executable:
```bash
chmod +x run_all_tests_comprehensive.sh
chmod +x run_all_tests.sh
```

### Missing Dependencies
Install required packages:
```bash
pip install numpy scipy matplotlib
```

---

## Summary

- **Total Test Files**: 29+ test files
- **Total Demo Files**: 9+ demo files
- **Total Phases**: 13 phases
- **Total Modules**: 39 modules

All commands are ready to use. Start with `./run_all_tests_comprehensive.sh` to verify everything works!

