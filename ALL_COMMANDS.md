# Complete Command List for Super AGI System

## 🎯 Quick Start (Most Important Commands)

```bash
# 1. Test everything
./run_all_tests_comprehensive.sh

# 2. Demonstrate everything
python demo_all_phases.py

# 3. Run benchmarks
python run_benchmarks.py
```

---

## 📝 Complete Command List

### Testing Commands

#### Comprehensive Test Suites
```bash
# Run all tests (recommended)
./run_all_tests_comprehensive.sh

# Run original test suite (Phases 6 & 8)
./run_all_tests.sh
```

#### Phase 1-5: Core Systems Tests
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

#### Phase 6-8: Super AGI Features Tests
```bash
# Phase 6: Creativity
python test_creativity_system.py
python test_creative_problem_solving.py
python test_artistic_creation.py

# Phase 8: Advanced Reasoning
python test_phase8_reasoning.py
```

#### Phase 9-13: Advanced Super AGI Features Tests
```bash
# Phase 9: Strategic Planning & Multi-Agent
python test_phase9_strategic.py

# Phase 10: Tool Use & Self-Improvement
python test_phase10_tooluse.py

# Phase 11: Language & Communication
python test_phase11_language.py

# Phase 12: Embodied Intelligence & Temporal
python test_phase12_embodied.py

# Phase 13: Robustness, Safety & Explainability
python test_phase13_safety.py
```

#### Verbose Test Output
```bash
python -m unittest test_phase9_strategic.py -v
python -m unittest test_phase10_tooluse.py -v
python -m unittest test_phase11_language.py -v
python -m unittest test_phase12_embodied.py -v
python -m unittest test_phase13_safety.py -v
```

---

### Demonstration Commands

#### Comprehensive Demonstrations
```bash
# Demonstrate all phases
python demo_all_phases.py
```

#### Individual Phase Demonstrations
```bash
# Phase 1-5 Demos
python demo_hierarchical_learning.py
python demo_semantic_representations.py
python demo_world_models.py
python demo_multimodal_integration.py
python demo_memory_consolidation.py
python demo_unsupervised_learning.py
python demo_reward_learning.py
python demo_phase4_social.py

# Phase 6 Demo
python demo_creativity_system.py

# Interactive Demos
python interactive_brain_demo.py
python final_demo.py
```

---

### Training Commands

#### Basic Training
```bash
# Train basic brain system
python final_enhanced_brain.py

# Train with specific neuron count
python final_enhanced_brain.py --neurons 10000
```

#### Scaling Training
```bash
# 10K neuron training
python simple_10k_demo.py

# 50K neuron training
python optimized_50k_v2.py

# Scaling analysis
python sequential_testing.py
```

#### Python Training Examples

**Phase 6: Creativity**
```python
from Phase6_Creativity.creativity_system import CreativitySystem
system = CreativitySystem()
ideas = system.generate_ideas(np.random.rand(10), context={'domain': 'tech'})
```

**Phase 7: Meta-Learning**
```python
from Phase7_AdvancedLearning.meta_learning import MetaLearningSystem
system = MetaLearningSystem()
examples = [(np.random.rand(5), np.random.rand(3)) for _ in range(5)]
proficiency = system.few_shot_learn(examples, 'classification')
```

**Phase 8: Reasoning**
```python
from Phase8_AdvancedReasoning.mathematical_reasoning import MathematicalReasoningSystem
system = MathematicalReasoningSystem()
result = system.solve_equation("x + 5 = 10")
```

**Phase 9: Strategic Planning**
```python
from Phase9_StrategicPlanning.strategic_planning import StrategicPlanningSystem
system = StrategicPlanningSystem()
goal = system.create_strategic_goal("Market expansion", np.array([1.0, 0.8]), 365.0)
plan = system.plan_for_goal(goal)
```

**Phase 10: Tool Use**
```python
from Phase10_ToolUse.tool_use import ToolUseSystem
system = ToolUseSystem()
tool = system.register_tool("AddOne", "Adds one", lambda x: x+1, {"math": 0.9})
result = system.use_tool_for_task({"math": 0.8}, 5)
```

**Phase 11: Language**
```python
from Phase11_Language.language_generation import LanguageGenerationSystem
system = LanguageGenerationSystem()
text = system.generate_text("AI", style="formal", length=5)
```

**Phase 12: Embodied**
```python
from Phase12_Embodied.embodied_intelligence import EmbodiedIntelligenceSystem
system = EmbodiedIntelligenceSystem()
mapping = system.learn_sensorimotor_mapping(np.array([0.5]), np.array([0.3]), True)
```

**Phase 13: Safety**
```python
from Phase13_Safety.safety_alignment import SafetyAlignmentSystem
system = SafetyAlignmentSystem()
value = system.register_value("Safety", "Prioritize safety", 0.9)
check = system.check_action_safety({'description': 'safe op', 'type': 'safe'})
```

---

### Benchmark Commands

```bash
# Run all benchmarks
python run_benchmarks.py

# Comprehensive benchmark suite
python comprehensive_benchmark_suite.py

# Super AGI benchmark integration
python super_agi_benchmark_integration.py

# Benchmark learning
python benchmark_learning.py
```

---

### Analysis & Visualization Commands

```bash
# Performance analysis
python performance_tracking.py
python analyze_emergent_behavior.py

# Visualization
python visualize_sequential_results.py
python visualize_enhanced_brain_results.py
python create_intelligence_visualization.py

# Scaling analysis
python sequential_testing.py
./sequential_testing.sh
```

---

## 📊 Command Summary by Category

### Testing (34+ commands)
- Comprehensive test suites: 2
- Phase 1-5 tests: 20
- Phase 6-8 tests: 4
- Phase 9-13 tests: 5
- Verbose tests: 5

### Demonstrations (10+ commands)
- Comprehensive: 1
- Individual phase: 8
- Interactive: 2

### Training (10+ commands)
- Basic: 2
- Scaling: 3
- Python examples: 8

### Benchmarks (4 commands)
- All benchmarks: 1
- Comprehensive: 1
- Integration: 1
- Learning: 1

### Analysis (6+ commands)
- Performance: 2
- Visualization: 3
- Scaling: 2

---

## 🎯 Recommended Workflow

### 1. Initial Setup
```bash
cd /Users/aichat/Downloads/BrianSimulation
chmod +x run_all_tests_comprehensive.sh
```

### 2. Verify Installation
```bash
./run_all_tests_comprehensive.sh
```

### 3. See Capabilities
```bash
python demo_all_phases.py
```

### 4. Run Benchmarks
```bash
python run_benchmarks.py
```

### 5. Test Specific Phase
```bash
python test_phase9_strategic.py
python test_phase10_tooluse.py
python test_phase11_language.py
python test_phase12_embodied.py
python test_phase13_safety.py
```

---

## 📚 Documentation Files

- `COMPLETE_COMMAND_REFERENCE.md` - Full detailed reference
- `QUICK_COMMAND_REFERENCE.md` - Quick reference card
- `ALL_COMMANDS.md` - This file (complete list)
- `TESTING_GUIDE.md` - Testing guide
- `HOW_TO_TEST_SUPER_AGI.md` - How to test guide

---

## ⚠️ Troubleshooting

### Import Errors
```bash
# Ensure you're in the project root
cd /Users/aichat/Downloads/BrianSimulation
python test_phase9_strategic.py
```

### Permission Errors
```bash
chmod +x run_all_tests_comprehensive.sh
chmod +x run_all_tests.sh
```

### Missing Dependencies
```bash
pip install numpy scipy matplotlib
```

---

## ✅ Quick Verification

Run these commands to verify everything works:

```bash
# 1. Test new phases
python test_phase9_strategic.py && echo "✓ Phase 9 OK"
python test_phase10_tooluse.py && echo "✓ Phase 10 OK"
python test_phase11_language.py && echo "✓ Phase 11 OK"
python test_phase12_embodied.py && echo "✓ Phase 12 OK"
python test_phase13_safety.py && echo "✓ Phase 13 OK"

# 2. Demo all phases
python demo_all_phases.py && echo "✓ Demo OK"
```

---

**Total Commands Available: 60+**

All commands are ready to use! Start with `./run_all_tests_comprehensive.sh` to test everything.

