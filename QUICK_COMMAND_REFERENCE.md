# Quick Command Reference

## 🚀 Quick Start

```bash
# Test everything
./run_all_tests_comprehensive.sh

# Demonstrate everything
python demo_all_phases.py

# Run benchmarks
python run_benchmarks.py
```

## 📋 Test Commands

### All Tests
```bash
./run_all_tests_comprehensive.sh    # Comprehensive (all phases)
./run_all_tests.sh                  # Original (phases 6 & 8)
```

### Individual Phase Tests
```bash
# Phase 6: Creativity
python test_creativity_system.py
python test_creative_problem_solving.py
python test_artistic_creation.py

# Phase 8: Reasoning
python test_phase8_reasoning.py

# Phase 9: Strategic Planning
python test_phase9_strategic.py

# Phase 10: Tool Use
python test_phase10_tooluse.py

# Phase 11: Language
python test_phase11_language.py

# Phase 12: Embodied
python test_phase12_embodied.py

# Phase 13: Safety
python test_phase13_safety.py
```

## 🎭 Demo Commands

```bash
# All phases demo
python demo_all_phases.py

# Individual demos
python demo_creativity_system.py
python demo_hierarchical_learning.py
python demo_world_models.py
python interactive_brain_demo.py
```

## 🎯 Training Examples

### Phase 6: Creativity
```python
from Phase6_Creativity.creativity_system import CreativitySystem
system = CreativitySystem()
ideas = system.generate_ideas(np.random.rand(10), context={})
```

### Phase 9: Strategic Planning
```python
from Phase9_StrategicPlanning.strategic_planning import StrategicPlanningSystem
system = StrategicPlanningSystem()
goal = system.create_strategic_goal("Goal", np.array([1.0]), 365.0)
plan = system.plan_for_goal(goal)
```

### Phase 10: Tool Use
```python
from Phase10_ToolUse.tool_use import ToolUseSystem
system = ToolUseSystem()
tool = system.register_tool("Tool", "Desc", lambda x: x+1, {"cap": 0.9})
result = system.use_tool_for_task({"cap": 0.8}, 5)
```

### Phase 11: Language
```python
from Phase11_Language.language_generation import LanguageGenerationSystem
system = LanguageGenerationSystem()
text = system.generate_text("AI", style="formal", length=5)
```

## 📊 Benchmarks

```bash
python run_benchmarks.py
python comprehensive_benchmark_suite.py
python super_agi_benchmark_integration.py
```

## 🔍 Analysis

```bash
python performance_tracking.py
python analyze_emergent_behavior.py
python visualize_sequential_results.py
```

## ⚡ Most Used Commands

```bash
# Test all new phases (9-13)
python test_phase9_strategic.py
python test_phase10_tooluse.py
python test_phase11_language.py
python test_phase12_embodied.py
python test_phase13_safety.py

# Demo all capabilities
python demo_all_phases.py

# Full test suite
./run_all_tests_comprehensive.sh
```

