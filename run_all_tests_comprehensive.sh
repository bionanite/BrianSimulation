#!/bin/bash
# Comprehensive Test Runner for All Super AGI Phases

echo "=========================================="
echo "COMPREHENSIVE SUPER AGI TEST SUITE"
echo "=========================================="
echo ""

# Phase 1-5: Core Systems
echo "--- Phase 1-5: Core Systems ---"
python test_plasticity.py && echo "✓ Phase 1.1: Plasticity" || echo "✗ Phase 1.1: Failed"
python test_structural_plasticity.py && echo "✓ Phase 1.2: Structural Plasticity" || echo "✗ Phase 1.2: Failed"
python test_reward_learning.py && echo "✓ Phase 1.3: Reward Learning" || echo "✗ Phase 1.3: Failed"
python test_unsupervised_learning.py && echo "✓ Phase 1.4: Unsupervised Learning" || echo "✗ Phase 1.4: Failed"
python test_memory_consolidation.py && echo "✓ Phase 1.5: Memory Consolidation" || echo "✗ Phase 1.5: Failed"

python test_hierarchical_learning.py && echo "✓ Phase 2.1: Hierarchical Learning" || echo "✗ Phase 2.1: Failed"
python test_semantic_representations.py && echo "✓ Phase 2.2: Semantic Representations" || echo "✗ Phase 2.2: Failed"
python test_world_models.py && echo "✓ Phase 2.3: World Models" || echo "✗ Phase 2.3: Failed"
python test_multimodal_integration.py && echo "✓ Phase 2.4: Multimodal Integration" || echo "✗ Phase 2.4: Failed"

python test_intrinsic_motivation.py && echo "✓ Phase 3.1: Intrinsic Motivation" || echo "✗ Phase 3.1: Failed"
python test_goal_setting_planning.py && echo "✓ Phase 3.2: Goal Setting" || echo "✗ Phase 3.2: Failed"
python test_value_systems.py && echo "✓ Phase 3.3: Value Systems" || echo "✗ Phase 3.3: Failed"
python test_executive_control.py && echo "✓ Phase 3.4: Executive Control" || echo "✗ Phase 3.4: Failed"

python test_theory_of_mind.py && echo "✓ Phase 4.1: Theory of Mind" || echo "✗ Phase 4.1: Failed"
python test_social_learning.py && echo "✓ Phase 4.2: Social Learning" || echo "✗ Phase 4.2: Failed"
python test_communication.py && echo "✓ Phase 4.3: Communication" || echo "✗ Phase 4.3: Failed"
python test_context_sensitivity.py && echo "✓ Phase 4.4: Context Sensitivity" || echo "✗ Phase 4.4: Failed"

python test_self_model.py && echo "✓ Phase 5.1: Self Model" || echo "✗ Phase 5.1: Failed"
python test_global_workspace.py && echo "✓ Phase 5.2: Global Workspace" || echo "✗ Phase 5.2: Failed"
python test_metacognition.py && echo "✓ Phase 5.3: Metacognition" || echo "✗ Phase 5.3: Failed"
python test_qualia_subjective.py && echo "✓ Phase 5.4: Qualia" || echo "✗ Phase 5.4: Failed"

echo ""
echo "--- Phase 6-8: Super AGI Features ---"
python test_creativity_system.py && echo "✓ Phase 6.1: Creativity" || echo "✗ Phase 6.1: Failed"
python test_creative_problem_solving.py && echo "✓ Phase 6.2: Creative Problem Solving" || echo "✗ Phase 6.2: Failed"
python test_artistic_creation.py && echo "✓ Phase 6.3: Artistic Creation" || echo "✗ Phase 6.3: Failed"

# Phase 7 tests would go here if they exist
# python test_meta_learning.py && echo "✓ Phase 7.1: Meta-Learning" || echo "✗ Phase 7.1: Failed"
# python test_continual_learning.py && echo "✓ Phase 7.2: Continual Learning" || echo "✗ Phase 7.2: Failed"
# python test_curriculum_learning.py && echo "✓ Phase 7.3: Curriculum Learning" || echo "✗ Phase 7.3: Failed"

python test_phase8_reasoning.py && echo "✓ Phase 8: Advanced Reasoning" || echo "✗ Phase 8: Failed"

echo ""
echo "--- Phase 9-13: Advanced Super AGI Features ---"
python test_phase9_strategic.py && echo "✓ Phase 9: Strategic Planning" || echo "✗ Phase 9: Failed"
python test_phase10_tooluse.py && echo "✓ Phase 10: Tool Use" || echo "✗ Phase 10: Failed"
python test_phase11_language.py && echo "✓ Phase 11: Language" || echo "✗ Phase 11: Failed"
python test_phase12_embodied.py && echo "✓ Phase 12: Embodied Intelligence" || echo "✗ Phase 12: Failed"
python test_phase13_safety.py && echo "✓ Phase 13: Safety" || echo "✗ Phase 13: Failed"

echo ""
echo "=========================================="
echo "ALL TESTS COMPLETE"
echo "=========================================="

