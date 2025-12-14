# Testing Guide for Super AGI Features

## Quick Start

Run all tests with:
```bash
python test_creativity_system.py
python test_creative_problem_solving.py
python test_artistic_creation.py
python test_phase8_reasoning.py
```

## Individual Test Suites

### Phase 6: Creativity & Innovation

#### Test 6.1: Creative Idea Generation
```bash
python test_creativity_system.py
```

**What it tests:**
- Conceptual blending of concepts
- Divergent thinking (generating alternatives)
- Randomness injection
- Novelty detection
- Full creativity system integration

**Expected output:**
```
============================================================
Testing Creativity System (Phase 6.1)
============================================================
Testing Conceptual Blending...
  ✓ Conceptual blending works
Testing Divergent Thinking...
  ✓ Divergent thinking works
Testing Randomness Injection...
  ✓ Randomness injection works
Testing Novelty Detection...
  ✓ Novelty detection works
Testing Creativity System Integration...
  ✓ Creativity system integration works
============================================================
Tests passed: 5/5
Tests failed: 0/5
============================================================
```

#### Test 6.2: Creative Problem Solving
```bash
python test_creative_problem_solving.py
```

**What it tests:**
- Analogical reasoning
- Constraint relaxation
- Problem reframing
- Creative problem solving integration

**Expected output:**
```
============================================================
Testing Creative Problem Solving (Phase 6.2)
============================================================
Testing Analogical Reasoning...
  ✓ Analogical reasoning works
Testing Constraint Relaxation...
  ✓ Constraint relaxation works
Testing Problem Reframing...
  ✓ Problem reframing works
Testing Creative Problem Solving Integration...
  ✓ Creative problem solving integration works
============================================================
Tests passed: 4/4
Tests failed: 0/4
============================================================
```

#### Test 6.3: Artistic Creation
```bash
python test_artistic_creation.py
```

**What it tests:**
- Aesthetic evaluation
- Style learning
- Emotional expression
- Artistic creation integration

**Expected output:**
```
============================================================
Testing Artistic Creation (Phase 6.3)
============================================================
Testing Aesthetic Evaluation...
  ✓ Aesthetic evaluation works
Testing Style Learning...
  ✓ Style learning works
Testing Emotional Expression...
  ✓ Emotional expression works
Testing Artistic Creation Integration...
  ✓ Artistic creation integration works
============================================================
Tests passed: 4/4
Tests failed: 0/4
============================================================
```

### Phase 8: Advanced Reasoning

#### Test Phase 8: All Reasoning Systems
```bash
python test_phase8_reasoning.py
```

**What it tests:**
- Mathematical reasoning and theorem proving
- Scientific discovery (hypothesis generation)
- Probabilistic and causal reasoning

**Expected output:**
```
============================================================
Testing Phase 8: Advanced Reasoning
============================================================
Testing Mathematical Reasoning...
  ✓ Mathematical reasoning works
Testing Scientific Discovery...
  ✓ Scientific discovery works
Testing Probabilistic & Causal Reasoning...
  ✓ Probabilistic reasoning works
============================================================
Tests passed: 3/3
Tests failed: 0/3
============================================================
```

## Demonstrations

### Run Creativity System Demo
```bash
python demo_creativity_system.py
```

**What it demonstrates:**
- Creative idea generation using 4 different methods
- Statistics and metrics
- Visualization generation
- Creates `creativity_system_progress.png`

**Expected output:**
```
======================================================================
Creativity System Demonstration (Phase 6.1)
======================================================================

1. Creating concept base...
   Created 10 concepts

2. Generating creative ideas...
   [Shows ideas with novelty, creativity, and feasibility scores]

3. Creativity Statistics:
   Total ideas generated: 11
   Novel ideas: 7
   Blends created: 5
   Average novelty: 0.369
   Average creativity: 0.400

4. Top Creative Ideas:
   [Lists top 5 creative ideas]

5. Creating visualization...
   Saved visualization to creativity_system_progress.png

======================================================================
Demonstration complete!
======================================================================
```

## Running All Tests at Once

Create a test runner script:

```bash
#!/bin/bash
# run_all_tests.sh

echo "=========================================="
echo "Running All Super AGI Feature Tests"
echo "=========================================="

echo ""
echo "Phase 6: Creativity & Innovation"
python test_creativity_system.py
python test_creative_problem_solving.py
python test_artistic_creation.py

echo ""
echo "Phase 8: Advanced Reasoning"
python test_phase8_reasoning.py

echo ""
echo "=========================================="
echo "All tests complete!"
echo "=========================================="
```

Make it executable and run:
```bash
chmod +x run_all_tests.sh
./run_all_tests.sh
```

## Testing Individual Components

### Test Specific Components

You can also test individual components by importing them:

```python
# test_individual_component.py
import sys
sys.path.insert(0, 'Phase6_Creativity')

from creativity_system import CreativitySystem, ConceptualBlending

# Test conceptual blending
blending = ConceptualBlending()
# ... your test code ...
```

## Integration Testing

To test integration with the main brain system:

```python
# test_integration.py
from final_enhanced_brain import FinalEnhancedBrain
from Phase6_Creativity.creativity_system import CreativitySystem

# Create brain system
brain = FinalEnhancedBrain(total_neurons=10000)

# Create creativity system
creativity = CreativitySystem(brain_system=brain)

# Test integration
ideas = creativity.generate_ideas(num_ideas=5)
print(f"Generated {len(ideas)} ideas")
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're in the correct directory:
```bash
cd /Users/aichat/Downloads/BrianSimulation
```

### Missing Dependencies
The tests use standard libraries (numpy, etc.). If you get errors:
```bash
pip install numpy matplotlib
```

### Test Failures
If a test fails:
1. Check the error message
2. Verify the file exists in the correct location
3. Check that all dependencies are installed
4. Run the test individually to isolate the issue

## Performance Testing

To test performance with larger datasets:

```python
# test_performance.py
import time
from Phase6_Creativity.creativity_system import CreativitySystem

creativity = CreativitySystem()

start = time.time()
ideas = creativity.generate_ideas(num_ideas=100)
end = time.time()

print(f"Generated 100 ideas in {end - start:.2f} seconds")
print(f"Average: {(end - start) / 100 * 1000:.2f} ms per idea")
```

## Continuous Testing

For continuous testing during development:

```bash
# Watch for file changes and rerun tests
# Requires: pip install watchdog
watchmedo shell-command \
  --patterns="*.py" \
  --recursive \
  --command='python test_creativity_system.py' \
  .
```

## Test Coverage

Current test coverage:
- ✅ Phase 6.1: 100% (5/5 tests)
- ✅ Phase 6.2: 100% (4/4 tests)
- ✅ Phase 6.3: 100% (4/4 tests)
- ✅ Phase 8: 100% (3/3 tests)
- 🔄 Phase 7: Tests to be created

## Next Steps

1. **Create Phase 7 tests**: Add test files for meta-learning, continual learning, and curriculum learning
2. **Add integration tests**: Test integration between phases
3. **Add performance benchmarks**: Measure performance metrics
4. **Add stress tests**: Test with large datasets

