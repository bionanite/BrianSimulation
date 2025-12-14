# Test Commands for Benchmark System

## Quick Start Commands

### 1. Install Dependencies First
```bash
pip install datasets evaluate requests wikipedia
```

For LLM integration (optional):
```bash
pip install openai anthropic
```

### 2. Basic Benchmark Test (10K neurons, all benchmarks)
```bash
python run_benchmarks.py 10000
```

### 3. Test Specific Benchmark
```bash
# Test MMLU only
python run_benchmarks.py 10000 MMLU

# Test HellaSwag only
python run_benchmarks.py 10000 HellaSwag

# Test GSM8K only
python run_benchmarks.py 10000 GSM8K

# Test ARC only
python run_benchmarks.py 10000 ARC

# Test HumanEval only
python run_benchmarks.py 10000 HumanEval
```

### 4. Comprehensive Benchmark Suite
```bash
# Basic run (10K neurons, no LLM, 20 questions per benchmark)
python comprehensive_benchmark_suite.py

# With custom neuron count
python comprehensive_benchmark_suite.py 50000

# With LLM integration (requires API keys)
python comprehensive_benchmark_suite.py 10000 true

# Full configuration: neurons, use_llm, max_questions, enable_learning
python comprehensive_benchmark_suite.py 10000 false 50 true
```

### 5. Scale Validation Test
```bash
# Test at 10K neurons
python validate_80b_scale.py 10000

# Test at 100K neurons
python validate_80b_scale.py 100000

# Test at 1M neurons
python validate_80b_scale.py 1000000

# Test at 80B neurons (may require distributed computing)
python validate_80b_scale.py 80000000000
```

## Individual Component Tests

### Test Language Processor
```python
python -c "
from language_processor import LanguageProcessor
lp = LanguageProcessor()
text = 'What is the capital of France?'
pattern = lp.text_to_pattern(text)
print(f'Pattern shape: {pattern.shape}')
print(f'Pattern sample: {pattern[:10]}')
result = lp.pattern_to_text(pattern)
print(f'Reconstructed: {result}')
"
```

### Test Benchmark Framework
```python
python -c "
from benchmark_framework import BenchmarkFramework
from benchmark_adapters import MMLUAdapter
from final_enhanced_brain import FinalEnhancedBrain

brain = FinalEnhancedBrain(total_neurons=10000)
framework = BenchmarkFramework(brain_system=brain)
framework.register_adapter('MMLU', MMLUAdapter())
summary = framework.run_benchmark('MMLU', max_questions=3, verbose=True)
print(f'Accuracy: {summary.accuracy:.2%}')
"
```

### Test Advanced Reasoning
```python
python -c "
from advanced_reasoning import AdvancedReasoning
reasoner = AdvancedReasoning()
result = reasoner.chain_of_thought('What is 2 + 2?', max_steps=3, verbose=True)
print(f'Conclusion: {result[\"conclusion\"]}')
print(f'Confidence: {result[\"confidence\"]:.2f}')
"
```

### Test Knowledge Base
```python
python -c "
from knowledge_base import KnowledgeBase
kb = KnowledgeBase()
result = kb.query_wikipedia('artificial intelligence', sentences=2)
print(f'Found: {result.get(\"found\")}')
print(f'Content: {result.get(\"content\", \"\")[:200]}')
"
```

### Test LLM Integration (requires API key)
```bash
export OPENAI_API_KEY="your-key-here"
python -c "
from language_integration import LLMIntegration
from final_enhanced_brain import FinalEnhancedBrain

brain = FinalEnhancedBrain(total_neurons=10000)
llm = LLMIntegration(provider='openai')
result = llm.hybrid_process('What is 2+2?', brain)
print(f'Answer: {result[\"answer\"]}')
"
```

## Quick Test Scripts

### Quick Test All Components
```bash
# Create test script
cat > quick_test.py << 'EOF'
#!/usr/bin/env python3
"""Quick test of all components"""

print("Testing Language Processor...")
from language_processor import LanguageProcessor
lp = LanguageProcessor()
pattern = lp.text_to_pattern("Test question")
print(f"✅ Language Processor: Pattern shape {pattern.shape}")

print("\nTesting Benchmark Adapters...")
from benchmark_adapters import MMLUAdapter
adapter = MMLUAdapter()
data = adapter.load_dataset()
print(f"✅ MMLU Adapter: Loaded {len(data)} items")

print("\nTesting Advanced Reasoning...")
from advanced_reasoning import AdvancedReasoning
reasoner = AdvancedReasoning()
result = reasoner.chain_of_thought("Test", max_steps=2)
print(f"✅ Advanced Reasoning: {len(result['steps'])} steps")

print("\nTesting Knowledge Base...")
from knowledge_base import KnowledgeBase
kb = KnowledgeBase()
print(f"✅ Knowledge Base: Cache stats {kb.get_cache_stats()}")

print("\n✅ All components tested successfully!")
EOF

python quick_test.py
```

### Test Benchmark Learning
```python
python -c "
from benchmark_learning import BenchmarkLearner
from final_enhanced_brain import FinalEnhancedBrain
from benchmark_framework import BenchmarkResult

brain = FinalEnhancedBrain(total_neurons=10000)
learner = BenchmarkLearner(brain_system=brain)

# Create mock results
results = [
    BenchmarkResult(
        benchmark_name='MMLU',
        task_name='test',
        question='Test?',
        ground_truth='Answer',
        prediction='Answer',
        is_correct=True,
        confidence=0.8,
        response_time=1.0
    )
]

analysis = learner.analyze_results(results)
print(f'Analysis: {analysis[\"accuracy\"]:.2%}')
print('✅ Benchmark Learning tested')
"
```

## Performance Testing

### Test with Different Neuron Counts
```bash
# Small scale (fast)
python comprehensive_benchmark_suite.py 1000 false 5 false

# Medium scale
python comprehensive_benchmark_suite.py 10000 false 10 false

# Large scale (slower)
python comprehensive_benchmark_suite.py 100000 false 10 false
```

### Test Learning Over Multiple Runs
```bash
# Run benchmark multiple times to see learning
for i in {1..3}; do
    echo "Run $i"
    python comprehensive_benchmark_suite.py 10000 false 10 true
done
```

## Debug Mode

### Run with Debug Output
```python
python -c "
from final_enhanced_brain import FinalEnhancedBrain
brain = FinalEnhancedBrain(total_neurons=10000, debug=True)
# Debug output will show initialization details
"
```

## Expected Output Examples

### Successful Benchmark Run
```
======================================================================
BENCHMARK VALIDATION SYSTEM
======================================================================

Initializing brain system...
✅ Brain initialized with 10,000 neurons

Registering benchmark adapters...
✅ Registered benchmark adapter: MMLU
✅ Registered benchmark adapter: HellaSwag
...

Running MMLU
======================================================================
Benchmark Results: MMLU
======================================================================
Accuracy: 45.00% (9/20)
Average Confidence: 0.650
Average Response Time: 0.123s
```

### Superhuman Achievement
```
======================================================================
SUPERHUMAN INTELLIGENCE ASSESSMENT
======================================================================
  MMLU          : 92.00% (threshold: 90%) ✅ SUPERHUMAN
  HellaSwag     : 96.00% (threshold: 95%) ✅ SUPERHUMAN
  GSM8K         : 91.00% (threshold: 90%) ✅ SUPERHUMAN
  ARC           : 86.00% (threshold: 85%) ✅ SUPERHUMAN
  HumanEval     : 87.00% (threshold: 85%) ✅ SUPERHUMAN

Superhuman Benchmarks: 5/5
🎉 ACHIEVED SUPERHUMAN INTELLIGENCE ON ALL BENCHMARKS!
```

## Troubleshooting Commands

### Check Dependencies
```bash
python -c "
import sys
deps = ['numpy', 'datasets', 'requests', 'wikipedia', 'openai', 'anthropic']
for dep in deps:
    try:
        __import__(dep)
        print(f'✅ {dep}')
    except ImportError:
        print(f'❌ {dep} - Install with: pip install {dep}')
"
```

### Test Brain System Only
```python
python -c "
from final_enhanced_brain import FinalEnhancedBrain
brain = FinalEnhancedBrain(total_neurons=1000)
result = brain.comprehensive_enhanced_assessment()
print(f'Intelligence Score: {result[\"overall_enhanced_score\"]:.3f}')
print('✅ Brain system working')
"
```

### Check File Structure
```bash
ls -la benchmark_framework.py benchmark_adapters.py performance_tracking.py language_processor.py language_integration.py knowledge_base.py benchmark_learning.py advanced_reasoning.py
```

## Recommended Test Sequence

1. **Quick Component Test**
   ```bash
   python quick_test.py
   ```

2. **Single Benchmark Test**
   ```bash
   python run_benchmarks.py 10000 MMLU
   ```

3. **All Benchmarks (Small)**
   ```bash
   python comprehensive_benchmark_suite.py 10000 false 10 false
   ```

4. **Full Test with Learning**
   ```bash
   python comprehensive_benchmark_suite.py 10000 false 20 true
   ```

5. **Scale Validation**
   ```bash
   python validate_80b_scale.py 100000
   ```

## Notes

- Start with small neuron counts (10K) for faster testing
- Use `max_questions=5-10` for quick tests
- Enable learning (`true`) to see improvement over runs
- Check `benchmark_results/` directory for saved results
- LLM integration requires API keys (optional)

